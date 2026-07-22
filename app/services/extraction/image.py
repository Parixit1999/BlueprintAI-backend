"""Image extraction via a vision model (Ollama locally, Bedrock Claude on AWS).

Vision output is the least reliable extraction path, which is exactly why every
field carries model-reported confidence and flows through HITL review before
ingestion. The model is instructed to return null rather than guess.
"""
import io
import json
import re
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from app.exceptions import ExtractionFailed, InvalidFile
from app.schemas import Confidence, ProvisionalChunk, RegionType
from app.services.ai.base import VisionProvider
from app.services.extraction.enhance import enhance_for_vision


@dataclass
class VisionRegion:
    """One region the vision model reported, with a resolution-independent
    bbox in percentages (top-left origin). Callers map it into whatever
    coordinate space their renderer uses - image pixels for a photo, PDF
    points for a rasterized scanned page."""

    region_type: RegionType
    text: str | None
    confidence: Confidence
    bbox_pct: list[float] | None  # [x1, y1, x2, y2] as 0-100 percentages
    is_drawing: bool | None = None  # summary region only: vision verdict

PROMPT = """You are extracting content from an engineering drawing image.
The image is {width}x{height} pixels.

Find EVERY piece of visible text in the image - title block fields, drawing
numbers, dimensions, notes, labels. Each one is its own region; do not merge
or skip any.

Return ONLY a JSON object of the form
{"is_drawing": true|false, "summary": "...", "regions": [...]}.

"is_drawing" is false when the image is NOT an engineering/technical drawing
(a photo, screenshot, document scan of prose, etc.). Judge honestly.

"summary" is one rich paragraph describing what the drawing DEPICTS as an
engineer would: what kind of drawing it is, what is shown (equipment,
structures, plans, sections), the overall layout, and anything notable.
Mention the drawing number and title if visible. Do not guess at values.

Each element of "regions" describes one text region:
{
  "text": "the exact text, or null if illegible - NEVER guess",
  "type": "note" | "dimension" | "title_block" | "bom",
  "bbox_pct": [x1, y1, x2, y2],
  "confidence": "high" | "medium" | "low"
}

bbox_pct is [x1, y1, x2, y2] in PIXELS of this {width}x{height} image,
measured from the TOP-LEFT corner. Draw the box TIGHTLY around the text it
contains - it is used to highlight the exact region on the drawing.
Use confidence "low" for anything small, blurry, or partially obscured. If a
value is illegible, set text to null and confidence to "low". No prose
outside the JSON object."""

_REGION_MAP = {
    "note": RegionType.note,
    "dimension": RegionType.dimension,
    "title_block": RegionType.title_block,
    "bom": RegionType.bom,
}


def _parse_is_drawing(raw: str) -> bool | None:
    """The explicit is-this-a-drawing verdict; None when absent/unparseable."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    verdict = obj.get("is_drawing") if isinstance(obj, dict) else None
    return verdict if isinstance(verdict, bool) else None


def _parse_summary(raw: str) -> str | None:
    """The whole-drawing description from the {"summary": ...} contract."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    summary = obj.get("summary") if isinstance(obj, dict) else None
    return summary.strip() if isinstance(summary, str) and summary.strip() else None


def _parse_response(raw: str) -> list[dict]:
    # Preferred: the {"regions": [...]} object contract. Fall back to a bare
    # array for models/providers that return one.
    raw = raw.strip()
    if raw.startswith("```"):  # some providers fence their JSON
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("regions"), list):
            return obj["regions"]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match is None:
        raise ExtractionFailed(
            "The vision model did not return structured output for this image. "
            "Try a clearer or higher-resolution image."
        )
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        raise ExtractionFailed(
            "The vision model returned malformed output for this image. "
            "Try again or use a clearer image."
        )
    if not isinstance(parsed, list):
        raise ExtractionFailed("The vision model returned an unexpected structure.")
    return parsed


class ImageExtractor:
    def __init__(self, vision: VisionProvider):
        self._vision = vision

    @staticmethod
    def _to_pct(values: list, sent_width: int, sent_height: int) -> list[float] | None:
        """Normalize model coords to 0-100 percentages.

        Vision models ignore coordinate instructions often enough that we
        detect the scale: fractions (0-1), percentages (0-100), or absolute
        pixels of the (downscaled) image we sent.
        """
        try:
            x1, y1, x2, y2 = (float(v) for v in values)
        except (TypeError, ValueError):
            return None
        peak = max(x1, y1, x2, y2)
        if peak <= 1:
            x1, y1, x2, y2 = (v * 100 for v in (x1, y1, x2, y2))
        elif peak <= 100:
            pass
        else:
            # absolute pixel coords of the sent image, per axis
            x1, x2 = (v / sent_width * 100 for v in (x1, x2))
            y1, y2 = (v / sent_height * 100 for v in (y1, y2))
        vals = [min(max(v, 0.0), 100.0) for v in (x1, y1, x2, y2)]
        x1, y1, x2, y2 = vals
        # models sometimes emit corners in reverse order - normalize, don't reject
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 0.1 or y2 - y1 < 0.1:
            return None  # degenerate box
        return [x1, y1, x2, y2]

    # Claude vision reads up to ~1568px on the long edge; sending more
    # resolution than the old 1024 cap materially improves both text
    # legibility and bbox precision on dense archive sheets.
    MAX_SIDE = 1568

    # Bedrock's 5 MB image limit applies to the BASE64-encoded payload
    # (raw x 4/3), so the raw bytes must stay under ~3.9 MB; keep margin
    MAX_BYTES = 3_600_000

    @staticmethod
    def _downscale(data: bytes) -> tuple[bytes, int, int]:
        """Send the model a bounded, known-size image so absolute pixel
        coordinates in its output can be mapped back reliably. Dense scans at
        1568px can exceed the provider's byte limit as PNG - fall back to JPEG
        (visually lossless for scans), shrinking further only if needed."""
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            if max(img.size) > ImageExtractor.MAX_SIDE:
                img.thumbnail((ImageExtractor.MAX_SIDE, ImageExtractor.MAX_SIDE))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            if buf.tell() <= ImageExtractor.MAX_BYTES:
                return buf.getvalue(), img.width, img.height
            while True:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=88)
                if buf.tell() <= ImageExtractor.MAX_BYTES or min(img.size) < 400:
                    return buf.getvalue(), img.width, img.height
                img = img.resize((int(img.width * 0.85), int(img.height * 0.85)), Image.LANCZOS)

    def analyze(self, data: bytes) -> list[VisionRegion]:
        """Run the vision model on raw image bytes and return regions with
        percentage bboxes. Coordinate-space-agnostic so both photo uploads and
        rasterized scanned-PDF pages can reuse it. Returns [] if nothing found;
        callers decide whether an empty result is an error."""
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
        except UnidentifiedImageError:
            raise InvalidFile("This file is not a valid image - it appears to be corrupt.")

        sent_bytes, sent_w, sent_h = self._downscale(data)
        # .replace, not .format - the prompt's JSON examples contain braces
        prompt = PROMPT.replace("{width}", str(sent_w)).replace("{height}", str(sent_h))
        raw = self._vision.analyze_image(sent_bytes, prompt)
        regions: list[VisionRegion] = []
        summary = _parse_summary(raw)
        if summary:
            regions.append(
                VisionRegion(
                    region_type=RegionType.summary,
                    text=summary,
                    confidence=Confidence.high,
                    bbox_pct=None,  # describes the whole drawing
                    is_drawing=_parse_is_drawing(raw),
                )
            )
        for item in _parse_response(raw):
            if not isinstance(item, dict):
                continue
            bbox_pct = None
            pct = item.get("bbox_pct")
            if isinstance(pct, list) and len(pct) == 4:
                bbox_pct = self._to_pct(pct, sent_w, sent_h)
            text = item.get("text")
            confidence = item.get("confidence")
            regions.append(
                VisionRegion(
                    region_type=_REGION_MAP.get(item.get("type"), RegionType.note),
                    text=str(text).strip() if text else None,
                    confidence=(
                        Confidence(confidence)
                        if confidence in {c.value for c in Confidence}
                        else Confidence.medium
                    ),
                    bbox_pct=bbox_pct,
                )
            )
        return regions

    @staticmethod
    def region_to_chunk(
        region: VisionRegion, xmax: float, ymax: float, page: int = 1
    ) -> ProvisionalChunk:
        """Map a percentage-space region into a chunk whose bbox is in the
        renderer's y-up coordinate space (extents [0, 0, xmax, ymax])."""
        bbox = None
        if region.bbox_pct is not None:
            x1, y1, x2, y2 = region.bbox_pct
            # percentages from top-left -> extents coords, y-up
            bbox = [
                round(x1 / 100 * xmax, 1),
                round(ymax - (y2 / 100 * ymax), 1),
                round(x2 / 100 * xmax, 1),
                round(ymax - (y1 / 100 * ymax), 1),
            ]
        return ProvisionalChunk(
            region_type=region.region_type,
            chunk_text=region.text,
            is_drawing=region.is_drawing,
            bbox=bbox,
            confidence=region.confidence,
            page=page,
        )

    def extract(self, path: str) -> list[ProvisionalChunk]:
        raw = open(path, "rb").read()
        # Enhancement (orientation, contrast, upscale) BEFORE the vision model;
        # dims come from the enhanced image so bbox percentages match the
        # (orientation-normalized) preview.
        data, _applied = enhance_for_vision(raw)
        try:
            with Image.open(io.BytesIO(data)) as img:
                width, height = img.size
        except UnidentifiedImageError:
            raise InvalidFile("This file is not a valid image - it appears to be corrupt.")

        regions = self.analyze(data)
        chunks = [self.region_to_chunk(r, width, height) for r in regions]
        if not chunks:
            raise ExtractionFailed(
                "No text regions were detected in this image. If the drawing has "
                "readable text, try a higher-resolution or better-lit image."
            )
        return chunks
