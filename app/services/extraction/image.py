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
from pillow_heif import register_heif_opener

from app.exceptions import ExtractionFailed, InvalidFile
from app.services.extraction.ocr import normalize_text, textract_lines

# Teach Pillow to open iPhone HEIC/HEIF photos (idempotent)
register_heif_opener()
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
    # component groups: every further instance's bbox (first is bbox_pct)
    extra_bboxes_pct: list[list[float]] | None = None
    is_drawing: bool | None = None  # summary region only: vision verdict

PROMPT = """You are extracting content from an engineering drawing image.
The image is {width}x{height} pixels.

Find EVERY piece of visible text in the image - title block fields, drawing
numbers, dimensions, notes, labels. Each one is its own region; do not merge
or skip any.

ALSO identify the DRAWN COMPONENTS - physical elements depicted in the
drawing itself, not text: stairs, pipes, valves, pumps, doors, walls, tanks,
ducts, structural members, fixtures, major equipment (these are examples;
label whatever the drawing actually depicts). Be EXHAUSTIVE but GROUPED:
emit ONE region per component TYPE, with "text" a short engineer's label
(e.g. "staircase, U-shaped", "gate valve", "door, single-swing") - never the
bare word "component" - and put the bbox of EVERY instance of that type in
"instances" (the first instance also goes in "bbox_pct"). Box instances
TIGHTLY. Every recognizable drawn element belongs to some group; do not
skip repeats - count them all via their boxes.

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
  "text": "the exact text (for component: a short label), or null if illegible - NEVER guess",
  "type": "note" | "dimension" | "title_block" | "bom" | "component",
  "bbox_pct": [x1, y1, x2, y2],
  "instances": [[x1, y1, x2, y2], ...],   // component regions only: every instance
  "confidence": "high" | "medium" | "low"
}

bbox_pct is [x1, y1, x2, y2] in PIXELS of this {width}x{height} image,
measured from the TOP-LEFT corner. Draw the box TIGHTLY around the text it
contains - it is used to highlight the exact region on the drawing.
Use confidence "low" for anything small, blurry, or partially obscured. If a
value is illegible, set text to null and confidence to "low". No prose
outside the JSON object.{ocr_section}"""

# Appended to the prompt when Textract OCR lines are available. The OCR pass
# reads the FULL-resolution image, so it sees small text the downscaled
# vision image may not - transcribe against it rather than squinting.
OCR_SECTION = """

MACHINE OCR REFERENCE (read from the full-resolution image; use it to get
exact characters and numbers right, especially small text; it has no
understanding, so YOU still decide what is a region and what type it is;
do not invent regions for OCR fragments that are not meaningful text):
{ocr_lines}"""

_REGION_MAP = {
    "note": RegionType.note,
    "dimension": RegionType.dimension,
    "title_block": RegionType.title_block,
    "bom": RegionType.bom,
    "component": RegionType.component,
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

    # OCR context cap: enough for a dense sheet without flooding the prompt
    MAX_OCR_LINES = 250

    def analyze(self, data: bytes, ocr_lines: list[dict] | None = None) -> list[VisionRegion]:
        """Run the vision model on raw image bytes and return regions with
        percentage bboxes. Coordinate-space-agnostic so both photo uploads and
        rasterized scanned-PDF pages can reuse it. Returns [] if nothing found;
        callers decide whether an empty result is an error.

        ocr_lines: precomputed Textract lines (tests / reuse); None fetches
        them, [] skips OCR entirely."""
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
        except UnidentifiedImageError:
            raise InvalidFile("This file is not a valid image - it appears to be corrupt.")

        if ocr_lines is None:
            # full-resolution OCR pass; [] on any failure (graceful fallback)
            ocr_lines = textract_lines(data)

        sent_bytes, sent_w, sent_h = self._downscale(data)
        # .replace, not .format - the prompt's JSON examples contain braces
        prompt = PROMPT.replace("{width}", str(sent_w)).replace("{height}", str(sent_h))
        if ocr_lines:
            listing = "\n".join(
                f'- "{line["text"]}"' for line in ocr_lines[: self.MAX_OCR_LINES]
            )
            prompt = prompt.replace(
                "{ocr_section}", OCR_SECTION.replace("{ocr_lines}", listing)
            )
        else:
            prompt = prompt.replace("{ocr_section}", "")
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
            # component groups carry every instance; first doubles as bbox
            extra: list[list[float]] = []
            instances = item.get("instances")
            if isinstance(instances, list):
                for inst in instances:
                    if isinstance(inst, list) and len(inst) == 4:
                        converted = self._to_pct(inst, sent_w, sent_h)
                        if converted:
                            extra.append(converted)
            if extra and bbox_pct is None:
                bbox_pct = extra[0]
            if bbox_pct in extra:
                extra = [b for b in extra if b != bbox_pct]
            text = item.get("text")
            n_instances = (1 if bbox_pct else 0) + len(extra)
            label = str(text).strip() if text else None
            if label and n_instances > 1:
                label = f"{label} — {n_instances} instances"
            confidence = item.get("confidence")
            regions.append(
                VisionRegion(
                    region_type=_REGION_MAP.get(item.get("type"), RegionType.note),
                    text=label,
                    confidence=(
                        Confidence(confidence)
                        if confidence in {c.value for c in Confidence}
                        else Confidence.medium
                    ),
                    bbox_pct=bbox_pct,
                    extra_bboxes_pct=extra or None,
                )
            )
        if ocr_lines:
            self._snap_to_ocr(regions, ocr_lines)
        return regions

    @staticmethod
    def _snap_to_ocr(regions: list[VisionRegion], ocr_lines: list[dict]) -> None:
        """Replace approximate vision bboxes with Textract's pixel-accurate
        ones when a region's text matches exactly one OCR line. A confirmed
        character-for-character OCR match also upgrades confidence: the value
        was machine-read at full resolution, not transcribed from a
        downscaled image."""
        by_text: dict[str, list[dict]] = {}
        for line in ocr_lines:
            by_text.setdefault(normalize_text(line["text"]), []).append(line)
        for region in regions:
            if not region.text:
                continue
            matches = by_text.get(normalize_text(region.text))
            if not matches or len(matches) != 1:
                continue  # absent or ambiguous (repeated text) - keep vision box
            line = matches[0]
            region.bbox_pct = list(line["bbox_pct"])
            if line["confidence"] >= 90 and region.confidence != Confidence.high:
                region.confidence = Confidence.high

    @staticmethod
    def _pct_to_extents(pct: list[float], xmax: float, ymax: float) -> list[float]:
        x1, y1, x2, y2 = pct
        # percentages from top-left -> extents coords, y-up
        return [
            round(x1 / 100 * xmax, 1),
            round(ymax - (y2 / 100 * ymax), 1),
            round(x2 / 100 * xmax, 1),
            round(ymax - (y1 / 100 * ymax), 1),
        ]

    @staticmethod
    def region_to_chunk(
        region: VisionRegion, xmax: float, ymax: float, page: int = 1
    ) -> ProvisionalChunk:
        """Map a percentage-space region into a chunk whose bbox is in the
        renderer's y-up coordinate space (extents [0, 0, xmax, ymax])."""
        bbox = None
        if region.bbox_pct is not None:
            bbox = ImageExtractor._pct_to_extents(region.bbox_pct, xmax, ymax)
        extra = None
        if region.extra_bboxes_pct:
            extra = [
                ImageExtractor._pct_to_extents(b, xmax, ymax)
                for b in region.extra_bboxes_pct
            ]
        return ProvisionalChunk(
            region_type=region.region_type,
            chunk_text=region.text,
            is_drawing=region.is_drawing,
            bbox=bbox,
            confidence=region.confidence,
            page=page,
            extra_bboxes=extra,
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
