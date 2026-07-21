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

PROMPT = """You are extracting content from an engineering drawing image.

Find EVERY piece of visible text in the image - title block fields, drawing
numbers, dimensions, notes, labels. Each one is its own region; do not merge
or skip any.

Return ONLY a JSON object of the form {"regions": [...]} where each element of
"regions" describes one text region:
{
  "text": "the exact text, or null if illegible - NEVER guess",
  "type": "note" | "dimension" | "title_block" | "bom",
  "bbox_pct": [x1, y1, x2, y2],
  "confidence": "high" | "medium" | "low"
}

bbox_pct values are percentages (0-100) of image width/height measured from
the TOP-LEFT corner. Use confidence "low" for anything small, blurry, or
partially obscured. If a value is illegible, set text to null and confidence
to "low". No prose outside the JSON object."""

_REGION_MAP = {
    "note": RegionType.note,
    "dimension": RegionType.dimension,
    "title_block": RegionType.title_block,
    "bom": RegionType.bom,
}


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

    MAX_SIDE = 1024

    @staticmethod
    def _downscale(data: bytes) -> tuple[bytes, int, int]:
        """Send the model a bounded, known-size image so absolute pixel
        coordinates in its output can be mapped back reliably."""
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            if max(img.size) > ImageExtractor.MAX_SIDE:
                img.thumbnail((ImageExtractor.MAX_SIDE, ImageExtractor.MAX_SIDE))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue(), img.width, img.height

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
        raw = self._vision.analyze_image(sent_bytes, PROMPT)
        regions: list[VisionRegion] = []
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
