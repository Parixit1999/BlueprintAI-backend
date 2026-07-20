"""Image extraction via a vision model (Ollama locally, Bedrock Claude on AWS).

Vision output is the least reliable extraction path, which is exactly why every
field carries model-reported confidence and flows through HITL review before
ingestion. The model is instructed to return null rather than guess.
"""
import io
import json
import re

from PIL import Image, UnidentifiedImageError

from app.exceptions import ExtractionFailed, InvalidFile
from app.schemas import Confidence, ProvisionalChunk, RegionType
from app.services.ai.base import VisionProvider

PROMPT = """You are extracting content from an engineering drawing image.

Return ONLY a JSON array. Each element describes one text region:
{
  "text": "the exact text, or null if illegible - NEVER guess",
  "type": "note" | "dimension" | "title_block" | "bom",
  "bbox_pct": [x1, y1, x2, y2],
  "confidence": "high" | "medium" | "low"
}

bbox_pct values are percentages (0-100) of image width/height measured from
the TOP-LEFT corner. Use confidence "low" for anything small, blurry, or
partially obscured. If a value is illegible, set text to null and confidence
to "low". Do not include any prose outside the JSON array."""

_REGION_MAP = {
    "note": RegionType.note,
    "dimension": RegionType.dimension,
    "title_block": RegionType.title_block,
    "bom": RegionType.bom,
}


def _parse_response(raw: str) -> list[dict]:
    # Models often wrap JSON in code fences or add prose - extract the array.
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

    def extract(self, path: str) -> list[ProvisionalChunk]:
        data = open(path, "rb").read()
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
            with Image.open(io.BytesIO(data)) as img:
                width, height = img.size
        except UnidentifiedImageError:
            raise InvalidFile("This file is not a valid image - it appears to be corrupt.")

        sent_bytes, sent_w, sent_h = self._downscale(data)
        raw = self._vision.analyze_image(sent_bytes, PROMPT)
        chunks: list[ProvisionalChunk] = []
        for item in _parse_response(raw):
            if not isinstance(item, dict):
                continue
            bbox = None
            pct = item.get("bbox_pct")
            if isinstance(pct, list) and len(pct) == 4:
                normalized = self._to_pct(pct, sent_w, sent_h)
                if normalized is not None:
                    x1, y1, x2, y2 = normalized
                    # percentages from top-left -> pixel coords, y-up
                    bbox = [
                        round(x1 / 100 * width, 1),
                        round(height - (y2 / 100 * height), 1),
                        round(x2 / 100 * width, 1),
                        round(height - (y1 / 100 * height), 1),
                    ]
            text = item.get("text")
            confidence = item.get("confidence")
            chunks.append(
                ProvisionalChunk(
                    region_type=_REGION_MAP.get(item.get("type"), RegionType.note),
                    chunk_text=str(text).strip() if text else None,
                    bbox=bbox,
                    confidence=(
                        Confidence(confidence)
                        if confidence in {c.value for c in Confidence}
                        else Confidence.medium
                    ),
                )
            )

        if not chunks:
            raise ExtractionFailed(
                "No text regions were detected in this image. If the drawing has "
                "readable text, try a higher-resolution or better-lit image."
            )
        return chunks
