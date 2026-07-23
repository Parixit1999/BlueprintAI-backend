"""High-precision OCR via Amazon Textract (American-based, pay-per-page).

Textract complements the vision model: it reads text at full resolution
with pixel-accurate word/line boxes but understands nothing about
drawings. The vision model understands drawings but downscales the image
and draws approximate boxes. The hybrid uses each for what it is good at:
Textract lines anchor exact text + location, the vision model classifies
and summarizes.

Best-effort by design: any failure (no permission, oversized image,
throttle) returns [] and extraction proceeds vision-only, so offline/
local development and permission gaps degrade gracefully instead of
breaking uploads.
"""
import io
import logging

from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# Textract synchronous limits: 10 MB payload, 10000 px on either side
_MAX_BYTES = 9_500_000
_MAX_SIDE = 9_800

_client = None


def _get_client():
    global _client
    if _client is None:
        import boto3

        _client = boto3.client("textract", region_name=settings.aws_region)
    return _client


def _fit_for_textract(data: bytes) -> bytes:
    """Resize/re-encode only when the original exceeds Textract's sync
    limits - otherwise send full resolution, which is the entire point."""
    if len(data) <= _MAX_BYTES:
        with Image.open(io.BytesIO(data)) as img:
            if max(img.size) <= _MAX_SIDE and img.format in ("PNG", "JPEG"):
                return data
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        if max(img.size) > _MAX_SIDE:
            img.thumbnail((_MAX_SIDE, _MAX_SIDE))
        quality = 90
        while True:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            if buf.tell() <= _MAX_BYTES or quality <= 50:
                return buf.getvalue()
            quality -= 10


def textract_lines(image_bytes: bytes) -> list[dict]:
    """OCR lines as [{text, confidence (0-100), bbox_pct [x1,y1,x2,y2]}]
    with bbox_pct in 0-100 percentages, top-left origin - the same space
    VisionRegion.bbox_pct uses. [] when Textract is disabled/unavailable."""
    if not settings.textract_enabled:
        return []
    try:
        payload = _fit_for_textract(image_bytes)
        response = _get_client().detect_document_text(Document={"Bytes": payload})
    except Exception as exc:  # AccessDenied, throttle, bad image, no creds...
        logger.info("Textract unavailable, proceeding vision-only: %s", exc)
        return []

    lines = []
    for block in response.get("Blocks", []):
        if block.get("BlockType") != "LINE" or not block.get("Text"):
            continue
        box = block.get("Geometry", {}).get("BoundingBox")
        if not box:
            continue
        x1 = box["Left"] * 100
        y1 = box["Top"] * 100
        lines.append(
            {
                "text": block["Text"].strip(),
                "confidence": round(float(block.get("Confidence", 0.0)), 1),
                "bbox_pct": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x1 + box["Width"] * 100, 2),
                    round(y1 + box["Height"] * 100, 2),
                ],
            }
        )
    return lines


def normalize_text(text: str) -> str:
    """Matching key for snapping vision regions to OCR lines: case- and
    whitespace-insensitive so cosmetic transcription differences still match."""
    return " ".join(text.upper().split())
