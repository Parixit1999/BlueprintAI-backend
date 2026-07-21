"""Image/document enhancement before vision ingestion.

Some scans and photos are unsuitable for direct ingestion: rotated phone
photos (EXIF orientation), faint low-contrast scans, tiny thumbnails. This
module normalizes them BEFORE the vision model sees them, so extraction
quality does not depend on how the drawing was captured.

Geometry rule: only EXIF orientation changes the image geometry, and the
renderer applies the same transpose (see rendering.render_image), so vision
bbox percentages always line up with the preview. Contrast/sharpen/upscale
never alter aspect ratio, and all bboxes are percentage-based.
"""
import io
import logging

from PIL import Image, ImageFilter, ImageOps, ImageStat

logger = logging.getLogger(__name__)

# Below this grayscale standard deviation the scan is considered faint /
# low-contrast and gets autocontrast + a mild sharpen.
LOW_CONTRAST_STDDEV = 40
# Images smaller than this on their longest side are upscaled 2x so small
# text survives the vision model's fixed input size.
MIN_SIDE = 900


def enhance_for_vision(data: bytes) -> tuple[bytes, list[str]]:
    """Return (enhanced PNG bytes, list of applied steps). Falls back to the
    original bytes on any failure - enhancement must never block ingestion."""
    try:
        img = Image.open(io.BytesIO(data))
        applied: list[str] = []

        # 1) EXIF orientation (sideways/upside-down phone photos)
        orientation = img.getexif().get(0x0112, 1) if hasattr(img, "getexif") else 1
        img = ImageOps.exif_transpose(img)
        if orientation != 1:
            applied.append("orientation")

        img = img.convert("RGB")

        # 2) faint scans: stretch contrast, then a mild sharpen
        stddev = ImageStat.Stat(img.convert("L")).stddev[0]
        if stddev < LOW_CONTRAST_STDDEV:
            img = ImageOps.autocontrast(img, cutoff=1)
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
            applied.append("contrast")

        # 3) tiny images: upscale so text is legible at the vision input size
        if max(img.size) < MIN_SIDE:
            img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
            applied.append("upscale")

        if not applied:
            return data, []
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        logger.info("image enhancement applied: %s", ", ".join(applied))
        return buf.getvalue(), applied
    except Exception:
        logger.exception("image enhancement failed; ingesting original")
        return data, []
