"""Extractor registry keyed by file extension.

To support a new format, add an extractor module and register it here -
upload code and services stay untouched (open/closed).
"""
import pymupdf
from PIL import Image

# PIL's decompression-bomb guard defaults to ~179 MP, which real wide-format
# drawing scans exceed (an E-size sheet at 600 DPI is ~230 MP). Uploads come
# from authenticated engineers and decode under the extraction concurrency
# cap, so raise the ceiling rather than reject the archive's largest sheets.
# 600 MP still stops genuinely hostile images (a 1 GP bomb never decodes).
Image.MAX_IMAGE_PIXELS = 600_000_000

from app.services.ai import get_text_generator, get_vision_provider
from app.services.extraction.base import Extractor
from app.services.extraction.dwg import DwgExtractor
from app.services.extraction.dxf import DxfExtractor
from app.services.extraction.image import ImageExtractor
from app.services.extraction.pdf import PdfExtractor
from app.services.extraction.rvt import RvtExtractor

_ONE_SHEET_IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".heic", ".heif"}


def document_page_count(path: str, suffix: str) -> int | None:
    """How many sheets the stored document actually has, read from the
    document itself.

    Everything downstream used to infer this from the extracted regions
    ("the highest page we produced a chunk for"). On a large set that is
    wrong the moment extraction is capped: a 229-page plan set whose
    extraction stopped at page 139 reported 139 sheets, rendered only 139,
    and left the last 90 unreachable in the viewer.

    Returns None for formats whose sheet count cannot be read cheaply here
    (CAD), so the caller keeps its previous behaviour for those.
    """
    suffix = suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(path) as doc:
            return doc.page_count or None
    if suffix in (".tif", ".tiff"):
        with Image.open(path) as im:  # scanned drawings are often multi-page
            return getattr(im, "n_frames", 1) or 1
    if suffix in _ONE_SHEET_IMAGES:
        return 1
    return None

_FACTORIES = {
    # CAD text comes from entities (exact); the vision extractor adds what a
    # structural read cannot see - drawn components and a sheet summary
    ".dxf": lambda: DxfExtractor(ImageExtractor(get_vision_provider())),
    ".dwg": lambda: DwgExtractor(  # LibreDWG (bundled) or ODA converter
        ImageExtractor(get_vision_provider())
    ),
    # PDF gets a vision extractor too, for the scanned-PDF (no text layer) fallback
    ".pdf": lambda: PdfExtractor(
        ImageExtractor(get_vision_provider()), generator=get_text_generator()
    ),
    # best-effort: embedded preview + metadata, with a limitation note
    ".rvt": lambda: RvtExtractor(ImageExtractor(get_vision_provider())),
    ".png": lambda: ImageExtractor(get_vision_provider()),
    ".jpg": lambda: ImageExtractor(get_vision_provider()),
    ".jpeg": lambda: ImageExtractor(get_vision_provider()),
    ".tif": lambda: ImageExtractor(get_vision_provider()),
    ".tiff": lambda: ImageExtractor(get_vision_provider()),
    ".bmp": lambda: ImageExtractor(get_vision_provider()),
    ".webp": lambda: ImageExtractor(get_vision_provider()),
    ".heic": lambda: ImageExtractor(get_vision_provider()),
    ".heif": lambda: ImageExtractor(get_vision_provider()),
}

# Formats we recognize but cannot parse, with actionable guidance instead of
# a generic "unsupported" error. (Currently none - every recognized format
# has at least best-effort extraction.)
FORMAT_GUIDANCE: dict[str, str] = {}


def supported_extensions() -> set[str]:
    return set(_FACTORIES)


def get_extractor(suffix: str) -> Extractor | None:
    factory = _FACTORIES.get(suffix.lower())
    return factory() if factory else None
