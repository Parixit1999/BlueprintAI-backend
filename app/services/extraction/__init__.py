"""Extractor registry keyed by file extension.

To support a new format, add an extractor module and register it here -
upload code and services stay untouched (open/closed).
"""
from app.services.ai import get_vision_provider
from app.services.extraction.base import Extractor
from app.services.extraction.dwg import DwgExtractor
from app.services.extraction.dxf import DxfExtractor
from app.services.extraction.image import ImageExtractor
from app.services.extraction.pdf import PdfExtractor

_FACTORIES = {
    ".dxf": lambda: DxfExtractor(),
    ".dwg": lambda: DwgExtractor(),  # via ODA converter when configured
    # PDF gets a vision extractor too, for the scanned-PDF (no text layer) fallback
    ".pdf": lambda: PdfExtractor(ImageExtractor(get_vision_provider())),
    ".png": lambda: ImageExtractor(get_vision_provider()),
    ".jpg": lambda: ImageExtractor(get_vision_provider()),
    ".jpeg": lambda: ImageExtractor(get_vision_provider()),
    ".tif": lambda: ImageExtractor(get_vision_provider()),
    ".tiff": lambda: ImageExtractor(get_vision_provider()),
    ".bmp": lambda: ImageExtractor(get_vision_provider()),
    ".webp": lambda: ImageExtractor(get_vision_provider()),
}

# Formats we recognize but cannot parse, with actionable guidance instead of
# a generic "unsupported" error.
FORMAT_GUIDANCE = {
    ".rvt": (
        "Revit models (.rvt) are a proprietary Autodesk format that cannot be "
        "parsed directly. From Revit, export the sheets as PDF (or the model as "
        "DXF/DWG) and upload those - the drawings will be fully searchable."
    ),
}


def supported_extensions() -> set[str]:
    return set(_FACTORIES)


def get_extractor(suffix: str) -> Extractor | None:
    factory = _FACTORIES.get(suffix.lower())
    return factory() if factory else None
