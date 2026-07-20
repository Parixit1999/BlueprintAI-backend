"""Extractor registry keyed by file extension.

To support a new format, add an extractor module and register it here -
upload code and services stay untouched (open/closed).
"""
from app.services.ai import get_vision_provider
from app.services.extraction.base import Extractor
from app.services.extraction.dxf import DxfExtractor
from app.services.extraction.image import ImageExtractor
from app.services.extraction.pdf import PdfExtractor

_FACTORIES = {
    ".dxf": lambda: DxfExtractor(),
    ".pdf": lambda: PdfExtractor(),
    ".png": lambda: ImageExtractor(get_vision_provider()),
    ".jpg": lambda: ImageExtractor(get_vision_provider()),
    ".jpeg": lambda: ImageExtractor(get_vision_provider()),
}


def supported_extensions() -> set[str]:
    return set(_FACTORIES)


def get_extractor(suffix: str) -> Extractor | None:
    factory = _FACTORIES.get(suffix.lower())
    return factory() if factory else None
