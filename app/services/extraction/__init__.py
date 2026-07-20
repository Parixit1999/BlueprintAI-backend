"""Extractor registry keyed by file extension.

To support a new format, add an extractor module and register it here -
upload code and services stay untouched (open/closed).
"""
from app.services.extraction.base import Extractor
from app.services.extraction.dxf import DxfExtractor

_REGISTRY: dict[str, Extractor] = {
    ".dxf": DxfExtractor(),
}


def supported_extensions() -> set[str]:
    return set(_REGISTRY)


def get_extractor(suffix: str) -> Extractor | None:
    return _REGISTRY.get(suffix.lower())
