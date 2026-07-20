from typing import Protocol

from app.schemas import ProvisionalChunk


class Extractor(Protocol):
    """Turns one drawing file into provisional chunks awaiting HITL review."""

    def extract(self, path: str) -> list[ProvisionalChunk]: ...
