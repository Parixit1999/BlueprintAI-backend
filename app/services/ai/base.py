"""Provider-agnostic AI interfaces (dependency inversion boundary).

Services depend on these protocols only; concrete providers (Ollama, Bedrock)
are selected by the factory in __init__.py. Adding a provider means adding a
module here - no changes to services or routers.
"""
from collections.abc import Iterator
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...


class TextGenerator(Protocol):
    def generate(self, system: str, user: str, image: bytes | None = None) -> str:
        """Generate an answer; `image` optionally lets multimodal providers
        SEE the drawing being discussed. Text-only providers ignore it."""
        ...

    def generate_stream(
        self, system: str, user: str, image: bytes | None = None
    ) -> Iterator[str]:
        """Yield the answer in chunks as the model produces them. Providers
        that cannot stream may yield the whole answer once."""
        ...


class VisionProvider(Protocol):
    def analyze_image(self, image_bytes: bytes, prompt: str) -> str: ...
