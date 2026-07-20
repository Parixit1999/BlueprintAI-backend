"""Provider-agnostic AI interfaces (dependency inversion boundary).

Services depend on these protocols only; concrete providers (Ollama, Bedrock)
are selected by the factory in __init__.py. Adding a provider means adding a
module here - no changes to services or routers.
"""
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...


class TextGenerator(Protocol):
    def generate(self, system: str, user: str) -> str: ...
