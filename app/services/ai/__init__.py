"""Provider factory - the single place that maps config to implementations."""
from functools import lru_cache

from app.config import settings
from app.services.ai.base import EmbeddingProvider, TextGenerator


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    if settings.ai_provider == "bedrock":
        from app.services.ai.bedrock import BedrockEmbedding

        return BedrockEmbedding()
    from app.services.ai.ollama import OllamaEmbedding

    return OllamaEmbedding()


@lru_cache(maxsize=1)
def get_text_generator() -> TextGenerator:
    if settings.ai_provider == "bedrock":
        from app.services.ai.bedrock import BedrockGenerator

        return BedrockGenerator()
    from app.services.ai.ollama import OllamaGenerator

    return OllamaGenerator()
