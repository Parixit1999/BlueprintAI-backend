"""Local Ollama provider - used while AWS is unavailable and for offline dev."""
import httpx

from app.config import settings


class OllamaEmbedding:
    def embed(self, text: str) -> list[float]:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.ollama_embed_model, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]


class OllamaGenerator:
    def generate(self, system: str, user: str) -> str:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_text_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
