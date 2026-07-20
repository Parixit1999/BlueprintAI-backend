"""Local Ollama provider - used while AWS is unavailable and for offline dev."""
import base64

import httpx

from app.config import settings
from app.exceptions import VisionUnavailable


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


class OllamaVision:
    def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        try:
            resp = httpx.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [base64.b64encode(image_bytes).decode()],
                        }
                    ],
                    "stream": False,
                },
                timeout=300,
            )
        except httpx.ConnectError:
            raise VisionUnavailable(
                "Image extraction is unavailable: the local AI service (Ollama) is not running."
            )
        if resp.status_code == 404:
            raise VisionUnavailable(
                f"Image extraction is unavailable: vision model "
                f"'{settings.ollama_vision_model}' is not installed. "
                f"Run: ollama pull {settings.ollama_vision_model}"
            )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
