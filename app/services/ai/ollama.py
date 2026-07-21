"""Local Ollama provider - used while AWS is unavailable and for offline dev."""
import base64
import json

import httpx

from app.config import settings
from app.exceptions import VisionUnavailable


def _stream_chat(payload: dict) -> str:
    """POST to Ollama /api/chat with stream=True and concatenate the message
    content. Streaming matters for slow local models: with a single non-streamed
    response the read timeout must cover the ENTIRE generation, so a long answer
    (e.g. a detailed scanned drawing) trips httpx.ReadTimeout and the whole
    request fails. Streaming makes the timeout a per-chunk gap instead, so a slow
    but progressing generation completes. The timeout below is that per-chunk gap
    (also covers model load + time-to-first-token)."""
    payload = {**payload, "stream": True}
    timeout = httpx.Timeout(settings.ollama_read_timeout, connect=30.0)
    with httpx.stream(
        "POST", f"{settings.ollama_base_url}/api/chat", json=payload, timeout=timeout
    ) as resp:
        if resp.status_code == 404:
            resp.read()
            raise VisionUnavailable(
                f"Model '{payload['model']}' is not installed. Run: ollama pull {payload['model']}"
            )
        resp.raise_for_status()
        parts = []
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            parts.append(chunk.get("message", {}).get("content", ""))
            if chunk.get("done"):
                break
        return "".join(parts)


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
        return _stream_chat(
            {
                "model": settings.ollama_text_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        )


class OllamaVision:
    def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        try:
            return _stream_chat(
                {
                    "model": settings.ollama_vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [base64.b64encode(image_bytes).decode()],
                        }
                    ],
                }
            )
        except httpx.ConnectError:
            raise VisionUnavailable(
                "Image extraction is unavailable: the local AI service (Ollama) is not running."
            )
        except httpx.TimeoutException:
            raise VisionUnavailable(
                "Image extraction timed out: the local vision model is taking too long to read "
                "this drawing. Try a smaller or lower-resolution image, or switch extraction to "
                "the faster cloud vision model (AWS Bedrock)."
            )
