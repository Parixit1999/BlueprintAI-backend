"""Amazon Bedrock provider - Titan embeddings + Claude generation."""
import json
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        # dense archive sheets extract dozens of regions in one response;
        # the default 60s read timeout aborts those long generations
        config=Config(read_timeout=600, connect_timeout=10, retries={"max_attempts": 2}),
    )


class BedrockEmbedding:
    # Titan v2 enforces BOTH 50,000 characters AND 8,192 tokens. Dense CAD
    # text tokenizes at ~3.7 chars/token, so the character clamp must be far
    # below the character limit; the halving retry covers any text that
    # tokenizes even worse. Discovered via a DXF whose document text hit 55k
    # chars / 12k tokens.
    MAX_EMBED_CHARS = 24_000

    def embed(self, text: str) -> list[float]:
        clamped = text[: self.MAX_EMBED_CHARS]
        for _ in range(3):
            try:
                resp = _client().invoke_model(
                    modelId=settings.bedrock_embed_model,
                    body=json.dumps({"inputText": clamped, "dimensions": 1024}),
                )
                return json.loads(resp["body"].read())["embedding"]
            except _client().exceptions.ValidationException:
                if len(clamped) < 2_000:
                    raise
                clamped = clamped[: len(clamped) // 2]
        raise RuntimeError("embedding input could not be reduced to fit model limits")


def _image_block(image: bytes) -> dict:
    fmt = "png" if image[:8] == b"\x89PNG\r\n\x1a\n" else "jpeg"
    return {"image": {"format": fmt, "source": {"bytes": image}}}


def _user_content(user: str, image: bytes | None) -> list[dict]:
    content: list[dict] = []
    if image:
        content.append(_image_block(image))
    content.append({"text": user})
    return content


class BedrockGenerator:
    def generate(self, system: str, user: str, image: bytes | None = None) -> str:
        resp = _client().converse(
            modelId=settings.bedrock_text_model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": _user_content(user, image)}],
            inferenceConfig={"maxTokens": 1024},
        )
        return resp["output"]["message"]["content"][0]["text"]

    def generate_stream(self, system: str, user: str, image: bytes | None = None):
        """converse_stream yields deltas as Bedrock produces them. Falls back
        to a single whole-answer chunk if the streaming API errors, so the
        chat contract holds either way. Untestable until the AWS account is
        unblocked - verify at migration."""
        try:
            resp = _client().converse_stream(
                modelId=settings.bedrock_text_model,
                system=[{"text": system}],
                messages=[{"role": "user", "content": _user_content(user, image)}],
                inferenceConfig={"maxTokens": 1024},
            )
            for event in resp["stream"]:
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                if "text" in delta:
                    yield delta["text"]
        except Exception:
            yield self.generate(system, user, image)


class BedrockVision:
    def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        resp = _client().converse(
            modelId=settings.bedrock_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [_image_block(image_bytes), {"text": prompt}],
                }
            ],
            # dense archive sheets extract dozens of regions; a low cap
            # truncates the JSON mid-array and the whole extraction fails
            inferenceConfig={"maxTokens": 32000},
        )
        return resp["output"]["message"]["content"][0]["text"]
