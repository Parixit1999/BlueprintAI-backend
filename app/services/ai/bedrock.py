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
    def embed(self, text: str) -> list[float]:
        resp = _client().invoke_model(
            modelId=settings.bedrock_embed_model,
            body=json.dumps({"inputText": text, "dimensions": 1024}),
        )
        return json.loads(resp["body"].read())["embedding"]


class BedrockGenerator:
    def generate(self, system: str, user: str) -> str:
        resp = _client().converse(
            modelId=settings.bedrock_text_model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": 1024},
        )
        return resp["output"]["message"]["content"][0]["text"]

    def generate_stream(self, system: str, user: str):
        """converse_stream yields deltas as Bedrock produces them. Falls back
        to a single whole-answer chunk if the streaming API errors, so the
        chat contract holds either way. Untestable until the AWS account is
        unblocked - verify at migration."""
        try:
            resp = _client().converse_stream(
                modelId=settings.bedrock_text_model,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": 1024},
            )
            for event in resp["stream"]:
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                if "text" in delta:
                    yield delta["text"]
        except Exception:
            yield self.generate(system, user)


class BedrockVision:
    def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        # the extractor sends PNG normally, JPEG when PNG would exceed the
        # size limit - detect from magic bytes rather than widening the API
        fmt = "png" if image_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "jpeg"
        resp = _client().converse(
            modelId=settings.bedrock_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
                        {"text": prompt},
                    ],
                }
            ],
            # dense archive sheets extract dozens of regions; a low cap
            # truncates the JSON mid-array and the whole extraction fails
            inferenceConfig={"maxTokens": 32000},
        )
        return resp["output"]["message"]["content"][0]["text"]
