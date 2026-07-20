"""Amazon Bedrock provider - Titan embeddings + Claude generation."""
import json
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _client():
    import boto3

    return boto3.client("bedrock-runtime", region_name=settings.aws_region)


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
