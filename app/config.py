from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_region: str = "us-east-1"
    s3_bucket: str = "blueprintai-files"
    # Set to http://localhost:9000 for local MinIO; leave unset for real S3
    s3_endpoint_url: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    bedrock_vision_model: str = "anthropic.claude-opus-4-8"
    bedrock_text_model: str = "anthropic.claude-opus-4-8"
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"

    # "ollama" (local) or "bedrock" (AWS) — controls embeddings + generation
    ai_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "mxbai-embed-large"  # 1024-dim, same as Titan v2
    ollama_text_model: str = "llama3.1:8b"
    ollama_vision_model: str = "qwen2.5vl:7b"
    # Vision/generation responses are streamed, so this is the max gap between
    # chunks (covers model load + time-to-first-token), not the whole-response
    # budget. Local vision on a detailed scanned drawing can be slow.
    ollama_read_timeout: float = 300.0

    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB

    # Two documents whose embeddings are at least this cosine-similar are
    # flagged as possible duplicates. Calibrated on real drawings: the same
    # drawing across formats scores ~0.98, genuinely different parts ~0.59.
    duplicate_similarity_threshold: float = 0.90

    # Single-user mode for now; auth later just replaces this per-request.
    default_user_id: str = "global"

    database_url: str = "postgresql://postgres:postgres@localhost:5432/blueprintai"


settings = Settings()
