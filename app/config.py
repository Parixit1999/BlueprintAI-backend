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

    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB

    database_url: str = "postgresql://postgres:postgres@localhost:5432/blueprintai"


settings = Settings()
