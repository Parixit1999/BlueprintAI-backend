from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_region: str = "us-east-1"
    s3_bucket: str = "blueprintai-files"
    # Set to http://localhost:9000 for local MinIO; leave unset for real S3
    s3_endpoint_url: str | None = None
    # Endpoint browsers can reach, used only for presigned URLs. Needed when
    # the backend runs in docker (it reaches MinIO as http://minio:9000, a
    # hostname the user's browser cannot resolve). Defaults to s3_endpoint_url.
    s3_public_endpoint_url: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    # Current Claude models on Bedrock require the cross-region inference
    # profile ID (us. prefix) - the bare model ID is rejected for on-demand
    bedrock_vision_model: str = "us.anthropic.claude-opus-4-8"
    bedrock_text_model: str = "us.anthropic.claude-opus-4-8"
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"

    # "ollama" (local) or "bedrock" (AWS) — controls embeddings + generation
    ai_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "snowflake-arctic-embed:335m"  # 1024-dim, same as Titan v2
    ollama_text_model: str = "llama3.1:8b"
    ollama_vision_model: str = "llama3.2-vision:11b"
    # Vision/generation responses are streamed, so this is the max gap between
    # chunks (covers model load + time-to-first-token), not the whole-response
    # budget. Local vision on a detailed scanned drawing can be slow.
    ollama_read_timeout: float = 300.0

    # DWG support: path to the free ODA File Converter binary; when unset,
    # DWG uploads get guidance to export DXF/PDF instead.
    oda_converter_path: str | None = None

    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB

    # Two documents whose embeddings are at least this cosine-similar are
    # flagged as possible duplicates. Calibrated on real drawings: the same
    # drawing across formats scores ~0.98, genuinely different parts ~0.59.
    duplicate_similarity_threshold: float = 0.90

    # Single-user mode for now; auth later just replaces this per-request.
    default_user_id: str = "global"

    database_url: str = "postgresql://postgres:postgres@localhost:5432/blueprintai"


settings = Settings()
