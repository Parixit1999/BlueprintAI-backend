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

    database_url: str = "postgresql://postgres:postgres@localhost:5432/blueprintai"


settings = Settings()
