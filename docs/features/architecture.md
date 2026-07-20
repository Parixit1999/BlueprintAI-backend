# Architecture & local dev stack

## Layers (SOLID)
- `app/routers/*` — thin HTTP adapters; validation + delegate. No business logic.
- `app/services/*` — orchestration; constructor-injected dependencies.
- `app/repositories.py` — the ONLY place SQL lives (FileRepository,
  ChunkRepository, ChatRepository, StatsRepository).
- `app/db.py` — psycopg connection pool, opened/closed by the FastAPI lifespan.
- `app/dependencies.py` — composition root: builds services with their deps.
- `app/config.py` — pydantic-settings from `.env`.
- `app/exceptions.py` — typed domain errors; `app/main.py` maps them to HTTP.

## Provider abstraction
- `services/ai/base.py` — `EmbeddingProvider`, `TextGenerator`, `VisionProvider`
  protocols. `services/ai/__init__.py` is the factory keyed on
  `settings.ai_provider`. Ollama and Bedrock implementations sit side by side.
- `services/storage.py` — `ObjectStorage` protocol; S3 impl targets MinIO or S3.
- `services/extraction/` — `Extractor` protocol + registry keyed by file
  extension (`.dxf`, `.pdf`, `.png/.jpg/.jpeg`). Add a format = add a module.

## Local dev
- `docker-compose.yml`: Postgres+pgvector and MinIO. `db/init.sql` is the schema.
- Everything runs offline (Ollama for AI, MinIO for S3, Docker for DB) — built
  this way because the AWS account was pending activation. Flip to AWS by
  setting `AI_PROVIDER=bedrock` and unsetting `S3_ENDPOINT_URL`.
