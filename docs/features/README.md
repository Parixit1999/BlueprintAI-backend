# BlueprintAI Backend — Feature Reference

Concise per-feature notes for future contributors and AI assistants. Feature
names roughly map to the git branch that introduced them.

| Feature | File | Branch |
|---|---|---|
| Architecture & local dev stack | `architecture.md` | feat/local-dev-stack |
| Database schema | `schema.md` | multiple |
| Extraction (DXF / PDF / image) | `extraction.md` | feat/local-dev-stack, feat/pdf-image-support |
| Rendering & evidence | `rendering.md` | feat/evidence-render, feat/pdf-image-support |
| Embeddings, retrieval & RAG | `rag.md` | feat/local-dev-stack, feat/scoped-retrieval |
| Duplicate detection (semantic) | `duplicate-detection.md` | feat/document-management, feat/embedding-dedup |
| Chat persistence & sessions | `chat.md` | feat/chat-and-stats, feat/chat-session-management |
| Stats | `stats.md` | feat/chat-and-stats |
| Errors & API surface | `api-and-errors.md` | feat/pdf-image-support |

## The core idea
Upload engineering drawings → extract each element as a region with a bounding
box + confidence → human verifies → embed into pgvector → answer questions with
**evidence** (the exact source region highlighted). The bbox is the product: it
carries verifiability through extraction → storage → retrieval → display.

## Stack
- **FastAPI** (Python 3.12), layered: `routers` (HTTP only) → `services`
  (orchestration, DI'd) → `repositories.py` (all SQL) → `db.py` (psycopg pool).
- **AI providers** behind protocols in `services/ai/` with a config factory:
  `AI_PROVIDER=ollama` (local: snowflake-arctic-embed, llama3.1:8b, llama3.2-vision:11b) or
  `bedrock` (Titan + Claude). One env var flips everything.
- **Storage** behind `ObjectStorage`: MinIO locally, S3 in AWS (same boto3 code,
  switched by `S3_ENDPOINT_URL`).
- **DB**: Postgres 16 + pgvector (Docker locally, RDS in AWS).
