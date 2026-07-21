# Local infrastructure

The entire stack runs locally with one command. AWS migration later is a
config swap (AI_PROVIDER=bedrock, unset S3_ENDPOINT_URL), not a rewrite.

## Prerequisites
- Docker Desktop
- Native Ollama (`brew install ollama && brew services start ollama`) -
  native rather than containerized because Docker on macOS cannot use the
  Apple GPU and the vision model is unusably slow on CPU
- The frontend checkout as a sibling directory: `../BlueprintAI-frontend`

## Commands (from this repo)
| Command | Does |
|---|---|
| `make models` | one-time pull of the American-based models (Llama 3.2 Vision, Llama 3.1, Snowflake Arctic Embed) |
| `make up` | build + start everything: Postgres/pgvector, MinIO (+ bucket init), backend API, frontend |
| `make health` | probe backend / frontend / Ollama |
| `make logs` | follow backend logs |
| `make down` | stop (data volumes kept) |
| `make clean` | stop and DELETE all data |

## Endpoints
- App: http://localhost:5175 (5173/5174 are left free for `npm run dev`)
- API + docs: http://localhost:8000/docs
- MinIO console: http://localhost:9001 (minioadmin/minioadmin)

## Notes
- The DB schema applies automatically on first start (db/init.sql); the
  MinIO bucket is created by the one-shot `minio-init` service.
- The backend container reaches host Ollama via `host.docker.internal`
  (extra_hosts covers Linux).
- Healthchecks gate startup order: backend waits for a healthy DB and the
  bucket; `docker compose ps` shows live health.
- Native development still works unchanged: `uvicorn app.main:app` +
  `npm run dev` against the same db/minio containers.
