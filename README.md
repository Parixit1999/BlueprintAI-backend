# BlueprintAI Backend

BlueprintAI replaces a spreadsheet-based engineering drawing registry with a secure web
application: engineers upload drawings in any common format, the system reads them with
AI, files them under projects / drawings / sets / versions, and answers plain-English
questions about their content — every answer cited down to the exact highlighted region
of the exact drawing it came from.

**Supported formats:** PDF (vector + scanned) · DXF · DWG (via bundled LibreDWG) ·
RVT (best-effort: embedded preview + metadata) · PNG · JPG/JPEG · TIF/TIFF · BMP ·
WEBP · HEIC/HEIF

## Architecture

```
Browser ── HTTPS ── CloudFront ──┬── S3 (React frontend)
                                 └── ALB ── Fargate (this API) ──┬── RDS Postgres + pgvector
                                                                 ├── S3 (drawing files)
                                                                 ├── Bedrock (Claude vision/text + Titan embeddings)
                                                                 └── Textract (full-resolution OCR)
```

Local development runs the same container against the same cloud services (RDS,
S3, Bedrock), so dev and prod behave identically. A fully offline mode exists —
see below.

## Prerequisites

- Docker Desktop
- AWS CLI v2, signed in via `aws login` with access to the project account
  (Bedrock, S3, RDS/Secrets Manager). Sessions expire ~12h; re-run `aws login`
  and `docker restart blueprintai-backend` when API calls start failing.
- The sibling repo `BlueprintAI-frontend` checked out next to this one
  (the compose stack builds the frontend from `../BlueprintAI-frontend`).

## Run

```bash
aws login        # once per ~12h session
make up          # builds and starts db + backend + frontend
```

- App: http://localhost:5175 · API: http://localhost:8000/api (docs at /docs)
- **First run:** an `admin` account is created with a random password printed once
  in the backend logs — `docker logs blueprintai-backend | grep password` — sign
  in and change it (account menu, bottom-left).

### Offline / local-storage mode

```bash
make up-local    # MinIO object storage + local Postgres instead of S3/RDS
```

AI stays on Bedrock (needs internet). For a fully offline AI, set
`AI_PROVIDER=ollama` — note that switching embedding models requires
re-ingesting the knowledge base. Details: `docs/LOCAL-INFRA.md`.

## Key configuration (docker-compose.yml / .env)

| Variable | Purpose |
|---|---|
| `DATABASE_SECRET_ID` + `DATABASE_HOST` | RDS credentials resolve from Secrets Manager at startup (no password in files) |
| `DATABASE_URL` | direct Postgres URL (local mode fallback) |
| `S3_BUCKET` (+ optional `S3_ENDPOINT_URL`) | drawing storage; endpoint set = MinIO |
| `AI_PROVIDER` | `bedrock` (default) or `ollama` |
| `TEXTRACT_ENABLED` | hybrid OCR on/off (degrades gracefully without access) |
| `INITIAL_ADMIN_PASSWORD` | optional deterministic seed password |

## Deploying to AWS

Backend: `docker buildx build --platform linux/amd64 -t <account>.dkr.ecr.us-east-1.amazonaws.com/blueprintai-backend:latest --push .`
then `aws ecs update-service --cluster blueprintai --service backend --force-new-deployment`.
Frontend: build with `VITE_API_BASE=/api`, `aws s3 sync dist s3://<web bucket>`, invalidate CloudFront.
Rolling deploys are zero-downtime behind the ALB.

## More documentation

- `docs/LOCAL-INFRA.md` — the local stack in depth
- `docs/SECURITY.md` — auth, secrets, network posture
- `docs/EDGE-CASES.md` — the edge-case catalog and how each is handled
- `docs/features/` — feature-by-feature reference
