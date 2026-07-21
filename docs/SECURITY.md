# Security posture

How each data class is handled, what is enforced today, and what is
deliberately deferred. Honest by design: this is a single-user local MVP with
an AWS migration path; items marked *deferred* are architectural slots, not
afterthoughts.

## Secure handling by data class

| Data | Storage | Protections |
|---|---|---|
| Projects & metadata | Postgres | Parameterized SQL everywhere (repositories are the only SQL layer); typed domain errors so internals never leak in messages |
| Uploaded files (originals) | MinIO/S3 object storage | Extension allowlist; 25 MB size cap; content validated by parsers (corrupt files -> clean 422); filenames sanitized to a basename with control/path characters stripped before any storage key or DB row is built |
| Drawing files / versions / sets | Postgres (registry) | FK integrity with explicit ON DELETE behavior (files detach rather than vanish; version groups split safely); cycle-protected folder moves |
| Chat sessions & messages | Postgres | Scoped by `user_id` on every query (single "global" user today - the auth seam is already in the schema and service signatures); input length caps |
| RAG indexes (chunks, registry cards, embeddings) | Postgres + pgvector | Derived data only - rebuildable from originals (`/registry/reindex`); retrieval scoped by project through SQL joins, not client-side filtering |
| Retrieved information / generated answers / citations | Postgres (`chat_messages.evidence`, `version_context`) | Answers are grounded-only by prompt contract; every claim carries evidence the user can inspect; answers are stored with their citations so history stays auditable |
| User feedback | Postgres (`answer_feedback` + clamped weights) | One rating per message (upsert with delta application - re-rating can never double-count); retrieval weights clamped to [0.3, 2.0] so no amount of feedback can bury or fabricate relevance |

## Enforced today

- **SQL injection**: psycopg parameterized queries only; no string-built values.
- **Upload abuse**: allowlist + size cap + parser validation + filename
  sanitization (`../../evil/../weird.dxf` -> `weird.dxf`, verified).
- **Input caps**: chat and query questions capped at 2000 chars; `top_k`
  bounded; folder/file/project names length-capped via Pydantic.
- **Event-loop safety**: all blocking work (extraction, LLM, DB) runs in
  worker threads - one slow request cannot starve the API.
- **External-service hangs**: vision calls carry socket timeouts plus an
  absolute deadline; a wedged model surfaces as a retryable failure, never a
  hang.
- **CORS**: locked to the local dev origins.
- **Error hygiene**: domain errors map to user-safe messages via one handler;
  unexpected exceptions return a generic 500 without internals.
- **No secrets in code**: connection settings come from environment/config;
  production plan is AWS Secrets Manager (see below).

## Deferred (by explicit product decision), with the seam already built

- **Authentication/authorization**: the client requested no login for the
  pilot. Every chat query already filters by `user_id`, and services take the
  user identity as a parameter - real auth (e.g. Cognito/JWT) slots in at the
  router layer without schema changes.
- **Rate limiting / audit log**: add at the reverse-proxy/API-gateway layer in
  the AWS deployment.
- **Encryption at rest**: local Docker volumes are unencrypted; the AWS plan
  uses RDS + S3 with encryption enabled and Secrets Manager for credentials.
- **Prompt injection**: drawing content is untrusted input to the LLM. The
  system prompt constrains answers to provided context and every answer is
  citation-checked by a human-visible evidence trail, which bounds the blast
  radius; a dedicated injection filter is future work for multi-tenant use.

## AI model provenance (American-based requirement)

All models in both deployment modes are from US companies:

| Role | Local (Ollama) | AWS (Bedrock) |
|---|---|---|
| Vision extraction | Llama 3.2 Vision - Meta (US) | Anthropic Claude (US) |
| Answer generation | Llama 3.1 - Meta (US) | Anthropic Claude (US) |
| Embeddings (1024-dim) | Snowflake Arctic Embed - Snowflake (US) | Amazon Titan Text Embeddings v2 (US) |

Both embedding models are 1024-dimensional, so the pgvector schema is
identical across modes and no migration is needed when switching providers.
