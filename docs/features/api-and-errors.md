# Errors & API surface

## Typed domain errors — `app/exceptions.py`
All extend `BlueprintError` (message is user-safe). `app/main.py` has ONE
exception handler mapping them to HTTP so routers stay try/except-free:

| Exception | HTTP | Meaning |
|---|---|---|
| UnsupportedFileType / InvalidFile / ExtractionFailed / RenderFailed | 422 | bad/unprocessable file |
| FileTooLarge | 413 | over `max_upload_bytes` (25 MB) |
| VisionUnavailable | 503 | vision model not reachable |
| FileNotFound | 404 | missing document/session |
| AlreadyIngested | 409 | re-ingest attempt |

Every message is actionable (e.g. scanned-PDF → "upload as an image instead").

## Endpoints
- `POST /files/upload` · `GET /files` · `GET /files/{id}/extraction` ·
  `GET /files/{id}/render?page=N` · `POST /files/{id}/retry` · `DELETE /files/{id}`
- `POST /review/{id}/confirm`
- `POST /chats` · `GET /chats` · `GET /chats/{id}` ·
  `POST /chats/{id}/messages` · `PATCH /chats/{id}` · `DELETE /chats/{id}`
- `GET /stats` · `GET /health`

## Upload validation & failure handling
`FileService.ingest_upload` rejects empty files, unsupported extensions (lists the
supported set), and files over the size cap before extraction. Bulk/zip upload is
handled client-side (frontend expands zips and calls this endpoint per file).

**Failed extraction keeps the row** with `status='failed'` and the message in the
`files.error` column (NOT deleted) — the UI shows the error on the Documents list
with a Retry action. `POST /files/{id}/retry` re-runs extraction from the stored
original (allowed only for `failed`/`uploaded` rows). `mark_extracted` clears
`error` on success.

## Vision hang protection
`OllamaVision.analyze_image` streams the response (per-chunk gap timeout
`ollama_read_timeout`, default 300s) AND wraps the whole call in a helper thread
with an absolute deadline of `2 x ollama_read_timeout`. Rationale: a wedged
Ollama was observed holding an accepted connection open for 7+ hours without
tripping httpx's read timeout, hanging the extraction worker thread forever and
leaving the row stuck in `uploaded`. The deadline guarantees the failure
surfaces as `ExtractionFailed` and the row flips to `failed`/retryable.
