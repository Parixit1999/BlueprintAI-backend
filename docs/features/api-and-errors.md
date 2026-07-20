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
  `GET /files/{id}/render?page=N` · `DELETE /files/{id}`
- `POST /review/{id}/confirm`
- `POST /chats` · `GET /chats` · `GET /chats/{id}` ·
  `POST /chats/{id}/messages` · `PATCH /chats/{id}` · `DELETE /chats/{id}`
- `GET /stats` · `GET /health`

## Upload validation
`FileService.ingest_upload` rejects empty files, unsupported extensions (lists the
supported set), and files over the size cap before extraction. Failed extraction
deletes the just-created row (no orphans). Bulk/zip upload is handled client-side
(frontend expands zips and calls this endpoint per file).
