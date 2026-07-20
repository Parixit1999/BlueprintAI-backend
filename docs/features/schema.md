# Database schema

Source of truth: `db/init.sql`. Two content tables + two chat tables.

## files (one row per uploaded drawing)
- `id`, `filename`, `file_type`, `s3_key`, `status`
  (uploaded→extracted→ingested), `created_at`
- `content_sha256` — exact-content hash (stored signal; not the dedup driver)
- `embedding vector(1024)` — document-level embedding for **semantic** duplicate
  detection (built from the extracted text at extraction time)
- `extraction jsonb` — provisional regions awaiting HITL review
- `render jsonb` — `{pages: {N: {s3_key, extents}}}` cached PNG renders

## chunks (one row per confirmed region, after review)
- `source_file_id` (FK, ON DELETE CASCADE), `page`, `region_type`
  (title_block/dimension/note/bom/view), `chunk_text`
- `bbox float8[]` — `[x1,y1,x2,y2]` on the drawing (the evidence pointer)
- `image_uri` — reserved for a per-region crop (currently unused; we render the
  full page and highlight instead)
- `confidence` (high/medium/low), `verification_status`
  (unverified/confirmed/corrected)
- `original_value` + `corrected_value` — the human-correction audit log
- `embedding vector(1024)` — for retrieval

## chat_sessions / chat_messages
- sessions: `user_id` (defaults `'global'` — single-user now, auth later),
  `title`, timestamps.
- messages: `role`, `content`, `evidence jsonb` (retrieval refs on assistant
  turns), FK cascade from session.
