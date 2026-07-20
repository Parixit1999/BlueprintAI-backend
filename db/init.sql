-- BlueprintAI schema: files + chunks with spatial/confidence metadata.
-- Embedding dimension is 1024 to match Bedrock Titan Text Embeddings v2,
-- so no schema change is needed when switching from local to AWS.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS files (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    filename      text NOT NULL,
    file_type     text NOT NULL,            -- dxf | pdf | image
    s3_key        text NOT NULL,            -- original file location
    status        text NOT NULL DEFAULT 'uploaded',  -- uploaded | extracted | reviewed | ingested
    extraction    jsonb,                    -- provisional chunks awaiting HITL review
    render        jsonb,                    -- {s3_key, extents [xmin,ymin,xmax,ymax]} of the PNG render
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file_id      uuid NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    page                int NOT NULL DEFAULT 1,
    region_type         text NOT NULL,      -- title_block | dimension | note | bom | view
    chunk_text          text NOT NULL,
    bbox                float8[],           -- [x1, y1, x2, y2] on the page
    image_uri           text,               -- S3/MinIO key of the evidence crop
    confidence          text NOT NULL DEFAULT 'high',        -- high | medium | low
    verification_status text NOT NULL DEFAULT 'unverified',  -- unverified | confirmed | corrected
    original_value      text,               -- model output before human correction
    corrected_value     text,               -- human-corrected value, if any
    embedding           vector(1024),
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_file_idx ON chunks (source_file_id);
-- Plain top-k for MVP; an ivfflat/hnsw index can be added when volume warrants it.

-- Chat history. user_id defaults to the single global user; real auth later
-- only needs to start writing real ids here.
CREATE TABLE IF NOT EXISTS chat_sessions (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    text NOT NULL DEFAULT 'global',
    title      text NOT NULL DEFAULT 'New chat',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id uuid NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role       text NOT NULL,       -- user | assistant
    content    text NOT NULL,
    evidence   jsonb,               -- retrieval references on assistant messages
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON chat_messages (session_id, created_at);
