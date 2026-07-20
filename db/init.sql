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
