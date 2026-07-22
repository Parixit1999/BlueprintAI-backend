-- BlueprintAI schema: files + chunks with spatial/confidence metadata.
-- Embedding dimension is 1024 to match Bedrock Titan Text Embeddings v2,
-- so no schema change is needed when switching from local to AWS.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Phase 1 drawing management: projects contain drawings; drawings may be
-- grouped into sets and may have multiple versions; each drawing has files
-- (sheets or iterations). Mirrors the client's "Drawings Number Book".

CREATE TABLE IF NOT EXISTS projects (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    number      text,                        -- project number, e.g. "1234" (matches pj1234 in filenames)
    name        text NOT NULL,
    description text,
    source      text NOT NULL DEFAULT 'manual',  -- manual | book_import
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS drawing_sets (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid REFERENCES projects(id) ON DELETE SET NULL,
    set_number text NOT NULL,                -- e.g. "12A" from the book's Set # column
    name       text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS drawings (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id       uuid REFERENCES projects(id) ON DELETE SET NULL,
    set_id           uuid REFERENCES drawing_sets(id) ON DELETE SET NULL,
    dwg_number       text,                   -- raw, as recorded ("12158-W-59")
    dwg_number_norm  text,                   -- normalized for matching ("12158-W-59" canonical form)
    description      text,
    contract_number  text,                   -- raw; the book mixes notes into this column
    drawing_date     text,                   -- raw as recorded ("2017-2018", "2018--")
    year             int,                    -- best-effort parsed year for version ordering
    sheet_count      int,
    version_group_id uuid,                   -- drawings sharing this id are versions of the same drawing
    version_note     text,
    source           text NOT NULL DEFAULT 'manual',  -- manual | book_import | upload
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS drawings_project_idx ON drawings (project_id);
CREATE INDEX IF NOT EXISTS drawings_dwg_norm_idx ON drawings (dwg_number_norm);
CREATE INDEX IF NOT EXISTS drawings_version_group_idx ON drawings (version_group_id);

-- Registry metadata as retrievable RAG content: one searchable "card" per
-- project/drawing/set, regenerated whenever the registry changes, so chat can
-- answer questions about projects, drawing metadata, sets, and versions - not
-- just file content.
CREATE TABLE IF NOT EXISTS registry_chunks (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type  text NOT NULL,          -- project | drawing | set
    entity_id    uuid NOT NULL,
    project_id   uuid,                   -- scope filter (project cards point at themselves)
    label        text NOT NULL,          -- display title, e.g. "11767-W-59" or the project name
    project_name text,
    chunk_text   text NOT NULL,          -- the searchable metadata card
    embedding    vector(1024),
    feedback_weight float8 NOT NULL DEFAULT 1.0,  -- RLHF: adjusted by answer ratings
    updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS registry_chunks_entity_idx ON registry_chunks (entity_type, entity_id);
CREATE INDEX IF NOT EXISTS registry_chunks_project_idx ON registry_chunks (project_id);

-- File manager: folders form a tree (parent_id); files live in at most one
-- folder (null = root). Folder location is organizational only - extraction,
-- project assignment, and RAG behavior are unaffected by where a file lives.
CREATE TABLE IF NOT EXISTS folders (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name       text NOT NULL,
    parent_id  uuid REFERENCES folders(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS folders_parent_idx ON folders (parent_id);

CREATE TABLE IF NOT EXISTS files (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    filename      text NOT NULL,
    file_type     text NOT NULL,            -- dxf | pdf | image
    s3_key        text NOT NULL,            -- original file location
    status        text NOT NULL DEFAULT 'uploaded',  -- uploaded | extracted | reviewed | ingested | failed
    error         text,                     -- extraction failure message when status = 'failed'
    content_sha256 text,                    -- hash of the original bytes (exact-match signal)
    embedding     vector(1024),
    is_drawing    boolean,             -- vision verdict; false = not an engineering drawing             -- document-level embedding for semantic duplicate/similarity detection
    extraction    jsonb,                    -- provisional chunks awaiting HITL review
    render        jsonb,                    -- {s3_key, extents [xmin,ymin,xmax,ymax]} of the PNG render
    drawing_id    uuid REFERENCES drawings(id) ON DELETE SET NULL,  -- the logical drawing this file belongs to
    sheet_number  text,                     -- e.g. "23" for "SHT 23", or "6 of 31"
    folder_id     uuid REFERENCES folders(id) ON DELETE SET NULL,   -- file-manager location (null = root)
    auto_assigned boolean NOT NULL DEFAULT false,  -- drawing assignment made automatically (exact DWG match)
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS files_content_hash_idx ON files (content_sha256);
CREATE INDEX IF NOT EXISTS files_drawing_idx ON files (drawing_id);
CREATE INDEX IF NOT EXISTS files_folder_idx ON files (folder_id);

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
    feedback_weight     float8 NOT NULL DEFAULT 1.0,  -- RLHF: adjusted by answer ratings
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
    version_context jsonb,          -- which drawing version answered + sibling versions
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON chat_messages (session_id, created_at);

-- RLHF: one rating per assistant message. Ratings adjust the feedback_weight
-- of the evidence the answer used, so retrieval learns from user feedback.
CREATE TABLE IF NOT EXISTS answer_feedback (
    message_id uuid PRIMARY KEY REFERENCES chat_messages(id) ON DELETE CASCADE,
    rating     smallint NOT NULL,   -- 1 = helpful, -1 = not helpful
    comment    text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- Authentication: simple username/password accounts with server-side
-- session tokens (revocable, expiring). Passwords are bcrypt-hashed;
-- tokens are stored as sha256 digests so a database leak exposes neither.
CREATE TABLE IF NOT EXISTS users (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    username      text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS auth_tokens (
    token_sha256 text PRIMARY KEY,
    user_id      uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at   timestamptz NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS auth_tokens_user_idx ON auth_tokens (user_id);
