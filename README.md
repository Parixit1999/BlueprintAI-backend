# BlueprintAI Backend

BlueprintAI lets engineers upload engineering drawings (CAD/DXF files or vector PDFs),
automatically extract their content, and ask questions against them in plain language.
Every answer comes with **evidence** — a cropped, highlighted region of the original
drawing — so users can verify the answer against the source instead of trusting it blindly.

## What the app does

1. **Upload** — user uploads a drawing; the original is stored in S3.
2. **Extract** — structured data (title block, dimensions, notes, BOM) is extracted with
   a confidence level and bounding box for every field. Illegible values are flagged,
   never guessed.
3. **Review (human-in-the-loop)** — before anything is ingested, the user reviews
   extracted fields beside the source crop and confirms or corrects them.
4. **Ingest** — verified content is chunked by semantic region, embedded, and stored in
   a vector database with its spatial metadata.
5. **Query** — the user asks a question; the system retrieves relevant chunks, generates
   an answer, and returns it with the source-region crop as evidence.

## Tech stack

- **FastAPI** (Python 3.12) — API layer
- **Amazon Bedrock** — vision extraction, text generation, and embeddings
- **Amazon S3** — original files and evidence crops
- **RDS PostgreSQL + pgvector** — vector storage and retrieval
- **ezdxf / PyMuPDF / Pillow** — DXF parsing, PDF text extraction, evidence cropping

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in AWS + DB values
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

API docs at http://localhost:8000/docs

## Project structure

- `app/main.py` — FastAPI app + routers
- `app/config.py` — settings from `.env`
- `app/schemas.py` — extraction/chunk/query data contracts (value + confidence + bbox)
- `app/routers/` — `files` (upload/extraction), `review` (HITL confirm), `query` (RAG)
- `app/services/` — extraction (ezdxf/PyMuPDF), embedding (Titan), retrieval (pgvector), storage (S3)
