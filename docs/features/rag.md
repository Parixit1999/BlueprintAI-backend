# Embeddings, retrieval & RAG

## Ingestion
`review_service.py`: on "confirm & ingest", each confirmed/corrected region's
text is embedded (`EmbeddingProvider`, 1024-dim) and inserted into `chunks` with
its bbox/confidence/verification metadata. Only reviewed regions enter the vector
DB (garbage-in-garbage-out guard).

## Retrieval + generation — `query_service.py`
Pipeline: embed question → `ChunkRepository.search` (pgvector cosine) → **scope to
one drawing** → generate with the scoped regions as context.

### Document scoping (feat/scoped-retrieval — important)
Naive top-k across the whole KB padded answers with regions from *unrelated*
drawings. Now:
1. Retrieve a wide candidate pool (`CANDIDATE_POOL = 30`).
2. If the best score < `MIN_RELEVANCE (0.50)` → "couldn't find that", **zero
   evidence** (kills phantom sources on off-topic questions).
3. Keep only regions from the drawing that owns the best-matching region.
4. Within that drawing, keep regions scoring ≥ `RELATIVE_FLOOR (0.60)` × top,
   capped at `MAX_EVIDENCE (6)`.
Thresholds calibrated on real data (genuine Qs score ≥0.52, off-topic ≤0.45).

**Retune the thresholds when switching to Bedrock Titan** — the score
distribution differs from the local mxbai model.

## Answer style
`SYSTEM_PROMPT` forces conversational prose and forbids leaking
"chunk/source/index" references into the text (the UI shows sources separately).

## Known limitation
Atomic chunks + top-k mean single-drawing point lookups are strong, but aggregate
("list everything") and cross-drawing questions are weak. Fix path: feed the whole
(small) drawing to the model, and the future "compile related documents" feature
for multi-drawing questions.
