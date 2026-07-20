# Duplicate detection (semantic)

Detects when the same drawing was uploaded more than once — **even across file
formats** (e.g. the same part as a DXF and as a PNG photo).

## How
- At extraction time, `FileService._document_embedding` builds one document-level
  embedding from the concatenated extracted text and stores it on
  `files.embedding`. Exists before ingestion, so dedup works at review time.
- `FileRepository.list_all(threshold)` self-joins `files` on cosine similarity:
  for each file it returns `similar_documents` (file_id, filename, similarity)
  above `settings.duplicate_similarity_threshold (0.90)`; `is_duplicate = has any`.
- The frontend shows a "Possible duplicate" tag + "N% similar to <file>" and a
  filter. Deletion is manual (user verifies first).

## Why not byte-hashing
`content_sha256` (exact hash) only catches identical bytes. It missed the same
drawing re-exported or in a different format. Embedding similarity catches those:
same drawing across formats ≈ 0.98, genuinely different parts ≈ 0.59 — a clean gap.

## Reuse
Same machinery seeds the future **"compile related documents"** feature: a lower
threshold surfaces related-but-not-identical drawings.

Note: calibrated on the local mxbai embedding model; re-check the 0.90 threshold
on Bedrock Titan.
