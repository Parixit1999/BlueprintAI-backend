# Extraction (DXF / PDF / image)

`app/services/extraction/`. One extractor per format, registered by extension.
Each returns `list[ProvisionalChunk]` (region_type, chunk_text, bbox, confidence,
page). **Chunking = one drawing element per chunk**, not fixed-size windows —
the chunk boundary IS the evidence boundary.

## DXF — `dxf.py` (primary, most accurate)
- ezdxf structural read. One chunk per `TEXT`/`MTEXT` entity (note) and per
  `DIMENSION` entity. Exact model-space bboxes. Illegible dimension → text=null,
  confidence=low (never guessed). Corrupt file → `InvalidFile`.

## Vector PDF — `pdf.py`
- PyMuPDF, span-level text with exact bboxes (y flipped to y-up to match the
  viewer). A regex classifies dimension-looking spans. Scanned/no-text PDF →
  clear `InvalidFile` ("upload as an image instead"); password-protected →
  `InvalidFile`.

## Image — `image.py`
- Vision model (`VisionProvider`: Ollama qwen2.5vl local, Bedrock Claude on AWS).
  Downscaled to a known size so returned pixel coords map back reliably; handles
  fraction/percent/absolute coord scales and reversed corners. Model returns a
  JSON array of regions with per-field confidence; illegible → null.
- Least reliable path by design → confidence flows into HITL review.

## Gotcha
Fine granularity means atomic chunks (often one line). Point-lookup Q&A is strong;
aggregate ("list all notes") is limited by top-k retrieval — see `rag.md`.
