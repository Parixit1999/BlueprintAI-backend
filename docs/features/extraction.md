# Extraction (DXF / PDF / image)

`app/services/extraction/`. One extractor per format, registered by extension.
Each returns `list[ProvisionalChunk]` (region_type, chunk_text, bbox, confidence,
page). **Chunking = one drawing element per chunk**, not fixed-size windows —
the chunk boundary IS the evidence boundary.

## DXF — `dxf.py` (primary, most accurate)
- ezdxf structural read. One chunk per `TEXT`/`MTEXT` entity (note) and per
  `DIMENSION` entity. Exact model-space bboxes. Illegible dimension → text=null,
  confidence=low (never guessed). Corrupt file → `InvalidFile`.

## PDF — `pdf.py`
- **Vector PDF**: PyMuPDF, span-level text with exact bboxes (y flipped to y-up
  to match the viewer). A regex classifies dimension-looking spans, confidence
  `high` (embedded text, not OCR).
- **Scanned PDF (no text layer)**: instead of rejecting, each page is rasterized
  (`get_pixmap`, `SCAN_DPI=200`) and run through the same vision model as image
  uploads. `PdfExtractor` takes an injected `ImageExtractor`; vision returns
  percentage bboxes which are mapped into PDF-point extents (`[0,0,w,h]`,
  matching `render_pdf_page`) so evidence highlights line up. Falls back to a
  clear error only if vision is unavailable or finds nothing.
- Password-protected or corrupt → `InvalidFile`.

## Image — `image.py`
- Vision model (`VisionProvider`: Ollama qwen2.5vl local, Bedrock Claude on AWS).
  `analyze(bytes) -> list[VisionRegion]` is coordinate-space-agnostic (percentage
  bboxes) so both image uploads and scanned-PDF pages reuse it; `region_to_chunk`
  maps a region into the caller's y-up extents. Downscaled to a known size so
  returned pixel coords map back reliably; handles fraction/percent/absolute
  coord scales and reversed corners. Illegible → null.
- Least reliable path by design → confidence flows into HITL review.

## Gotcha
Fine granularity means atomic chunks (often one line). Point-lookup Q&A is strong;
aggregate ("list all notes") is limited by top-k retrieval — see `rag.md`.
