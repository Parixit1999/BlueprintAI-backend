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
- Vision model (`VisionProvider`: Ollama llama3.2-vision local, Bedrock Claude on AWS).
  `analyze(bytes) -> list[VisionRegion]` is coordinate-space-agnostic (percentage
  bboxes) so both image uploads and scanned-PDF pages reuse it; `region_to_chunk`
  maps a region into the caller's y-up extents. Downscaled to a known size so
  returned pixel coords map back reliably; handles fraction/percent/absolute
  coord scales and reversed corners. Illegible → null.
- This **overview pass** also reports the sheet's discipline, title and drawing
  number — context every zoomed pass below is given, because a cropped symbol is
  unidentifiable until you know what kind of drawing it came from.
- Least reliable path by design → confidence flows into HITL review.
- OCR snapping (`_snap_to_ocr`) replaces a vision bbox with Textract's
  pixel-accurate one when the text matches. Repeated text ("1", "TYP", a
  dimension value) is disambiguated by POSITION — the nearest OCR line wins,
  provided it is unambiguously nearest — which is what gives numbered callouts
  exact boxes. Components are never snapped: their labels describe a drawn
  symbol, not text on the sheet.

## Components — `components.py`
A single whole-sheet pass cannot find small components: an E-size sheet arrives
at 1568px, so a 1/2" valve is ~2 pixels. Asked to be exhaustive at that
resolution the model reports the big things and lumps the rest into a few huge
boxes. The fix is to zoom in.

- **Tile sweep** — the sheet is cut into an overlapping NxN grid, each tile read
  at near-native resolution. Gated on the sheet actually having detail the
  overview lost (`component_tile_min_scale`). Coverage is all-or-nothing: the
  grid shrinks to what the budget covers *completely*, because half a sweep
  reads as a detection bias, not a budget limit.
- **Split** — a box that is group-sized (`component_oversized_area_pct`, large
  in BOTH axes) gets cropped and re-read so its contents come back individually
  boxed. Runs only on groups the tile sweep did not already dissolve, since
  tiling breaks groups up as a side effect. A group that survives every attempt
  is marked `suspect` → forced to low confidence rather than sold as a component.
- **Merge** — IoU dedup across passes; zoomed boxes beat overview boxes, and an
  overview box containing ≥2 zoomed detections is dropped as a failed grouping.
- **Tighten** — `tighten_to_ink` shrinks each box to its actual ink extent using
  an adaptive threshold. Free (no model call) and the cheapest bbox accuracy
  available on line art.
- **Naming** — components no pass could name are composed into ONE numbered
  contact sheet (set-of-mark) and named in a single call, instead of one call
  each.
- **Grouping** — instances of the same resolved text collapse into one review
  card carrying every instance box, so 60 valves are one row with 60 highlights.

Budget: `component_detail_calls` (default 4) caps the extra calls per sheet;
naming adds at most one more. **Set it to 0 for `AI_PROVIDER=ollama`** — local
vision runs ~60s per call. At 0, the free stages (tighten, merge, OCR snapping,
keynotes) still run, so behaviour degrades to single-pass, not to worse boxes.

## Callouts and keynotes — `callouts.py`
Drawings label components with numbered bubbles whose meaning lives in a keynote
legend ("3) FLOOR DRAIN WITH TRAP PRIMER"). Pure functions parse that legend out
of whatever text the extractor already holds (CAD entities, PDF text layer, OCR
lines — DXF and PDF pass theirs in via `analyze(sheet_texts=...)`), then attach
each number to a component: the model's own reported association first (it can
see the leader line), nearest-neighbour within a distance cap as fallback. A
resolved keynote is the strongest identification a component can have — the
drafter's words, not a model's reading — and is weighted accordingly below.

## Confidence
`resolve_confidence` derives confidence from EVIDENCE rather than trusting the
model's self-report, which is systematically cautious about drawn symbols:
`+2` keynote resolved, `+1` corroborated by an independent pass, `+1` read on a
zoomed crop; capped to low for an unsplittable group box, and to medium for
something tiny that was only ever seen in the downscale. The size cap is judged
on the box **as the model reported it**, never the tightened one — otherwise
improving a box would lower its confidence.

## Gotcha
Fine granularity means atomic chunks (often one line). Point-lookup Q&A is strong;
aggregate ("list all notes") is limited by top-k retrieval — see `rag.md`.

`extra_bboxes` (a component card's further instance boxes) lives in the review
payload only — `chunks` stores one bbox per row, so after ingestion a card
highlights its first instance. Review shows all of them.
