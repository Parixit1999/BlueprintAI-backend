"""PDF extraction via PyMuPDF.

Vector PDFs use the embedded text layer, giving exact bboxes at high
confidence. Scanned PDFs (no text layer) are rasterized page by page and run
through the vision model - the same path image uploads use - so a photographed
or scanned drawing works without asking the user to convert it to PNG/JPG.

Bboxes are converted to y-up coordinates (origin bottom-left) so the same
viewer mapping works for every file type.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor

import pymupdf

from app.config import settings
from app.exceptions import ExtractionFailed, InvalidFile
from app.schemas import Confidence, ProvisionalChunk, RegionType
from app.services.extraction.enhance import enhance_for_vision
from app.services.extraction.image import ImageExtractor

# Text that looks like a dimension callout: "120", "12.5 mm", "R6.5", "Ø13 ±0.1"
_DIMENSION_RE = re.compile(
    r"^[\sØ⌀R]*\d+(\.\d+)?\s*(mm|cm|in|\")?\s*(±\s*\d+(\.\d+)?)?\s*$", re.IGNORECASE
)


def _classify(text: str) -> RegionType:
    return RegionType.dimension if _DIMENSION_RE.match(text.strip()) else RegionType.note


class PdfExtractor:
    # DPI to rasterize scanned pages before handing them to the vision model.
    # The image extractor downscales to <=1024px, so a higher DPI just yields a
    # crisper source before that cap.
    SCAN_DPI = 200

    def __init__(self, image_extractor: ImageExtractor | None = None, generator=None):
        # image_extractor: scanned pages fall back to vision. generator (a
        # TextGenerator): judges text-layer PDFs (drawing vs prose document)
        # and writes their summary. Both optional and best-effort.
        self._image = image_extractor
        self._generator = generator

    def extract(self, path: str) -> list[ProvisionalChunk]:
        try:
            doc = pymupdf.open(path)
        except Exception:
            raise InvalidFile("This PDF could not be opened - the file appears to be corrupt.")

        if doc.needs_pass:
            raise InvalidFile("This PDF is password-protected. Remove the password and re-upload.")

        chunks = self._extract_text(doc)
        if chunks:
            # Text-layer PDFs never meet the vision model, so judge them from
            # their text: is this actually a drawing sheet, or a prose document
            # (resume, report, letter) that shouldn't be in a drawing archive?
            # Also yields the summary region that vision-path documents get.
            verdict_chunk = self._judge_text(chunks)
            if verdict_chunk is not None:
                chunks = [verdict_chunk] + chunks
            return chunks

        # No embedded text layer -> scanned/photographed PDF. Rasterize each
        # page and run it through vision, the same way image uploads are handled.
        if self._image is not None:
            chunks = self._extract_scanned(doc)
            if chunks:
                return chunks
            raise ExtractionFailed(
                "This looks like a scanned PDF, but no readable text regions were "
                "detected on it. Try a higher-resolution scan."
            )

        raise InvalidFile(
            "No extractable text found in this PDF - it appears to be a scanned "
            "document, and vision extraction is not available in this deployment."
        )

    _JUDGE_PROMPT = (
        "You are looking at text extracted from a PDF in an engineering "
        "drawing archive. Decide whether the document is an engineering/"
        "technical DRAWING sheet (plans, sections, title blocks, dimensions, "
        "callouts) or a PROSE document (resume, report, letter, specification "
        "text, form). Then write one paragraph describing the document.\n"
        "Return ONLY JSON: {\"is_drawing\": true|false, \"summary\": \"...\"}"
    )

    def _judge_text(self, chunks: list[ProvisionalChunk]) -> ProvisionalChunk | None:
        """Text-based is-it-a-drawing verdict + summary for vector PDFs.
        Best-effort: any failure returns None and extraction proceeds as
        before (no summary, no verdict)."""
        if self._generator is None:
            return None
        try:
            sample = "\n".join(
                (c.chunk_text or "") for c in chunks[:60] if c.chunk_text
            )[:4000]
            raw = self._generator.generate(
                self._JUDGE_PROMPT, f"Extracted text:\n{sample}"
            ).strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw)
            obj = json.loads(raw)
            summary = obj.get("summary")
            verdict = obj.get("is_drawing")
            if not isinstance(summary, str) or not summary.strip():
                return None
            return ProvisionalChunk(
                region_type=RegionType.summary,
                chunk_text=summary.strip(),
                confidence=Confidence.high,
                page=1,
                is_drawing=verdict if isinstance(verdict, bool) else None,
            )
        except Exception:
            return None

    @staticmethod
    def _extract_text(doc: "pymupdf.Document") -> list[ProvisionalChunk]:
        chunks: list[ProvisionalChunk] = []
        for page_index, page in enumerate(doc):
            height = page.rect.height
            # span granularity keeps separately-placed callouts as separate chunks
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        content = " ".join(span["text"].split())
                        if not content:
                            continue
                        x0, y0, x1, y1 = span["bbox"]
                        chunks.append(
                            ProvisionalChunk(
                                region_type=_classify(content),
                                chunk_text=content,
                                # PDF y-axis points down; flip to y-up for the viewer
                                bbox=[
                                    round(x0, 3),
                                    round(height - y1, 3),
                                    round(x1, 3),
                                    round(height - y0, 3),
                                ],
                                confidence=Confidence.high,  # embedded text, not OCR
                                page=page_index + 1,
                            )
                        )
        return chunks

    def _extract_scanned(self, doc: "pymupdf.Document") -> list[ProvisionalChunk]:
        # Rasterize serially (PyMuPDF documents are not thread-safe), then run
        # the slow part - vision + OCR per page - concurrently. Results keep
        # page order regardless of completion order.
        pages: list[tuple[int, bytes, float, float]] = []
        for page_index, page in enumerate(doc):
            png = page.get_pixmap(dpi=self.SCAN_DPI).tobytes("png")
            png, _applied = enhance_for_vision(png)
            pages.append((page_index, png, page.rect.width, page.rect.height))

        with ThreadPoolExecutor(max_workers=settings.vision_page_concurrency) as pool:
            page_regions = list(pool.map(lambda p: self._image.analyze(p[1]), pages))

        chunks: list[ProvisionalChunk] = []
        for (page_index, _png, width, height), regions in zip(pages, page_regions):
            # Map vision percentages into PDF-point extents, matching the viewer's
            # render_pdf_page extents [0, 0, page.width, page.height].
            for region in regions:
                chunks.append(
                    ImageExtractor.region_to_chunk(
                        region, width, height, page=page_index + 1
                    )
                )
        return chunks
