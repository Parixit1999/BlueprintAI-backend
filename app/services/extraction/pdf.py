"""PDF extraction via PyMuPDF.

Vector PDFs use the embedded text layer, giving exact bboxes at high
confidence. Scanned PDFs (no text layer) are rasterized page by page and run
through the vision model - the same path image uploads use - so a photographed
or scanned drawing works without asking the user to convert it to PNG/JPG.

Bboxes are converted to y-up coordinates (origin bottom-left) so the same
viewer mapping works for every file type.
"""
import re

import pymupdf

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

    def __init__(self, image_extractor: ImageExtractor | None = None):
        # Injected so scanned pages can fall back to vision. Optional so a
        # text-only deployment (no vision provider) still works for vector PDFs.
        self._image = image_extractor

    def extract(self, path: str) -> list[ProvisionalChunk]:
        try:
            doc = pymupdf.open(path)
        except Exception:
            raise InvalidFile("This PDF could not be opened - the file appears to be corrupt.")

        if doc.needs_pass:
            raise InvalidFile("This PDF is password-protected. Remove the password and re-upload.")

        chunks = self._extract_text(doc)
        if chunks:
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
        chunks: list[ProvisionalChunk] = []
        for page_index, page in enumerate(doc):
            png = page.get_pixmap(dpi=self.SCAN_DPI).tobytes("png")
            png, _applied = enhance_for_vision(png)
            regions = self._image.analyze(png)
            # Map vision percentages into PDF-point extents, matching the viewer's
            # render_pdf_page extents [0, 0, page.width, page.height].
            for region in regions:
                chunks.append(
                    ImageExtractor.region_to_chunk(
                        region,
                        page.rect.width,
                        page.rect.height,
                        page=page_index + 1,
                    )
                )
        return chunks
