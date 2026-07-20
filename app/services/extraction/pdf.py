"""Vector-PDF extraction via PyMuPDF - embedded text with exact bboxes.

Bboxes are converted to y-up coordinates (origin bottom-left) so the same
viewer mapping works for every file type. Scanned PDFs (no text layer) are
rejected with a clear, actionable error.
"""
import re

import pymupdf

from app.exceptions import InvalidFile
from app.schemas import Confidence, ProvisionalChunk, RegionType

# Text that looks like a dimension callout: "120", "12.5 mm", "R6.5", "Ø13 ±0.1"
_DIMENSION_RE = re.compile(
    r"^[\sØ⌀R]*\d+(\.\d+)?\s*(mm|cm|in|\")?\s*(±\s*\d+(\.\d+)?)?\s*$", re.IGNORECASE
)


def _classify(text: str) -> RegionType:
    return RegionType.dimension if _DIMENSION_RE.match(text.strip()) else RegionType.note


class PdfExtractor:
    def extract(self, path: str) -> list[ProvisionalChunk]:
        try:
            doc = pymupdf.open(path)
        except Exception:
            raise InvalidFile("This PDF could not be opened - the file appears to be corrupt.")

        if doc.needs_pass:
            raise InvalidFile("This PDF is password-protected. Remove the password and re-upload.")

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

        if not chunks:
            raise InvalidFile(
                "No extractable text found in this PDF - it appears to be a scanned "
                "document. Upload the drawing as a PNG/JPG image instead, and it will "
                "be processed with vision extraction."
            )
        return chunks
