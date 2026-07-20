"""Extraction: DXF via ezdxf (structured, primary path) and vector PDF via PyMuPDF.

Day 2-3. Output must conform to app.schemas.ExtractionResult — every field with
value + confidence + bbox; illegible -> value=None, confidence=low, never guess.
"""
from app.schemas import ExtractionResult


def extract_dxf(path: str) -> ExtractionResult:
    raise NotImplementedError


def extract_pdf(path: str) -> ExtractionResult:
    raise NotImplementedError
