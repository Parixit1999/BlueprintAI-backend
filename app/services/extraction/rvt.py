"""Revit (.rvt) support - best effort, clearly labeled.

RVT is a proprietary Autodesk format whose sheet geometry cannot be parsed
by any free library. What CAN be read for free: an .rvt file is an OLE
compound document carrying an embedded preview image (RevitPreview4.0
stream) and a UTF-16 metadata block (BasicFileInfo stream). We extract
both, run the preview through the vision model, and attach a prominent
note explaining the accuracy limitation - per product decision, partial
support with an honest note beats rejecting the file.
"""
import io
import re

import olefile
from PIL import Image

from app.exceptions import ExtractionFailed
from app.schemas import Confidence, ProvisionalChunk, RegionType
from app.services.extraction.image import ImageExtractor

LIMITATION_NOTE = (
    "Revit model: the .rvt format is proprietary and its full sheet geometry "
    "cannot be extracted by free tools. Shown here are the model's embedded "
    "preview image and file metadata only - accuracy is limited to what the "
    "preview reveals. For complete, searchable drawings, export the sheets "
    "from Revit as PDF or DWG and upload those."
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PNG_END = b"IEND\xaeB`\x82"

# BasicFileInfo lines worth surfacing (key: value pairs in UTF-16 text)
_INFO_KEYS = ("Revit Build", "Format", "Build", "Last Save Path", "Username")


def extract_preview_png(path: str) -> bytes | None:
    """The embedded preview thumbnail as PNG bytes, if present."""
    with olefile.OleFileIO(path) as ole:
        for stream in ole.listdir(streams=True, storages=False):
            if "RevitPreview4.0" in stream:
                data = ole.openstream(stream).read()
                start = data.find(_PNG_MAGIC)
                end = data.rfind(_PNG_END)
                if start != -1 and end != -1:
                    return data[start : end + len(_PNG_END)]
    return None


def extract_basic_info(path: str) -> list[str]:
    """Human-readable metadata lines from the BasicFileInfo stream."""
    with olefile.OleFileIO(path) as ole:
        for stream in ole.listdir(streams=True, storages=False):
            if "BasicFileInfo" in stream:
                raw = ole.openstream(stream).read()
                text = raw.decode("utf-16-le", errors="ignore")
                lines = []
                for line in re.split(r"[\r\n\x00]+", text):
                    line = line.strip()
                    if any(line.startswith(k) for k in _INFO_KEYS) and ":" in line:
                        lines.append(line)
                return lines[:8]
    return []


class RvtExtractor:
    def __init__(self, image_extractor: ImageExtractor | None = None):
        # Optional so metadata still extracts in a text-only deployment.
        self._image = image_extractor

    def extract(self, path: str) -> list[ProvisionalChunk]:
        if not olefile.isOleFile(path):
            raise ExtractionFailed(
                "This does not look like a valid Revit file - it may be corrupt "
                "or from a very old Revit version."
            )

        chunks: list[ProvisionalChunk] = [
            ProvisionalChunk(
                region_type=RegionType.note,
                chunk_text=LIMITATION_NOTE,
                bbox=None,
                confidence=Confidence.low,
                page=1,
            )
        ]

        for line in extract_basic_info(path):
            chunks.append(
                ProvisionalChunk(
                    region_type=RegionType.title_block,
                    chunk_text=line,
                    bbox=None,
                    confidence=Confidence.high,  # decoded from the file, not OCR
                    page=1,
                )
            )

        preview = extract_preview_png(path)
        if preview is not None and self._image is not None:
            with Image.open(io.BytesIO(preview)) as img:
                width, height = img.size
            for region in self._image.analyze(preview):
                chunks.append(
                    ImageExtractor.region_to_chunk(region, float(width), float(height), page=1)
                )

        if len(chunks) == 1 and preview is None:
            raise ExtractionFailed(
                "No readable preview or metadata found in this Revit file. "
                "Export the sheets as PDF or DWG from Revit and upload those."
            )
        return chunks
