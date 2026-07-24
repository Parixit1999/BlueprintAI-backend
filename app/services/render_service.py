"""Evidence rendering: per-page PNG of the drawing + extents for bbox overlay.

Renders lazily on first request and caches per page in object storage.
"""
import tempfile
from pathlib import Path

from app.exceptions import FileNotFound, RenderFailed
from app.repositories import FileRepository
from app.services.extraction.dwg import convert_to_dxf
from app.services.extraction.rvt import extract_preview_png
from app.services.rendering import render_dxf, render_image, render_pdf_page
from app.services.storage import ObjectStorage


class RenderService:
    def __init__(self, files: FileRepository, storage: ObjectStorage):
        self._files = files
        self._storage = storage

    def get_render(self, file_id: str, page: int = 1) -> dict:
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound("File not found")

        pages = self._page_map(record)
        entry = pages.get(str(page))
        if entry is None:
            entry = self._generate(record, page)
            pages[str(page)] = entry
            self._files.set_render(file_id, {"pages": pages})

        return {
            "file_id": file_id,
            "page": page,
            "url": self._storage.presigned_url(entry["s3_key"]),
            "extents": entry["extents"],
        }

    def get_render_bytes(self, file_id: str, page: int = 1) -> bytes:
        """The rendered page as PNG bytes - used to let the answer model SEE
        the drawing it is describing (visual answers)."""
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound("File not found")
        pages = self._page_map(record)
        entry = pages.get(str(page))
        if entry is None:
            entry = self._generate(record, page)
            pages[str(page)] = entry
            self._files.set_render(file_id, {"pages": pages})
        return self._storage.download_bytes(entry["s3_key"])

    @staticmethod
    def _page_map(record: dict) -> dict:
        render = record["render"] or {}
        if "pages" in render:
            return dict(render["pages"])
        if "s3_key" in render:  # legacy single-page format
            return {"1": render}
        return {}

    def _generate(self, record: dict, page: int) -> dict:
        suffix = Path(record["filename"]).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            # stream to disk - originals can be up to 1 GB
            self._storage.download_to_path(record["s3_key"], tmp.name)
            try:
                if suffix == ".dxf":
                    png, extents = render_dxf(tmp.name)
                elif suffix == ".dwg":
                    # render the same DXF conversion extraction used, so
                    # region bboxes line up with what is on screen
                    with tempfile.TemporaryDirectory() as out_dir:
                        dxf_path = convert_to_dxf(tmp.name, out_dir)
                        png, extents = render_dxf(str(dxf_path))
                elif suffix == ".rvt":
                    preview = extract_preview_png(tmp.name)
                    if preview is None:
                        raise RenderFailed(
                            "This Revit file has no embedded preview image to display."
                        )
                    with tempfile.NamedTemporaryFile(suffix=".png") as ptmp:
                        ptmp.write(preview)
                        ptmp.flush()
                        png, extents = render_image(ptmp.name)
                elif suffix == ".pdf":
                    png, extents = render_pdf_page(tmp.name, page)
                else:
                    png, extents = render_image(tmp.name)
            except RenderFailed:
                raise
            except Exception as exc:
                raise RenderFailed(f"Could not render this drawing: {exc}") from exc
        s3_key = f"renders/{record['file_id']}_p{page}.png"
        self._storage.upload_bytes(png, s3_key, content_type="image/png")
        return {"s3_key": s3_key, "extents": extents}
