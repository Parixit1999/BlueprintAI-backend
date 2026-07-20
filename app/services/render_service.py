"""Evidence rendering: per-page PNG of the drawing + extents for bbox overlay.

Renders lazily on first request and caches per page in object storage.
"""
import tempfile
from pathlib import Path

from app.exceptions import FileNotFound, RenderFailed
from app.repositories import FileRepository
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

    @staticmethod
    def _page_map(record: dict) -> dict:
        render = record["render"] or {}
        if "pages" in render:
            return dict(render["pages"])
        if "s3_key" in render:  # legacy single-page format
            return {"1": render}
        return {}

    def _generate(self, record: dict, page: int) -> dict:
        data = self._storage.download_bytes(record["s3_key"])
        suffix = Path(record["filename"]).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                if suffix == ".dxf":
                    png, extents = render_dxf(tmp.name)
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
