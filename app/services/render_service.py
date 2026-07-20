"""Evidence rendering: PNG of the drawing + model-space extents for bbox overlay.

Renders lazily on first request and caches the PNG in object storage, so files
uploaded before this feature existed work too.
"""
import tempfile
from pathlib import Path

from app.repositories import FileRepository
from app.services.rendering import render_dxf
from app.services.storage import ObjectStorage


class FileNotFound(Exception):
    pass


class RenderFailed(Exception):
    pass


class RenderService:
    def __init__(self, files: FileRepository, storage: ObjectStorage):
        self._files = files
        self._storage = storage

    def get_render(self, file_id: str) -> dict:
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound(file_id)

        render = record["render"]
        if render is None:
            render = self._generate(record)
            self._files.set_render(file_id, render)

        return {
            "file_id": file_id,
            "url": self._storage.presigned_url(render["s3_key"]),
            "extents": render["extents"],
        }

    def _generate(self, record: dict) -> dict:
        data = self._storage.download_bytes(record["s3_key"])
        suffix = Path(record["filename"]).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                png, extents = render_dxf(tmp.name)
            except Exception as exc:
                raise RenderFailed(str(exc)) from exc
        s3_key = f"renders/{record['file_id']}.png"
        self._storage.upload_bytes(png, s3_key, content_type="image/png")
        return {"s3_key": s3_key, "extents": extents}
