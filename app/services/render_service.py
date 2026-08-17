"""Evidence rendering: per-page PNG of the drawing + extents for bbox overlay.

Every sheet is pre-rendered right after extraction (prerender_all, called
from the background worker); the lazy per-request path remains as fallback
for legacy records and prerender failures.
"""
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

from app.config import settings
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
            "url": self._render_url(entry["s3_key"]),
            "extents": entry["extents"],
        }

    def _render_url(self, s3_key: str) -> str:
        """CDN path in production (CloudFront caches /renders/* at the edge;
        the key is unguessable - file uuid + page); presigned URL otherwise."""
        if settings.render_cdn:
            return f"/{s3_key}"
        return self._storage.presigned_url(s3_key)

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

    def prerender_all(self, file_id: str) -> None:
        """Render every sheet ONCE, right after extraction, in the background
        worker - so no viewer request ever waits on (or competes with) render
        generation. Heavy formats are handled efficiently: the original
        downloads once, a DWG converts once, and a CAD document parses once
        for all its sheets - unlike the lazy path, which repeats all of that
        per page. Best-effort: any failure leaves the lazy path as fallback.
        """
        record = self._files.get(file_id)
        if record is None:
            return
        pages = self._page_map(record)
        suffix = Path(record["filename"]).suffix.lower()
        # The document's own sheet count when we have it. Falling back to the
        # highest extracted page means a capped extraction silently stops the
        # pre-render early, leaving later sheets unviewable.
        page_count = record.get("page_count") or max(
            [c.get("page") or 1 for c in (record.get("extraction") or [])] or [1]
        )
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                self._storage.download_to_path(record["s3_key"], tmp.name)
                if suffix in (".dxf", ".dwg"):
                    with tempfile.TemporaryDirectory() as out_dir:
                        dxf_path = tmp.name
                        if suffix == ".dwg":
                            converted, _ = convert_to_dxf(tmp.name, out_dir)
                            dxf_path = str(converted)
                        import ezdxf

                        from app.services.rendering import (
                            dxf_sheet_names,
                            render_dxf_layout,
                        )

                        doc = ezdxf.readfile(dxf_path)
                        sheets = dxf_sheet_names(doc)
                        for i, name in enumerate(sheets, start=1):
                            if str(i) in pages:
                                continue
                            try:
                                png, extents = render_dxf_layout(doc, name)
                            except Exception:
                                continue  # one bad sheet must not stop the rest
                            pages[str(i)] = self._store_render(record, i, png, extents)
                            self._files.set_render(file_id, {"pages": pages})
                elif suffix == ".pdf":
                    for p in range(1, page_count + 1):
                        if str(p) in pages:
                            continue
                        try:
                            png, extents = render_pdf_page(tmp.name, p)
                        except Exception:
                            continue
                        pages[str(p)] = self._store_render(record, p, png, extents)
                        self._files.set_render(file_id, {"pages": pages})
                else:
                    if "1" not in pages:
                        # single-page formats reuse the normal lazy generator
                        self.get_render(file_id, 1)
        except Exception:
            logger.warning("prerender failed for %s", file_id, exc_info=True)

    def _generate(self, record: dict, page: int) -> dict:
        suffix = Path(record["filename"]).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            # stream to disk - originals can be up to 1 GB
            self._storage.download_to_path(record["s3_key"], tmp.name)
            try:
                if suffix == ".dxf":
                    png, extents = render_dxf(tmp.name, page)
                elif suffix == ".dwg":
                    # render the same DXF conversion extraction used, so
                    # region bboxes line up with what is on screen
                    with tempfile.TemporaryDirectory() as out_dir:
                        dxf_path, _converter = convert_to_dxf(tmp.name, out_dir)
                        png, extents = render_dxf(str(dxf_path), page)
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
        return self._store_render(record, page, png, extents)

    def _store_render(self, record: dict, page: int, png: bytes, extents: list) -> dict:
        # DXF renders stay PNG (line art); everything else is JPEG now
        is_png = png[:8] == b"\x89PNG\r\n\x1a\n"
        ext = "png" if is_png else "jpg"
        s3_key = f"renders/{record['file_id']}_p{page}.{ext}"
        self._storage.upload_bytes(
            png, s3_key, content_type="image/png" if is_png else "image/jpeg"
        )
        return {"s3_key": s3_key, "extents": extents}
