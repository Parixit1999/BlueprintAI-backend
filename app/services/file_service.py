"""Upload orchestration: validate -> store original -> extract -> persist chunks."""
import hashlib
import tempfile
from pathlib import Path

from app.config import settings
from app.exceptions import FileNotFound, FileTooLarge, UnsupportedFileType
from app.repositories import FileRepository
from app.services import extraction
from app.services.storage import ObjectStorage


class FileService:
    def __init__(self, files: FileRepository, storage: ObjectStorage):
        self._files = files
        self._storage = storage

    def ingest_upload(self, filename: str, data: bytes) -> dict:
        suffix = Path(filename).suffix.lower()
        extractor = extraction.get_extractor(suffix)
        if extractor is None:
            supported = ", ".join(sorted(extraction.supported_extensions()))
            raise UnsupportedFileType(
                f"'{suffix or filename}' is not a supported file type. "
                f"Upload one of: {supported}."
            )
        if len(data) == 0:
            raise UnsupportedFileType("The uploaded file is empty.")
        if len(data) > settings.max_upload_bytes:
            raise FileTooLarge(
                f"File is {len(data) / 1024 / 1024:.1f} MB; the maximum is "
                f"{settings.max_upload_bytes // (1024 * 1024)} MB."
            )

        content_sha256 = hashlib.sha256(data).hexdigest()
        file_id = self._files.create(filename, suffix.lstrip("."), content_sha256)
        s3_key = f"originals/{file_id}/{filename}"
        self._storage.upload_bytes(data, s3_key)

        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                chunks = [c.model_dump(mode="json") for c in extractor.extract(tmp.name)]
            except Exception:
                # keep no orphaned records for failed extractions
                self._files.delete(file_id)
                raise

        self._files.mark_extracted(file_id, s3_key, chunks)
        return {"file_id": file_id, "filename": filename, "chunks": chunks}

    def list_files(self) -> list[dict]:
        return self._files.list_all()

    def get_extraction(self, file_id: str) -> dict | None:
        record = self._files.get(file_id)
        if record is None:
            return None
        return {"file_id": file_id, "status": record["status"], "chunks": record["extraction"]}

    def delete_file(self, file_id: str) -> None:
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound("Document not found")
        # remove stored objects first; ignore missing keys so a partial state
        # still deletes cleanly. Chunks cascade via the DB foreign key.
        for key in self._files.list_render_keys(file_id):
            try:
                self._storage.delete_bytes(key)
            except Exception:
                pass
        self._files.delete(file_id)
