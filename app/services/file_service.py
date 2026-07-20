"""Upload orchestration: store original -> extract -> persist provisional chunks."""
import tempfile
from pathlib import Path

from app.repositories import FileRepository
from app.services import extraction
from app.services.storage import ObjectStorage


class UnsupportedFileType(Exception):
    pass


class ExtractionFailed(Exception):
    pass


class FileService:
    def __init__(self, files: FileRepository, storage: ObjectStorage):
        self._files = files
        self._storage = storage

    def ingest_upload(self, filename: str, data: bytes) -> dict:
        suffix = Path(filename).suffix.lower()
        extractor = extraction.get_extractor(suffix)
        if extractor is None:
            raise UnsupportedFileType(
                f"Unsupported file type '{suffix}'. Supported: {sorted(extraction.supported_extensions())}"
            )

        file_id = self._files.create(filename, suffix.lstrip("."))
        s3_key = f"originals/{file_id}/{filename}"
        self._storage.upload_bytes(data, s3_key)

        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                chunks = [c.model_dump(mode="json") for c in extractor.extract(tmp.name)]
            except Exception as exc:
                self._files.delete(file_id)
                raise ExtractionFailed(str(exc)) from exc

        self._files.mark_extracted(file_id, s3_key, chunks)
        return {"file_id": file_id, "filename": filename, "chunks": chunks}

    def list_files(self) -> list[dict]:
        return self._files.list_all()

    def get_extraction(self, file_id: str) -> dict | None:
        record = self._files.get(file_id)
        if record is None:
            return None
        return {"file_id": file_id, "status": record["status"], "chunks": record["extraction"]}
