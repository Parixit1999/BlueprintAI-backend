"""Upload orchestration: validate -> store original -> extract -> persist chunks."""
import hashlib
import tempfile
from pathlib import Path

from app.config import settings
from app.exceptions import BlueprintError, FileNotFound, FileTooLarge, UnsupportedFileType
from app.repositories import FileRepository
from app.services import extraction
from app.services.ai.base import EmbeddingProvider
from app.services.storage import ObjectStorage


class FileService:
    def __init__(self, files: FileRepository, storage: ObjectStorage, embedder: EmbeddingProvider):
        self._files = files
        self._storage = storage
        self._embedder = embedder

    def _document_embedding(self, chunks: list[dict]) -> list[float] | None:
        """One embedding representing the whole document, for semantic
        duplicate/similarity detection. Built from the extracted text so it
        exists before ingestion (dedup works at review time)."""
        texts = [c["chunk_text"] for c in chunks if c.get("chunk_text")]
        if not texts:
            return None
        return self._embedder.embed(" ".join(texts))

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
        return self._extract_and_store(file_id, filename, suffix, s3_key, data)

    def _extract_and_store(
        self, file_id: str, filename: str, suffix: str, s3_key: str, data: bytes
    ) -> dict:
        """Run extraction and persist the result. On failure the row is kept
        with status 'failed' and the error message, so the user sees what went
        wrong in the document list and can retry without re-uploading."""
        extractor = extraction.get_extractor(suffix)
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                chunks = [c.model_dump(mode="json") for c in extractor.extract(tmp.name)]
            except BlueprintError as exc:
                self._files.mark_failed(file_id, s3_key, str(exc))
                raise
            except Exception as exc:
                self._files.mark_failed(
                    file_id, s3_key, f"Unexpected extraction error: {exc}"
                )
                raise

        embedding = self._document_embedding(chunks)
        self._files.mark_extracted(file_id, s3_key, chunks, embedding)
        return {"file_id": file_id, "filename": filename, "chunks": chunks}

    def retry_extraction(self, file_id: str) -> dict:
        """Re-run extraction on the stored original of a failed (or stuck
        'uploaded') document."""
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound("Document not found")
        if record["status"] not in ("failed", "uploaded"):
            raise UnsupportedFileType(
                "This document was already extracted - retry only applies to failed uploads."
            )
        filename = record["filename"]
        s3_key = record["s3_key"]
        if not s3_key or s3_key == "pending":
            s3_key = f"originals/{file_id}/{filename}"
        data = self._storage.download_bytes(s3_key)
        return self._extract_and_store(file_id, filename, Path(filename).suffix.lower(), s3_key, data)

    def list_files(self) -> list[dict]:
        return self._files.list_all(settings.duplicate_similarity_threshold)

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
