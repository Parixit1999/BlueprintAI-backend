"""Upload orchestration: validate -> store original -> extract -> persist chunks."""
import hashlib
import logging
import threading
import tempfile
from pathlib import Path

from app.config import settings
from app.exceptions import BlueprintError, FileNotFound, FileTooLarge, UnsupportedFileType
from app.repositories import FileRepository
from app.services import extraction
from app.services.ai.base import EmbeddingProvider
from app.services.storage import ObjectStorage


class FileService:
    def __init__(
        self,
        files: FileRepository,
        storage: ObjectStorage,
        embedder: EmbeddingProvider,
        index=None,  # RegistryIndexService; optional to keep tests/tools light
        drawings=None,  # DrawingRepository; optional, enables orphan cleanup
    ):
        self._files = files
        self._storage = storage
        self._embedder = embedder
        self._index = index
        self._drawings = drawings

    def _document_embedding(self, chunks: list[dict]) -> list[float] | None:
        """One embedding representing the whole document, for semantic
        duplicate/similarity detection. Built from the extracted text so it
        exists before ingestion (dedup works at review time)."""
        texts = [c["chunk_text"] for c in chunks if c.get("chunk_text")]
        if not texts:
            return None
        return self._embedder.embed(" ".join(texts))

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Uploads control this string: strip any path components and control
        characters so it can never influence storage keys, temp paths, or UI
        rendering; cap the length for sane display and key sizes."""
        clean = Path(filename or "").name
        clean = "".join(ch for ch in clean if ch.isprintable() and ch not in '\\/:*?"<>|')
        return clean[:200].strip() or "unnamed"

    # Extraction concurrency cap: pages/files beyond this WAIT instead of
    # competing for memory. Two heavy multi-page extractions already use
    # 4 concurrent vision calls; a third joins the line. This is what turns
    # "many simultaneous uploads" from an OOM risk into a short queue.
    _extract_slots = threading.Semaphore(2)

    def store_upload(self, filename: str, data: bytes, folder_id: str | None = None) -> dict:
        """The FAST half of an upload: validate, persist the original to
        object storage, create the record in status 'uploaded'. Returns in
        seconds so the HTTP request is never at the mercy of extraction time
        (proxies cut connections at 60s; a dense multi-sheet scan takes 12
        minutes). Extraction happens in process_upload, in the background."""
        filename = self._sanitize_filename(filename)
        suffix = Path(filename).suffix.lower()
        extractor = extraction.get_extractor(suffix)
        if extractor is None:
            guidance = extraction.FORMAT_GUIDANCE.get(suffix)
            if guidance:
                raise UnsupportedFileType(guidance)
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
        file_id = self._files.create(filename, suffix.lstrip("."), content_sha256, folder_id)
        s3_key = f"originals/{file_id}/{filename}"
        self._storage.upload_bytes(data, s3_key)
        # persist the key NOW - the background task reads it from the record
        self._files.set_s3_key(file_id, s3_key)
        return {"file_id": file_id, "filename": filename, "status": "uploaded"}

    def process_upload(self, file_id: str, run_matcher=None) -> None:
        """The SLOW half: extract, store regions, and (for fresh uploads) run
        the assignment matcher. Runs as a background task under the extraction
        semaphore; all outcomes land in the DB for the frontend to poll -
        errors are recorded on the file row, never raised to a caller."""
        record = self._files.get(file_id)
        if record is None:
            return
        s3_key = record["s3_key"]
        try:
            data = self._storage.download_bytes(s3_key)
            with self._extract_slots:
                self._extract_and_store(
                    file_id, record["filename"],
                    Path(record["filename"]).suffix.lower(), s3_key, data,
                )
            if run_matcher is not None:
                run_matcher(file_id)
        except Exception:
            logging.getLogger(__name__).exception(
                "background processing failed for file %s", file_id
            )
            # _extract_and_store already recorded the failure on the row;
            # anything before/after it gets a generic record so the user can
            # see and retry rather than staring at a stuck 'uploaded'
            current = self._files.get(file_id)
            if current is not None and current["status"] == "uploaded":
                self._files.mark_failed(
                    file_id, s3_key, "Processing failed unexpectedly - use Retry."
                )

    def ingest_upload(self, filename: str, data: bytes, folder_id: str | None = None) -> dict:
        """Synchronous upload+extract, kept for scripts/tests - the API route
        uses store_upload + background process_upload instead."""
        stored = self.store_upload(filename, data, folder_id)
        record = self._files.get(stored["file_id"])
        return self._extract_and_store(
            stored["file_id"], stored["filename"],
            Path(stored["filename"]).suffix.lower(), record["s3_key"],
            self._storage.download_bytes(record["s3_key"]),
        )

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
        # File-level verdict from the vision summaries (one per page): flagged
        # as not-a-drawing only when EVERY judged page says so; None = the
        # extractor made no judgement (CAD files are drawings by definition).
        verdicts = [c.get("is_drawing") for c in chunks if c.get("is_drawing") is not None]
        is_drawing = None if not verdicts else any(verdicts)
        self._files.mark_extracted(file_id, s3_key, chunks, embedding, is_drawing)
        return {"file_id": file_id, "filename": filename, "chunks": chunks,
                "is_drawing": is_drawing}

    def prepare_retry(self, file_id: str) -> dict:
        """Validate and mark a failed upload for background re-processing."""
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound("Document not found")
        if record["status"] not in ("failed", "uploaded"):
            raise UnsupportedFileType(
                "This document was already extracted - retry only applies to failed uploads."
            )
        self._files.mark_uploaded(file_id)
        return {"file_id": file_id, "filename": record["filename"], "status": "uploaded"}

    def prepare_reextract(self, file_id: str) -> dict:
        """Validate and mark an extracted/ingested document for background
        re-extraction: knowledge-base chunks drop now (nothing enters twice),
        the fresh regions arrive when the background pass finishes."""
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound("Document not found")
        s3_key = record["s3_key"]
        if not s3_key or s3_key == "pending":
            raise UnsupportedFileType(
                "The original file is not available for this document."
            )
        self._files.release_ingest_claim(file_id)
        self._files.mark_uploaded(file_id)
        return {"file_id": file_id, "filename": record["filename"], "status": "uploaded"}

    def reextract(self, file_id: str) -> dict:
        """Re-read an already-extracted (or ingested) document with the current
        extraction pipeline - used when the models improve (better text, better
        region boxes). Existing knowledge-base chunks for the file are removed
        and the document returns to 'needs review', so nothing enters the
        knowledge base twice and the human checkpoint still applies."""
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound("Document not found")
        s3_key = record["s3_key"]
        if not s3_key or s3_key == "pending":
            raise UnsupportedFileType(
                "The original file is not available for this document."
            )
        # drops this file's chunks and returns status to 'extracted'
        self._files.release_ingest_claim(file_id)
        data = self._storage.download_bytes(s3_key)
        result = self._extract_and_store(
            file_id, record["filename"], Path(record["filename"]).suffix.lower(),
            s3_key, data,
        )
        # region boxes changed: the cached page renders are still valid (same
        # original), but any drawing card mentioning this file is unaffected.
        return result

    def list_files(self) -> list[dict]:
        return self._files.list_all(settings.duplicate_similarity_threshold)

    def get_extraction(self, file_id: str) -> dict | None:
        record = self._files.get(file_id)
        if record is None:
            return None
        return {"file_id": file_id, "status": record["status"],
                "filename": record["filename"], "is_drawing": record.get("is_drawing"),
                "dwg_number": record.get("dwg_number"),
                "project_name": record.get("project_name"),
                "auto_assigned": record.get("auto_assigned"),
                "error": record.get("error"),
                "chunks": record["extraction"]}

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
        # Drawing records auto-created by an upload should not outlive their
        # last file - otherwise projects count ghost drawings. Human work is
        # protected: manually created drawings, any hand-entered metadata
        # (description, contract number, version note), or explicit version
        # links keep the record even with zero files.
        if self._drawings and record.get("drawing_id"):
            if self._cleanup_orphan_drawing(record["drawing_id"]):
                return
        # the parent drawing's registry card lists its files - refresh it so
        # registry answers stop mentioning the deleted document (best-effort:
        # a stale card must never block the delete)
        if self._index and record.get("drawing_id"):
            try:
                self._index.index_drawing(record["drawing_id"])
            except Exception:
                pass

    def _cleanup_orphan_drawing(self, drawing_id: str) -> bool:
        """Delete a drawing left empty by a file deletion, but ONLY when it is
        a bare upload-created shell. Returns True when the drawing was removed
        (registry card included). Best-effort: cleanup must never fail the
        file deletion that triggered it."""
        try:
            drawing = self._drawings.get(drawing_id)
            if drawing is None:
                return False
            protected = (
                drawing.get("source") != "upload"
                or drawing.get("description")
                or drawing.get("contract_number")
                or drawing.get("version_note")
                or self._drawings.version_sibling_count(drawing_id) > 0
            )
            if protected or self._drawings.files_count(drawing_id) > 0:
                return False
            self._drawings.delete(drawing_id)
            if self._index:
                self._index.remove("drawing", drawing_id)
                if drawing.get("project_id"):
                    self._index.index_project(drawing["project_id"])
            return True
        except Exception:
            return False
