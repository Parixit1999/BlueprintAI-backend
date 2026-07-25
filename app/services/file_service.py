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

    def store_upload(self, filename: str, fileobj, folder_id: str | None = None) -> dict:
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
        # `fileobj` streams: size via seek, hash in chunks, multipart to S3 -
        # a 1 GB Revit model never lives in this process's memory
        fileobj.seek(0, 2)
        size = fileobj.tell()
        fileobj.seek(0)
        if size == 0:
            raise UnsupportedFileType("The uploaded file is empty.")
        if size > settings.max_upload_bytes:
            raise FileTooLarge(
                f"File is {size / 1024 / 1024:.1f} MB; the maximum is "
                f"{settings.max_upload_bytes // (1024 * 1024)} MB."
            )

        digest = hashlib.sha256()
        for block in iter(lambda: fileobj.read(1024 * 1024), b""):
            digest.update(block)
        fileobj.seek(0)
        content_sha256 = digest.hexdigest()

        file_id = self._files.create(filename, suffix.lstrip("."), content_sha256, folder_id)
        s3_key = f"originals/{file_id}/{filename}"
        self._storage.upload_fileobj(fileobj, s3_key)
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
            suffix = Path(record["filename"]).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                self._storage.download_to_path(s3_key, tmp.name)
                with self._extract_slots:
                    self._extract_and_store(
                        file_id, record["filename"], suffix, s3_key, path=tmp.name,
                    )
            if run_matcher is not None:
                run_matcher(file_id)
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "background processing failed for file %s", file_id
            )
            # _extract_and_store already recorded the failure on the row;
            # anything before/after it gets a record CARRYING THE REAL ERROR -
            # a generic "failed unexpectedly" makes every incident a log dig
            current = self._files.get(file_id)
            if current is not None and current["status"] == "uploaded":
                detail = f"{type(exc).__name__}: {exc}"[:300]
                self._files.mark_failed(
                    file_id, s3_key, f"Processing failed - use Retry. ({detail})"
                )

    # Region priority when a file exceeds the region cap: identity and
    # structure first, bulk dimensions and notes last.
    _REGION_PRIORITY = {"summary": 0, "title_block": 1, "component": 2, "bom": 3,
                        "dimension": 4, "note": 5}

    def _guard_region_count(self, chunks: list[dict]) -> list[dict]:
        """A pathological file (98k-entity DXF) must not produce an
        unreviewable extraction. Keep the most valuable regions up to the cap
        and say exactly what was dropped."""
        cap = settings.max_regions_per_file
        if len(chunks) <= cap:
            return chunks
        ordered = sorted(
            range(len(chunks)),
            key=lambda i: (self._REGION_PRIORITY.get(chunks[i].get("region_type"), 9), i),
        )
        keep = set(ordered[:cap])
        kept = [c for i, c in enumerate(chunks) if i in keep]
        dropped = len(chunks) - cap
        advisory = {
            "region_type": "note",
            "chunk_text": (
                f"This file contains {len(chunks):,} text regions - beyond what a "
                f"review can meaningfully cover. The {cap:,} most significant "
                f"(title blocks, components, BOMs first) were kept; {dropped:,} "
                "bulk entities were left out. If something specific is missing, "
                "export a focused extract of the drawing and upload that."
            ),
            "bbox": None,
            "confidence": "low",
            "page": 1,
            "is_drawing": None,
            "advisory": True,
        }
        return [advisory, *kept]

    def ingest_upload(self, filename: str, data: bytes, folder_id: str | None = None) -> dict:
        """Synchronous upload+extract, kept for scripts/tests - the API route
        uses store_upload + background process_upload instead."""
        import io

        stored = self.store_upload(filename, io.BytesIO(data), folder_id)
        record = self._files.get(stored["file_id"])
        return self._extract_and_store(
            stored["file_id"], stored["filename"],
            Path(stored["filename"]).suffix.lower(), record["s3_key"], data=data,
        )

    def _extract_and_store(
        self, file_id: str, filename: str, suffix: str, s3_key: str,
        data: bytes | None = None, path: str | None = None,
    ) -> dict:
        """Run extraction and persist the result. On failure the row is kept
        with status 'failed' and the error message, so the user sees what went
        wrong in the document list and can retry without re-uploading.
        Accepts an on-disk path (streaming pipeline) or raw bytes (scripts)."""
        extractor = extraction.get_extractor(suffix)
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            if path is None:
                tmp.write(data or b"")
                tmp.flush()
                path = tmp.name
            try:
                chunks = [c.model_dump(mode="json") for c in extractor.extract(path)]
            except BlueprintError as exc:
                self._files.mark_failed(file_id, s3_key, str(exc))
                raise
            except Exception as exc:
                self._files.mark_failed(
                    file_id, s3_key, f"Unexpected extraction error: {exc}"
                )
                raise

        chunks = self._guard_region_count(chunks)
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
        suffix = Path(record["filename"]).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            self._storage.download_to_path(s3_key, tmp.name)
            result = self._extract_and_store(
                file_id, record["filename"], suffix, s3_key, path=tmp.name,
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
