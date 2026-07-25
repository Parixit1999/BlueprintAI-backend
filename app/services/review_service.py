"""HITL checkpoint: apply human corrections, embed, and ingest confirmed chunks."""
import logging
from concurrent.futures import ThreadPoolExecutor

from app.config import settings
from app.exceptions import AlreadyIngested, FileNotFound
from app.repositories import ChunkRepository, FileRepository
from app.services.ai.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class ReviewService:
    def __init__(self, files: FileRepository, chunks: ChunkRepository, embedder: EmbeddingProvider):
        self._files = files
        self._chunks = chunks
        self._embedder = embedder

    def start_ingest(
        self, file_id: str, corrections: dict[int, str], rejected: list[int]
    ) -> dict:
        """Validate + claim, WITHOUT doing the slow work.

        Embedding a dense sheet means thousands of model calls - minutes of
        work. Run inside the HTTP request, that outlives every proxy timeout
        (CloudFront cuts at ~60s), so the client sees a failure while the
        server keeps ingesting: 'ingestion is failing' with a half-full
        knowledge base. The request now only claims the file; the caller
        schedules run_ingest as a background task and clients poll status,
        exactly like uploads.
        """
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound("File not found")
        if record["status"] == "ingested":
            raise AlreadyIngested("This document is already in the knowledge base.")
        # Atomic claim (extracted -> ingesting): without it a second confirm
        # (double-click, back button, another tab) double-inserts every chunk.
        if not self._files.claim_for_ingest(file_id):
            raise AlreadyIngested(
                "This document is already being added to the knowledge base."
            )
        return {"file_id": file_id, "status": "ingesting"}

    def run_ingest(
        self, file_id: str, corrections: dict[int, str], rejected: list[int]
    ) -> None:
        """The slow half: embed + insert every confirmed chunk. Runs in the
        background after start_ingest claimed the file."""
        record = self._files.get(file_id)
        if record is None:
            return
        try:
            # collect what actually gets ingested, then embed CONCURRENTLY -
            # embedding is the slow step (one model call per region; dense
            # sheets have hundreds), and the calls are independent
            to_ingest = []
            for i, chunk in enumerate(record["extraction"]):
                if i in rejected:
                    continue
                if chunk.get("advisory"):
                    continue  # pipeline disclosure, not drawing content
                original = chunk.get("chunk_text")
                corrected = corrections.get(i)
                text = corrected if corrected is not None else original
                if not text:
                    continue  # unreadable value with no human correction - skip
                to_ingest.append((chunk, original, corrected, text))

            with ThreadPoolExecutor(max_workers=settings.embed_concurrency) as pool:
                embeddings = list(
                    pool.map(lambda item: self._embedder.embed(item[3]), to_ingest)
                )

            for (chunk, original, corrected, text), embedding in zip(to_ingest, embeddings):
                self._chunks.insert(
                    source_file_id=file_id,
                    region_type=chunk.get("region_type", "note"),
                    chunk_text=text,
                    bbox=chunk.get("bbox"),
                    page=chunk.get("page", 1),
                    confidence=chunk.get("confidence", "high"),
                    verification_status="corrected" if corrected is not None else "confirmed",
                    original_value=original,
                    corrected_value=corrected,
                    embedding=embedding,
                )
        except Exception:
            # failed midway: drop partial chunks and return to 'extracted'
            # so the review can simply be confirmed again
            logger.exception("Ingestion failed for %s", file_id)
            self._files.release_ingest_claim(file_id)
            return

        self._files.mark_ingested(file_id)
