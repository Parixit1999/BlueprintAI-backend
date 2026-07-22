"""HITL checkpoint: apply human corrections, embed, and ingest confirmed chunks."""
from app.exceptions import AlreadyIngested, FileNotFound
from app.repositories import ChunkRepository, FileRepository
from app.services.ai.base import EmbeddingProvider


class ReviewService:
    def __init__(self, files: FileRepository, chunks: ChunkRepository, embedder: EmbeddingProvider):
        self._files = files
        self._chunks = chunks
        self._embedder = embedder

    def confirm_and_ingest(
        self, file_id: str, corrections: dict[int, str], rejected: list[int]
    ) -> dict:
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound("File not found")
        if record["status"] == "ingested":
            raise AlreadyIngested("This document is already in the knowledge base.")
        # Atomic claim (extracted -> ingesting): embedding every region takes
        # minutes on dense sheets, and without the claim a second confirm
        # (double-click, back button, another tab) double-inserts every chunk.
        if not self._files.claim_for_ingest(file_id):
            raise AlreadyIngested(
                "This document is already being added to the knowledge base."
            )

        try:
            ingested = 0
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
                    embedding=self._embedder.embed(text),
                )
                ingested += 1
        except Exception:
            # failed midway: drop partial chunks and return to 'extracted'
            # so the review can simply be confirmed again
            self._files.release_ingest_claim(file_id)
            raise

        self._files.mark_ingested(file_id)
        return {"file_id": file_id, "ingested_chunks": ingested, "rejected": len(rejected)}
