"""HITL checkpoint: apply human corrections, embed, and ingest confirmed chunks."""
from app.repositories import ChunkRepository, FileRepository
from app.services.ai.base import EmbeddingProvider


class FileNotFound(Exception):
    pass


class AlreadyIngested(Exception):
    pass


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
            raise FileNotFound(file_id)
        if record["status"] == "ingested":
            raise AlreadyIngested(file_id)

        ingested = 0
        for i, chunk in enumerate(record["extraction"]):
            if i in rejected:
                continue
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
                confidence=chunk.get("confidence", "high"),
                verification_status="corrected" if corrected is not None else "confirmed",
                original_value=original,
                corrected_value=corrected,
                embedding=self._embedder.embed(text),
            )
            ingested += 1

        self._files.mark_ingested(file_id)
        return {"file_id": file_id, "ingested_chunks": ingested, "rejected": len(rejected)}
