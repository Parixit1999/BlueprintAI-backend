"""RAG: embed question -> top-k retrieval -> grounded answer with evidence."""
from app.repositories import ChunkRepository
from app.services.ai.base import EmbeddingProvider, TextGenerator

SYSTEM_PROMPT = (
    "You answer questions about engineering drawings using ONLY the provided "
    "extracted context. Each context chunk is labeled with its source region. "
    "If the context does not contain the answer, say so plainly - never guess. "
    "Keep answers short and factual; cite values exactly as written in the context."
)


class QueryService:
    def __init__(self, chunks: ChunkRepository, embedder: EmbeddingProvider, generator: TextGenerator):
        self._chunks = chunks
        self._embedder = embedder
        self._generator = generator

    def ask(self, question: str, top_k: int = 5) -> dict:
        hits = self._chunks.search(self._embedder.embed(question), top_k)
        if not hits:
            return {"answer": "No drawings have been ingested yet.", "evidence": []}

        context = "\n\n".join(
            f"[chunk {i + 1}] ({h['region_type']}) {h['chunk_text']}" for i, h in enumerate(hits)
        )
        answer = self._generator.generate(
            SYSTEM_PROMPT,
            f"Context from the drawing:\n{context}\n\nQuestion: {question}",
        )
        return {"answer": answer, "evidence": hits}
