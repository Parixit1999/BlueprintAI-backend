"""RAG: embed question -> top-k retrieval -> grounded answer with evidence."""
from app.repositories import ChunkRepository
from app.services.ai.base import EmbeddingProvider, TextGenerator

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about engineering "
    "drawings, using ONLY the extracted context provided with each question.\n\n"
    "Write like a knowledgeable colleague: a natural, complete sentence or two. "
    "State the answer directly and quote any values (dimensions, tolerances, "
    "part numbers, materials) exactly as they appear in the context.\n\n"
    "Rules:\n"
    "- Never invent or guess. If the context does not contain the answer, say "
    "so plainly (e.g. \"I couldn't find that in the drawing.\").\n"
    "- Do NOT mention chunks, context, sources, indices, or reference numbers in "
    "your answer. The user is shown the source regions separately, so never write "
    "things like \"(Source: chunk 3)\" or \"according to the context\".\n"
    "- Keep it concise and conversational; no bullet lists unless the answer is "
    "genuinely a list of items from the drawing."
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
