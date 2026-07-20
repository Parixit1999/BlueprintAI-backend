"""RAG: embed question -> retrieve -> scope to the most relevant drawing ->
grounded answer with evidence.

Retrieval scopes to a single drawing on purpose: a question about one part
should never cite regions from an unrelated drawing. We cast a wide net, pick
the drawing that contains the best-matching region, and keep only that
drawing's regions as both the model's context and the shown evidence.
"""
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

# Cast a wide net, then narrow to one drawing.
CANDIDATE_POOL = 30
# Below this cosine similarity the best match is off-topic; nothing in the
# knowledge base answers the question. Calibrated on real data: genuine
# questions score >= ~0.52, off-topic ones <= ~0.45.
MIN_RELEVANCE = 0.50
# Within the chosen drawing, keep only regions scoring at least this fraction
# of the best region's score, so evidence is what actually supports the answer
# rather than weak padding.
RELATIVE_FLOOR = 0.60
# Hard cap on how many regions we surface as evidence.
MAX_EVIDENCE = 6

NO_MATCH = "I couldn't find anything about that in the ingested drawings."


class QueryService:
    def __init__(self, chunks: ChunkRepository, embedder: EmbeddingProvider, generator: TextGenerator):
        self._chunks = chunks
        self._embedder = embedder
        self._generator = generator

    def ask(self, question: str, top_k: int = 5) -> dict:
        candidates = self._chunks.search(self._embedder.embed(question), CANDIDATE_POOL)
        top_score = candidates[0]["score"] if candidates else 0.0
        if top_score < MIN_RELEVANCE:
            return {"answer": NO_MATCH, "evidence": []}

        # The drawing that owns the single best-matching region is the one the
        # question is about; keep only its regions so evidence stays coherent.
        primary_file_id = candidates[0]["source_file_id"]
        floor = top_score * RELATIVE_FLOOR
        hits = [
            h
            for h in candidates
            if h["source_file_id"] == primary_file_id and h["score"] >= floor
        ][:MAX_EVIDENCE]

        context = "\n\n".join(
            f"[{i + 1}] ({h['region_type']}) {h['chunk_text']}" for i, h in enumerate(hits)
        )
        answer = self._generator.generate(
            SYSTEM_PROMPT,
            f"Context from the drawing:\n{context}\n\nQuestion: {question}",
        )
        return {"answer": answer, "evidence": hits}
