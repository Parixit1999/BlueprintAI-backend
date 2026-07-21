"""RAG: embed question -> retrieve -> scope to the most relevant drawing ->
grounded answer with evidence.

Retrieval scopes to a single drawing on purpose: a question about one part
should never cite regions from an unrelated drawing. We cast a wide net, pick
the drawing that contains the best-matching region, and keep only that
drawing's regions as both the model's context and the shown evidence.
"""
from app.repositories import ChunkRepository, RegistryChunkRepository
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


REGISTRY_POOL = 10


class QueryService:
    def __init__(
        self,
        chunks: ChunkRepository,
        embedder: EmbeddingProvider,
        generator: TextGenerator,
        registry: RegistryChunkRepository | None = None,
    ):
        self._chunks = chunks
        self._embedder = embedder
        self._generator = generator
        self._registry = registry

    def _registry_answer(self, question: str, meta_hits: list[dict]) -> dict:
        """Answer from registry metadata cards (projects, drawings, sets,
        versions) when they match the question better than any file content."""
        top = meta_hits[0]["score"]
        floor = top * RELATIVE_FLOOR
        hits = [h for h in meta_hits if h["score"] >= floor][:MAX_EVIDENCE]
        context = "\n\n".join(
            f"[{i + 1}] ({h['entity_type']} record) {h['chunk_text']}"
            for i, h in enumerate(hits)
        )
        answer = self._generator.generate(
            SYSTEM_PROMPT,
            "Context from the drawing registry (projects, drawings, sets, versions):\n"
            f"{context}\n\nQuestion: {question}",
        )
        return {"answer": answer, "evidence": hits}

    def ask(self, question: str, top_k: int = 5, project_id: str | None = None) -> dict:
        """Answer a question over the ingested drawings AND the registry
        metadata (projects, drawing metadata, sets, versions, file metadata),
        optionally scoped to one project."""
        q_embedding = self._embedder.embed(question)
        candidates = self._chunks.search(q_embedding, CANDIDATE_POOL, project_id)
        meta_hits = (
            self._registry.search(q_embedding, REGISTRY_POOL, project_id)
            if self._registry is not None
            else []
        )

        top_score = candidates[0]["score"] if candidates else 0.0
        top_meta = meta_hits[0]["score"] if meta_hits else 0.0

        # Registry metadata wins when it matches the question better than any
        # extracted drawing content ("what contract covers 11767-W-59?").
        if top_meta >= MIN_RELEVANCE and top_meta >= top_score:
            return self._registry_answer(question, meta_hits)
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

        # Tell the model where the regions come from, so answers can reference
        # the drawing naturally ("on drawing 11767-W-59 ...").
        primary = hits[0]
        source_bits = [b for b in (
            primary.get("dwg_number") and f"drawing {primary['dwg_number']}",
            primary.get("filename") and f"file {primary['filename']}",
            primary.get("project_name") and f"project {primary['project_name']}",
        ) if b]
        source_line = ", ".join(source_bits)
        context = "\n\n".join(
            f"[{i + 1}] ({h['region_type']}) {h['chunk_text']}" for i, h in enumerate(hits)
        )
        answer = self._generator.generate(
            SYSTEM_PROMPT,
            f"Context from {source_line}:\n{context}\n\nQuestion: {question}",
        )
        return {"answer": answer, "evidence": hits}
