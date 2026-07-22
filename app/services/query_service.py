"""RAG: embed question -> retrieve -> scope to the most relevant drawing ->
grounded answer with evidence.

Retrieval scopes to a single drawing on purpose: a question about one part
should never cite regions from an unrelated drawing. We cast a wide net, pick
the drawing that contains the best-matching region, and keep only that
drawing's regions as both the model's context and the shown evidence.
"""
from app.repositories import ChunkRepository, DrawingRepository, RegistryChunkRepository
from app.services.ai.base import EmbeddingProvider, TextGenerator
from app.services.matching import parse_content

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
    "- Keep it concise and conversational.\n"
    "- Format for readability using GitHub-flavored markdown when it helps: use "
    "a markdown table when the answer compares or enumerates several items "
    "(drawings, versions, dimensions, materials), a short list when the answer "
    "is genuinely a list, and plain prose otherwise. Never force a table onto a "
    "single-fact answer."
)

# Cast a wide net, then narrow to one drawing.
CANDIDATE_POOL = 30
# Below this cosine similarity the best match is off-topic; nothing in the
# knowledge base answers the question. Calibrated for Titan embed v2, whose
# distribution runs much cooler and wider than the local embedders: measured
# on-topic questions score >= ~0.42, off-topic ones <= ~0.13. (For
# snowflake-arctic-embed this was 0.70: on-topic >= ~0.82, off-topic <= ~0.61.)
MIN_RELEVANCE = 0.30
# Within the chosen drawing, keep only regions scoring at least this fraction
# of the best region's score, so evidence is what actually supports the answer
# rather than weak padding.
RELATIVE_FLOOR = 0.60
# Hard cap on how many regions we surface as evidence.
MAX_EVIDENCE = 6

# Multi-drawing responses: a second drawing joins the answer only when its own
# best region scores at least this fraction of the overall best - i.e. the
# question genuinely concerns it too, not just vaguely.
MULTI_DRAWING_FLOOR = 0.85
# In multi-drawing mode, cap regions per drawing and drawings per answer so the
# combined context stays focused.
MAX_PER_DRAWING = 2
MAX_DRAWINGS = 4

NO_MATCH = "I couldn't find anything about that in the ingested drawings."


REGISTRY_POOL = 10
# Registry cards are dense entity summaries full of registry vocabulary
# ("drawing", "project", "set"), so they score generically high on any
# drawing-flavored phrasing. A registry answer must therefore beat the best
# file content by a clear margin. Calibrated for Titan embed v2: registry
# questions ("what versions exist of X?") show margins >= +0.14, content
# questions <= +0.09 (DWG-number-heavy comparisons run positive because the
# cards name the drawings - the margin keeps them on the content path, and
# borderline registry questions still get their cards via the blend +
# identifier anchoring). (For snowflake-arctic-embed this was 0.03.)
REGISTRY_MARGIN = 0.12


class QueryService:
    def __init__(
        self,
        chunks: ChunkRepository,
        embedder: EmbeddingProvider,
        generator: TextGenerator,
        registry: RegistryChunkRepository | None = None,
        drawings: DrawingRepository | None = None,
        renders=None,  # RenderService; enables visual answers when multimodal
    ):
        self._chunks = chunks
        self._embedder = embedder
        self._generator = generator
        self._registry = registry
        self._drawings = drawings
        self._renders = renders

    @staticmethod
    def _version_label(d: dict) -> str:
        label = d.get("dwg_number") or "unnumbered"
        when = d.get("drawing_date") or d.get("year")
        if when:
            label += f" ({when})"
        if d.get("version_note"):
            label += f" - {d['version_note']}"
        return label

    def _version_context(self, hits: list[dict], candidates: list[dict]) -> dict | None:
        """Version-aware retrieval: identify which drawing version answered and
        disclose sibling versions, so answers never silently blend or hide
        versions. Returns None when the answering drawing has no other versions."""
        primary = hits[0]
        group = primary.get("version_group_id")
        if not group or self._drawings is None:
            return None
        siblings = [
            v for v in self._drawings.versions(group)
            if v["drawing_id"] != primary.get("drawing_id")
        ]
        if not siblings:
            return None
        # which sibling versions ALSO matched this question (their files appear
        # in the candidate pool) - these are the "several relevant versions"
        matched_ids = {
            c.get("drawing_id")
            for c in candidates
            if c.get("version_group_id") == group and c.get("drawing_id") != primary.get("drawing_id")
        }
        return {
            "used": {
                "drawing_id": primary.get("drawing_id"),
                "label": self._version_label(primary),
            },
            "other_versions": [
                {
                    "drawing_id": v["drawing_id"],
                    "label": self._version_label(v),
                    "also_matched": v["drawing_id"] in matched_ids,
                }
                for v in siblings
            ],
        }

    @staticmethod
    def _group_label(h: dict) -> str:
        """Human-readable identity of the drawing a region came from."""
        if h.get("dwg_number"):
            label = h["dwg_number"]
            when = h.get("drawing_date") or h.get("year")
            if when:
                label += f" ({when})"
            if h.get("version_note"):
                label += f" - {h['version_note']}"
        else:
            label = h.get("filename") or "unassigned file"
        if h.get("project_name"):
            label += f", project {h['project_name']}"
        return label

    def _anchored_cards(self, question: str, already: list[dict]) -> list[dict]:
        """Identifier-anchored retrieval: a DWG number written in the question
        is an exact reference, so include those drawings' registry cards
        deterministically (every version in the group) instead of relying on
        embedding similarity to surface them."""
        if self._registry is None or self._drawings is None:
            return []
        norms = {c["norm"] for c in parse_content([question])["dwg_candidates"]}
        if not norms:
            return []
        ids = [
            d["drawing_id"]
            for d in self._drawings.search_registry()
            if d["dwg_number_norm"] in norms
        ]
        seen = {h["entity_id"] for h in already}
        return [c for c in self._registry.get_by_entity(ids) if c["entity_id"] not in seen]

    @staticmethod
    def _registry_section(registry_extra: list[dict], start: int) -> str:
        """Supplementary registry-card context appended to content answers.
        Numbered continuing from the content citations so evidence refs align."""
        if not registry_extra:
            return ""
        lines = "\n".join(
            f"[{start + i}] ({h['entity_type']} record) {h['chunk_text']}"
            for i, h in enumerate(registry_extra)
        )
        return (
            "\n\n--- Registry records (project/drawing/set metadata; use these "
            f"for counts, version lists, and set membership) ---\n{lines}"
        )

    def _multi_drawing_answer(
        self, question: str, groups: list[list[dict]],
        registry_extra: list[dict] | None = None,
    ) -> dict:
        """Combine regions from several relevant drawings, clearly attributed
        per drawing, with every region cited as evidence."""
        hits: list[dict] = []
        sections: list[str] = []
        for group in groups[:MAX_DRAWINGS]:
            top = group[0]
            kept = group[:MAX_PER_DRAWING]
            lines = "\n".join(
                f"[{len(hits) + j + 1}] ({h['region_type']}) {h['chunk_text']}"
                for j, h in enumerate(kept)
            )
            sections.append(f"--- From drawing {self._group_label(top)} ---\n{lines}")
            hits.extend(kept)
        registry_extra = registry_extra or []
        context = "\n\n".join(sections) + self._registry_section(
            registry_extra, len(hits) + 1
        )
        hits = hits + registry_extra
        prompt = (
            SYSTEM_PROMPT,
            "The relevant information spans MULTIPLE drawings. For every fact in "
            "your answer, say which drawing it comes from (use the drawing names "
            "given in the section headers). Do not blend facts from different "
            "drawings into one unattributed statement. If the question compares "
            "attributes across the drawings, answer with a GitHub-flavored "
            "markdown table: one row per attribute, one column per drawing, and "
            "a dash for anything the context does not state.\n\n"
            f"{context}\n\nQuestion: {question}",
        )
        return {
            "answer": None,
            "prompt": prompt,
            "evidence": hits,
            "version_context": None,
            "multi_drawing": True,
        }

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
        prompt = (
            SYSTEM_PROMPT,
            "Context from the drawing registry (projects, drawings, sets, versions):\n"
            f"{context}\n\nQuestion: {question}",
        )
        # registry cards describe their own version relationships in the text;
        # answers may combine several records, each cited as evidence
        return {"answer": None, "prompt": prompt, "evidence": hits, "version_context": None,
                "multi_drawing": len({h["entity_id"] for h in hits}) > 1}

    @staticmethod
    def _conversation_block(history: list[dict]) -> str:
        """Recent turns, truncated, so the model can resolve follow-ups
        ("what about its material?") within the session."""
        lines = []
        for m in history[-6:]:
            speaker = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {m['content'][:300]}")
        return "\n".join(lines)

    def ask(
        self,
        question: str,
        top_k: int = 5,
        project_id: str | None = None,
        history: list[dict] | None = None,
        file_id: str | None = None,
    ) -> dict:
        """Answer a question in one shot: plan (retrieve + build the prompt),
        then generate. Streaming callers use plan() + stream() instead."""
        result = self.plan(question, top_k, project_id, history, file_id)
        prompt = result.pop("prompt", None)
        image = result.pop("image", None)
        if result["answer"] is None and prompt:
            result["answer"] = self._generator.generate(*prompt, image=image)
        return result

    def stream(self, prompt: tuple[str, str], image: bytes | None = None):
        """Token stream for a prompt built by plan()."""
        yield from self._generator.generate_stream(*prompt, image=image)

    def _drawing_image(self, file_id: str, page: int) -> bytes | None:
        """Rendered page bytes for visual answers, bounded to the provider's
        image limits. Best-effort: a missing render never blocks an answer."""
        if self._renders is None:
            return None
        try:
            from app.services.extraction.image import ImageExtractor

            raw = self._renders.get_render_bytes(file_id, page)
            sent, _w, _h = ImageExtractor._downscale(raw)
            return sent
        except Exception:
            return None

    def plan(
        self,
        question: str,
        top_k: int = 5,
        project_id: str | None = None,
        history: list[dict] | None = None,
        file_id: str | None = None,
    ) -> dict:
        """Everything except generation: retrieve over the ingested drawings
        AND the registry metadata (projects, drawing metadata, sets, versions,
        file metadata), optionally scoped to one project, and assemble the
        generation prompt. Returns evidence/version_context/multi_drawing
        immediately plus either a canned `answer` (no-match) or a `prompt` to
        generate from - which is what makes evidence-first streaming possible.
        `history` (recent session turns) lets follow-up questions keep their
        conversation context."""
        q_embedding = self._embedder.embed(question)
        candidates = self._chunks.search(q_embedding, CANDIDATE_POOL, project_id)
        meta_hits = (
            self._registry.search(q_embedding, REGISTRY_POOL, project_id)
            if self._registry is not None
            else []
        )

        top_score = candidates[0]["score"] if candidates else 0.0
        top_meta = meta_hits[0]["score"] if meta_hits else 0.0

        # Follow-up carry-over: a terse follow-up ("and its part number?") may
        # not retrieve on its own. Re-embed it together with the previous user
        # question and retry before giving up.
        if history and max(top_score, top_meta) < MIN_RELEVANCE:
            prev_user = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"), None
            )
            if prev_user:
                carry_embedding = self._embedder.embed(f"{prev_user}\n{question}")
                candidates = self._chunks.search(carry_embedding, CANDIDATE_POOL, project_id, file_id)
                if self._registry is not None and file_id is None:
                    meta_hits = self._registry.search(carry_embedding, REGISTRY_POOL, project_id)
                top_score = candidates[0]["score"] if candidates else 0.0
                top_meta = meta_hits[0]["score"] if meta_hits else 0.0

        # Registry metadata wins only when it clearly dominates the extracted
        # content ("what contract covers 11767-W-59?") - see REGISTRY_MARGIN.
        convo = self._conversation_block(history) if history else ""
        convo_prefix = (
            f"Conversation so far (the question may refer back to it):\n{convo}\n\n"
            if convo
            else ""
        )
        if top_meta >= MIN_RELEVANCE and top_meta >= top_score + REGISTRY_MARGIN:
            return self._registry_answer(convo_prefix + question, meta_hits)
        if top_score < MIN_RELEVANCE:
            return {"answer": NO_MATCH, "prompt": None, "evidence": [],
                    "version_context": None, "multi_drawing": False}

        # Registry cards that are relevant but did not win outright still know
        # things the file content cannot (full drawing lists, version links,
        # set membership) - blend them into the content answer as supplementary
        # context instead of dropping them, so "how many drawings / which
        # versions" phrasings get complete answers.
        registry_extra = [h for h in meta_hits if h["score"] >= MIN_RELEVANCE]
        registry_extra = (
            self._anchored_cards(question, registry_extra) + registry_extra
        )[:3]

        # Group candidates by the drawing (or file, when unassigned) they belong
        # to. If several drawings each match the question strongly, answer from
        # all of them with per-drawing attribution; otherwise keep the original
        # single-drawing scoping so narrow questions stay precise.
        floor = top_score * RELATIVE_FLOOR
        grouped: dict[str, list[dict]] = {}
        for h in candidates:
            if h["score"] < floor:
                continue
            key = h.get("drawing_id") or h["source_file_id"]
            grouped.setdefault(key, []).append(h)
        multi_floor = max(MIN_RELEVANCE, top_score * MULTI_DRAWING_FLOOR)
        qualifying = sorted(
            (g for g in grouped.values() if g[0]["score"] >= multi_floor),
            key=lambda g: g[0]["score"],
            reverse=True,
        )
        if len(qualifying) >= 2:
            return self._multi_drawing_answer(
                convo_prefix + question, qualifying, registry_extra
            )

        # Single-drawing mode: the drawing that owns the best-matching region is
        # the one the question is about; keep only its regions.
        primary_file_id = candidates[0]["source_file_id"]
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

        version_context = self._version_context(hits, candidates)
        version_line = ""
        if version_context:
            others = "; ".join(v["label"] for v in version_context["other_versions"])
            version_line = (
                f"\nNote: this context is from version {version_context['used']['label']} "
                f"of the drawing. Other versions exist: {others}. Mention in your answer "
                "which version the information comes from, and do NOT claim anything about "
                "the other versions' content - you have not seen them."
            )

        context = "\n\n".join(
            f"[{i + 1}] ({h['region_type']}) {h['chunk_text']}" for i, h in enumerate(hits)
        ) + self._registry_section(registry_extra, len(hits) + 1)
        # Visual answers: the question concerns ONE drawing, so the model also
        # SEES its rendered page and can describe what is depicted - layout,
        # geometry, how parts relate - not just recite extracted text.
        image = self._drawing_image(primary_file_id, hits[0].get("page") or 1)
        image_note = (
            "\nYou can also SEE the drawing image. Use it to describe what the "
            "drawing depicts (layout, geometry, how parts relate) when the "
            "question calls for description - but quote exact values "
            "(dimensions, part numbers, materials) only from the extracted "
            "context, since that is what the citations point to."
            if image
            else ""
        )
        prompt = (
            SYSTEM_PROMPT,
            f"{convo_prefix}Context from {source_line}:{version_line}{image_note}\n{context}\n\n"
            f"Question: {question}",
        )
        return {"answer": None, "prompt": prompt, "image": image,
                "evidence": hits + registry_extra,
                "version_context": version_context, "multi_drawing": False}
