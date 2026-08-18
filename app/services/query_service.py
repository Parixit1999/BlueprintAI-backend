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
    "drawings, using the extracted context provided with each question and, "
    "when one is attached, the drawing image itself.\n\n"
    "Write like a knowledgeable colleague: a natural, complete sentence or two. "
    "State the answer directly and quote any values (dimensions, tolerances, "
    "part numbers, materials) exactly as they appear in the context.\n\n"
    "Rules:\n"
    "- Never invent or guess. But before saying anything is missing or "
    "unknown, EXHAUST your evidence: when a drawing image is attached, "
    "examine it carefully - markings, symbols, callouts, line work, how parts "
    "relate - and answer from direct observation, saying so (e.g. \"Looking "
    "at the drawing, ...\"). A bare refusal when the image was never examined "
    "is a wrong answer.\n"
    "- If neither the context nor careful inspection of the image settles the "
    "question, say plainly what you looked for and what you DID observe, then "
    "state what is missing - never just \"I couldn't find that\".\n"
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

# When nothing in the knowledge base matches, the answer is generated from
# this prompt instead of a flat refusal: greetings and "how do I use this?"
# get a helpful, app-aware reply, and drawing-content questions get an honest
# "not found" plus what to check. No evidence is attached either way.
ASSISTANT_PROMPT = (
    "You are the assistant inside BlueprintAI, a drawing-intelligence tool for "
    "engineering drawing archives. The user's message did not match any content "
    "in their drawing knowledge base.\n\n"
    "How to respond:\n"
    "- If it is a greeting or small talk, reply warmly in one or two sentences "
    "and mention you can answer questions about their drawings.\n"
    "- If they ask what you can do or how to use the tool, explain briefly and "
    "concretely using the capabilities below.\n"
    "- If the message looks like a question about drawing content, say plainly "
    "that you couldn't find it in the ingested drawings, and suggest checking "
    "that the document is uploaded and confirmed (Documents page), or asking "
    "with the drawing number.\n"
    "- Never invent drawing content. Keep it short and friendly.\n\n"
    "What BlueprintAI can do:\n"
    "- Upload drawings (DXF, DWG, PDF, PNG, JPG, TIFF - or a ZIP of many); the "
    "AI reads each drawing, extracts every text region, and writes a short "
    "description of what it depicts.\n"
    "- Review before ingest: every extracted region is verified by a human on "
    "the Documents page (correct or reject, then Confirm & ingest).\n"
    "- Organize: projects, drawing sets, versions of the same drawing, and a "
    "Files page with folders. Uploads are auto-assigned to drawings when the "
    "file name or content identifies them.\n"
    "- Ask questions in this chat: answers cite the exact source regions, "
    "highlight them on the drawing, handle multiple drawings and versions, and "
    "can be scoped to one project with the selector on the left.\n"
    "- Rate answers with the thumbs - feedback improves future retrieval."
)


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


# Follow-up questions lose their meaning when searched literally ("how many
# drawings does IT have?") - the pronoun matches nothing, and generic words
# can hit the wrong document. Before retrieval, follow-ups are rewritten into
# a standalone query using the conversation, so the search targets what the
# user actually meant. The original wording still goes to the generator.
REWRITE_PROMPT = (
    "You turn a follow-up chat message into ONE standalone search query about "
    "engineering drawings. Resolve references like 'it', 'this document', "
    "'each one' from the conversation. Keep drawing numbers and file names "
    "verbatim. If the message is already self-contained, return it unchanged. "
    "Return ONLY the query text - no quotes, no explanation."
)


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

    def _anchored_cards(
        self, question: str, already: list[dict],
        allowed_project_ids: list[str] | None = None,
    ) -> list[dict]:
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
        cards = [c for c in self._registry.get_by_entity(ids) if c["entity_id"] not in seen]
        if allowed_project_ids is not None:
            # role scope: anchored cards obey the same rule as searched ones -
            # owned by an allowed sheet, or owned by no sheet at all
            cards = [
                c for c in cards
                if c.get("project_id") is None or c["project_id"] in allowed_project_ids
            ]
        return cards

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
        # Visual evidence for the BEST-matching drawing: text spans several
        # drawings, but visual questions (markings, symbols, what something
        # depicts) are answered by looking at the most relevant sheet.
        # Prefer the drawing the user actually NAMED - as a chunk group, or
        # failing that as a registry card (its content may not have matched
        # the query even though the user asked about it directly). Fall back
        # to the best-scoring group.
        q_lower = question.lower()
        named = next(
            (
                g
                for g in groups[:MAX_DRAWINGS]
                if g[0].get("dwg_number") and g[0]["dwg_number"].lower() in q_lower
            ),
            None,
        )
        top_hit = (named or groups[0])[0]
        image = (
            self._drawing_image(top_hit["source_file_id"], top_hit.get("page") or 1)
            if top_hit.get("source_file_id")
            else None
        )
        if named is None and self._drawings is not None:
            named_card = next(
                (
                    r
                    for r in registry_extra
                    if r.get("entity_type") == "drawing"
                    and r.get("label")
                    and r["label"].lower() in q_lower
                ),
                None,
            )
            if named_card:
                files = self._drawings.files_for_drawing(named_card["entity_id"])
                usable = [f for f in files if f.get("status") in ("extracted", "ingested")]
                if usable:
                    named_image = self._drawing_image(usable[0]["file_id"], 1)
                    if named_image:
                        image = named_image
                        top_hit = {"dwg_number": named_card["label"]}
        image_note = (
            f"\nThe sheet image for drawing {self._group_label(top_hit)} is "
            "ATTACHED - it is first-class evidence. For questions about "
            "markings, symbols, components, or what the drawing depicts, "
            "examine the image and answer from direct observation (prefixed "
            "like \"Looking at the drawing, ...\") before concluding anything "
            "is missing. The other drawings are represented by text only.\n"
            if image
            else ""
        )
        prompt = (
            SYSTEM_PROMPT,
            "The relevant information spans MULTIPLE drawings. For every fact in "
            "your answer, say which drawing it comes from (use the drawing names "
            "given in the section headers). Do not blend facts from different "
            "drawings into one unattributed statement. If the question compares "
            "attributes across the drawings, answer with a GitHub-flavored "
            "markdown table: one row per attribute, one column per drawing, and "
            f"a dash for anything the context does not state.\n{image_note}\n"
            f"{context}\n\nQuestion: {question}",
        )
        return {
            "answer": None,
            "prompt": prompt,
            "image": image,
            "evidence": hits,
            "version_context": None,
            "multi_drawing": True,
        }

    def _registry_answer(
        self, question: str, meta_hits: list[dict],
        allowed_project_ids: list[str] | None = None,
    ) -> dict:
        """Answer from registry metadata cards (projects, drawings, sets,
        versions) when they match the question better than any file content."""
        top = meta_hits[0]["score"]
        floor = top * RELATIVE_FLOOR
        hits = [h for h in meta_hits if h["score"] >= floor][:MAX_EVIDENCE]
        # A DWG number typed in the question is an exact reference. At
        # thousands-of-cards scale, embeddings cannot tell 10951-W-20 from
        # 10551-W-25 - anchored cards are deterministic, so they LEAD and
        # are never crowded out by the semantic pool.
        anchored = self._anchored_cards(question, hits, allowed_project_ids)
        if anchored:
            hits = (anchored + hits)[: max(MAX_EVIDENCE, len(anchored))]
        context = "\n\n".join(
            f"[{i + 1}] ({h['entity_type']} record) {h['chunk_text']}"
            for i, h in enumerate(hits)
        )
        # Registry cards are METADATA - when the question is really about the
        # drawing itself ("what does the 6 marking mean?"), the answer is on
        # the sheet, not in the registry. If the hits name one drawing that
        # has a file, attach its image so the model can LOOK.
        image = None
        drawing_ids = {h["entity_id"] for h in hits if h["entity_type"] == "drawing"}
        if len(drawing_ids) == 1 and self._drawings is not None:
            files = self._drawings.files_for_drawing(next(iter(drawing_ids)))
            usable = [f for f in files if f.get("status") in ("extracted", "ingested")]
            if usable:
                image = self._drawing_image(usable[0]["file_id"], 1)
        image_note = (
            "\nThe registry records above are METADATA. The drawing image "
            "itself is ATTACHED - examine it and answer visual questions "
            "(markings, symbols, components, layout) from direct observation, "
            "prefixed like \"Looking at the drawing, ...\"."
            if image
            else ""
        )
        prompt = (
            SYSTEM_PROMPT,
            "Context from the drawing registry (projects, drawings, sets, versions):"
            f"{image_note}\n{context}\n\nQuestion: {question}",
        )
        # registry cards describe their own version relationships in the text;
        # answers may combine several records, each cited as evidence
        return {"answer": None, "prompt": prompt, "image": image, "evidence": hits,
                "version_context": None,
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
        allowed_project_ids: list[str] | None = None,
    ) -> dict:
        """Answer a question in one shot: plan (retrieve + build the prompt),
        then generate. Streaming callers use plan() + stream() instead."""
        result = self.plan(question, top_k, project_id, history, file_id,
                           allowed_project_ids)
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

    def _contextualize(self, question: str, history: list[dict]) -> str:
        """Standalone search query for a follow-up. Best-effort: any failure
        or degenerate rewrite falls back to the original question."""
        try:
            convo = self._conversation_block(history)
            rewritten = self._generator.generate(
                REWRITE_PROMPT,
                f"Conversation so far:\n{convo}\n\nLatest message: {question}",
            ).strip().strip('"')
            if 0 < len(rewritten) <= 400:
                return rewritten
        except Exception:
            pass
        return question

    def plan(
        self,
        question: str,
        top_k: int = 5,
        project_id: str | None = None,
        history: list[dict] | None = None,
        file_id: str | None = None,
        allowed_project_ids: list[str] | None = None,
    ) -> dict:
        """Everything except generation: retrieve over the ingested drawings
        AND the registry metadata (projects, drawing metadata, sets, versions,
        file metadata), optionally scoped to one project, and assemble the
        generation prompt. Returns evidence/version_context/multi_drawing
        immediately plus either a canned `answer` (no-match) or a `prompt` to
        generate from - which is what makes evidence-first streaming possible.
        `history` (recent session turns) lets follow-up questions keep their
        conversation context."""
        search_question = self._contextualize(question, history) if history else question
        q_embedding = self._embedder.embed(search_question)
        candidates = self._chunks.search(
            q_embedding, CANDIDATE_POOL, project_id, file_id,
            allowed_project_ids=allowed_project_ids,
        )
        # file-scoped chat is about ONE document; other entities' registry
        # cards are noise (identifier-anchored cards still apply below)
        meta_hits = (
            self._registry.search(q_embedding, REGISTRY_POOL, project_id,
                                  allowed_project_ids=allowed_project_ids)
            if self._registry is not None and file_id is None
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
                candidates = self._chunks.search(
                    carry_embedding, CANDIDATE_POOL, project_id, file_id,
                    allowed_project_ids=allowed_project_ids,
                )
                if self._registry is not None and file_id is None:
                    meta_hits = self._registry.search(
                        carry_embedding, REGISTRY_POOL, project_id,
                        allowed_project_ids=allowed_project_ids,
                    )
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
            return self._registry_answer(
                convo_prefix + question, meta_hits, allowed_project_ids
            )
        if top_score < MIN_RELEVANCE and file_id and candidates:
            # scoped to one document the intent is unambiguous - answer from
            # its best regions (the summary embeds close to almost anything)
            pass
        elif top_score < MIN_RELEVANCE:
            # off-corpus: greet / explain the tool / honest not-found - see
            # ASSISTANT_PROMPT. Generated (so it streams), never cited.
            return {"answer": None,
                    "prompt": (ASSISTANT_PROMPT, f"{convo_prefix}User message: {question}"),
                    "evidence": [], "version_context": None, "multi_drawing": False}

        # Registry cards that are relevant but did not win outright still know
        # things the file content cannot (full drawing lists, version links,
        # set membership) - blend them into the content answer as supplementary
        # context instead of dropping them, so "how many drawings / which
        # versions" phrasings get complete answers.
        registry_extra = [h for h in meta_hits if h["score"] >= MIN_RELEVANCE]
        registry_extra = (
            self._anchored_cards(search_question, registry_extra, allowed_project_ids)
            + registry_extra
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
            "\nThe drawing image is ATTACHED - it is first-class evidence, not "
            "decoration. Examine it before answering: identify components, "
            "symbols, balloons/callouts, and spatial relationships by looking. "
            "If the extracted context does not answer the question, search the "
            "image for the answer and report what you observe (prefixed like "
            "\"Looking at the drawing, ...\") before concluding anything is "
            "missing. For exact printed values (dimensions, part numbers, "
            "materials), prefer the extracted context when it has them, since "
            "citations point there; observed values you read off the image "
            "yourself should be flagged as read from the drawing."
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
