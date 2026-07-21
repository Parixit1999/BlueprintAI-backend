"""Persistent chat: every question runs RAG; both turns are stored with the
assistant's evidence references so past answers stay verifiable.

RLHF: users rate assistant answers; each rating shifts the retrieval weight of
the evidence that answer used (clamped), so consistently-downvoted regions rank
lower in future retrieval and upvoted ones rank higher.
"""
from app.exceptions import FileNotFound
from app.repositories import ChatRepository, ChunkRepository, RegistryChunkRepository
from app.services.query_service import QueryService

TITLE_MAX = 60

# Retrieval-weight shift per rating step (rating delta of +/-1 or +/-2 when
# flipping an existing rating). Clamped in the repositories to [0.3, 2.0].
FEEDBACK_STEP = 0.1


class ChatService:
    def __init__(
        self,
        chats: ChatRepository,
        query: QueryService,
        user_id: str,
        chunks: ChunkRepository | None = None,
        registry: RegistryChunkRepository | None = None,
    ):
        self._chats = chats
        self._query = query
        self._user_id = user_id
        self._chunks = chunks
        self._registry = registry

    def rate_message(
        self, session_id: str, message_id: str, rating: int, comment: str | None = None
    ) -> dict:
        """Record feedback on an assistant answer and reweight its evidence.
        rating: 1 (helpful), -1 (not helpful), 0 (clear feedback)."""
        if self._chats.get_session(session_id, self._user_id) is None:
            raise FileNotFound("Chat session not found")
        message = self._chats.get_message(session_id, message_id)
        if message is None or message["role"] != "assistant":
            raise FileNotFound("Assistant message not found")

        if rating == 0:
            previous = self._chats.clear_rating(message_id)
        else:
            previous = self._chats.set_rating(message_id, rating, comment)
        delta = (rating - previous) * FEEDBACK_STEP
        if delta:
            evidence = message.get("evidence") or []
            chunk_ids = [e["chunk_id"] for e in evidence if e.get("chunk_id")]
            entities = [
                (e["entity_type"], e["entity_id"])
                for e in evidence
                if e.get("entity_type") and e.get("entity_id")
            ]
            if self._chunks and chunk_ids:
                self._chunks.adjust_weights(chunk_ids, delta)
            if self._registry and entities:
                self._registry.adjust_weights(entities, delta)
        return {
            "message_id": message_id,
            "rating": rating if rating != 0 else None,
            "reweighted_sources": len((message.get("evidence") or [])) if delta else 0,
        }

    def create_session(self) -> dict:
        return self._chats.create_session(self._user_id, "New chat")

    def list_sessions(self) -> list[dict]:
        return self._chats.list_sessions(self._user_id)

    def get_messages(self, session_id: str) -> dict:
        session = self._chats.get_session(session_id, self._user_id)
        if session is None:
            raise FileNotFound("Chat session not found")
        return {**session, "messages": self._chats.list_messages(session_id)}

    def rename_session(self, session_id: str, title: str) -> dict:
        if self._chats.get_session(session_id, self._user_id) is None:
            raise FileNotFound("Chat session not found")
        clean = title.strip()[:TITLE_MAX] or "New chat"
        self._chats.set_title(session_id, clean)
        return {"session_id": session_id, "title": clean}

    def delete_session(self, session_id: str) -> None:
        if self._chats.get_session(session_id, self._user_id) is None:
            raise FileNotFound("Chat session not found")
        self._chats.delete_session(session_id)

    def ask(self, session_id: str, question: str, project_id: str | None = None) -> dict:
        session = self._chats.get_session(session_id, self._user_id)
        if session is None:
            raise FileNotFound("Chat session not found")

        # capture history BEFORE storing the new question, so the model sees
        # the conversation exactly as the user did when asking
        history = self._chats.list_messages(session_id)
        user_msg = self._chats.add_message(session_id, "user", question)
        result = self._query.ask(question, project_id=project_id, history=history)
        assistant_msg = self._chats.add_message(
            session_id, "assistant", result["answer"], result["evidence"],
            result.get("version_context"),
        )

        if session["title"] == "New chat":
            title = question[:TITLE_MAX] + ("…" if len(question) > TITLE_MAX else "")
            self._chats.set_title(session_id, title)
        else:
            self._chats.touch(session_id)

        return {"user_message": user_msg, "assistant_message": assistant_msg}

    def ask_stream(self, session_id: str, question: str, project_id: str | None = None):
        """Streaming variant of ask(): yields (event, data) tuples.

        Retrieval finishes before generation starts, so the evidence goes out
        FIRST (`meta` event) and the answer text streams after it (`token`
        events) - the user sees where the answer comes from while it is still
        being written. The assistant message is stored only once generation
        completes, so a failed stream leaves no half-answer in the session.
        """
        session = self._chats.get_session(session_id, self._user_id)
        if session is None:
            raise FileNotFound("Chat session not found")

        history = self._chats.list_messages(session_id)
        user_msg = self._chats.add_message(session_id, "user", question)
        plan = self._query.plan(question, project_id=project_id, history=history)
        yield "meta", {
            "user_message": user_msg,
            "evidence": plan["evidence"],
            "version_context": plan.get("version_context"),
            "multi_drawing": plan.get("multi_drawing", False),
        }

        answer = plan["answer"]  # canned (no-match) answers skip generation
        if answer is not None:
            yield "token", {"t": answer}
        else:
            parts: list[str] = []
            try:
                for piece in self._query.stream(plan["prompt"]):
                    parts.append(piece)
                    yield "token", {"t": piece}
            except Exception as exc:  # surface mid-generation failures to the UI
                yield "error", {"detail": f"Generation failed: {exc}"}
                return
            answer = "".join(parts)

        assistant_msg = self._chats.add_message(
            session_id, "assistant", answer, plan["evidence"],
            plan.get("version_context"),
        )
        if session["title"] == "New chat":
            title = question[:TITLE_MAX] + ("…" if len(question) > TITLE_MAX else "")
            self._chats.set_title(session_id, title)
        else:
            self._chats.touch(session_id)
        yield "done", {"assistant_message": assistant_msg}
