"""Persistent chat: every question runs RAG; both turns are stored with the
assistant's evidence references so past answers stay verifiable."""
from app.exceptions import FileNotFound
from app.repositories import ChatRepository
from app.services.query_service import QueryService

TITLE_MAX = 60


class ChatService:
    def __init__(self, chats: ChatRepository, query: QueryService, user_id: str):
        self._chats = chats
        self._query = query
        self._user_id = user_id

    def create_session(self) -> dict:
        return self._chats.create_session(self._user_id, "New chat")

    def list_sessions(self) -> list[dict]:
        return self._chats.list_sessions(self._user_id)

    def get_messages(self, session_id: str) -> dict:
        session = self._chats.get_session(session_id, self._user_id)
        if session is None:
            raise FileNotFound("Chat session not found")
        return {**session, "messages": self._chats.list_messages(session_id)}

    def ask(self, session_id: str, question: str) -> dict:
        session = self._chats.get_session(session_id, self._user_id)
        if session is None:
            raise FileNotFound("Chat session not found")

        user_msg = self._chats.add_message(session_id, "user", question)
        result = self._query.ask(question)
        assistant_msg = self._chats.add_message(
            session_id, "assistant", result["answer"], result["evidence"]
        )

        if session["title"] == "New chat":
            title = question[:TITLE_MAX] + ("…" if len(question) > TITLE_MAX else "")
            self._chats.set_title(session_id, title)
        else:
            self._chats.touch(session_id)

        return {"user_message": user_msg, "assistant_message": assistant_msg}
