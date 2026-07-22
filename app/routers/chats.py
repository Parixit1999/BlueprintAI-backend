import json
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.dependencies import chat_service
from app.exceptions import BlueprintError
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chats", tags=["chats"])

Service = Annotated[ChatService, Depends(chat_service)]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    project_id: str | None = None  # optional: scope retrieval to one project
    file_id: str | None = None  # optional: chat about one specific document


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class FeedbackRequest(BaseModel):
    rating: int = Field(ge=-1, le=1)  # 1 helpful, -1 not helpful, 0 clear
    comment: str | None = Field(default=None, max_length=1000)


# Sync def handlers: all do blocking work (DB, and for `ask` the embedding + LLM
# generation), so FastAPI runs them in its worker threadpool, off the event loop.
@router.post("")
def create_session(service: Service):
    return service.create_session()


@router.get("")
def list_sessions(service: Service):
    return service.list_sessions()


@router.get("/{session_id}")
def get_messages(session_id: str, service: Service):
    return service.get_messages(session_id)


@router.patch("/{session_id}")
def rename_session(session_id: str, body: RenameRequest, service: Service):
    return service.rename_session(session_id, body.title)


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str, service: Service):
    service.delete_session(session_id)


@router.post("/{session_id}/messages")
def ask(session_id: str, body: AskRequest, service: Service):
    return service.ask(session_id, body.question, body.project_id, body.file_id)


@router.post("/{session_id}/messages/stream")
def ask_stream(session_id: str, body: AskRequest, service: Service):
    """SSE stream: `meta` (user message + evidence, sent before generation
    starts), then `token` events as the answer is written, then `done` with
    the stored assistant message. Sync generator: FastAPI iterates it in the
    worker threadpool, so the blocking LLM stream stays off the event loop."""

    def sse():
        try:
            for event, data in service.ask_stream(session_id, body.question, body.project_id, body.file_id):
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
        except BlueprintError as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{session_id}/messages/{message_id}/feedback")
def rate_message(session_id: str, message_id: str, body: FeedbackRequest, service: Service):
    """RLHF: rate an answer; the rating reweights the evidence it used."""
    return service.rate_message(session_id, message_id, body.rating, body.comment)
