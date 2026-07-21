from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import chat_service
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chats", tags=["chats"])

Service = Annotated[ChatService, Depends(chat_service)]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    project_id: str | None = None  # optional: scope retrieval to one project


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
    return service.ask(session_id, body.question, body.project_id)


@router.post("/{session_id}/messages/{message_id}/feedback")
def rate_message(session_id: str, message_id: str, body: FeedbackRequest, service: Service):
    """RLHF: rate an answer; the rating reweights the evidence it used."""
    return service.rate_message(session_id, message_id, body.rating, body.comment)
