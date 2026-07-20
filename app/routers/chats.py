from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import chat_service
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chats", tags=["chats"])

Service = Annotated[ChatService, Depends(chat_service)]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@router.post("")
async def create_session(service: Service):
    return service.create_session()


@router.get("")
async def list_sessions(service: Service):
    return service.list_sessions()


@router.get("/{session_id}")
async def get_messages(session_id: str, service: Service):
    return service.get_messages(session_id)


@router.post("/{session_id}/messages")
async def ask(session_id: str, body: AskRequest, service: Service):
    return service.ask(session_id, body.question)
