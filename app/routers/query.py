from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import query_service
from app.services.query_service import QueryService

router = APIRouter(prefix="/query", tags=["query"])

Service = Annotated[QueryService, Depends(query_service)]


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@router.post("")
async def query(request: QueryRequest, service: Service):
    return service.ask(request.question, request.top_k)
