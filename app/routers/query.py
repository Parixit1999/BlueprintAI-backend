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
def query(request: QueryRequest, service: Service):
    # Sync def: embedding + LLM generation are blocking, so FastAPI runs this in
    # its worker threadpool instead of on the event loop.
    return service.ask(request.question, request.top_k)
