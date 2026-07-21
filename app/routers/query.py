from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import query_service
from app.services.query_service import QueryService

router = APIRouter(prefix="/query", tags=["query"])

Service = Annotated[QueryService, Depends(query_service)]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    project_id: str | None = None  # optional: scope retrieval to one project


@router.post("")
def query(request: QueryRequest, service: Service):
    # Sync def: embedding + LLM generation are blocking, so FastAPI runs this in
    # its worker threadpool instead of on the event loop.
    return service.ask(request.question, request.top_k, request.project_id)
