from fastapi import APIRouter, HTTPException

from app.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest):
    """RAG: embed question -> top-k retrieval -> Bedrock generation -> answer + evidence crops."""
    raise HTTPException(status_code=501, detail="Not implemented: Day 5 — retrieve + generate")
