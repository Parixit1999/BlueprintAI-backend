from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.dependencies import review_service
from app.services.review_service import AlreadyIngested, FileNotFound, ReviewService

router = APIRouter(prefix="/review", tags=["review"])

Service = Annotated[ReviewService, Depends(review_service)]


class ConfirmRequest(BaseModel):
    # chunk index -> corrected text; anything not listed is confirmed as-is
    corrections: dict[int, str] = {}
    # chunk indexes to drop entirely (junk extractions)
    rejected: list[int] = []


@router.post("/{file_id}/confirm")
async def confirm_and_ingest(file_id: str, body: ConfirmRequest, service: Service):
    try:
        return service.confirm_and_ingest(file_id, body.corrections, body.rejected)
    except FileNotFound:
        raise HTTPException(status_code=404, detail="File not found")
    except AlreadyIngested:
        raise HTTPException(status_code=409, detail="File already ingested")
