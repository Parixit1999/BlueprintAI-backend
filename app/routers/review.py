from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import review_service
from app.services.review_service import ReviewService

router = APIRouter(prefix="/review", tags=["review"])

Service = Annotated[ReviewService, Depends(review_service)]


class ConfirmRequest(BaseModel):
    # chunk index -> corrected text; anything not listed is confirmed as-is
    corrections: dict[int, str] = {}
    # chunk indexes to drop entirely (junk extractions)
    rejected: list[int] = []


@router.post("/{file_id}/confirm")
async def confirm_and_ingest(file_id: str, body: ConfirmRequest, service: Service):
    """Domain errors (not found / already ingested) map to HTTP via the app-level handler."""
    return service.confirm_and_ingest(file_id, body.corrections, body.rejected)
