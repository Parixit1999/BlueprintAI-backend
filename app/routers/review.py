from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends
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
def confirm_and_ingest(
    file_id: str, body: ConfirmRequest, service: Service, background: BackgroundTasks
):
    """Domain errors (not found / already ingested) map to HTTP via the app-level handler.

    Returns in milliseconds: the file is atomically claimed ('ingesting') and
    the embedding work runs as a background task - a dense sheet takes minutes,
    far beyond proxy timeouts. Clients poll the document status, same as uploads.
    """
    result = service.start_ingest(file_id, body.corrections, body.rejected)
    background.add_task(service.run_ingest, file_id, body.corrections, body.rejected)
    return result
