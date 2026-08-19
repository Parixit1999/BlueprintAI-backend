from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app import authz
from app.db import pool
from app.repositories import FileRepository
from app.dependencies import review_service
from app.services import jobs
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
    request: Request, file_id: str, body: ConfirmRequest, service: Service
):
    """Domain errors (not found / already ingested) map to HTTP via the app-level handler.

    Returns in milliseconds: the file is atomically claimed ('ingesting') and
    the embedding work runs on the bounded ingest pool - NOT BackgroundTasks,
    which shares the request thread pool and lets a bulk ingest starve every
    API call into ALB 504s. Clients poll the document status, same as uploads.
    """
    authz.check_file(request.state.user, FileRepository(pool).project_of, file_id)
    result = service.start_ingest(file_id, body.corrections, body.rejected)
    jobs.submit(
        jobs.ingest_pool, file_id, service.run_ingest, file_id, body.corrections, body.rejected
    )
    return result
