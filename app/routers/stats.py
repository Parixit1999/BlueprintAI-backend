from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app import authz
from app.dependencies import stats_repository
from app.repositories import StatsRepository

router = APIRouter(prefix="/stats", tags=["stats"])

Repo = Annotated[StatsRepository, Depends(stats_repository)]


@router.get("")
def stats(request: Request, repo: Repo):
    # Sync def: DB query runs in FastAPI's worker threadpool, off the event
    # loop. Global counts stay global for every dashboard holder (they leak
    # magnitudes, not content); the per-project panel is role-scoped.
    return repo.snapshot(authz.allowed_project_ids(request.state.user))
