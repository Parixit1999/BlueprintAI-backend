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
    # loop. The dashboard describes what THIS person can see: archive counts
    # cover the sheets their role allows, chat counts cover their own
    # questions. Admins and all-sheets roles get the whole archive.
    user = request.state.user
    return repo.snapshot(
        authz.allowed_project_ids(user), user_id=str(user["id"])
    )
