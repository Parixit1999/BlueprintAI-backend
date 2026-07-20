from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import stats_repository
from app.repositories import StatsRepository

router = APIRouter(prefix="/stats", tags=["stats"])

Repo = Annotated[StatsRepository, Depends(stats_repository)]


@router.get("")
async def stats(repo: Repo):
    return repo.snapshot()
