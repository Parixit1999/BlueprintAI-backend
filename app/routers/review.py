from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/review", tags=["review"])


@router.post("/{file_id}/confirm")
async def confirm_and_ingest(file_id: str):
    """HITL checkpoint 1: apply confirmations/corrections, then embed + store chunks.

    Only confirmed/corrected (or high-confidence) fields flow into the vector DB.
    """
    raise HTTPException(status_code=501, detail="Not implemented: Day 3-4 — HITL confirm + ingest")
