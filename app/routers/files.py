from fastapi import APIRouter, HTTPException, UploadFile

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload")
async def upload_file(file: UploadFile):
    """Accept a drawing (DXF / vector PDF for MVP), store to S3, kick off extraction."""
    raise HTTPException(status_code=501, detail="Not implemented: Day 2 — upload to S3 + extraction")


@router.get("")
async def list_files():
    raise HTTPException(status_code=501, detail="Not implemented: Day 2")


@router.get("/{file_id}/extraction")
async def get_extraction(file_id: str):
    """Return the extraction result for a file, for the HITL review UI."""
    raise HTTPException(status_code=501, detail="Not implemented: Day 3")
