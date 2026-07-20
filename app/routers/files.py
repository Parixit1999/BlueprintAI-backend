from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.dependencies import file_service
from app.services.file_service import ExtractionFailed, FileService, UnsupportedFileType

router = APIRouter(prefix="/files", tags=["files"])

Service = Annotated[FileService, Depends(file_service)]


@router.post("/upload")
async def upload_file(file: UploadFile, service: Service):
    data = await file.read()
    try:
        return service.ingest_upload(file.filename or "unnamed", data)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ExtractionFailed as exc:
        raise HTTPException(status_code=422, detail=f"Extraction failed: {exc}")


@router.get("")
async def list_files(service: Service):
    return service.list_files()


@router.get("/{file_id}/extraction")
async def get_extraction(file_id: str, service: Service):
    result = service.get_extraction(file_id)
    if result is None:
        raise HTTPException(status_code=404, detail="File not found")
    return result
