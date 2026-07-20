from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile

from app.dependencies import file_service, render_service
from app.services.file_service import FileService
from app.services.render_service import RenderService

router = APIRouter(prefix="/files", tags=["files"])

Service = Annotated[FileService, Depends(file_service)]
Renderer = Annotated[RenderService, Depends(render_service)]


@router.post("/upload")
async def upload_file(file: UploadFile, service: Service):
    """Upload a drawing (DXF, PDF, PNG, JPG); domain errors map to HTTP via the app-level handler."""
    data = await file.read()
    return service.ingest_upload(file.filename or "unnamed", data)


@router.get("")
async def list_files(service: Service):
    return service.list_files()


@router.get("/{file_id}/extraction")
async def get_extraction(file_id: str, service: Service):
    result = service.get_extraction(file_id)
    if result is None:
        raise HTTPException(status_code=404, detail="File not found")
    return result


@router.get("/{file_id}/render")
async def get_render(file_id: str, renderer: Renderer, page: Annotated[int, Query(ge=1)] = 1):
    """PNG render of one page + its extents, for the evidence viewer."""
    return renderer.get_render(file_id, page)


@router.delete("/{file_id}", status_code=204)
async def delete_file(file_id: str, service: Service):
    """Delete a document, its chunks, and its stored files. Domain errors map via the app handler."""
    service.delete_file(file_id)
