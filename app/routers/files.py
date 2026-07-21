from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.dependencies import file_service, render_service
from app.services.file_service import FileService
from app.services.render_service import RenderService

router = APIRouter(prefix="/files", tags=["files"])

Service = Annotated[FileService, Depends(file_service)]
Renderer = Annotated[RenderService, Depends(render_service)]


@router.post("/upload")
async def upload_file(
    file: UploadFile,
    service: Service,
    folder_id: Annotated[str | None, Form()] = None,
):
    """Upload a drawing (DXF, PDF, PNG, JPG), optionally into a folder; domain
    errors map to HTTP via the app-level handler."""
    data = await file.read()
    # Extraction is slow and blocking (vision/LLM calls, DB writes). Run it in a
    # worker thread so a single in-progress upload can't freeze the event loop
    # and stall every other request (document list, chat, etc.).
    return await run_in_threadpool(
        service.ingest_upload, file.filename or "unnamed", data, folder_id
    )


# The handlers below do only synchronous, blocking work (DB queries, rendering),
# so they are declared `def`, not `async def`: FastAPI then runs them in its
# worker threadpool, keeping the event loop free to serve requests concurrently.
@router.get("")
def list_files(service: Service):
    return service.list_files()


@router.get("/{file_id}/extraction")
def get_extraction(file_id: str, service: Service):
    result = service.get_extraction(file_id)
    if result is None:
        raise HTTPException(status_code=404, detail="File not found")
    return result


@router.get("/{file_id}/render")
def get_render(file_id: str, renderer: Renderer, page: Annotated[int, Query(ge=1)] = 1):
    """PNG render of one page + its extents, for the evidence viewer."""
    return renderer.get_render(file_id, page)


@router.post("/{file_id}/retry")
def retry_extraction(file_id: str, service: Service):
    """Re-run extraction on a failed upload from the stored original. Sync def:
    extraction is blocking, so it runs in the worker threadpool."""
    return service.retry_extraction(file_id)


@router.delete("/{file_id}", status_code=204)
def delete_file(file_id: str, service: Service):
    """Delete a document, its chunks, and its stored files. Domain errors map via the app handler."""
    service.delete_file(file_id)
