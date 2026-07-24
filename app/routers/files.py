from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from pydantic import BaseModel

from app.db import pool
from app.dependencies import drawing_service, file_service, render_service
from app.repositories import DismissedDuplicateRepository
from app.services.file_service import FileService
from app.services.project_service import DrawingService
from app.services.render_service import RenderService

router = APIRouter(prefix="/files", tags=["files"])

Service = Annotated[FileService, Depends(file_service)]
Drawings = Annotated[DrawingService, Depends(drawing_service)]
Renderer = Annotated[RenderService, Depends(render_service)]


@router.post("/upload")
async def upload_file(
    file: UploadFile,
    service: Service,
    drawings: Drawings,
    background: BackgroundTasks,
    folder_id: Annotated[str | None, Form()] = None,
):
    """Upload a drawing: the request only validates and stores the original
    (seconds), then extraction + the assignment matcher run as a background
    task. Proxies cut connections at ~60s and dense scans extract for
    minutes, so the client POLLS the file status instead of waiting here."""
    data = await file.read()
    stored = await run_in_threadpool(
        service.store_upload, file.filename or "unnamed", data, folder_id
    )
    background.add_task(
        service.process_upload, stored["file_id"], drawings.suggest_and_maybe_assign
    )
    return stored


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
def retry_extraction(
    file_id: str, service: Service, drawings: Drawings, background: BackgroundTasks
):
    """Re-run extraction on a failed upload, in the background; poll status."""
    prepared = service.prepare_retry(file_id)
    background.add_task(
        service.process_upload, file_id, drawings.suggest_and_maybe_assign
    )
    return prepared


@router.post("/{file_id}/reextract")
def reextract(file_id: str, service: Service, background: BackgroundTasks):
    """Re-read an extracted/ingested document with the current pipeline, in
    the background. Drops its knowledge-base chunks now; the document shows
    as processing until the fresh regions land - poll status."""
    prepared = service.prepare_reextract(file_id)
    background.add_task(service.process_upload, file_id, None)
    return prepared


class NotDuplicateRequest(BaseModel):
    other_file_id: str


@router.post("/{file_id}/not-duplicate", status_code=204)
def dismiss_duplicate(file_id: str, body: NotDuplicateRequest, service: Service):
    """Human veto: this pair is not a duplicate - the flag never returns."""
    if service.get_extraction(file_id) is None:
        raise HTTPException(404, "Document not found")
    DismissedDuplicateRepository(pool).dismiss(file_id, body.other_file_id)


@router.delete("/{file_id}", status_code=204)
def delete_file(file_id: str, service: Service):
    """Delete a document, its chunks, and its stored files. Domain errors map via the app handler."""
    service.delete_file(file_id)
