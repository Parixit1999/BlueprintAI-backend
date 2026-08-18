from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from pydantic import BaseModel

from app import authz
from app.db import pool
from app.exceptions import Forbidden
from app.dependencies import drawing_service, file_service, render_service
from app.services import jobs
from app.repositories import DismissedDuplicateRepository, FileRepository
from app.services.file_service import FileService
from app.services.project_service import DrawingService
from app.services.render_service import RenderService

router = APIRouter(prefix="/files", tags=["files"])

Service = Annotated[FileService, Depends(file_service)]
Drawings = Annotated[DrawingService, Depends(drawing_service)]
Renderer = Annotated[RenderService, Depends(render_service)]


class StatusBatch(BaseModel):
    ids: list[str]


@router.post("/statuses")
def file_statuses(body: StatusBatch, service: Service):
    """Poll many uploads with ONE request/query: status + region count,
    never the multi-megabyte extraction payload. POST because 200+ UUIDs
    exceed sane URL lengths. Sync def: DB-only, runs in the threadpool."""
    if len(body.ids) > 500:
        raise HTTPException(status_code=422, detail="At most 500 ids per request.")
    return {"statuses": service.get_statuses(body.ids)}


@router.post("/upload")
async def upload_file(
    file: UploadFile,
    service: Service,
    drawings: Drawings,
    folder_id: Annotated[str | None, Form()] = None,
):
    """Upload a drawing: the request only validates and stores the original
    (seconds), then extraction + the assignment matcher run as a background
    task. Proxies cut connections at ~60s and dense scans extract for
    minutes, so the client POLLS the file status instead of waiting here."""
    # hand the spooled temp file straight to the streaming store - the
    # request body never materializes as one bytes object
    stored = await run_in_threadpool(
        service.store_upload, file.filename or "unnamed", file.file, folder_id
    )
    # byte-identical re-upload: the document already exists (possibly already
    # extracted or ingested) - do not restart extraction over it
    if not stored.get("existing"):
        jobs.submit(
            jobs.extract_pool, stored["file_id"],
            service.process_upload, stored["file_id"], drawings.suggest_and_maybe_assign,
        )
    return stored


# The handlers below do only synchronous, blocking work (DB queries, rendering),
# so they are declared `def`, not `async def`: FastAPI then runs them in its
# worker threadpool, keeping the event loop free to serve requests concurrently.
@router.get("")
def list_files(
    request: Request,
    service: Service,
    page: int | None = None,
    page_size: int = 10,
    q: str | None = None,
    file_type: str | None = None,
    status: str | None = None,
    assigned: str | None = None,
    drawing: str | None = None,
    dup_only: bool = False,
    sort: str = "uploaded",
    dir: str = "desc",
):
    """Paged when `page` is given ({items, total, ...} envelope with filters
    and sorting run in SQL); legacy full array otherwise (internal callers)."""
    allowed = authz.allowed_project_ids(request.state.user)
    if page is None:
        # the legacy unpaged listing has no scoping and no callers in the
        # role-restricted UI - refuse rather than leak
        if allowed is not None:
            raise Forbidden("Use the paged document listing.")
        return service.list_files()
    return service.list_files_paged(
        q=q, file_type=file_type, status=status, assigned=assigned,
        drawing=drawing, dup_only=dup_only, sort=sort, direction=dir,
        page=page, page_size=page_size, allowed_project_ids=allowed,
    )


@router.get("/{file_id}/extraction")
def get_extraction(request: Request, file_id: str, service: Service):
    authz.check_file(request.state.user, service.project_of, file_id)
    result = service.get_extraction(file_id)
    if result is None:
        raise HTTPException(status_code=404, detail="File not found")
    return result


@router.get("/{file_id}/render")
def get_render(
    request: Request, file_id: str, renderer: Renderer,
    page: Annotated[int, Query(ge=1)] = 1,
):
    """PNG render of one page + its extents, for the evidence viewer."""
    authz.check_file(request.state.user, FileRepository(pool).project_of, file_id)
    return renderer.get_render(file_id, page)


@router.post("/{file_id}/retry")
def retry_extraction(
    request: Request, file_id: str, service: Service, drawings: Drawings
):
    """Re-run extraction on a failed upload, in the background; poll status."""
    authz.check_file(request.state.user, service.project_of, file_id)
    prepared = service.prepare_retry(file_id)
    jobs.submit(
        jobs.extract_pool, file_id,
        service.process_upload, file_id, drawings.suggest_and_maybe_assign,
    )
    return prepared


@router.post("/{file_id}/reextract")
def reextract(request: Request, file_id: str, service: Service):
    """Re-read an extracted/ingested document with the current pipeline, in
    the background. Drops its knowledge-base chunks now; the document shows
    as processing until the fresh regions land - poll status."""
    authz.check_file(request.state.user, service.project_of, file_id)
    prepared = service.prepare_reextract(file_id)
    jobs.submit(jobs.extract_pool, file_id, service.process_upload, file_id, None)
    return prepared


class NotDuplicateRequest(BaseModel):
    other_file_id: str


@router.post("/{file_id}/not-duplicate", status_code=204)
def dismiss_duplicate(
    request: Request, file_id: str, body: NotDuplicateRequest, service: Service
):
    """Human veto: this pair is not a duplicate - the flag never returns."""
    authz.check_file(request.state.user, service.project_of, file_id)
    if service.get_extraction(file_id) is None:
        raise HTTPException(404, "Document not found")
    DismissedDuplicateRepository(pool).dismiss(file_id, body.other_file_id)


@router.delete("/{file_id}", status_code=204)
def delete_file(request: Request, file_id: str, service: Service):
    """Delete a document, its chunks, and its stored files. Domain errors map via the app handler."""
    authz.check_file(request.state.user, service.project_of, file_id)
    service.delete_file(file_id)
