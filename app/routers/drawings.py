from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app import authz
from app.dependencies import drawing_service, registry_index_service
from app.services.project_service import DrawingService
from app.services.registry_index import RegistryIndexService

router = APIRouter(tags=["drawings"])

Service = Annotated[DrawingService, Depends(drawing_service)]
Index = Annotated[RegistryIndexService, Depends(registry_index_service)]


class DrawingCreate(BaseModel):
    project_id: str | None = None
    set_id: str | None = None
    dwg_number: str | None = None
    description: str | None = None
    contract_number: str | None = None
    drawing_date: str | None = None
    sheet_count: int | None = None
    version_note: str | None = None


class DrawingUpdate(BaseModel):
    project_id: str | None = None
    set_id: str | None = None
    dwg_number: str | None = None
    description: str | None = None
    contract_number: str | None = None
    drawing_date: str | None = None
    sheet_count: int | None = None
    version_note: str | None = None


class LinkVersion(BaseModel):
    other_drawing_id: str = Field(min_length=1)


class AssignFile(BaseModel):
    drawing_id: str | None = None
    sheet_number: str | None = None
    new_drawing: DrawingCreate | None = None
    # create a sibling drawing linked as a new version of this drawing id,
    # and attach the file to it (the version-suggestion accept path)
    version_of: str | None = None


def _check_drawing(request: Request, service: DrawingService, drawing_id: str) -> None:
    """Role guard for one drawing: its owning project must be allowed."""
    user = request.state.user
    if authz.allowed_project_ids(user) is not None:
        authz.check_project(user, service.project_of_drawing(drawing_id))


# Sync def handlers: DB-only work runs in FastAPI's worker threadpool.
@router.post("/drawings")
def create_drawing(request: Request, body: DrawingCreate, service: Service):
    user = request.state.user
    if authz.allowed_project_ids(user) is not None:
        authz.check_project(user, body.project_id)
    return service.create(body.model_dump(exclude_unset=True))


@router.get("/drawings/{drawing_id}")
def drawing_detail(request: Request, drawing_id: str, service: Service):
    _check_drawing(request, service, drawing_id)
    return service.get_detail(drawing_id)


@router.patch("/drawings/{drawing_id}")
def update_drawing(
    request: Request, drawing_id: str, body: DrawingUpdate, service: Service
):
    _check_drawing(request, service, drawing_id)
    fields = body.model_dump(exclude_unset=True)
    if "project_id" in fields and authz.allowed_project_ids(request.state.user) is not None:
        authz.check_project(request.state.user, fields["project_id"])
    return service.update(drawing_id, fields)


@router.delete("/drawings/{drawing_id}", status_code=204)
def delete_drawing(request: Request, drawing_id: str, service: Service):
    _check_drawing(request, service, drawing_id)
    service.delete(drawing_id)


@router.post("/drawings/{drawing_id}/link-version")
def link_versions(request: Request, drawing_id: str, body: LinkVersion, service: Service):
    _check_drawing(request, service, drawing_id)
    _check_drawing(request, service, body.other_drawing_id)
    return service.link_versions(drawing_id, body.other_drawing_id)


@router.post("/drawings/{drawing_id}/unlink-version")
def unlink_version(request: Request, drawing_id: str, service: Service):
    _check_drawing(request, service, drawing_id)
    return service.unlink_version(drawing_id)


@router.delete("/sets/{set_id}", status_code=204)
def delete_set(request: Request, set_id: str, service: Service):
    user = request.state.user
    if authz.allowed_project_ids(user) is not None:
        authz.check_project(user, service.project_of_set(set_id))
    service.delete_set(set_id)


@router.get("/files/{file_id}/suggestions")
def file_suggestions(request: Request, file_id: str, service: Service):
    """Ranked project/drawing suggestions for a file, from filename signals
    (DWG number, pj####, name fragments, initials) against the registry."""
    authz.check_file(request.state.user, service.file_project_of, file_id)
    return service.suggestions_for_file(file_id)


@router.post("/files/{file_id}/assign")
def assign_file(request: Request, file_id: str, body: AssignFile, service: Service):
    """Attach a file to an existing drawing, create a new drawing and attach,
    or accept a version suggestion (version_of)."""
    user = request.state.user
    authz.check_file(user, service.file_project_of, file_id)
    if authz.allowed_project_ids(user) is not None:
        # the TARGET must be on an allowed sheet too - assigning is how a
        # document would otherwise escape into (or out of) a sheet
        if body.drawing_id:
            authz.check_project(user, service.project_of_drawing(body.drawing_id))
        if body.new_drawing is not None:
            authz.check_project(user, body.new_drawing.project_id)
        if body.version_of:
            authz.check_project(user, service.project_of_drawing(body.version_of))
    if body.version_of:
        return service.add_as_version(file_id, body.version_of)
    return service.assign_file(
        file_id,
        body.drawing_id,
        body.sheet_number,
        body.new_drawing.model_dump(exclude_unset=True) if body.new_drawing else None,
    )


@router.post("/files/{file_id}/unassign", status_code=204)
def unassign_file(request: Request, file_id: str, service: Service):
    authz.check_file(request.state.user, service.file_project_of, file_id)
    service.unassign_file(file_id)


@router.post("/registry/reindex")
def reindex_registry(index: Index):
    """Rebuild every registry metadata card (used after bulk changes or if
    the embedding service was down during edits). Sync def: embedding work
    runs in the threadpool."""
    return index.reindex_all()
