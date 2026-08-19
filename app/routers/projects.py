from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app import authz
from app.dependencies import drawing_service, project_service
from app.services.project_service import DrawingService, ProjectService

router = APIRouter(prefix="/projects", tags=["projects"])

Projects = Annotated[ProjectService, Depends(project_service)]
Drawings = Annotated[DrawingService, Depends(drawing_service)]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=300)
    number: str | None = None
    description: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    number: str | None = None
    description: str | None = None


class SetCreate(BaseModel):
    set_number: str = Field(min_length=1, max_length=100)
    name: str | None = None


# Sync def handlers: DB-only work runs in FastAPI's worker threadpool.
@router.post("")
def create_project(body: ProjectCreate, service: Projects):
    return service.create(body.name, body.number, body.description)


@router.get("")
def list_projects(request: Request, service: Projects):
    allowed = authz.allowed_project_ids(request.state.user)
    projects = service.list_all()
    if allowed is not None:
        projects = [p for p in projects if p["project_id"] in allowed]
    return projects


@router.get("/{project_id}")
def project_detail(request: Request, project_id: str, service: Projects):
    authz.check_project(request.state.user, project_id)
    return service.get_detail(project_id)


@router.patch("/{project_id}")
def update_project(
    request: Request, project_id: str, body: ProjectUpdate, service: Projects
):
    authz.check_project(request.state.user, project_id)
    return service.update(project_id, body.model_dump(exclude_unset=True))


@router.delete("/{project_id}", status_code=204)
def delete_project(request: Request, project_id: str, service: Projects):
    authz.check_project(request.state.user, project_id)
    service.delete(project_id)


@router.post("/{project_id}/sets")
def create_set(request: Request, project_id: str, body: SetCreate, drawings: Drawings):
    authz.check_project(request.state.user, project_id)
    return drawings.create_set(project_id, body.set_number, body.name)
