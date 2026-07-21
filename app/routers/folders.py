from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.dependencies import folder_service
from app.services.folder_service import FolderService

router = APIRouter(tags=["folders"])

Service = Annotated[FolderService, Depends(folder_service)]


class FolderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    parent_id: str | None = None


class FolderRename(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class FolderMove(BaseModel):
    parent_id: str | None = None  # null = move to root


class FileRename(BaseModel):
    filename: str = Field(min_length=1, max_length=300)


class FileMove(BaseModel):
    folder_id: str | None = None  # null = move to root


# Sync def handlers: DB-only work runs in FastAPI's worker threadpool.
@router.get("/folders/browse")
def browse(service: Service, folder_id: Annotated[str | None, Query()] = None):
    """File-manager view of one folder (root when folder_id omitted)."""
    return service.browse(folder_id)


@router.get("/folders")
def list_folders(service: Service):
    """Flat folder list for move dialogs."""
    return service.list_all()


@router.post("/folders")
def create_folder(body: FolderCreate, service: Service):
    return service.create(body.name, body.parent_id)


@router.patch("/folders/{folder_id}")
def rename_folder(folder_id: str, body: FolderRename, service: Service):
    return service.rename(folder_id, body.name)


@router.post("/folders/{folder_id}/move")
def move_folder(folder_id: str, body: FolderMove, service: Service):
    return service.move(folder_id, body.parent_id)


@router.delete("/folders/{folder_id}")
def delete_folder(folder_id: str, service: Service):
    """Recursive: deletes subfolders and the files inside (storage included)."""
    return service.delete(folder_id)


@router.patch("/files/{file_id}/name")
def rename_file(file_id: str, body: FileRename, service: Service):
    return service.rename_file(file_id, body.filename)


@router.post("/files/{file_id}/move")
def move_file(file_id: str, body: FileMove, service: Service):
    return service.move_file(file_id, body.folder_id)
