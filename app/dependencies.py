"""FastAPI dependency providers - composition root for services."""
from fastapi import Request

from app.config import settings
from app.db import pool
from app.repositories import (
    ChatRepository,
    ChunkRepository,
    DrawingRepository,
    FileRepository,
    FolderRepository,
    ProjectRepository,
    RegistryChunkRepository,
    StatsRepository,
)
from app.services.ai import get_embedding_provider, get_text_generator
from app.services.chat_service import ChatService
from app.services.file_service import FileService
from app.services.folder_service import FolderService
from app.services.project_service import DrawingService, ProjectService
from app.services.query_service import QueryService
from app.services.registry_index import RegistryIndexService
from app.services.render_service import RenderService
from app.services.review_service import ReviewService
from app.services.storage import get_storage


def file_service() -> FileService:
    return FileService(
        FileRepository(pool), get_storage(), get_embedding_provider(),
        index=registry_index_service(),
        drawings=DrawingRepository(pool),
        renders=render_service(),
    )


def folder_service() -> FolderService:
    return FolderService(FolderRepository(pool), FileRepository(pool), file_service())


def registry_index_service() -> RegistryIndexService:
    return RegistryIndexService(
        RegistryChunkRepository(pool),
        ProjectRepository(pool),
        DrawingRepository(pool),
        get_embedding_provider(),
    )


def project_service() -> ProjectService:
    return ProjectService(ProjectRepository(pool), DrawingRepository(pool), registry_index_service())


def drawing_service() -> DrawingService:
    return DrawingService(
        DrawingRepository(pool), ProjectRepository(pool), FileRepository(pool),
        registry_index_service(),
    )


def review_service() -> ReviewService:
    return ReviewService(FileRepository(pool), ChunkRepository(pool), get_embedding_provider())


def query_service() -> QueryService:
    return QueryService(
        ChunkRepository(pool), get_embedding_provider(), get_text_generator(),
        RegistryChunkRepository(pool), DrawingRepository(pool),
        renders=render_service(),
    )


def render_service() -> RenderService:
    return RenderService(FileRepository(pool), get_storage())


def chat_service(request: Request) -> ChatService:
    # Chats belong to the person who had them: every session is created,
    # listed, opened, and deleted under the signed-in user's id, so one
    # person's conversations never appear in another's sidebar. The
    # configured default is only a fallback for unauthenticated internal
    # callers - the auth middleware means requests always carry a user.
    user = getattr(request.state, "user", None)
    user_id = str(user["id"]) if user else settings.default_user_id
    return ChatService(
        ChatRepository(pool), query_service(), user_id,
        ChunkRepository(pool), RegistryChunkRepository(pool),
    )


def stats_repository() -> StatsRepository:
    return StatsRepository(pool)
