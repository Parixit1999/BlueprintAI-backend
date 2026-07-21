"""FastAPI dependency providers - composition root for services."""
from app.config import settings
from app.db import pool
from app.repositories import (
    ChatRepository,
    ChunkRepository,
    DrawingRepository,
    FileRepository,
    ProjectRepository,
    RegistryChunkRepository,
    StatsRepository,
)
from app.services.ai import get_embedding_provider, get_text_generator
from app.services.chat_service import ChatService
from app.services.file_service import FileService
from app.services.project_service import DrawingService, ProjectService
from app.services.query_service import QueryService
from app.services.registry_index import RegistryIndexService
from app.services.render_service import RenderService
from app.services.review_service import ReviewService
from app.services.storage import get_storage


def file_service() -> FileService:
    return FileService(FileRepository(pool), get_storage(), get_embedding_provider())


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
    )


def render_service() -> RenderService:
    return RenderService(FileRepository(pool), get_storage())


def chat_service() -> ChatService:
    # Single global user until auth arrives; then resolve user_id per request.
    return ChatService(
        ChatRepository(pool), query_service(), settings.default_user_id,
        ChunkRepository(pool), RegistryChunkRepository(pool),
    )


def stats_repository() -> StatsRepository:
    return StatsRepository(pool)
