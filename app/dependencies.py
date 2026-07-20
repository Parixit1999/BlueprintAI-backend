"""FastAPI dependency providers - composition root for services."""
from app.db import pool
from app.repositories import ChunkRepository, FileRepository
from app.services.ai import get_embedding_provider, get_text_generator
from app.services.file_service import FileService
from app.services.query_service import QueryService
from app.services.review_service import ReviewService
from app.services.storage import get_storage


def file_service() -> FileService:
    return FileService(FileRepository(pool), get_storage())


def review_service() -> ReviewService:
    return ReviewService(FileRepository(pool), ChunkRepository(pool), get_embedding_provider())


def query_service() -> QueryService:
    return QueryService(ChunkRepository(pool), get_embedding_provider(), get_text_generator())
