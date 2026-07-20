"""Postgres connection pool, opened/closed by the FastAPI lifespan."""
from psycopg_pool import ConnectionPool

from app.config import settings

pool = ConnectionPool(
    settings.database_url,
    min_size=1,
    max_size=10,
    open=False,
    kwargs={"autocommit": True},
)
