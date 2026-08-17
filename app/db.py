"""Postgres connection pool, opened/closed by the FastAPI lifespan."""
from psycopg_pool import ConnectionPool

from app.config import settings


def _resolve_database_url() -> str:
    # Runtime secret resolution (RDS): credentials come from Secrets Manager
    # at startup, so no password lives in env files or compose. Falls back to
    # database_url for local development against the docker Postgres.
    if settings.database_secret_id and settings.database_host:
        import json
        from urllib.parse import quote

        import boto3

        secret = json.loads(
            boto3.client("secretsmanager", region_name=settings.aws_region)
            .get_secret_value(SecretId=settings.database_secret_id)["SecretString"]
        )
        return (
            f"postgresql://{secret['username']}:{quote(secret['password'], safe='')}"
            f"@{settings.database_host}:{secret.get('port', 5432)}"
            f"/{secret.get('dbname', 'blueprintai')}"
        )
    return settings.database_url


pool = ConnectionPool(
    _resolve_database_url(),
    min_size=1,
    # Sized for the whole process: API threads + extraction jobs + ingest
    # jobs. 30 x 4 workers is well under RDS t4g.medium's connection budget.
    max_size=30,
    open=False,
    kwargs={"autocommit": True},
)

# Tiny pool RESERVED for job liveness stamps. Heartbeats must never queue
# behind embedding batches on the main pool: a starved heartbeat makes the
# sweeper kill a healthy hours-long extraction (false positive we hit in
# production during the first bulk ingest).
liveness_pool = ConnectionPool(
    _resolve_database_url(),
    min_size=0,
    max_size=2,
    open=False,
    kwargs={"autocommit": True},
)
