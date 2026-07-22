import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import pool
from app.repositories import AuthRepository
from app.services.auth_service import AuthFailed, AuthService
from app.exceptions import (
    AlreadyIngested,
    BlueprintError,
    ExtractionFailed,
    FileNotFound,
    FileTooLarge,
    InvalidFile,
    RenderFailed,
    UnsupportedFileType,
    VisionUnavailable,
)
from app.routers import auth, chats, drawings, files, folders, projects, query, review, stats

_ERROR_STATUS: list[tuple[type[BlueprintError], int]] = [
    (AuthFailed, 401),
    (UnsupportedFileType, 422),
    (InvalidFile, 422),
    (ExtractionFailed, 422),
    (RenderFailed, 422),
    (FileTooLarge, 413),
    (VisionUnavailable, 503),
    (FileNotFound, 404),
    (AlreadyIngested, 409),
]


def _recover_orphaned_ingests() -> None:
    """A restart guarantees no ingest worker survived (single-process app),
    so any file still marked 'ingesting' is an orphaned claim - e.g. the
    server restarted mid-ingest. Release it: drop the partial chunks and
    return the file to 'extracted' so it can simply be confirmed again."""
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT id FROM files WHERE status = 'ingesting'"
        ).fetchall()
        for (file_id,) in rows:
            conn.execute("DELETE FROM chunks WHERE source_file_id = %s", (file_id,))
            conn.execute(
                "UPDATE files SET status = 'extracted' WHERE id = %s", (file_id,)
            )


def _ensure_auth_schema() -> None:
    """Idempotent auth tables for databases created before auth existed
    (init.sql only runs on a fresh local volume, never on RDS)."""
    with pool.connection() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                   id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                   username      text NOT NULL UNIQUE,
                   password_hash text NOT NULL,
                   created_at    timestamptz NOT NULL DEFAULT now()
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS auth_tokens (
                   token_sha256 text PRIMARY KEY,
                   user_id      uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                   expires_at   timestamptz NOT NULL,
                   created_at   timestamptz NOT NULL DEFAULT now()
               )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS auth_tokens_user_idx ON auth_tokens (user_id)"
        )


def _seed_first_user() -> None:
    generated = AuthService(AuthRepository(pool)).ensure_seed_user(
        "admin", settings.initial_admin_password
    )
    if generated:
        # printed exactly once, on the run that creates the account
        logging.getLogger("uvicorn.error").warning(
            "Created initial user 'admin' with password: %s "
            "(sign in and change it; this will not be shown again)",
            generated,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool.open()
    _recover_orphaned_ingests()
    _ensure_auth_schema()
    _seed_first_user()
    yield
    pool.close()


app = FastAPI(title="BlueprintAI API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(BlueprintError)
async def blueprint_error_handler(request: Request, exc: BlueprintError):
    status = next((code for cls, code in _ERROR_STATUS if isinstance(exc, cls)), 400)
    return JSONResponse(status_code=status, content={"detail": str(exc)})


# Everything except signing in (and the health probe) requires a session.
_PUBLIC_PATHS = {"/health", "/auth/login"}


@app.middleware("http")
async def require_auth(request: Request, call_next):
    # CORS preflights never carry credentials; CORSMiddleware (outermost)
    # answers them before this runs, but skip defensively anyway
    if request.method == "OPTIONS" or request.url.path in _PUBLIC_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    token = header.removeprefix("Bearer ").strip()
    user = AuthService(AuthRepository(pool)).authenticate(token) if token else None
    if user is None:
        return JSONResponse(
            status_code=401, content={"detail": "Please sign in to continue."}
        )
    request.state.user = user
    return await call_next(request)


# added after the auth middleware so CORS is outermost (handles preflight
# and stamps headers onto auth 401s)
app.add_middleware(
    CORSMiddleware,
    # any localhost port: vite dev picks a free port when 5173 is taken, and
    # the containerized frontend publishes on 5175 — local-only either way
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(files.router)
app.include_router(review.router)
app.include_router(query.router)
app.include_router(chats.router)
app.include_router(stats.router)
app.include_router(projects.router)
app.include_router(drawings.router)
app.include_router(folders.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
