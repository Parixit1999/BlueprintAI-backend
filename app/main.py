import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import liveness_pool, pool
from app.services import heartbeat, jobs
from app.repositories import AuthRepository
from app.services.auth_service import AuthFailed, AuthService
from app.exceptions import (
    AlreadyIngested,
    BlueprintError,
    ExtractionFailed,
    FileNotFound,
    FileTooLarge,
    Forbidden,
    InvalidFile,
    RenderFailed,
    UnsupportedFileType,
    VisionUnavailable,
)
from app import authz
from app.routers import auth, chats, drawings, files, folders, projects, query, registry, review, stats

_ERROR_STATUS: list[tuple[type[BlueprintError], int]] = [
    (AuthFailed, 401),
    (Forbidden, 403),
    (UnsupportedFileType, 422),
    (InvalidFile, 422),
    (ExtractionFailed, 422),
    (RenderFailed, 422),
    (FileTooLarge, 413),
    (VisionUnavailable, 503),
    (FileNotFound, 404),
    (AlreadyIngested, 409),
]


# A busy row is reclaimable once its liveness signal has been silent this
# long. Workers stamp last_heartbeat_at every 45s (app.services.heartbeat),
# so ~4 missed beats = dead. Rows WITHOUT a heartbeat (claimed by an
# instance running pre-heartbeat code - mixed fleet during a rolling
# deploy, or the sub-second window before the first stamp) keep the old
# conservative 2-hour age rule so a new instance never reclaims work that
# is alive on an old one.
_STALE_SQL = (
    f"(last_heartbeat_at IS NOT NULL AND last_heartbeat_at < now() - interval '{heartbeat.STALE_AFTER}') "
    "OR (last_heartbeat_at IS NULL AND COALESCE(processing_started_at, created_at) < now() - interval '2 hours')"
)


def _reclaim_stale_work() -> None:
    """Free busy rows whose worker has stopped heartbeating.

    Multiple instances share the database (2+ ECS tasks with rolling
    deploys and auto scale-out, plus the local dev stack), so nothing may
    assume an in-flight row is orphaned just because it exists: the row may
    be actively processing on ANOTHER instance right now - a scale-out is
    even TRIGGERED by heavy extraction. Liveness, not age, is the test:
    a healthy 2-hour extraction heartbeats throughout and is never touched;
    a dead worker's row frees within ~3 minutes. Runs at startup and every
    minute from the sweeper thread; idempotent and multi-instance safe."""
    with pool.connection() as conn:
        rows = conn.execute(
            f"SELECT id FROM files WHERE status = 'ingesting' AND ({_STALE_SQL})"
        ).fetchall()
        for (file_id,) in rows:
            conn.execute("DELETE FROM chunks WHERE source_file_id = %s", (file_id,))
            conn.execute(
                "UPDATE files SET status = 'extracted' WHERE id = %s", (file_id,)
            )
        conn.execute(
            "UPDATE files SET status = 'failed', "
            "error = 'The processing worker stopped responding - use Retry.' "
            f"WHERE status = 'uploaded' AND ({_STALE_SQL})"
        )


_sweeper_stop = threading.Event()


def _keepalive_queued_jobs() -> None:
    """Stamp heartbeats for rows whose job is still WAITING in this
    instance's executor queues: a queued job is alive, not orphaned, and
    must not be reclaimed. If this instance dies the stamps stop and the
    normal reclaim takes over."""
    ids = jobs.queued_ids()
    if not ids:
        return
    with pool.connection() as conn:
        conn.execute(
            "UPDATE files SET last_heartbeat_at = now() WHERE id = ANY(%s)", (ids,)
        )


def _sweeper_loop() -> None:
    while not _sweeper_stop.wait(60):
        try:
            _keepalive_queued_jobs()
            _reclaim_stale_work()
        except Exception:
            logging.getLogger(__name__).warning("stale-work sweep failed", exc_info=True)


def _ensure_files_schema() -> None:
    """Idempotent columns/indexes for databases that predate them (RDS never
    re-runs init.sql)."""
    with pool.connection() as conn:
        conn.execute(
            "ALTER TABLE files ADD COLUMN IF NOT EXISTS processing_started_at timestamptz DEFAULT now()"
        )
        conn.execute(
            "ALTER TABLE files ADD COLUMN IF NOT EXISTS last_heartbeat_at timestamptz"
        )
        # soft delete for registry rows: the book hides them, the Deleted page
        # lists them, and a restore puts the row back with its files intact
        conn.execute(
            "ALTER TABLE drawings ADD COLUMN IF NOT EXISTS deleted_at timestamptz"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS drawings_deleted_idx "
            "ON drawings (deleted_at) WHERE deleted_at IS NOT NULL"
        )
        # the document's OWN sheet count, so the viewer and the book stop
        # inferring it from however far extraction happened to get
        conn.execute("ALTER TABLE files ADD COLUMN IF NOT EXISTS page_count int")
        # people, not just logins: the workspace has real teammates in it now
        conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name text")
        conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email text")
        # HNSW vector indexes: retrieval was a sequential scan over every
        # embedding - fine at demo scale, quadratic pain later. First boot
        # after this ships pays a one-time build (seconds at current volume).
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS registry_chunks_embedding_hnsw "
            "ON registry_chunks USING hnsw (embedding vector_cosine_ops)"
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
        # human veto over embedding-similarity duplicate flags: pairs the
        # user has marked "not a duplicate" (stored in both directions)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS dismissed_duplicates (
                   file_id       uuid NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                   other_file_id uuid NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                   created_at    timestamptz NOT NULL DEFAULT now(),
                   PRIMARY KEY (file_id, other_file_id)
               )"""
        )
        # --- role-based access ---
        # a role names a set of PAGES plus Number Book sheet access (all
        # sheets, or the projects listed in role_projects); users carry one
        # role, and admins bypass roles entirely
        conn.execute(
            """CREATE TABLE IF NOT EXISTS roles (
                   id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                   name       text NOT NULL UNIQUE,
                   pages      text[] NOT NULL DEFAULT '{}',
                   all_sheets boolean NOT NULL DEFAULT false,
                   created_at timestamptz NOT NULL DEFAULT now()
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS role_projects (
                   role_id    uuid NOT NULL REFERENCES roles(id)    ON DELETE CASCADE,
                   project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                   PRIMARY KEY (role_id, project_id)
               )"""
        )
        # one switch per role: editors change things, viewers only look
        conn.execute(
            "ALTER TABLE roles ADD COLUMN IF NOT EXISTS can_edit boolean NOT NULL DEFAULT true"
        )
        conn.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS role_id uuid REFERENCES roles(id) ON DELETE SET NULL"
        )
        conn.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false"
        )
        # one-time promotion of the seeded account. Guarded: these patches
        # run on every boot, and re-promoting a deliberately demoted 'admin'
        # would silently undo an administrative decision.
        conn.execute(
            "UPDATE users SET is_admin = true "
            "WHERE username = 'admin' "
            "AND NOT EXISTS (SELECT 1 FROM users WHERE is_admin)"
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
    liveness_pool.open()
    _ensure_files_schema()
    _reclaim_stale_work()
    _ensure_auth_schema()
    _seed_first_user()
    sweeper = threading.Thread(target=_sweeper_loop, daemon=True)
    sweeper.start()
    yield
    _sweeper_stop.set()
    liveness_pool.close()
    pool.close()


app = FastAPI(title="BlueprintAI API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(BlueprintError)
async def blueprint_error_handler(request: Request, exc: BlueprintError):
    status = next((code for cls, code in _ERROR_STATUS if isinstance(exc, cls)), 400)
    return JSONResponse(status_code=status, content={"detail": str(exc)})


# Per-request timing: one log line per API request with the wall-clock the
# server spent on it, and a Server-Timing header so browser devtools show the
# split between server work and network. This is THE tool for "why is the
# app slow" - grep the access-timing lines and sort.
_TIMING_LOG = logging.getLogger("access.timing")


@app.middleware("http")
async def time_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["Server-Timing"] = f"app;dur={ms:.0f}"
    path = request.url.path
    if not path.endswith("/health"):
        level = logging.WARNING if ms >= 1000 else logging.INFO
        _TIMING_LOG.log(level, "%s %s -> %s in %.0fms",
                        request.method, path, response.status_code, ms)
    return response


# Everything except signing in (and the health probe) requires a session.
_PUBLIC_PATHS = {"/health", "/api/health", "/api/auth/login"}


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
    # Page gate: the role must include the feature this URL belongs to.
    # 403, never 401 - the frontend treats 401 as an expired session.
    denial = authz.check_page(user, request.method, request.url.path)
    if denial:
        return JSONResponse(status_code=403, content={"detail": denial})
    return await call_next(request)


# Region-heavy documents produce multi-MB JSON payloads (a 5,000-region
# extraction is ~4 MB raw, ~10x smaller gzipped) - compress everything
# sizeable. Inner to auth/CORS; SSE streaming responses are not buffered
# by GZipMiddleware, so chat streaming keeps flowing chunk by chunk.
app.add_middleware(GZipMiddleware, minimum_size=1024)

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

# Every API route lives under /api so page URLs (/files, /projects in the
# SPA) can never collide with API paths of the same name behind one domain -
# CloudFront routes /api/* to the backend and everything else to the site.
_API_PREFIX = "/api"
for _r in (auth, files, review, query, chats, stats, projects, drawings, folders, registry):
    app.include_router(_r.router, prefix=_API_PREFIX)


@app.get("/api/health")
async def api_health():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}
