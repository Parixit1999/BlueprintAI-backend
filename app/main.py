from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.db import pool
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
from app.routers import chats, drawings, files, folders, projects, query, review, stats

_ERROR_STATUS: list[tuple[type[BlueprintError], int]] = [
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool.open()
    _recover_orphaned_ingests()
    yield
    pool.close()


app = FastAPI(title="BlueprintAI API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(BlueprintError)
async def blueprint_error_handler(request: Request, exc: BlueprintError):
    status = next((code for cls, code in _ERROR_STATUS if isinstance(exc, cls)), 400)
    return JSONResponse(status_code=status, content={"detail": str(exc)})


app.add_middleware(
    CORSMiddleware,
    # any localhost port: vite dev picks a free port when 5173 is taken, and
    # the containerized frontend publishes on 5175 — local-only either way
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
