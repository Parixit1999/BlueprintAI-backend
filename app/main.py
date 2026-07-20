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
from app.routers import chats, files, query, review, stats

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool.open()
    yield
    pool.close()


app = FastAPI(title="BlueprintAI API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(BlueprintError)
async def blueprint_error_handler(request: Request, exc: BlueprintError):
    status = next((code for cls, code in _ERROR_STATUS if isinstance(exc, cls)), 400)
    return JSONResponse(status_code=status, content={"detail": str(exc)})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(files.router)
app.include_router(review.router)
app.include_router(query.router)
app.include_router(chats.router)
app.include_router(stats.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
