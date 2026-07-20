from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import files, query, review

app = FastAPI(title="BlueprintAI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(files.router)
app.include_router(review.router)
app.include_router(query.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
