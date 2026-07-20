"""Top-k vector retrieval from pgvector. Day 4-5."""
from app.schemas import Chunk


def retrieve(question: str, top_k: int = 5) -> list[Chunk]:
    raise NotImplementedError
