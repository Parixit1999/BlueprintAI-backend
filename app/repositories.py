"""Data access layer - the only place SQL lives."""
import json
from typing import Any

from psycopg_pool import ConnectionPool


class FileRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create(self, filename: str, file_type: str) -> str:
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO files (filename, file_type, s3_key) VALUES (%s, %s, 'pending') RETURNING id",
                (filename, file_type),
            ).fetchone()
        return str(row[0])

    def mark_extracted(self, file_id: str, s3_key: str, chunks: list[dict]) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE files SET s3_key = %s, status = 'extracted', extraction = %s WHERE id = %s",
                (s3_key, json.dumps(chunks), file_id),
            )

    def mark_ingested(self, file_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE files SET status = 'ingested' WHERE id = %s", (file_id,))

    def delete(self, file_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM files WHERE id = %s", (file_id,))

    def get(self, file_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, filename, file_type, status, extraction, created_at FROM files WHERE id = %s",
                (file_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "file_id": str(row[0]),
            "filename": row[1],
            "file_type": row[2],
            "status": row[3],
            "extraction": row[4] or [],
            "created_at": row[5].isoformat(),
        }

    def list_all(self) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, filename, file_type, status, created_at FROM files ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "file_id": str(r[0]),
                "filename": r[1],
                "file_type": r[2],
                "status": r[3],
                "created_at": r[4].isoformat(),
            }
            for r in rows
        ]


class ChunkRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def insert(
        self,
        source_file_id: str,
        region_type: str,
        chunk_text: str,
        bbox: list[float] | None,
        confidence: str,
        verification_status: str,
        original_value: str | None,
        corrected_value: str | None,
        embedding: list[float],
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO chunks (source_file_id, region_type, chunk_text, bbox,
                       confidence, verification_status, original_value, corrected_value, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    source_file_id,
                    region_type,
                    chunk_text,
                    bbox,
                    confidence,
                    verification_status,
                    original_value,
                    corrected_value,
                    json.dumps(embedding),
                ),
            )

    def search(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        vector = json.dumps(embedding)
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT source_file_id, region_type, chunk_text, bbox, image_uri,
                          1 - (embedding <=> %s::vector) AS score
                   FROM chunks
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (vector, vector, top_k),
            ).fetchall()
        return [
            {
                "source_file_id": str(r[0]),
                "region_type": r[1],
                "chunk_text": r[2],
                "bbox": r[3],
                "image_uri": r[4],
                "score": round(float(r[5]), 4),
            }
            for r in rows
        ]
