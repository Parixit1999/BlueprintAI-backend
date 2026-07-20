"""Data access layer - the only place SQL lives."""
import json
from typing import Any

from psycopg_pool import ConnectionPool


class FileRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create(self, filename: str, file_type: str, content_sha256: str | None = None) -> str:
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO files (filename, file_type, s3_key, content_sha256) "
                "VALUES (%s, %s, 'pending', %s) RETURNING id",
                (filename, file_type, content_sha256),
            ).fetchone()
        return str(row[0])

    def mark_extracted(
        self, file_id: str, s3_key: str, chunks: list[dict], embedding: list[float] | None = None
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE files SET s3_key = %s, status = 'extracted', extraction = %s, "
                "embedding = %s WHERE id = %s",
                (s3_key, json.dumps(chunks), json.dumps(embedding) if embedding else None, file_id),
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
                "SELECT id, filename, file_type, status, extraction, created_at, s3_key, render, content_sha256 "
                "FROM files WHERE id = %s",
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
            "s3_key": row[6],
            "render": row[7],
            "content_sha256": row[8],
        }

    def list_render_keys(self, file_id: str) -> list[str]:
        """Every object-storage key produced for a file: the original plus any
        per-page renders. Used to clean up storage on delete."""
        record = self.get(file_id)
        if record is None:
            return []
        keys = [record["s3_key"]] if record["s3_key"] and record["s3_key"] != "pending" else []
        render = record["render"] or {}
        for entry in render.get("pages", {}).values():
            if entry.get("s3_key"):
                keys.append(entry["s3_key"])
        if "s3_key" in render:  # legacy single-page format
            keys.append(render["s3_key"])
        return keys

    def set_render(self, file_id: str, render: dict[str, Any]) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE files SET render = %s WHERE id = %s",
                (json.dumps(render), file_id),
            )

    def list_all(self, similarity_threshold: float = 0.90) -> list[dict[str, Any]]:
        """List documents, each tagged with any other documents whose content is
        semantically near-identical (cosine similarity >= threshold on the
        document embedding). Catches the same drawing across file formats, not
        just byte-identical files."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT f.id, f.filename, f.file_type, f.status, f.created_at,
                          count(c.id),
                          (
                            SELECT json_agg(json_build_object(
                                     'file_id', o.id, 'filename', o.filename,
                                     'similarity', round((1 - (f.embedding <=> o.embedding))::numeric, 4))
                                   ORDER BY f.embedding <=> o.embedding)
                            FROM files o
                            WHERE o.id <> f.id AND o.embedding IS NOT NULL
                              AND (1 - (f.embedding <=> o.embedding)) >= %s
                          ) AS similar
                   FROM files f LEFT JOIN chunks c ON c.source_file_id = f.id
                   GROUP BY f.id ORDER BY f.created_at DESC""",
                (similarity_threshold,),
            ).fetchall()
        return [
            {
                "file_id": str(r[0]),
                "filename": r[1],
                "file_type": r[2],
                "status": r[3],
                "created_at": r[4].isoformat(),
                "chunk_count": r[5],
                "similar_documents": r[6] or [],
                "is_duplicate": bool(r[6]),
            }
            for r in rows
        ]


class ChatRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create_session(self, user_id: str, title: str) -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO chat_sessions (user_id, title) VALUES (%s, %s) RETURNING id, title, created_at",
                (user_id, title),
            ).fetchone()
        return {"session_id": str(row[0]), "title": row[1], "created_at": row[2].isoformat()}

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT s.id, s.title, s.created_at, s.updated_at, count(m.id)
                   FROM chat_sessions s LEFT JOIN chat_messages m ON m.session_id = s.id
                   WHERE s.user_id = %s
                   GROUP BY s.id ORDER BY s.updated_at DESC""",
                (user_id,),
            ).fetchall()
        return [
            {
                "session_id": str(r[0]),
                "title": r[1],
                "created_at": r[2].isoformat(),
                "updated_at": r[3].isoformat(),
                "message_count": r[4],
            }
            for r in rows
        ]

    def get_session(self, session_id: str, user_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, title FROM chat_sessions WHERE id = %s AND user_id = %s",
                (session_id, user_id),
            ).fetchone()
        return None if row is None else {"session_id": str(row[0]), "title": row[1]}

    def set_title(self, session_id: str, title: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title = %s, updated_at = now() WHERE id = %s",
                (title, session_id),
            )

    def touch(self, session_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE chat_sessions SET updated_at = now() WHERE id = %s", (session_id,))

    def add_message(
        self, session_id: str, role: str, content: str, evidence: list[dict] | None = None
    ) -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                """INSERT INTO chat_messages (session_id, role, content, evidence)
                   VALUES (%s, %s, %s, %s) RETURNING id, created_at""",
                (session_id, role, content, json.dumps(evidence) if evidence is not None else None),
            ).fetchone()
        return {
            "message_id": str(row[0]),
            "role": role,
            "content": content,
            "evidence": evidence,
            "created_at": row[1].isoformat(),
        }

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT id, role, content, evidence, created_at
                   FROM chat_messages WHERE session_id = %s ORDER BY created_at""",
                (session_id,),
            ).fetchall()
        return [
            {
                "message_id": str(r[0]),
                "role": r[1],
                "content": r[2],
                "evidence": r[3],
                "created_at": r[4].isoformat(),
            }
            for r in rows
        ]


class StatsRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def snapshot(self) -> dict[str, Any]:
        with self._pool.connection() as conn:
            files_by_status = dict(
                conn.execute("SELECT status, count(*) FROM files GROUP BY status").fetchall()
            )
            files_by_type = dict(
                conn.execute("SELECT file_type, count(*) FROM files GROUP BY file_type").fetchall()
            )
            chunks_total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            chunks_by_confidence = dict(
                conn.execute("SELECT confidence, count(*) FROM chunks GROUP BY confidence").fetchall()
            )
            corrected = conn.execute(
                "SELECT count(*) FROM chunks WHERE verification_status = 'corrected'"
            ).fetchone()[0]
            sessions = conn.execute("SELECT count(*) FROM chat_sessions").fetchone()[0]
            questions = conn.execute(
                "SELECT count(*) FROM chat_messages WHERE role = 'user'"
            ).fetchone()[0]
        return {
            "documents_total": sum(files_by_status.values()),
            "documents_by_status": files_by_status,
            "documents_by_type": files_by_type,
            "chunks_total": chunks_total,
            "chunks_by_confidence": chunks_by_confidence,
            "chunks_corrected": corrected,
            "chat_sessions": sessions,
            "questions_asked": questions,
        }


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
        page: int = 1,
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO chunks (source_file_id, page, region_type, chunk_text, bbox,
                       confidence, verification_status, original_value, corrected_value, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    source_file_id,
                    page,
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
                """SELECT source_file_id, region_type, chunk_text, bbox, image_uri, page,
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
                "page": r[5],
                "score": round(float(r[6]), 4),
            }
            for r in rows
        ]
