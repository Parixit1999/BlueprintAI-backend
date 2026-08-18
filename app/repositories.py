"""Data access layer - the only place SQL lives."""
import json
import time
from datetime import timedelta
from typing import Any

from psycopg_pool import ConnectionPool


class FileRepository:
    # per-process cache for the archive-wide duplicate count (see list_paged)
    _dup_count_cache: dict[str, Any] = {"count": 0, "at": 0.0, "thr": None}

    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create(
        self, filename: str, file_type: str, content_sha256: str | None = None,
        folder_id: str | None = None,
    ) -> str:
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO files (filename, file_type, s3_key, content_sha256, folder_id) "
                "VALUES (%s, %s, 'pending', %s, %s) RETURNING id",
                (filename, file_type, content_sha256, folder_id),
            ).fetchone()
        return str(row[0])

    def rename(self, file_id: str, filename: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE files SET filename = %s WHERE id = %s", (filename, file_id))

    def move_to_folder(self, file_id: str, folder_id: str | None) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE files SET folder_id = %s WHERE id = %s", (folder_id, file_id))

    def list_in_folder(self, folder_id: str | None) -> list[dict[str, Any]]:
        """Files directly inside one folder (root = null), for the browser."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT f.id, f.filename, f.file_type, f.status, f.created_at, f.error,
                          f.drawing_id, d.dwg_number
                   FROM files f LEFT JOIN drawings d ON f.drawing_id = d.id
                   WHERE f.folder_id IS NOT DISTINCT FROM %s
                   ORDER BY f.filename""",
                (folder_id,),
            ).fetchall()
        return [
            {
                "file_id": str(r[0]), "filename": r[1], "file_type": r[2], "status": r[3],
                "created_at": r[4].isoformat(), "error": r[5],
                "drawing_id": str(r[6]) if r[6] else None, "dwg_number": r[7],
            }
            for r in rows
        ]

    def mark_extracted(
        self, file_id: str, s3_key: str, chunks: list[dict], embedding: list[float] | None = None,
        is_drawing: bool | None = None,
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE files SET s3_key = %s, status = 'extracted', extraction = %s, "
                "embedding = %s, is_drawing = %s, error = NULL WHERE id = %s",
                (s3_key, json.dumps(chunks), json.dumps(embedding) if embedding else None,
                 is_drawing, file_id),
            )

    def set_s3_key(self, file_id: str, s3_key: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE files SET s3_key = %s WHERE id = %s", (s3_key, file_id)
            )

    def heartbeat(self, file_id: str) -> None:
        """Liveness stamp for in-flight extraction/ingestion (see
        app.services.heartbeat)."""
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE files SET last_heartbeat_at = now() WHERE id = %s", (file_id,)
            )

    def mark_uploaded(self, file_id: str) -> None:
        """Back to 'processing' state for background re-extraction/retry."""
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE files SET status = 'uploaded', error = NULL, "
                "processing_started_at = now() WHERE id = %s",
                (file_id,),
            )

    def mark_failed(self, file_id: str, s3_key: str, error: str) -> None:
        """Keep the row on extraction failure (instead of deleting) so the UI
        can show what went wrong and offer a retry without re-uploading."""
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE files SET s3_key = %s, status = 'failed', error = %s WHERE id = %s",
                (s3_key, error, file_id),
            )

    def mark_ingested(self, file_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE files SET status = 'ingested' WHERE id = %s", (file_id,))

    def claim_for_ingest(self, file_id: str) -> bool:
        """Atomically move extracted -> ingesting. Ingestion embeds every
        region (minutes for dense sheets), so this claim is what prevents a
        second confirm - double-click, back button, second tab - from
        double-inserting every chunk while the first run is still going."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "UPDATE files SET status = 'ingesting', processing_started_at = now(), "
                "last_heartbeat_at = now() "
                "WHERE id = %s AND status = 'extracted' RETURNING id",
                (file_id,),
            ).fetchone()
        return row is not None

    def release_ingest_claim(self, file_id: str) -> None:
        """Failed ingest: drop any partially inserted chunks and return the
        file to 'extracted' so the user can review and confirm again."""
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE source_file_id = %s", (file_id,))
            conn.execute(
                "UPDATE files SET status = 'extracted' WHERE id = %s", (file_id,)
            )

    def delete(self, file_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM files WHERE id = %s", (file_id,))

    def get(self, file_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT f.id, f.filename, f.file_type, f.status, f.extraction, f.created_at, "
                "f.s3_key, f.render, f.content_sha256, f.error, f.drawing_id, f.is_drawing, "
                "d.dwg_number, p.name, f.auto_assigned, f.page_count "
                "FROM files f LEFT JOIN drawings d ON f.drawing_id = d.id "
                "LEFT JOIN projects p ON d.project_id = p.id WHERE f.id = %s",
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
            "error": row[9],
            "drawing_id": str(row[10]) if row[10] else None,
            "is_drawing": row[11],
            "dwg_number": row[12],
            "project_name": row[13],
            "auto_assigned": row[14],
            # the document's real sheet count; None for older rows and CAD
            "page_count": row[15],
        }

    def project_of(self, file_id: str) -> tuple[str | None, str | None] | None:
        """(project_id, drawing_id) for one file - the ownership fact the
        per-document role check needs. None when the file doesn't exist."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT d.project_id, f.drawing_id FROM files f "
                "LEFT JOIN drawings d ON d.id = f.drawing_id WHERE f.id = %s",
                (file_id,),
            ).fetchone()
        if row is None:
            return None
        return (str(row[0]) if row[0] else None, str(row[1]) if row[1] else None)

    def set_page_count(self, file_id: str, page_count: int) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE files SET page_count = %s WHERE id = %s", (page_count, file_id)
            )

    def find_by_sha(self, content_sha256: str) -> dict[str, Any] | None:
        """Newest document with these exact bytes; a live row wins over a
        failed one so re-uploads resume rather than duplicate."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, filename, status FROM files "
                "WHERE content_sha256 = %s "
                "ORDER BY (status = 'failed') ASC, created_at DESC LIMIT 1",
                (content_sha256,),
            ).fetchone()
        if row is None:
            return None
        return {"file_id": str(row[0]), "filename": row[1], "status": row[2]}

    def reset_for_reprocess(self, file_id: str) -> None:
        """Put a failed row back to square one for a fresh extraction run."""
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE files SET status = 'uploaded', error = NULL, "
                "extraction = NULL, processing_started_at = now(), "
                "last_heartbeat_at = NULL WHERE id = %s",
                (file_id,),
            )

    def get_statuses(self, ids: list[str]) -> list[dict[str, Any]]:
        """Light polling payload for many files at once: status and metadata
        WITHOUT the extraction JSON (which can be megabytes per document).
        One query regardless of batch size - built for the upload page's
        single poller."""
        if not ids:
            return []
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT f.id, f.status, f.error, f.is_drawing, d.dwg_number,
                          p.name, f.auto_assigned,
                          COALESCE(jsonb_array_length(f.extraction), 0)
                   FROM files f
                   LEFT JOIN drawings d ON f.drawing_id = d.id
                   LEFT JOIN projects p ON d.project_id = p.id
                   WHERE f.id = ANY(%s)""",
                (ids,),
            ).fetchall()
        return [
            {
                "file_id": str(r[0]),
                "status": r[1],
                "error": r[2],
                "is_drawing": r[3],
                "dwg_number": r[4],
                "project_name": r[5],
                "auto_assigned": r[6],
                "region_count": r[7],
            }
            for r in rows
        ]

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

    def similarity_to_drawing(self, file_id: str, drawing_id: str) -> float | None:
        """Best cosine similarity between this file's document embedding and
        any file already attached to the drawing. AI evidence for 'same
        drawing, different iteration'. None when either side lacks embeddings."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT max(1 - (a.embedding <=> b.embedding)) "
                "FROM files a, files b "
                "WHERE a.id = %s AND b.drawing_id = %s AND b.id <> a.id "
                "  AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL",
                (file_id, drawing_id),
            ).fetchone()
        return round(float(row[0]), 4) if row and row[0] is not None else None

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
                          f.error, f.drawing_id, d.dwg_number, f.auto_assigned, count(c.id),
                          f.is_drawing, p.name AS project_name,
                          (
                            SELECT json_agg(json_build_object(
                                     'file_id', o.id, 'filename', o.filename,
                                     'similarity', round((1 - (f.embedding <=> o.embedding))::numeric, 4))
                                   ORDER BY f.embedding <=> o.embedding)
                            FROM files o
                            WHERE o.id <> f.id AND o.embedding IS NOT NULL
                              AND (1 - (f.embedding <=> o.embedding)) >= %s
                              AND NOT EXISTS (
                                SELECT 1 FROM dismissed_duplicates dd
                                WHERE dd.file_id = f.id AND dd.other_file_id = o.id
                              )
                          ) AS similar
                   FROM files f
                        LEFT JOIN chunks c ON c.source_file_id = f.id
                        LEFT JOIN drawings d ON f.drawing_id = d.id
                        LEFT JOIN projects p ON d.project_id = p.id
                   GROUP BY f.id, d.dwg_number, p.name ORDER BY f.created_at DESC""",
                (similarity_threshold,),
            ).fetchall()
        return [
            {
                "file_id": str(r[0]),
                "filename": r[1],
                "file_type": r[2],
                "status": r[3],
                "created_at": r[4].isoformat(),
                "error": r[5],
                "drawing_id": str(r[6]) if r[6] else None,
                "dwg_number": r[7],
                "auto_assigned": r[8],
                "chunk_count": r[9],
                "is_drawing": r[10],
                "project_name": r[11],
                "similar_documents": r[12] or [],
                "is_duplicate": bool(r[12]),
            }
            for r in rows
        ]


    _LIST_SORTS = {
        "name": "f.filename",
        "assignment": "d.dwg_number",
        "type": "f.file_type",
        "status": "f.status",
        "uploaded": "f.created_at",
        # rank each row by how many documents share its type (within the
        # current filters), so the most common formats surface first
        "type_count": "count(*) OVER (PARTITION BY f.file_type)",
    }

    def list_paged(
        self,
        similarity_threshold: float = 0.90,
        q: str | None = None,
        file_type: str | None = None,
        status: str | None = None,
        assigned: str | None = None,
        drawing: str | None = None,
        dup_only: bool = False,
        sort: str = "uploaded",
        direction: str = "desc",
        page: int = 1,
        page_size: int = 10,
        allowed_project_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Server-side paged listing: filters, sorting, and LIMIT/OFFSET run
        in SQL, so listing cost is bound by the page size, not the archive -
        including the per-row duplicate check, which only runs for the page.
        """
        order_col = self._LIST_SORTS.get(sort, "f.created_at")
        order_dir = "ASC" if direction == "asc" else "DESC"
        # stable secondary key so pages never overlap
        if sort == "type_count":
            # equal counts tie-break alphabetically; within a type, newest first
            order_sql = f"{order_col} {order_dir}, f.file_type, f.created_at DESC, f.id"
        else:
            order_sql = f"{order_col} {order_dir} NULLS LAST, f.id"

        where = ["TRUE"]
        params: list[Any] = []
        if q:
            where.append("f.filename ILIKE %s")
            params.append(f"%{q}%")
        if file_type:
            where.append("f.file_type = %s")
            params.append(file_type)
        if status == "processing":
            where.append("f.status IN ('uploaded', 'ingesting')")
        elif status:
            where.append("f.status = %s")
            params.append(status)
        if assigned == "yes":
            where.append("f.drawing_id IS NOT NULL")
        elif assigned == "no":
            where.append("f.drawing_id IS NULL")
        # the vision verdict: 'no' surfaces content flagged as not an
        # engineering drawing (photos, forms, ...) for quick review/deletion
        if drawing == "yes":
            where.append("f.is_drawing IS TRUE")
        elif drawing == "no":
            where.append("f.is_drawing IS FALSE")
        if allowed_project_ids is not None:
            # role-scoped: documents on the caller's sheets, plus unassigned
            # uploads (which belong to no sheet - hiding them would make a
            # user's own uploads vanish until someone files them)
            where.append("(d.project_id = ANY(%s::uuid[]) OR f.drawing_id IS NULL)")
            params.append(allowed_project_ids)
        dup_exists = (
            """EXISTS (
                 SELECT 1 FROM files o
                 WHERE o.id <> f.id AND o.embedding IS NOT NULL
                   AND (1 - (f.embedding <=> o.embedding)) >= %s
                   AND NOT EXISTS (
                     SELECT 1 FROM dismissed_duplicates dd
                     WHERE dd.file_id = f.id AND dd.other_file_id = o.id
                   )
               )"""
        )
        if dup_only:
            where.append(dup_exists)
            params.append(similarity_threshold)
        where_sql = " AND ".join(where)

        page = max(1, page)
        page_size = max(1, min(page_size, 100))
        offset = (page - 1) * page_size

        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""SELECT f.id, f.filename, f.file_type, f.status, f.created_at,
                          f.error, f.drawing_id, d.dwg_number, f.auto_assigned,
                          (SELECT count(*) FROM chunks c WHERE c.source_file_id = f.id),
                          f.is_drawing, p.name AS project_name,
                          (
                            SELECT json_agg(json_build_object(
                                     'file_id', o.id, 'filename', o.filename,
                                     'similarity', round((1 - (f.embedding <=> o.embedding))::numeric, 4))
                                   ORDER BY f.embedding <=> o.embedding)
                            FROM files o
                            WHERE o.id <> f.id AND o.embedding IS NOT NULL
                              AND (1 - (f.embedding <=> o.embedding)) >= %s
                              AND NOT EXISTS (
                                SELECT 1 FROM dismissed_duplicates dd
                                WHERE dd.file_id = f.id AND dd.other_file_id = o.id
                              )
                          ) AS similar,
                          count(*) OVER() AS total
                   FROM files f
                        LEFT JOIN drawings d ON f.drawing_id = d.id
                        LEFT JOIN projects p ON d.project_id = p.id
                   WHERE {where_sql}
                   ORDER BY {order_sql}
                   LIMIT %s OFFSET %s""",
                (similarity_threshold, *params, page_size, offset),
            ).fetchall()
            # cheap whole-archive facts for the page chrome
            grand_total, pending_review, failed_count = conn.execute(
                "SELECT count(*), count(*) FILTER (WHERE status = 'extracted'), "
                "count(*) FILTER (WHERE status = 'failed') FROM files"
            ).fetchone()
            types = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT file_type FROM files ORDER BY file_type"
                ).fetchall()
            ]
            # The duplicate badge compares every embedding against every other
            # (O(n^2) vector math) - ~1.5s at a few hundred documents, and the
            # Documents page polls this endpoint. The count moves slowly, so
            # each process reuses it for 60s instead of recomputing per poll.
            now = time.monotonic()
            cache = FileRepository._dup_count_cache
            if cache["at"] and now - cache["at"] < 60 and cache["thr"] == similarity_threshold:
                duplicate_count = cache["count"]
            else:
                duplicate_count = conn.execute(
                    f"SELECT count(*) FROM files f WHERE {dup_exists}",
                    (similarity_threshold,),
                ).fetchone()[0]
                FileRepository._dup_count_cache = {
                    "count": duplicate_count, "at": now, "thr": similarity_threshold,
                }

        total = rows[0][13] if rows else 0
        items = [
            {
                "file_id": str(r[0]),
                "filename": r[1],
                "file_type": r[2],
                "status": r[3],
                "created_at": r[4].isoformat(),
                "error": r[5],
                "drawing_id": str(r[6]) if r[6] else None,
                "dwg_number": r[7],
                "auto_assigned": r[8],
                "chunk_count": r[9],
                "is_drawing": r[10],
                "project_name": r[11],
                "similar_documents": r[12] or [],
                "is_duplicate": bool(r[12]),
            }
            for r in rows
        ]
        return {
            "items": items,
            "total": total,
            "grand_total": grand_total,
            "pending_review_count": pending_review,
            "failed_count": failed_count,
            "duplicate_count": duplicate_count,
            "types": types,
            "page": page,
            "page_size": page_size,
        }


_PROJECT_COLS = "id, number, name, description, source, created_at"


def _project_dict(r) -> dict[str, Any]:
    return {
        "project_id": str(r[0]),
        "number": r[1],
        "name": r[2],
        "description": r[3],
        "source": r[4],
        "created_at": r[5].isoformat(),
    }


_DRAWING_COLS = (
    "id, project_id, set_id, dwg_number, dwg_number_norm, description, contract_number, "
    "drawing_date, year, sheet_count, version_group_id, version_note, source, created_at"
)


def _drawing_dict(r) -> dict[str, Any]:
    return {
        "drawing_id": str(r[0]),
        "project_id": str(r[1]) if r[1] else None,
        "set_id": str(r[2]) if r[2] else None,
        "dwg_number": r[3],
        "dwg_number_norm": r[4],
        "description": r[5],
        "contract_number": r[6],
        "drawing_date": r[7],
        "year": r[8],
        "sheet_count": r[9],
        "version_group_id": str(r[10]) if r[10] else None,
        "version_note": r[11],
        "source": r[12],
        "created_at": r[13].isoformat(),
    }


class FolderRepository:
    """File-manager folder tree."""

    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create(self, name: str, parent_id: str | None) -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO folders (name, parent_id) VALUES (%s, %s) "
                "RETURNING id, name, parent_id, created_at",
                (name, parent_id),
            ).fetchone()
        return {"folder_id": str(row[0]), "name": row[1],
                "parent_id": str(row[2]) if row[2] else None,
                "created_at": row[3].isoformat()}

    def get(self, folder_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, name, parent_id FROM folders WHERE id = %s", (folder_id,)
            ).fetchone()
        if row is None:
            return None
        return {"folder_id": str(row[0]), "name": row[1],
                "parent_id": str(row[2]) if row[2] else None}

    def list_all(self) -> list[dict[str, Any]]:
        """Flat list of every folder (for move dialogs); small scale."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, name, parent_id FROM folders ORDER BY name"
            ).fetchall()
        return [
            {"folder_id": str(r[0]), "name": r[1], "parent_id": str(r[2]) if r[2] else None}
            for r in rows
        ]

    def children(self, parent_id: str | None) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT f.id, f.name,
                          (SELECT count(*) FROM folders c WHERE c.parent_id = f.id),
                          (SELECT count(*) FROM files x WHERE x.folder_id = f.id)
                   FROM folders f
                   WHERE f.parent_id IS NOT DISTINCT FROM %s ORDER BY f.name""",
                (parent_id,),
            ).fetchall()
        return [
            {"folder_id": str(r[0]), "name": r[1], "subfolder_count": r[2], "file_count": r[3]}
            for r in rows
        ]

    def rename(self, folder_id: str, name: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE folders SET name = %s WHERE id = %s", (name, folder_id))

    def move(self, folder_id: str, parent_id: str | None) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE folders SET parent_id = %s WHERE id = %s", (parent_id, folder_id))

    def subtree_ids(self, folder_id: str) -> list[str]:
        """The folder and every descendant (for cycle checks and recursive delete)."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                """WITH RECURSIVE sub AS (
                       SELECT id FROM folders WHERE id = %s
                       UNION ALL
                       SELECT f.id FROM folders f JOIN sub ON f.parent_id = sub.id
                   ) SELECT id FROM sub""",
                (folder_id,),
            ).fetchall()
        return [str(r[0]) for r in rows]

    def file_ids_in(self, folder_ids: list[str]) -> list[str]:
        if not folder_ids:
            return []
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id FROM files WHERE folder_id = ANY(%s::uuid[])", (folder_ids,)
            ).fetchall()
        return [str(r[0]) for r in rows]

    def delete(self, folder_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM folders WHERE id = %s", (folder_id,))


class ProjectRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create(self, name: str, number: str | None, description: str | None,
               source: str = "manual") -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"INSERT INTO projects (name, number, description, source) "
                f"VALUES (%s, %s, %s, %s) RETURNING {_PROJECT_COLS}",
                (name, number, description, source),
            ).fetchone()
        return _project_dict(row)

    def get(self, project_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {_PROJECT_COLS} FROM projects WHERE id = %s", (project_id,)
            ).fetchone()
        return _project_dict(row) if row else None

    def update(self, project_id: str, fields: dict[str, Any]) -> None:
        allowed = {"name", "number", "description"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        clause = ", ".join(f"{k} = %s" for k in sets)
        with self._pool.connection() as conn:
            conn.execute(
                f"UPDATE projects SET {clause} WHERE id = %s", (*sets.values(), project_id)
            )

    def delete(self, project_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM projects WHERE id = %s", (project_id,))

    def list_all(self) -> list[dict[str, Any]]:
        """Projects with drawing/set/file counts for the list page."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT p.id, p.number, p.name, p.description, p.source, p.created_at,
                          (SELECT count(*) FROM drawings d
                            WHERE d.project_id = p.id AND d.deleted_at IS NULL),
                          (SELECT count(*) FROM drawing_sets s WHERE s.project_id = p.id),
                          (SELECT count(*) FROM files f
                             JOIN drawings d ON f.drawing_id = d.id
                            WHERE d.project_id = p.id)
                   FROM projects p ORDER BY p.name"""
            ).fetchall()
        return [
            {**_project_dict(r), "drawing_count": r[6], "set_count": r[7], "file_count": r[8]}
            for r in rows
        ]


class DrawingRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Insert a drawing; version_group_id defaults to the drawing's own id
        so every drawing starts as the sole member of its version group."""
        with self._pool.connection() as conn:
            row = conn.execute(
                f"""INSERT INTO drawings
                       (project_id, set_id, dwg_number, dwg_number_norm, description,
                        contract_number, drawing_date, year, sheet_count, version_note, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING {_DRAWING_COLS}""",
                (
                    fields.get("project_id"),
                    fields.get("set_id"),
                    fields.get("dwg_number"),
                    fields.get("dwg_number_norm"),
                    fields.get("description"),
                    fields.get("contract_number"),
                    fields.get("drawing_date"),
                    fields.get("year"),
                    fields.get("sheet_count"),
                    fields.get("version_note"),
                    fields.get("source", "manual"),
                ),
            ).fetchone()
            conn.execute(
                "UPDATE drawings SET version_group_id = id WHERE id = %s AND version_group_id IS NULL",
                (row[0],),
            )
        drawing = _drawing_dict(row)
        drawing["version_group_id"] = drawing["version_group_id"] or drawing["drawing_id"]
        return drawing

    def get(self, drawing_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {_DRAWING_COLS} FROM drawings WHERE id = %s", (drawing_id,)
            ).fetchone()
        return _drawing_dict(row) if row else None

    def update(self, drawing_id: str, fields: dict[str, Any]) -> None:
        allowed = {
            "project_id", "set_id", "dwg_number", "dwg_number_norm", "description",
            "contract_number", "drawing_date", "year", "sheet_count", "version_note",
        }
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        clause = ", ".join(f"{k} = %s" for k in sets)
        with self._pool.connection() as conn:
            conn.execute(
                f"UPDATE drawings SET {clause} WHERE id = %s", (*sets.values(), drawing_id)
            )

    def delete(self, drawing_id: str) -> None:
        """Soft delete: the row leaves the book but stays recoverable from the
        Deleted page. Attached files keep their drawing_id, so restoring brings
        the row back with its scans still linked."""
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE drawings SET deleted_at = now() WHERE id = %s", (drawing_id,)
            )

    def restore(self, drawing_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE drawings SET deleted_at = NULL WHERE id = %s", (drawing_id,)
            )

    def list_deleted(self, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        """The recycle bin behind the book, newest deletion first."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""SELECT {', '.join('d.' + c for c in _DRAWING_COLS.split(', '))},
                           (SELECT count(*) FROM files f WHERE f.drawing_id = d.id),
                           s.set_number, p.name, d.deleted_at
                    FROM drawings d
                    LEFT JOIN drawing_sets s ON d.set_id = s.id
                    LEFT JOIN projects p ON d.project_id = p.id
                    WHERE d.deleted_at IS NOT NULL
                      AND (%s::uuid[] IS NULL OR d.project_id = ANY(%s::uuid[]))
                    ORDER BY d.deleted_at DESC""",
                (allowed, allowed),
            ).fetchall()
        return [
            {**_drawing_dict(r), "file_count": r[14], "set_number": r[15],
             "project_name": r[16], "deleted_at": r[17].isoformat()}
            for r in rows
        ]

    def files_count(self, drawing_id: str) -> int:
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT count(*) FROM files WHERE drawing_id = %s", (drawing_id,)
            ).fetchone()[0]

    def version_sibling_count(self, drawing_id: str) -> int:
        """Other drawings explicitly linked as versions of this one."""
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT count(*) FROM drawings WHERE version_group_id = "
                "(SELECT version_group_id FROM drawings WHERE id = %s) AND id <> %s",
                (drawing_id, drawing_id),
            ).fetchone()[0]

    def list_for_project(self, project_id: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""SELECT {', '.join('d.' + c for c in _DRAWING_COLS.split(', '))},
                           (SELECT count(*) FROM files f WHERE f.drawing_id = d.id),
                           s.set_number
                    FROM drawings d LEFT JOIN drawing_sets s ON d.set_id = s.id
                    WHERE d.project_id = %s
                    ORDER BY d.dwg_number_norm NULLS LAST, d.created_at""",
                (project_id,),
            ).fetchall()
        return [
            {**_drawing_dict(r), "file_count": r[14], "set_number": r[15]} for r in rows
        ]

    def list_registry(
        self, project_id: str | None, allowed: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Registry rows for the spreadsheet view: all drawings (Main Book)
        or one project's drawings, with set number, project name, and file
        count joined in. The book is thousands of rows, not millions - one
        query per tab is fine and keeps the grid snappy."""
        # soft-deleted rows live in the Deleted page, never in the book
        where = (
            "WHERE d.deleted_at IS NULL AND d.project_id = %s"
            if project_id
            else "WHERE d.deleted_at IS NULL"
        )
        params: tuple = (project_id,) if project_id else ()
        if allowed is not None:
            # role-scoped: only rows owned by the caller's sheets
            where += " AND d.project_id = ANY(%s::uuid[])"
            params += (allowed,)
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""SELECT {', '.join('d.' + c for c in _DRAWING_COLS.split(', '))},
                           (SELECT count(*) FROM files f WHERE f.drawing_id = d.id),
                           s.set_number, p.name,
                           (SELECT f.filename FROM files f WHERE f.drawing_id = d.id
                             ORDER BY f.created_at LIMIT 1)
                    FROM drawings d
                    LEFT JOIN drawing_sets s ON d.set_id = s.id
                    LEFT JOIN projects p ON d.project_id = p.id
                    {where}
                    ORDER BY d.dwg_number_norm NULLS LAST, d.created_at""",
                params,
            ).fetchall()
        return [
            # filename is derived from the attached scan, not stored on the
            # row: the book shows what the drawing actually came from, and it
            # stays right when files are reassigned.
            {**_drawing_dict(r), "file_count": r[14], "set_number": r[15],
             "project_name": r[16], "filename": r[17]}
            for r in rows
        ]

    def count_all(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT count(*) FROM drawings WHERE deleted_at IS NULL"
            ).fetchone()[0]

    def count_deleted(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT count(*) FROM drawings WHERE deleted_at IS NOT NULL"
            ).fetchone()[0]

    def set_project(self, set_id: str) -> str | None:
        """The owning project of a drawing set (None = unowned or missing)."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT project_id FROM drawing_sets WHERE id = %s", (set_id,)
            ).fetchone()
        return str(row[0]) if row and row[0] else None

    def find_set(self, project_id: str | None, set_number: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, project_id, set_number, name FROM drawing_sets "
                "WHERE project_id IS NOT DISTINCT FROM %s AND set_number = %s",
                (project_id, set_number),
            ).fetchone()
        if row is None:
            return None
        return {
            "set_id": str(row[0]),
            "project_id": str(row[1]) if row[1] else None,
            "set_number": row[2],
            "name": row[3],
        }

    def versions(self, version_group_id: str) -> list[dict[str, Any]]:
        """All drawings in a version group, oldest first (year, then created)."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {_DRAWING_COLS} FROM drawings WHERE version_group_id = %s "
                "ORDER BY year NULLS LAST, created_at",
                (version_group_id,),
            ).fetchall()
        return [_drawing_dict(r) for r in rows]

    def link_versions(self, drawing_id: str, other_drawing_id: str) -> None:
        """Merge the two drawings' version groups into one."""
        with self._pool.connection() as conn:
            conn.execute(
                """UPDATE drawings SET version_group_id =
                       (SELECT version_group_id FROM drawings WHERE id = %s)
                   WHERE version_group_id = (SELECT version_group_id FROM drawings WHERE id = %s)""",
                (drawing_id, other_drawing_id),
            )

    def unlink_version(self, drawing_id: str) -> None:
        """Split a drawing back out into its own version group."""
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE drawings SET version_group_id = id WHERE id = %s", (drawing_id,)
            )

    def find_by_norm(self, dwg_number_norm: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {_DRAWING_COLS} FROM drawings WHERE dwg_number_norm = %s",
                (dwg_number_norm,),
            ).fetchall()
        return [_drawing_dict(r) for r in rows]

    def search_registry(self) -> list[dict[str, Any]]:
        """Lightweight full-registry scan for the matcher: id, numbers, description,
        project. 7k rows is fine to scan in-process for an MVP."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT d.id, d.dwg_number, d.dwg_number_norm, d.description,
                          d.project_id, p.name, d.year
                   FROM drawings d LEFT JOIN projects p ON d.project_id = p.id"""
            ).fetchall()
        return [
            {
                "drawing_id": str(r[0]),
                "dwg_number": r[1],
                "dwg_number_norm": r[2],
                "description": r[3],
                "project_id": str(r[4]) if r[4] else None,
                "project_name": r[5],
                "year": r[6],
            }
            for r in rows
        ]

    # --- sets ---

    def create_set(self, project_id: str | None, set_number: str, name: str | None) -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO drawing_sets (project_id, set_number, name) VALUES (%s, %s, %s) "
                "RETURNING id, project_id, set_number, name, created_at",
                (project_id, set_number, name),
            ).fetchone()
        return {
            "set_id": str(row[0]),
            "project_id": str(row[1]) if row[1] else None,
            "set_number": row[2],
            "name": row[3],
            "created_at": row[4].isoformat(),
        }

    def list_sets(self, project_id: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT s.id, s.project_id, s.set_number, s.name, s.created_at,
                          (SELECT count(*) FROM drawings d WHERE d.set_id = s.id)
                   FROM drawing_sets s WHERE s.project_id = %s ORDER BY s.set_number""",
                (project_id,),
            ).fetchall()
        return [
            {
                "set_id": str(r[0]),
                "project_id": str(r[1]) if r[1] else None,
                "set_number": r[2],
                "name": r[3],
                "created_at": r[4].isoformat(),
                "drawing_count": r[5],
            }
            for r in rows
        ]

    def delete_set(self, set_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE drawings SET set_id = NULL WHERE set_id = %s", (set_id,))
            conn.execute("DELETE FROM drawing_sets WHERE id = %s", (set_id,))

    # --- files on drawings ---

    def files_for_project(self, project_id: str) -> list[dict[str, Any]]:
        """Every file attached to any drawing of the project, one query -
        feeds the project file explorer without N+1 per drawing."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT f.id, f.filename, f.file_type, f.status, f.sheet_number,
                          f.created_at, f.drawing_id
                   FROM files f JOIN drawings d ON f.drawing_id = d.id
                   WHERE d.project_id = %s ORDER BY f.filename""",
                (project_id,),
            ).fetchall()
        return [
            {
                "file_id": str(r[0]),
                "filename": r[1],
                "file_type": r[2],
                "status": r[3],
                "sheet_number": r[4],
                "created_at": r[5].isoformat(),
                "drawing_id": str(r[6]),
            }
            for r in rows
        ]

    def files_for_drawing(self, drawing_id: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT id, filename, file_type, status, sheet_number, created_at
                   FROM files WHERE drawing_id = %s ORDER BY created_at""",
                (drawing_id,),
            ).fetchall()
        return [
            {
                "file_id": str(r[0]),
                "filename": r[1],
                "file_type": r[2],
                "status": r[3],
                "sheet_number": r[4],
                "created_at": r[5].isoformat(),
            }
            for r in rows
        ]

    def attach_file(
        self, file_id: str, drawing_id: str | None, sheet_number: str | None,
        auto: bool = False,
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE files SET drawing_id = %s, sheet_number = %s, auto_assigned = %s "
                "WHERE id = %s",
                (drawing_id, sheet_number, auto if drawing_id else False, file_id),
            )


class RegistryChunkRepository:
    """Searchable metadata cards for registry entities (projects, drawings,
    sets). One card per entity, upserted whenever the entity changes."""

    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def upsert(
        self,
        entity_type: str,
        entity_id: str,
        project_id: str | None,
        label: str,
        project_name: str | None,
        chunk_text: str,
        embedding: list[float],
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO registry_chunks
                       (entity_type, entity_id, project_id, label, project_name, chunk_text, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (entity_type, entity_id) DO UPDATE SET
                       project_id = EXCLUDED.project_id,
                       label = EXCLUDED.label,
                       project_name = EXCLUDED.project_name,
                       chunk_text = EXCLUDED.chunk_text,
                       embedding = EXCLUDED.embedding,
                       updated_at = now()""",
                (entity_type, entity_id, project_id, label, project_name,
                 chunk_text, json.dumps(embedding)),
            )

    def remove(self, entity_type: str, entity_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "DELETE FROM registry_chunks WHERE entity_type = %s AND entity_id = %s",
                (entity_type, entity_id),
            )

    def search(
        self, embedding: list[float], top_k: int, project_id: str | None = None,
        allowed_project_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        vector = json.dumps(embedding)
        # Same two-stage shape as chunk search: index-served distance pool,
        # then RLHF-weighted re-rank.
        pool_size = max(top_k * 4, 100)
        with self._pool.connection() as conn, conn.transaction():
            # SET LOCAL needs a real transaction (pool runs autocommit)
            conn.execute("SET LOCAL hnsw.ef_search = 200")
            rows = conn.execute(
                """SELECT * FROM (
                       SELECT entity_type, entity_id, project_id, label, project_name, chunk_text,
                              (1 - (embedding <=> %s::vector)) * feedback_weight AS score
                       FROM registry_chunks
                       WHERE (%s::uuid IS NULL OR project_id = %s::uuid)
                         AND (%s::uuid[] IS NULL
                              OR project_id = ANY(%s::uuid[])
                              OR project_id IS NULL)
                       ORDER BY embedding <=> %s::vector
                       LIMIT %s
                   ) candidates
                   ORDER BY score DESC
                   LIMIT %s""",
                (vector, project_id, project_id,
                 allowed_project_ids, allowed_project_ids,
                 vector, pool_size, top_k),
            ).fetchall()
        return [
            {
                "region_type": "registry",
                "entity_type": r[0],
                "entity_id": str(r[1]),
                "project_id": str(r[2]) if r[2] else None,
                "label": r[3],
                "project_name": r[4],
                "chunk_text": r[5],
                "score": round(float(r[6]), 4),
                # keep the evidence shape compatible with file-content hits
                "source_file_id": None,
                "bbox": None,
                "image_uri": None,
                "page": None,
                "filename": None,
                "dwg_number": r[3] if r[0] == "drawing" else None,
                "drawing_id": str(r[1]) if r[0] == "drawing" else None,
            }
            for r in rows
        ]

    def get_by_entity(self, entity_ids: list[str]) -> list[dict[str, Any]]:
        """Exact card lookup for identifier-anchored retrieval: when a question
        names a DWG number outright, its card is included deterministically
        rather than hoping embedding similarity clears the floor."""
        if not entity_ids:
            return []
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT entity_type, entity_id, project_id, label, project_name, chunk_text
                   FROM registry_chunks WHERE entity_id = ANY(%s::uuid[])""",
                (entity_ids,),
            ).fetchall()
        return [
            {
                "region_type": "registry",
                "entity_type": r[0],
                "entity_id": str(r[1]),
                "project_id": str(r[2]) if r[2] else None,
                "label": r[3],
                "project_name": r[4],
                "chunk_text": r[5],
                # exact identifier match - outranks any similarity score
                "score": 0.99,
                "source_file_id": None,
                "bbox": None,
                "image_uri": None,
                "page": None,
                "filename": None,
                "dwg_number": r[3] if r[0] == "drawing" else None,
                "drawing_id": str(r[1]) if r[0] == "drawing" else None,
            }
            for r in rows
        ]

    def count(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute("SELECT count(*) FROM registry_chunks").fetchone()[0]

    def adjust_weights(self, entities: list[tuple[str, str]], delta: float) -> None:
        """RLHF weight shift for registry cards, by (entity_type, entity_id)."""
        if not entities:
            return
        with self._pool.connection() as conn:
            for etype, eid in entities:
                conn.execute(
                    "UPDATE registry_chunks SET feedback_weight = "
                    "GREATEST(0.3, LEAST(2.0, feedback_weight + %s)) "
                    "WHERE entity_type = %s AND entity_id = %s",
                    (delta, etype, eid),
                )


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

    def delete_session(self, session_id: str) -> None:
        # chat_messages cascade via the session_id foreign key
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        evidence: list[dict] | None = None,
        version_context: dict | None = None,
    ) -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                """INSERT INTO chat_messages (session_id, role, content, evidence, version_context)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id, created_at""",
                (
                    session_id, role, content,
                    json.dumps(evidence) if evidence is not None else None,
                    json.dumps(version_context) if version_context is not None else None,
                ),
            ).fetchone()
        return {
            "message_id": str(row[0]),
            "role": role,
            "content": content,
            "evidence": evidence,
            "version_context": version_context,
            "created_at": row[1].isoformat(),
        }

    def get_message(self, session_id: str, message_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, role, content, evidence FROM chat_messages "
                "WHERE id = %s AND session_id = %s",
                (message_id, session_id),
            ).fetchone()
        if row is None:
            return None
        return {"message_id": str(row[0]), "role": row[1], "content": row[2], "evidence": row[3]}

    def set_rating(self, message_id: str, rating: int, comment: str | None) -> int:
        """Upsert a rating; returns the PREVIOUS rating (0 if none) so the
        caller can apply a weight delta rather than double-counting."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT rating FROM answer_feedback WHERE message_id = %s", (message_id,)
            ).fetchone()
            previous = row[0] if row else 0
            conn.execute(
                """INSERT INTO answer_feedback (message_id, rating, comment)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (message_id) DO UPDATE SET
                       rating = EXCLUDED.rating, comment = EXCLUDED.comment, updated_at = now()""",
                (message_id, rating, comment),
            )
        return previous

    def clear_rating(self, message_id: str) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT rating FROM answer_feedback WHERE message_id = %s", (message_id,)
            ).fetchone()
            previous = row[0] if row else 0
            conn.execute("DELETE FROM answer_feedback WHERE message_id = %s", (message_id,))
        return previous

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT m.id, m.role, m.content, m.evidence, m.version_context, m.created_at,
                          fb.rating
                   FROM chat_messages m LEFT JOIN answer_feedback fb ON fb.message_id = m.id
                   WHERE m.session_id = %s ORDER BY m.created_at""",
                (session_id,),
            ).fetchall()
        return [
            {
                "message_id": str(r[0]),
                "role": r[1],
                "content": r[2],
                "evidence": r[3],
                "version_context": r[4],
                "created_at": r[5].isoformat(),
                "feedback": r[6],
            }
            for r in rows
        ]


class StatsRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def snapshot(self, allowed_project_ids: list[str] | None = None) -> dict[str, Any]:
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
            projects = conn.execute("SELECT count(*) FROM projects").fetchone()[0]
            drawings = conn.execute(
                "SELECT count(*) FROM drawings WHERE deleted_at IS NULL"
            ).fetchone()[0]
            sets = conn.execute("SELECT count(*) FROM drawing_sets").fetchone()[0]
            # sheet totals from the registry, plus how many drawings still
            # have no count recorded (an actionable gap, not just a stat)
            sheets_total, sheets_missing = conn.execute(
                """SELECT coalesce(sum(sheet_count), 0),
                          count(*) FILTER (WHERE sheet_count IS NULL)
                   FROM drawings WHERE deleted_at IS NULL"""
            ).fetchone()
            # distinct document pages the extractor has actually read - the
            # "how much of the archive has been processed" number
            pages_extracted = conn.execute(
                "SELECT count(DISTINCT (source_file_id, page)) FROM chunks"
            ).fetchone()[0]
            unassigned = conn.execute(
                "SELECT count(*) FROM files WHERE drawing_id IS NULL"
            ).fetchone()[0]
            feedback = dict(
                conn.execute(
                    "SELECT rating, count(*) FROM answer_feedback GROUP BY rating"
                ).fetchall()
            )
            # top projects by drawing count, for the dashboard breakdown
            per_project = conn.execute(
                """SELECT p.id, p.name, p.number, count(d.id) AS drawings
                   FROM projects p
                        LEFT JOIN drawings d
                          ON d.project_id = p.id AND d.deleted_at IS NULL
                   WHERE %s::uuid[] IS NULL OR p.id = ANY(%s::uuid[])
                   GROUP BY p.id ORDER BY drawings DESC, p.name LIMIT 8""",
                (allowed_project_ids, allowed_project_ids),
            ).fetchall()
            # daily activity for the dashboard trend: uploads and questions,
            # last 14 calendar days (UTC), zero-filled client-agnostically here
            uploads_by_day = dict(
                conn.execute(
                    """SELECT date_trunc('day', created_at)::date, count(*)
                       FROM files
                       WHERE created_at >= date_trunc('day', now()) - interval '13 days'
                       GROUP BY 1"""
                ).fetchall()
            )
            questions_by_day = dict(
                conn.execute(
                    """SELECT date_trunc('day', created_at)::date, count(*)
                       FROM chat_messages
                       WHERE role = 'user'
                         AND created_at >= date_trunc('day', now()) - interval '13 days'
                       GROUP BY 1"""
                ).fetchall()
            )
            today = conn.execute("SELECT date_trunc('day', now())::date").fetchone()[0]
        activity_daily = [
            {
                "date": (day := today - timedelta(days=13 - i)).isoformat(),
                "uploads": uploads_by_day.get(day, 0),
                "questions": questions_by_day.get(day, 0),
            }
            for i in range(14)
        ]
        return {
            "activity_daily": activity_daily,
            "documents_total": sum(files_by_status.values()),
            "documents_by_status": files_by_status,
            "documents_by_type": files_by_type,
            "documents_unassigned": unassigned,
            "chunks_total": chunks_total,
            "chunks_by_confidence": chunks_by_confidence,
            "chunks_corrected": corrected,
            "chat_sessions": sessions,
            "questions_asked": questions,
            "projects_total": projects,
            "drawings_total": drawings,
            "sets_total": sets,
            "sheets_total": sheets_total,
            "sheets_missing": sheets_missing,
            "pages_extracted": pages_extracted,
            "feedback_helpful": feedback.get(1, 0),
            "feedback_unhelpful": feedback.get(-1, 0),
            "drawings_per_project": [
                {
                    "project_id": str(r[0]),
                    "name": r[1],
                    "number": r[2],
                    "drawings": r[3],
                }
                for r in per_project
            ],
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

    def adjust_weights(self, chunk_ids: list[str], delta: float) -> None:
        """RLHF: shift the retrieval weight of rated evidence, clamped so no
        region can be boosted or buried without limit."""
        if not chunk_ids:
            return
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE chunks SET feedback_weight = GREATEST(0.3, LEAST(2.0, feedback_weight + %s)) "
                "WHERE id = ANY(%s::uuid[])",
                (delta, chunk_ids),
            )

    def search(
        self, embedding: list[float], top_k: int, project_id: str | None = None,
        file_id: str | None = None, allowed_project_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector search over ingested regions. Optionally scoped to one
        project (via the file -> drawing -> project chain) or to one document
        (file-scoped chat); results carry the drawing/project context so
        evidence can show where a region lives. allowed_project_ids is the
        caller's ROLE scope: content on other sheets never enters the pool
        (unassigned content, owned by no sheet, stays retrievable)."""
        vector = json.dumps(embedding)
        # Two-stage retrieval so the HNSW index can serve the search: the
        # inner query fetches a candidate pool ordered by PURE distance (the
        # only ordering a vector index accelerates), the outer re-ranks by
        # the RLHF-weighted score. The pool is wide enough that a weight
        # boost cannot promote something from outside it in practice.
        pool_size = max(top_k * 4, 100)
        with self._pool.connection() as conn, conn.transaction():
            # HNSW returns at most ef_search candidates per scan; keep it
            # ahead of the pool. SET LOCAL needs a real transaction (the
            # pool runs autocommit, where it would silently no-op).
            conn.execute("SET LOCAL hnsw.ef_search = 200")
            rows = conn.execute(
                """SELECT * FROM (
                       SELECT c.source_file_id, c.region_type, c.chunk_text, c.bbox,
                              c.image_uri, c.page, f.filename,
                              f.drawing_id, d.dwg_number, p.name AS project_name,
                              d.version_group_id, d.year, d.drawing_date, d.version_note,
                              s.set_number, c.id AS chunk_id, c.feedback_weight,
                              (1 - (c.embedding <=> %s::vector)) * c.feedback_weight AS score
                       FROM chunks c
                            JOIN files f ON f.id = c.source_file_id
                            LEFT JOIN drawings d ON f.drawing_id = d.id
                            LEFT JOIN projects p ON d.project_id = p.id
                            LEFT JOIN drawing_sets s ON d.set_id = s.id
                       WHERE (%s::uuid IS NULL OR d.project_id = %s::uuid)
                         AND (%s::uuid IS NULL OR c.source_file_id = %s::uuid)
                         AND (%s::uuid[] IS NULL
                              OR d.project_id = ANY(%s::uuid[])
                              OR f.drawing_id IS NULL)
                       ORDER BY c.embedding <=> %s::vector
                       LIMIT %s
                   ) candidates
                   ORDER BY score DESC
                   LIMIT %s""",
                (vector, project_id, project_id, file_id, file_id,
                 allowed_project_ids, allowed_project_ids, vector,
                 pool_size, top_k),
            ).fetchall()
        return [
            {
                "source_file_id": str(r[0]),
                "region_type": r[1],
                "chunk_text": r[2],
                "bbox": r[3],
                "image_uri": r[4],
                "page": r[5],
                "filename": r[6],
                "drawing_id": str(r[7]) if r[7] else None,
                "dwg_number": r[8],
                "project_name": r[9],
                "version_group_id": str(r[10]) if r[10] else None,
                "year": r[11],
                "drawing_date": r[12],
                "version_note": r[13],
                "set_number": r[14],
                "chunk_id": str(r[15]),
                "feedback_weight": round(float(r[16]), 3),
                "score": round(float(r[17]), 4),
            }
            for r in rows
        ]


class AuthRepository:
    """Users and session tokens. Token digests only - never raw tokens."""

    def __init__(self, pool):
        self._pool = pool

    def count_users(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute("SELECT count(*) FROM users").fetchone()[0]

    def create_user(
        self,
        username: str,
        password_hash: str,
        full_name: str | None = None,
        email: str | None = None,
        role_id: str | None = None,
        is_admin: bool = False,
    ) -> str:
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO users (username, password_hash, full_name, email, role_id, is_admin) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (username.strip().lower(), password_hash, full_name, email, role_id, is_admin),
            ).fetchone()
        return str(row[0])

    def list_users(self) -> list[dict]:
        """Everyone with an account, for the Users page. Never returns hashes."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT u.id, u.username, u.full_name, u.email, u.created_at, "
                "u.is_admin, u.role_id, r.name "
                "FROM users u LEFT JOIN roles r ON r.id = u.role_id "
                "ORDER BY u.created_at"
            ).fetchall()
        return [
            {
                "user_id": str(r[0]),
                "username": r[1],
                "full_name": r[2],
                "email": r[3],
                "created_at": r[4].isoformat(),
                "is_admin": r[5],
                "role_id": str(r[6]) if r[6] else None,
                "role_name": r[7],
            }
            for r in rows
        ]

    def delete_user(self, user_id: str) -> None:
        with self._pool.connection() as conn:
            # auth_tokens cascade, so the removed account's sessions die with it
            conn.execute("DELETE FROM users WHERE id = %s", (user_id,))

    def get_user_by_username(self, username: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash FROM users WHERE username = %s",
                (username,),
            ).fetchone()
        if row is None:
            return None
        return {"id": str(row[0]), "username": row[1], "password_hash": row[2]}

    def get_user_by_id(self, user_id: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, is_admin FROM users WHERE id = %s",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": str(row[0]),
            "username": row[1],
            "password_hash": row[2],
            "is_admin": row[3],
        }

    def set_password_hash(self, user_id: str, password_hash: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (password_hash, user_id),
            )

    def insert_token(self, token_sha256: str, user_id: str, ttl_days: int) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO auth_tokens (token_sha256, user_id, expires_at) "
                "VALUES (%s, %s, now() + make_interval(days => %s))",
                (token_sha256, user_id, ttl_days),
            )
            # opportunistic cleanup so expired rows never accumulate
            conn.execute("DELETE FROM auth_tokens WHERE expires_at < now()")

    def get_user_by_token(self, token_sha256: str) -> dict | None:
        """The per-request identity, role included. Joining the role here
        (rather than caching it in the token) means editing a role takes
        effect on every holder's NEXT request - no re-login required."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT u.id, u.username, u.full_name, u.email, u.is_admin, "
                "r.id, r.name, r.pages, r.all_sheets, "
                "(SELECT array_agg(rp.project_id::text) FROM role_projects rp "
                " WHERE rp.role_id = r.id) "
                "FROM auth_tokens t "
                "JOIN users u ON u.id = t.user_id "
                "LEFT JOIN roles r ON r.id = u.role_id "
                "WHERE t.token_sha256 = %s AND t.expires_at > now()",
                (token_sha256,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": str(row[0]),
            "username": row[1],
            "full_name": row[2],
            "email": row[3],
            "is_admin": row[4],
            "role": None if row[5] is None else {
                "id": str(row[5]),
                "name": row[6],
                "pages": list(row[7] or []),
                "all_sheets": row[8],
                "project_ids": list(row[9] or []),
            },
        }

    # --- roles ---

    def list_roles(self) -> list[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT r.id, r.name, r.pages, r.all_sheets, r.created_at, "
                "(SELECT count(*) FROM users u WHERE u.role_id = r.id), "
                "(SELECT array_agg(rp.project_id::text) FROM role_projects rp "
                " WHERE rp.role_id = r.id) "
                "FROM roles r ORDER BY r.created_at"
            ).fetchall()
        return [
            {
                "role_id": str(r[0]),
                "name": r[1],
                "pages": list(r[2] or []),
                "all_sheets": r[3],
                "created_at": r[4].isoformat(),
                "user_count": r[5],
                "project_ids": list(r[6] or []),
            }
            for r in rows
        ]

    def get_role_by_name(self, name: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, name FROM roles WHERE lower(name) = lower(%s)", (name,)
            ).fetchone()
        return None if row is None else {"role_id": str(row[0]), "name": row[1]}

    def create_role(
        self, name: str, pages: list[str], all_sheets: bool, project_ids: list[str]
    ) -> str:
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO roles (name, pages, all_sheets) VALUES (%s, %s, %s) RETURNING id",
                (name, pages, all_sheets),
            ).fetchone()
            role_id = str(row[0])
            for pid in project_ids:
                conn.execute(
                    "INSERT INTO role_projects (role_id, project_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (role_id, pid),
                )
        return role_id

    def update_role(
        self,
        role_id: str,
        name: str,
        pages: list[str],
        all_sheets: bool,
        project_ids: list[str],
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE roles SET name = %s, pages = %s, all_sheets = %s WHERE id = %s",
                (name, pages, all_sheets, role_id),
            )
            # replace the sheet list wholesale - it is tiny and explicit
            conn.execute("DELETE FROM role_projects WHERE role_id = %s", (role_id,))
            for pid in project_ids:
                conn.execute(
                    "INSERT INTO role_projects (role_id, project_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (role_id, pid),
                )

    def delete_role(self, role_id: str) -> None:
        # users.role_id has ON DELETE SET NULL: holders become roleless and
        # lose access on their next request - the caller warns about this
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM roles WHERE id = %s", (role_id,))

    def update_user(
        self, user_id: str, role_id: str | None = ..., is_admin: bool | None = None
    ) -> None:
        sets, params = [], []
        if role_id is not ...:
            sets.append("role_id = %s")
            params.append(role_id)
        if is_admin is not None:
            sets.append("is_admin = %s")
            params.append(is_admin)
        if not sets:
            return
        params.append(user_id)
        with self._pool.connection() as conn:
            conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = %s", params)

    def count_admins(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute("SELECT count(*) FROM users WHERE is_admin").fetchone()[0]

    def delete_token(self, token_sha256: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM auth_tokens WHERE token_sha256 = %s", (token_sha256,))

    def delete_tokens_for_user(self, user_id: str, except_sha: str | None = None) -> None:
        with self._pool.connection() as conn:
            if except_sha is None:
                conn.execute("DELETE FROM auth_tokens WHERE user_id = %s", (user_id,))
            else:
                conn.execute(
                    "DELETE FROM auth_tokens WHERE user_id = %s AND token_sha256 <> %s",
                    (user_id, except_sha),
                )


class DismissedDuplicateRepository:
    """Pairs the user has ruled out as duplicates - symmetric, so both
    directions are stored and either file's listing suppresses the other."""

    def __init__(self, pool):
        self._pool = pool

    def dismiss(self, file_id: str, other_file_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO dismissed_duplicates (file_id, other_file_id) "
                "VALUES (%s, %s), (%s, %s) ON CONFLICT DO NOTHING",
                (file_id, other_file_id, other_file_id, file_id),
            )
