"""Data access layer - the only place SQL lives."""
import json
from typing import Any

from psycopg_pool import ConnectionPool


class FileRepository:
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
        self, file_id: str, s3_key: str, chunks: list[dict], embedding: list[float] | None = None
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE files SET s3_key = %s, status = 'extracted', extraction = %s, "
                "embedding = %s, error = NULL WHERE id = %s",
                (s3_key, json.dumps(chunks), json.dumps(embedding) if embedding else None, file_id),
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

    def delete(self, file_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM files WHERE id = %s", (file_id,))

    def get(self, file_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, filename, file_type, status, extraction, created_at, s3_key, render, "
                "content_sha256, error, drawing_id FROM files WHERE id = %s",
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
                          f.error, f.drawing_id, d.dwg_number, f.auto_assigned, count(c.id),
                          (
                            SELECT json_agg(json_build_object(
                                     'file_id', o.id, 'filename', o.filename,
                                     'similarity', round((1 - (f.embedding <=> o.embedding))::numeric, 4))
                                   ORDER BY f.embedding <=> o.embedding)
                            FROM files o
                            WHERE o.id <> f.id AND o.embedding IS NOT NULL
                              AND (1 - (f.embedding <=> o.embedding)) >= %s
                          ) AS similar
                   FROM files f
                        LEFT JOIN chunks c ON c.source_file_id = f.id
                        LEFT JOIN drawings d ON f.drawing_id = d.id
                   GROUP BY f.id, d.dwg_number ORDER BY f.created_at DESC""",
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
                "similar_documents": r[10] or [],
                "is_duplicate": bool(r[10]),
            }
            for r in rows
        ]


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
                          (SELECT count(*) FROM drawings d WHERE d.project_id = p.id),
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
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM drawings WHERE id = %s", (drawing_id,))

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
        self, embedding: list[float], top_k: int, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        vector = json.dumps(embedding)
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT entity_type, entity_id, project_id, label, project_name, chunk_text,
                          (1 - (embedding <=> %s::vector)) * feedback_weight AS score
                   FROM registry_chunks
                   WHERE %s::uuid IS NULL OR project_id = %s::uuid
                   ORDER BY (1 - (embedding <=> %s::vector)) * feedback_weight DESC
                   LIMIT %s""",
                (vector, project_id, project_id, vector, top_k),
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
            projects = conn.execute("SELECT count(*) FROM projects").fetchone()[0]
            drawings = conn.execute("SELECT count(*) FROM drawings").fetchone()[0]
            sets = conn.execute("SELECT count(*) FROM drawing_sets").fetchone()[0]
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
                   FROM projects p LEFT JOIN drawings d ON d.project_id = p.id
                   GROUP BY p.id ORDER BY drawings DESC, p.name LIMIT 8"""
            ).fetchall()
        return {
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
        self, embedding: list[float], top_k: int, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Vector search over ingested regions. Optionally scoped to one
        project (via the file -> drawing -> project chain); results carry the
        drawing/project context so evidence can show where a region lives."""
        vector = json.dumps(embedding)
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT c.source_file_id, c.region_type, c.chunk_text, c.bbox,
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
                   WHERE %s::uuid IS NULL OR d.project_id = %s::uuid
                   ORDER BY (1 - (c.embedding <=> %s::vector)) * c.feedback_weight DESC
                   LIMIT %s""",
                (vector, project_id, project_id, vector, top_k),
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
