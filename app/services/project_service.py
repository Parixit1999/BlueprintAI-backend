"""Project and drawing management (Phase 1: replace the spreadsheet book).

Every mutation also refreshes the affected registry metadata cards (see
registry_index.py) so the RAG layer can answer questions about the registry
itself. Card sync is best-effort and never fails the mutation.
"""
from typing import Any

from app.exceptions import FileNotFound
from app.repositories import DrawingRepository, FileRepository, ProjectRepository
from app.services import matching
from app.services.registry_index import RegistryIndexService


class ProjectService:
    def __init__(
        self,
        projects: ProjectRepository,
        drawings: DrawingRepository,
        index: RegistryIndexService,
    ):
        self._projects = projects
        self._drawings = drawings
        self._index = index

    def create(self, name: str, number: str | None, description: str | None) -> dict:
        project = self._projects.create(name.strip(), number, description)
        self._index.index_project(project["project_id"])
        return project

    def list_all(self) -> list[dict]:
        return self._projects.list_all()

    def get_detail(self, project_id: str) -> dict:
        project = self._projects.get(project_id)
        if project is None:
            raise FileNotFound("Project not found")
        return {
            **project,
            "drawings": self._drawings.list_for_project(project_id),
            "sets": self._drawings.list_sets(project_id),
        }

    def update(self, project_id: str, fields: dict) -> dict:
        if self._projects.get(project_id) is None:
            raise FileNotFound("Project not found")
        self._projects.update(project_id, fields)
        # cascade: the project name appears on its drawings'/sets' cards
        self._index.index_project(project_id, cascade=True)
        return self._projects.get(project_id)

    def delete(self, project_id: str) -> None:
        if self._projects.get(project_id) is None:
            raise FileNotFound("Project not found")
        # capture members before the FK sets their project_id to NULL
        orphaned = [d["drawing_id"] for d in self._drawings.list_for_project(project_id)]
        orphaned_sets = [s["set_id"] for s in self._drawings.list_sets(project_id)]
        self._projects.delete(project_id)
        self._index.remove("project", project_id)
        for sid in orphaned_sets:
            self._index.remove("set", sid)
        for did in orphaned:
            self._index.index_drawing(did)


class DrawingService:
    def __init__(
        self,
        drawings: DrawingRepository,
        projects: ProjectRepository,
        files: FileRepository,
        index: RegistryIndexService,
    ):
        self._drawings = drawings
        self._projects = projects
        self._files = files
        self._index = index

    def create(self, fields: dict) -> dict:
        fields = dict(fields)
        if fields.get("dwg_number") and not fields.get("dwg_number_norm"):
            parsed = matching.parse_filename(fields["dwg_number"])
            if parsed["dwg_candidates"]:
                fields["dwg_number_norm"] = parsed["dwg_candidates"][0]["norm"]
        if fields.get("drawing_date") and not fields.get("year"):
            fields["year"] = matching.parse_year(fields["drawing_date"])
        drawing = self._drawings.create(fields)
        self._index.index_drawing(drawing["drawing_id"])
        if drawing.get("project_id"):
            self._index.index_project(drawing["project_id"])
        return drawing

    def get_detail(self, drawing_id: str) -> dict:
        drawing = self._drawings.get(drawing_id)
        if drawing is None:
            raise FileNotFound("Drawing not found")
        versions = []
        if drawing["version_group_id"]:
            versions = [
                v for v in self._drawings.versions(drawing["version_group_id"])
                if v["drawing_id"] != drawing_id
            ]
        project = self._projects.get(drawing["project_id"]) if drawing["project_id"] else None
        return {
            **drawing,
            "project_name": project["name"] if project else None,
            "files": self._drawings.files_for_drawing(drawing_id),
            "other_versions": versions,
        }

    def update(self, drawing_id: str, fields: dict) -> dict:
        before = self._drawings.get(drawing_id)
        if before is None:
            raise FileNotFound("Drawing not found")
        fields = dict(fields)
        if "dwg_number" in fields:
            parsed = matching.parse_filename(fields["dwg_number"] or "")
            fields["dwg_number_norm"] = (
                parsed["dwg_candidates"][0]["norm"] if parsed["dwg_candidates"] else None
            )
        if "drawing_date" in fields:
            fields["year"] = matching.parse_year(fields["drawing_date"])
        self._drawings.update(drawing_id, fields)
        after = self._drawings.get(drawing_id)
        self._index.index_drawing(drawing_id)
        for pid in {before.get("project_id"), after.get("project_id")} - {None}:
            self._index.index_project(pid)
        for sid, pid in {
            (before.get("set_id"), before.get("project_id")),
            (after.get("set_id"), after.get("project_id")),
        }:
            if sid:
                self._index.index_set(sid, pid)
        return after

    def delete(self, drawing_id: str) -> None:
        drawing = self._drawings.get(drawing_id)
        if drawing is None:
            raise FileNotFound("Drawing not found")
        self._drawings.delete(drawing_id)
        self._index.remove("drawing", drawing_id)
        if drawing.get("project_id"):
            self._index.index_project(drawing["project_id"])

    def link_versions(self, drawing_id: str, other_drawing_id: str) -> dict:
        """Declare two drawings as versions of the same drawing."""
        for did in (drawing_id, other_drawing_id):
            if self._drawings.get(did) is None:
                raise FileNotFound("Drawing not found")
        self._drawings.link_versions(drawing_id, other_drawing_id)
        # version lists appear on every card in the group
        group = self._drawings.get(drawing_id)["version_group_id"]
        for v in self._drawings.versions(group):
            self._index.index_drawing(v["drawing_id"])
        return self.get_detail(drawing_id)

    def unlink_version(self, drawing_id: str) -> dict:
        drawing = self._drawings.get(drawing_id)
        if drawing is None:
            raise FileNotFound("Drawing not found")
        old_group = drawing["version_group_id"]
        self._drawings.unlink_version(drawing_id)
        self._index.index_drawing(drawing_id)
        if old_group:
            for v in self._drawings.versions(old_group):
                self._index.index_drawing(v["drawing_id"])
        return self.get_detail(drawing_id)

    def create_set(self, project_id: str | None, set_number: str, name: str | None) -> dict:
        created = self._drawings.create_set(project_id, set_number, name)
        self._index.index_set(created["set_id"], project_id)
        return created

    def delete_set(self, set_id: str) -> None:
        self._drawings.delete_set(set_id)
        self._index.remove("set", set_id)

    # --- file association ---

    def suggestions_for_file(self, file_id: str) -> dict:
        """Ranked project/drawing suggestions for an uploaded file, from its
        filename signals matched against projects and the drawing registry."""
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound("File not found")
        filename = record["filename"]
        # "information found within the actual file content": the extracted
        # region texts (title blocks especially) carry DWG/project signals even
        # when the filename is meaningless (scan0001.png)
        content_texts = [
            c.get("chunk_text") for c in (record.get("extraction") or []) if c.get("chunk_text")
        ]
        parsed = matching.parse_filename(filename)
        return {
            "file_id": file_id,
            "filename": filename,
            "parsed": parsed,
            "content_signals": matching.parse_content(content_texts),
            "project_suggestions": matching.suggest_projects(
                filename, self._projects.list_all(), content_texts
            ),
            "drawing_suggestions": matching.suggest_drawings(
                filename, self._drawings.search_registry(), content_texts
            ),
        }

    # Auto-assign gate: only an EXACT normalized DWG-number match (score 0.95,
    # from filename or content), and only when that number maps to exactly one
    # drawing - version groups share numbers, and picking a version is a human
    # decision. Everything weaker stays a suggestion.
    AUTO_ASSIGN_SCORE = 0.95

    def suggest_and_maybe_assign(self, file_id: str) -> dict:
        """Post-extraction hook: compute suggestions and auto-assign when the
        gate passes. Returns {auto_assignment, suggestions} for the upload UI."""
        suggestions = self.suggestions_for_file(file_id)
        exact = [
            d for d in suggestions["drawing_suggestions"]
            if d["score"] >= self.AUTO_ASSIGN_SCORE
        ]
        exact_ids = {d["drawing_id"] for d in exact}
        auto = None
        if len(exact_ids) == 1:
            target = exact[0]
            record = self._files.get(file_id)
            sheet = matching.parse_filename(record["filename"])["sheet_number"]
            self._drawings.attach_file(file_id, target["drawing_id"], sheet, auto=True)
            auto = {
                "drawing_id": target["drawing_id"],
                "dwg_number": target["dwg_number"],
                "project_name": target.get("project_name"),
                "sheet_number": sheet,
                "reason": target["reason"],
            }
        return {"auto_assignment": auto, "suggestions": suggestions}

    def assign_file(
        self,
        file_id: str,
        drawing_id: str | None,
        sheet_number: str | None,
        new_drawing: dict | None,
    ) -> dict:
        """Attach a file to an existing drawing, or create a drawing (optionally
        under a project) and attach to it. Sheet number defaults from filename."""
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound("File not found")

        if drawing_id is None and new_drawing is not None:
            fields = dict(new_drawing)
            if not fields.get("dwg_number"):
                parsed = matching.parse_filename(record["filename"])
                if parsed["dwg_candidates"]:
                    fields["dwg_number"] = parsed["dwg_candidates"][0]["norm"]
            created = self.create({**fields, "source": "upload"})
            drawing_id = created["drawing_id"]
        if drawing_id is None:
            raise FileNotFound("Provide drawing_id or new_drawing")
        if self._drawings.get(drawing_id) is None:
            raise FileNotFound("Drawing not found")

        if sheet_number is None:
            sheet_number = matching.parse_filename(record["filename"])["sheet_number"]
        self._drawings.attach_file(file_id, drawing_id, sheet_number)
        self._index.index_drawing(drawing_id)  # file list appears on the card
        return self.get_detail(drawing_id)

    def unassign_file(self, file_id: str) -> None:
        if self._files.get(file_id) is None:
            raise FileNotFound("File not found")
        with self._files._pool.connection() as conn:
            row = conn.execute(
                "SELECT drawing_id FROM files WHERE id = %s", (file_id,)
            ).fetchone()
        old_drawing_id = str(row[0]) if row and row[0] else None
        self._drawings.attach_file(file_id, None, None)
        if old_drawing_id:
            self._index.index_drawing(old_drawing_id)
