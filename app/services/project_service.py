"""Project and drawing management (Phase 1: replace the spreadsheet book)."""
from typing import Any

from app.exceptions import FileNotFound
from app.repositories import DrawingRepository, FileRepository, ProjectRepository
from app.services import matching


class ProjectService:
    def __init__(self, projects: ProjectRepository, drawings: DrawingRepository):
        self._projects = projects
        self._drawings = drawings

    def create(self, name: str, number: str | None, description: str | None) -> dict:
        return self._projects.create(name.strip(), number, description)

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
        return self._projects.get(project_id)

    def delete(self, project_id: str) -> None:
        if self._projects.get(project_id) is None:
            raise FileNotFound("Project not found")
        self._projects.delete(project_id)


class DrawingService:
    def __init__(
        self,
        drawings: DrawingRepository,
        projects: ProjectRepository,
        files: FileRepository,
    ):
        self._drawings = drawings
        self._projects = projects
        self._files = files

    def create(self, fields: dict) -> dict:
        fields = dict(fields)
        if fields.get("dwg_number") and not fields.get("dwg_number_norm"):
            parsed = matching.parse_filename(fields["dwg_number"])
            if parsed["dwg_candidates"]:
                fields["dwg_number_norm"] = parsed["dwg_candidates"][0]["norm"]
        if fields.get("drawing_date") and not fields.get("year"):
            fields["year"] = matching.parse_year(fields["drawing_date"])
        return self._drawings.create(fields)

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
        if self._drawings.get(drawing_id) is None:
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
        return self._drawings.get(drawing_id)

    def delete(self, drawing_id: str) -> None:
        if self._drawings.get(drawing_id) is None:
            raise FileNotFound("Drawing not found")
        self._drawings.delete(drawing_id)

    def link_versions(self, drawing_id: str, other_drawing_id: str) -> dict:
        """Declare two drawings as versions of the same drawing."""
        for did in (drawing_id, other_drawing_id):
            if self._drawings.get(did) is None:
                raise FileNotFound("Drawing not found")
        self._drawings.link_versions(drawing_id, other_drawing_id)
        return self.get_detail(drawing_id)

    def unlink_version(self, drawing_id: str) -> dict:
        if self._drawings.get(drawing_id) is None:
            raise FileNotFound("Drawing not found")
        self._drawings.unlink_version(drawing_id)
        return self.get_detail(drawing_id)

    def create_set(self, project_id: str | None, set_number: str, name: str | None) -> dict:
        return self._drawings.create_set(project_id, set_number, name)

    def delete_set(self, set_id: str) -> None:
        self._drawings.delete_set(set_id)

    # --- file association ---

    def suggestions_for_file(self, file_id: str) -> dict:
        """Ranked project/drawing suggestions for an uploaded file, from its
        filename signals matched against projects and the drawing registry."""
        record = self._files.get(file_id)
        if record is None:
            raise FileNotFound("File not found")
        filename = record["filename"]
        parsed = matching.parse_filename(filename)
        return {
            "file_id": file_id,
            "filename": filename,
            "parsed": parsed,
            "project_suggestions": matching.suggest_projects(
                filename, self._projects.list_all()
            ),
            "drawing_suggestions": matching.suggest_drawings(
                filename, self._drawings.search_registry()
            ),
        }

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
        return self.get_detail(drawing_id)

    def unassign_file(self, file_id: str) -> None:
        if self._files.get(file_id) is None:
            raise FileNotFound("File not found")
        self._drawings.attach_file(file_id, None, None)
