"""Registry metadata as RAG content.

Every project, drawing, and set gets one searchable "metadata card" - a plain
sentence rendering of its fields plus its relationships (project membership,
set membership, versions, attached files). Cards are (re)embedded whenever the
entity changes, so chat can answer questions like "what contract covers
drawing 11767-W-59?" or "which drawings are in set 12A?" even when that
information exists only in the registry, not on any drawing.

Card generation is best-effort: a failed embed (e.g. Ollama down) must never
fail the registry mutation that triggered it - the card is simply stale until
the next change or a reindex.
"""
import logging

from app.repositories import (
    DrawingRepository,
    ProjectRepository,
    RegistryChunkRepository,
)
from app.services.ai.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class RegistryIndexService:
    def __init__(
        self,
        registry: RegistryChunkRepository,
        projects: ProjectRepository,
        drawings: DrawingRepository,
        embedder: EmbeddingProvider,
    ):
        self._registry = registry
        self._projects = projects
        self._drawings = drawings
        self._embedder = embedder

    # ---- card builders ----

    def _project_card(self, project: dict) -> str:
        bits = [f"Project: {project['name']}"]
        if project.get("number"):
            bits.append(f"(project number {project['number']})")
        text = " ".join(bits) + "."
        if project.get("description"):
            text += f" {project['description']}"
        drawings = self._drawings.list_for_project(project["project_id"])
        if drawings:
            names = ", ".join(d["dwg_number"] or "unnumbered" for d in drawings[:40])
            text += f" Contains {len(drawings)} drawing(s): {names}."
        return text

    def _drawing_card(self, drawing: dict) -> str:
        title = drawing.get("dwg_number") or "Unnumbered drawing"
        text = f"Drawing {title}"
        if drawing.get("description"):
            text += f": {drawing['description']}"
        text += "."
        project = self._projects.get(drawing["project_id"]) if drawing.get("project_id") else None
        if project:
            text += f" Project: {project['name']}"
            if project.get("number"):
                text += f" (number {project['number']})"
            text += "."
        if drawing.get("contract_number"):
            text += f" Contract number: {drawing['contract_number']}."
        if drawing.get("drawing_date") or drawing.get("year"):
            text += f" Date: {drawing.get('drawing_date') or drawing.get('year')}."
        if drawing.get("sheet_count"):
            text += f" {drawing['sheet_count']} sheets."
        if drawing.get("version_note"):
            text += f" Version note: {drawing['version_note']}."

        files = self._drawings.files_for_drawing(drawing["drawing_id"])
        if files:
            listed = ", ".join(
                f"{f['filename']}" + (f" (sheet {f['sheet_number']})" if f["sheet_number"] else "")
                for f in files[:20]
            )
            text += f" Files: {listed}."
        if drawing.get("version_group_id"):
            others = [
                v for v in self._drawings.versions(drawing["version_group_id"])
                if v["drawing_id"] != drawing["drawing_id"]
            ]
            if others:
                listed = "; ".join(
                    (v["dwg_number"] or "unnumbered")
                    + (f" ({v['year']})" if v.get("year") else "")
                    + (f", {v['version_note']}" if v.get("version_note") else "")
                    for v in others
                )
                text += f" Other versions of this drawing: {listed}."
        return text

    def _set_card(self, s: dict, project: dict | None, drawings: list[dict]) -> str:
        text = f"Drawing set {s['set_number']}"
        if s.get("name"):
            text += f" ({s['name']})"
        if project:
            text += f" in project {project['name']}"
        text += "."
        members = [d for d in drawings if d.get("set_id") == s["set_id"]]
        if members:
            listed = ", ".join(d["dwg_number"] or "unnumbered" for d in members[:40])
            text += f" Contains drawings: {listed}."
        return text

    # ---- sync entry points (best-effort, never raise) ----

    def index_project(self, project_id: str, cascade: bool = False) -> None:
        try:
            project = self._projects.get(project_id)
            if project is None:
                self._registry.remove("project", project_id)
                return
            card = self._project_card(project)
            self._registry.upsert(
                "project", project_id, project_id, project["name"], project["name"],
                card, self._embedder.embed(card),
            )
            if cascade:
                # project name appears on drawing/set cards
                for d in self._drawings.list_for_project(project_id):
                    self.index_drawing(d["drawing_id"])
                for s in self._drawings.list_sets(project_id):
                    self.index_set(s["set_id"], project_id)
        except Exception:
            logger.exception("registry card sync failed for project %s", project_id)

    def index_drawing(self, drawing_id: str) -> None:
        try:
            drawing = self._drawings.get(drawing_id)
            if drawing is None:
                self._registry.remove("drawing", drawing_id)
                return
            project = self._projects.get(drawing["project_id"]) if drawing.get("project_id") else None
            card = self._drawing_card(drawing)
            self._registry.upsert(
                "drawing", drawing_id, drawing.get("project_id"),
                drawing.get("dwg_number") or "Drawing",
                project["name"] if project else None,
                card, self._embedder.embed(card),
            )
        except Exception:
            logger.exception("registry card sync failed for drawing %s", drawing_id)

    def index_set(self, set_id: str, project_id: str | None) -> None:
        try:
            sets = self._drawings.list_sets(project_id) if project_id else []
            s = next((x for x in sets if x["set_id"] == set_id), None)
            if s is None:
                self._registry.remove("set", set_id)
                return
            project = self._projects.get(project_id) if project_id else None
            drawings = self._drawings.list_for_project(project_id) if project_id else []
            card = self._set_card(s, project, drawings)
            self._registry.upsert(
                "set", set_id, project_id, f"Set {s['set_number']}",
                project["name"] if project else None,
                card, self._embedder.embed(card),
            )
        except Exception:
            logger.exception("registry card sync failed for set %s", set_id)

    def remove(self, entity_type: str, entity_id: str) -> None:
        try:
            self._registry.remove(entity_type, entity_id)
        except Exception:
            logger.exception("registry card removal failed for %s %s", entity_type, entity_id)

    def reindex_all(self) -> dict:
        """Rebuild every card. Cheap for a manually-curated registry; the
        importer-scale case (thousands of drawings) should run this offline."""
        counts = {"projects": 0, "drawings": 0, "sets": 0}
        for p in self._projects.list_all():
            self.index_project(p["project_id"])
            counts["projects"] += 1
            for s in self._drawings.list_sets(p["project_id"]):
                self.index_set(s["set_id"], p["project_id"])
                counts["sets"] += 1
        with self._drawings._pool.connection() as conn:  # all drawings incl. unassigned
            ids = [str(r[0]) for r in conn.execute("SELECT id FROM drawings").fetchall()]
        for did in ids:
            self.index_drawing(did)
            counts["drawings"] += 1
        return counts
