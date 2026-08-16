"""The registry as a spreadsheet: the Drawings Number Book, live.

Engineers coming from the Excel book get the same mental model - sheet tabs
(Main Book + one per project), the same six columns, cell editing - but every
cell is backed by the real registry, so edits re-embed metadata cards and
show up in chat/search immediately.
"""
import io
import re
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.dependencies import drawing_service, project_service
from app.services.project_service import DrawingService, ProjectService

router = APIRouter(prefix="/registry", tags=["registry"])

Drawings = Annotated[DrawingService, Depends(drawing_service)]
Projects = Annotated[ProjectService, Depends(project_service)]

# The book's column order, reused for every export sheet.
_BOOK_HEADER = ["DWG #", "# of Sheets", "Description", "Contract #", "Date", "Set #"]


class RegistryRowUpdate(BaseModel):
    dwg_number: str | None = None
    description: str | None = None
    contract_number: str | None = None
    drawing_date: str | None = None
    sheet_count: int | None = None
    set_number: str | None = None
    project_id: str | None = None


class RegistryRowCreate(RegistryRowUpdate):
    pass


@router.get("/tabs")
def registry_tabs(drawings: Drawings, projects: Projects):
    """Sheet tabs: Main Book (every drawing) then one tab per project,
    mirroring the workbook's structure."""
    return {
        "tabs": [
            {"id": None, "name": "Main Book", "count": drawings.registry_count()},
            *(
                {"id": p["project_id"], "name": p["name"], "count": p["drawing_count"]}
                for p in projects.list_all()
            ),
        ]
    }


@router.get("/rows")
def registry_rows(drawings: Drawings, project_id: str | None = None):
    return {"rows": drawings.registry_rows(project_id)}


@router.patch("/rows/{drawing_id}")
def update_registry_row(drawing_id: str, body: RegistryRowUpdate, drawings: Drawings):
    return drawings.registry_update(drawing_id, body.model_dump(exclude_unset=True))


@router.post("/rows")
def create_registry_row(body: RegistryRowCreate, drawings: Drawings):
    return drawings.registry_create(body.model_dump(exclude_unset=True))


def _sheet_title(name: str, used: set[str]) -> str:
    """Excel sheet names: max 31 chars, no []:*?/\\ - and unique."""
    clean = re.sub(r"[\[\]:*?/\\]", "-", name).strip() or "Sheet"
    clean = clean[:31]
    title, n = clean, 2
    while title.lower() in used:
        suffix = f" ({n})"
        title = clean[: 31 - len(suffix)] + suffix
        n += 1
    used.add(title.lower())
    return title


def _write_sheet(ws, rows: list[dict]) -> None:
    ws.append(_BOOK_HEADER)
    for r in rows:
        ws.append([
            r["dwg_number"], r["sheet_count"], r["description"],
            r["contract_number"], r["drawing_date"], r["set_number"],
        ])
    # column widths tuned to the original book's proportions
    for col, width in zip("ABCDEF", (16, 10, 60, 24, 12, 8)):
        ws.column_dimensions[col].width = width


@router.get("/export")
def export_book(drawings: Drawings, projects: Projects, project_id: str | None = None):
    """Round-trip back to Excel: one tab exports a single sheet; no project_id
    exports the whole book (Main Book + a sheet per project)."""
    from openpyxl import Workbook  # deferred: only the export path needs it

    wb = Workbook()
    used: set[str] = set()
    if project_id:
        project = projects.get_detail(project_id)
        ws = wb.active
        ws.title = _sheet_title(project["name"], used)
        _write_sheet(ws, drawings.registry_rows(project_id))
    else:
        ws = wb.active
        ws.title = _sheet_title("Main Book", used)
        _write_sheet(ws, drawings.registry_rows(None))
        for p in projects.list_all():
            _write_sheet(
                wb.create_sheet(_sheet_title(p["name"], used)),
                drawings.registry_rows(p["project_id"]),
            )

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="Drawings Number Book.xlsx"'},
    )
