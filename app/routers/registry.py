"""The registry as a spreadsheet: the Drawings Number Book, live.

Engineers coming from the Excel book get the same mental model - sheet tabs
(Main Book + one per project), the same six columns, cell editing - but every
cell is backed by the real registry, so edits re-embed metadata cards and
show up in chat/search immediately.
"""
import io
import re
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import authz
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
def registry_tabs(request: Request, drawings: Drawings, projects: Projects):
    """Sheet tabs: Main Book (every drawing) then one tab per project,
    mirroring the workbook's structure. Role-restricted callers get only
    their sheets - and no Main Book, which by definition spans everything."""
    allowed = authz.allowed_project_ids(request.state.user)
    sheets = [
        {"id": p["project_id"], "name": p["name"], "count": p["drawing_count"]}
        for p in projects.list_all()
        if allowed is None or p["project_id"] in allowed
    ]
    if allowed is not None:
        return {"tabs": sheets}
    return {
        "tabs": [
            {"id": None, "name": "Main Book", "count": drawings.registry_count()},
            *sheets,
        ]
    }


@router.get("/rows")
def registry_rows(request: Request, drawings: Drawings, project_id: str | None = None):
    user = request.state.user
    authz.check_project(user, project_id)
    return {"rows": drawings.registry_rows(project_id, authz.allowed_project_ids(user))}


@router.get("/deleted")
def deleted_rows(request: Request, drawings: Drawings):
    """The book's recycle bin: rows removed from the sheets but recoverable."""
    return {"rows": drawings.list_deleted(authz.allowed_project_ids(request.state.user))}


@router.post("/rows/{drawing_id}/restore")
def restore_registry_row(request: Request, drawing_id: str, drawings: Drawings):
    authz.check_project(request.state.user, drawings.project_of_drawing(drawing_id))
    return drawings.restore(drawing_id)


@router.patch("/rows/{drawing_id}")
def update_registry_row(
    request: Request, drawing_id: str, body: RegistryRowUpdate, drawings: Drawings
):
    user = request.state.user
    fields = body.model_dump(exclude_unset=True)
    if authz.allowed_project_ids(user) is not None:
        # the row must already be on one of the caller's sheets...
        authz.check_project(user, drawings.project_of_drawing(drawing_id))
        # ...and may only move to another of them
        if "project_id" in fields:
            authz.check_project(user, fields["project_id"])
    return drawings.registry_update(drawing_id, fields)


@router.post("/rows")
def create_registry_row(request: Request, body: RegistryRowCreate, drawings: Drawings):
    user = request.state.user
    fields = body.model_dump(exclude_unset=True)
    if authz.allowed_project_ids(user) is not None:
        # restricted roles create rows on their sheets, never in the Main Book
        authz.check_project(user, fields.get("project_id"))
    return drawings.registry_create(fields)


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
def export_book(
    request: Request, drawings: Drawings, projects: Projects,
    project_id: str | None = None,
):
    """Round-trip back to Excel: one tab exports a single sheet; no project_id
    exports the whole book (Main Book + a sheet per project). A restricted
    role's whole-book export fans over ITS sheets only - no Main Book tab."""
    from openpyxl import Workbook  # deferred: only the export path needs it

    user = request.state.user
    allowed = authz.allowed_project_ids(user)
    wb = Workbook()
    used: set[str] = set()
    if project_id:
        authz.check_project(user, project_id)
        project = projects.get_detail(project_id)
        ws = wb.active
        ws.title = _sheet_title(project["name"], used)
        _write_sheet(ws, drawings.registry_rows(project_id))
    else:
        exportable = [
            p for p in projects.list_all()
            if allowed is None or p["project_id"] in allowed
        ]
        ws = wb.active
        if allowed is None:
            ws.title = _sheet_title("Main Book", used)
            _write_sheet(ws, drawings.registry_rows(None))
        elif exportable:
            first = exportable.pop(0)
            ws.title = _sheet_title(first["name"], used)
            _write_sheet(ws, drawings.registry_rows(first["project_id"]))
        else:
            ws.title = _sheet_title("No sheets", used)
            _write_sheet(ws, [])
        for p in exportable:
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
