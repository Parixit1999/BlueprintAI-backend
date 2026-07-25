"""DXF extraction via ezdxf - structured read, so confidence is high and
bboxes are exact model-space coordinates. Null values are flagged
low-confidence for the review UI, never guessed.

Sheets: professional CAD files often keep the modelspace empty and store
every plottable sheet in a paperspace layout, so extraction walks the same
sheet list the renderer exposes (modelspace when non-empty + every
non-empty layout) and stamps chunks with the sheet's page number - the
multi-page viewer then works exactly like it does for PDFs.

Text harvest recurses into block INSERTs (including attributes): title
blocks and even entire sheets are commonly one INSERT whose text a
top-level query would never see.

The entity read never SEES the drawing, so it cannot detect drawn
components (stairs, valves, ducts...) or describe the sheet. When a vision
provider is supplied, each sheet is rendered (with the same renderer the
viewer uses, so bboxes line up) and a vision pass adds component groups
and a summary - text regions from vision are dropped because the entity
read already has them with exact coordinates. Best-effort: a render or
vision failure keeps the entity extraction intact.
"""
import logging

import ezdxf
from ezdxf import bbox as ezbbox

from app.exceptions import InvalidFile
from app.schemas import Confidence, ProvisionalChunk, RegionType
from app.services.extraction.image import ImageExtractor

logger = logging.getLogger(__name__)

# Vision output kept for CAD files: what the entity read cannot provide.
_VISION_KEEP = {RegionType.component, RegionType.summary}

# Vision cost ceiling for huge sheet sets; sheets beyond it still get full
# entity text extraction, and an advisory notes the limit.
_MAX_VISION_SHEETS = 8

# Block nesting depth for the INSERT text harvest (sheets-in-a-block are one
# level; title blocks inside them one more; deeper nesting is exotic).
_MAX_INSERT_DEPTH = 4


def _entity_bbox(entity) -> list[float] | None:
    try:
        extents = ezbbox.extents([entity], fast=True)
        if extents.has_data:
            (x1, y1, _), (x2, y2, _) = extents.extmin, extents.extmax
            return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]
    except Exception:
        pass
    return None


def _iter_text_entities(container, depth: int = _MAX_INSERT_DEPTH):
    """Yield TEXT/MTEXT/ATTRIB entities, recursing through block INSERTs.

    virtual_entities() yields transformed copies, so bboxes computed on
    them are in the container's own coordinate space.
    """
    for entity in container:
        kind = entity.dxftype()
        if kind in ("TEXT", "MTEXT"):
            yield entity
        elif kind == "INSERT" and depth > 0:
            yield from entity.attribs
            try:
                yield from _iter_text_entities(
                    entity.virtual_entities(), depth - 1
                )
            except Exception:
                # a malformed block reference must not sink the whole sheet
                continue


def _text_content(entity) -> str:
    if entity.dxftype() == "MTEXT":
        return entity.plain_text()
    return entity.dxf.text  # TEXT and ATTRIB


def _pct_to_model_space(pct: list[float], extents: list[float]) -> list[float]:
    """Map a top-left-origin percentage bbox onto DXF model-space extents.

    Unlike images/PDFs, DXF extents rarely start at (0, 0) - the drawing
    lives wherever the drafter put it - so the offset matters.
    """
    xmin, ymin, xmax, ymax = extents
    w, h = xmax - xmin, ymax - ymin
    x1, y1, x2, y2 = pct
    return [
        round(xmin + x1 / 100 * w, 3),
        round(ymin + (1 - y2 / 100) * h, 3),
        round(xmin + x2 / 100 * w, 3),
        round(ymin + (1 - y1 / 100) * h, 3),
    ]


class DxfExtractor:
    def __init__(self, vision: ImageExtractor | None = None):
        self._vision = vision

    def _sheet_text_chunks(self, layout, page: int) -> list[ProvisionalChunk]:
        chunks: list[ProvisionalChunk] = []
        for text in _iter_text_entities(layout):
            content = _text_content(text).strip()
            if not content:
                continue
            chunks.append(
                ProvisionalChunk(
                    region_type=RegionType.note,
                    chunk_text=content,
                    bbox=_entity_bbox(text),
                    confidence=Confidence.high,
                    page=page,
                )
            )

        for dim in layout.query("DIMENSION"):
            try:
                measurement = dim.get_measurement()
            except Exception:
                measurement = None
            override = (dim.dxf.text or "").strip()
            if override and override != "<>":
                value = override
            elif isinstance(measurement, (int, float)):
                value = f"{round(measurement, 4)}"
            else:
                value = None
            chunks.append(
                ProvisionalChunk(
                    region_type=RegionType.dimension,
                    chunk_text=f"Dimension: {value}" if value else None,
                    bbox=_entity_bbox(dim),
                    confidence=Confidence.high if value else Confidence.low,
                    page=page,
                )
            )
        return chunks

    def _vision_sheet_chunks(self, doc, sheet_name: str, page: int) -> list[ProvisionalChunk]:
        """Render one sheet and vision-detect components + summary.

        Best-effort by design: CAD text extraction is already complete and
        exact, so a failure here must never fail the file.
        """
        if self._vision is None:
            return []
        try:
            from app.services.rendering import render_dxf_layout

            png, extents = render_dxf_layout(doc, sheet_name)
            # skip Textract: every piece of text is already extracted from
            # entities with exact coordinates - OCR would only add cost
            regions = self._vision.analyze(png, ocr_lines=[])
        except Exception:
            logger.warning(
                "CAD vision pass failed for sheet %r", sheet_name, exc_info=True
            )
            return []
        chunks: list[ProvisionalChunk] = []
        for region in regions:
            if region.region_type not in _VISION_KEEP:
                continue
            # text-dense sheets can render near-blank at vision resolution;
            # when the model saw nothing drawing-like, its summary describes
            # a blank image - worse than no summary at all
            if region.region_type == RegionType.summary and region.is_drawing is False:
                continue
            bbox = (
                _pct_to_model_space(region.bbox_pct, extents)
                if region.bbox_pct
                else None
            )
            extra = (
                [_pct_to_model_space(b, extents) for b in region.extra_bboxes_pct]
                if region.extra_bboxes_pct
                else None
            )
            chunks.append(
                ProvisionalChunk(
                    region_type=region.region_type,
                    chunk_text=region.text,
                    bbox=bbox,
                    confidence=region.confidence,
                    extra_bboxes=extra,
                    page=page,
                    # a parsed CAD file IS a drawing - don't let a vision
                    # misread of the rendered image say otherwise
                    is_drawing=True if region.region_type == RegionType.summary else None,
                )
            )
        return chunks

    def extract(self, path: str) -> list[ProvisionalChunk]:
        try:
            doc = ezdxf.readfile(path)
        except Exception:
            # ezdxf raises DXFError subclasses but also IO/decode errors on garbage input
            raise InvalidFile(
                "This file could not be read as a DXF drawing - it appears to be "
                "corrupt or is not really a DXF file."
            )
        from app.services.rendering import dxf_sheet_names

        sheets = dxf_sheet_names(doc)
        vision_chunks: list[ProvisionalChunk] = []
        text_chunks: list[ProvisionalChunk] = []
        for page, name in enumerate(sheets, start=1):
            layout = doc.modelspace() if name == "Model" else doc.layout(name)
            if name != "Model":
                # the layout tab name is real metadata - drafters name sheets
                # by number (A191, C101...) and users search by it
                text_chunks.append(
                    ProvisionalChunk(
                        region_type=RegionType.note,
                        chunk_text=f"Sheet {name}",
                        bbox=None,
                        confidence=Confidence.high,
                        page=page,
                    )
                )
            text_chunks.extend(self._sheet_text_chunks(layout, page))
            if page <= _MAX_VISION_SHEETS:
                vision_chunks.extend(self._vision_sheet_chunks(doc, name, page))

        if self._vision is not None and len(sheets) > _MAX_VISION_SHEETS:
            vision_chunks.append(
                ProvisionalChunk(
                    region_type=RegionType.note,
                    chunk_text=(
                        f"Component detection ran on the first {_MAX_VISION_SHEETS} "
                        f"of {len(sheets)} sheets; text extraction covers all sheets."
                    ),
                    bbox=None,
                    confidence=Confidence.high,
                    page=1,
                    advisory=True,
                )
            )

        # summary + component groups lead, matching the other extraction paths
        return [*vision_chunks, *text_chunks]
