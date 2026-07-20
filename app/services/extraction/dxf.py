"""DXF extraction via ezdxf - structured read, so confidence is high and
bboxes are exact model-space coordinates. Null values are flagged
low-confidence for the review UI, never guessed.
"""
import ezdxf
from ezdxf import bbox as ezbbox

from app.schemas import Confidence, ProvisionalChunk, RegionType


def _entity_bbox(entity) -> list[float] | None:
    try:
        extents = ezbbox.extents([entity], fast=True)
        if extents.has_data:
            (x1, y1, _), (x2, y2, _) = extents.extmin, extents.extmax
            return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]
    except Exception:
        pass
    return None


class DxfExtractor:
    def extract(self, path: str) -> list[ProvisionalChunk]:
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        chunks: list[ProvisionalChunk] = []

        for text in msp.query("TEXT MTEXT"):
            content = (
                text.plain_text() if text.dxftype() == "MTEXT" else text.dxf.text
            ).strip()
            if not content:
                continue
            chunks.append(
                ProvisionalChunk(
                    region_type=RegionType.note,
                    chunk_text=content,
                    bbox=_entity_bbox(text),
                    confidence=Confidence.high,
                )
            )

        for dim in msp.query("DIMENSION"):
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
                )
            )

        return chunks
