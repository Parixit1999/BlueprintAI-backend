"""DWG support via conversion to DXF.

DWG is Autodesk's proprietary binary format. Two converters are supported,
tried in order:

1. ODA File Converter (free, closed-source, best fidelity) - used when
   ODA_CONVERTER_PATH is configured.
2. LibreDWG's dwg2dxf (free, open-source, bundled in the docker image) -
   the default. Handles most DWG versions well, but very new or complex
   drawings may lose some entities, so extractions carry an accuracy note.

After conversion the drawing flows through the normal DXF extractor, and
the viewer renders the converted DXF, so bboxes line up.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.config import settings
from app.exceptions import ExtractionFailed, UnsupportedFileType
from app.schemas import Confidence, ProvisionalChunk, RegionType
from app.services.extraction.dxf import DxfExtractor

GUIDANCE = (
    "DWG is a proprietary AutoCAD format and no converter is available in this "
    "deployment. Export the drawing as DXF (AutoCAD: SAVEAS > DXF) or PDF and "
    "upload that instead."
)

ACCURACY_NOTE = (
    "This drawing was converted from DWG to DXF with the open-source LibreDWG "
    "converter. Most drawings convert cleanly, but very new DWG versions or "
    "complex entities may be dropped in conversion - if anything looks "
    "incomplete, export the drawing as DXF or PDF from AutoCAD and re-upload "
    "for full accuracy."
)


def convert_to_dxf(path: str, out_dir: str) -> Path:
    """Convert a DWG to DXF in out_dir, using the best available converter.

    Raises UnsupportedFileType when no converter exists, ExtractionFailed when
    conversion fails. Returns the produced DXF path.
    """
    oda = settings.oda_converter_path
    if oda and shutil.which(oda):
        with tempfile.TemporaryDirectory() as in_dir:
            src = (Path(in_dir) / Path(path).name).with_suffix(".dwg")
            src.write_bytes(Path(path).read_bytes())
            # ODAFileConverter <in> <out> <outver> <outtype> <recurse> <audit>
            result = subprocess.run(
                [oda, in_dir, out_dir, "ACAD2018", "DXF", "0", "1"],
                capture_output=True, timeout=120,
            )
            produced = list(Path(out_dir).glob("*.dxf"))
            if result.returncode == 0 and produced:
                return produced[0]

    if shutil.which("dwg2dxf"):
        out = Path(out_dir) / (Path(path).stem + ".dxf")
        result = subprocess.run(
            ["dwg2dxf", "-o", str(out), path],
            capture_output=True, timeout=120,
        )
        # dwg2dxf can exit non-zero on recoverable warnings; a usable DXF is
        # the real success signal
        if out.exists() and out.stat().st_size > 0:
            return out
        raise ExtractionFailed(
            "DWG conversion failed - the file may be corrupt or use a DWG "
            "version LibreDWG cannot read. Export it as DXF or PDF and re-upload."
        )

    raise UnsupportedFileType(GUIDANCE)


class DwgExtractor:
    def extract(self, path: str) -> list[ProvisionalChunk]:
        with tempfile.TemporaryDirectory() as out_dir:
            dxf_path = convert_to_dxf(path, out_dir)
            chunks = DxfExtractor().extract(str(dxf_path))
        note = ProvisionalChunk(
            region_type=RegionType.note,
            chunk_text=ACCURACY_NOTE,
            bbox=None,
            confidence=Confidence.low,
            page=1,
            advisory=True,
        )
        return [note, *chunks]
