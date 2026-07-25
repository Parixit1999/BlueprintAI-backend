"""DWG support via conversion to DXF.

DWG is Autodesk's proprietary binary format. Two converters are supported,
tried in order:

1. ODA File Converter (free, closed-source, reference-quality - reads every
   DWG version including AutoCAD 2018+ AC1032) - used when
   ODA_CONVERTER_PATH is configured. The production image ships it (amd64).
2. LibreDWG's dwg2dxf (free, open-source, bundled in the docker image) -
   the fallback. Handles older DWG versions well, but cannot read AC1032
   and may drop entities, so its extractions carry an accuracy note.

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


def convert_to_dxf(path: str, out_dir: str) -> tuple[Path, str]:
    """Convert a DWG to DXF in out_dir, using the best available converter.

    Raises UnsupportedFileType when no converter exists, ExtractionFailed when
    conversion fails. Returns (produced DXF path, converter name) where the
    converter name is "oda" or "libredwg".
    """
    oda = settings.oda_converter_path
    if oda and shutil.which(oda):
        with tempfile.TemporaryDirectory() as in_dir:
            src = (Path(in_dir) / Path(path).name).with_suffix(".dwg")
            src.write_bytes(Path(path).read_bytes())
            # ODAFileConverter <in> <out> <outver> <outtype> <recurse> <audit>
            result = subprocess.run(
                [oda, in_dir, out_dir, "ACAD2018", "DXF", "0", "1"],
                capture_output=True, timeout=600,
            )
            produced = list(Path(out_dir).glob("*.dxf"))
            if result.returncode == 0 and produced:
                return produced[0], "oda"

    if shutil.which("dwg2dxf"):
        out = Path(out_dir) / (Path(path).stem + ".dxf")
        result = subprocess.run(
            ["dwg2dxf", "-o", str(out), path],
            capture_output=True, timeout=120,
        )
        # dwg2dxf can exit non-zero on recoverable warnings; a usable DXF is
        # the real success signal
        if out.exists() and out.stat().st_size > 0:
            return out, "libredwg"
        raise ExtractionFailed(
            "DWG conversion failed - the file may be corrupt or use a DWG "
            "version LibreDWG cannot read. Export it as DXF or PDF and re-upload."
        )

    raise UnsupportedFileType(GUIDANCE)


class DwgExtractor:
    def __init__(self, vision=None):
        # forwarded to the DXF extractor after conversion, so DWG drawings
        # get the same component-detection vision pass as native DXF
        self._vision = vision

    def extract(self, path: str) -> list[ProvisionalChunk]:
        with tempfile.TemporaryDirectory() as out_dir:
            dxf_path, converter = convert_to_dxf(path, out_dir)
            chunks = DxfExtractor(self._vision).extract(str(dxf_path))
        if converter == "oda":
            # reference-quality conversion - no accuracy caveat needed
            return chunks
        note = ProvisionalChunk(
            region_type=RegionType.note,
            chunk_text=ACCURACY_NOTE,
            bbox=None,
            confidence=Confidence.low,
            page=1,
            advisory=True,
        )
        return [note, *chunks]
