"""DWG support via external conversion.

DWG is Autodesk's proprietary binary format with no reliable open-source
parser. The industry-standard free tool is the ODA File Converter
(https://www.opendesign.com/guestfiles/oda_file_converter): when its path is
configured (ODA_CONVERTER_PATH), DWG uploads are converted to DXF on the fly
and flow through the normal DXF extractor. Without it, the user gets precise
guidance instead of a generic failure.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.config import settings
from app.exceptions import ExtractionFailed, UnsupportedFileType
from app.schemas import ProvisionalChunk
from app.services.extraction.dxf import DxfExtractor

GUIDANCE = (
    "DWG is a proprietary AutoCAD format. To ingest it directly, install the free "
    "ODA File Converter and set ODA_CONVERTER_PATH; otherwise export the drawing "
    "as DXF (AutoCAD: SAVEAS > DXF) or PDF and upload that instead."
)


class DwgExtractor:
    def extract(self, path: str) -> list[ProvisionalChunk]:
        converter = settings.oda_converter_path
        if not converter or not shutil.which(converter):
            raise UnsupportedFileType(GUIDANCE)

        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            src = Path(in_dir) / Path(path).name
            src = src.with_suffix(".dwg")
            src.write_bytes(Path(path).read_bytes())
            # ODAFileConverter <in> <out> <outver> <outtype> <recurse> <audit>
            result = subprocess.run(
                [converter, in_dir, out_dir, "ACAD2018", "DXF", "0", "1"],
                capture_output=True, timeout=120,
            )
            produced = list(Path(out_dir).glob("*.dxf"))
            if result.returncode != 0 or not produced:
                raise ExtractionFailed(
                    "DWG conversion failed - the file may be corrupt or use an "
                    "unsupported DWG version. Export it as DXF or PDF and re-upload."
                )
            return DxfExtractor().extract(str(produced[0]))
