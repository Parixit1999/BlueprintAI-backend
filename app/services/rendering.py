"""Render drawings to PNG for the evidence viewer.

Every renderer returns (png_bytes, extents) where extents [xmin, ymin, xmax,
ymax] describe the coordinate space of the image in the same y-up convention
the extractors use, so the frontend maps chunk bboxes linearly.
"""
import io

import matplotlib

matplotlib.use("Agg")

import ezdxf
import matplotlib.pyplot as plt
import pymupdf
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import BackgroundPolicy, Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from app.exceptions import RenderFailed

# HEIC/HEIF renders through the normal image path once Pillow can open it
register_heif_opener()

MAX_WIDTH_INCHES = 12
DPI = 150
# Render derivatives are for VIEWING - the original in object storage keeps
# full fidelity. JPEG for continuous-tone scans (5-10x smaller than PNG =
# 5-10x faster loads); PNG stays for line art (DXF renders), where JPEG
# artifacts would fuzz thin lines. Long side capped so a huge sheet's
# derivative stays a fast download.
JPEG_QUALITY = 80
MAX_RENDER_SIDE = 3000


def render_dxf(path: str) -> tuple[bytes, list[float]]:
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    backend = MatplotlibBackend(ax)
    config = Configuration(background_policy=BackgroundPolicy.WHITE)
    Frontend(ctx, backend, config=config).draw_layout(msp, finalize=True)

    (xmin, xmax), (ymin, ymax) = ax.get_xlim(), ax.get_ylim()
    width, height = xmax - xmin, ymax - ymin
    if width <= 0 or height <= 0:
        plt.close(fig)
        raise RenderFailed("Drawing has no visible extents")
    fig.set_size_inches(MAX_WIDTH_INCHES, MAX_WIDTH_INCHES * height / width)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    return buf.getvalue(), [round(float(v), 3) for v in (xmin, ymin, xmax, ymax)]


def render_pdf_page(path: str, page: int) -> tuple[bytes, list[float]]:
    doc = pymupdf.open(path)
    if not 1 <= page <= len(doc):
        raise RenderFailed(f"Page {page} does not exist (document has {len(doc)} pages)")
    pdf_page = doc[page - 1]
    # adaptive DPI: huge sheets render at whatever DPI keeps the long side
    # under the cap, instead of producing a 12000px derivative
    long_side_pts = max(pdf_page.rect.width, pdf_page.rect.height)
    dpi = min(DPI, int(MAX_RENDER_SIDE / (long_side_pts / 72)) or DPI)
    jpg = pdf_page.get_pixmap(dpi=dpi).tobytes("jpeg", jpg_quality=JPEG_QUALITY)
    # extents in PDF points, y-up (extractor bboxes are flipped to match)
    return jpg, [0.0, 0.0, round(pdf_page.rect.width, 3), round(pdf_page.rect.height, 3)]


def render_image(path: str) -> tuple[bytes, list[float]]:
    with Image.open(path) as img:
        # same EXIF-orientation normalization as extraction, so vision bboxes
        # line up with the preview
        img = ImageOps.exif_transpose(img)
        width, height = img.size  # extents stay in ORIGINAL pixels
        render = img.convert("RGB")
        if max(render.size) > MAX_RENDER_SIDE:
            render.thumbnail((MAX_RENDER_SIDE, MAX_RENDER_SIDE))
        buf = io.BytesIO()
        render.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue(), [0.0, 0.0, float(width), float(height)]
