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
from PIL import Image

from app.exceptions import RenderFailed

MAX_WIDTH_INCHES = 12
DPI = 150


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
    png = pdf_page.get_pixmap(dpi=DPI).tobytes("png")
    # extents in PDF points, y-up (extractor bboxes are flipped to match)
    return png, [0.0, 0.0, round(pdf_page.rect.width, 3), round(pdf_page.rect.height, 3)]


def render_image(path: str) -> tuple[bytes, list[float]]:
    with Image.open(path) as img:
        width, height = img.size
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue(), [0.0, 0.0, float(width), float(height)]
