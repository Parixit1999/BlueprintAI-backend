"""Render DXF model space to PNG for the evidence viewer.

Returns the image bytes plus the model-space extents the image covers, so the
frontend can map chunk bboxes (model-space coords) onto image pixels linearly.
"""
import io

import matplotlib

matplotlib.use("Agg")

import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import BackgroundPolicy, Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

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
        raise ValueError("Drawing has no visible extents")
    fig.set_size_inches(MAX_WIDTH_INCHES, MAX_WIDTH_INCHES * height / width)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    return buf.getvalue(), [round(float(v), 3) for v in (xmin, ymin, xmax, ymax)]
