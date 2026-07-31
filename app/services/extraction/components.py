"""Drawn-component detection: find them all, box them tightly, name them.

One whole-sheet vision pass is structurally incapable of finding every
component. The model reads a 1568px image; an E-size sheet is 34x44 inches, so
a 1/2" valve symbol arrives as roughly two pixels. Asked to be exhaustive at
that resolution the model does the only thing it can: it reports the big
obvious things, and lumps the rest into a few enormous boxes - one rectangle
covering a pump, its housekeeping pad and half its piping, labelled "equipment".
That is the behaviour this module exists to fix, and it fixes it by ZOOMING IN
rather than by asking the same question louder.

Three sources of detections, merged:

1. the overview pass (whole sheet, cheap, sees layout and large elements)
2. a SPLIT pass - any box big enough to be a group gets cropped from the
   full-resolution image and re-read, so the elements inside come back
   individually boxed
3. a TILE sweep - the sheet is cut into overlapping tiles, each read at
   near-native resolution, which is the only way small components are visible
   at all

Every zoomed pass carries the sheet's context (discipline, title, drawing
number, what the overview said the sheet shows) because a crop out of context
is unidentifiable - a circle with two triangles is meaningless until you know
you are looking at a plumbing riser.

Cost is the constraint, so nothing here is unconditional: passes are gated on
evidence that they will pay (a box that looks like a group; a sheet whose
resolution the overview actually lost) and share one per-sheet call budget
(`component_detail_calls`). Set that budget to 0 and this module still earns
its place - tightening boxes to their ink, merging duplicates and resolving
keynotes cost nothing but arithmetic.
"""
import io
import json
import logging
import re
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.schemas import Confidence, RegionType
from app.services.extraction import callouts as callout_util

logger = logging.getLogger(__name__)

# Claude vision reads up to ~1568px on the long edge. Shared with image.py:
# every image handed to the model - whole sheet, tile or crop - goes through
# downscale_for_vision, so returned pixel coordinates always map back.
MAX_VISION_SIDE = 1568
# Bedrock's 5 MB image limit applies to the BASE64-encoded payload (raw x 4/3),
# so raw bytes must stay under ~3.9 MB; keep margin.
MAX_VISION_BYTES = 3_600_000


# --------------------------------------------------------------------------
# geometry - everything is [x1, y1, x2, y2] in 0-100 percentages of the sheet,
# top-left origin, the same space VisionRegion.bbox_pct uses
# --------------------------------------------------------------------------


def to_pct(values, sent_width: int, sent_height: int) -> list[float] | None:
    """Normalize model coords to 0-100 percentages.

    Vision models ignore coordinate instructions often enough that we detect
    the scale: fractions (0-1), percentages (0-100), or absolute pixels of the
    (downscaled) image we sent.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in values)
    except (TypeError, ValueError):
        return None
    if any(v != v for v in (x1, y1, x2, y2)):  # NaN survives float() and poisons sorts
        return None
    peak = max(x1, y1, x2, y2)
    if peak <= 1:
        x1, y1, x2, y2 = (v * 100 for v in (x1, y1, x2, y2))
    elif peak <= 100:
        pass
    else:
        # absolute pixel coords of the sent image, per axis
        x1, x2 = (v / sent_width * 100 for v in (x1, x2))
        y1, y2 = (v / sent_height * 100 for v in (y1, y2))
    vals = [min(max(v, 0.0), 100.0) for v in (x1, y1, x2, y2)]
    x1, y1, x2, y2 = vals
    # models sometimes emit corners in reverse order - normalize, don't reject
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 - x1 < 0.1 or y2 - y1 < 0.1:
        return None  # degenerate box
    return [x1, y1, x2, y2]


def area_pct(box: list[float]) -> float:
    """Share of the sheet a box covers, in percent."""
    return (box[2] - box[0]) * (box[3] - box[1]) / 100.0


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    )
    return intersection / union if union > 0 else 0.0


def contains(outer: list[float], inner: list[float], slack: float = 1.0) -> bool:
    """Whether `inner` sits inside `outer`, with a little slack for the model's
    imprecision at the edges."""
    return (
        outer[0] - slack <= inner[0]
        and outer[1] - slack <= inner[1]
        and outer[2] + slack >= inner[2]
        and outer[3] + slack >= inner[3]
    )


def child_to_sheet(child: list[float], crop: list[float]) -> list[float]:
    """Map a box measured inside a crop back onto the whole sheet.

    `crop` is the region of the sheet the crop covers, in sheet percentages;
    `child` is in percentages OF THE CROP. Getting this wrong puts every
    zoomed detection in the wrong place, so it is deliberately one expression.
    """
    cx1, cy1, cx2, cy2 = crop
    width, height = cx2 - cx1, cy2 - cy1
    return [
        round(cx1 + child[0] / 100 * width, 3),
        round(cy1 + child[1] / 100 * height, 3),
        round(cx1 + child[2] / 100 * width, 3),
        round(cy1 + child[3] / 100 * height, 3),
    ]


# --------------------------------------------------------------------------
# detections
# --------------------------------------------------------------------------

# Source ranking: a detection made on a zoomed crop saw real pixels, so its box
# and its identification both beat the overview's guess at the same object.
SOURCE_OVERVIEW = "overview"
SOURCE_SPLIT = "split"
SOURCE_TILE = "tile"
_ZOOMED = {SOURCE_SPLIT, SOURCE_TILE}


@dataclass
class Detection:
    """One component instance, wherever it was found."""

    label: str | None
    box: list[float]
    confidence: Confidence = Confidence.medium
    source: str = SOURCE_OVERVIEW
    callout: str | None = None
    keynote: str | None = None
    # set when an independent pass found the same object in the same place
    corroborated: bool = False
    # the box was replaced by its own ink extent, i.e. it is pixel-tight
    tightened: bool = False
    # area of the box AS THE MODEL REPORTED IT. Tightening shrinks boxes on
    # purpose, so the size-based confidence cap has to judge what the model
    # actually looked at - otherwise improving a box would lower its
    # confidence, which is exactly backwards.
    read_area: float | None = None
    # still group-sized after every attempt to break it apart
    suspect: bool = False
    # superseded by the components a split pass found inside it
    dropped: bool = False

    @property
    def zoomed(self) -> bool:
        return self.source in _ZOOMED


# Confidence as an ordinal, so evidence can move a detection up or down it.
_LEVEL = {Confidence.low: 0, Confidence.medium: 1, Confidence.high: 2}
_BY_LEVEL = {0: Confidence.low, 1: Confidence.medium, 2: Confidence.high}


@dataclass
class SheetContext:
    """What the overview pass learned about the sheet, handed to every zoomed
    pass so a crop can be identified in context rather than in isolation."""

    discipline: str | None = None
    sheet_title: str | None = None
    drawing_number: str | None = None
    summary: str | None = None

    def block(self) -> str:
        lines = []
        if self.discipline:
            lines.append(f"- discipline: {self.discipline}")
        if self.sheet_title:
            lines.append(f"- sheet title: {self.sheet_title}")
        if self.drawing_number:
            lines.append(f"- drawing number: {self.drawing_number}")
        if self.summary:
            lines.append(f"- what this sheet shows: {self.summary[:600]}")
        if not lines:
            return ""
        return "Context for the sheet this crop was taken from:\n" + "\n".join(lines)


@dataclass
class ComponentResult:
    components: list[Detection] = field(default_factory=list)
    callouts: list[Detection] = field(default_factory=list)
    calls_used: int = 0


# Labels that identify nothing. The model is told never to use them; when it
# does anyway, the component goes to the naming pass instead of reaching a
# reviewer as a row that says "component".
_GENERIC_LABELS = {
    "component",
    "components",
    "part",
    "parts",
    "object",
    "item",
    "element",
    "equipment",
    "unknown",
    "unidentified",
    "symbol",
    "shape",
    "detail",
    "feature",
    "n/a",
    "none",
    "null",
}


def is_usable_label(label: str | None) -> bool:
    """Whether a label actually tells a reader what the thing IS."""
    if not label:
        return False
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", label.casefold()).strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned or cleaned in _GENERIC_LABELS:
        return False
    # a bare number is a callout, not an identification
    if cleaned.replace(" ", "").isdigit():
        return False
    return len(cleaned) >= 3


def normalize_label(label: str | None) -> str:
    """Grouping key: same component type, however the model phrased it."""
    if not label:
        return ""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", label.casefold())
    return " ".join(cleaned.split())


# --------------------------------------------------------------------------
# vision plumbing
# --------------------------------------------------------------------------


def downscale_for_vision(data: bytes) -> tuple[bytes, int, int]:
    """Send the model a bounded, known-size image so absolute pixel
    coordinates in its output can be mapped back reliably. Dense scans at
    1568px can exceed the provider's byte limit as PNG - fall back to JPEG
    (visually lossless for scans), shrinking further only if needed."""
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        return encode_for_vision(img)


def encode_for_vision(img: "Image.Image") -> tuple[bytes, int, int]:
    """The same bounding applied to an already-open image (tiles and crops,
    which are produced in memory and never round-trip through bytes)."""
    # copy first: thumbnail() resizes IN PLACE, and callers pass images they
    # still need at full resolution (the sheet is cropped many times)
    img = img.convert("RGB").copy()
    if max(img.size) > MAX_VISION_SIDE:
        img.thumbnail((MAX_VISION_SIDE, MAX_VISION_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    if buf.tell() <= MAX_VISION_BYTES:
        return buf.getvalue(), img.width, img.height
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        if buf.tell() <= MAX_VISION_BYTES or min(img.size) < 400:
            return buf.getvalue(), img.width, img.height
        img = img.resize((int(img.width * 0.85), int(img.height * 0.85)), Image.LANCZOS)


def crop_pct(
    img: "Image.Image", box: list[float], margin_pct: float = 0.0, min_side_px: int = 96
) -> tuple["Image.Image", list[float]]:
    """Crop the sheet region a box covers, returning the crop AND the sheet
    percentages it actually spans.

    The returned box is what child coordinates are mapped against, so it must
    reflect the margin and the clamping, not what was asked for.
    """
    width, height = img.size
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin_pct / 100, (y2 - y1) * margin_pct / 100
    x1, y1 = max(0.0, x1 - mx), max(0.0, y1 - my)
    x2, y2 = min(100.0, x2 + mx), min(100.0, y2 + my)

    px1, py1 = int(x1 / 100 * width), int(y1 / 100 * height)
    px2, py2 = int(round(x2 / 100 * width)), int(round(y2 / 100 * height))
    # a hairline box still has to produce an image the model can read
    if px2 - px1 < min_side_px:
        pad = (min_side_px - (px2 - px1)) // 2 + 1
        px1, px2 = max(0, px1 - pad), min(width, px2 + pad)
    if py2 - py1 < min_side_px:
        pad = (min_side_px - (py2 - py1)) // 2 + 1
        py1, py2 = max(0, py1 - pad), min(height, py2 + pad)
    px2, py2 = max(px1 + 1, px2), max(py1 + 1, py2)

    actual = [
        px1 / width * 100,
        py1 / height * 100,
        px2 / width * 100,
        py2 / height * 100,
    ]
    return img.crop((px1, py1, px2, py2)), actual


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

# Shared by every zoomed pass. Two things in here are load-bearing:
# "one entry per DISTINCT instance" (the overview pass's grouping is what
# produced the giant boxes) and the confidence rubric, which explicitly
# decouples confidence from SIZE - the old prompt said to mark anything small
# as low, which is exactly backwards for a pass whose whole purpose is small
# components read at high resolution.
DETAIL_PROMPT = """You are looking at a CROP of a larger engineering drawing, not a whole sheet.
The crop is {width}x{height} pixels.

{context}

Identify every DRAWN COMPONENT visible in this crop - the physical elements the
drawing depicts: valves, pumps, fittings, pipes, ducts, dampers, stairs, doors,
windows, walls, beams, columns, footings, fixtures, drains, panels, outlets,
tanks, major equipment. Name each from its drawn shape, its standard
symbology, and the sheet context above - not merely from text next to it.

Rules:
- ONE entry per DISTINCT component instance. Two valves = two entries.
- NEVER draw one box around several different components. If you are tempted
  to box an assembly, emit its identifiable parts separately instead.
- Box each component TIGHTLY around the symbol itself - not the white space
  around it, not its leader line, not its dimension.
- Small components matter as much as large ones. Size is NEVER a reason to
  skip a component or to lower its confidence.
- "label" must be a specific engineer's noun phrase: "gate valve", "floor
  drain", "W12x26 beam", "single-swing door". NEVER "component", "part",
  "object", "equipment", "unknown", or a bare number.
- If a numbered callout bubble points at a component, put that number in
  "callout".
- Ignore dimensions, notes and title-block text; another pass reads those.

Return ONLY this JSON object:
{"components": [{"label": "...", "bbox_pct": [x1, y1, x2, y2], "callout": "3" or null, "confidence": "high"}]}

bbox_pct is [x1, y1, x2, y2] in PIXELS of this {width}x{height} crop, measured
from its TOP-LEFT corner.

confidence describes how sure you are of the IDENTIFICATION, never how small
the component is:
- "high": the symbol is clear and you can name the element confidently.
- "medium": you can name it, but the symbol is ambiguous or partly obscured.
- "low": you can see something is drawn there but genuinely cannot say what.
No prose outside the JSON object."""


# Set-of-mark naming: rather than one call per unnamed component, their crops
# are composed into a single numbered contact sheet and named in one request.
# Twelve components cost one call instead of twelve.
NAMING_PROMPT = """This image is a contact sheet: {count} numbered panels, each a close-up crop of
ONE component taken from the same engineering drawing. The number in the corner
of each panel is its panel number.

{context}

For each panel, say what the component IS, as an engineer would label it on a
schedule: "gate valve", "roof drain", "grab bar", "W12x26 beam", "supply
diffuser", "cleanout". Judge from the drawn shape, standard symbology, and the
sheet context above.

Never answer "component", "part", "object", "equipment", "unknown" or a bare
number. If a panel is genuinely unreadable, use null for that panel.

Return ONLY this JSON object, with one key per panel number:
{"labels": {"1": "gate valve", "2": null}, "confidence": {"1": "high", "2": "low"}}

confidence describes how sure you are of the identification, not how small the
component is. No prose outside the JSON object."""


def _strip_fence(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw)
    return raw.strip()


def parse_detail(raw: str, sent_w: int, sent_h: int) -> list[dict]:
    """Components out of a detail-pass response. Tolerant by design: a zoomed
    pass is an enhancement, so malformed output loses its findings rather than
    failing the extraction."""
    text = _strip_fence(raw)
    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None
    if isinstance(payload, dict):
        items = payload.get("components")
        if not isinstance(items, list):
            items = payload.get("regions") if isinstance(payload.get("regions"), list) else []
    elif isinstance(payload, list):
        items = payload
    else:
        return []

    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        box = to_pct(item.get("bbox_pct") or item.get("bbox") or [], sent_w, sent_h)
        if box is None:
            continue
        confidence = item.get("confidence")
        out.append(
            {
                "label": (str(item["label"]).strip() if item.get("label") else None),
                "box": box,
                "confidence": (
                    Confidence(confidence)
                    if confidence in {c.value for c in Confidence}
                    else Confidence.medium
                ),
                "callout": callout_util.normalize_callout(item.get("callout")),
            }
        )
    return out


def parse_names(raw: str) -> dict[str, tuple[str | None, Confidence | None]]:
    """{panel number -> (label, confidence)} from the naming pass."""
    text = _strip_fence(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, dict):
        return {}
    labels = payload.get("labels")
    # some models answer with the flat {"1": "gate valve"} shape
    if not isinstance(labels, dict):
        labels = {k: v for k, v in payload.items() if isinstance(v, (str, type(None)))}
    confidences = payload.get("confidence")
    if not isinstance(confidences, dict):
        confidences = {}
    out: dict[str, tuple[str | None, Confidence | None]] = {}
    for key, value in labels.items():
        index = re.sub(r"\D", "", str(key))
        if not index:
            continue
        level = confidences.get(key) or confidences.get(str(index))
        out[str(int(index))] = (
            str(value).strip() if isinstance(value, str) and value.strip() else None,
            Confidence(level) if level in {c.value for c in Confidence} else None,
        )
    return out


# --------------------------------------------------------------------------
# box tightening - free accuracy, no model call
# --------------------------------------------------------------------------

# Floors that stop tightening from collapsing a box onto a speck. Both are
# deliberately loose, because the boxes most worth fixing are the worst ones:
# a box nine times too large in each axis holds ink across barely 1% of its
# area, and an area floor strict enough to "feel safe" would decline to fix
# exactly the grossly-oversized boxes this exists for. What actually keeps
# tightening honest is that getbbox() spans ALL the ink in the box - it can
# only land on a speck if the box contained nothing but that speck, in which
# case the box was already pointing at the wrong thing.
_MIN_TIGHTEN_RATIO = 0.005
_MIN_TIGHTEN_PX = 6
# Grown back around the ink so strokes are not clipped at the boundary.
_TIGHTEN_MARGIN = 0.04


def tighten_to_ink(img: "Image.Image", box: list[float]) -> list[float] | None:
    """Shrink a box to the drawn ink inside it.

    A vision model's box is a gesture at where something is; on line art the
    actual extent of the symbol is exactly computable - it is where the pixels
    stop being paper. This costs a crop and a threshold, and it is the single
    cheapest accuracy improvement available to component boxes.

    Returns the tightened box, or None when tightening would be unsafe
    (blank crop, ink filling the whole box, or a collapse onto a speck).
    """
    try:
        width, height = img.size
        px1, py1 = int(box[0] / 100 * width), int(box[1] / 100 * height)
        px2, py2 = int(round(box[2] / 100 * width)), int(round(box[3] / 100 * height))
        if px2 - px1 < 4 or py2 - py1 < 4:
            return None
        crop = img.crop((px1, py1, px2, py2)).convert("L")
        # invert so ink is bright, then threshold adaptively: a faint scan's
        # darkest ink may only reach mid-grey, and a fixed cut would erase it
        inverted = ImageOps.invert(crop)
        peak = inverted.getextrema()[1]
        if peak < 24:
            return None  # blank paper - nothing to tighten onto
        threshold = max(24, int(peak * 0.45))
        mask = inverted.point(lambda v, t=threshold: 255 if v >= t else 0)
        extent = mask.getbbox()
        if extent is None:
            return None
        ex1, ey1, ex2, ey2 = extent
        crop_w, crop_h = px2 - px1, py2 - py1
        # ink spans the whole crop: either the box is already tight or the
        # component sits on hatching - either way, changing nothing is right
        if (ex2 - ex1) >= crop_w * 0.98 and (ey2 - ey1) >= crop_h * 0.98:
            return None
        if (ex2 - ex1) < _MIN_TIGHTEN_PX or (ey2 - ey1) < _MIN_TIGHTEN_PX:
            return None
        if (ex2 - ex1) * (ey2 - ey1) < crop_w * crop_h * _MIN_TIGHTEN_RATIO:
            return None
        mx, my = (ex2 - ex1) * _TIGHTEN_MARGIN, (ey2 - ey1) * _TIGHTEN_MARGIN
        return [
            round(max(0.0, (px1 + ex1 - mx) / width * 100), 3),
            round(max(0.0, (py1 + ey1 - my) / height * 100), 3),
            round(min(100.0, (px1 + ex2 + mx) / width * 100), 3),
            round(min(100.0, (py1 + ey2 + my) / height * 100), 3),
        ]
    except Exception:  # a tightening failure must never cost the detection
        logger.debug("box tightening failed", exc_info=True)
        return None


# --------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------

# Two boxes describing the same object. Deliberately moderate: tiles overlap,
# so the SAME valve seen from two tiles lands at slightly different
# coordinates, and a threshold too high would report it twice.
_SAME_OBJECT_IOU = 0.45
# An overview box is discarded as a failed grouping when it swallows at least
# this many zoomed detections - that is the giant-box case, now solved.
_GROUP_CHILD_COUNT = 2


def _better_label(a: str | None, b: str | None) -> str | None:
    """The more informative of two labels for the same object."""
    a_ok, b_ok = is_usable_label(a), is_usable_label(b)
    if a_ok and not b_ok:
        return a
    if b_ok and not a_ok:
        return b
    if not a_ok and not b_ok:
        return a or b
    # both usable: the longer one carries more specifics ("6in gate valve")
    return a if len(a or "") >= len(b or "") else b


def merge(detections: list[Detection]) -> list[Detection]:
    """Collapse detections of the same object from different passes.

    Zoomed detections are seeded first so their boxes win: they were measured
    on real pixels, while the overview's were guessed from a downscale.
    Overlap marks corroboration, which the confidence policy then rewards -
    two independent passes finding the same valve in the same place is real
    evidence, and it is the main reason component confidence can now rise
    above what a single cautious pass reported.
    """
    ordered = sorted(detections, key=lambda d: (not d.zoomed, -area_pct(d.box)))
    kept: list[Detection] = []
    for detection in ordered:
        match = None
        for existing in kept:
            if iou(existing.box, detection.box) >= _SAME_OBJECT_IOU:
                match = existing
                break
        if match is None:
            kept.append(detection)
            continue
        # same object, seen twice
        if detection.source != match.source:
            match.corroborated = True
        match.label = _better_label(match.label, detection.label)
        match.callout = match.callout or detection.callout
        if _LEVEL[detection.confidence] > _LEVEL[match.confidence]:
            match.confidence = detection.confidence
        if detection.zoomed and not match.zoomed:
            match.box, match.source = detection.box, detection.source

    # Drop overview boxes that turned out to be groups: if a zoomed pass found
    # several distinct components inside one overview box, that box was the
    # "several components in one rectangle" problem and must not survive
    # alongside its own contents.
    survivors: list[Detection] = []
    for detection in kept:
        if detection.zoomed:
            survivors.append(detection)
            continue
        children = [
            other
            for other in kept
            if other is not detection
            and other.zoomed
            and contains(detection.box, other.box)
        ]
        if len(children) >= _GROUP_CHILD_COUNT:
            logger.debug(
                "dropping grouped overview box %s covering %d zoomed detections",
                detection.label,
                len(children),
            )
            continue
        survivors.append(detection)
    return survivors


# --------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------

# Below this share of the sheet, a component seen ONLY in the downscaled
# overview was a handful of pixels when the model read it - the identification
# cannot honestly be called high-confidence no matter what the model said.
# A zoomed pass looking at the same component is exempt: it saw it properly.
_TINY_AREA_PCT = 0.05


def resolve_confidence(detection: Detection) -> Confidence:
    """Confidence from EVIDENCE, not from the model's mood.

    Vision models are systematically cautious about drawn symbols: they report
    "medium" for a valve they have identified perfectly well, and the old
    pipeline had no way to ever say otherwise, so component confidence sat at
    medium/low forever. It also had no way to say a box was untrustworthy.
    Both directions are now earned:

      +2  the callout resolves against the sheet's own keynote legend - the
          drafter's words, the strongest identification available
      +1  an independent pass found the same object in the same place
      +1  the component was read on a zoomed crop rather than a downscale
      cap the box is still group-sized after refinement -> low
      cap tiny AND only ever seen in the downscale -> at most medium
    """
    score = _LEVEL[detection.confidence]
    if detection.keynote:
        score += 2
    if detection.corroborated:
        score += 1
    if detection.zoomed:
        score += 1
    # Size cap, judged on the box the model reported rather than the tightened
    # one, and never applied to a keynote-resolved component: its identity came
    # from the drafter's legend, not from how many pixels the symbol occupied.
    read_area = detection.read_area
    if read_area is None:
        read_area = area_pct(detection.box)
    if not detection.zoomed and not detection.keynote and read_area < _TINY_AREA_PCT:
        score = min(score, _LEVEL[Confidence.medium])
    if detection.suspect:
        score = min(score, _LEVEL[Confidence.low])
    return _BY_LEVEL[max(0, min(2, score))]


# --------------------------------------------------------------------------
# the zoomed passes
# --------------------------------------------------------------------------


def _run_detail(vision, image: "Image.Image", context: SheetContext) -> list[dict]:
    """One detail call over an in-memory crop. Returns [] on any failure."""
    payload, sent_w, sent_h = encode_for_vision(image)
    prompt = (
        DETAIL_PROMPT.replace("{width}", str(sent_w))
        .replace("{height}", str(sent_h))
        .replace("{context}", context.block())
    )
    try:
        raw = vision.analyze_image(payload, prompt)
    except Exception:
        logger.warning("component detail pass failed", exc_info=True)
        return []
    return parse_detail(raw, sent_w, sent_h)


def is_group_sized(box: list[float], oversized_area_pct: float) -> bool:
    """Whether a box is big enough to be a group rather than a component.

    Both axes have to be large: a pipe run legitimately spans half the sheet
    while staying a few percent tall, and splitting it would be wrong.
    """
    return (
        area_pct(box) >= oversized_area_pct
        and (box[2] - box[0]) >= 20
        and (box[3] - box[1]) >= 20
    )


def split_oversized(
    vision,
    img: "Image.Image",
    candidates: list[Detection],
    context: SheetContext,
    budget: int,
) -> tuple[list[Detection], int]:
    """Break apart boxes big enough to be groups rather than components.

    Returns (new detections, calls used). Parents that could not be broken up
    are marked `suspect` so the confidence policy stops presenting them as
    trustworthy - an unsplittable 40%-of-the-sheet box is a failure the
    reviewer should see flagged, not a high-confidence "equipment".
    """
    if budget <= 0:
        return [], 0
    # biggest first: the worst offenders are the ones users notice
    candidates = sorted(candidates, key=lambda d: -area_pct(d.box))
    produced: list[Detection] = []
    used = 0
    for parent in candidates:
        if used >= budget:
            break
        crop, crop_box = crop_pct(img, parent.box, margin_pct=3.0)
        used += 1
        children = _run_detail(vision, crop, context)
        mapped = [
            Detection(
                label=item["label"],
                box=child_to_sheet(item["box"], crop_box),
                confidence=item["confidence"],
                source=SOURCE_SPLIT,
                callout=item["callout"],
            )
            for item in children
        ]
        # a "split" that returns one box the size of the parent has not split
        # anything - keep the parent, flagged, rather than renaming it
        useful = [
            child
            for child in mapped
            if area_pct(child.box) < area_pct(parent.box) * 0.85
        ]
        if len(useful) >= 2:
            produced.extend(useful)
            parent.suspect = False
            parent.dropped = True  # its contents replace it
        else:
            parent.suspect = True
    return produced, used


def _tile_grid(scale: float, budget: int) -> int:
    """Side length of the tile grid, or 0 when tiling is not worth a call.

    Coverage has to be uniform: a partial sweep finds components in the
    top-left of the sheet and none in the bottom-right, which reads as a
    detection bias rather than a budget limit. So the grid shrinks to what the
    budget can cover completely, and is skipped entirely if even 2x2 does not
    fit.
    """
    wanted = 3 if scale >= 2.6 else 2
    while wanted >= 2 and wanted * wanted > budget:
        wanted -= 1
    return wanted if wanted >= 2 else 0


# Tiles overlap so a component sitting on a seam is whole in at least one of
# them; the merge step removes the resulting duplicates.
_TILE_OVERLAP = 0.08


def tile_sweep(
    vision,
    img: "Image.Image",
    context: SheetContext,
    budget: int,
    min_scale: float,
) -> tuple[list[Detection], int]:
    """Read the sheet in overlapping tiles at near-native resolution.

    This is the pass that finds small components at all: at 1/3 of the sheet
    per tile the model sees roughly three times the linear detail the overview
    had. Gated on the sheet actually having detail the overview lost - a phone
    photo of a single part gains nothing from being cut into quarters.
    """
    if budget <= 0:
        return [], 0
    scale = max(img.size) / MAX_VISION_SIDE
    if scale < min_scale:
        return [], 0
    grid = _tile_grid(scale, budget)
    if grid == 0:
        return [], 0

    step = 100.0 / grid
    produced: list[Detection] = []
    used = 0
    for row in range(grid):
        for col in range(grid):
            if used >= budget:
                break
            box = [
                max(0.0, col * step - step * _TILE_OVERLAP),
                max(0.0, row * step - step * _TILE_OVERLAP),
                min(100.0, (col + 1) * step + step * _TILE_OVERLAP),
                min(100.0, (row + 1) * step + step * _TILE_OVERLAP),
            ]
            crop, crop_box = crop_pct(img, box)
            used += 1
            for item in _run_detail(vision, crop, context):
                produced.append(
                    Detection(
                        label=item["label"],
                        box=child_to_sheet(item["box"], crop_box),
                        confidence=item["confidence"],
                        source=SOURCE_TILE,
                        callout=item["callout"],
                    )
                )
    return produced, used


# --------------------------------------------------------------------------
# naming pass (set-of-mark contact sheet)
# --------------------------------------------------------------------------

_CELL_PX = 320
_MONTAGE_COLUMNS = 4
_BADGE_PX = 46


def _badge_font():
    try:
        return ImageFont.load_default(size=34)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def build_montage(
    img: "Image.Image", boxes: list[list[float]]
) -> "Image.Image":
    """Compose numbered crops into one contact sheet.

    Naming N components in N calls is the obvious implementation and the wrong
    one. Each crop is drawn into a fixed cell with its panel number stamped in
    the corner, so a single vision call names the whole batch - and each panel
    is still a high-resolution view of the component, which is the reason the
    naming works at all.
    """
    rows = (len(boxes) + _MONTAGE_COLUMNS - 1) // _MONTAGE_COLUMNS
    columns = min(len(boxes), _MONTAGE_COLUMNS)
    sheet = Image.new("RGB", (columns * _CELL_PX, rows * _CELL_PX), "white")
    draw = ImageDraw.Draw(sheet)
    font = _badge_font()

    for index, box in enumerate(boxes):
        # generous margin: a symbol is identified by what surrounds it
        crop, _ = crop_pct(img, box, margin_pct=45.0, min_side_px=140)
        crop = crop.convert("RGB")
        crop.thumbnail((_CELL_PX - _BADGE_PX // 2, _CELL_PX - _BADGE_PX // 2))
        col, row = index % _MONTAGE_COLUMNS, index // _MONTAGE_COLUMNS
        ox, oy = col * _CELL_PX, row * _CELL_PX
        sheet.paste(
            crop,
            (
                ox + (_CELL_PX - crop.width) // 2,
                oy + (_CELL_PX - crop.height) // 2,
            ),
        )
        draw.rectangle([ox, oy, ox + _CELL_PX - 1, oy + _CELL_PX - 1], outline="black", width=2)
        # panel number on a solid badge so it can never be mistaken for part
        # of the drawing
        draw.rectangle([ox + 2, oy + 2, ox + _BADGE_PX, oy + _BADGE_PX], fill="black")
        draw.text((ox + 14, oy + 6), str(index + 1), fill="white", font=font)
    return sheet


def name_unlabeled(
    vision,
    img: "Image.Image",
    detections: list[Detection],
    context: SheetContext,
    limit: int,
) -> int:
    """Give every component a meaningful tag, using the drawing for context.

    Runs once per sheet over the components no pass could name, which is
    exactly the case the reviewer cannot act on: a box labelled "component"
    tells them nothing. Returns the number of calls made (0 or 1).
    """
    unnamed = [d for d in detections if not is_usable_label(d.label)]
    if not unnamed:
        return 0
    batch = unnamed[:limit]
    try:
        montage = build_montage(img, [d.box for d in batch])
        payload, _w, _h = encode_for_vision(montage)
        prompt = NAMING_PROMPT.replace("{count}", str(len(batch))).replace(
            "{context}", context.block()
        )
        raw = vision.analyze_image(payload, prompt)
    except Exception:
        logger.warning("component naming pass failed", exc_info=True)
        return 0
    names = parse_names(raw)
    for index, detection in enumerate(batch, start=1):
        label, level = names.get(str(index), (None, None))
        if is_usable_label(label):
            detection.label = label
            if level is not None:
                detection.confidence = level
    return 1


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def detect(
    *,
    vision,
    image_bytes: bytes,
    seeds: list[Detection],
    callout_marks: list[Detection],
    context: SheetContext,
    detail_calls: int,
    oversized_area_pct: float,
    tile_min_scale: float,
    name_limit: int,
) -> ComponentResult:
    """Run the full component pipeline over one sheet.

    Order is what makes the budget go far. TILING RUNS FIRST, because it does
    double duty: it finds the small components the overview never saw AND it
    dissolves the giant boxes on its own - once a tile has reported the pump,
    the pad and the valve individually, `merge` drops the overview rectangle
    that covered all three. Spending the budget on splits first would buy the
    second effect at the price of the first.

    Splitting is therefore the FALLBACK, aimed only at group boxes the tile
    sweep did not already break up - which is the normal case on a sheet whose
    resolution never justified tiling. Naming runs last and outside the budget:
    a component nobody can name is useless to a reviewer however cheap it was
    to find.

    Every stage is best-effort - this runs inside extraction, and a component
    pass must never cost a user their upload.
    """
    result = ComponentResult(callouts=list(callout_marks))
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            img = opened.convert("RGB")
    except Exception:
        logger.warning("component pass could not open the sheet image", exc_info=True)
        result.components = seeds
        return result

    detections = list(seeds)
    remaining = max(0, detail_calls)

    if remaining:
        produced, used = tile_sweep(vision, img, context, remaining, tile_min_scale)
        remaining -= used
        result.calls_used += used
        detections.extend(produced)

    if remaining:
        zoomed = [d for d in detections if d.zoomed]
        candidates = [
            seed
            for seed in seeds
            if not seed.dropped
            and is_group_sized(seed.box, oversized_area_pct)
            # already resolved: the tile sweep found its contents separately
            and sum(1 for z in zoomed if contains(seed.box, z.box)) < _GROUP_CHILD_COUNT
        ]
        produced, used = split_oversized(vision, img, candidates, context, remaining)
        remaining -= used
        result.calls_used += used
        # a parent replaced by its own contents must not survive beside them
        detections = [d for d in detections if not d.dropped]
        detections.extend(produced)

    detections = merge(detections)

    # tighten AFTER merging: the surviving box is the one worth measuring, and
    # tightening first would move boxes apart and break the overlap matching
    for detection in detections:
        tightened = tighten_to_ink(img, detection.box)
        if tightened is not None:
            detection.read_area = area_pct(detection.box)
            detection.box, detection.tightened = tightened, True

    if detail_calls > 0 and name_limit > 0:
        result.calls_used += name_unlabeled(vision, img, detections, context, name_limit)

    result.components = detections
    return result


def apply_keynotes(
    components: list[Detection], callout_marks: list[Detection], keynotes: dict[str, str]
) -> None:
    """Resolve callout numbers against the sheet's keynote legend and attach
    the meaning to both the component and the bubble."""
    if callout_marks and components:
        # link_by_proximity works on plain dicts (it stays free of this
        # module's types), so the assignments it makes are copied back
        proxies = [
            {"box": component.box, "callout": component.callout}
            for component in components
        ]
        callout_util.link_by_proximity(
            [{"number": mark.callout, "box": mark.box} for mark in callout_marks],
            proxies,
        )
        for component, proxy in zip(components, proxies):
            component.callout = proxy.get("callout")
    for component in components:
        if component.callout:
            component.keynote = keynotes.get(component.callout)
    for mark in callout_marks:
        if mark.callout:
            mark.keynote = keynotes.get(mark.callout)


def group_instances(detections: list[Detection]) -> list[dict]:
    """Collapse instances of the same component into one review card.

    A sheet with sixty identical gate valves must not become sixty rows a
    human has to confirm one by one - but every one of those valves still
    needs its own tight box, so the card carries them all and the viewer
    highlights all of them. Cards are keyed on the RESOLVED text, so "gate
    valve (keynote 3: 6\" GATE VALVE)" and a plain "gate valve" elsewhere on
    the sheet stay separate: they are different findings about different
    parts of the drawing.

    Returns plain dicts - image.py owns the VisionRegion type, so the
    dependency runs one way only.
    """
    grouped: dict[str, dict] = {}
    for detection in detections:
        text = callout_util.describe(
            detection.label, detection.callout, detection.keynote
        )
        if not text:
            continue  # nothing identifiable to show a reviewer
        key = normalize_label(text)
        confidence = resolve_confidence(detection)
        entry = grouped.get(key)
        if entry is None:
            grouped[key] = {
                "text": text,
                "confidence": confidence,
                "boxes": [detection.box],
            }
            continue
        entry["boxes"].append(detection.box)
        if _LEVEL[confidence] > _LEVEL[entry["confidence"]]:
            entry["confidence"] = confidence

    cards = []
    for entry in grouped.values():
        # reading order, so the first box (the one that survives ingestion as
        # the card's bbox) is the top-left instance rather than whichever pass
        # happened to run first
        boxes = sorted(entry["boxes"], key=lambda b: (round(b[1], 1), round(b[0], 1)))
        count = len(boxes)
        cards.append(
            {
                "region_type": RegionType.component,
                "text": f"{entry['text']} — {count} instances" if count > 1 else entry["text"],
                "confidence": entry["confidence"],
                "boxes": boxes,
            }
        )
    return cards
