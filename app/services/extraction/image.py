"""Image extraction via a vision model (Ollama locally, Bedrock Claude on AWS).

Vision output is the least reliable extraction path, which is exactly why every
field carries model-reported confidence and flows through HITL review before
ingestion. The model is instructed to return null rather than guess.

Two passes, not one. This overview pass reads the whole sheet: text, layout,
what the drawing depicts, and a first look at the drawn components. It cannot
see small components - a whole E-size sheet arrives at 1568px - so the
components it reports are only seeds for `components.detect`, which zooms in
and does the real work. See that module for why.
"""
import io
import json
import logging
import re
from dataclasses import dataclass, field

from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener

from app.config import settings
from app.exceptions import ExtractionFailed, InvalidFile
from app.services.extraction.ocr import normalize_text, textract_lines

# Teach Pillow to open iPhone HEIC/HEIF photos (idempotent)
register_heif_opener()
from app.schemas import Confidence, ProvisionalChunk, RegionType
from app.services.ai.base import VisionProvider
from app.services.extraction import callouts as callout_util
from app.services.extraction import components as component_util
from app.services.extraction.components import (
    Detection,
    SheetContext,
    downscale_for_vision,
    to_pct,
)
from app.services.extraction.enhance import enhance_for_vision

logger = logging.getLogger(__name__)


@dataclass
class VisionRegion:
    """One region the vision model reported, with a resolution-independent
    bbox in percentages (top-left origin). Callers map it into whatever
    coordinate space their renderer uses - image pixels for a photo, PDF
    points for a rasterized scanned page."""

    region_type: RegionType
    text: str | None
    confidence: Confidence
    bbox_pct: list[float] | None  # [x1, y1, x2, y2] as 0-100 percentages
    # component groups: every further instance's bbox (first is bbox_pct)
    extra_bboxes_pct: list[list[float]] | None = None
    is_drawing: bool | None = None  # summary region only: vision verdict


PROMPT = """You are extracting content from an engineering drawing image.
The image is {width}x{height} pixels.

Return ONLY a JSON object of this form:
{"is_drawing": true|false, "discipline": "...", "sheet_title": "...",
 "drawing_number": "...", "summary": "...", "regions": [...]}

"is_drawing" is false when the image is NOT an engineering/technical drawing
(a photo, screenshot, document scan of prose, etc.). Judge honestly.

"discipline" is the drawing's trade - architectural, structural, mechanical,
electrical, plumbing, civil, process, survey, or other. A later pass uses this
to read the symbols correctly, so answer it even when you have to infer it.

"sheet_title" and "drawing_number" are the title-block values, or null.

"summary" is one rich paragraph describing what the drawing DEPICTS as an
engineer would: what kind of drawing it is, what is shown (equipment,
structures, plans, sections), the overall layout, and anything notable.
Mention the drawing number and title if visible. Do not guess at values.

"regions" holds three kinds of entry:

1. TEXT - every piece of visible text: title block fields, drawing numbers,
   dimensions, notes, labels. Each one is its own region; do not merge or skip
   any.

2. DRAWN COMPONENTS - physical elements the drawing depicts, not text: valves,
   pumps, fittings, pipes, ducts, dampers, stairs, doors, windows, walls,
   beams, columns, footings, fixtures, drains, panels, tanks, major equipment
   (these are examples - label whatever this drawing actually depicts).
   Identify each from its drawn shape, standard symbology and its context in
   the sheet, not merely from text printed near it. Emit ONE region per
   component TYPE and list EVERY instance of that type in "instances", one
   box each.
   - Box each instance TIGHTLY around the symbol itself.
   - One instance box contains ONE component. NEVER put a box around a group
     of different components: if a rectangle would cover a pump AND its pad
     AND its piping, emit those as separate components instead. Any box
     covering more than a quarter of the sheet is almost certainly this
     mistake.
   - Small components matter as much as large ones - do not skip them.
   - "text" is a short engineer's label ("gate valve", "single-swing door",
     "W12x26 beam"). NEVER the bare word "component", "part" or "equipment".

3. NUMBERED CALLOUTS - the numbered bubbles (1, 2, 3 ... usually circled,
   hexagonal or boxed, often on a leader line) that key a component to a
   keynote list. Emit type "callout", "text" set to just the number, and
   "bbox_pct" on the bubble itself. When you can see which component a bubble
   points at, ALSO put its number in that component's "callout" field.

Each element of "regions":
{
  "text": "the exact text (component: a short label), or null if illegible - NEVER guess",
  "type": "note" | "dimension" | "title_block" | "bom" | "component" | "callout",
  "bbox_pct": [x1, y1, x2, y2],
  "instances": [[x1, y1, x2, y2], ...],   // components only: every instance
  "callout": "3",                          // components only: its callout number, if any
  "confidence": "high" | "medium" | "low"
}

bbox_pct is [x1, y1, x2, y2] in PIXELS of this {width}x{height} image,
measured from the TOP-LEFT corner. Draw the box TIGHTLY around what it
contains - it is used to highlight the exact region on the drawing.

"confidence" describes how certain you are of what you reported:
- "high": you can read the text exactly, or name the component confidently
  from a clear symbol.
- "medium": mostly certain, with real ambiguity.
- "low": genuinely uncertain. If a value is illegible, also set "text" to null.
Size is NOT a confidence criterion. A small but clearly drawn valve is high
confidence; judge legibility and certainty, never scale.
No prose outside the JSON object.{ocr_section}"""

# Appended to the prompt when Textract OCR lines are available. The OCR pass
# reads the FULL-resolution image, so it sees small text the downscaled
# vision image may not - transcribe against it rather than squinting.
OCR_SECTION = """

MACHINE OCR REFERENCE (read from the full-resolution image; use it to get
exact characters and numbers right, especially small text; it has no
understanding, so YOU still decide what is a region and what type it is;
do not invent regions for OCR fragments that are not meaningful text):
{ocr_lines}"""

_REGION_MAP = {
    "note": RegionType.note,
    "dimension": RegionType.dimension,
    "title_block": RegionType.title_block,
    "bom": RegionType.bom,
    "component": RegionType.component,
    # A callout bubble IS text on the sheet; it is stored as a note whose
    # meaning has been resolved against the keynote legend. It deliberately
    # does not become its own RegionType: the value is in the mapping, and a
    # new type would need every consumer to learn about it.
    "callout": RegionType.note,
}


def _parse_payload(raw: str) -> dict:
    """The model's response as one object.

    Preferred: the documented {"regions": [...]} contract. Two fallbacks kept
    from earlier providers - a bare array of regions, and an object embedded in
    prose. Parsed ONCE here rather than three times by three helpers.
    """
    text = (raw or "").strip()
    if text.startswith("```"):  # some providers fence their JSON
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"regions": parsed}
    except json.JSONDecodeError:
        pass
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    arr_match = re.search(r"\[.*\]", text, re.DOTALL)
    if arr_match is None:
        raise ExtractionFailed(
            "The vision model did not return structured output for this image. "
            "Try a clearer or higher-resolution image."
        )
    try:
        parsed = json.loads(arr_match.group(0))
    except json.JSONDecodeError:
        raise ExtractionFailed(
            "The vision model returned malformed output for this image. "
            "Try again or use a clearer image."
        )
    if not isinstance(parsed, list):
        raise ExtractionFailed("The vision model returned an unexpected structure.")
    return {"regions": parsed}


def _as_confidence(value) -> Confidence:
    return (
        Confidence(value)
        if value in {c.value for c in Confidence}
        else Confidence.medium
    )


def _clean_str(value) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


@dataclass
class _Parsed:
    """The overview pass, split into what each downstream stage needs."""

    summary: VisionRegion | None = None
    context: SheetContext = field(default_factory=SheetContext)
    text_regions: list[VisionRegion] = field(default_factory=list)
    component_seeds: list[Detection] = field(default_factory=list)
    callout_marks: list[Detection] = field(default_factory=list)
    callout_regions: list[tuple[VisionRegion, str]] = field(default_factory=list)


class ImageExtractor:
    def __init__(self, vision: VisionProvider):
        self._vision = vision

    # Kept as the class's own knobs because callers and tests reference them;
    # the implementations live in components.py, which every zoomed pass shares.
    MAX_SIDE = component_util.MAX_VISION_SIDE
    MAX_BYTES = component_util.MAX_VISION_BYTES

    _downscale = staticmethod(downscale_for_vision)
    _to_pct = staticmethod(to_pct)

    # OCR context cap: enough for a dense sheet without flooding the prompt
    MAX_OCR_LINES = 250

    def _overview(self, raw: str, sent_w: int, sent_h: int) -> _Parsed:
        """Turn the overview response into summary, context, text regions,
        component seeds and callout bubbles."""
        payload = _parse_payload(raw)
        parsed = _Parsed()

        summary = _clean_str(payload.get("summary"))
        verdict = payload.get("is_drawing")
        parsed.context = SheetContext(
            discipline=_clean_str(payload.get("discipline")),
            sheet_title=_clean_str(payload.get("sheet_title")),
            drawing_number=_clean_str(payload.get("drawing_number")),
            summary=summary,
        )
        if summary:
            parsed.summary = VisionRegion(
                region_type=RegionType.summary,
                text=summary,
                confidence=Confidence.high,
                bbox_pct=None,  # describes the whole drawing
                is_drawing=verdict if isinstance(verdict, bool) else None,
            )

        regions = payload.get("regions")
        if not isinstance(regions, list):
            regions = []
        for item in regions:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            text = _clean_str(item.get("text"))
            confidence = _as_confidence(item.get("confidence"))
            primary = None
            raw_box = item.get("bbox_pct")
            if isinstance(raw_box, list) and len(raw_box) == 4:
                primary = to_pct(raw_box, sent_w, sent_h)

            if kind == "component":
                # Every instance becomes its own detection here. The old
                # pipeline kept them welded into one region, which is why a
                # single sloppy box could stand for a dozen real components;
                # they are regrouped into one review card only at the end,
                # after each box has been refined and tightened on its own.
                boxes = []
                if primary:
                    boxes.append(primary)
                instances = item.get("instances")
                if isinstance(instances, list):
                    for instance in instances:
                        if isinstance(instance, list) and len(instance) == 4:
                            converted = to_pct(instance, sent_w, sent_h)
                            if converted and converted not in boxes:
                                boxes.append(converted)
                number = callout_util.normalize_callout(item.get("callout"))
                for box in boxes:
                    parsed.component_seeds.append(
                        Detection(
                            label=text,
                            box=box,
                            confidence=confidence,
                            source=component_util.SOURCE_OVERVIEW,
                            callout=number,
                        )
                    )
                continue

            if kind == "callout":
                number = callout_util.normalize_callout(text or item.get("callout"))
                if number and primary:
                    parsed.callout_marks.append(
                        Detection(
                            label=None,
                            box=primary,
                            confidence=confidence,
                            source=component_util.SOURCE_OVERVIEW,
                            callout=number,
                        )
                    )
                    region = VisionRegion(
                        region_type=RegionType.note,
                        text=number,
                        confidence=confidence,
                        bbox_pct=primary,
                    )
                    parsed.callout_regions.append((region, number))
                    continue
                # not a usable number - fall through and keep it as text

            parsed.text_regions.append(
                VisionRegion(
                    region_type=_REGION_MAP.get(kind, RegionType.note),
                    text=text,
                    confidence=confidence,
                    bbox_pct=primary,
                )
            )
        return parsed

    def analyze(
        self,
        data: bytes,
        ocr_lines: list[dict] | None = None,
        sheet_texts: list[str] | None = None,
    ) -> list[VisionRegion]:
        """Run the vision model on raw image bytes and return regions with
        percentage bboxes. Coordinate-space-agnostic so both photo uploads and
        rasterized scanned-PDF pages can reuse it. Returns [] if nothing found;
        callers decide whether an empty result is an error.

        ocr_lines: precomputed Textract lines (tests / reuse); None fetches
        them, [] skips OCR entirely.
        sheet_texts: text the caller already holds exactly (CAD entities, a PDF
        text layer). Used to resolve numbered callouts against the sheet's
        keynote legend, which OCR-free paths could not otherwise read.
        """
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
        except UnidentifiedImageError:
            raise InvalidFile("This file is not a valid image - it appears to be corrupt.")

        if ocr_lines is None:
            # full-resolution OCR pass; [] on any failure (graceful fallback)
            ocr_lines = textract_lines(data)

        sent_bytes, sent_w, sent_h = downscale_for_vision(data)
        # .replace, not .format - the prompt's JSON examples contain braces
        prompt = PROMPT.replace("{width}", str(sent_w)).replace("{height}", str(sent_h))
        if ocr_lines:
            listing = "\n".join(
                f'- "{line["text"]}"' for line in ocr_lines[: self.MAX_OCR_LINES]
            )
            prompt = prompt.replace(
                "{ocr_section}", OCR_SECTION.replace("{ocr_lines}", listing)
            )
        else:
            prompt = prompt.replace("{ocr_section}", "")

        raw = self._vision.analyze_image(sent_bytes, prompt)
        parsed = self._overview(raw, sent_w, sent_h)

        # Zoom in on the components: split group-sized boxes, sweep the sheet
        # in tiles, then name whatever is still anonymous.
        detected = component_util.detect(
            vision=self._vision,
            image_bytes=data,
            seeds=parsed.component_seeds,
            callout_marks=parsed.callout_marks,
            context=parsed.context,
            detail_calls=settings.component_detail_calls,
            oversized_area_pct=settings.component_oversized_area_pct,
            tile_min_scale=settings.component_tile_min_scale,
            name_limit=settings.component_max_named_per_sheet,
        )

        # Numbered callouts only mean something once they are resolved against
        # the sheet's own keynote legend, so gather every text the sheet has:
        # what the model transcribed, what OCR read, and what the caller
        # already knows exactly.
        keynote_sources = [r.text for r in parsed.text_regions if r.text]
        keynote_sources += [line["text"] for line in ocr_lines]
        keynote_sources += list(sheet_texts or [])
        keynotes = callout_util.parse_keynotes(keynote_sources)
        component_util.apply_keynotes(
            detected.components, detected.callouts, keynotes
        )

        regions: list[VisionRegion] = []
        if parsed.summary:
            regions.append(parsed.summary)
        for card in component_util.group_instances(detected.components):
            boxes = card["boxes"]
            regions.append(
                VisionRegion(
                    region_type=card["region_type"],
                    text=card["text"],
                    confidence=card["confidence"],
                    bbox_pct=boxes[0],
                    extra_bboxes_pct=boxes[1:] or None,
                )
            )
        # a bubble reading "3" is unsearchable; resolved it becomes an answer
        for region, number in parsed.callout_regions:
            region.text = callout_util.callout_text(number, keynotes.get(number))
            if keynotes.get(number):
                region.confidence = Confidence.high
            regions.append(region)
        regions.extend(parsed.text_regions)

        if ocr_lines:
            self._snap_to_ocr(regions, ocr_lines)
        return regions

    # How far a vision box may sit from an OCR line that shares its text
    # before the "match" is a coincidence, as a percentage of the sheet.
    _MAX_SNAP_DISTANCE_PCT = 12.0
    # With repeated text ("1", "TYP", "6"), the nearest OCR line only wins if
    # the runner-up is clearly farther - otherwise which one it is is a guess.
    _SNAP_AMBIGUITY_RATIO = 1.5

    @classmethod
    def _snap_to_ocr(cls, regions: list[VisionRegion], ocr_lines: list[dict]) -> None:
        """Replace approximate vision bboxes with Textract's pixel-accurate
        ones. A confirmed character-for-character OCR match also upgrades
        confidence: the value was machine-read at full resolution, not
        transcribed from a downscaled image.

        Repeated text used to be given up on entirely, which is exactly the
        case numbered callouts fall into - a sheet has a dozen "3"s and none of
        them got a precise box. They are now disambiguated by POSITION: of the
        lines carrying that text, the one nearest the vision box wins, provided
        it is unambiguously nearest.
        """
        by_text: dict[str, list[dict]] = {}
        for line in ocr_lines:
            by_text.setdefault(normalize_text(line["text"]), []).append(line)
        for region in regions:
            # component labels describe a drawn symbol, not text on the sheet;
            # matching one against a legend line would move its box onto the
            # legend
            if region.region_type == RegionType.component or not region.text:
                continue
            matches = by_text.get(normalize_text(region.text))
            if not matches:
                continue
            if len(matches) == 1:
                line = matches[0]
            else:
                line = cls._nearest_line(matches, region.bbox_pct)
                if line is None:
                    continue
            region.bbox_pct = list(line["bbox_pct"])
            if line["confidence"] >= 90 and region.confidence != Confidence.high:
                region.confidence = Confidence.high

    @classmethod
    def _nearest_line(cls, matches: list[dict], box: list[float] | None) -> dict | None:
        """The OCR line a repeated-text region refers to, or None when it
        cannot be told apart from another candidate."""
        if not box:
            return None
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        scored = []
        for line in matches:
            lb = line["bbox_pct"]
            lx, ly = (lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2
            scored.append((((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5, line))
        scored.sort(key=lambda pair: pair[0])
        best_distance, best = scored[0]
        if best_distance > cls._MAX_SNAP_DISTANCE_PCT:
            return None
        runner_up = scored[1][0]
        if runner_up < best_distance * cls._SNAP_AMBIGUITY_RATIO:
            return None
        return best

    @staticmethod
    def _pct_to_extents(pct: list[float], xmax: float, ymax: float) -> list[float]:
        x1, y1, x2, y2 = pct
        # percentages from top-left -> extents coords, y-up
        return [
            round(x1 / 100 * xmax, 1),
            round(ymax - (y2 / 100 * ymax), 1),
            round(x2 / 100 * xmax, 1),
            round(ymax - (y1 / 100 * ymax), 1),
        ]

    @staticmethod
    def region_to_chunk(
        region: VisionRegion, xmax: float, ymax: float, page: int = 1
    ) -> ProvisionalChunk:
        """Map a percentage-space region into a chunk whose bbox is in the
        renderer's y-up coordinate space (extents [0, 0, xmax, ymax])."""
        bbox = None
        if region.bbox_pct is not None:
            bbox = ImageExtractor._pct_to_extents(region.bbox_pct, xmax, ymax)
        extra = None
        if region.extra_bboxes_pct:
            extra = [
                ImageExtractor._pct_to_extents(b, xmax, ymax)
                for b in region.extra_bboxes_pct
            ]
        return ProvisionalChunk(
            region_type=region.region_type,
            chunk_text=region.text,
            is_drawing=region.is_drawing,
            bbox=bbox,
            confidence=region.confidence,
            page=page,
            extra_bboxes=extra,
        )

    def extract(self, path: str) -> list[ProvisionalChunk]:
        raw = open(path, "rb").read()
        # Enhancement (orientation, contrast, upscale) BEFORE the vision model;
        # dims come from the enhanced image so bbox percentages match the
        # (orientation-normalized) preview.
        data, _applied = enhance_for_vision(raw)
        try:
            with Image.open(io.BytesIO(data)) as img:
                width, height = img.size
        except UnidentifiedImageError:
            raise InvalidFile("This file is not a valid image - it appears to be corrupt.")

        regions = self.analyze(data)
        chunks = [self.region_to_chunk(r, width, height) for r in regions]
        if not chunks:
            raise ExtractionFailed(
                "No text regions were detected in this image. If the drawing has "
                "readable text, try a higher-resolution or better-lit image."
            )
        return chunks
