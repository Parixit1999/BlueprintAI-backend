"""Numbered callouts and the keynote legend they point at.

Engineering sheets rarely write "gate valve" next to a gate valve. They draw a
numbered bubble - 1, 2, 3 ... - with a leader line, and put the meaning in a
keynote list somewhere else on the sheet:

    1.  6" DUCTILE IRON PIPE
    2   GATE VALVE, FLANGED
    3)  CONCRETE THRUST BLOCK

The number on its own is worthless to a reader and to retrieval ("what is 3?").
The number RESOLVED against the legend is the single most reliable
identification a component can carry, because it is the drafter's own words
rather than a vision model's reading of a symbol - which is why a resolved
keynote is the strongest confidence signal in the component pipeline.

Pure functions on purpose: no model calls, no I/O. The vision pass reports
which bubble sits on which component; everything here is text parsing and
nearest-neighbour geometry, so it costs nothing and behaves identically for
DXF entity text, a PDF text layer, and OCR lines.
"""
import re

# "1. DESCRIPTION" / "2 - DESCRIPTION" / "3) DESCRIPTION" / "04  DESCRIPTION".
# The separator is optional (drafters align keynotes with spaces alone), which
# is what forces the strict guards in parse_keynotes below - without them
# "3.5 MM" parses as keynote 3 meaning "5 MM".
_KEYNOTE_RE = re.compile(r"^\(?(\d{1,2})\)?\s*[.):\-–]?\s+(.{3,120})$")

# A keynote number is a small ordinal. Sheets number keynotes 1..~50; anything
# larger is a dimension, a year, or a part number that happens to lead a line.
_MAX_KEYNOTE = 60

# Lines that look like keynotes but are not: a description has to read like
# words, not like a measurement or a coordinate.
_MEASUREMENT_RE = re.compile(
    r"^[\d\s.,'\"/x×\-+±Ø⌀RØ°%()]*(mm|cm|m|in|ft|kg|lb|psi|deg)?[\d\s.,'\"/x×\-+±°%()]*$",
    re.IGNORECASE,
)


def _looks_like_description(text: str) -> bool:
    """A keynote description is prose-ish: it has letters, and it is not just a
    dimension or a bare code. Rejecting here is cheap; a wrong keynote silently
    mislabels a component, which is far more expensive than a missed one."""
    letters = sum(1 for ch in text if ch.isalpha())
    if letters < 3:
        return False
    if _MEASUREMENT_RE.match(text):
        return False
    # at least a third of the characters should be letters/spaces - filters
    # "12-3456-7890 REV" style part numbers that carry a stray word
    alpha_space = sum(1 for ch in text if ch.isalpha() or ch.isspace())
    return alpha_space / len(text) >= 0.34


def parse_keynotes(texts) -> dict[str, str]:
    """Build {callout number -> description} from a sheet's text regions.

    Takes whatever text the extractor already has (CAD entity text, a PDF text
    layer, OCR lines) - a legend is frequently ONE multi-line MTEXT entity, so
    every input is split on newlines first.

    Later entries lose to earlier ones: a legend block is normally transcribed
    once, and a stray "1 SEE DETAIL" caption elsewhere on the sheet must not
    overwrite a real keynote.
    """
    keynotes: dict[str, str] = {}
    for raw in texts:
        if not raw:
            continue
        for line in str(raw).splitlines():
            line = " ".join(line.split())
            if not line:
                continue
            match = _KEYNOTE_RE.match(line)
            if not match:
                continue
            number, description = match.group(1), match.group(2).strip(" .-:–")
            if not 1 <= int(number) <= _MAX_KEYNOTE:
                continue
            if not _looks_like_description(description):
                continue
            keynotes.setdefault(str(int(number)), description)
    return keynotes


def normalize_callout(value) -> str | None:
    """Canonical form of a callout number as the model may report it: "03",
    "(3)", "3." and 3 are all keynote 3. Anything that is not a small integer
    is not a callout."""
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    try:
        number = int(digits)
    except ValueError:
        return None
    return str(number) if 1 <= number <= _MAX_KEYNOTE else None


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


# How far a callout bubble may sit from the component it labels, as a
# percentage of the sheet's diagonal. Leader lines are short by drafting
# convention; beyond this the "nearest" component is a coincidence, and a
# wrong link is worse than no link.
_MAX_LINK_DISTANCE_PCT = 12.0


def link_by_proximity(callouts: list[dict], components: list[dict]) -> None:
    """Attach each unclaimed callout number to its nearest component.

    The vision model is asked to report the association directly (it can see
    the leader line, which geometry cannot); this is the fallback for the
    bubbles it reported without saying what they point at. Components that
    already carry a callout keep it.

    Both arguments are lists of mutable dicts with "box" ([x1,y1,x2,y2] in
    sheet percentages) and, for components, "callout".
    """
    free = [c for c in components if not c.get("callout")]
    if not free:
        return
    for callout in callouts:
        number = normalize_callout(callout.get("number"))
        box = callout.get("box")
        if not number or not box:
            continue
        cx, cy = _center(box)
        best, best_distance = None, _MAX_LINK_DISTANCE_PCT
        for component in free:
            if component.get("callout"):
                continue  # claimed earlier in this loop
            px, py = _center(component["box"])
            distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if distance < best_distance:
                best, best_distance = component, distance
        if best is not None:
            best["callout"] = number


def describe(label: str | None, callout: str | None, keynote: str | None) -> str | None:
    """The text a component region carries once its callout is resolved.

    Order matters: the drafter's keynote wording leads when the model had no
    label of its own, because "6\" GATE VALVE, FLANGED" beats "valve". When
    both exist they are combined - the reader gets the model's plain-language
    identification AND the sheet's own designation, and retrieval indexes both.
    """
    label = (label or "").strip() or None
    if keynote:
        if label and label.casefold() not in keynote.casefold():
            return f"{label} (keynote {callout}: {keynote})"
        return f"{keynote} (keynote {callout})"
    if label and callout:
        return f"{label} (callout {callout})"
    return label


def callout_text(number: str, keynote: str | None) -> str:
    """Region text for the bubble itself. A chunk reading "3" is unsearchable
    and meaningless in a citation; resolved, it becomes a real answer."""
    return f"Callout {number}: {keynote}" if keynote else f"Callout {number}"
