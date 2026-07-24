"""Flexible file-to-project/drawing recognition.

The client's archive has an open-ended naming convention: "11767 W-59 1A 1.1
FIRST FLR..pdf", "G-13 6118W.pdf", "11772 W-54 sht 6 of 31.tif", "pj1234_...".
This module parses whatever signals a filename carries (DWG number patterns,
project numbers, sheet markers, name fragments) and scores candidates against
the known projects and the imported drawing registry. Results are ranked
suggestions with human-readable reasons - the user confirms; nothing is
auto-assigned silently (same HITL philosophy as extraction review).
"""
import re

# "12158-W-59", "11767 W-59", "6118W", "6118-W" - seq number, W, optional facility code
_DWG_RE = re.compile(r"(\d{3,5})\s*[-_ ]?\s*W\b\s*[-_ ]?\s*(\d{1,3})?", re.IGNORECASE)
# "pj1234", "PJ-1234", "pj 1234", "proj1234" - plus municipal alphanumeric
# numbers like "pj490-W" / "pj490W" (digits with a short letter suffix; the
# lookahead stops "pj1234-sheet" from swallowing "she" as a suffix)
_PJ_RE = re.compile(
    r"\bp(?:ro)?j\s*[-_#]?\s*(\d{2,6}(?:-?[A-Za-z]{1,3})?(?![A-Za-z]))", re.IGNORECASE
)


def _norm_number(value: str) -> str:
    """Compare project numbers ignoring case and separators: '490-W' ==
    '490w' == '490 W'."""
    return re.sub(r"[^A-Z0-9]", "", value.upper())
# "SHT 23", "sht. 6 of 31", "sheet 4"
_SHEET_RE = re.compile(r"\bsh(?:ee)?t\.?\s*#?\s*(\d+)(?:\s*of\s*(\d+))?", re.IGNORECASE)

_STOPWORDS = {"the", "of", "and", "for", "at", "to", "a", "in", "on", "project", "dwg", "drawing"}


def normalize_dwg(seq: str, facility: str | None) -> str:
    return f"{int(seq)}-W-{int(facility)}" if facility else f"{int(seq)}-W"


def parse_filename(filename: str) -> dict:
    """Extract every recognizable signal from a filename."""
    stem = re.sub(r"\.[A-Za-z0-9]{1,4}$", "", filename)  # drop extension
    out: dict = {"stem": stem, "dwg_candidates": [], "project_numbers": [], "sheet_number": None}

    for m in _DWG_RE.finditer(stem):
        seq, fac = m.group(1), m.group(2)
        out["dwg_candidates"].append(
            {"seq": int(seq), "facility": int(fac) if fac else None, "norm": normalize_dwg(seq, fac)}
        )
    for m in _PJ_RE.finditer(stem):
        out["project_numbers"].append(m.group(1))
    m = _SHEET_RE.search(stem)
    if m:
        out["sheet_number"] = f"{m.group(1)} of {m.group(2)}" if m.group(2) else m.group(1)
    return out


def parse_content(texts: list[str]) -> dict:
    """Extract DWG-number and project-number signals from a file's extracted
    region text (title blocks especially) - 'information found within the
    actual file content'. Essential for scans with meaningless filenames."""
    out: dict = {"dwg_candidates": [], "project_numbers": []}
    seen_norms: set[str] = set()
    for text in texts[:200]:
        if not text:
            continue
        for m in _DWG_RE.finditer(text):
            seq, fac = m.group(1), m.group(2)
            norm = normalize_dwg(seq, fac)
            if norm not in seen_norms:
                seen_norms.add(norm)
                out["dwg_candidates"].append(
                    {"seq": int(seq), "facility": int(fac) if fac else None, "norm": norm}
                )
        for m in _PJ_RE.finditer(text):
            if m.group(1) not in out["project_numbers"]:
                out["project_numbers"].append(m.group(1))
    return out


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in _STOPWORDS]


def _initials(name: str) -> str:
    words = [t for t in _tokens(name) if not t.isdigit()]
    return "".join(w[0] for w in words) if len(words) >= 2 else ""


def suggest_projects(
    filename: str, projects: list[dict], content_texts: list[str] | None = None
) -> list[dict]:
    """Rank projects a file plausibly belongs to. Signals, strongest first:
    explicit pj#### number (filename or content), full name containment
    (filename or content), token overlap, initials ("Project Alpha Gamma" ~
    "AG"). Content signals come from the file's extracted regions."""
    parsed = parse_filename(filename)
    content = parse_content(content_texts or [])
    fname_tokens = set(_tokens(parsed["stem"]))
    fname_compact = "".join(_tokens(parsed["stem"]))
    content_blob = " ".join(content_texts or [])
    content_compact = "".join(_tokens(content_blob))
    suggestions = []

    for p in projects:
        best: tuple[float, str] | None = None

        number_norm = _norm_number(p["number"]) if p.get("number") else None
        if number_norm and number_norm in {_norm_number(n) for n in parsed["project_numbers"]}:
            best = (0.95, f"filename contains project number pj{p['number']}")
        elif number_norm and number_norm in {_norm_number(n) for n in content["project_numbers"]}:
            best = (0.95, f"the drawing content contains project number {p['number']}")

        name_tokens = [t for t in _tokens(p["name"]) if not t.isdigit()]
        if not best and name_tokens:
            name_compact = "".join(name_tokens)
            if name_compact and name_compact in fname_compact:
                best = (0.9, "filename contains the full project name")
            elif name_compact and content_compact and name_compact in content_compact:
                best = (0.85, "the drawing content contains the project name")
            else:
                overlap = sum(1 for t in name_tokens if t in fname_tokens)
                if overlap and overlap / len(name_tokens) >= 0.5:
                    score = 0.5 + 0.3 * (overlap / len(name_tokens))
                    matched = [t for t in name_tokens if t in fname_tokens]
                    best = (round(score, 2), f"filename matches part of the project name ({', '.join(matched)})")
                else:
                    initials = _initials(p["name"])
                    if len(initials) >= 2 and initials in fname_tokens:
                        best = (0.6, f"filename contains the project initials '{initials.upper()}'")

        if best:
            suggestions.append(
                {
                    "project_id": p["project_id"],
                    "name": p["name"],
                    "number": p.get("number"),
                    "score": best[0],
                    "reason": best[1],
                }
            )

    suggestions.sort(key=lambda s: s["score"], reverse=True)
    return suggestions[:5]


def suggest_drawings(
    filename: str, registry: list[dict], content_texts: list[str] | None = None
) -> list[dict]:
    """Rank registry drawings this file is plausibly a sheet/version of, by DWG
    number found in the filename OR in the file's extracted content (title
    blocks): exact normalized match, then sequence-only match."""
    parsed = parse_filename(filename)
    candidates = [(c, "filename") for c in parsed["dwg_candidates"]]
    if content_texts:
        seen = {c["norm"] for c in parsed["dwg_candidates"]}
        for c in parse_content(content_texts)["dwg_candidates"]:
            if c["norm"] not in seen:
                candidates.append((c, "the drawing content"))
    if not candidates:
        return []

    suggestions = []
    for cand, source in candidates:
        for d in registry:
            norm = d.get("dwg_number_norm") or ""
            if not norm:
                continue
            score_reason = None
            if norm == cand["norm"]:
                score_reason = (0.95, f"DWG number {d['dwg_number']} matches {source} exactly")
            elif cand["facility"] is None and norm.startswith(f"{cand['seq']}-W"):
                score_reason = (0.75, f"drawing sequence {cand['seq']} matches {source} (facility code missing)")
            elif norm.split("-")[0] == str(cand["seq"]):
                score_reason = (0.6, f"drawing sequence {cand['seq']} matches {source}")
            if score_reason:
                suggestions.append(
                    {
                        "drawing_id": d["drawing_id"],
                        "dwg_number": d["dwg_number"],
                        "description": d.get("description"),
                        "project_id": d.get("project_id"),
                        "project_name": d.get("project_name"),
                        "year": d.get("year"),
                        "score": score_reason[0],
                        "reason": score_reason[1],
                    }
                )

    # keep the best score per drawing
    best_by_id: dict[str, dict] = {}
    for s in suggestions:
        cur = best_by_id.get(s["drawing_id"])
        if cur is None or s["score"] > cur["score"]:
            best_by_id[s["drawing_id"]] = s
    ranked = sorted(best_by_id.values(), key=lambda s: s["score"], reverse=True)
    return ranked[:5]


def parse_year(raw) -> int | None:
    """Best-effort year from the book's messy date column (2018, "2017-2018",
    "2018--", datetime objects)."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw if 1800 <= raw <= 2100 else None
    m = re.search(r"\b(18|19|20)\d{2}\b", str(raw))
    return int(m.group(0)) if m else None
