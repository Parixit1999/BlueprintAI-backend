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
# "pj1234", "PJ-1234", "pj 1234", "proj1234"
_PJ_RE = re.compile(r"\bp(?:ro)?j\s*[-_#]?\s*(\d{2,6})", re.IGNORECASE)
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


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in _STOPWORDS]


def _initials(name: str) -> str:
    words = [t for t in _tokens(name) if not t.isdigit()]
    return "".join(w[0] for w in words) if len(words) >= 2 else ""


def suggest_projects(filename: str, projects: list[dict]) -> list[dict]:
    """Rank projects a file plausibly belongs to. Signals, strongest first:
    explicit pj#### number, full name containment, token overlap, initials
    ("Project Alpha Gamma" ~ "AG")."""
    parsed = parse_filename(filename)
    fname_tokens = set(_tokens(parsed["stem"]))
    fname_compact = "".join(_tokens(parsed["stem"]))
    suggestions = []

    for p in projects:
        best: tuple[float, str] | None = None

        if p.get("number") and p["number"] in parsed["project_numbers"]:
            best = (0.95, f"filename contains project number pj{p['number']}")

        name_tokens = [t for t in _tokens(p["name"]) if not t.isdigit()]
        if not best and name_tokens:
            name_compact = "".join(name_tokens)
            if name_compact and name_compact in fname_compact:
                best = (0.9, "filename contains the full project name")
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


def suggest_drawings(filename: str, registry: list[dict]) -> list[dict]:
    """Rank registry drawings this file is plausibly a sheet/version of, by DWG
    number: exact normalized match, then sequence-only match (facility code
    absent or different in the filename)."""
    parsed = parse_filename(filename)
    if not parsed["dwg_candidates"]:
        return []

    suggestions = []
    for cand in parsed["dwg_candidates"]:
        for d in registry:
            norm = d.get("dwg_number_norm") or ""
            if not norm:
                continue
            score_reason = None
            if norm == cand["norm"]:
                score_reason = (0.95, f"DWG number {d['dwg_number']} matches the filename exactly")
            elif cand["facility"] is None and norm.startswith(f"{cand['seq']}-W"):
                score_reason = (0.75, f"drawing sequence {cand['seq']} matches (facility code missing in filename)")
            elif norm.split("-")[0] == str(cand["seq"]):
                score_reason = (0.6, f"drawing sequence {cand['seq']} matches")
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
