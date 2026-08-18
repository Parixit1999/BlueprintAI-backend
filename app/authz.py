"""Role-based access control: the page gate and sheet (project) scoping.

The model is deliberately small: a user is an admin (bypasses everything),
or carries one role, or has no role (and no access). A role grants PAGES -
whole features, no view/edit split - plus sheet access inside the Number
Book: every sheet, or an explicit list of projects. Sheet access applies
everywhere content is project-owned: registry rows, documents, and chat
retrieval. Content owned by no project (unassigned uploads, cross-project
registry cards) stays visible to anyone who holds the relevant page,
otherwise uploaders would lose sight of their own uploads.
"""
from __future__ import annotations

from typing import Any

from app.exceptions import Forbidden

PAGES = ("dashboard", "numberbook", "chat", "upload", "documents")

_PAGE_LABEL = {
    "dashboard": "the Dashboard",
    "numberbook": "the Number Book",
    "chat": "Chat",
    "upload": "Upload",
    "documents": "Documents",
}

_ADMIN = "admin"   # people/roles management
_OPEN = "open"     # any authenticated user, role or not

# Ordered (methods, prefix, requirement) rules; first match wins.
# `methods` of None means any method. Requirement is _OPEN, _ADMIN, or a
# frozenset of pages of which the user needs AT LEAST ONE - several pages
# legitimately share endpoints (the chat evidence viewer renders drawing
# pages; the upload flow polls extraction and reads assignment suggestions).
_RULES: list[tuple[frozenset[str] | None, str, Any]] = [
    (frozenset({"GET"}), "/api/auth/me", _OPEN),
    (frozenset({"POST"}), "/api/auth/logout", _OPEN),
    (frozenset({"POST"}), "/api/auth/password", _OPEN),
    (None, "/api/auth/users", _ADMIN),
    (None, "/api/auth/roles", _ADMIN),
    (None, "/api/stats", frozenset({"dashboard"})),
    (None, "/api/registry", frozenset({"numberbook"})),
    (None, "/api/drawings", frozenset({"numberbook"})),
    (None, "/api/sets", frozenset({"numberbook"})),
    # the project LIST feeds scope pickers on several pages; detail and
    # mutations stay Number Book territory (matched below)
    (frozenset({"GET"}), "/api/projects__exact__",
     frozenset({"numberbook", "chat", "upload", "documents"})),
    (None, "/api/projects", frozenset({"numberbook"})),
    (frozenset({"POST"}), "/api/files/upload", frozenset({"upload"})),
    (frozenset({"POST"}), "/api/files/statuses", frozenset({"upload", "documents"})),
    (None, "/api/files/__id__/extraction", frozenset({"upload", "documents"})),
    (None, "/api/files/__id__/render",
     frozenset({"upload", "documents", "chat", "numberbook"})),
    (None, "/api/files/__id__/suggestions",
     frozenset({"upload", "documents", "numberbook"})),
    (None, "/api/files/__id__/assign",
     frozenset({"upload", "documents", "numberbook"})),
    (None, "/api/files/__id__/unassign",
     frozenset({"upload", "documents", "numberbook"})),
    (None, "/api/files", frozenset({"documents"})),
    (None, "/api/review", frozenset({"documents"})),
    (None, "/api/folders", frozenset({"documents"})),
    (None, "/api/query", frozenset({"chat"})),
    (None, "/api/chats", frozenset({"chat"})),
]


def _match(method: str, path: str):
    for methods, pattern, req in _RULES:
        if methods is not None and method not in methods:
            continue
        if pattern.endswith("__exact__"):
            base = pattern.removesuffix("__exact__")
            if path == base or path == base + "/":
                return req
            continue
        if "__id__" in pattern:
            head, tail = pattern.split("__id__", 1)
            if path.startswith(head) and path.endswith(tail) and len(path) > len(head) + len(tail):
                # one path segment where __id__ sits
                middle = path[len(head):len(path) - len(tail)]
                if middle and "/" not in middle:
                    return req
            continue
        if path == pattern or path.startswith(pattern + "/"):
            return req
    return None


def user_pages(user: dict) -> frozenset[str]:
    role = user.get("role")
    return frozenset(role["pages"]) if role else frozenset()


def check_page(user: dict, method: str, path: str) -> str | None:
    """The middleware page gate: a human-readable denial, or None to pass.
    Default-deny: an /api path no rule recognizes is refused rather than
    silently open - a new router must be added to the map on purpose."""
    req = _match(method, path)
    if req is _OPEN:
        return None
    if user.get("is_admin"):
        return None
    if req is _ADMIN:
        return "Only administrators can manage people and roles."
    if user.get("role") is None:
        return ("Your account doesn't have access yet. "
                "Ask an administrator to assign you a role.")
    if req is None:
        return "Your role doesn't include access to this."
    granted = user_pages(user)
    if granted & req:
        return None
    label = _PAGE_LABEL.get(min(req), "this")
    if len(req) == 1:
        label = _PAGE_LABEL[next(iter(req))]
    return f"Your role doesn't include access to {label}."


def allowed_project_ids(user: dict) -> list[str] | None:
    """None = unrestricted (admin, or a role with every sheet); otherwise the
    explicit allowlist (possibly empty: such a user sees only unowned
    content)."""
    if user.get("is_admin"):
        return None
    role = user.get("role") or {}
    if role.get("all_sheets"):
        return None
    return list(role.get("project_ids") or [])


def check_project(user: dict, project_id: str | None) -> None:
    """Raise unless the user may touch content owned by this project.
    project_id None means Main-Book / whole-archive scope, which restricted
    roles do not get."""
    allowed = allowed_project_ids(user)
    if allowed is None:
        return
    if project_id is None:
        raise Forbidden("The Main Book isn't included in your role - open one of your sheets instead.")
    if project_id not in allowed:
        raise Forbidden("That sheet isn't included in your role.")


def check_file(user: dict, locate, file_id: str) -> None:
    """Per-document guard: a file is reachable when it is unassigned or its
    drawing's project is allowed. `locate` is any callable returning
    (project_id, drawing_id) or None; missing files fall through so the
    handler returns its usual 404."""
    allowed = allowed_project_ids(user)
    if allowed is None:
        return
    located = locate(file_id)
    if located is None:
        return  # not found: let the endpoint 404 as it always has
    project_id, drawing_id = located
    if drawing_id is None or project_id is None:
        return  # unassigned content stays visible
    if project_id not in allowed:
        raise Forbidden("That document belongs to a sheet outside your role.")
