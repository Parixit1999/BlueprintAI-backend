"""Login, logout, session introspection, and password change."""
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.db import pool
from app.repositories import AuthRepository
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


def auth_service() -> AuthService:
    return AuthService(AuthRepository(pool))


class LoginRequest(BaseModel):
    username: str
    password: str


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    full_name: str | None = None
    email: str | None = None
    role_id: str | None = None


class UserUpdateRequest(BaseModel):
    # role_id is tri-state: absent = leave alone, null = clear, id = set.
    # model_fields_set distinguishes absent from null.
    role_id: str | None = None
    is_admin: bool | None = None


class RoleRequest(BaseModel):
    name: str
    pages: list[str] = []
    all_sheets: bool = False
    project_ids: list[str] = []


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    return header.removeprefix("Bearer ").strip() or None


@router.post("/login")
def login(body: LoginRequest, svc: AuthService = Depends(auth_service)):
    return svc.login(body.username, body.password)


@router.post("/logout", status_code=204)
def logout(request: Request, svc: AuthService = Depends(auth_service)):
    token = _bearer(request)
    if token:
        svc.logout(token)


@router.get("/me")
def me(request: Request):
    # request.state.user is set by the auth middleware
    user = request.state.user
    role = user.get("role")
    return {
        "id": user["id"],
        "username": user["username"],
        "full_name": user.get("full_name"),
        "email": user.get("email"),
        "is_admin": bool(user.get("is_admin")),
        # no project_ids on purpose: the server filters everything, the
        # client only needs to know which pages to draw
        "role": None if role is None else {
            "name": role["name"],
            "pages": role["pages"],
            "all_sheets": role["all_sheets"],
        },
    }


@router.get("/users")
def list_users(svc: AuthService = Depends(auth_service)):
    """Everyone who can sign in, with their role. Admin-gated by the
    auth middleware."""
    return {"users": svc.list_users()}


@router.post("/users", status_code=201)
def create_user(body: UserCreateRequest, svc: AuthService = Depends(auth_service)):
    return svc.create_user(
        body.username, body.password, body.full_name, body.email, body.role_id
    )


@router.patch("/users/{user_id}", status_code=204)
def update_user(
    user_id: str, body: UserUpdateRequest, svc: AuthService = Depends(auth_service)
):
    svc.update_user(
        user_id,
        role_id=body.role_id if "role_id" in body.model_fields_set else ...,
        is_admin=body.is_admin,
    )


@router.get("/roles")
def list_roles(svc: AuthService = Depends(auth_service)):
    """Roles an admin can hand out. Admin-gated by the auth middleware."""
    return {"roles": svc.list_roles()}


@router.post("/roles", status_code=201)
def create_role(body: RoleRequest, svc: AuthService = Depends(auth_service)):
    return svc.create_role(body.name, body.pages, body.all_sheets, body.project_ids)


@router.patch("/roles/{role_id}")
def update_role(
    role_id: str, body: RoleRequest, svc: AuthService = Depends(auth_service)
):
    return svc.update_role(
        role_id, body.name, body.pages, body.all_sheets, body.project_ids
    )


@router.delete("/roles/{role_id}", status_code=204)
def delete_role(role_id: str, svc: AuthService = Depends(auth_service)):
    svc.delete_role(role_id)


@router.delete("/users/{user_id}", status_code=204)
def delete_user(
    user_id: str, request: Request, svc: AuthService = Depends(auth_service)
):
    svc.delete_user(user_id, request.state.user["id"])


@router.post("/password", status_code=204)
def change_password(
    body: PasswordChangeRequest,
    request: Request,
    svc: AuthService = Depends(auth_service),
):
    svc.change_password(
        request.state.user["id"],
        body.current_password,
        body.new_password,
        keep_token=_bearer(request),
    )
