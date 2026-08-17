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
    return {
        "username": user["username"],
        "full_name": user.get("full_name"),
        "email": user.get("email"),
    }


@router.get("/users")
def list_users(svc: AuthService = Depends(auth_service)):
    """Everyone who can sign in. One shared workspace - no roles, so any
    signed-in teammate can see and manage the account list."""
    return {"users": svc.list_users()}


@router.post("/users", status_code=201)
def create_user(body: UserCreateRequest, svc: AuthService = Depends(auth_service)):
    return svc.create_user(
        body.username, body.password, body.full_name, body.email
    )


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
