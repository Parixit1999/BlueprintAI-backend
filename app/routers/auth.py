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
    return {"username": request.state.user["username"]}


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
