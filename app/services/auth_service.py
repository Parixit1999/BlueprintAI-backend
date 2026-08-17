"""Simple username/password authentication with server-side session tokens.

Design (pilot-appropriate, still done properly):
- passwords bcrypt-hashed, never stored or logged in plain text
- opaque bearer tokens (256-bit random); only their sha256 lands in the
  database, so neither a DB leak nor a log leak yields a usable session
- tokens expire after TOKEN_TTL_DAYS and can be revoked (logout)
- a short sleep on failed logins blunts brute-force attempts
"""
import hashlib
import secrets
import time

import bcrypt

from app.exceptions import BlueprintError
from app.repositories import AuthRepository

TOKEN_TTL_DAYS = 7
_FAILED_LOGIN_DELAY_S = 0.5


class AuthFailed(BlueprintError):
    """Wrong credentials or missing/expired token (mapped to 401)."""


class UserRejected(BlueprintError):
    """Account creation/removal validation failure. Like PasswordChangeRejected
    it must NOT be a 401, or the frontend would sign the admin out mid-form."""


class PasswordChangeRejected(BlueprintError):
    """Password change validation failure. Deliberately NOT AuthFailed:
    the frontend treats 401 as an expired session and signs the user out,
    which must not happen on a wrong current password or a too-short new
    one. Falls through to the default 400."""


class AuthService:
    def __init__(self, repo: AuthRepository):
        self._repo = repo

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def login(self, username: str, password: str) -> dict:
        user = self._repo.get_user_by_username(username.strip().lower())
        ok = user is not None and bcrypt.checkpw(
            password.encode(), user["password_hash"].encode()
        )
        if not ok:
            time.sleep(_FAILED_LOGIN_DELAY_S)
            raise AuthFailed("Incorrect username or password.")
        token = secrets.token_urlsafe(32)
        self._repo.insert_token(self._hash_token(token), user["id"], TOKEN_TTL_DAYS)
        return {"token": token, "username": user["username"]}

    def authenticate(self, token: str) -> dict | None:
        """The user for a valid, unexpired token; None otherwise."""
        return self._repo.get_user_by_token(self._hash_token(token))

    def logout(self, token: str) -> None:
        self._repo.delete_token(self._hash_token(token))

    def change_password(
        self, user_id: str, current: str, new: str, keep_token: str | None = None
    ) -> None:
        user = self._repo.get_user_by_id(user_id)
        if user is None or not bcrypt.checkpw(current.encode(), user["password_hash"].encode()):
            time.sleep(_FAILED_LOGIN_DELAY_S)
            raise PasswordChangeRejected("Current password is incorrect.")
        if len(new) < 8:
            raise PasswordChangeRejected(
                "New password must be at least 8 characters long."
            )
        self._repo.set_password_hash(
            user_id, bcrypt.hashpw(new.encode(), bcrypt.gensalt()).decode()
        )
        # sign out every OTHER session; the one making the change stays valid
        self._repo.delete_tokens_for_user(
            user_id, except_sha=self._hash_token(keep_token) if keep_token else None
        )

    # --- account management (single shared workspace, no roles) ---

    def list_users(self) -> list[dict]:
        return self._repo.list_users()

    def create_user(
        self,
        username: str,
        password: str,
        full_name: str | None = None,
        email: str | None = None,
    ) -> dict:
        """Add a teammate. Validation mirrors change_password so a new account
        can never be weaker than an existing one."""
        username = (username or "").strip().lower()
        if not username:
            raise UserRejected("Username is required.")
        if len(username) < 3:
            raise UserRejected("Username must be at least 3 characters long.")
        if len(password or "") < 8:
            raise UserRejected("Password must be at least 8 characters long.")
        if self._repo.get_user_by_username(username) is not None:
            raise UserRejected(f"The username “{username}” is already taken.")
        user_id = self._repo.create_user(
            username,
            bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
            (full_name or "").strip() or None,
            (email or "").strip().lower() or None,
        )
        return {
            "user_id": user_id,
            "username": username,
            "full_name": (full_name or "").strip() or None,
            "email": (email or "").strip().lower() or None,
        }

    def delete_user(self, user_id: str, acting_user_id: str) -> None:
        if user_id == acting_user_id:
            raise UserRejected("You cannot remove your own account.")
        if self._repo.get_user_by_id(user_id) is None:
            raise UserRejected("That account no longer exists.")
        self._repo.delete_user(user_id)

    def ensure_seed_user(self, username: str, password: str | None) -> str | None:
        """Create the first account when no users exist. Returns the
        generated password when one had to be invented (caller logs it once);
        None when a password was supplied or users already exist."""
        if self._repo.count_users() > 0:
            return None
        generated = None
        if not password:
            generated = secrets.token_urlsafe(12)
            password = generated
        self._repo.create_user(
            username, bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        )
        return generated
