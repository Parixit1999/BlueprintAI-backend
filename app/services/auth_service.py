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
    """Wrong credentials or missing/expired token."""


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

    def change_password(self, user_id: str, current: str, new: str) -> None:
        user = self._repo.get_user_by_id(user_id)
        if user is None or not bcrypt.checkpw(current.encode(), user["password_hash"].encode()):
            time.sleep(_FAILED_LOGIN_DELAY_S)
            raise AuthFailed("Current password is incorrect.")
        if len(new) < 8:
            raise AuthFailed("New password must be at least 8 characters.")
        self._repo.set_password_hash(
            user_id, bcrypt.hashpw(new.encode(), bcrypt.gensalt()).decode()
        )
        # changing the password signs out every other session
        self._repo.delete_tokens_for_user(user_id)

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
