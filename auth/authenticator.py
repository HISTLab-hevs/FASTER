"""Authentication service for JWT login and account-management workflows.

This module owns authentication policy and token issuance for the backend API.
Persistence concerns stay behind the repository/database layer where possible;
the authenticator coordinates validation, password policy, and authenticated
account-management flows on top of that storage.
"""

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import jwt

try:
    import bcrypt
except Exception:
    bcrypt = None

try:
    from database.config import DatabaseSettings
    from database.connection import MariaDBConnectionFactory, MariaDBSession
    from database.repositories.users_repository import MariaUsersRepository
except Exception:
    DatabaseSettings = None
    MariaDBConnectionFactory = None
    MariaDBSession = None
    MariaUsersRepository = None

_ALGORITHM = "HS256"
_TOKEN_TTL_HOURS = 12
_PASSWORD_MIN_LEN = 10


logger = logging.getLogger(__name__)


def _weak_passwords_allowed() -> bool:
    """Return whether weak-password checks are disabled via environment flag."""
    return str(os.getenv("FL_AUTH_ALLOW_WEAK_PASSWORDS", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> datetime:
    """Return the current UTC time with timezone information."""
    return datetime.now(timezone.utc)


def _hash(password: str) -> str:
    """Hash a plain-text password for persistent storage.

    Args:
        password: Plain-text password value.

    Returns:
        Encoded bcrypt password hash.
    """
    if bcrypt is None:
        raise RuntimeError("bcrypt is required for shared password hashing")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_hash(password: str, stored: str) -> bool:
    """Verify a plain-text password against a stored hash.

    Args:
        password: Plain-text password value.
        stored: Persisted password hash value.

    Returns:
        ``True`` when the password matches the stored hash.
    """
    if not stored or not stored.startswith("$2") or bcrypt is None:
        return False
    try:
        return bool(bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8")))
    except Exception:
        return False


def _normalize_email(email: str) -> str:
    """Normalize an email address for backend storage and lookup.

    Args:
        email: Email address value.

    Returns:
        Lowercased and trimmed email string.
    """
    return str(email or "").strip().lower()


def _is_valid_email(email: str) -> bool:
    """Return whether an email address matches the backend validation rule.

    Args:
        email: Email address value.

    Returns:
        ``True`` when the address matches the accepted format.
    """
    return bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", _normalize_email(email)))


def _password_policy_errors(password: str) -> List[str]:
    """Return human-readable password policy failures.

    Args:
        password: Plain-text password value.

    Returns:
        List of validation errors. Empty when the password satisfies the
        current backend policy.
    """
    if _weak_passwords_allowed():
        return []

    pwd = str(password or "")
    errors: List[str] = []
    if len(pwd) < _PASSWORD_MIN_LEN:
        errors.append(f"Password must be at least {_PASSWORD_MIN_LEN} characters")
    if re.search(r"\s", pwd):
        errors.append("Password must not contain spaces")
    if not re.search(r"[a-z]", pwd):
        errors.append("Password must include at least one lowercase letter")
    if not re.search(r"[A-Z]", pwd):
        errors.append("Password must include at least one uppercase letter")
    if not re.search(r"\d", pwd):
        errors.append("Password must include at least one number")
    if not re.search(r"[^A-Za-z0-9\s]", pwd):
        errors.append("Password must include at least one special character")
    return errors


class Authenticator:
    """Authenticate users and issue JWT access tokens.

    Passwords are hashed with bcrypt and accounts live in the shared MariaDB
    users table, whose repository is resolved during initialization. Use
    :meth:`backend_error` to inspect why the database backend is unavailable.
    """

    def __init__(self, secret_key: str) -> None:
        """Initialize the authenticator.

        Args:
            secret_key: JWT signing secret used for access tokens.
        """
        self._secret = secret_key
        self._db_enabled = False
        self._users_repo = None
        self._db_factory = None
        self._db_unavailable_reason = ""
        if DatabaseSettings and MariaDBConnectionFactory and MariaUsersRepository:
            try:
                settings = DatabaseSettings.from_env()
                self._db_factory = MariaDBConnectionFactory(settings)
                self._users_repo = MariaUsersRepository(self._db_factory)
                self._db_enabled = True
            except Exception as exc:
                raise RuntimeError("Authenticator requires DB-backed auth storage") from exc
        else:
            self._db_unavailable_reason = (
                "DB-backed auth storage requires configured MariaDB dependencies."
            )
            raise RuntimeError(self._db_unavailable_reason)

    @staticmethod
    def _safe_user_from_record(user: Dict[str, Any]) -> Dict[str, Any]:
        """Project an internal user record into the API-safe shape.

        Args:
            user: Internal user mapping from repository/database code.

        Returns:
            Sanitized user payload without password material.
        """
        return {
            "identifier": str(user.get("identifier") or ""),
            "email": user.get("email"),
            "role": "admin" if str(user.get("role") or "user") == "admin" else "user",
            "active": bool(user.get("active", True)),
            "created_at": str(user.get("created_at") or ""),
            "created_by": str(user.get("created_by") or ""),
        }

    def _db_find_user(self, identifier_or_email: str) -> Optional[Dict[str, Any]]:
        """Load one user record by identifier first, then normalized email.

        Args:
            identifier_or_email: User identifier or email address.

        Returns:
            Internal user mapping when found, otherwise ``None``.
        """
        if not self._db_enabled or self._users_repo is None:
            return None
        needle = str(identifier_or_email or "").strip()
        if not needle:
            return None

        try:
            user = self._users_repo.get_user_by_identifier(needle)
            if user:
                return {
                    "identifier": user.identifier,
                    "email": user.email,
                    "password_hash": user.password_hash,
                    "role": user.role,
                    "active": user.active,
                    "created_at": user.created_at,
                    "created_by": user.created_by,
                }
        except Exception as exc:
            logger.warning("Authenticator failed DB lookup by identifier for '%s': %s", needle, exc)
            return None

        email_needle = _normalize_email(needle)
        try:
            row = self._users_repo.get_user_by_email(email_needle)
            if row:
                return {
                    "identifier": row.identifier,
                    "email": row.email,
                    "password_hash": row.password_hash,
                    "role": row.role,
                    "active": row.active,
                    "created_at": row.created_at,
                    "created_by": row.created_by,
                }
        except Exception as exc:
            logger.warning("Authenticator failed DB lookup by email for '%s': %s", needle, exc)
            return None
        return None

    def _db_ready(self) -> bool:
        """Return whether DB-backed auth storage is available."""
        return bool(self._db_enabled and self._users_repo is not None)

    def backend_error(self) -> str:
        """Return a human-readable reason when DB-backed auth is unavailable."""
        return (
            self._db_unavailable_reason
            or "DB-backed auth storage is unavailable."
        )

    def _build_token(self, user: Dict[str, Any]) -> str:
        """Build a signed JWT access token for a verified user.

        Args:
            user: Internal user mapping.

        Returns:
            Encoded JWT string.
        """
        now = _utc_now()
        exp = now + timedelta(hours=_TOKEN_TTL_HOURS)
        payload = {
            "sub": str(user.get("identifier") or ""),
            "role": "admin" if str(user.get("role") or "user") == "admin" else "user",
            "email": user.get("email"),
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
        }
        return jwt.encode(payload, self._secret, algorithm=_ALGORITHM)

    def authenticate(self, identifier_or_email: str, password: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Authenticate a user and return a token plus safe user payload.

        Args:
            identifier_or_email: User identifier or email address.
            password: Plain-text password value.

        Returns:
            Tuple ``(is_authenticated, token, user)``. The token and user payload
            are empty when authentication fails.
        """
        if not self._db_ready():
            return False, "", {}
        user = self._db_find_user(identifier_or_email)
        if not user or not bool(user.get("active", True)):
            return False, "", {}
        if not _verify_hash(password, str(user.get("password_hash") or "")):
            return False, "", {}
        token = self._build_token(user)
        return True, token, self._safe_user_from_record(user)

    def verify_token(self, token: str) -> Tuple[bool, str]:
        """Validate a JWT and return the subject identifier.

        Args:
            token: Encoded JWT access token.

        Returns:
            Tuple ``(is_valid, identifier)`` where ``identifier`` is empty when
            the token cannot be validated.

        This compact return shape is used by request handlers that only need
        truthiness plus the authenticated identifier.
        """
        if not token:
            return False, ""
        try:
            payload = jwt.decode(token, self._secret, algorithms=[_ALGORITHM])
            return True, str(payload.get("sub") or "")
        except jwt.PyJWTError:
            return False, ""

    def verify_token_claims(self, token: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate a JWT and return the claims used by API handlers.

        Args:
            token: Encoded JWT access token.

        Returns:
            Tuple ``(is_valid, claims)`` where ``claims`` contains the sanitized
            identifier, role, and email fields when validation succeeds.
        """
        if not token:
            return False, {}
        try:
            payload = jwt.decode(token, self._secret, algorithms=[_ALGORITHM])
            return True, {
                "identifier": str(payload.get("sub") or ""),
                "role": "admin" if str(payload.get("role") or "user") == "admin" else "user",
                "email": payload.get("email"),
            }
        except jwt.PyJWTError:
            return False, {}

    def get_user(self, identifier_or_email: str) -> Optional[Dict[str, Any]]:
        """Return one API-safe user payload by identifier or email.

        Args:
            identifier_or_email: User identifier or email address.

        Returns:
            Sanitized user mapping when found, otherwise ``None``.
        """
        if not self._db_ready():
            return None
        user = self._db_find_user(identifier_or_email)
        return self._safe_user_from_record(user) if user else None

    def list_users(self) -> List[Dict[str, Any]]:
        """Return API-safe user payloads for administrative listing views.

        Returns:
            Sanitized user mappings for all users, including inactive accounts
            when the DB backend is available, otherwise an empty list.
        """
        if not self._db_ready():
            return []

        out: List[Dict[str, Any]] = []
        try:
            rows = self._users_repo.list_users(include_inactive=True)
        except Exception as exc:
            logger.warning("Authenticator failed DB user listing: %s", exc)
            return []

        for row in rows or []:
            try:
                if isinstance(row, dict):
                    identifier = row.get("identifier")
                    email = row.get("email")
                    role = row.get("role")
                    active = row.get("active")
                    created_at = row.get("created_at")
                    created_by = row.get("created_by")
                else:
                    identifier = row.identifier
                    email = row.email
                    role = row.role
                    active = row.active
                    created_at = row.created_at
                    created_by = row.created_by
            except Exception:
                continue
            out.append({
                "identifier": str(identifier or ""),
                "email": email,
                "role": "admin" if str(role or "user") == "admin" else "user",
                "active": bool(active),
                "created_at": str(created_at or ""),
                "created_by": str(created_by or ""),
            })
        return out

    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate a password against the current backend policy.

        Args:
            password: Plain-text password value.

        Returns:
            Tuple ``(is_valid, errors)`` where ``errors`` is empty on success.
        """
        errors = _password_policy_errors(password)
        return len(errors) == 0, errors

    def register_user(
        self,
        username: str,
        password: str,
        role: str = "user",
        email: Optional[str] = None,
        created_by: str = "cli",
    ) -> bool:
        """Register a user account for bootstrap or non-admin creation flows.

        Args:
            username: Account identifier to create.
            password: Plain-text password value.
            role: Role to assign to the new account.
            email: Optional email address for the new account.
            created_by: Actor label recorded with the created account.

        Returns:
            ``True`` when the account is created successfully, otherwise
            ``False``.
        """
        identifier = str(username or "").strip()
        if not identifier:
            return False
        mail = _normalize_email(email or "") or None

        if not self._db_ready():
            return False
        if self._db_find_user(identifier):
            return False
        if mail and self._db_find_user(mail):
            return False
        ok_pwd, _ = self.validate_password(password)
        if not ok_pwd:
            return False
        try:
            self._users_repo.create_user({
                "identifier": identifier,
                "email": mail,
                "password_hash": _hash(password),
                "role": "admin" if role == "admin" else "user",
                "active": True,
                "created_by": created_by,
            })
        except Exception as exc:
            logger.warning("Authenticator failed to register user '%s': %s", identifier, exc)
            return False
        return True

    def create_user(
        self,
        identifier: str,
        email: str,
        password: str,
        role: str,
        created_by: str,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Create a user account from an authenticated administrative action.

        Args:
            identifier: Account identifier to create.
            email: Email address for the new account.
            password: Plain-text password value.
            role: Role to assign to the new account.
            created_by: Identifier of the actor creating the account.

        Returns:
            Tuple ``(is_created, message, user)`` where ``user`` contains a
            minimal safe payload on success.
        """
        identifier_clean = str(identifier or "").strip()
        if not identifier_clean:
            return False, "Identifier is required", {}

        mail = _normalize_email(email)
        if not _is_valid_email(mail):
            return False, "Invalid email format", {}

        ok_pwd, pwd_errors = self.validate_password(password)
        if not ok_pwd:
            return False, "; ".join(pwd_errors), {}

        if not self._db_ready():
            return False, self.backend_error(), {}
        if self._db_find_user(identifier_clean):
            return False, "Identifier already registered", {}
        if self._db_find_user(mail):
            return False, "Email already registered", {}
        try:
            self._users_repo.create_user({
                "identifier": identifier_clean,
                "email": mail,
                "password_hash": _hash(password),
                "role": "admin" if role == "admin" else "user",
                "active": True,
                "created_by": created_by,
            })
        except Exception as exc:
            logger.warning("Authenticator failed to create user '%s': %s", identifier_clean, exc)
            return False, "Could not create user", {}
        return True, "User created", {
            "identifier": identifier_clean,
            "email": mail,
            "role": "admin" if role == "admin" else "user",
        }

    def admin_set_user_role(self, actor_identifier: str, target_identifier: str, new_role: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Update a non-admin user's role from an administrative action.

        Args:
            actor_identifier: Identifier of the acting administrator.
            target_identifier: Identifier or email of the user to update.
            new_role: Requested role value.

        Returns:
            Tuple ``(is_updated, message, user)`` where ``user`` is the updated
            sanitized payload on success.
        """
        role_clean = "admin" if str(new_role or "").lower() == "admin" else "user"

        if not self._db_ready():
            return False, self.backend_error(), {}
        actor = self._db_find_user(actor_identifier)
        if not actor or str(actor.get("role") or "user") != "admin":
            return False, "Forbidden", {}
        target = self._db_find_user(target_identifier)
        if not target:
            return False, "User not found", {}
        if str(target.get("role") or "user") == "admin":
            return False, "Cannot change role of an existing admin", {}
        self._users_repo.update_user(str(target.get("identifier") or ""), {"role": role_clean})
        updated = self._db_find_user(str(target.get("identifier") or "")) or {}
        return True, "Role updated", self._safe_user_from_record(updated)

    def admin_delete_user(self, actor_identifier: str, target_identifier: str) -> Tuple[bool, str]:
        """Delete a non-admin account from an administrative action.

        Args:
            actor_identifier: Identifier of the acting administrator.
            target_identifier: Identifier or email of the user to delete.

        Returns:
            Tuple ``(is_deleted, message)`` describing the outcome.
        """
        if not self._db_ready():
            return False, self.backend_error()
        actor = self._db_find_user(actor_identifier)
        if not actor or str(actor.get("role") or "user") != "admin":
            return False, "Forbidden"
        target = self._db_find_user(target_identifier)
        if not target:
            return False, "User not found"
        if str(target.get("role") or "user") == "admin":
            return False, "Cannot delete admin users"
        deleted = self._users_repo.delete_user(str(target.get("identifier") or ""))
        return (True, "User deleted") if deleted else (False, "User not found")

    def admin_set_password(self, target_identifier: str, new_password: str) -> Tuple[bool, str]:
        """Set a new password for an existing account.

        Args:
            target_identifier: Identifier or email of the account to update.
            new_password: Replacement plain-text password value.

        Returns:
            Tuple ``(is_updated, message)`` describing the outcome.

        Intended for operational CLI/bootstrap workflows.
        """
        ok_pwd, pwd_errors = self.validate_password(new_password)
        if not ok_pwd:
            return False, "; ".join(pwd_errors)

        if not self._db_ready():
            return False, self.backend_error()
        target = self._db_find_user(target_identifier)
        if not target:
            return False, "User not found"
        self._users_repo.update_user(str(target.get("identifier") or ""), {"password_hash": _hash(new_password)})
        return True, "Password updated"

    def change_password_authenticated(self, identifier: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change a password for an authenticated user after verification.

        Args:
            identifier: Identifier of the authenticated account.
            old_password: Current plain-text password value.
            new_password: Replacement plain-text password value.

        Returns:
            Tuple ``(is_updated, message)`` describing the outcome.
        """
        ok_pwd, pwd_errors = self.validate_password(new_password)
        if not ok_pwd:
            return False, "; ".join(pwd_errors)

        if not self._db_ready():
            return False, self.backend_error()
        user = self._db_find_user(identifier)
        if not user:
            return False, "Account not found"
        if not _verify_hash(str(old_password or ""), str(user.get("password_hash") or "")):
            return False, "Current password is incorrect"
        self._users_repo.update_user(str(user.get("identifier") or ""), {"password_hash": _hash(new_password)})
        return True, "Password updated"
