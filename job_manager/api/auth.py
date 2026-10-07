"""Authentication dependency and rate limiting for FastAPI.

Provides:
    - :func:`get_current_user` — FastAPI dependency that extracts and
      validates a JWT token from the request.
    - :func:`get_client_ip` — resolves the real client IP behind
      reverse proxies using ``X-Forwarded-For``.
    - :class:`RateLimiter` — simple in-memory sliding-window rate limiter.
    - :data:`login_limiter` — pre-configured limiter for the login endpoint.

Module-level constants:
    - ``JWT_SECRET`` — signing key (env ``JWT_SECRET`` or config fallback).
    - ``JWT_EXPIRATION_HOURS`` — token lifetime in hours.
"""

from __future__ import annotations

import os
import time
import threading
import logging
from collections import defaultdict

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt as pyjwt

from api.config import Config

logger = logging.getLogger(__name__)

config = Config()
security = HTTPBearer(auto_error=False)

_jwt_cfg = config.get("jwt", {})
JWT_SECRET: str = os.environ.get("JWT_SECRET", _jwt_cfg.get("secret"))
"""JWT signing secret. Overridden by the ``JWT_SECRET`` env var."""

JWT_EXPIRATION_HOURS: int = int(_jwt_cfg.get("expiration_hours", 24))
"""Token lifetime in hours."""


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """In-memory sliding-window rate limiter, keyed by arbitrary string.

    Tracks timestamps of recent attempts per key and rejects new ones
    once the window is full.

    Args:
        max_attempts: Maximum allowed attempts within the window.
        window_seconds: Length of the sliding window in seconds.
    """

    def __init__(self, max_attempts: int, window_seconds: int) -> None:
        """Initialize the instance.

        Args:
            max_attempts: Input value for `max_attempts`.
            window_seconds: Input value for `window_seconds`.

        """
        self.max_attempts = max_attempts
        self.window = window_seconds
        self._attempts: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        """Check whether *key* is allowed to make another attempt.

        Args:
            key: Identifier to rate-limit (e.g. an IP address).

        Returns:
            ``True`` if under the limit, ``False`` otherwise.
        """
        now = time.time()
        with self._lock:
            attempts = self._attempts[key]
            self._attempts[key] = [t for t in attempts if now - t < self.window]
            if len(self._attempts[key]) >= self.max_attempts:
                return False
            self._attempts[key].append(now)
            return True


login_limiter = RateLimiter(
    max_attempts=config.get("login_rate_limit", 10),
    window_seconds=config.get("login_rate_window_seconds", 300),
)
"""Pre-configured rate limiter for the ``/auth/login`` endpoint."""


# ---------------------------------------------------------------------------
# JWT dependency
# ---------------------------------------------------------------------------

def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """FastAPI dependency that extracts and validates a JWT token.

    The token is read from the ``Authorization: Bearer <token>`` header.

    Args:
        request: The incoming HTTP request.
        credentials: Credentials extracted by the ``HTTPBearer`` scheme.

    Returns:
        The ``sub`` (username) claim from the decoded JWT.

    Raises:
        HTTPException: 401 if the token is missing, expired, or invalid.
    """
    token = ""
    if credentials:
        token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization")
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        username = payload.get("sub", "")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return username
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_client_ip(request: Request) -> str:
    """Resolve the real client IP address.

    If the direct connection comes from a trusted proxy (as listed in
    ``config.yaml → trusted_proxies``), the first address in the
    ``X-Forwarded-For`` header is returned instead.

    Args:
        request: The incoming HTTP request.

    Returns:
        The client's IP address as a string.
    """
    trusted_proxies = config.get("trusted_proxies", ["127.0.0.1", "::1"])
    client_ip = request.client.host if request.client else "unknown"

    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for and client_ip in trusted_proxies:
        client_ip = forwarded_for.split(",")[0].strip()

    return client_ip
