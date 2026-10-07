"""Minimal Job Manager HTTP client used by the Faster web backend."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any
from urllib import error, request

import yaml


class JobManagerClient:
    """Small wrapper around the Job Manager REST API.

    This client uses a service-account login so the web backend can query queue
    state and submit or cancel jobs without coupling frontend sessions to Job
    Manager JWTs.
    """

    _LOGIN_FAILED_ERROR = "JM service login failed"
    _INTEGRATION_DISABLED_ERROR = "Job Manager integration disabled"

    _USER_JOB_LIMIT_MIN = 1
    _USER_JOB_LIMIT_MAX = 200
    _JOB_LIST_LIMIT_MIN = 1
    _JOB_LIST_LIMIT_MAX = 500

    def __init__(self) -> None:
        """Initialize the client from environment configuration.

        The following environment variables are read:

        ``JM_API_BASE_URL``
            Base URL for the Job Manager API.
        ``JM_API_TIMEOUT``
            Request timeout in seconds.
        ``FL_JM_SOURCE_APP``
            Source application name attached to submissions.
        ``JM_SERVICE_USERNAME``
            Service account username.
        ``JM_SERVICE_PASSWORD``
            Service account password.
        ``FL_JM_TOKEN_CACHE``
            Optional path to a cross-process service-token cache (used by the
            CLI to avoid re-login on every invocation).
        """
        self.base_url = str(
            os.getenv("JM_API_BASE_URL", "http://job-manager-api:5000") or ""
        ).rstrip("/")
        self.timeout = float(os.getenv("JM_API_TIMEOUT", "8") or "8")
        self.enabled = bool(self.base_url)

        self.source_app = (
            str(os.getenv("FL_JM_SOURCE_APP", "faster") or "faster").strip() or "faster"
        )
        self._service_username = str(os.getenv("JM_SERVICE_USERNAME", "") or "")
        self._service_password = str(os.getenv("JM_SERVICE_PASSWORD", "") or "")

        # Optional cross-process token cache. Short-lived CLI invocations each
        # build a fresh client, so without this every command would re-login and
        # quickly trip the Job Manager's per-IP login rate limit. When this path
        # is set, the service token is shared on disk between invocations. The
        # long-lived web app leaves it unset and keeps its in-memory cache.
        self._token_cache_path = str(os.getenv("FL_JM_TOKEN_CACHE", "") or "").strip()

        self._token = ""
        self._token_expire_epoch = 0.0

    def _url(self, path: str) -> str:
        """Build an absolute API URL for a given path.

        Args:
            path: API path beginning with ``/``.

        Returns:
            Fully qualified URL composed from the configured base URL and path.
        """
        return f"{self.base_url}{path}"

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[bool, int, dict[str, Any]]:
        """Send an HTTP request and parse a JSON response.

        Args:
            method: HTTP method to use.
            path: API path relative to the configured base URL.
            payload: Optional JSON payload for the request body.
            headers: Optional extra headers to merge into the request.

        Returns:
            A tuple ``(ok, status_code, data)`` where:
                - ``ok`` indicates whether the request completed without an
                  HTTP or transport error,
                - ``status_code`` is the HTTP status code, or ``0`` for
                  transport-level failures,
                - ``data`` is the parsed JSON object or an error payload.

        Notes:
            Non-dictionary JSON payloads are wrapped as ``{"data": ...}`` to
            preserve a dictionary return shape.
        """
        if not self.enabled:
            return False, 0, {"error": self._INTEGRATION_DISABLED_ERROR}

        body = None
        req_headers = {"Accept": "application/json"}

        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            req_headers["Content-Type"] = "application/json"

        if headers:
            req_headers.update(headers)

        req = request.Request(
            self._url(path),
            data=body,
            method=method.upper(),
            headers=req_headers,
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    return True, int(resp.status), parsed
                return True, int(resp.status), {"data": parsed}
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="ignore")
            try:
                parsed = json.loads(raw) if raw else {}
            except Exception:
                parsed = {"error": raw or str(exc)}

            if not isinstance(parsed, dict):
                parsed = {"error": raw or str(exc)}

            return False, int(exc.code), parsed
        except Exception as exc:
            return False, 0, {"error": str(exc)}

    def _service_login(self) -> bool:
        """Authenticate the service account and cache the bearer token.

        Returns:
            ``True`` if a valid token is available after this call,
            otherwise ``False``.

        Notes:
            The token is cached for a fixed local TTL of 50 minutes to avoid
            repeated logins while staying comfortably below a typical
            one-hour expiration window.
        """
        if not self._service_password:
            return False

        now = time.time()
        if self._token and now < self._token_expire_epoch:
            return True

        # Reuse a still-valid token persisted by a previous process, if any.
        cached_token, cached_expiry = self._read_cached_token()
        if cached_token and now < cached_expiry:
            self._token = cached_token
            self._token_expire_epoch = cached_expiry
            return True

        ok, status, data = self._request_json(
            "POST",
            "/auth/login",
            payload={
                "username": self._service_username,
                "password": self._service_password,
            },
        )
        if not ok or status != 200:
            self._token = ""
            self._token_expire_epoch = 0.0
            return False

        token = str(data.get("token") or "")
        if not token:
            return False

        self._token = token
        self._token_expire_epoch = now + (50 * 60)
        self._write_cached_token(token, self._token_expire_epoch)
        return True

    def _read_cached_token(self) -> tuple[str, float]:
        """Return ``(token, expiry_epoch)`` from the on-disk cache, if enabled.

        Returns ``("", 0.0)`` when no cache path is configured or the cache is
        missing, unreadable, or malformed — callers then fall back to login.
        """
        if not self._token_cache_path:
            return "", 0.0
        try:
            with open(self._token_cache_path, encoding="utf-8") as fh:
                data = json.load(fh)
            return str(data.get("token") or ""), float(data.get("expiry") or 0.0)
        except Exception:
            return "", 0.0

    def _write_cached_token(self, token: str, expiry_epoch: float) -> None:
        """Persist the service token to the on-disk cache when one is enabled.

        Best-effort and owner-only (0600); failures are ignored so a non-writable
        cache never blocks an otherwise successful login.
        """
        if not self._token_cache_path:
            return
        try:
            fd = os.open(self._token_cache_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"token": token, "expiry": expiry_epoch}, fh)
        except Exception:
            pass

    def _auth_headers(self) -> dict[str, str] | None:
        """Return authorization headers for authenticated API requests.

        Returns:
            A headers dictionary containing a bearer token if login succeeds,
            otherwise ``None``.
        """
        if not self._service_login():
            return None
        return {"Authorization": f"Bearer {self._token}"}

    def ping(self) -> tuple[bool, dict[str, Any]]:
        """Check whether the Job Manager API is reachable.

        Returns:
            A tuple ``(ok, data)`` where ``ok`` indicates request success and
            ``data`` contains the parsed API response.
        """
        ok, _, data = self._request_json("GET", "/ping")
        return ok, data

    def get_metrics(self) -> tuple[bool, dict[str, Any]]:
        """Fetch Job Manager metrics using service-account authentication.

        Returns:
            A tuple ``(ok, data)`` where ``ok`` indicates request success and
            ``data`` contains the parsed API response or an error payload.
        """
        headers = self._auth_headers()
        if not headers:
            return False, {"error": self._LOGIN_FAILED_ERROR}

        ok, _, data = self._request_json("GET", "/metrics", headers=headers)
        return ok, data

    def list_jobs_for_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """List jobs associated with a specific user.

        Args:
            user_id: Identifier of the user whose jobs should be fetched.
            limit: Maximum number of jobs to request.

        Returns:
            A tuple ``(ok, jobs)`` where ``jobs`` is a list of job dictionaries.
            On failure, ``jobs`` is an empty list.
        """
        headers = self._auth_headers()
        if not headers:
            return False, []

        safe_limit = max(self._USER_JOB_LIMIT_MIN, min(limit, self._USER_JOB_LIMIT_MAX))
        path = f"/jobs/user/{user_id}?offset=0&limit={safe_limit}"

        ok, _, data = self._request_json("GET", path, headers=headers)
        if not ok:
            return False, []

        jobs = data.get("jobs") if isinstance(data, dict) else []
        return True, jobs if isinstance(jobs, list) else []

    def list_jobs(self, limit: int = 200) -> tuple[bool, list[dict[str, Any]]]:
        """List jobs visible to the service account.

        Args:
            limit: Maximum number of jobs to request.

        Returns:
            A tuple ``(ok, jobs)`` where ``jobs`` is a list of job dictionaries.
            On failure, ``jobs`` is an empty list.
        """
        headers = self._auth_headers()
        if not headers:
            return False, []

        safe_limit = max(self._JOB_LIST_LIMIT_MIN, min(limit, self._JOB_LIST_LIMIT_MAX))
        path = f"/jobs?offset=0&limit={safe_limit}"

        ok, _, data = self._request_json("GET", path, headers=headers)
        if not ok:
            return False, []

        jobs = data.get("jobs") if isinstance(data, dict) else []
        return True, jobs if isinstance(jobs, list) else []

    def get_job_status(self, job_id: str) -> tuple[bool, dict[str, Any]]:
        """Fetch the current status for a job.

        Args:
            job_id: Job identifier.

        Returns:
            A tuple ``(ok, data)`` where ``data`` contains the parsed API
            response or an error payload.
        """
        headers = self._auth_headers()
        if not headers:
            return False, {"error": self._LOGIN_FAILED_ERROR}

        ok, _, data = self._request_json("GET", f"/job-status/{job_id}", headers=headers)
        return ok, data

    def cancel_job(self, job_id: str) -> tuple[bool, dict[str, Any]]:
        """Request cancellation of a queued or running job.

        Args:
            job_id: Job identifier.

        Returns:
            A tuple ``(ok, data)`` where ``data`` contains the parsed API
            response or an error payload.
        """
        headers = self._auth_headers()
        if not headers:
            return False, {"error": self._LOGIN_FAILED_ERROR}

        ok, _, data = self._request_json("POST", f"/jobs/{job_id}/cancel", headers=headers)
        return ok, data

    def delete_job(self, job_id: str) -> tuple[bool, dict[str, Any]]:
        """Delete a job row when it is known to be stale.

        Args:
            job_id: Job identifier.

        Returns:
            A tuple ``(ok, data)`` where ``data`` contains the parsed API
            response or an error payload.
        """
        headers = self._auth_headers()
        if not headers:
            return False, {"error": self._LOGIN_FAILED_ERROR}

        ok, _, data = self._request_json("DELETE", f"/jobs/{job_id}", headers=headers)
        return ok, data

    def submit_config_job(
        self,
        user_id: str,
        config: dict[str, Any],
        description: str = "",
        job_type: str = "faster_run",
        run_id: str = "",
    ) -> tuple[bool, dict[str, Any]]:
        """Submit a lightweight config snapshot file to the Job Manager queue.

        Args:
            user_id: Identifier of the submitting user.
            config: Configuration dictionary serialized as YAML and uploaded as
                ``run_config.yaml``.
            description: Optional human-readable job description.
            job_type: Job type forwarded to the Job Manager API.
            run_id: Optional external run identifier.

        Returns:
            A tuple ``(ok, data)`` where ``data`` contains the parsed API
            response or an error payload.
        """
        headers = self._auth_headers()
        if not headers:
            return False, {"error": self._LOGIN_FAILED_ERROR}

        file_bytes = yaml.safe_dump(config or {}, sort_keys=False).encode("utf-8")
        boundary = f"----faster-jm-{uuid.uuid4().hex}"
        parts: list[bytes] = []

        def add_field(name: str, value: str) -> None:
            """Append a standard multipart form-data field."""
            parts.append(f"--{boundary}\r\n".encode("utf-8"))
            parts.append(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
            )
            parts.append((value or "").encode("utf-8"))
            parts.append(b"\r\n")

        add_field("user_id", str(user_id or ""))
        add_field("source_app", self.source_app)
        add_field("description", str(description or ""))
        add_field("job_type", str(job_type or "faster_run"))
        add_field("run_id", str(run_id or ""))

        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(
            b'Content-Disposition: form-data; name="file"; filename="run_config.yaml"\r\n'
        )
        parts.append(b"Content-Type: application/x-yaml\r\n\r\n")
        parts.append(file_bytes)
        parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))

        body = b"".join(parts)
        req_headers = {
            "Accept": "application/json",
            "Authorization": headers["Authorization"],
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        req = request.Request(
            self._url("/submit-job"),
            data=body,
            method="POST",
            headers=req_headers,
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
                return True, payload if isinstance(payload, dict) else {}
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="ignore")
            try:
                payload = json.loads(raw) if raw else {}
            except Exception:
                payload = {"error": raw or str(exc)}

            if not isinstance(payload, dict):
                payload = {"error": raw or str(exc)}

            return False, payload
        except Exception as exc:
            return False, {"error": str(exc)}
