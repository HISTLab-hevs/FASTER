"""FastAPI runtime for Faster.

This module owns the HTTP surface for the web app while keeping the legacy
launcher available as a thin compatibility wrapper.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, DefaultDict, Callable

from fastapi import FastAPI, Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.staticfiles import StaticFiles

APP_PORT = 7860

_LOGGER = logging.getLogger("web_backend.app")

# API body cap in bytes. Set FL_API_BODY_MAX_BYTES=0 (default) for no limit.
_MAX_API_BODY_BYTES = int(os.getenv("FL_API_BODY_MAX_BYTES", "0") or "0")
# Optional streamed dataset upload cap in bytes. Set to 0 for no extra limit.
_MAX_CUSTOM_DATASET_UPLOAD_BYTES = int(
    os.getenv("FL_CUSTOM_DATASET_UPLOAD_MAX_BYTES", "0") or "0"
)
_CUSTOM_DATASET_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024

# Basic in-memory rate limiting for login attempts by client IP.
_LOGIN_WINDOW_S = 60
_LOGIN_MAX_ATTEMPTS = 10
_login_attempts: DefaultDict[str, Deque[float]] = defaultdict(deque)

# Static SPA paths.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WRAPPER_DIR = _PROJECT_ROOT / "wrapper"
_WRAPPER_INDEX = _WRAPPER_DIR / "index.html"

# Conservative validation for download route identifiers.
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{3,160}")


def _load_runtime_deps() -> tuple[Callable[[str, str], bool], Callable[[str], dict[str, Any]], Callable[[str], tuple[bool, str]], Any]:
    """Load runtime-facing helpers lazily to keep module import lightweight."""
    from web_backend.api_handlers import can_user_access_run, fl_api_handler, verify_access_token
    from web_backend.service import svc

    return can_user_access_run, fl_api_handler, verify_access_token, svc


def _with_security_headers(response: JSONResponse | FileResponse | RedirectResponse) -> Any:
    """Attach security headers to one outgoing HTTP response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: blob:; "
        "connect-src 'self' https://cdn.jsdelivr.net; "
        "font-src 'self' data: https://cdn.jsdelivr.net; "
        "frame-ancestors 'none';"
    )
    return response


def _json_response(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    """Create one JSON response with the app's standard security headers."""
    return _with_security_headers(JSONResponse(payload, status_code=status_code))


def _masked_json_error(
    message: str,
    *,
    status_code: int,
    log_message: str | None = None,
) -> JSONResponse:
    """Return one stable JSON error payload and optionally log the active exception."""
    if log_message:
        _LOGGER.exception(log_message)
    return _json_response({"error": message}, status_code=status_code)


def _resolve_bearer_token(request: Request) -> str:
    """Resolve one bearer token from the Authorization header."""
    auth_header = str(request.headers.get("authorization") or "").strip()
    if auth_header.lower().startswith("bearer "):
        bearer = auth_header[7:].strip()
        if bearer:
            return bearer
    return ""


def _custom_dataset_upload_payload(meta: dict[str, Any]) -> dict[str, Any]:
    """Build the stable response payload returned after one dataset upload."""
    return {
        "success": True,
        "dataset_ref": meta.get("dataset_ref"),
        "dataset_name": meta.get("dataset_name"),
        "dataset_type": meta.get("dataset_type"),
        "dataset_format": meta.get("dataset_format"),
        "rows": meta.get("rows"),
        "features": meta.get("features"),
        "classes": meta.get("classes"),
        "description": meta.get("description", ""),
    }


def _decode_multipart_value(value: bytes | None) -> str:
    """Decode one multipart field or file name to text."""
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace")


async def _stream_custom_dataset_upload_form(
    request: Request,
) -> tuple[str, str, str, int, str]:
    """Stream one multipart dataset upload directly to disk with minimal buffering."""
    from python_multipart.multipart import create_form_parser

    content_type = str(request.headers.get("content-type") or "")
    if "multipart/form-data" not in content_type.lower():
        raise ValueError("Dataset upload must use multipart form data")

    upload_tmp_dir = tempfile.mkdtemp(prefix=".custom_dataset_upload_")
    parsed_fields: dict[str, str] = {}
    parsed_files = []
    total_bytes = 0
    try:
        parser = create_form_parser(
            {
                "Content-Type": content_type,
                "Content-Length": str(request.headers.get("content-length") or ""),
            },
            lambda field: parsed_fields.__setitem__(
                _decode_multipart_value(field.field_name),
                _decode_multipart_value(field.value),
            ),
            parsed_files.append,
            config={
                "UPLOAD_DIR": upload_tmp_dir,
                "UPLOAD_DELETE_TMP": False,
                "UPLOAD_KEEP_EXTENSIONS": True,
                "MAX_MEMORY_FILE_SIZE": 0,
            },
        )
        async for chunk in request.stream():
            if not chunk:
                continue
            total_bytes += len(chunk)
            if (
                _MAX_CUSTOM_DATASET_UPLOAD_BYTES > 0
                and total_bytes > _MAX_CUSTOM_DATASET_UPLOAD_BYTES
            ):
                raise ValueError("Dataset upload too large")
            for offset in range(0, len(chunk), _CUSTOM_DATASET_UPLOAD_CHUNK_BYTES):
                parser.write(chunk[offset : offset + _CUSTOM_DATASET_UPLOAD_CHUNK_BYTES])
        parser.finalize()
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Invalid multipart dataset upload") from error
    finally:
        for item in parsed_files:
            try:
                item.close()
            except Exception:
                _LOGGER.warning("Failed to close streamed multipart upload file", exc_info=True)
        if not parsed_files:
            shutil.rmtree(upload_tmp_dir, ignore_errors=True)

    upload_files = [
        item
        for item in parsed_files
        if _decode_multipart_value(item.field_name) == "file"
    ]
    if not upload_files:
        shutil.rmtree(upload_tmp_dir, ignore_errors=True)
        raise ValueError("Dataset file is required")
    if len(upload_files) != 1:
        shutil.rmtree(upload_tmp_dir, ignore_errors=True)
        raise ValueError("Exactly one dataset file must be uploaded")

    upload_file = upload_files[0]
    tmp_path = os.fsdecode(upload_file.actual_file_name or b"")
    if not tmp_path:
        shutil.rmtree(upload_tmp_dir, ignore_errors=True)
        raise ValueError("Failed to stage uploaded dataset file")
    return (
        _decode_multipart_value(upload_file.file_name) or os.path.basename(tmp_path),
        parsed_fields.get("description", ""),
        tmp_path,
        total_bytes,
        upload_tmp_dir,
    )


def _prune_login_attempts(client_ip: str, now: float) -> Deque[float]:
    """Prune expired login attempts for one client and return the active window."""
    attempts = _login_attempts[client_ip]
    while attempts and now - attempts[0] > _LOGIN_WINDOW_S:
        attempts.popleft()
    return attempts


def create_app() -> FastAPI:
    """Create the target FastAPI app and register the current HTTP surface."""
    app = FastAPI(title="Faster Web Backend")

    def _log_runtime_startup() -> None:
        _LOGGER.info(
            "Starting Faster FastAPI runtime on port %s; wrapper_dir=%s",
            APP_PORT,
            _WRAPPER_DIR,
        )

    def _seed_admin_from_env() -> None:
        """Seed the bootstrap accounts from env on first startup if absent.

        Two distinct identities are created so that human and machine roles do
        not share a credential:

        1. The human bootstrap admin (``FASTER_ADMIN_*``), used to sign in to
           the dashboard.
        2. A dedicated Job Manager service account (``JM_SERVICE_USERNAME`` /
           ``JM_SERVICE_PASSWORD``), used only for the single ``faster-app`` →
           Job Manager connection. Decoupling it means human admins can be
           renamed, added, or deleted without ever breaking that channel.

        Both services share one ``users`` table and the schema seeds none, so
        seeding here makes a fresh deployment work without a manual
        ``auth.user_cli create-admin`` step.

        The seed is idempotent and conservative: it runs only when the users
        table is empty, so it never overrides an existing deployment's accounts.
        """
        admin_password = str(os.getenv("FASTER_ADMIN_PASSWORD", "") or "")
        svc_username = str(os.getenv("JM_SERVICE_USERNAME", "") or "").strip()
        svc_password = str(os.getenv("JM_SERVICE_PASSWORD", "") or "")

        # Nothing requested → nothing to do.
        if not admin_password and not (svc_username and svc_password):
            return

        admin_username = str(os.getenv("FASTER_ADMIN_USERNAME", "admin") or "admin").strip()
        admin_email = str(os.getenv("FASTER_ADMIN_EMAIL", "") or "") or None

        import secrets as _secrets

        from auth.authenticator import Authenticator

        # The secret only signs JWTs; seeding just writes a row, so a per-call
        # fallback is fine when FL_SECRET_KEY is unset (mirrors user_cli).
        secret = os.getenv("FL_SECRET_KEY") or _secrets.token_urlsafe(16)
        auth = Authenticator(secret_key=secret)
        if not auth._db_ready():
            _LOGGER.warning(
                "Account seed skipped: auth DB backend not ready (%s)",
                auth.backend_error(),
            )
            return

        # Only seed into a fresh deployment; never touch existing accounts.
        if auth.list_users():
            return

        def _seed_account(label: str, username: str, password: str, env_var: str) -> None:
            if not username or not password:
                return
            ok = auth.register_user(
                username=username,
                password=password,
                role="admin",
                email=admin_email if label == "admin" else None,
                created_by="bootstrap",
            )
            if ok:
                _LOGGER.info("Seeded %s '%s' on empty users table", label, username)
                return
            valid, errors = auth.validate_password(password)
            if not valid:
                _LOGGER.warning(
                    "%s seed failed: %s rejected (%s)",
                    label,
                    env_var,
                    "; ".join(errors),
                )
            else:
                _LOGGER.warning("%s seed failed for '%s'", label, username)

        _seed_account("admin", admin_username, admin_password, "FASTER_ADMIN_PASSWORD")
        # Skip the service account when it would collide with the admin row
        # (single shared identity); otherwise seed a dedicated one.
        if svc_username and svc_username != admin_username:
            _seed_account(
                "Job Manager service account",
                svc_username,
                svc_password,
                "JM_SERVICE_PASSWORD",
            )

    if _WRAPPER_DIR.is_dir():
        app.mount(
            "/wrapper",
            StaticFiles(directory=str(_WRAPPER_DIR), html=True),
            name="wrapper-spa",
        )
    else:
        _LOGGER.warning("wrapper directory not found at %s", _WRAPPER_DIR)

    if not _WRAPPER_INDEX.exists():
        _LOGGER.warning("wrapper index not found at %s", _WRAPPER_INDEX)

    app.add_event_handler("startup", _log_runtime_startup)
    app.add_event_handler("startup", _seed_admin_from_env)

    @app.get("/", include_in_schema=False)
    async def spa_redirect_root(request: Request):
        """Redirect root requests to the SPA sign-in route."""
        del request
        return _with_security_headers(
            RedirectResponse(url="/Faster/signin", status_code=307)
        )

    @app.get("/Faster/{path:path}", include_in_schema=False)
    async def spa_root(request: Request, path: str):
        """Serve the SPA index file for direct root or history-based navigation."""
        del request, path
        if not _WRAPPER_INDEX.exists():
            return _json_response({"error": f"Missing {_WRAPPER_INDEX}"}, status_code=500)
        return _with_security_headers(
            FileResponse(str(_WRAPPER_INDEX), media_type="text/html")
        )

    @app.post("/api", include_in_schema=False)
    async def api_endpoint(request: Request):
        """Validate and dispatch wrapper API requests through ``fl_api_handler``."""
        try:
            content_length = request.headers.get("content-length")
            if content_length:
                if _MAX_API_BODY_BYTES > 0 and int(content_length) > _MAX_API_BODY_BYTES:
                    return _json_response({"error": "Payload too large"}, status_code=413)

            body = await request.json()
            if not isinstance(body, dict):
                return _json_response({"error": "Invalid API body"}, status_code=400)

            client_ip = request.client.host if request.client else "unknown"

            if body.get("command") == "login":
                now = time.time()
                attempts = _prune_login_attempts(client_ip, now)
                if len(attempts) >= _LOGIN_MAX_ATTEMPTS:
                    return _json_response(
                        {"error": "Too many login attempts, retry later"},
                        status_code=429,
                    )
                attempts.append(now)

            _, fl_api_handler, _, _ = _load_runtime_deps()
            result = fl_api_handler(json.dumps(body))
            return _json_response(result)
        except Exception:
            return _masked_json_error(
                "Internal server error",
                status_code=500,
                log_message="Unhandled exception in /api endpoint",
            )

    @app.post("/api/upload-custom-dataset", include_in_schema=False)
    async def upload_custom_dataset_endpoint(request: Request):
        """Handle multipart custom-dataset uploads with streamed temporary-file staging."""
        tmp_path = ""
        upload_tmp_dir = ""
        request_start = time.perf_counter()
        try:
            content_length = request.headers.get("content-length")
            if content_length:
                if (
                    _MAX_CUSTOM_DATASET_UPLOAD_BYTES > 0
                    and int(content_length) > _MAX_CUSTOM_DATASET_UPLOAD_BYTES
                ):
                    return _json_response(
                        {"error": "Dataset upload too large"},
                        status_code=413,
                    )

            _, _, verify_access_token, svc = _load_runtime_deps()
            auth_token = _resolve_bearer_token(request)
            valid, username = verify_access_token(auth_token)
            if not valid:
                return _json_response({"error": "Unauthorized"}, status_code=401)

            upload_start = time.perf_counter()
            (
                original_name,
                description,
                tmp_path,
                total_bytes,
                upload_tmp_dir,
            ) = await _stream_custom_dataset_upload_form(request)

            ingest_start = time.perf_counter()
            meta = svc.save_custom_dataset_file(
                username,
                original_name,
                tmp_path,
                description=description,
            )
            _LOGGER.info(
                "Custom dataset upload '%s' for user '%s': receive=%.3fs ingest=%.3fs total=%.3fs size_mb=%.2f",
                original_name,
                username,
                ingest_start - upload_start,
                time.perf_counter() - ingest_start,
                time.perf_counter() - request_start,
                total_bytes / (1024 * 1024) if total_bytes else 0.0,
            )
            return _json_response(_custom_dataset_upload_payload(meta))
        except ValueError as exc:
            return _json_response({"error": str(exc)}, status_code=400)
        except Exception:
            return _masked_json_error(
                "Failed to process custom dataset",
                status_code=500,
                log_message="Unhandled exception in streamed custom dataset upload endpoint",
            )
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except FileNotFoundError:
                    pass
                except Exception:
                    _LOGGER.warning(
                        "Failed to remove temporary upload file '%s'",
                        tmp_path,
                        exc_info=True,
                    )
            if upload_tmp_dir:
                shutil.rmtree(upload_tmp_dir, ignore_errors=True)

    @app.get("/fl-download/{run_id}.zip", include_in_schema=False)
    async def download_zip(request: Request, run_id: str):
        """Stream a run archive after token and ownership checks."""
        try:
            if not _RUN_ID_PATTERN.fullmatch(run_id):
                return _json_response({"error": "Invalid run id"}, status_code=400)

            can_user_access_run, _, verify_access_token, svc = _load_runtime_deps()
            auth_token = _resolve_bearer_token(request)
            valid, username = verify_access_token(auth_token)
            if not valid:
                return _json_response({"error": "Unauthorized"}, status_code=401)

            if not can_user_access_run(username, run_id):
                return _json_response({"error": "Forbidden"}, status_code=403)

            zip_path = svc.export_run_to_zip(run_id)
        except Exception:
            return _masked_json_error(
                "Failed to export run archive",
                status_code=500,
                log_message=f"Unhandled exception while exporting archive for run '{run_id}'",
            )

        return _with_security_headers(
            FileResponse(
                zip_path,
                media_type="application/zip",
                filename=f"{run_id}.zip",
            )
        )

    return app


app = create_app()

__all__ = ["APP_PORT", "app", "create_app"]
