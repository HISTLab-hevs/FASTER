"""API dispatcher and shared wiring for the backend handler package."""

from __future__ import annotations

import json
import logging
import os
import secrets

from auth.authenticator import Authenticator
from .admin import handle_admin_command
from .auth import handle_account_command, handle_public_auth_command
from .content import handle_content_command
from .run_workflows import (
    attach_custom_dataset_path,
    heal_best_accuracy,
    is_active_job_entry,
    normalized_jobs_snapshot,
    reconcile_running_runs,
    resolve_job_id_for_cancel,
    submit_run_to_job_manager,
    validate_run_config,
)
from .runs import handle_run_command
from .shared import (
    build_admin_analytics,
    clean_nan_inf,
    error_payload,
    is_admin_role,
    is_user_run_visible,
    is_valid_dataset_ref,
    is_valid_job_id,
    is_valid_run_id,
    log_and_mask_exception,
    matches_tokens,
    row_summary_from_meta,
    run_path,
    sanitize_config_obj,
    search_tokens,
)
from web_backend.jm_client import JobManagerClient
from web_backend.service import BASE_RESULTS_PATH, svc, validate_run_display_name

try:
    import GPUtil

    _GPUTIL = True
except ImportError:
    _GPUTIL = False


_SECRET = os.getenv("FL_SECRET_KEY", secrets.token_urlsafe(32))
_auth = Authenticator(secret_key=_SECRET)
_jm = JobManagerClient()

ADMIN_ANALYTICS_MAX_RUNS = 300
ADMIN_ANALYTICS_RESOURCE_SCAN_LIMIT = 120
ADMIN_ANALYTICS_TOP_USERS = 8
ADMIN_ANALYTICS_TOP_METHODS = 6
ADMIN_ANALYTICS_TOP_DATASETS = 6

logger = logging.getLogger("web_backend.api_handlers")


def verify_access_token(token: str):
    """Verify an auth token."""
    return _auth.verify_token(token)


def can_user_access_run(username: str, run_id: str) -> bool:
    """Return whether a user can access one run."""
    user = _auth.get_user(username) or {}
    role = str(user.get("role") or "user")
    return svc.user_can_access_run(username, run_id, is_admin=is_admin_role(role))


def _run_path(name: str) -> str:
    """Return the results path for one run name."""
    return run_path(BASE_RESULTS_PATH, name)


def _is_user_run_visible(username: str, run_id: str, role: str = "user") -> bool:
    """Return whether a user may see one run."""
    return is_user_run_visible(username, run_id, role=role, svc=svc)


def _validate_run_config(cfg: dict) -> tuple[bool, str]:
    """Validate one run configuration payload against backend rules."""
    return validate_run_config(
        cfg,
        validate_run_display_name=validate_run_display_name,
        is_valid_dataset_ref=is_valid_dataset_ref,
    )


def _attach_custom_dataset_path(
    cfg: dict,
    username: str,
    is_admin: bool = False,
) -> tuple[bool, str]:
    """Resolve custom dataset references for a run submission payload."""
    return attach_custom_dataset_path(
        cfg,
        username,
        svc=svc,
        is_valid_dataset_ref=is_valid_dataset_ref,
        is_admin=is_admin,
    )


def _row_summary_from_meta(meta: dict) -> dict:
    """Build one API run row from service metadata."""
    return row_summary_from_meta(meta, svc=svc)


def _heal_best_accuracy(run_id: str) -> None:
    """Backfill best-accuracy metadata from persisted metrics when available."""
    heal_best_accuracy(run_id, svc=svc, run_path=_run_path, logger=logger)


def _reconcile_running_runs(username: str, is_admin: bool, include_all: bool) -> dict:
    """Reconcile active run metadata with Job Manager state."""
    return reconcile_running_runs(
        username,
        is_admin=is_admin,
        include_all=include_all,
        svc=svc,
        jm=_jm,
        logger=logger,
        heal_best_accuracy_fn=_heal_best_accuracy,
    )


def _normalized_jobs_snapshot(username: str, is_admin: bool = False) -> list[dict]:
    """Return normalized active jobs with role-aware visibility."""
    return normalized_jobs_snapshot(
        username,
        svc=svc,
        jm=_jm,
        run_path=_run_path,
        is_admin=is_admin,
    )


def _build_admin_analytics(run_rows: list, jobs: list) -> dict:
    """Build aggregate analytics for the admin overview."""
    return build_admin_analytics(
        run_rows,
        jobs,
        svc=svc,
        resource_scan_limit=ADMIN_ANALYTICS_RESOURCE_SCAN_LIMIT,
        top_users=ADMIN_ANALYTICS_TOP_USERS,
        top_methods=ADMIN_ANALYTICS_TOP_METHODS,
        top_datasets=ADMIN_ANALYTICS_TOP_DATASETS,
    )


def _submit_run_to_job_manager(
    config: dict,
    username: str,
    is_admin: bool = False,
) -> tuple[bool, dict]:
    """Validate, reserve, and submit a run through the Job Manager."""
    return submit_run_to_job_manager(
        config,
        username,
        svc=svc,
        jm=_jm,
        validate_run_config_fn=_validate_run_config,
        attach_custom_dataset_path_fn=_attach_custom_dataset_path,
        is_admin=is_admin,
    )


def _resolve_job_id_for_cancel(
    identifier: str,
    username: str,
    is_admin: bool,
) -> tuple[str, str | None]:
    """Resolve a UI cancel target to a concrete Job Manager job id."""
    return resolve_job_id_for_cancel(
        identifier,
        username,
        is_admin=is_admin,
        jm=_jm,
        is_valid_job_id=is_valid_job_id,
        is_valid_run_id=is_valid_run_id,
    )


def fl_api_handler(request_json: str) -> dict:
    """Dispatch one SPA API call and return a JSON-serializable response."""
    try:
        req = json.loads(request_json)
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid JSON request"}

    if not isinstance(req, dict):
        return {"error": "Invalid JSON request"}

    command = req.get("command", "")
    token = req.get("token", "")
    data = req.get("data", {})

    if not isinstance(data, dict):
        return {"error": "Invalid JSON request"}

    public_response = handle_public_auth_command(
        command,
        data,
        {"auth": _auth, "is_admin_role": is_admin_role},
    )
    if public_response is not None:
        return public_response

    try:
        valid, claims = _auth.verify_token_claims(token)
    except Exception:
        return log_and_mask_exception(
            logger,
            user_message="Internal server error",
            log_message=f"Unhandled API error while processing command '{locals().get('command', 'unknown')}'",
        )
    if not valid:
        return {"error": "Unauthorized"}

    username = str(claims.get("identifier") or "")
    role = str(claims.get("role") or "user")
    is_admin = is_admin_role(role)

    ctx = {
        "auth": _auth,
        "claims": claims,
        "jm": _jm,
        "svc": svc,
        "username": username,
        "role": role,
        "is_admin": is_admin,
        "is_admin_role": is_admin_role,
        "validate_run_id": is_valid_run_id,
        "validate_job_id": is_valid_job_id,
        "validate_run_display_name": validate_run_display_name,
        "is_user_run_visible": _is_user_run_visible,
        "run_path": _run_path,
        "sanitize_config_obj": sanitize_config_obj,
        "submit_run_to_job_manager": _submit_run_to_job_manager,
        "resolve_job_id_for_cancel": _resolve_job_id_for_cancel,
        "normalized_jobs_snapshot": _normalized_jobs_snapshot,
        "is_active_job_entry": is_active_job_entry,
        "row_summary_from_meta": _row_summary_from_meta,
        "reconcile_running_runs": _reconcile_running_runs,
        "search_tokens": search_tokens,
        "matches_tokens": matches_tokens,
        "build_admin_analytics": _build_admin_analytics,
        "clean_nan_inf": clean_nan_inf,
        "logger": logger,
        "gputil_enabled": _GPUTIL,
        "gputil_module": GPUtil if _GPUTIL else None,
        "admin_analytics_max_runs": ADMIN_ANALYTICS_MAX_RUNS,
    }

    try:
        for handler in (
            handle_account_command,
            handle_run_command,
            handle_content_command,
            handle_admin_command,
        ):
            response = handler(command, data, ctx)
            if response is not None:
                return response
        return error_payload(f"Unknown command: '{command}'")
    except Exception:
        return log_and_mask_exception(
            logger,
            user_message="Internal server error",
            log_message=f"Unhandled API error while processing command '{locals().get('command', 'unknown')}'",
        )
