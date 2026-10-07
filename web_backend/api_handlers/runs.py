"""Run-oriented API handlers.

This module owns commands that expose persisted run state, history, and
Job-Manager-backed lifecycle actions. It relies on shared helpers prepared by
``web_backend.api_handlers`` so command behavior stays stable while domain logic
is easier to navigate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import re


_RUN_TS_PATTERN = re.compile(r"(?:^|_)(\d{8})_(\d{6})(?:_|$)")


def _coerce_int(
    value: Any,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Coerce one value to int and clamp it to the configured range."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _extract_run_sort_ts(row: dict[str, Any]) -> int:
    """Extract a sortable run timestamp from metadata or the run name."""
    created_at = str(row.get("created_at") or "").strip()
    if created_at:
        try:
            return int(datetime.fromisoformat(created_at).timestamp() * 1000)
        except Exception:
            pass

    name = str(row.get("name") or row.get("id") or "")
    match = _RUN_TS_PATTERN.search(name)
    if not match:
        return 0
    try:
        return int(f"{match.group(1)}{match.group(2)}")
    except Exception:
        return 0


def _row_matches_history_search(
    row: dict[str, Any],
    query: str,
    field: str,
    *,
    can_see_owner: bool,
) -> bool:
    """Return whether one history row matches the requested search filter."""
    if not query:
        return True

    fields = {
        "name": str(row.get("name") or ""),
        "method": str(row.get("method") or ""),
        "dataset": str(row.get("dataset") or ""),
        "model_name": str(row.get("model_name") or ""),
        "status": str(row.get("status") or ""),
    }
    if can_see_owner:
        fields["owner"] = str(row.get("owner") or "")

    effective_field = "all" if (field == "owner" and not can_see_owner) else field
    if effective_field == "all":
        return query in " ".join(fields.values()).lower()
    return query in str(fields.get(effective_field) or "").lower()


def _sort_history_rows(rows: list[dict[str, Any]], sort_key: str, sort_dir: str) -> list[dict[str, Any]]:
    """Sort history rows using the requested key and direction."""
    key = str(sort_key or "run_ts")
    reverse = str(sort_dir or "desc").lower() == "desc"

    def _best_accuracy(val: Any) -> float | None:
        """Coerce one best-accuracy value to float when possible."""
        try:
            return float(val) if val is not None else None
        except Exception:
            return None

    def _sort_value(row: dict[str, Any]) -> tuple[int, Any]:
        """Return the normalized sort tuple for one history row."""
        if key == "best_accuracy":
            val = _best_accuracy(row.get("best_accuracy"))
            return (1 if val is None else 0, val if val is not None else 0.0)
        if key == "run_ts":
            val = _extract_run_sort_ts(row)
            return (1 if not val else 0, val)
        return (0, str(row.get(key) or "").lower())

    return sorted(rows, key=_sort_value, reverse=reverse)


def _paginate_rows(rows: list[dict[str, Any]], page: int, page_size: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Slice rows for one page and return matching pagination metadata."""
    total = len(rows)
    pages = max(1, (total + page_size - 1) // page_size)
    safe_page = min(max(1, page), pages)
    start = (safe_page - 1) * page_size
    end = start + page_size
    return rows[start:end], {
        "page": safe_page,
        "page_size": page_size,
        "pages": pages,
        "total": total,
    }


def _active_job_for_run(jobs: list[Any], identifier: str) -> dict[str, Any] | None:
    """Return the active Job Manager row for a run or job identifier, if present."""
    ident = str(identifier or "")
    return next(
        (
            job
            for job in jobs
            if isinstance(job, dict)
            if str(job.get("status") or "").lower() in {"queued", "pending", "running"}
            and (
                str(job.get("run_id") or "") == ident
                or str(job.get("job_id") or "") == ident
            )
        ),
        None,
    )


def _job_cancel_failure_message(payload: Any) -> str:
    """Return the stable user-facing cancel failure message."""
    payload = payload or {}
    return str(
        payload.get("error")
        or payload.get("detail")
        or "Failed to cancel Job Manager job"
    )


def _cancel_active_job_manager_run(
    *,
    jm: Any,
    svc: Any,
    run_id: str,
    username: str,
    is_admin: bool,
    job_id: str = "",
) -> dict[str, Any]:
    """Cancel an active Job Manager run or return the standard failure payload."""
    identifier = str(run_id or job_id or "").strip()

    def _delete_stale_job(row_job_id: str) -> None:
        """Best-effort removal for a stale Job Manager row that no longer runs."""
        candidate = str(row_job_id or "").strip()
        if not candidate or not hasattr(jm, "delete_job"):
            return
        try:
            jm.delete_job(candidate)
        except Exception:
            pass

    ok_jobs, jobs = (
        jm.list_jobs(limit=300)
        if is_admin
        else jm.list_jobs_for_user(username, limit=300)
    )
    active = _active_job_for_run(jobs, identifier) if ok_jobs and isinstance(jobs, list) else None
    resolved_job_id = str((active or {}).get("job_id") or "").strip()
    active_run_id = str((active or {}).get("run_id") or "")
    # Only mark the Faster run row when we can tie the job back to a run_id.
    mark_run_id = active_run_id or str(run_id or "")

    if not resolved_job_id:
        # No matching job found in Job Manager — treat as a stale active entry.
        if mark_run_id:
            svc.mark_run_status(mark_run_id, "stopped", "Stopped by user (job missing)")
        _delete_stale_job(job_id or identifier)
        return {
            "success": True,
            "message": "Stopped (no active job in Job Manager)",
        }

    ok_cancel, payload = jm.cancel_job(resolved_job_id)
    if not ok_cancel:
        message = _job_cancel_failure_message(payload)
        lowered = message.lower()
        if any(token in lowered for token in ("process not found", "job not found", "already exited")):
            if mark_run_id:
                svc.mark_run_status(mark_run_id, "stopped", "Stopped by user")
            _delete_stale_job(resolved_job_id)
            return {
                "success": True,
                "message": "Stopped",
            }
        return {"success": False, "message": message}

    if mark_run_id:
        svc.mark_run_status(mark_run_id, "stopped", "Stopped by user")
    return {
        "success": True,
        "message": str((payload or {}).get("message") or "Stopped"),
    }


def _validate_visible_run_id(
    run_id: str,
    *,
    username: str,
    role: str,
    validate_run_id: Any,
    is_user_run_visible: Any,
) -> dict[str, str] | None:
    """Validate one run id and its role-scoped visibility."""
    if run_id and not validate_run_id(run_id):
        return {"error": "Invalid run id"}
    if run_id and not is_user_run_visible(username, run_id, role=role):
        return {"error": "Forbidden"}
    return None


def _handle_get_runs(
    data: dict[str, Any],
    *,
    svc: Any,
    username: str,
    is_admin: bool,
    row_summary_from_meta: Any,
    reconcile_running_runs: Any,
) -> dict[str, Any]:
    """Handle the paged and unpaged run history listing command."""
    include_all = bool(data.get("all_users", False)) and is_admin
    refresh_db = bool(data.get("refresh_db", False))
    paged = any(key in data for key in ("page", "page_size", "sort_key", "sort_dir", "search_field"))
    reconcile_report = None

    if refresh_db:
        reconcile_report = reconcile_running_runs(
            username,
            is_admin=is_admin,
            include_all=include_all,
        )

    search_query = str(data.get("search", "") or "").strip().lower()
    search_field = str(data.get("search_field", "all") or "all").strip().lower()
    run_meta = svc.list_run_metadata(username, include_all=include_all)
    rows = []
    for meta in run_meta or []:
        row = row_summary_from_meta(meta)
        if str(row.get("id") or "").strip():
            rows.append(row)
    rows = [
        row
        for row in rows
        if _row_matches_history_search(row, search_query, search_field, can_see_owner=is_admin)
    ]

    if paged:
        sort_key = str(data.get("sort_key", "run_ts") or "run_ts")
        sort_dir = str(data.get("sort_dir", "desc") or "desc")
        page = _coerce_int(data.get("page", 1), default=1, minimum=1)
        page_size = _coerce_int(
            data.get("page_size", 10),
            default=10,
            minimum=1,
            maximum=100,
        )
        rows = _sort_history_rows(rows, sort_key, sort_dir)
        paged_rows, pagination = _paginate_rows(rows, page, page_size)
        response = {"success": True, "runs": paged_rows, "pagination": pagination}
    else:
        response = {"success": True, "runs": rows}
    if reconcile_report is not None:
        response["reconcile"] = reconcile_report
    return response


def _handle_get_runs_by_ids(
    data: dict[str, Any],
    *,
    svc: Any,
    username: str,
    role: str,
    validate_run_id: Any,
    is_user_run_visible: Any,
    row_summary_from_meta: Any,
) -> dict[str, Any]:
    """Handle run summary lookups for a bounded list of run ids."""
    ids_raw = data.get("ids", [])
    if not isinstance(ids_raw, list):
        return {"error": "ids must be a list"}

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in ids_raw[:20]:
        run_id = str(raw or "").strip()
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        if not validate_run_id(run_id):
            continue
        if not is_user_run_visible(username, run_id, role=role):
            continue

        meta = svc.get_run_metadata(run_id) or {}
        if not meta:
            continue
        meta["run_id"] = str(meta.get("run_id") or run_id)
        row = row_summary_from_meta(meta)
        if str(row.get("id") or "").strip():
            rows.append(row)

    return {"success": True, "runs": rows}


def _handle_validate_run_names(
    data: dict[str, Any],
    *,
    svc: Any,
    username: str,
    validate_run_display_name: Any,
) -> dict[str, Any]:
    """Handle conflict checks for prospective run display names."""
    names_raw = data.get("names", [])
    if not isinstance(names_raw, list):
        return {"error": "names must be a list"}

    names = []
    for raw in names_raw:
        ok_name, msg_name, normalized_name = validate_run_display_name(
            raw,
            field_name="run_name",
            required=False,
        )
        if not ok_name:
            return {"error": msg_name}
        if not normalized_name:
            continue
        names.append(normalized_name)

    conflicts = svc.find_run_name_conflicts(names, username=username)
    return {"success": True, "conflicts": conflicts}


def _handle_get_run_status(
    data: dict[str, Any],
    *,
    svc: Any,
    username: str,
    role: str,
    validate_run_id: Any,
    is_user_run_visible: Any,
) -> dict[str, Any]:
    """Handle status lookups for one visible run."""
    run_id = data.get("run_id", "")
    run_err = _validate_visible_run_id(
        run_id,
        username=username,
        role=role,
        validate_run_id=validate_run_id,
        is_user_run_visible=is_user_run_visible,
    )
    if run_err:
        return run_err
    status = str(svc.get_run_status(run_id) or "unknown")
    return {
        "success": True,
        "status": status,
        "active": status in {"queued", "running"},
        "running": status == "running",
        "stop_reason": svc.get_run_stop_reason(run_id),
    }


def _handle_start_or_queue_run(
    command: str,
    data: dict[str, Any],
    *,
    username: str,
    is_admin: bool,
    sanitize_config_obj: Any,
    submit_run_to_job_manager: Any,
) -> dict[str, Any]:
    """Handle run submission for immediate and queued execution flows."""
    config = sanitize_config_obj(data.get("config", {}))
    ok_submit, submit = submit_run_to_job_manager(config, username, is_admin=is_admin)
    if not ok_submit:
        return {"error": submit.get("error") or "Job Manager submission failed"}

    if command == "start_run":
        return {
            "success": True,
            "run_id": submit.get("run_id"),
            "job_id": submit.get("job_id"),
            "queued": True,
            "started": False,
            "message": "Queued in Job Manager",
        }

    jm_payload = (
        submit.get("jm_payload")
        if isinstance(submit.get("jm_payload"), dict)
        else {}
    )
    return {
        "success": True,
        "queued": True,
        "started": False,
        "run_id": submit.get("run_id"),
        "job_id": submit.get("job_id"),
        "position": None,
        "message": "Queued in Job Manager",
        "job_manager": {
            "enabled": True,
            "submitted": True,
            "job_id": submit.get("job_id"),
            "status": jm_payload.get("status"),
            "error": None,
        },
    }


def _handle_stop_run(
    data: dict[str, Any],
    *,
    svc: Any,
    jm: Any,
    username: str,
    role: str,
    is_admin: bool,
    validate_run_id: Any,
    is_user_run_visible: Any,
) -> dict[str, Any]:
    """Handle stopping one active run via Job Manager."""
    run_id = str(data.get("run_id", "") or "").strip()
    job_id = str(data.get("job_id", "") or "").strip()
    identifier = run_id or job_id
    if identifier and not validate_run_id(identifier):
        return {"error": "Invalid run id"}
    # Block stopping a run the caller cannot see (would otherwise corrupt another user's row).
    if run_id and not is_user_run_visible(username, run_id, role=role):
        return {"error": "Forbidden"}

    if not (jm.enabled and identifier):
        return {
            "success": False,
            "message": "Job Manager integration is required to stop runs.",
        }

    return _cancel_active_job_manager_run(
        jm=jm,
        svc=svc,
        run_id=run_id,
        job_id=job_id,
        username=username,
        is_admin=is_admin,
    )


def _handle_get_metrics(
    data: dict[str, Any],
    *,
    svc: Any,
    username: str,
    role: str,
    validate_run_id: Any,
    is_user_run_visible: Any,
    run_path: Any,
    clean_nan_inf: Any,
) -> dict[str, Any]:
    """Handle metrics retrieval with resource summary backfilling."""
    run_id = data.get("run_id", "")
    run_err = _validate_visible_run_id(
        run_id,
        username=username,
        role=role,
        validate_run_id=validate_run_id,
        is_user_run_visible=is_user_run_visible,
    )
    if run_err:
        return run_err

    try:
        metrics = svc.load_metrics(run_path(run_id)) or {}
        resource_usage = svc.load_run_resource_usage(run_id) or {}
        summary = resource_usage.get("summary") if isinstance(resource_usage, dict) else {}

        if isinstance(summary, dict) and summary:
            merged = dict(metrics.get("resource_usage") or {})
            merged.update(summary)
            merged["samples_count"] = summary.get(
                "samples_count",
                merged.get("samples_count"),
            )
            metrics["resource_usage"] = merged
            if merged.get("vram_used_gb_avg") is not None:
                metrics["vram_avg_gb"] = merged.get("vram_used_gb_avg")

        return {"success": True, "metrics": clean_nan_inf(metrics)}
    except Exception:
        return {"success": True, "metrics": {}}


def _handle_get_logs(
    data: dict[str, Any],
    *,
    svc: Any,
    username: str,
    role: str,
    validate_run_id: Any,
    is_user_run_visible: Any,
    run_path: Any,
) -> dict[str, Any]:
    """Handle log tail retrieval for one visible run."""
    run_id = data.get("run_id", "")
    run_err = _validate_visible_run_id(
        run_id,
        username=username,
        role=role,
        validate_run_id=validate_run_id,
        is_user_run_visible=is_user_run_visible,
    )
    if run_err:
        return run_err

    n = _coerce_int(data.get("n", 100), default=100, minimum=1, maximum=2000)

    try:
        logs = svc.read_logs(run_path(run_id), n)
    except Exception:
        logs = ""

    return {"success": True, "logs": logs}


def _handle_get_run_config(
    data: dict[str, Any],
    *,
    svc: Any,
    username: str,
    role: str,
    validate_run_id: Any,
    is_user_run_visible: Any,
    run_path: Any,
) -> dict[str, Any]:
    """Handle persisted run-config retrieval for one visible run."""
    run_id = data.get("run_id", "")
    run_err = _validate_visible_run_id(
        run_id,
        username=username,
        role=role,
        validate_run_id=validate_run_id,
        is_user_run_visible=is_user_run_visible,
    )
    if run_err:
        return run_err

    try:
        cfg = svc.load_run_config(run_path(run_id)) or {}
    except Exception:
        cfg = {}

    return {"success": True, "config": cfg}


def _handle_delete_run(
    data: dict[str, Any],
    *,
    svc: Any,
    username: str,
    role: str,
    validate_run_id: Any,
    is_user_run_visible: Any,
) -> dict[str, Any]:
    """Handle deleting one run and its persisted artifacts."""
    run_id = data.get("run_id", "")
    if run_id:
        run_err = _validate_visible_run_id(
            run_id,
            username=username,
            role=role,
            validate_run_id=validate_run_id,
            is_user_run_visible=is_user_run_visible,
        )
        if run_err:
            return run_err
    elif not is_user_run_visible(username, run_id, role=role):
        return {"error": "Forbidden"}

    ok = svc.delete_run(run_id)
    if not ok:
        return {"error": f"Run '{run_id}' not found or already deleted"}

    return {"success": True, "message": f"Deleted '{run_id}'"}


def _handle_rename_run(
    data: dict[str, Any],
    *,
    svc: Any,
    username: str,
    role: str,
    validate_run_id: Any,
    is_user_run_visible: Any,
    validate_run_display_name: Any,
) -> dict[str, Any]:
    """Handle renaming one visible run."""
    run_id = data.get("run_id", "")
    new_name = data.get("new_name", "")
    if run_id:
        run_err = _validate_visible_run_id(
            run_id,
            username=username,
            role=role,
            validate_run_id=validate_run_id,
            is_user_run_visible=is_user_run_visible,
        )
        if run_err:
            return run_err
    elif not is_user_run_visible(username, run_id, role=role):
        return {"error": "Forbidden"}

    if not new_name or not isinstance(new_name, str):
        return {"error": "new_name is required"}
    ok_name, msg_name, normalized_name = validate_run_display_name(
        new_name,
        field_name="new_name",
        required=True,
    )
    if not ok_name:
        return {"error": msg_name}
    new_name = normalized_name

    ok, msg = svc.rename_run(run_id, new_name)
    if not ok:
        return {"error": msg}

    return {"success": True, "message": msg}


def _handle_cancel_queued_job(
    data: dict[str, Any],
    *,
    svc: Any,
    jm: Any,
    username: str,
    role: str,
    is_admin: bool,
    validate_run_id: Any,
    is_user_run_visible: Any,
    resolve_job_id_for_cancel: Any,
) -> dict[str, Any]:
    """Handle queued-job cancellation through Job Manager."""
    raw_id = str(data.get("job_id", "") or "")
    # A run id identifies an owned run; block cancelling one the caller cannot see.
    if validate_run_id(raw_id) and not is_user_run_visible(username, raw_id, role=role):
        return {"error": "Forbidden"}
    job_id, err = resolve_job_id_for_cancel(
        raw_id,
        username=username,
        is_admin=is_admin,
    )
    if err:
        return {"error": err}

    if not jm.enabled:
        return {"error": "Job Manager integration is required"}

    ok_jm, jm_payload = jm.cancel_job(job_id)
    if ok_jm:
        # Keep the local run row in sync — without this the Faster DB stays
        # "queued" until a later refresh reconciles it (frontend shows a stale job).
        if validate_run_id(raw_id):
            svc.mark_run_status(raw_id, "stopped", "Cancelled by user")
        return {
            "success": True,
            "message": str(
                (jm_payload or {}).get("message") or "Cancelled via Job Manager"
            ),
            "job_manager": {"cancelled": True, "job_id": job_id},
        }

    jm_err = str(
        (jm_payload or {}).get("error")
        or (jm_payload or {}).get("detail")
        or "Failed to cancel job"
    )
    # Idempotency: the job is already gone from the JM (e.g. cancelled twice).
    # If the local run is already terminal, report success instead of an error.
    if "not found" in jm_err.lower() and validate_run_id(raw_id):
        if str(svc.get_run_status(raw_id) or "").lower() in {"stopped", "cancelled", "canceled", "failed", "completed", "done"}:
            return {"success": True, "message": "Already cancelled"}
    return {"error": jm_err}


def _build_job_manager_diagnostics(jm: Any, *, username: str) -> dict[str, Any]:
    """Build the stable Job Manager diagnostics payload."""
    jm_diag = {
        "enabled": bool(jm.enabled),
        "source_app": str(jm.source_app or ""),
        "ping_ok": False,
        "metrics_ok": False,
        "jobs_ok": False,
        "ping": {},
        "metrics": {},
        "jobs_for_user": [],
        "jobs_by_status": {},
    }

    if jm.enabled:
        ping_ok, ping_payload = jm.ping()
        jm_diag["ping_ok"] = bool(ping_ok)
        jm_diag["ping"] = ping_payload if isinstance(ping_payload, dict) else {}

        metrics_ok, metrics_payload = jm.get_metrics()
        jm_diag["metrics_ok"] = bool(metrics_ok)
        jm_diag["metrics"] = (
            metrics_payload if isinstance(metrics_payload, dict) else {}
        )

        jobs_ok, jobs_payload = jm.list_jobs_for_user(username, limit=50)
        jm_diag["jobs_ok"] = bool(jobs_ok)
        jobs = jobs_payload if isinstance(jobs_payload, list) else []
        jm_diag["jobs_for_user"] = jobs

        status_counts = {}
        for job in jobs:
            status = str(job.get("status") or "unknown")
            status_counts[status] = int(status_counts.get(status, 0)) + 1
        jm_diag["jobs_by_status"] = status_counts

    return jm_diag


def _handle_job_manager_diagnostics(
    *,
    jm: Any,
    username: str,
) -> dict[str, Any]:
    """Handle the Job Manager diagnostics command."""
    return {
        "success": True,
        "execution_backend": "job_manager_canonical",
        "local_runtime": {
            "enabled": False,
            "running": 0,
            "pending": 0,
        },
        "job_manager": _build_job_manager_diagnostics(jm, username=username),
        "note": "Job Manager is the only active queue and execution scheduler for run submissions.",
    }


def handle_run_command(command: str, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch one run/history command or return ``None`` when unmatched."""
    svc = ctx["svc"]
    jm = ctx["jm"]
    username = ctx["username"]
    role = ctx["role"]
    is_admin = ctx["is_admin"]
    validate_run_id = ctx["validate_run_id"]
    is_user_run_visible = ctx["is_user_run_visible"]
    sanitize_config_obj = ctx["sanitize_config_obj"]
    submit_run_to_job_manager = ctx["submit_run_to_job_manager"]
    row_summary_from_meta = ctx["row_summary_from_meta"]
    reconcile_running_runs = ctx["reconcile_running_runs"]
    validate_run_display_name = ctx["validate_run_display_name"]
    run_path = ctx["run_path"]
    clean_nan_inf = ctx["clean_nan_inf"]
    resolve_job_id_for_cancel = ctx["resolve_job_id_for_cancel"]

    if command == "get_runs":
        return _handle_get_runs(
            data,
            svc=svc,
            username=username,
            is_admin=is_admin,
            row_summary_from_meta=row_summary_from_meta,
            reconcile_running_runs=reconcile_running_runs,
        )

    if command == "get_runs_by_ids":
        return _handle_get_runs_by_ids(
            data,
            svc=svc,
            username=username,
            role=role,
            validate_run_id=validate_run_id,
            is_user_run_visible=is_user_run_visible,
            row_summary_from_meta=row_summary_from_meta,
        )

    if command == "validate_run_names":
        return _handle_validate_run_names(
            data,
            svc=svc,
            username=username,
            validate_run_display_name=validate_run_display_name,
        )

    if command == "get_run_status":
        return _handle_get_run_status(
            data,
            svc=svc,
            username=username,
            role=role,
            validate_run_id=validate_run_id,
            is_user_run_visible=is_user_run_visible,
        )

    if command in {"start_run", "queue_run"}:
        return _handle_start_or_queue_run(
            command,
            data,
            username=username,
            is_admin=is_admin,
            sanitize_config_obj=sanitize_config_obj,
            submit_run_to_job_manager=submit_run_to_job_manager,
        )

    if command == "stop_run":
        return _handle_stop_run(
            data,
            svc=svc,
            jm=jm,
            username=username,
            role=role,
            is_admin=is_admin,
            validate_run_id=validate_run_id,
            is_user_run_visible=is_user_run_visible,
        )

    if command == "get_metrics":
        return _handle_get_metrics(
            data,
            svc=svc,
            username=username,
            role=role,
            validate_run_id=validate_run_id,
            is_user_run_visible=is_user_run_visible,
            run_path=run_path,
            clean_nan_inf=clean_nan_inf,
        )

    if command == "get_logs":
        return _handle_get_logs(
            data,
            svc=svc,
            username=username,
            role=role,
            validate_run_id=validate_run_id,
            is_user_run_visible=is_user_run_visible,
            run_path=run_path,
        )

    if command == "get_run_config":
        return _handle_get_run_config(
            data,
            svc=svc,
            username=username,
            role=role,
            validate_run_id=validate_run_id,
            is_user_run_visible=is_user_run_visible,
            run_path=run_path,
        )

    if command == "delete_run":
        return _handle_delete_run(
            data,
            svc=svc,
            username=username,
            role=role,
            validate_run_id=validate_run_id,
            is_user_run_visible=is_user_run_visible,
        )

    if command == "rename_run":
        return _handle_rename_run(
            data,
            svc=svc,
            username=username,
            role=role,
            validate_run_id=validate_run_id,
            is_user_run_visible=is_user_run_visible,
            validate_run_display_name=validate_run_display_name,
        )

    if command == "cancel_queued_job":
        return _handle_cancel_queued_job(
            data,
            svc=svc,
            jm=jm,
            username=username,
            role=role,
            is_admin=is_admin,
            validate_run_id=validate_run_id,
            is_user_run_visible=is_user_run_visible,
            resolve_job_id_for_cancel=resolve_job_id_for_cancel,
        )

    if command == "job_manager_diagnostics":
        return _handle_job_manager_diagnostics(jm=jm, username=username)

    return None
