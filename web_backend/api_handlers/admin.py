"""Domain handlers for admin, analytics, alerts, and runtime status."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import psutil


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


def _handle_admin_list_users(*, auth: Any, is_admin: bool) -> dict[str, Any]:
    """Handle the admin user-list command."""
    if not is_admin:
        return {"error": "Forbidden"}

    users = auth.list_users()
    users = sorted(
        users,
        key=lambda item: (item.get("role") != "admin", str(item.get("identifier") or "")),
    )
    return {"success": True, "users": users}


def _handle_admin_create_user(
    data: dict[str, Any],
    *,
    auth: Any,
    username: str,
    is_admin: bool,
) -> dict[str, Any]:
    """Handle admin-driven user creation."""
    if not is_admin:
        return {"error": "Forbidden"}

    identifier = str(data.get("identifier", "") or "").strip()
    email = str(data.get("email", "") or "").strip().lower()
    password = str(data.get("password", "") or "")
    role_new = str(data.get("role", "user") or "user").lower()

    if not identifier:
        return {"error": "Identifier is required"}
    if not email:
        return {"error": "Email is required"}
    if role_new not in {"user", "admin"}:
        return {"error": "Invalid role"}
    if not password:
        return {"error": "Password is required"}

    ok, msg_create, created = auth.create_user(
        identifier,
        email,
        password,
        role_new,
        created_by=username,
    )
    if not ok:
        return {"error": msg_create}

    return {
        "success": True,
        "user": created,
        "message": msg_create or "User created",
    }


def _handle_admin_delete_user(
    data: dict[str, Any],
    *,
    auth: Any,
    username: str,
    is_admin: bool,
) -> dict[str, Any]:
    """Handle admin-driven user deletion."""
    if not is_admin:
        return {"error": "Forbidden"}

    identifier = str(data.get("identifier", "") or "").strip()
    if not identifier:
        return {"error": "User identifier is required"}

    ok, msg = auth.admin_delete_user(username, identifier)
    if not ok:
        return {"error": msg}

    return {"success": True, "message": msg}


def _handle_admin_set_user_role(
    data: dict[str, Any],
    *,
    auth: Any,
    username: str,
    is_admin: bool,
) -> dict[str, Any]:
    """Handle admin-driven user role updates."""
    if not is_admin:
        return {"error": "Forbidden"}

    identifier = str(data.get("identifier", "") or "").strip()
    role_target = str(data.get("role", "") or "").strip().lower()

    if not identifier:
        return {"error": "User identifier is required"}
    if role_target not in {"user", "admin"}:
        return {"error": "Invalid role"}

    ok, msg, user_out = auth.admin_set_user_role(
        username,
        identifier,
        role_target,
    )
    if not ok:
        return {"error": msg}

    return {"success": True, "message": msg, "user": user_out}


def _admin_run_search_text(row: dict[str, Any], search_field: str) -> str:
    """Return the searchable text for one admin run row."""
    if search_field == "owner":
        return str(row.get("owner") or "")
    if search_field == "method":
        return str(row.get("method") or "")
    if search_field == "dataset":
        return str(row.get("dataset") or "")
    if search_field == "status":
        return str(row.get("status") or "")
    if search_field == "id":
        return str(row.get("id") or "")
    return " ".join(
        [
            str(row.get("id") or ""),
            str(row.get("name") or ""),
            str(row.get("owner") or ""),
            str(row.get("method") or ""),
            str(row.get("dataset") or ""),
            str(row.get("model_name") or ""),
            str(row.get("status") or ""),
        ]
    )


def _admin_job_search_text(item: dict[str, Any], search_field: str) -> str:
    """Return the searchable text for one admin job row."""
    if search_field == "owner":
        return str(item.get("owner") or "")
    if search_field == "method":
        return str(item.get("summary", {}).get("method") or "")
    if search_field == "id":
        return str(item.get("run_id") or item.get("job_id") or "")
    if search_field == "kind":
        return str(item.get("kind") or "")
    return " ".join(
        [
            str(item.get("run_id") or ""),
            str(item.get("job_id") or ""),
            str(item.get("owner") or ""),
            str(item.get("kind") or ""),
            str(item.get("status") or ""),
            str((item.get("summary") or {}).get("method") or ""),
            str((item.get("summary") or {}).get("dataset_name") or ""),
        ]
    )


def _paginate_admin_jobs(
    jobs: list[dict[str, Any]],
    *,
    page: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Paginate admin job rows with the current clamping semantics."""
    jobs_total = len(jobs)
    jobs_pages = max(1, (jobs_total + page_size - 1) // page_size)
    safe_page = min(page, jobs_pages)
    jobs_start = (safe_page - 1) * page_size
    jobs_end = jobs_start + page_size
    return jobs[jobs_start:jobs_end], {
        "page": safe_page,
        "page_size": page_size,
        "pages": jobs_pages,
        "total": jobs_total,
    }


def _handle_admin_platform_overview(
    data: dict[str, Any],
    *,
    auth: Any,
    svc: Any,
    username: str,
    is_admin: bool,
    search_tokens: Any,
    matches_tokens: Any,
    row_summary_from_meta: Any,
    build_admin_analytics: Any,
    normalized_jobs_snapshot: Any,
    admin_runs_limit: int,
) -> dict[str, Any]:
    """Handle the admin platform overview command."""
    if not is_admin:
        return {"error": "Forbidden"}

    search_query = str(data.get("search", "") or "").strip().lower()
    search_field = str(data.get("search_field", "all") or "all").strip().lower()
    jobs_page_size = _coerce_int(
        data.get("jobs_page_size", 10),
        default=10,
        minimum=1,
        maximum=100,
    )
    jobs_page = _coerce_int(data.get("jobs_page", 1), default=1, minimum=1)
    search_token_list = search_tokens(search_query)

    run_ids_all = svc.list_runs(username, include_all=True)
    run_meta_sample = svc.list_run_metadata(username, include_all=True, limit=admin_runs_limit)
    run_rows_scoped = []
    for meta in run_meta_sample or []:
        row = row_summary_from_meta(meta)
        if str(row.get("id") or "").strip():
            run_rows_scoped.append(row)
    filtered_run_rows = [
        row
        for row in run_rows_scoped
        if matches_tokens(_admin_run_search_text(row, search_field), search_token_list)
    ]

    users = auth.list_users()
    jobs = normalized_jobs_snapshot(username, is_admin=True)
    filtered_jobs = [
        job
        for job in jobs
        if matches_tokens(_admin_job_search_text(job, search_field), search_token_list)
    ]
    analytics = build_admin_analytics(filtered_run_rows, filtered_jobs)
    jobs_page_items, jobs_pagination = _paginate_admin_jobs(
        filtered_jobs,
        page=jobs_page,
        page_size=jobs_page_size,
    )

    return {
        "success": True,
        "totals": {
            "users": len(users),
            "runs": len(run_ids_all),
            "running_jobs": len(
                [job for job in filtered_jobs if str(job.get("kind") or "") == "running"]
            ),
            "pending_jobs": len(
                [job for job in filtered_jobs if str(job.get("kind") or "") == "pending"]
            ),
        },
        "jobs": jobs_page_items,
        "jobs_pagination": jobs_pagination,
        "analytics": analytics,
    }


def _handle_get_alerts(*, svc: Any, username: str) -> dict[str, Any]:
    """Handle user alert retrieval."""
    return {"success": True, "alerts": svc.consume_user_alerts(username)}


def _partition_active_jobs(
    jobs_all: list[dict[str, Any]],
    *,
    username: str,
    is_active_job_entry: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition active jobs into running and queued groupings."""
    jobs = [job for job in jobs_all if is_active_job_entry(job)]
    running_jobs = [
        job
        for job in jobs
        if str(job.get("status") or job.get("kind") or "").lower() == "running"
    ]
    queued_jobs = [
        job
        for job in jobs
        if str(job.get("status") or job.get("kind") or "").lower() in {"queued", "pending"}
    ]
    mine_running = [job for job in running_jobs if str(job.get("owner") or "") == username]
    mine_queued = [job for job in queued_jobs if str(job.get("owner") or "") == username]
    return jobs, running_jobs, queued_jobs, mine_running, mine_queued


def _build_active_user_job(
    mine_running: list[dict[str, Any]],
    *,
    svc: Any,
    run_path: Any,
) -> dict[str, Any] | None:
    """Build the active user-job payload when a running job exists."""
    if not mine_running:
        return None

    active_job = mine_running[0]
    run_id = str(active_job.get("run_id") or "")
    if not run_id:
        return None

    cfg = active_job.get("config") if isinstance(active_job.get("config"), dict) else None
    if cfg is None:
        cfg = svc.load_run_config(run_path(run_id)) or {}
    return {
        "run_id": run_id,
        "method": cfg.get("method"),
        "dataset_name": svc.dataset_display_name_from_config(cfg),
        "num_clients": cfg.get("num_clients"),
        "local_model_epochs": cfg.get("local_model_epochs"),
        "weights_sending_frequency": cfg.get("weights_sending_frequency"),
        "config": cfg,
    }


def _build_queued_user_job(mine_queued: list[dict[str, Any]]) -> tuple[int | None, int]:
    """Build queue-position details for the current user."""
    queue_position = None
    if mine_queued:
        queue_positions = [
            job.get("position")
            for job in mine_queued
            if isinstance(job.get("position"), int)
        ]
        queue_position = min(queue_positions) if queue_positions else None

    ahead = max(queue_position - 1, 0) if queue_position is not None else 0
    return queue_position, ahead


def _build_user_job_payload(
    *,
    active: dict[str, Any] | None,
    mine_queued: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the user-job status payload for runtime status responses."""
    queue_position, ahead = _build_queued_user_job(mine_queued)
    return {
        "status": "active" if active else ("queued" if mine_queued else "idle"),
        "active": active,
        "queue": {
            "enabled": True,
            "position": queue_position,
            "ahead": ahead,
        },
    }


def _build_job_manager_status(*, jm: Any, is_admin: bool) -> dict[str, Any]:
    """Build the Job Manager runtime status payload."""
    jm_status = {"enabled": bool(jm.enabled), "reachable": False}
    if jm.enabled:
        ok_ping, ping_payload = jm.ping()
        jm_status["reachable"] = bool(ok_ping)
        if ok_ping:
            jm_status["ping"] = ping_payload
        if is_admin:
            ok_metrics, metrics_payload = jm.get_metrics()
            jm_status["metrics"] = (
                metrics_payload if ok_metrics else {"error": "metrics unavailable"}
            )
    return jm_status


def _load_gpu_info(
    *,
    gputil_enabled: bool,
    gputil_module: Any,
    logger: Any,
) -> dict[str, Any]:
    """Load GPU information for runtime status responses."""
    gpu_info = {
        "available": False,
        "message": "No GPU detected",
        "gpus": [],
    }
    if gputil_enabled and gputil_module is not None:
        try:
            gpus = gputil_module.getGPUs()
            if gpus:
                gpu_info = {
                    "available": True,
                    "gpus": [
                        {
                            "id": gpu.id,
                            "name": gpu.name,
                            "temperature": gpu.temperature,
                            "load": round(gpu.load * 100, 1),
                            "memory_used_mb": round(gpu.memoryUsed),
                            "memory_total_mb": round(gpu.memoryTotal),
                        }
                        for gpu in gpus
                    ],
                }
        except Exception as exc:
            logger.warning("Failed to query GPU info for runtime status: %s", exc)
    return gpu_info


def _append_opportunistic_resource_sample(
    *,
    svc: Any,
    logger: Any,
    active_run_id: str,
    cpu: float,
    ram: Any,
    gpu_info: dict[str, Any],
) -> None:
    """Append one opportunistic resource sample when an active run lacks samples."""
    if not active_run_id:
        return

    try:
        usage = svc.load_run_resource_usage(active_run_id) or {}
        summary = usage.get("summary") if isinstance(usage, dict) else {}
        has_samples = int((summary or {}).get("samples_count") or 0) > 0

        if not has_samples:
            sample = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "cpu_percent": round(cpu, 1),
                "ram_used_gb": round(ram.used / (1024**3), 2),
                "ram_total_gb": round(ram.total / (1024**3), 2),
                "ram_percent": round(ram.percent, 1),
                "gpus": [
                    {
                        "id": gpu.get("id"),
                        "name": gpu.get("name"),
                        "temperature": gpu.get("temperature"),
                        "load": gpu.get("load"),
                        "memory_used_mb": gpu.get("memory_used_mb"),
                        "memory_total_mb": gpu.get("memory_total_mb"),
                    }
                    for gpu in (gpu_info.get("gpus") or [])
                    if isinstance(gpu, dict)
                ],
            }
            svc.append_run_resource_sample(active_run_id, sample)
    except Exception as exc:
        logger.warning(
            "Failed to append opportunistic resource sample for active run '%s': %s",
            active_run_id,
            exc,
        )


def _handle_system_status(
    *,
    svc: Any,
    jm: Any,
    username: str,
    is_admin: bool,
    normalized_jobs_snapshot: Any,
    is_active_job_entry: Any,
    run_path: Any,
    logger: Any,
    gputil_enabled: bool,
    gputil_module: Any,
) -> dict[str, Any]:
    """Handle the system runtime-status command."""
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()

    jobs_all = normalized_jobs_snapshot(username, is_admin=is_admin)
    jobs, _running_jobs, _queued_jobs, mine_running, mine_queued = _partition_active_jobs(
        jobs_all,
        username=username,
        is_active_job_entry=is_active_job_entry,
    )
    active = _build_active_user_job(mine_running, svc=svc, run_path=run_path)
    user_job = _build_user_job_payload(active=active, mine_queued=mine_queued)
    jm_status = _build_job_manager_status(jm=jm, is_admin=is_admin)
    gpu_info = _load_gpu_info(
        gputil_enabled=gputil_enabled,
        gputil_module=gputil_module,
        logger=logger,
    )

    active_run_id = ((user_job or {}).get("active") or {}).get("run_id")
    _append_opportunistic_resource_sample(
        svc=svc,
        logger=logger,
        active_run_id=active_run_id,
        cpu=cpu,
        ram=ram,
        gpu_info=gpu_info,
    )

    return {
        "success": True,
        "cpu_percent": round(cpu, 1),
        "ram_used_gb": round(ram.used / (1024**3), 2),
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "ram_percent": round(ram.percent, 1),
        "gpu_info": gpu_info,
        "user_job": user_job,
        "jobs": jobs,
        "job_manager": jm_status,
        "alerts": svc.consume_user_alerts(username),
    }


def handle_admin_command(command: str, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Handle admin and platform/runtime-oriented API commands."""
    auth = ctx["auth"]
    svc = ctx["svc"]
    jm = ctx["jm"]
    username = ctx["username"]
    is_admin = ctx["is_admin"]
    search_tokens = ctx["search_tokens"]
    matches_tokens = ctx["matches_tokens"]
    row_summary_from_meta = ctx["row_summary_from_meta"]
    build_admin_analytics = ctx["build_admin_analytics"]
    normalized_jobs_snapshot = ctx["normalized_jobs_snapshot"]
    is_active_job_entry = ctx["is_active_job_entry"]
    run_path = ctx["run_path"]
    logger = ctx["logger"]
    gputil_enabled = ctx["gputil_enabled"]
    gputil_module = ctx["gputil_module"]
    admin_runs_limit = ctx["admin_analytics_max_runs"]

    if command == "admin_list_users":
        return _handle_admin_list_users(auth=auth, is_admin=is_admin)

    if command == "admin_create_user":
        return _handle_admin_create_user(
            data,
            auth=auth,
            username=username,
            is_admin=is_admin,
        )

    if command == "admin_delete_user":
        return _handle_admin_delete_user(
            data,
            auth=auth,
            username=username,
            is_admin=is_admin,
        )

    if command == "admin_set_user_role":
        return _handle_admin_set_user_role(
            data,
            auth=auth,
            username=username,
            is_admin=is_admin,
        )

    if command == "admin_platform_overview":
        return _handle_admin_platform_overview(
            data,
            auth=auth,
            svc=svc,
            username=username,
            is_admin=is_admin,
            search_tokens=search_tokens,
            matches_tokens=matches_tokens,
            row_summary_from_meta=row_summary_from_meta,
            build_admin_analytics=build_admin_analytics,
            normalized_jobs_snapshot=normalized_jobs_snapshot,
            admin_runs_limit=admin_runs_limit,
        )

    if command == "get_alerts":
        return _handle_get_alerts(svc=svc, username=username)

    if command == "system_status":
        return _handle_system_status(
            svc=svc,
            jm=jm,
            username=username,
            is_admin=is_admin,
            normalized_jobs_snapshot=normalized_jobs_snapshot,
            is_active_job_entry=is_active_job_entry,
            run_path=run_path,
            logger=logger,
            gputil_enabled=gputil_enabled,
            gputil_module=gputil_module,
        )

    return None
