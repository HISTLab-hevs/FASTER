"""Shared run-oriented workflow helpers used by the API dispatcher."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Callable

from web_backend.contracts import CUSTOM_DATASET_NAMES, is_custom_dataset_name, is_supported_dataset_name
from utils.config import normalize_run_config_aliases
from utils.round_schedule import (
    normalize_round_client_allocation_config,
    normalize_round_train_schedule_config,
)
from utils.run_config_validation import validate_core_run_config


RUN_CONFIG_REQUIRED_KEYS = [
    "method",
    "dataset_name",
    "num_clients",
    "batch_size",
    "local_model_epochs",
    "weights_sending_frequency",
    "learning_rate",
    "server_learning_rate",
]


def validate_run_config(
    cfg: dict[str, Any],
    *,
    validate_run_display_name: Callable[..., tuple[bool, str, str]],
    is_valid_dataset_ref: Callable[[str], bool],
) -> tuple[bool, str]:
    """Validate a run configuration payload."""
    if not isinstance(cfg, dict):
        return False, "Invalid config payload"

    normalize_run_config_aliases(cfg)

    missing = [key for key in RUN_CONFIG_REQUIRED_KEYS if cfg.get(key) in (None, "")]
    if missing:
        return False, f"Missing required config keys: {', '.join(missing)}"

    dataset_name = str(cfg.get("dataset_name") or "").strip().lower()

    if not is_supported_dataset_name(dataset_name):
        return False, f"Unsupported dataset_name: {dataset_name}"

    if is_custom_dataset_name(dataset_name):
        ds_ref = str(cfg.get("custom_dataset_ref") or "")
        if not is_valid_dataset_ref(ds_ref):
            return False, "custom_dataset_ref is required and invalid for custom dataset"

        ds_name = str(cfg.get("custom_dataset_name") or "").strip()
        if ds_name and (len(ds_name) > 80 or not re.fullmatch(r"[A-Za-z0-9_.\- ]+", ds_name)):
            return False, "custom_dataset_name contains invalid characters"

    ok_name, msg_name, normalized_run_name = validate_run_display_name(
        cfg.get("run_name"),
        field_name="run_name",
        required=False,
    )
    if not ok_name:
        return False, msg_name
    if normalized_run_name:
        cfg["run_name"] = normalized_run_name

    # Web-independent value validation (method, numeric ranges, split mode, FedGP, mu,
    # custom model) lives in utils so the worker (main.py) enforces the same rules.
    ok_core, core_message = validate_core_run_config(cfg)
    if not ok_core:
        return False, core_message

    try:
        normalize_round_train_schedule_config(cfg)
        normalize_round_client_allocation_config(cfg)
    except ValueError as exc:
        return False, str(exc)

    return True, ""


def attach_custom_dataset_path(
    cfg: dict[str, Any],
    username: str,
    *,
    svc: Any,
    is_valid_dataset_ref: Callable[[str], bool],
    is_admin: bool = False,
) -> tuple[bool, str]:
    """Resolve a custom dataset reference to an absolute path."""
    dataset_name = str(cfg.get("dataset_name") or "").strip().lower()
    if dataset_name not in CUSTOM_DATASET_NAMES:
        cfg.pop("custom_dataset_ref", None)
        cfg.pop("custom_dataset_path", None)
        cfg.pop("custom_dataset_name", None)
        return True, ""

    dataset_ref = str(cfg.get("custom_dataset_ref") or "")
    if not is_valid_dataset_ref(dataset_ref):
        return False, "Invalid custom_dataset_ref"

    resolved_path = svc.resolve_custom_dataset_path(username, dataset_ref, is_admin=is_admin)
    if not resolved_path:
        return False, "Custom dataset not found. Please upload it again."

    ok_structure, structure_message = svc.validate_custom_dataset_structure(resolved_path, dataset_name)
    if not ok_structure:
        return False, structure_message

    cfg["custom_dataset_path"] = resolved_path
    cfg.setdefault("custom_dataset_name", os.path.splitext(os.path.basename(dataset_ref))[0])
    return True, ""


def heal_best_accuracy(run_id: str, *, svc: Any, run_path: Callable[[str], str], logger: Any) -> None:
    """Attempt to backfill ``best_accuracy`` from persisted metrics."""
    try:
        metrics = svc.load_metrics(run_path(run_id))
        if not metrics:
            return
        best = None
        for key in ["aggregated_test", "aggregated_val"]:
            accuracy_list = metrics.get(key, {}).get("accuracy")
            if isinstance(accuracy_list, list) and accuracy_list:
                finite_values = []
                for value in accuracy_list:
                    try:
                        if value is not None:
                            numeric = float(value)
                            if math.isfinite(numeric):
                                finite_values.append(numeric)
                    except (TypeError, ValueError):
                        continue
                if finite_values:
                    best = max(finite_values)
                    break
        if best is not None:
            svc.mark_run_best_accuracy(run_id, best)
    except Exception as exc:
        logger.warning("Failed to backfill best_accuracy for run '%s': %s", run_id, exc)


def reconcile_running_runs(
    username: str,
    *,
    is_admin: bool,
    include_all: bool,
    svc: Any,
    jm: Any,
    logger: Any,
    heal_best_accuracy_fn: Callable[[str], None],
) -> dict[str, int]:
    """Reconcile stale run states against Job Manager-backed execution."""
    checked = 0
    fixed = 0
    skipped = 0
    confirmed_terminal = 0
    no_evidence = 0
    incomplete = 0
    stale_or_partial = 0

    try:
        candidate_runs = svc.list_runs(username, include_all=include_all)
    except Exception as exc:
        logger.warning(
            "Failed to gather candidate runs for reconciliation for user '%s' (include_all=%s): %s",
            username,
            include_all,
            exc,
        )
        return {
            "checked": 0,
            "fixed": 0,
            "skipped": 0,
            "confirmed_terminal": 0,
            "no_evidence": 0,
            "incomplete": 0,
            "stale_or_partial": 0,
        }

    reconcilable = []
    for run_id in candidate_runs:
        state = str(svc.get_run_status(run_id) or "").lower()
        if state in {"running", "queued"}:
            reconcilable.append((run_id, state))
        else:
            heal_best_accuracy_fn(run_id)

    if not reconcilable:
        return {
            "checked": 0,
            "fixed": 0,
            "skipped": 0,
            "confirmed_terminal": 0,
            "no_evidence": 0,
            "incomplete": 0,
            "stale_or_partial": 0,
        }

    def _result() -> dict[str, int]:
        """Return the current reconciliation counters as a payload."""
        return {
            "checked": checked,
            "fixed": fixed,
            "skipped": skipped,
            "confirmed_terminal": confirmed_terminal,
            "no_evidence": no_evidence,
            "incomplete": incomplete,
            "stale_or_partial": stale_or_partial,
        }

    if not jm.enabled:
        incomplete = len(reconcilable)
        skipped = len(reconcilable)
        checked = len(reconcilable)
        return _result()

    snapshot_limit = _job_snapshot_limit(jm, is_admin=is_admin)
    ok_jobs, jobs = _list_visible_jobs(
        jm,
        username,
        is_admin=is_admin,
        limit=snapshot_limit,
    )
    if not ok_jobs or not isinstance(jobs, list):
        incomplete = len(reconcilable)
        skipped = len(reconcilable)
        checked = len(reconcilable)
        return _result()

    active_priority = {"running": 2, "queued": 1, "pending": 1}
    terminal_priority = {"failed": 4, "error": 4, "cancelled": 3, "canceled": 3, "stopped": 3, "done": 2, "completed": 2, "success": 2}
    known_statuses = set(active_priority) | set(terminal_priority)

    jm_by_run: dict[str, str] = {}
    malformed_rows = 0
    unknown_status_rows = 0
    conflicting_rows = 0
    truncated_snapshot = len(jobs) >= snapshot_limit
    for job in jobs:
        if not isinstance(job, dict):
            malformed_rows += 1
            continue
        run_id = str(job.get("run_id") or "").strip()
        status = str(job.get("status") or "").lower().strip()
        if not run_id or not status:
            malformed_rows += 1
            continue
        if status not in known_statuses:
            unknown_status_rows += 1
            continue

        previous = jm_by_run.get(run_id)
        if previous is None:
            jm_by_run[run_id] = status
            continue
        if previous == status:
            continue

        previous_active = active_priority.get(previous, 0)
        current_active = active_priority.get(status, 0)
        if previous_active or current_active:
            chosen = status if current_active > previous_active else previous
            if chosen != previous:
                jm_by_run[run_id] = chosen
            conflicting_rows += 1
            continue

        previous_terminal = terminal_priority.get(previous, 0)
        current_terminal = terminal_priority.get(status, 0)
        if current_terminal > previous_terminal:
            jm_by_run[run_id] = status
        conflicting_rows += 1

    snapshot_is_partial = bool(malformed_rows or unknown_status_rows or conflicting_rows or truncated_snapshot)
    if snapshot_is_partial:
        stale_or_partial += malformed_rows + unknown_status_rows + conflicting_rows
        if truncated_snapshot:
            stale_or_partial += 1

    for run_id, current_state in reconcilable:
        checked += 1
        jm_state = jm_by_run.get(run_id)
        if jm_state in {"running", "queued", "pending"}:
            desired_state = "running" if jm_state == "running" else "queued"
            if desired_state != current_state:
                svc.mark_run_status(run_id, desired_state, "Recovered by refresh: synchronized with Job Manager")
                fixed += 1
            else:
                skipped += 1
            continue
        if jm_state in {"done", "completed", "success"}:
            svc.mark_run_status(run_id, "completed", "")
            fixed += 1
            confirmed_terminal += 1
            continue
        if jm_state in {"cancelled", "canceled", "stopped"}:
            svc.mark_run_status(run_id, "stopped", "Recovered by refresh: cancelled in Job Manager")
            fixed += 1
            confirmed_terminal += 1
            continue
        if jm_state in {"error", "failed"}:
            svc.mark_run_status(run_id, "failed", "Recovered by refresh: failed in Job Manager")
            fixed += 1
            confirmed_terminal += 1
            continue

        skipped += 1
        if snapshot_is_partial:
            stale_or_partial += 1
        else:
            no_evidence += 1

    return _result()


def _job_snapshot_limit(jm: Any, *, is_admin: bool) -> int:
    """Return the effective Job Manager snapshot limit for the current role."""
    attr_name = "_JOB_LIST_LIMIT_MAX" if is_admin else "_USER_JOB_LIMIT_MAX"
    default = 500 if is_admin else 200
    try:
        value = int(getattr(jm, attr_name, default) or default)
    except Exception:
        value = default
    return max(1, value)


def _list_visible_jobs(
    jm: Any,
    username: str,
    *,
    is_admin: bool,
    limit: int | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Fetch the user-scoped Job Manager snapshot for reconciliation/cancel.

    Reconciliation and cancellation only ever act on the caller's own runs, so
    a normal user is correctly limited to their own jobs here. The queue-position
    snapshot deliberately uses the global list instead (see
    ``normalized_jobs_snapshot``) so a user can see how many jobs sit ahead.
    """
    safe_limit = limit if limit is not None else _job_snapshot_limit(jm, is_admin=is_admin)
    if is_admin:
        return jm.list_jobs(limit=safe_limit)
    return jm.list_jobs_for_user(username, limit=safe_limit)


def normalized_jobs_snapshot(
    username: str,
    *,
    svc: Any,
    jm: Any,
    run_path: Callable[[str], str],
    is_admin: bool = False,
) -> list[dict[str, Any]]:
    """Return normalized active jobs with role-aware visibility.

    The queue snapshot is always built from the *global* job list so a normal
    user's queue position reflects every job ahead of them. Other users' jobs
    are then masked below to status+position only (no job_id, run_id, or owner).
    """
    if not jm.enabled:
        return []
    ok_jobs, jm_jobs = jm.list_jobs(limit=_job_snapshot_limit(jm, is_admin=True))
    if not ok_jobs or not isinstance(jm_jobs, list):
        return []

    terminal_run_statuses = {
        "completed",
        "failed",
        "stopped",
        "cancelled",
        "canceled",
        "done",
        "success",
        "error",
    }

    active_rows = []
    for job in jm_jobs:
        if not isinstance(job, dict):
            continue
        if str(job.get("status") or "").lower() in {"queued", "pending", "running"}:
            run_id = str(job.get("run_id") or "").strip()
            if run_id and str(svc.get_run_status(run_id) or "").lower() in terminal_run_statuses:
                continue
            active_rows.append(job)

    def _queue_key(job: dict[str, Any]) -> tuple[int, str]:
        """Return the sort key that keeps running jobs ahead of queued jobs."""
        status = str(job.get("status") or "").lower()
        return (0 if status == "running" else 1, str(job.get("job_id") or ""))

    active_rows.sort(key=_queue_key)

    queue_positions: dict[str, int] = {}
    queue_counter = 0
    for job in active_rows:
        if str(job.get("status") or "").lower() in {"queued", "pending"}:
            queue_counter += 1
            queue_positions[str(job.get("job_id") or "")] = queue_counter

    my_queued_positions = [
        queue_positions.get(str(job.get("job_id") or ""))
        for job in active_rows
        if str(job.get("user_id") or "") == username and str(job.get("status") or "").lower() in {"queued", "pending"}
    ]
    my_queued_positions = [pos for pos in my_queued_positions if isinstance(pos, int)]
    my_first_queue_pos = min(my_queued_positions) if my_queued_positions else None

    output = []
    config_cache: dict[str, dict[str, Any]] = {}
    for job in active_rows:
        status = str(job.get("status") or "").lower()
        owner = str(job.get("user_id") or "unknown")
        run_id = str(job.get("run_id") or "")
        job_id = str(job.get("job_id") or "")
        is_mine = is_admin or owner == username
        position = queue_positions.get(job_id)

        is_ahead_for_user = False
        if not is_admin and not is_mine:
            if status == "running":
                is_ahead_for_user = bool(my_queued_positions)
            elif status in {"queued", "pending"} and my_first_queue_pos is not None and isinstance(position, int):
                is_ahead_for_user = position < my_first_queue_pos
        if not is_admin and not is_mine and not is_ahead_for_user:
            continue

        # Other users' jobs shown to a normal user expose no identity: only the
        # status and queue position are kept, so the user can gauge how many
        # jobs sit ahead of theirs without learning who or what they are.
        masked = not is_admin and not is_mine
        item = {
            "job_id": "" if masked else job_id,
            "run_id": "" if masked else run_id,
            "owner": "" if masked else owner,
            "status": status,
            "kind": "running" if status == "running" else ("pending" if status == "queued" else status),
            "position": position,
            "is_mine": is_mine,
            "locked": not is_mine,
        }
        summary = {
            "progress_pct": job.get("progress_pct") if isinstance(job.get("progress_pct"), (int, float)) else None,
        }
        if run_id and is_mine:
            cfg = config_cache.get(run_id)
            if cfg is None:
                cfg = svc.load_run_config(run_path(run_id)) or {}
                config_cache[run_id] = cfg
            summary.update({"method": cfg.get("method"), "dataset_name": svc.dataset_display_name_from_config(cfg)})
            item["config"] = cfg
        item["summary"] = summary
        output.append(item)
    return output


def submit_run_to_job_manager(
    config: dict[str, Any],
    username: str,
    *,
    svc: Any,
    jm: Any,
    validate_run_config_fn: Callable[[dict[str, Any]], tuple[bool, str]],
    attach_custom_dataset_path_fn: Callable[[dict[str, Any], str, bool], tuple[bool, str]],
    is_admin: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Validate, reserve, and submit a run payload to the Job Manager."""
    ok_cfg, msg_cfg = validate_run_config_fn(config)
    if not ok_cfg:
        return False, {"error": msg_cfg}

    ok_dataset, msg_dataset = attach_custom_dataset_path_fn(config, username, is_admin)
    if not ok_dataset:
        return False, {"error": msg_dataset}

    if not jm.enabled:
        return False, {"error": "Job Manager integration is required but not reachable"}

    ok_reserve, msg_reserve, run_id, _, reserved_cfg = svc.reserve_run_submission(config, username)
    if not ok_reserve:
        return False, {"error": msg_reserve}

    ok_submit, jm_payload = jm.submit_config_job(
        user_id=username,
        config=reserved_cfg,
        description=str(reserved_cfg.get("display_run_name") or reserved_cfg.get("run_name") or "FASTER run"),
        job_type="faster_run",
        run_id=run_id,
    )
    if not ok_submit:
        svc.mark_run_status(run_id, "failed", "Job Manager submission failed")
        return False, {"error": (jm_payload or {}).get("error") or "Job Manager submission failed"}

    return True, {
        "run_id": run_id,
        "job_id": jm_payload.get("job_id") if isinstance(jm_payload, dict) else None,
        "reserved_cfg": reserved_cfg,
        "jm_payload": jm_payload if isinstance(jm_payload, dict) else {},
    }


def resolve_job_id_for_cancel(
    identifier: str,
    username: str,
    *,
    is_admin: bool,
    jm: Any,
    is_valid_job_id: Callable[[str], bool],
    is_valid_run_id: Callable[[str], bool],
) -> tuple[str, str | None]:
    """Resolve a cancel identifier to a concrete Job Manager job ID."""
    ident = str(identifier or "").strip()
    if not ident:
        return "", "Invalid job id"
    if not is_valid_job_id(ident) and not is_valid_run_id(ident):
        return "", "Invalid job id"
    if not jm.enabled:
        if is_valid_job_id(ident):
            return ident, None
        return "", "Invalid job id"

    ok_jobs, jobs = jm.list_jobs(limit=300) if is_admin else jm.list_jobs_for_user(username, limit=300)
    if not ok_jobs or not isinstance(jobs, list):
        return "", "Could not resolve queued job id"

    by_job_id = next((job for job in jobs if str(job.get("job_id") or "") == ident), None)
    if by_job_id:
        return ident, None

    match = next(
        (
            job
            for job in jobs
            if str(job.get("run_id") or "") == ident and str(job.get("status") or "").lower() == "queued"
        ),
        None,
    )
    if not match:
        match = next(
            (
                job
                for job in jobs
                if str(job.get("run_id") or "") == ident and str(job.get("status") or "").lower() in {"queued", "running"}
            ),
            None,
        )
    job_id = str((match or {}).get("job_id") or "")
    if not job_id:
        if is_valid_job_id(ident):
            return ident, None
        return "", f"No queued job found for run id '{ident}'"
    return job_id, None


def is_active_job_entry(item: dict[str, Any]) -> bool:
    """Return whether a normalized job row is actively running or queued."""
    status = str(item.get("status") or item.get("kind") or "").strip().lower()
    return status in {"running", "queued", "pending"}
