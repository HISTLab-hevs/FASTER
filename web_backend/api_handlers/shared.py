"""Shared handler utilities used by the API dispatcher and domain modules."""

from __future__ import annotations

import math
import os
import re
from typing import Any


def run_path(base_results_path: str, name: str) -> str:
    """Return the results path for one run directory name."""
    return os.path.join(base_results_path, name)


def is_valid_run_id(run_id: str) -> bool:
    """Return whether a run identifier matches the backend route format."""
    return bool(re.fullmatch(r"[A-Za-z0-9_.\- ]{1,160}", run_id or ""))


def is_valid_job_id(job_id: str) -> bool:
    """Return whether a Job Manager identifier matches the backend format."""
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{3,120}", job_id or ""))


def is_valid_dataset_ref(dataset_ref: str) -> bool:
    """Return whether a custom-dataset reference token is valid."""
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{6,180}", dataset_ref or ""))


def is_admin_role(role: str) -> bool:
    """Return whether one role string represents an administrator."""
    return str(role or "").lower() == "admin"


def is_user_run_visible(username: str, run_id: str, *, role: str, svc: Any) -> bool:
    """Return whether a user may see one run."""
    if is_admin_role(role):
        return True
    return svc.user_can_access_run(username, run_id, is_admin=False)


def _meta_value(meta: Any, key: str, default: Any = None) -> Any:
    """Return one metadata field from dict-like or attribute-like rows."""
    if isinstance(meta, dict):
        return meta.get(key, default)
    return getattr(meta, key, default)


def sanitize_config_obj(obj: Any) -> Any:
    """Recursively sanitize configuration payload values."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            key_text = str(key)
            if key_text == "custom_model_code" and isinstance(value, str):
                out[key_text] = value[:250_000]
            else:
                out[key_text] = sanitize_config_obj(value)
        return out
    if isinstance(obj, list):
        return [sanitize_config_obj(value) for value in obj]
    if isinstance(obj, str):
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", obj)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:250_000]
    return obj


def error_payload(message: str) -> dict[str, str]:
    """Return the standard user-facing API error payload shape."""
    text = str(message or "").strip() or "Request failed"
    return {"error": text}


def log_and_mask_exception(
    logger: Any,
    *,
    user_message: str,
    log_message: str,
) -> dict[str, str]:
    """Log one unexpected exception and return a stable user-facing error."""
    if logger is not None:
        logger.exception(log_message)
    return error_payload(user_message)


def clean_nan_inf(obj: Any) -> Any:
    """Recursively replace NaN and infinity values with ``None``."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {key: clean_nan_inf(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_inf(value) for value in obj]
    return obj


def search_tokens(raw: str) -> list[str]:
    """Split a free-text search string into lowercase tokens."""
    return [token for token in str(raw or "").strip().lower().split() if token]


def matches_tokens(text: str, tokens: list[str]) -> bool:
    """Return whether all search tokens are present in one text."""
    if not tokens:
        return True
    haystack = str(text or "").lower()
    return all(token in haystack for token in tokens)


def top_with_other(counter: dict[str, int], top_n: int) -> dict[str, list[Any]]:
    """Build a chart-friendly top-N series with an aggregated ``others`` bucket."""
    ranked = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    limit = max(1, int(top_n or 1))
    top_items = ranked[:limit]
    other_total = sum(value for _, value in ranked[limit:])

    labels = [label for label, _ in top_items]
    values = [int(value) for _, value in top_items]
    if other_total > 0:
        labels.append("others")
        values.append(int(other_total))
    return {"labels": labels, "values": values}


def duration_bucket(seconds_val: Any) -> str:
    """Bucket a duration in seconds for analytics output."""
    try:
        seconds = float(seconds_val)
    except Exception:
        return "unknown"
    if seconds < 300:
        return "<5m"
    if seconds < 900:
        return "5-15m"
    if seconds < 1800:
        return "15-30m"
    if seconds < 3600:
        return "30-60m"
    return ">=60m"


def build_admin_analytics(
    run_rows: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    *,
    svc: Any,
    resource_scan_limit: int,
    top_users: int,
    top_methods: int,
    top_datasets: int,
) -> dict[str, Any]:
    """Build aggregate analytics payloads for the admin overview."""
    by_status: dict[str, int] = {}
    by_owner: dict[str, int] = {}
    by_method: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    duration_buckets = {
        "<5m": 0,
        "5-15m": 0,
        "15-30m": 0,
        "30-60m": 0,
        ">=60m": 0,
        "unknown": 0,
    }

    successes = 0
    failures = 0
    stopped = 0
    running = 0
    cpu_values: list[float] = []
    ram_values: list[float] = []
    gpu_values: list[float] = []
    scanned_resources = 0

    for index, row in enumerate(run_rows):
        status = str(row.get("status") or "unknown").lower()
        by_status[status] = by_status.get(status, 0) + 1

        owner = str(row.get("owner") or "unknown")
        by_owner[owner] = by_owner.get(owner, 0) + 1

        method = str(row.get("method") or "unknown")
        by_method[method] = by_method.get(method, 0) + 1

        dataset = str(row.get("dataset") or "unknown")
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1

        if status == "completed":
            successes += 1
        elif status == "failed":
            failures += 1
        elif status == "stopped":
            stopped += 1
        elif status == "running":
            running += 1

        if index < resource_scan_limit:
            run_id = str(row.get("id") or "")
            if run_id:
                usage = svc.load_run_resource_usage(run_id) or {}
                summary = usage.get("summary") if isinstance(usage.get("summary"), dict) else {}

                bucket = duration_bucket(summary.get("duration_s"))
                duration_buckets[bucket] = duration_buckets.get(bucket, 0) + 1

                cpu = summary.get("cpu_percent_avg")
                ram = summary.get("ram_percent_avg")
                gpu = summary.get("gpu_load_avg")
                if isinstance(cpu, (int, float)):
                    cpu_values.append(float(cpu))
                if isinstance(ram, (int, float)):
                    ram_values.append(float(ram))
                if isinstance(gpu, (int, float)):
                    gpu_values.append(float(gpu))
                scanned_resources += 1

    running_jobs = [job for job in jobs if str(job.get("kind") or "") == "running"]
    pending_jobs = [job for job in jobs if str(job.get("kind") or "") == "pending"]

    pending_by_method: dict[str, int] = {}
    for job in pending_jobs:
        method = str((job.get("summary") or {}).get("method") or "unknown")
        pending_by_method[method] = pending_by_method.get(method, 0) + 1

    active_workers = len({str(job.get("owner") or "unknown") for job in running_jobs})
    queued_workers = len({str(job.get("owner") or "unknown") for job in pending_jobs})
    queue_total = len(running_jobs) + len(pending_jobs)
    queue_pressure = round((len(pending_jobs) / queue_total) * 100.0, 1) if queue_total > 0 else 0.0

    terminal_total = successes + failures
    success_ratio = round((successes / terminal_total) * 100.0, 1) if terminal_total > 0 else 0.0

    def _avg(values: list[float]) -> float | None:
        """Return the rounded arithmetic mean for a list of numeric values."""
        return round(sum(values) / len(values), 1) if values else None

    return {
        "sampled_runs": len(run_rows),
        "resource_scanned_runs": scanned_resources,
        "runs_by_status": {
            "labels": list(by_status.keys()),
            "values": [int(value) for value in by_status.values()],
        },
        "top_users": top_with_other(by_owner, top_users),
        "top_methods": top_with_other(by_method, top_methods),
        "top_datasets": top_with_other(by_dataset, top_datasets),
        "run_duration_buckets": {
            "labels": ["<5m", "5-15m", "15-30m", "30-60m", ">=60m", "unknown"],
            "values": [
                int(duration_buckets.get("<5m", 0)),
                int(duration_buckets.get("5-15m", 0)),
                int(duration_buckets.get("15-30m", 0)),
                int(duration_buckets.get("30-60m", 0)),
                int(duration_buckets.get(">=60m", 0)),
                int(duration_buckets.get("unknown", 0)),
            ],
        },
        "run_outcomes": {
            "success": int(successes),
            "stopped": int(stopped),
            "failed": int(failures),
            "running": int(running),
            "success_ratio": success_ratio,
        },
        "worker_state": {
            "active_workers": active_workers,
            "queued_workers": queued_workers,
            "running_jobs": len(running_jobs),
            "pending_jobs": len(pending_jobs),
        },
        "pending_by_method": top_with_other(pending_by_method, top_methods),
        "queue_kpis": {
            "queue_total": queue_total,
            "queue_pressure": queue_pressure,
        },
        "resource_usage": {
            "cpu_avg": _avg(cpu_values),
            "ram_avg": _avg(ram_values),
            "gpu_avg": _avg(gpu_values),
        },
    }


def row_summary_from_meta(meta: Any, *, svc: Any) -> dict[str, Any]:
    """Build a run summary row from preloaded service metadata."""
    run_id = str(_meta_value(meta, "run_id") or "")
    status = str(_meta_value(meta, "status") or "unknown")
    stop_reason = str(_meta_value(meta, "stop_reason") or "")

    if status == "unknown":
        if svc.is_running(run_id):
            status = "running"
        elif stop_reason:
            status = "stopped"
        else:
            status = "completed"

    best_accuracy = _meta_value(meta, "best_accuracy")
    try:
        if best_accuracy is not None:
            best_accuracy = round(float(best_accuracy), 4)
    except Exception:
        best_accuracy = None

    return {
        "id": run_id,
        "name": str(_meta_value(meta, "display_name") or run_id),
        "owner": str(_meta_value(meta, "owner") or "unknown"),
        "status": status,
        "method": str(_meta_value(meta, "method") or ""),
        "dataset": str(_meta_value(meta, "dataset_name") or ""),
        "evaluation_split_mode": str(_meta_value(meta, "evaluation_split_mode") or "train_val_test"),
        "model_name": str(_meta_value(meta, "model_name") or "DefaultNet"),
        "stop_reason": stop_reason,
        "best_accuracy": best_accuracy,
        "created_at": str(_meta_value(meta, "created_at") or ""),
    }
