"""Job database operations for the shared MariaDB schema.

Provides functions for the full lifecycle of a job record:
create -> queue -> run -> complete (or error), plus query, reset,
and delete operations.

All functions use the shared :func:`api.db.connection` context
manager for automatic commit / rollback.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Any

from api.db import connection

logger = logging.getLogger(__name__)

_JOB_COLUMNS = (
    "job_id, owner_identifier AS user_id, run_id, source_app, description, job_type, status, "
    "start_time, end_time, error"
)
"""Column list used in all SELECT queries."""

_JOB_SELECT_SQL = f"SELECT {_JOB_COLUMNS} FROM jobs"
"""Base SELECT statement shared by job lookup functions."""


def _utc_timestamp() -> str:
    """Return the current UTC timestamp string used in job rows.

    Returns:
        An ISO-8601 timestamp generated from the current UTC time.
    """
    return datetime.now(timezone.utc).isoformat()


def _normalize_job_type(job_type: str | None) -> str:
    """Normalize a job-type label and apply the default value.

    Args:
        job_type: Raw job type supplied by a caller.

    Returns:
        A stripped non-empty job type, or ``"default"`` when the input is empty.
    """
    return (job_type or "default").strip() or "default"


def init_db() -> None:
    """Verify that the ``jobs`` table is available.

    Side Effects:
        Executes a lightweight read against ``jobs`` so startup fails clearly
        when ``database/schema/baseline.sql`` has not initialized MariaDB.

    Notes:
        ``database/schema/baseline.sql`` remains the schema source.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM jobs LIMIT 1")


def queue_job(
    job_id: str,
    user_id: str,
    source_app: str,
    description: str | None = None,
    job_type: str = "default",
    run_id: str | None = None,
) -> None:
    """Insert a new job record in ``queued`` state.

    Args:
        job_id: Unique external job identifier.
        user_id: Owner identifier to persist as ``owner_identifier``.
        source_app: Source application that submitted the job.
        description: Optional human-readable job description.
        job_type: Optional job type label; blank values become ``"default"``.
        run_id: Optional run identifier associated with the job.

    Side Effects:
        Inserts a row with ``INSERT IGNORE`` semantics, so duplicate ``job_id``
        values are skipped without raising.
    """
    job_type = _normalize_job_type(job_type)
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT IGNORE INTO jobs
                    (job_id, kind, owner_identifier, run_id, source_app, description, job_type, status, start_time, end_time, error, created_at, updated_at)
                VALUES
                    (%s, 'pending', %s, %s, %s, %s, %s, 'queued', NULL, NULL, NULL, UTC_TIMESTAMP(6), UTC_TIMESTAMP(6))
                """,
                (job_id, user_id, run_id, source_app, description, job_type),
            )


def register_job(
    job_id: str,
    user_id: str,
    source_app: str,
    description: str | None = None,
    job_type: str = "default",
    run_id: str | None = None,
) -> None:
    """Transition a job to ``running`` and stamp ``start_time``.

    Args:
        job_id: Unique external job identifier.
        user_id: Owner identifier used if a new row must be inserted.
        source_app: Source application used if a new row must be inserted.
        description: Optional description used if a new row must be inserted.
        job_type: Optional job type used if a new row must be inserted.
        run_id: Optional run identifier associated with the job.

    Side Effects:
        Updates an existing row when possible; if no row exists, inserts one in
        ``running`` state.
    """
    now_iso = _utc_timestamp()
    job_type = _normalize_job_type(job_type)
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = 'running', kind = 'running', run_id = COALESCE(run_id, %s), start_time = %s, error = NULL, updated_at = UTC_TIMESTAMP(6) "
                "WHERE job_id = %s",
                (run_id, now_iso, job_id),
            )
            if cur.rowcount == 0:
                cur.execute(
                    """
                    INSERT INTO jobs
                        (job_id, kind, owner_identifier, run_id, source_app, description, job_type, status, start_time, end_time, error, created_at, updated_at)
                    VALUES
                        (%s, 'running', %s, %s, %s, %s, %s, 'running', %s, NULL, NULL, UTC_TIMESTAMP(6), UTC_TIMESTAMP(6))
                    """,
                    (job_id, user_id, run_id, source_app, description, job_type, now_iso),
                )


def complete_job(job_id: str, success: bool = True, error_msg: str | None = None) -> None:
    """Mark a job as ``done`` or ``error``.

    Args:
        job_id: Unique external job identifier.
        success: Whether to store ``done`` instead of ``error``.
        error_msg: Optional error message persisted for the terminal state.

    Side Effects:
        Updates the job status, resets ``kind`` to ``pending``, stores an
        ``end_time`` timestamp, and records the optional error message.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = %s, kind = 'pending', end_time = %s, error = %s, updated_at = UTC_TIMESTAMP(6) WHERE job_id = %s",
                (
                    "done" if success else "error",
                    _utc_timestamp(),
                    error_msg,
                    job_id,
                ),
            )


def cancel_queued_job(job_id: str) -> bool:
    """Mark a queued job as cancelled.

    Args:
        job_id: Unique external job identifier.

    Returns:
        ``True`` when a queued row was updated, otherwise ``False``.

    Side Effects:
        Sets ``status`` to ``cancelled``, records a cancellation reason, and
        stamps ``end_time`` only if the current row is still queued.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = 'cancelled', kind = 'pending', end_time = %s, error = %s, updated_at = UTC_TIMESTAMP(6) "
                "WHERE job_id = %s AND status = 'queued'",
                (_utc_timestamp(), "Cancelled by user", job_id),
            )
            return cur.rowcount > 0


def reset_job(job_id: str) -> dict | None:
    """Reset a non-running job back to ``queued`` for retry.

    Args:
        job_id: Unique external job identifier.

    Returns:
        The pre-update job record with its ``status`` changed to ``queued`` in
        memory, or ``None`` when no job exists or the job is currently running.

    Side Effects:
        Clears timing/error columns and sets ``kind`` to ``pending`` for the
        persisted row when it is eligible for reset.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"{_JOB_SELECT_SQL} WHERE job_id = %s", (job_id,))
            record = cur.fetchone()
            if record is None:
                return None
            if record["status"] == "running":
                return None
            cur.execute(
                "UPDATE jobs SET status = 'queued', kind = 'pending', start_time = NULL, end_time = NULL, error = NULL, updated_at = UTC_TIMESTAMP(6) WHERE job_id = %s",
                (job_id,),
            )
            record["status"] = "queued"
            return record


def get_job_status(job_id: str) -> dict | None:
    """Fetch a single job by its external identifier.

    Args:
        job_id: Unique external job identifier.

    Returns:
        The matching job dictionary if found, otherwise ``None``.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"{_JOB_SELECT_SQL} WHERE job_id = %s", (job_id,))
            return cur.fetchone()


def get_all_jobs(
    offset: int = 0,
    limit: int = 100,
    status: str | None = None,
    job_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    search: str | None = None,
) -> list[dict]:
    """Retrieve all jobs ordered by ``start_time`` descending.

    Args:
        offset: Zero-based pagination offset.
        limit: Maximum number of rows to return.
        status: Optional exact status filter.
        job_type: Optional exact job type filter.
        date_from: Optional inclusive lower bound for ``start_time``.
        date_to: Optional inclusive upper bound for ``start_time``.
        search: Optional substring filter applied to description and job ID.

    Returns:
        A list of job dictionaries matching the requested filters.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            query = _JOB_SELECT_SQL
            params: list[Any] = []
            clauses: list[str] = []
            _apply_filters(clauses, params, status, job_type, date_from, date_to, search)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY start_time DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            cur.execute(query, params)
            return cur.fetchall()


def get_jobs_by_user(
    user_id: str,
    offset: int = 0,
    limit: int = 100,
    status: str | None = None,
    job_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    search: str | None = None,
) -> list[dict]:
    """Retrieve jobs for a specific user with optional filters.

    Args:
        user_id: Owner identifier used to filter jobs.
        offset: Zero-based pagination offset.
        limit: Maximum number of rows to return.
        status: Optional exact status filter.
        job_type: Optional exact job type filter.
        date_from: Optional inclusive lower bound for ``start_time``.
        date_to: Optional inclusive upper bound for ``start_time``.
        search: Optional substring filter applied to description and job ID.

    Returns:
        A list of job dictionaries matching the user and requested filters.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            clauses = ["owner_identifier = %s"]
            params: list[Any] = [user_id]
            _apply_filters(clauses, params, status, job_type, date_from, date_to, search)
            query = (
                f"{_JOB_SELECT_SQL} "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY start_time DESC LIMIT %s OFFSET %s"
            )
            params.extend([limit, offset])
            cur.execute(query, params)
            return cur.fetchall()


def get_job_counts() -> dict[str, int]:
    """Return the number of jobs grouped by status.

    Returns:
        Mapping from status string to count.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status")
            return {row["status"]: row["count"] for row in cur.fetchall()}


def delete_job(job_id: str) -> dict | None:
    """Delete a single job record.

    Args:
        job_id: Unique external job identifier.

    Returns:
        The deleted job record if it existed, otherwise ``None``.

    Side Effects:
        Removes the matching row from ``jobs`` after reading it.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"{_JOB_SELECT_SQL} WHERE job_id = %s", (job_id,))
            record = cur.fetchone()
            if record is None:
                return None
            cur.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
            return record


def delete_jobs(job_ids: list[str]) -> int:
    """Delete multiple non-running job records in one statement.

    Args:
        job_ids: Job identifiers to delete.

    Returns:
        Number of rows deleted. Running jobs are excluded by the SQL predicate.
    """
    if not job_ids:
        return 0
    with connection() as conn:
        with conn.cursor() as cur:
            placeholders = ", ".join(["%s"] * len(job_ids))
            cur.execute(
                f"DELETE FROM jobs WHERE job_id IN ({placeholders}) AND status != 'running'",
                job_ids,
            )
            return cur.rowcount


def _apply_filters(
    clauses: list[str],
    params: list[Any],
    status: str | None,
    job_type: str | None,
    date_from: str | None,
    date_to: str | None,
    search: str | None,
) -> None:
    """Append SQL WHERE clauses for shared filter parameters.

    Args:
        clauses: Mutable list of SQL predicate fragments to append to.
        params: Mutable positional-parameter list to append to.
        status: Optional exact status filter.
        job_type: Optional exact job type filter.
        date_from: Optional inclusive lower bound for ``start_time``.
        date_to: Optional inclusive upper bound for ``start_time``.
        search: Optional substring filter applied to description and job ID.

    Side Effects:
        Mutates ``clauses`` and ``params`` in place.
    """
    if status:
        clauses.append("status = %s")
        params.append(status)
    if job_type:
        clauses.append("job_type = %s")
        params.append(job_type)
    if date_from:
        clauses.append("start_time >= %s")
        params.append(date_from)
    if date_to:
        clauses.append("start_time <= %s")
        params.append(date_to)
    if search:
        clauses.append("(description LIKE %s OR job_id LIKE %s)")
        like = f"%{search}%"
        params.extend([like, like])


def upsert_run_metadata(
    run_id: str,
    owner_identifier: str,
    display_name: str,
    method: str | None,
    dataset_name: str | None,
    evaluation_split_mode: str | None,
    model_name: str | None,
    config_json: dict | None,
) -> None:
    """Ensure the submitted run row exists without overwriting Faster metadata.

    Args:
        run_id: Unique external run identifier.
        owner_identifier: Owner identifier to store when inserting a missing
            run row.
        display_name: Human-readable run display name to store when inserting
            a missing run row.
        method: Optional training method label used for missing run rows.
        dataset_name: Optional dataset label used for missing run rows.
        evaluation_split_mode: Optional evaluation split mode label used for
            missing run rows.
        model_name: Optional model label used for missing run rows.
        config_json: Optional JSON-serializable config payload to insert when a
            ``run_configs`` row does not already exist.

    Side Effects:
        Inserts a missing ``runs`` row for direct Job Manager submissions. If
        Faster already reserved the run, product-level metadata is left
        unchanged so Faster remains the config owner. When
        ``config_json`` is not ``None``, inserts the associated
        ``run_configs`` row only if it is missing.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs
                    (run_id, owner_identifier, display_name, status, method, dataset_name, evaluation_split_mode, model_name, stop_reason, best_accuracy, created_at, updated_at)
                VALUES
                    (%s, %s, %s, 'running', %s, %s, %s, %s, NULL, NULL, UTC_TIMESTAMP(6), UTC_TIMESTAMP(6))
                ON DUPLICATE KEY UPDATE
                    updated_at = UTC_TIMESTAMP(6)
                """,
                (
                    run_id,
                    owner_identifier,
                    display_name,
                    method,
                    dataset_name,
                    evaluation_split_mode,
                    model_name,
                ),
            )
            if config_json is not None:
                cur.execute(
                    """
                    INSERT INTO run_configs (run_id, config_json, created_at, updated_at)
                    VALUES (%s, %s, UTC_TIMESTAMP(6), UTC_TIMESTAMP(6))
                    ON DUPLICATE KEY UPDATE
                        updated_at = UTC_TIMESTAMP(6)
                    """,
                    (run_id, json.dumps(config_json, ensure_ascii=False)),
                )


def update_run_status(run_id: str, status: str, stop_reason: str | None = None) -> None:
    """Update run lifecycle state for one run id.

    Args:
        run_id: Unique external run identifier.
        status: New run status value to persist.
        stop_reason: Optional stop reason or error text to store.

    Side Effects:
        Updates the matching row in ``runs`` and refreshes ``updated_at``.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE runs SET status = %s, stop_reason = %s, updated_at = UTC_TIMESTAMP(6) WHERE run_id = %s",
                (status, stop_reason, run_id),
            )
