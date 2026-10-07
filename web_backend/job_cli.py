"""CLI utility to submit, inspect, and cancel FASTER Job Manager jobs.

Submits runs through the same pipeline as the web UI (validation + DB run row
+ Job Manager queue), lists active/queued jobs, and cancels them — all from
the terminal. Runs inside the ``faster-app`` container so it can reach both the
Job Manager (via the service account) and MariaDB.

Role-aware, mirroring the web UI: a normal ``--user`` may list the active
queue (other users' jobs masked to status + queue position only) but may
cancel or attach to **their own** runs only; an admin sees and controls every
job. The role is resolved from the database for the given ``--user``.

Examples:
    python3 -m web_backend.job_cli submit --config config.yaml --user admin
    python3 -m web_backend.job_cli list --user alice
    python3 -m web_backend.job_cli list --user admin --all
    python3 -m web_backend.job_cli cancel --job-id 20260617_071623_99e212 --user alice
    python3 -m web_backend.job_cli attach --run-id admin_20260617_071623 --user admin
"""

import argparse
import os
import sys
import time

import yaml

from web_backend.jm_client import JobManagerClient
from web_backend.service import BASE_RESULTS_PATH, svc

_ACTIVE = {"queued", "pending", "running"}
_TERMINAL = {"done", "completed", "failed", "cancelled", "stopped", "error"}


def _resolve_user(user: str) -> tuple[str, bool] | None:
    """Resolve ``--user`` to ``(identifier, is_admin)`` from the database.

    Returns ``None`` (after printing an error) if the user does not exist, so
    callers can stop with a non-zero exit code.
    """
    from web_backend.api_handlers.api import _auth

    account = _auth.get_user(user)
    if not account:
        print(f"User '{user}' not found. Create it with auth.user_cli first.", file=sys.stderr)
        return None
    return str(account["identifier"]), account.get("role") == "admin"


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for Job Manager job commands."""
    p = argparse.ArgumentParser(description="Submit, inspect, and cancel FASTER Job Manager jobs")
    sub = p.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="Submit a run from a config file (queued via Job Manager)")
    submit.add_argument("--config", required=True, help="Path to a run config YAML file")
    submit.add_argument("--user", default="admin", help="Owner of the run (default: admin)")

    list_cmd = sub.add_parser("list", help="List active (queued/running) jobs")
    list_cmd.add_argument("--user", default="admin", help="Acting user; non-admins see others' jobs masked (default: admin)")
    list_cmd.add_argument("--all", action="store_true", help="Include terminal jobs (done/failed/cancelled)")
    list_cmd.add_argument("--limit", type=int, default=200, help="Max jobs to fetch (default: 200)")

    cancel = sub.add_parser("cancel", help="Cancel a queued or running job by id")
    cancel.add_argument("--job-id", required=True, help="Job Manager job id to cancel")
    cancel.add_argument("--user", default="admin", help="Acting user; non-admins may cancel only their own jobs (default: admin)")

    attach = sub.add_parser("attach", help="Live-tail a run's logs (Ctrl+C to detach)")
    target = attach.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id", help="Run id (results/<run-id>/) to follow")
    target.add_argument("--job-id", help="Job Manager job id to resolve to its run and follow")
    attach.add_argument("--user", default="admin", help="Acting user; non-admins may attach only to their own runs (default: admin)")
    attach.add_argument("--lines", type=int, default=40, help="Backlog lines to show first (default: 40)")
    attach.add_argument("--interval", type=float, default=2.0, help="Poll interval in seconds (default: 2.0)")
    return p


def _cmd_submit(config_path: str, user: str) -> int:
    """Submit a run config through the same pipeline as the web UI."""
    # Imported here so `list`/`cancel` don't pay the cost of wiring the app layer.
    from web_backend.api_handlers.api import _submit_run_to_job_manager

    if not os.path.isfile(config_path):
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1
    try:
        with open(config_path) as fh:
            config = yaml.safe_load(fh) or {}
    except Exception as exc:
        print(f"Failed to read config: {exc}", file=sys.stderr)
        return 1
    if not isinstance(config, dict):
        print("Config must be a YAML mapping.", file=sys.stderr)
        return 1

    resolved = _resolve_user(user)
    if resolved is None:
        return 1
    identifier, is_admin = resolved

    ok, payload = _submit_run_to_job_manager(config, identifier, is_admin=is_admin)
    if not ok:
        print(f"Submit failed: {payload.get('error') or payload}", file=sys.stderr)
        return 1

    run_id = str(payload.get("run_id") or "")
    print(f"Submitted run {run_id} (job {payload.get('job_id')}) for user {identifier}")
    print(f"Results path: {os.path.join(BASE_RESULTS_PATH, run_id)}")
    return 0


def _cmd_list(jm: JobManagerClient, user: str, include_all: bool, limit: int) -> int:
    """Print active jobs, masking other users' jobs for non-admins.

    A normal user sees every active job so they can gauge how many sit ahead of
    theirs, but other users' jobs show only status + queue position — no job id,
    owner, or run id (mirroring the web UI). Admins see everything unmasked.
    """
    resolved = _resolve_user(user)
    if resolved is None:
        return 1
    identifier, is_admin = resolved

    ok, jobs = jm.list_jobs(limit=limit)
    if not ok:
        print("Failed to reach Job Manager (check service account / connectivity).", file=sys.stderr)
        return 1
    rows = [
        j for j in jobs
        if include_all or str(j.get("status") or "").lower() in _ACTIVE
    ]
    if not rows:
        print("No active jobs." if not include_all else "No jobs.")
        return 0

    # Queue position: running jobs first, then queued in id order (matches backend).
    active = [j for j in rows if str(j.get("status") or "").lower() in _ACTIVE]
    active.sort(key=lambda j: (0 if str(j.get("status") or "").lower() == "running" else 1, str(j.get("job_id") or "")))
    position = {}
    counter = 0
    for j in active:
        if str(j.get("status") or "").lower() in {"queued", "pending"}:
            counter += 1
            position[str(j.get("job_id") or "")] = counter

    print(f"{'JOB ID':<24} {'STATUS':<10} {'USER':<16} {'RUN ID':<24} {'QUEUE':<6}")
    for j in rows:
        job_id = str(j.get("job_id") or "")
        is_mine = is_admin or str(j.get("user_id") or "") == identifier
        pos = position.get(job_id)
        pos_str = f"#{pos}" if isinstance(pos, int) else "-"
        if is_mine:
            print(
                f"{job_id:<24} "
                f"{str(j.get('status') or ''):<10} "
                f"{str(j.get('user_id') or ''):<16} "
                f"{str(j.get('run_id') or ''):<24} "
                f"{pos_str:<6}"
            )
        else:
            # Masked: identity hidden, only status + queue position remain.
            print(
                f"{'(hidden)':<24} "
                f"{str(j.get('status') or ''):<10} "
                f"{'(hidden)':<16} "
                f"{'(hidden)':<24} "
                f"{pos_str:<6}"
            )
    return 0


def _cmd_cancel(jm: JobManagerClient, job_id: str, user: str) -> int:
    """Cancel one job (own jobs only for non-admins) and sync the Faster run row."""
    resolved = _resolve_user(user)
    if resolved is None:
        return 1
    identifier, is_admin = resolved

    ok, jobs = jm.list_jobs(limit=500)
    run_id = ""
    if ok:
        match = next((j for j in jobs if str(j.get("job_id") or "") == job_id), None)
        if not match:
            print(f"Job '{job_id}' not found.", file=sys.stderr)
            return 1
        if not is_admin and str(match.get("user_id") or "") != identifier:
            print(f"Forbidden: job '{job_id}' belongs to another user.", file=sys.stderr)
            return 1
        run_id = str(match.get("run_id") or "")
    ok_cancel, payload = jm.cancel_job(job_id)
    if not ok_cancel:
        print(f"Cancel failed: {payload.get('error') or payload.get('detail') or payload}", file=sys.stderr)
        return 1
    if run_id:
        svc.mark_run_status(run_id, "stopped", "Cancelled via CLI")
    print(f"Cancelled job {job_id}" + (f" (run {run_id})" if run_id else ""))
    return 0


def _resolve_run_id(
    jm: JobManagerClient, run_id: str, job_id: str, identifier: str, is_admin: bool
) -> str:
    """Return the run id to follow, resolving from a job id and enforcing ownership.

    Non-admins may follow only their own runs; an unowned or unknown target is
    rejected with an empty return (after printing why).
    """
    ok, jobs = jm.list_jobs(limit=500)
    if not ok:
        print("Failed to reach Job Manager.", file=sys.stderr)
        return ""
    if run_id:
        match = next((j for j in jobs if str(j.get("run_id") or "") == run_id), None)
    else:
        match = next((j for j in jobs if str(j.get("job_id") or "") == job_id), None)

    if not match:
        # A --job-id must resolve to a live job; a --run-id may point at a run
        # whose job has aged out of the queue. Admins can still follow it; a
        # non-admin can't be ownership-checked without the job row, so refuse.
        if job_id:
            print(f"Job '{job_id}' not found.", file=sys.stderr)
            return ""
        if not is_admin:
            print("Forbidden: cannot verify ownership of that run.", file=sys.stderr)
            return ""
        return run_id

    if not is_admin and str(match.get("user_id") or "") != identifier:
        print("Forbidden: that run belongs to another user.", file=sys.stderr)
        return ""
    return str(match.get("run_id") or run_id)


def _cmd_attach(
    jm: JobManagerClient, run_id: str, job_id: str, user: str, lines: int, interval: float
) -> int:
    """Live-tail a run's train.log until the run finishes or the user detaches."""
    resolved = _resolve_user(user)
    if resolved is None:
        return 1
    identifier, is_admin = resolved

    run_id = _resolve_run_id(jm, run_id, job_id, identifier, is_admin)
    if not run_id:
        return 1

    run_path = os.path.join(BASE_RESULTS_PATH, run_id)
    print(f"Attached to run {run_id} (status: {svc.get_run_status(run_id) or 'unknown'}). Ctrl+C to detach.\n")

    seen = 0  # number of log lines already printed
    backlog = True
    try:
        while True:
            text = svc.read_logs(run_path, n=lines if backlog else 5000)
            log_lines = text.splitlines() if text else []
            if backlog:
                for line in log_lines:
                    print(line)
                seen = len(log_lines)
                backlog = False
            elif len(log_lines) > seen:
                for line in log_lines[seen:]:
                    print(line)
                seen = len(log_lines)

            status = (svc.get_run_status(run_id) or "").lower()
            if status in _TERMINAL:
                print(f"\n-- run {run_id} {status}; detaching --")
                return 0
            time.sleep(max(0.5, interval))
    except KeyboardInterrupt:
        print(f"\n-- detached from {run_id} (run keeps going in the background) --")
        return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    # Each CLI call is a fresh process, so without a shared token cache every
    # command would re-login and quickly hit the Job Manager's per-IP login
    # rate limit when several commands run in quick succession. Enable the
    # on-disk cache by default (overridable via the env var).
    os.environ.setdefault("FL_JM_TOKEN_CACHE", "/tmp/.faster_jm_token.json")
    jm = JobManagerClient()
    if not jm.enabled:
        print("Job Manager integration is not configured (JM_API_BASE_URL).", file=sys.stderr)
        return 1
    if args.command == "submit":
        return _cmd_submit(args.config, args.user)
    if args.command == "list":
        return _cmd_list(jm, args.user, args.all, args.limit)
    if args.command == "cancel":
        return _cmd_cancel(jm, args.job_id, args.user)
    if args.command == "attach":
        return _cmd_attach(jm, args.run_id or "", args.job_id or "", args.user, args.lines, args.interval)
    return 2


if __name__ == "__main__":
    sys.exit(main())
