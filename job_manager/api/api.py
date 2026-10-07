"""
FastAPI router for the Simple Job Manager.

Endpoints:
  POST   /submit-job              — submit a new job
  GET    /job-status/{job_id}     — get status of a single job
  GET    /jobs                    — list all jobs (paginated, filtered)
  GET    /jobs/user/{user_id}    — list jobs for a user (paginated)
  GET    /job-result/{job_id}    — download job output as ZIP
  POST   /jobs/{job_id}/retry    — retry a failed/done job
  POST   /jobs/{job_id}/cancel   — cancel a running job
  DELETE /jobs/{job_id}          — delete a job
  POST   /jobs/bulk-delete       — delete multiple jobs
  GET    /metrics                — system metrics
  GET    /ping                   — health check
  POST   /auth/login             — authenticate and get JWT
"""

import os
import re
import shutil
import uuid
import datetime
import threading
import tempfile
import zipfile
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import jwt as pyjwt

from api.auth import (
    JWT_SECRET,
    JWT_EXPIRATION_HOURS,
    get_current_user,
    get_client_ip,
    login_limiter,
)
from api.config import Config
from api.job_manager import JobQueue
from api.job_db import (
    cancel_queued_job,
    get_job_status,
    get_all_jobs,
    get_jobs_by_user,
    get_job_counts,
    delete_job,
    delete_jobs,
    reset_job,
)
from api.models import (
    BulkDeleteRequest,
    JobListResponse,
    JobSubmitResponse,
    LoginRequest,
    LoginResponse,
    UserJobListResponse,
)
from api.simulation import run_fake_job
from api.user_db import verify_user, is_admin

router = APIRouter()

config = Config()
job_base_path = config.get("job_base_path")
MAX_UPLOAD_BYTES = config.get("max_upload_size_mb", 50) * 1024 * 1024

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    """Simple filename sanitizer: keep only safe characters."""
    name = os.path.basename(name)
    name = re.sub(r'[^\w\s\-.]', '', name).strip()
    return name or "input"


def _resolve_job_dir(job_record: dict) -> str:
    """Internal helper to resolve job dir.

    Args:
        job_record: Input value for `job_record`.

    Returns:
        Result of the operation.

    """
    user_id = job_record.get("user_id", "")
    return os.path.join(job_base_path, _sanitize_filename(user_id), job_record["job_id"])


def _resolve_output_dir(job_dir: str) -> str:
    """Internal helper to resolve output dir.

    Args:
        job_dir: Input value for `job_dir`.

    Returns:
        Result of the operation.

    """
    output_dir = os.path.join(job_dir, "output")
    return output_dir if os.path.isdir(output_dir) else job_dir


def _build_output_zip(output_dir: str, job_id: str) -> str:
    """Internal helper to build output zip.

    Args:
        output_dir: Input value for `output_dir`.
        job_id: Job identifier.

    Returns:
        Result of the operation.

    """
    tmp = tempfile.NamedTemporaryFile(suffix=f"_{job_id}.zip", delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, output_dir)
                zipf.write(full_path, arcname)
    return tmp.name


def _check_job_access(record: dict, current_user: str) -> None:
    """Raise 403 if the user doesn't own the job and isn't admin."""
    if record.get("user_id") != current_user and not is_admin(current_user):
        raise HTTPException(status_code=403, detail="Access denied")


# ---------------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------------

@router.post("/submit-job", response_model=JobSubmitResponse, status_code=202)
async def submit_job(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(""),
    source_app: str = Form(...),
    description: str = Form(""),
    job_type: str = Form("default"),
    run_id: str = Form(""),
    current_user: str = Depends(get_current_user),
):
    """Submit a new job with an uploaded file."""
    user_id = user_id or current_user
    client_ip = get_client_ip(request)
    job_type = job_type.strip() or "default"

    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    if not config.get_app_config(source_app):
        raise HTTPException(status_code=403, detail=f"Unknown app: {source_app}")

    if not config.is_ip_allowed(source_app, client_ip):
        raise HTTPException(status_code=403, detail=f"IP {client_ip} not allowed for {source_app}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)",
        )

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
    user_dir = os.path.join(job_base_path, _sanitize_filename(user_id))
    job_dir = os.path.join(user_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    filename = _sanitize_filename(file.filename or "input")
    input_path = os.path.join(job_dir, filename)
    with open(input_path, "wb") as f:
        f.write(content)

    job_payload = {
        "job_id": job_id,
        "run_id": (run_id or "").strip() or None,
        "user_id": user_id,
        "source_app": source_app,
        "description": description,
        "job_type": job_type,
        "job_dir": job_dir,
        "input_path": input_path,
    }

    sim_cfg = config.get("simulation", {})
    if sim_cfg.get("enabled", False):
        threading.Thread(target=run_fake_job, args=(job_payload,), daemon=True).start()
    else:
        JobQueue.add_job(job_payload)

    return JobSubmitResponse(job_id=job_id, status="queued", job_type=job_type)


@router.get("/job-status/{job_id}")
async def job_status(job_id: str, current_user: str = Depends(get_current_user)):
    """Get the status of a single job."""
    record = get_job_status(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    _check_job_access(record, current_user)
    return record


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    offset: int = 0,
    limit: int = 100,
    status: str | None = None,
    job_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    search: str | None = None,
    current_user: str = Depends(get_current_user),
):
    """List jobs with pagination and optional filters.

    Admin sees all jobs; regular users see only their own.
    """
    kwargs = dict(offset=offset, limit=limit, status=status, job_type=job_type,
                  date_from=date_from, date_to=date_to, search=search)
    if is_admin(current_user):
        jobs = get_all_jobs(**kwargs)
    else:
        jobs = get_jobs_by_user(current_user, **kwargs)
    return JobListResponse(offset=offset, limit=limit, count=len(jobs), jobs=jobs)


@router.get("/jobs/user/{user_id}", response_model=UserJobListResponse)
async def list_user_jobs(
    user_id: str,
    offset: int = 0,
    limit: int = 100,
    status: str | None = None,
    job_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    search: str | None = None,
    current_user: str = Depends(get_current_user),
):
    """List jobs for a specific user with pagination and filters."""
    if user_id != current_user and not is_admin(current_user):
        raise HTTPException(status_code=403, detail="Access denied")
    jobs = get_jobs_by_user(user_id, offset=offset, limit=limit, status=status,
                            job_type=job_type, date_from=date_from, date_to=date_to,
                            search=search)
    return UserJobListResponse(
        user_id=user_id, offset=offset, limit=limit, count=len(jobs), jobs=jobs
    )


@router.get("/job-result/{job_id}")
async def download_job_result(job_id: str, current_user: str = Depends(get_current_user)):
    """Download job output as a ZIP file. Returns 409 if not completed."""
    record = get_job_status(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    _check_job_access(record, current_user)
    if record.get("status") != "done":
        raise HTTPException(status_code=409, detail="Job not completed")

    job_dir = _resolve_job_dir(record)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Job directory not found")

    output_dir = _resolve_output_dir(job_dir)
    zip_path = _build_output_zip(output_dir, job_id)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{job_id}_output.zip",
        background=BackgroundTask(os.remove, zip_path),
    )


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, current_user: str = Depends(get_current_user)):
    """Retry a failed or completed job by re-queuing it.

    Args:
        job_id: Job identifier to reset and enqueue.
        current_user: Authenticated username injected by FastAPI.

    Returns:
        A response payload confirming the queued retry.
    """
    record = get_job_status(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    _check_job_access(record, current_user)

    if record.get("status") == "running":
        raise HTTPException(status_code=409, detail="Job is already running")
    if record.get("status") == "queued":
        raise HTTPException(status_code=409, detail="Job is already queued")

    reset_record = reset_job(job_id)
    if not reset_record:
        raise HTTPException(status_code=409, detail="Could not reset job")

    job_dir = _resolve_job_dir(reset_record)
    job_payload = {
        "job_id": job_id,
        "user_id": reset_record["user_id"],
        "source_app": reset_record.get("source_app"),
        "description": reset_record.get("description"),
        "job_type": reset_record.get("job_type"),
        "run_id": reset_record.get("run_id"),
        "job_dir": job_dir,
        "input_path": "",
    }

    # Find the original input file if it still exists
    if os.path.isdir(job_dir):
        for f in os.listdir(job_dir):
            fpath = os.path.join(job_dir, f)
            if os.path.isfile(fpath) and f != "output":
                job_payload["input_path"] = fpath
                break

    # Clean old output
    output_dir = os.path.join(job_dir, "output")
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)

    JobQueue.add_job(job_payload)
    return {"job_id": job_id, "status": "queued", "message": "Job re-queued for retry"}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, current_user: str = Depends(get_current_user)):
    """Cancel a queued or running job."""
    record = get_job_status(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    _check_job_access(record, current_user)

    status = str(record.get("status") or "").lower()
    if status == "queued":
        if not cancel_queued_job(job_id):
            raise HTTPException(status_code=409, detail="Could not cancel queued job")
        return {"job_id": job_id, "message": "Queued job cancelled"}

    if status == "running":
        if not JobQueue.cancel_job(job_id):
            raise HTTPException(status_code=409, detail="Could not cancel job (process not found)")
        return {"job_id": job_id, "message": "Cancel signal sent"}

    raise HTTPException(status_code=409, detail=f"Job is not cancellable from status '{status}'")


@router.delete("/jobs/{job_id}")
async def delete_job_endpoint(job_id: str, current_user: str = Depends(get_current_user)):
    """Delete a job by ID, removing both DB record and job directory."""
    record = get_job_status(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    _check_job_access(record, current_user)
    if record.get("status") == "running" and JobQueue.is_job_active(job_id):
        raise HTTPException(status_code=409, detail="Cannot delete a running job")

    job_dir = _resolve_job_dir(record)
    delete_job(job_id)
    if os.path.isdir(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)

    return {"deleted": True, "job_id": job_id}


@router.post("/jobs/bulk-delete")
async def bulk_delete_jobs(
    body: BulkDeleteRequest,
    current_user: str = Depends(get_current_user),
):
    """Delete multiple jobs at once. Running jobs are skipped."""
    if not body.job_ids:
        raise HTTPException(status_code=400, detail="No job_ids provided")

    ids_to_delete = body.job_ids
    if not is_admin(current_user):
        allowed = []
        for jid in body.job_ids:
            record = get_job_status(jid)
            if record and record.get("user_id") == current_user:
                allowed.append(jid)
        ids_to_delete = allowed

    deleted_count = delete_jobs(ids_to_delete)

    # Clean up directories (best effort)
    if os.path.isdir(job_base_path):
        for jid in ids_to_delete:
            for entry in os.scandir(job_base_path):
                job_dir = os.path.join(entry.path, jid)
                if os.path.isdir(job_dir):
                    shutil.rmtree(job_dir, ignore_errors=True)
                    break

    return {"deleted_count": deleted_count, "requested": len(body.job_ids)}


# ---------------------------------------------------------------------------
# Metrics & health
# ---------------------------------------------------------------------------

@router.get("/metrics")
async def metrics(current_user: str = Depends(get_current_user)):
    """System metrics: job counts by status, queue size, active workers. Admin only."""
    if not is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin only")

    counts = get_job_counts()
    return {
        "jobs_by_status": counts,
        "jobs_total": sum(counts.values()),
        "queue_pending": JobQueue.queue_size(),
        "workers_active": JobQueue.active_count(),
        "workers_max": config.get("max_concurrent_jobs", 3),
    }


@router.api_route("/ping", methods=["GET", "POST"])
async def ping(request: Request):
    """Health-check endpoint (no auth required)."""
    return {
        "message": "pong",
        "method": request.method,
        "ip": get_client_ip(request),
    }


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@router.post("/auth/login", response_model=LoginResponse)
async def auth_login(body: LoginRequest, request: Request):
    """Authenticate and return a JWT token. Rate-limited per IP."""
    client_ip = get_client_ip(request)

    if not login_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many login attempts, try again later")

    if not verify_user(body.username, body.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    payload = {
        "sub": body.username,
        "iat": datetime.datetime.now(datetime.timezone.utc),
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    token = pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return LoginResponse(token=token, username=body.username)
