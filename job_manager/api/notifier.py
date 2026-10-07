"""Callback notification module with retry support.

Sends job results to a configured callback URL when the app's
``notify_mode`` is set to ``"callback"``.  Retries with exponential
backoff on transient failures (5xx, 429, network errors).

Module-level constants (from ``config.yaml``):
    ``MAX_SIZE``
        Maximum allowed ZIP size in bytes before the callback is
        skipped.
    ``MAX_RETRIES``
        Number of delivery attempts before giving up.
    ``RETRY_BASE_DELAY``
        Initial delay in seconds; doubles after each retry.
"""

import os
import json
import time
import zipfile
import tempfile
import logging

import requests

from api.config import Config

logger = logging.getLogger(__name__)
config = Config()

MAX_SIZE: int = config.get("max_callback_size_kb", 5000) * 1024
MAX_RETRIES: int = config.get("callback_max_retries", 3)
RETRY_BASE_DELAY: int = config.get("callback_retry_delay_seconds", 2)


def notify_callback(
    job: dict, success: bool = True, error: str | None = None
) -> None:
    """Send job results to the source app's callback URL.

    Compresses the ``output/`` directory into a ZIP and POSTs it
    together with a JSON metadata payload.  The call is silently
    skipped when the app's ``notify_mode`` is not ``"callback"``.

    Args:
        job: Job payload dict with keys ``job_id``, ``user_id``,
            ``job_dir``, and ``source_app``.
        success: Whether the job completed successfully.
        error: Error message to include in the metadata (only
            meaningful when *success* is ``False``).
    """
    job_id = job["job_id"]
    user_id = job["user_id"]
    job_dir = job["job_dir"]
    source_app = job.get("source_app")

    notify_mode = config.get_notify_mode(source_app, default="callback")
    if notify_mode != "callback":
        logger.info(f"[{job_id}] notify_mode='{notify_mode}', skipping callback")
        return

    callback_url = config.get_callback_url(source_app)
    if not callback_url:
        logger.error(f"[{job_id}] No callback URL configured for app '{source_app}'")
        return

    output_dir = os.path.join(job_dir, "output")
    if not os.path.isdir(output_dir):
        output_dir = job_dir

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
            tmp_path = tmp_zip.name
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, output_dir)
                        zipf.write(full_path, arcname)

        zip_size = os.path.getsize(tmp_path)
        if zip_size > MAX_SIZE:
            logger.error(f"[{job_id}] ZIP exceeds max size ({zip_size} bytes)")
            return

        metadata = {
            "job_id": job_id,
            "user_id": user_id,
            "source_app": source_app,
            "status": "done" if success else "error",
            "error": error,
        }

        _send_with_retry(callback_url, tmp_path, metadata, job_id, source_app)

    except Exception as e:
        logger.error(f"[{job_id}] Failed to send callback: {e}")
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _send_with_retry(
    callback_url: str,
    zip_path: str,
    metadata: dict,
    job_id: str,
    source_app: str,
) -> None:
    """POST the ZIP + metadata with exponential-backoff retries.

    Retryable conditions: HTTP 5xx, 429, and network-level errors.
    Client errors (4xx except 429) are treated as permanent failures
    and are not retried.

    Args:
        callback_url: The target URL.
        zip_path: Path to the temporary ZIP file on disk.
        metadata: JSON-serializable dict with job metadata.
        job_id: Used for log messages.
        source_app: Used for log messages.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(zip_path, "rb") as f:
                files = {"file": (f"{job_id}_output.zip", f, "application/zip")}
                data = {"metadata": json.dumps(metadata)}
                response = requests.post(
                    callback_url, data=data, files=files, timeout=30
                )

            if response.status_code == 200:
                logger.info(f"[{job_id}] Callback sent to {source_app} successfully")
                return

            if 400 <= response.status_code < 500 and response.status_code != 429:
                logger.warning(
                    f"[{job_id}] {source_app} responded with {response.status_code}, not retrying"
                )
                return

            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            logger.warning(
                f"[{job_id}] Callback attempt {attempt}/{MAX_RETRIES} failed: {last_error}"
            )

        except requests.RequestException as e:
            last_error = str(e)
            logger.warning(
                f"[{job_id}] Callback attempt {attempt}/{MAX_RETRIES} error: {last_error}"
            )

        if attempt < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            time.sleep(delay)

    logger.error(
        f"[{job_id}] Callback to {source_app} failed after {MAX_RETRIES} attempts: {last_error}"
    )
