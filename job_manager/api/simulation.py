"""Simulation module for fake job execution.

When simulation mode is enabled in ``config.yaml``, jobs are not
executed for real but instead wait for a configurable delay and then
complete, optionally copying fixture files as output and sending a
callback notification.

This is useful for end-to-end testing of the job submission pipeline
without requiring an actual worker command.
"""

import os
import shutil
import time
import logging

from api.config import Config
from api.job_db import register_job, complete_job
from api.notifier import notify_callback

logger = logging.getLogger(__name__)
config = Config()


def run_fake_job(job: dict) -> None:
    """Simulate a job execution with a configurable delay.

    Intended to be called in a daemon thread.  The function:

    1. Registers the job as ``running`` in the database.
    2. Sleeps for ``simulation.delay_seconds``.
    3. Creates an ``output/`` directory with either fixture files
       (if ``simulation.fixture_output_path`` points to a directory)
       or a simple ``result.txt`` placeholder.
    4. Marks the job as ``done`` and sends a callback notification.

    On any exception the job is marked as ``error`` instead.

    Args:
        job: Job payload dict with keys ``job_id``, ``user_id``,
            ``source_app``, ``description``, ``job_type``, and
            ``job_dir``.
    """
    job_id = job["job_id"]
    user_id = job["user_id"]
    source_app = job.get("source_app")
    description = job.get("description")
    job_type = job.get("job_type")
    job_dir = job.get("job_dir", "")

    sim_cfg = config.get("simulation", {})
    delay = sim_cfg.get("delay_seconds", 1)
    fixture_path = sim_cfg.get("fixture_output_path")

    try:
        register_job(job_id, user_id, source_app, description, job_type)
        logger.info(f"[SIM] Job {job_id} started (delay={delay}s)")

        time.sleep(delay)

        output_dir = os.path.join(job_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        if fixture_path and os.path.isdir(fixture_path):
            for item in os.listdir(fixture_path):
                src = os.path.join(fixture_path, item)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(output_dir, item))
        else:
            with open(os.path.join(output_dir, "result.txt"), "w") as f:
                f.write(f"Simulated output for job {job_id}\n")

        complete_job(job_id, success=True)
        logger.info(f"[SIM] Job {job_id} completed successfully")
        notify_callback(job, success=True)

    except Exception as e:
        logger.error(f"[SIM] Job {job_id} failed: {e}")
        complete_job(job_id, success=False, error_msg=str(e))
        notify_callback(job, success=False, error=str(e))
