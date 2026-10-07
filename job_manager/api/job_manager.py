"""Background job manager with thread-safe queue and concurrency control.

Jobs are executed as subprocess commands.  The command template is
configurable in ``config.yaml`` under ``job_command``.  If no command
is configured, the job completes immediately as a no-op (useful for
testing the queue infrastructure).

Architecture:
    :class:`JobQueue`
        Class-level (singleton) thread-safe FIFO queue with active
        job tracking and process-level cancellation support.

    :class:`JobManager`
        Background loop that pulls jobs from :class:`JobQueue` and
        executes them in worker threads, respecting
        ``max_concurrent_jobs``.
"""

import os
import signal
import subprocess
import threading
import time
from queue import Queue
from datetime import datetime
import yaml

import psutil

try:
    import GPUtil
except Exception:  # pragma: no cover - optional runtime dependency
    GPUtil = None

from api.logger import get_logger
from api.job_db import (
    queue_job,
    register_job,
    complete_job,
    get_job_status,
    upsert_run_metadata,
    update_run_status,
)
from api.notifier import notify_callback
from api.config import Config
from api.resource_usage import append_resource_sample_file, safe_float
from utils.config import normalize_run_config_aliases

logger = get_logger()
config = Config()
MAX_RESOURCE_SAMPLES = 2000
RESOURCE_SAMPLE_INTERVAL_S = 5


def _dataset_display_name_from_cfg(cfg: dict) -> str:
    """Return the user-facing dataset label from a run config."""
    dataset_name = str(cfg.get("dataset_name") or "")
    if dataset_name in {"custom_csv", "custom_image_npz", "custom_image_folder"}:
        return str(cfg.get("custom_dataset_name") or dataset_name)
    return dataset_name


class JobQueue:
    """Thread-safe job queue with active job tracking and cancellation.

    All methods are ``@classmethod`` — the queue is a process-wide
    singleton backed by class-level data structures.
    """

    _queue: Queue = Queue()
    _active_jobs: dict[str, dict] = {}
    _processes: dict[str, subprocess.Popen] = {}
    _lock = threading.Lock()

    @classmethod
    def add_job(cls, job: dict) -> None:
        """Enqueue a job and persist it in the database.

        Args:
            job: Job payload dict with at least ``job_id`` and ``user_id``.
        """
        logger.info(f"Job queued: {job['job_id']} from app {job.get('source_app')}")
        try:
            queue_job(
                job["job_id"],
                job["user_id"],
                job.get("source_app"),
                job.get("description"),
                job.get("job_type"),
                job.get("run_id"),
            )
        except Exception as e:
            logger.error(f"Error registering queued job {job['job_id']}: {e}")
        cls._queue.put(job)

    @classmethod
    def get_next_job(cls) -> dict:
        """Block until a job is available and return it.

        Returns:
            The next job payload dict from the queue.
        """
        return cls._queue.get()

    @classmethod
    def mark_job_start(cls, job: dict) -> None:
        """Record a job as actively running.

        Args:
            job: The job payload dict.
        """
        with cls._lock:
            cls._active_jobs[job["job_id"]] = job

    @classmethod
    def mark_job_done(cls, job: dict) -> None:
        """Remove a job from the active set and process registry.

        Args:
            job: The job payload dict.
        """
        with cls._lock:
            cls._active_jobs.pop(job["job_id"], None)
            cls._processes.pop(job["job_id"], None)

    @classmethod
    def register_process(cls, job_id: str, process: subprocess.Popen) -> None:
        """Associate a running subprocess with a job for cancellation.

        Args:
            job_id: Unique job identifier.
            process: The :class:`subprocess.Popen` instance.
        """
        with cls._lock:
            cls._processes[job_id] = process

    @classmethod
    def cancel_job(cls, job_id: str) -> bool:
        """Send SIGTERM to the process group of a running job.

        Falls back to ``proc.kill()`` if the signal cannot be delivered
        and escalates to ``SIGKILL`` when a process ignores the first
        termination attempt.

        Args:
            job_id: Unique job identifier.

        Returns:
            ``True`` if a signal was sent, ``False`` if the process
            was not found or had already exited.
        """
        proc = None
        with cls._lock:
            proc = cls._processes.get(job_id)
        if not proc or proc.poll() is not None:
            return False

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            proc.kill()

        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                proc.kill()
            try:
                proc.wait(timeout=2.0)
            except Exception:
                pass
        return True

    @classmethod
    def active_count(cls) -> int:
        """Return the number of currently running jobs."""
        with cls._lock:
            return len(cls._active_jobs)

    @classmethod
    def is_job_active(cls, job_id: str) -> bool:
        """Check whether a specific job is currently running.

        Args:
            job_id: Unique job identifier.

        Returns:
            ``True`` if the job is in the active set.
        """
        with cls._lock:
            return job_id in cls._active_jobs

    @classmethod
    def queue_size(cls) -> int:
        """Return the approximate number of jobs waiting in the queue."""
        return cls._queue.qsize()


class JobManager:
    """Background manager that dequeues and executes jobs.

    Args:
        config: Application :class:`~api.config.Config` instance.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the instance.

        Args:
            config: Configuration payload.

        """
        self.config = config
        self.max_jobs: int = config.get("max_concurrent_jobs", 3)

    def run(self) -> None:
        """Start the blocking manager loop.

        Continuously polls :class:`JobQueue` and spawns worker threads
        up to ``max_concurrent_jobs``.  This method never returns under
        normal operation.
        """
        logger.info("JobManager started")
        while True:
            if JobQueue.active_count() < self.max_jobs:
                try:
                    job = JobQueue.get_next_job()
                    db_row = get_job_status(str(job.get("job_id") or ""))
                    if db_row and str(db_row.get("status") or "").lower() not in {"queued", "running"}:
                        continue
                    JobQueue.mark_job_start(job)
                    threading.Thread(
                        target=self._run_job_thread, args=(job,), daemon=True
                    ).start()
                except Exception as e:
                    logger.error(f"Error getting job from queue: {e}")
            time.sleep(1)

    def _run_job_thread(self, job: dict) -> None:
        """Worker thread: register, execute, and finalize a job.

        On success the job is marked ``done`` and a callback is sent.
        On failure the error is recorded; cancelled jobs are logged
        separately without triggering a callback.

        Args:
            job: Job payload dict.
        """
        job_id = job["job_id"]
        user_id = job["user_id"]
        source_app = job.get("source_app")
        description = job.get("description")
        job_type = job.get("job_type")
        run_id = str(job.get("run_id") or "")
        try:
            self._prepare_faster_run_job(job)
            run_id = str(job.get("run_id") or "")

            register_job(job_id, user_id, source_app, description, job_type, run_id=run_id or None)
            if run_id:
                update_run_status(run_id, "running", None)
            logger.info(f"Running job: {job_id} (app: {source_app})")
            self._execute_job(job)
            complete_job(job_id, success=True)
            if run_id:
                update_run_status(run_id, "completed", None)
            logger.info(f"Job completed: {job_id}")
            notify_callback(job, success=True)
        except Exception as e:
            error_msg = str(e)
            if "cancelled" in error_msg.lower():
                logger.info(f"Job cancelled: {job_id}")
                complete_job(job_id, success=False, error_msg="Cancelled by user")
                if run_id:
                    update_run_status(run_id, "stopped", "Cancelled by user")
            else:
                logger.error(f"Error running job {job_id}: {e}")
                complete_job(job_id, success=False, error_msg=error_msg)
                if run_id:
                    update_run_status(run_id, "failed", error_msg[:1000])
                notify_callback(job, success=False, error=error_msg)
        finally:
            JobQueue.mark_job_done(job)

    def _gpu_samples_from_nvidia_smi(self) -> list[dict]:
        """Collect GPU usage samples from ``nvidia-smi``.

        Returns:
            A list of GPU sample dictionaries, or an empty list when
            ``nvidia-smi`` is unavailable or returns an unexpected payload.
        """
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return []

        if res.returncode != 0:
            return []

        out: list[dict] = []
        for line in (res.stdout or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpu_id, name, temp, load, mem_used, mem_total = parts[:6]
            out.append(
                {
                    "id": int(gpu_id) if str(gpu_id).isdigit() else gpu_id,
                    "name": name,
                    "temperature": safe_float(temp),
                    "load": safe_float(load),
                    "memory_used_mb": safe_float(mem_used),
                    "memory_total_mb": safe_float(mem_total),
                }
            )
        return out

    def _collect_hardware_sample(self) -> dict:
        """Collect one CPU, RAM, and optional GPU usage sample.

        Returns:
            A resource sample dictionary compatible with ``resource_usage.json``.
        """
        vm = psutil.virtual_memory()
        gpus: list[dict] = []

        if GPUtil is not None:
            try:
                for g in GPUtil.getGPUs():
                    gpus.append(
                        {
                            "id": g.id,
                            "name": g.name,
                            "temperature": safe_float(g.temperature),
                            "load": round(float(g.load) * 100.0, 1),
                            "memory_used_mb": safe_float(g.memoryUsed),
                            "memory_total_mb": safe_float(g.memoryTotal),
                        }
                    )
            except Exception:
                gpus = []

        if not gpus:
            gpus = self._gpu_samples_from_nvidia_smi()

        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cpu_percent": round(psutil.cpu_percent(interval=0.1), 1),
            "ram_used_gb": round(vm.used / (1024 ** 3), 2),
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "ram_percent": round(vm.percent, 1),
            "gpus": gpus,
        }

    def _append_run_resource_sample(self, run_id: str, sample: dict) -> None:
        """Append a hardware sample to a Faster run's resource usage file.

        Args:
            run_id: Run identifier.
            sample: Hardware usage sample collected while the worker runs.
        """
        if not run_id or not isinstance(sample, dict):
            return
        run_dir = os.path.join("/app/results", run_id)
        if not os.path.isdir(run_dir):
            return

        path = os.path.join(run_dir, "resource_usage.json")
        append_resource_sample_file(path, sample, MAX_RESOURCE_SAMPLES)

    def _start_resource_sampler(self, run_id: str, stop_event: threading.Event) -> threading.Thread:
        """Start a daemon thread that samples resource usage until stopped.

        Args:
            run_id: Run identifier.
            stop_event: Event used to stop the sampler loop.

        Returns:
            The started sampler thread.
        """
        def _loop() -> None:
            while not stop_event.is_set():
                try:
                    self._append_run_resource_sample(run_id, self._collect_hardware_sample())
                except Exception:
                    pass
                stop_event.wait(RESOURCE_SAMPLE_INTERVAL_S)

        th = threading.Thread(target=_loop, daemon=True)
        th.start()
        return th

    def _build_faster_worker_config_snapshot(
        self,
        cfg: dict,
        *,
        run_id: str,
        run_dir: str,
        job: dict,
    ) -> dict:
        """Build the worker-facing Faster config snapshot.

        Args:
            cfg: Configuration payload submitted by Faster.
            run_id: Run identifier selected by Faster or derived from a direct
                Job Manager submission.
            run_dir: Execution directory visible to the Job Manager worker.
            job: Job payload used only to fill missing direct-submission owner
                fields.

        Returns:
            A config dictionary suitable for ``main.py`` execution.

        Notes:
            Faster owns product-level decisions such as display name, dataset,
            method, custom model metadata, and custom dataset path attachment.
            Job Manager only creates an execution snapshot for the worker and
            adjusts the execution-only ``save_path`` to the container run
            directory. Missing ``run_name``/``owner`` values are filled for
            direct submissions but existing values are not
            overwritten.
        """
        worker_cfg = dict(cfg)
        worker_cfg.setdefault("run_name", run_id)
        worker_cfg.setdefault("owner", str(job.get("user_id") or ""))
        worker_cfg["save_path"] = run_dir
        return worker_cfg

    def _prepare_faster_run_job(self, job: dict) -> None:
        """Write the worker snapshot for a Faster-submitted run.

        Args:
            job: Queued job payload containing the uploaded config.

        Side Effects:
            Creates ``/app/results/<run_id>/config.yaml`` for ``main.py``,
            points the job input/output paths at that worker snapshot, and
            ensures direct Job Manager submissions have a DB run row without
            overwriting Faster-owned metadata.
        """
        if str(job.get("source_app") or "") != "faster":
            return

        input_path = str(job.get("input_path") or "")
        if not input_path or not os.path.isfile(input_path):
            return

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if not isinstance(cfg, dict):
                return
        except Exception:
            return

        normalize_run_config_aliases(cfg)

        run_id = str(job.get("run_id") or cfg.get("run_name") or "").strip()
        if not run_id:
            run_id = f"run_{job.get('job_id')}"

        run_dir = os.path.join("/app/results", run_id)
        os.makedirs(run_dir, exist_ok=True)

        worker_cfg = self._build_faster_worker_config_snapshot(
            cfg,
            run_id=run_id,
            run_dir=run_dir,
            job=job,
        )

        cfg_path = os.path.join(run_dir, "config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(worker_cfg, f, sort_keys=False)

        job["run_id"] = run_id
        job["run_output_dir"] = run_dir
        job["input_path"] = cfg_path

        upsert_run_metadata(
            run_id=run_id,
            owner_identifier=str(job.get("user_id") or ""),
            display_name=str(worker_cfg.get("display_run_name") or run_id),
            method=worker_cfg.get("method"),
            dataset_name=_dataset_display_name_from_cfg(worker_cfg),
            evaluation_split_mode=worker_cfg.get("evaluation_split_mode"),
            model_name=worker_cfg.get("custom_model_name") or worker_cfg.get("model"),
            config_json=worker_cfg,
        )

    def _execute_job(self, job: dict) -> None:
        """Run the configured shell command for a job.

        If ``job_command`` is not set in the configuration, the job
        completes immediately as a no-op (an empty ``output/`` directory
        is created).

        The command receives the following environment variables:
        ``JOB_ID``, ``JOB_DIR``, ``JOB_INPUT_PATH``, ``JOB_USER_ID``,
        ``JOB_TYPE``, ``JOB_SOURCE_APP``, ``JOB_OUTPUT_DIR``.

        Args:
            job: Job payload dict with ``job_dir`` and ``input_path``.

        Raises:
            RuntimeError: If the command exits with a non-zero code
                or exceeds ``job_timeout_seconds``.
        """
        command_template = self.config.get("job_command")
        source_app = str(job.get("source_app") or "")

        if not command_template and source_app == "faster":
            command_template = "python3 -u /app/main.py --config \"$JOB_INPUT_PATH\" --save_path \"$JOB_OUTPUT_DIR\""

        if not command_template:
            output_dir = os.path.join(job.get("job_dir", ""), "output")
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"No job_command configured, job {job['job_id']} completed as no-op")
            return

        job_dir = job.get("job_dir", "")
        if source_app == "faster":
            output_dir = str(job.get("run_output_dir") or job_dir)
        else:
            output_dir = os.path.join(job_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        env = os.environ.copy()
        env.update({
            "JOB_ID": job["job_id"],
            "JOB_DIR": job_dir,
            "JOB_INPUT_PATH": job.get("input_path", ""),
            "JOB_USER_ID": job.get("user_id", ""),
            "JOB_TYPE": job.get("job_type", ""),
            "JOB_SOURCE_APP": job.get("source_app", ""),
            "JOB_OUTPUT_DIR": output_dir,
        })

        proc = subprocess.Popen(
            command_template,
            shell=True,
            cwd=job_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        JobQueue.register_process(job["job_id"], proc)

        run_id = str(job.get("run_id") or "")
        sample_stop = threading.Event()
        sample_thread = None
        if run_id and source_app == "faster":
            sample_thread = self._start_resource_sampler(run_id, sample_stop)
            
        #timeout updated from 3600 to 36000 to be more conservative 10h
        timeout = self.config.get("job_timeout_seconds", 36000)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait()
            raise RuntimeError(f"Job timed out after {timeout}s")
        finally:
            sample_stop.set()
            if sample_thread is not None:
                sample_thread.join(timeout=1.0)

        if proc.returncode != 0:
            error_msg = stderr.strip() or f"Command exited with code {proc.returncode}"
            if proc.returncode < 0:
                raise RuntimeError(f"Job cancelled (signal {-proc.returncode})")
            raise RuntimeError(error_msg)

        if run_id and source_app == "faster":
            try:
                self._append_run_resource_sample(run_id, self._collect_hardware_sample())
            except Exception:
                pass

        if stdout and stdout.strip():
            logger.info(f"[{job['job_id']}] stdout: {stdout.strip()[:500]}")
