"""Integration tests for the bundled Job Manager API contract."""

from __future__ import annotations

import os
import subprocess
import signal
import sys
import tempfile
import types
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.helpers import reload_module


class JobManagerApiTests(unittest.TestCase):
    """Exercise the bundled Job Manager API integration contract."""

    @classmethod
    def setUpClass(cls) -> None:
        pymysql_module = types.ModuleType("pymysql")
        cursors_module = types.ModuleType("pymysql.cursors")
        cursors_module.DictCursor = object
        pymysql_module.cursors = cursors_module

        pooled_db_module = types.ModuleType("dbutils.pooled_db")

        class FakePooledDB:
            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs

            def connection(self):
                conn = mock.Mock()
                conn.cursor.return_value.__enter__ = mock.Mock(return_value=mock.Mock())
                conn.cursor.return_value.__exit__ = mock.Mock(return_value=False)
                return conn

        pooled_db_module.PooledDB = FakePooledDB

        dbutils_module = types.ModuleType("dbutils")
        dbutils_module.pooled_db = pooled_db_module

        requests_module = types.ModuleType("requests")
        requests_module.post = mock.Mock()

        bcrypt_module = types.ModuleType("bcrypt")
        bcrypt_module.hashpw = mock.Mock(return_value=b"hashed")
        bcrypt_module.checkpw = mock.Mock(return_value=True)
        bcrypt_module.gensalt = mock.Mock(return_value=b"salt")

        cls._module_patches = [
            mock.patch.dict(
                sys.modules,
                {
                    "pymysql": pymysql_module,
                    "pymysql.cursors": cursors_module,
                    "dbutils": dbutils_module,
                    "dbutils.pooled_db": pooled_db_module,
                    "requests": requests_module,
                    "bcrypt": bcrypt_module,
                },
            )
        ]
        for patcher in cls._module_patches:
            patcher.start()

        cls.api_module = reload_module("api.api")
        cls.auth_module = reload_module("api.auth")

    @classmethod
    def tearDownClass(cls) -> None:
        for patcher in reversed(getattr(cls, "_module_patches", [])):
            patcher.stop()

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        self.app = FastAPI()
        self.app.include_router(self.api_module.router)
        self.app.dependency_overrides[self.api_module.get_current_user] = lambda: "alice"
        self.client = TestClient(self.app)

        self.job_base_patch = mock.patch.object(
            self.api_module,
            "job_base_path",
            self.tmpdir.name,
        )
        self.job_base_patch.start()
        self.addCleanup(self.job_base_patch.stop)

    def test_auth_login_returns_token_for_valid_credentials(self) -> None:
        """Ensure valid credentials return a JWT-bearing login payload."""
        limiter = mock.Mock()
        limiter.is_allowed.return_value = True

        with mock.patch.object(self.api_module, "login_limiter", limiter), mock.patch.object(
            self.api_module,
            "verify_user",
            return_value=True,
        ), mock.patch.object(
            self.api_module,
            "JWT_SECRET",
            "job-manager-secret-with-safe-length-1234567890",
        ), mock.patch.object(
            self.api_module,
            "JWT_EXPIRATION_HOURS",
            1,
        ):
            response = self.client.post(
                "/auth/login",
                json={"username": "alice", "password": "StrongPass1!"},
            )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("alice", payload["username"])
        self.assertTrue(payload["token"])

    def test_submit_job_stores_uploaded_file_and_queues_job(self) -> None:
        """Ensure job submission stores the uploaded snapshot and queues the job."""
        add_job = mock.Mock()

        with mock.patch.object(
            self.api_module.config,
            "get_app_config",
            return_value={"name": "faster"},
        ), mock.patch.object(
            self.api_module.config,
            "is_ip_allowed",
            return_value=True,
        ), mock.patch.object(
            self.api_module.JobQueue,
            "add_job",
            add_job,
        ):
            response = self.client.post(
                "/submit-job",
                data={
                    "user_id": "alice",
                    "source_app": "faster",
                    "description": "Demo run",
                    "job_type": "faster_run",
                    "run_id": "run-123",
                },
                files={"file": ("run_config.yaml", b"method: fed_avg\n", "application/x-yaml")},
            )

        self.assertEqual(202, response.status_code)
        payload = response.json()
        self.assertEqual("queued", payload["status"])
        add_job.assert_called_once()

        queued_payload = add_job.call_args.args[0]
        self.assertEqual("alice", queued_payload["user_id"])
        self.assertEqual("run-123", queued_payload["run_id"])
        self.assertTrue(os.path.isfile(queued_payload["input_path"]))

    def test_faster_worker_snapshot_preserves_product_fields(self) -> None:
        """Ensure worker snapshots preserve Faster-owned run fields."""
        manager_module = reload_module("api.job_manager")
        manager = manager_module.JobManager(mock.Mock(get=mock.Mock(return_value=1)))

        snapshot = manager._build_faster_worker_config_snapshot(
            {
                "run_name": "primary-run",
                "owner": "alice",
                "display_run_name": "Experiment A",
                "dataset_name": "custom_csv",
                "custom_dataset_name": "Uploaded Dataset",
                "custom_dataset_path": "/app/data/custom_datasets/alice/data.csv",
                "method": "fed_avg",
                "save_path": "/app/results/primary-run",
            },
            run_id="primary-run",
            run_dir="/app/results/primary-run",
            job={"user_id": "job-manager-service"},
        )

        self.assertEqual("primary-run", snapshot["run_name"])
        self.assertEqual("alice", snapshot["owner"])
        self.assertEqual("Experiment A", snapshot["display_run_name"])
        self.assertEqual("custom_csv", snapshot["dataset_name"])
        self.assertEqual("Uploaded Dataset", snapshot["custom_dataset_name"])
        self.assertEqual(
            "/app/data/custom_datasets/alice/data.csv",
            snapshot["custom_dataset_path"],
        )
        self.assertEqual("/app/results/primary-run", snapshot["save_path"])

    def test_list_jobs_scopes_regular_user_to_own_jobs(self) -> None:
        """Ensure non-admin users only see their own jobs in list responses."""
        jobs = [{"job_id": "job-1", "user_id": "alice", "status": "queued"}]

        with mock.patch.object(self.api_module, "is_admin", return_value=False), mock.patch.object(
            self.api_module,
            "get_jobs_by_user",
            return_value=jobs,
        ) as get_jobs_by_user, mock.patch.object(
            self.api_module,
            "get_all_jobs",
            return_value=[],
        ) as get_all_jobs:
            response = self.client.get("/jobs")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(1, payload["count"])
        self.assertEqual("job-1", payload["jobs"][0]["job_id"])
        get_jobs_by_user.assert_called_once()
        get_all_jobs.assert_not_called()

    def test_cancel_job_cancels_running_job_for_owner(self) -> None:
        """Ensure a job owner can cancel a running job through the API."""
        with mock.patch.object(
            self.api_module,
            "get_job_status",
            return_value={"job_id": "job-2", "user_id": "alice", "status": "running"},
        ), mock.patch.object(
            self.api_module,
            "is_admin",
            return_value=False,
        ), mock.patch.object(
            self.api_module.JobQueue,
            "cancel_job",
            return_value=True,
        ) as cancel_job:
            response = self.client.post("/jobs/job-2/cancel")

        self.assertEqual(200, response.status_code)
        self.assertEqual("Cancel signal sent", response.json()["message"])
        cancel_job.assert_called_once_with("job-2")

    def test_delete_job_endpoint_removes_stale_running_job_without_active_process(self) -> None:
        """Ensure stale running jobs can be deleted once the process is gone."""
        record = {"job_id": "job-2", "user_id": "alice", "status": "running"}

        with mock.patch.object(
            self.api_module,
            "get_job_status",
            return_value=record,
        ), mock.patch.object(
            self.api_module,
            "delete_job",
            return_value=record,
        ) as delete_job, mock.patch.object(
            self.api_module.JobQueue,
            "is_job_active",
            return_value=False,
        ), mock.patch.object(
            self.api_module.shutil,
            "rmtree",
        ) as rmtree:
            response = self.client.delete("/jobs/job-2")

        self.assertEqual(200, response.status_code)
        self.assertEqual({"deleted": True, "job_id": "job-2"}, response.json())
        delete_job.assert_called_once_with("job-2")

    def test_cancel_job_escalates_to_sigkill_when_process_survives_sigterm(self) -> None:
        """Ensure cancellation becomes forceful when a process ignores SIGTERM."""
        proc = mock.Mock()
        proc.pid = 4242
        proc.poll.return_value = None
        proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="job-2", timeout=2.0),
            None,
        ]

        with mock.patch.object(
            self.api_module.JobQueue,
            "_processes",
            {"job-2": proc},
        ), mock.patch.object(
            self.api_module.os,
            "getpgid",
            return_value=4242,
        ), mock.patch.object(
            self.api_module.os,
            "killpg",
        ) as killpg:
            result = self.api_module.JobQueue.cancel_job("job-2")

        self.assertTrue(result)
        self.assertEqual(2, killpg.call_count)
        killpg.assert_any_call(4242, signal.SIGTERM)
        killpg.assert_any_call(4242, signal.SIGKILL)
        self.assertEqual(2, proc.wait.call_count)
