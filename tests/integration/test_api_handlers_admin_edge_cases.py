"""Integration tests for admin and runtime-status API edge cases."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from tests.helpers import reload_module


class ApiHandlerAdminEdgeCaseTests(unittest.TestCase):
    """Cover frontend-visible admin and runtime-status edge cases."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api_handlers = reload_module("web_backend.api_handlers.api")

    def _protected_claims(self, role: str = "admin") -> tuple[bool, dict]:
        """Return one authenticated claims payload for API handler tests."""
        return True, {
            "identifier": "alice",
            "role": role,
            "email": "alice@example.com",
        }

    def test_admin_platform_overview_clamps_invalid_paging_inputs(self) -> None:
        """Ensure malformed admin pagination values degrade to safe defaults."""
        payload = json.dumps(
            {
                "command": "admin_platform_overview",
                "token": "good",
                "data": {"jobs_page": "oops", "jobs_page_size": "5000"},
            }
        )

        jobs = [
            {"job_id": "job-1", "run_id": "run-1", "owner": "alice", "kind": "running", "status": "running", "summary": {"method": "fed_avg", "dataset_name": "pathmnist"}},
            {"job_id": "job-2", "run_id": "run-2", "owner": "bob", "kind": "pending", "status": "pending", "summary": {"method": "fed_prox", "dataset_name": "bloodmnist"}},
        ]

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_run_metadata",
            return_value=[],
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_runs",
            return_value=[],
        ), mock.patch.object(
            self.api_handlers._auth,
            "list_users",
            return_value=[{"identifier": "alice"}, {"identifier": "bob"}],
        ), mock.patch.object(
            self.api_handlers,
            "_normalized_jobs_snapshot",
            return_value=jobs,
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual(["job-1", "job-2"], [item["job_id"] for item in response["jobs"]])
        self.assertEqual(
            {"page": 1, "page_size": 100, "pages": 1, "total": 2},
            response["jobs_pagination"],
        )

    def test_admin_platform_overview_filters_jobs_by_owner_search_field(self) -> None:
        """Ensure admin job filtering respects the selected owner search field."""
        payload = json.dumps(
            {
                "command": "admin_platform_overview",
                "token": "good",
                "data": {"search": "bob", "search_field": "owner"},
            }
        )

        jobs = [
            {"job_id": "job-1", "run_id": "run-1", "owner": "alice", "kind": "running", "status": "running", "summary": {"method": "fed_avg", "dataset_name": "pathmnist"}},
            {"job_id": "job-2", "run_id": "run-2", "owner": "bob", "kind": "pending", "status": "pending", "summary": {"method": "fed_prox", "dataset_name": "bloodmnist"}},
        ]

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_run_metadata",
            return_value=[],
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_runs",
            return_value=[],
        ), mock.patch.object(
            self.api_handlers._auth,
            "list_users",
            return_value=[{"identifier": "alice"}, {"identifier": "bob"}],
        ), mock.patch.object(
            self.api_handlers,
            "_normalized_jobs_snapshot",
            return_value=jobs,
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual(["job-2"], [item["job_id"] for item in response["jobs"]])
        self.assertEqual(1, response["totals"]["pending_jobs"])
        self.assertEqual(0, response["totals"]["running_jobs"])

    def test_admin_platform_overview_skips_malformed_job_rows_from_job_manager(self) -> None:
        """Ensure admin overview tolerates malformed Job Manager snapshot rows."""
        payload = json.dumps(
            {
                "command": "admin_platform_overview",
                "token": "good",
                "data": {"search": "", "search_field": "all"},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_run_metadata",
            return_value=[],
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_runs",
            return_value=[],
        ), mock.patch.object(
            self.api_handlers.svc,
            "load_run_config",
            return_value={"method": "fed_avg", "dataset_name": "pathmnist"},
        ), mock.patch.object(
            self.api_handlers._auth,
            "list_users",
            return_value=[{"identifier": "alice"}],
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "list_jobs",
            return_value=(
                True,
                [
                    None,
                    {
                        "job_id": "job-1",
                        "run_id": "run-1",
                        "user_id": "alice",
                        "status": "running",
                        "kind": "running",
                    },
                ],
            ),
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual(["job-1"], [item["job_id"] for item in response["jobs"]])
        self.assertEqual(1, response["totals"]["running_jobs"])

    def test_admin_create_user_rejects_invalid_role(self) -> None:
        """Ensure admin_create_user rejects unsupported roles before hitting auth."""
        payload = json.dumps(
            {
                "command": "admin_create_user",
                "token": "good",
                "data": {
                    "identifier": "bob",
                    "email": "bob@example.com",
                    "password": "StrongPass1!",
                    "role": "superuser",
                },
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers._auth,
            "create_user",
        ) as create_user:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "Invalid role"}, response)
        create_user.assert_not_called()

    def test_system_status_reports_queue_position_for_waiting_user_job(self) -> None:
        """Ensure system_status reports queue position and ahead count for queued jobs."""
        payload = json.dumps(
            {
                "command": "system_status",
                "token": "good",
                "data": {},
            }
        )

        ram = mock.Mock()
        ram.used = 4 * 1024**3
        ram.total = 8 * 1024**3
        ram.percent = 50.0

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(role="user"),
        ), mock.patch.object(
            self.api_handlers,
            "_normalized_jobs_snapshot",
            return_value=[
                {
                    "job_id": "job-123",
                    "run_id": "run-123",
                    "owner": "alice",
                    "status": "queued",
                    "kind": "pending",
                    "position": 3,
                }
            ],
        ), mock.patch.object(
            self.api_handlers.svc,
            "consume_user_alerts",
            return_value=[],
        ), mock.patch(
            "web_backend.api_handlers.admin.psutil.cpu_percent",
            return_value=12.5,
        ), mock.patch(
            "web_backend.api_handlers.admin.psutil.virtual_memory",
            return_value=ram,
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            False,
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual("queued", response["user_job"]["status"])
        self.assertEqual(3, response["user_job"]["queue"]["position"])
        self.assertEqual(2, response["user_job"]["queue"]["ahead"])
