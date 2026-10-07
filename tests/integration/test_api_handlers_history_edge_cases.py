"""Integration tests for history and run API edge cases."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from tests.helpers import reload_module


class ApiHandlerHistoryEdgeCaseTests(unittest.TestCase):
    """Cover frontend-visible history and run API edge cases."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api_handlers = reload_module("web_backend.api_handlers.api")

    def _protected_claims(self, role: str = "user") -> tuple[bool, dict]:
        """Return one authenticated claims payload for API handler tests."""
        return True, {
            "identifier": "alice",
            "role": role,
            "email": "alice@example.com",
        }

    def test_get_runs_clamps_invalid_pagination_inputs_to_safe_defaults(self) -> None:
        """Ensure malformed history pagination input degrades to safe defaults."""
        payload = json.dumps(
            {
                "command": "get_runs",
                "token": "good",
                "data": {
                    "page": "oops",
                    "page_size": "5000",
                    "sort_key": "name",
                    "sort_dir": "asc",
                },
            }
        )

        run_rows = [
            {
                "run_id": "run-1",
                "display_name": "alpha",
                "owner": "alice",
                "status": "completed",
                "method": "fed_avg",
                "dataset_name": "pathmnist",
                "evaluation_split_mode": "train_val_test",
                "model_name": "DefaultNet",
                "stop_reason": "",
                "best_accuracy": 0.9,
                "created_at": "2026-04-03T10:00:00",
            },
            {
                "run_id": "run-2",
                "display_name": "bravo",
                "owner": "alice",
                "status": "completed",
                "method": "fed_prox",
                "dataset_name": "pathmnist",
                "evaluation_split_mode": "train_val_test",
                "model_name": "DefaultNet",
                "stop_reason": "",
                "best_accuracy": 0.8,
                "created_at": "2026-04-02T10:00:00",
            },
        ]

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_run_metadata",
            return_value=run_rows,
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual(["alpha", "bravo"], [item["name"] for item in response["runs"]])
        self.assertEqual(
            {"page": 1, "page_size": 100, "pages": 1, "total": 2},
            response["pagination"],
        )

    def test_get_runs_owner_search_does_not_match_hidden_owner_for_regular_user(self) -> None:
        """Ensure owner-only search does not leak hidden owner matches to regular users."""
        payload = json.dumps(
            {
                "command": "get_runs",
                "token": "good",
                "data": {"search": "alice", "search_field": "owner"},
            }
        )

        run_rows = [
            {
                "run_id": "run-1",
                "display_name": "experiment-one",
                "owner": "alice",
                "status": "completed",
                "method": "fed_avg",
                "dataset_name": "pathmnist",
                "evaluation_split_mode": "train_val_test",
                "model_name": "DefaultNet",
                "stop_reason": "",
                "best_accuracy": 0.9,
                "created_at": "2026-04-03T10:00:00",
            }
        ]

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(role="user"),
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_run_metadata",
            return_value=run_rows,
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual([], response["runs"])

    def test_get_logs_invalid_n_falls_back_to_default_tail_size(self) -> None:
        """Ensure malformed log-tail sizes fall back to the default window."""
        payload = json.dumps(
            {
                "command": "get_logs",
                "token": "good",
                "data": {"run_id": "run-alpha", "n": "not-a-number"},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers,
            "_is_user_run_visible",
            return_value=True,
        ), mock.patch.object(
            self.api_handlers.svc,
            "read_logs",
            return_value="line1\n",
        ) as read_logs:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"success": True, "logs": "line1\n"}, response)
        read_logs.assert_called_once_with(self.api_handlers._run_path("run-alpha"), 100)

    def test_get_runs_by_ids_rejects_non_list_payloads(self) -> None:
        """Ensure get_runs_by_ids rejects malformed non-list id payloads."""
        payload = json.dumps(
            {
                "command": "get_runs_by_ids",
                "token": "good",
                "data": {"ids": "run-1"},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "ids must be a list"}, response)

    def test_get_runs_skips_malformed_run_metadata_rows(self) -> None:
        """Ensure get_runs tolerates malformed metadata rows from the service layer."""
        payload = json.dumps(
            {
                "command": "get_runs",
                "token": "good",
                "data": {},
            }
        )

        run_rows = [
            None,
            {
                "run_id": "run-1",
                "display_name": "alpha",
                "owner": "alice",
                "status": "completed",
                "method": "fed_avg",
                "dataset_name": "pathmnist",
                "evaluation_split_mode": "train_val_test",
                "model_name": "DefaultNet",
                "stop_reason": "",
                "best_accuracy": 0.9,
                "created_at": "2026-04-03T10:00:00",
            },
        ]

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_run_metadata",
            return_value=run_rows,
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual(["alpha"], [item["name"] for item in response["runs"]])
        self.assertEqual(1, len(response["runs"]))

    def test_stop_run_skips_malformed_job_rows_before_canceling(self) -> None:
        """Ensure stop_run tolerates malformed Job Manager rows before cancel logic."""
        payload = json.dumps(
            {
                "command": "stop_run",
                "token": "good",
                "data": {"run_id": "run-123"},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "user_can_access_run",
            return_value=True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "list_jobs_for_user",
            return_value=(
                True,
                [
                    None,
                    {"run_id": "run-123", "job_id": "job-123", "status": "running"},
                ],
            ),
        ), mock.patch.object(
            self.api_handlers._jm,
            "cancel_job",
            return_value=(True, {"message": "Cancel signal sent"}),
        ), mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"success": True, "message": "Cancel signal sent"}, response)
        mark_run_status.assert_called_once_with("run-123", "stopped", "Stopped by user")

    def test_stop_run_can_cancel_job_id_only_active_rows(self) -> None:
        """Ensure stop_run can stop a running job even when the identifier is job_id-only."""
        payload = json.dumps(
            {
                "command": "stop_run",
                "token": "good",
                "data": {"job_id": "job-123"},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "list_jobs_for_user",
            return_value=(
                True,
                [
                    {"job_id": "job-123", "status": "running", "user_id": "alice"},
                ],
            ),
        ), mock.patch.object(
            self.api_handlers._jm,
            "cancel_job",
            return_value=(True, {"message": "Cancel signal sent"}),
        ), mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"success": True, "message": "Cancel signal sent"}, response)
        mark_run_status.assert_not_called()

    def test_stop_run_treats_unavailable_snapshot_as_stale_job(self) -> None:
        """Ensure stop_run handles an unavailable Job Manager snapshot as stale state."""
        payload = json.dumps(
            {
                "command": "stop_run",
                "token": "good",
                "data": {"run_id": "run-123", "job_id": "job-123"},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers,
            "_is_user_run_visible",
            return_value=True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "list_jobs_for_user",
            return_value=(False, []),
        ), mock.patch.object(
            self.api_handlers._jm,
            "delete_job",
        ) as delete_job, mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"success": True, "message": "Stopped (no active job in Job Manager)"}, response)
        delete_job.assert_called_once_with("job-123")
        mark_run_status.assert_called_once_with("run-123", "stopped", "Stopped by user (job missing)")
