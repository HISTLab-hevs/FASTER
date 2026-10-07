"""Integration tests for the public API handler workflows and failure paths."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from tests.helpers import reload_module, valid_run_config


class ApiHandlerWorkflowTests(unittest.TestCase):
    """Exercise the public API handler workflows and failure paths."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api_handlers = reload_module("web_backend.api_handlers.api")

    def _protected_claims(self, role: str = "user") -> tuple[bool, dict]:
        return True, {
            "identifier": "alice",
            "role": role,
            "email": "alice@example.com",
        }

    def test_login_command_returns_identity_payload(self) -> None:
        """Ensure login returns the authenticated identity payload."""
        payload = json.dumps(
            {
                "command": "login",
                "data": {"identifier": "alice", "password": "StrongPass1!"},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "authenticate",
            return_value=(
                True,
                "jwt-token",
                {
                    "identifier": "alice",
                    "email": "alice@example.com",
                    "role": "user",
                },
            ),
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual("jwt-token", response["token"])
        self.assertEqual("alice", response["username"])
        self.assertFalse(response["is_admin"])

    def test_start_run_returns_queued_job_payload(self) -> None:
        """Ensure start_run returns the queued Job Manager payload."""
        payload = json.dumps(
            {
                "command": "start_run",
                "token": "good",
                "data": {"config": {"run_name": "demo"}},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers,
            "_submit_run_to_job_manager",
            return_value=(True, {"run_id": "run-123", "job_id": "job-123"}),
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual(
            {
                "success": True,
                "run_id": "run-123",
                "job_id": "job-123",
                "queued": True,
                "started": False,
                "message": "Queued in Job Manager",
            },
            response,
        )

    def test_submit_run_returns_error_tuple_when_job_manager_submission_fails(self) -> None:
        """Ensure submission failures return the Job Manager error and mark the run failed."""
        cfg = valid_run_config()

        with mock.patch.object(
            self.api_handlers,
            "_attach_custom_dataset_path",
            return_value=(True, ""),
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers.svc,
            "reserve_run_submission",
            return_value=(True, "", "run-123", None, dict(cfg)),
        ), mock.patch.object(
            self.api_handlers._jm,
            "submit_config_job",
            return_value=(False, {"error": "JM unavailable"}),
        ), mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            ok, payload = self.api_handlers._submit_run_to_job_manager(cfg, "alice")

        self.assertFalse(ok)
        self.assertEqual({"error": "JM unavailable"}, payload)
        mark_run_status.assert_called_once_with("run-123", "failed", "Job Manager submission failed")

    def test_get_logs_returns_tail_for_visible_run(self) -> None:
        """Ensure visible runs can fetch a tailed log payload."""
        payload = json.dumps(
            {
                "command": "get_logs",
                "token": "good",
                "data": {"run_id": "run_alpha", "n": 25},
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
            return_value="line1\nline2\n",
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertIn("line1", response["logs"])

    def test_get_defaults_returns_service_defaults_payload(self) -> None:
        """Ensure get_defaults forwards the service defaults payload."""
        payload = json.dumps(
            {
                "command": "get_defaults",
                "token": "good",
                "data": {},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "load_defaults",
            return_value={"learning_rate": 0.002},
        ) as load_defaults:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual(
            {"success": True, "defaults": {"learning_rate": 0.002}},
            response,
        )
        load_defaults.assert_called_once_with(username="alice")

    def test_save_defaults_persists_using_authenticated_user(self) -> None:
        """Ensure save_defaults persists using the authenticated user context."""
        payload = json.dumps(
            {
                "command": "save_defaults",
                "token": "good",
                "data": {"config": {"learning_rate": 0.002}},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "save_defaults",
            return_value=True,
        ) as save_defaults:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual(
            {"success": True, "message": "Defaults saved"},
            response,
        )
        save_defaults.assert_called_once_with(
            {"learning_rate": 0.002},
            username="alice",
            updated_by="alice",
        )

    def test_stop_run_cancels_via_job_manager_only(self) -> None:
        """Ensure stop_run cancels through Job Manager and updates status accordingly."""
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
            return_value=(True, [{"run_id": "run-123", "job_id": "job-123", "status": "running"}]),
        ), mock.patch.object(
            self.api_handlers._jm,
            "cancel_job",
            return_value=(True, {"message": "Cancel signal sent"}),
        ), mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual(
            {"success": True, "message": "Cancel signal sent"},
            response,
        )
        mark_run_status.assert_called_once_with("run-123", "stopped", "Stopped by user")

    def test_stop_run_marks_stopped_when_job_missing(self) -> None:
        """Ensure stop_run treats missing Job Manager job as stale and marks stopped."""
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
            return_value=(True, [{"run_id": "other-run", "status": "running"}]),
        ), mock.patch.object(
            self.api_handlers._jm,
            "delete_job",
        ) as delete_job, mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual(
            {"success": True, "message": "Stopped (no active job in Job Manager)"},
            response,
        )
        mark_run_status.assert_called_once_with("run-123", "stopped", "Stopped by user (job missing)")
        delete_job.assert_called_once_with("run-123")

    def test_stop_run_does_not_mark_status_when_only_job_id_is_available(self) -> None:
        """Ensure stop_run never falls back to job_id when marking run status."""
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
            return_value=(True, [{"run_id": "other-run", "status": "running"}]),
        ), mock.patch.object(
            self.api_handlers._jm,
            "delete_job",
        ) as delete_job, mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"success": True, "message": "Stopped (no active job in Job Manager)"}, response)
        mark_run_status.assert_not_called()
        delete_job.assert_called_once_with("job-123")

    def test_delete_stale_run_after_missing_job_allows_delete(self) -> None:
        """Ensure stale run (missing Job Manager job) can be stopped and deleted."""
        stop_payload = json.dumps(
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
            return_value=(True, [{"run_id": "other-run", "status": "running"}]),
        ), mock.patch.object(
            self.api_handlers._jm,
            "delete_job",
        ) as delete_job, mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            stop_resp = self.api_handlers.fl_api_handler(stop_payload)

        self.assertEqual({"success": True, "message": "Stopped (no active job in Job Manager)"}, stop_resp)
        mark_run_status.assert_called_once_with("run-123", "stopped", "Stopped by user (job missing)")
        delete_job.assert_called_once_with("run-123")

        delete_payload = json.dumps(
            {
                "command": "delete_run",
                "token": "good",
                "data": {"run_id": "run-123"},
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
            "delete_run",
            return_value=True,
        ) as delete_run:
            del_resp = self.api_handlers.fl_api_handler(delete_payload)

        self.assertEqual({"success": True, "message": "Deleted 'run-123'"}, del_resp)
        delete_run.assert_called_once_with("run-123")

    def test_stop_run_cleans_stale_job_when_cancel_reports_missing_process(self) -> None:
        """Ensure stop_run deletes stale Job Manager rows when the process is gone."""
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
            return_value=(True, [{"run_id": "run-123", "job_id": "job-123", "status": "running"}]),
        ), mock.patch.object(
            self.api_handlers._jm,
            "cancel_job",
            return_value=(False, {"error": "Process not found"}),
        ), mock.patch.object(
            self.api_handlers._jm,
            "delete_job",
        ) as delete_job, mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"success": True, "message": "Stopped"}, response)
        mark_run_status.assert_called_once_with("run-123", "stopped", "Stopped by user")
        delete_job.assert_called_once_with("job-123")

    def test_reconcile_running_runs_skips_terminal_fallback_when_job_manager_unavailable(self) -> None:
        """Ensure reconciliation skips terminal fallback when Job Manager is unavailable."""
        with mock.patch.object(
            self.api_handlers.svc,
            "list_runs",
            return_value=["run-123"],
        ), mock.patch.object(
            self.api_handlers.svc,
            "get_run_status",
            return_value="running",
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "list_jobs_for_user",
            return_value=(False, []),
        ), mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            report = self.api_handlers._reconcile_running_runs(
                "alice",
                is_admin=False,
                include_all=False,
            )

        self.assertEqual(1, report["checked"])
        self.assertEqual(0, report["fixed"])
        self.assertEqual(1, report["skipped"])
        self.assertEqual(1, report["incomplete"])
        self.assertEqual(0, report["confirmed_terminal"])
        self.assertEqual(0, report["no_evidence"])
        self.assertEqual(0, report["stale_or_partial"])
        mark_run_status.assert_not_called()

    def test_reconcile_running_runs_does_not_stop_run_missing_from_job_manager_snapshot(self) -> None:
        """Ensure reconciliation does not stop runs missing from an untrusted snapshot."""
        with mock.patch.object(
            self.api_handlers.svc,
            "list_runs",
            return_value=["run-123"],
        ), mock.patch.object(
            self.api_handlers.svc,
            "get_run_status",
            return_value="running",
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "list_jobs_for_user",
            return_value=(True, [{"run_id": "other-run", "status": "running"}]),
        ), mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            report = self.api_handlers._reconcile_running_runs(
                "alice",
                is_admin=False,
                include_all=False,
            )

        self.assertEqual(1, report["checked"])
        self.assertEqual(0, report["fixed"])
        self.assertEqual(1, report["skipped"])
        self.assertEqual(1, report["no_evidence"])
        self.assertEqual(0, report["incomplete"])
        self.assertEqual(0, report["stale_or_partial"])
        mark_run_status.assert_not_called()

    def test_reconcile_running_runs_treats_partial_job_manager_rows_as_untrusted(self) -> None:
        """Ensure reconciliation treats partial Job Manager rows as untrusted evidence."""
        with mock.patch.object(
            self.api_handlers.svc,
            "list_runs",
            return_value=["run-123"],
        ), mock.patch.object(
            self.api_handlers.svc,
            "get_run_status",
            return_value="running",
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "list_jobs_for_user",
            return_value=(True, [{"status": "running"}]),
        ), mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            report = self.api_handlers._reconcile_running_runs(
                "alice",
                is_admin=False,
                include_all=False,
            )

        self.assertEqual(1, report["checked"])
        self.assertEqual(0, report["fixed"])
        self.assertEqual(1, report["skipped"])
        self.assertEqual(0, report["no_evidence"])
        self.assertGreaterEqual(report["stale_or_partial"], 1)
        mark_run_status.assert_not_called()

    def test_reconcile_running_runs_applies_confirmed_terminal_state_from_job_manager(self) -> None:
        """Ensure reconciliation applies confirmed terminal states from Job Manager."""
        with mock.patch.object(
            self.api_handlers.svc,
            "list_runs",
            return_value=["run-123"],
        ), mock.patch.object(
            self.api_handlers.svc,
            "get_run_status",
            return_value="running",
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "list_jobs_for_user",
            return_value=(True, [{"run_id": "run-123", "status": "failed"}]),
        ), mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            report = self.api_handlers._reconcile_running_runs(
                "alice",
                is_admin=False,
                include_all=False,
            )

        self.assertEqual(1, report["checked"])
        self.assertEqual(1, report["fixed"])
        self.assertEqual(1, report["confirmed_terminal"])
        self.assertEqual(0, report["skipped"])
        mark_run_status.assert_called_once_with(
            "run-123",
            "failed",
            "Recovered by refresh: failed in Job Manager",
        )

    def test_heal_best_accuracy_logs_and_preserves_refresh_on_metrics_failure(self) -> None:
        """Ensure best-accuracy backfill logs metrics failures without mutating state."""
        with mock.patch.object(
            self.api_handlers.svc,
            "load_metrics",
            side_effect=RuntimeError("metrics broken"),
        ), mock.patch.object(
            self.api_handlers.svc,
            "mark_run_best_accuracy",
        ) as mark_run_best_accuracy, self.assertLogs(
            "web_backend.api_handlers",
            level="WARNING",
        ) as captured:
            self.api_handlers._heal_best_accuracy("run-123")

        self.assertTrue(
            any("Failed to backfill best_accuracy for run 'run-123'" in message for message in captured.output)
        )
        mark_run_best_accuracy.assert_not_called()

    def test_get_runs_refresh_reports_incomplete_reconciliation_without_mutating_run(self) -> None:
        """Ensure get_runs reports incomplete reconciliation without mutating the run."""
        payload = json.dumps(
            {
                "command": "get_runs",
                "token": "good",
                "data": {"refresh_db": True},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_runs",
            return_value=["run-123"],
        ), mock.patch.object(
            self.api_handlers.svc,
            "get_run_status",
            return_value="running",
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_run_metadata",
            return_value=[
                {
                    "run_id": "run-123",
                    "display_name": "run-123",
                    "owner": "alice",
                    "status": "running",
                    "method": "fed_avg",
                    "dataset_name": "pathmnist",
                    "evaluation_split_mode": "train_val_test",
                    "model_name": "DefaultNet",
                    "stop_reason": "",
                    "best_accuracy": None,
                    "created_at": "",
                }
            ],
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "list_jobs_for_user",
            return_value=(False, []),
        ), mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual("running", response["runs"][0]["status"])
        self.assertEqual(1, response["reconcile"]["incomplete"])
        self.assertEqual(0, response["reconcile"]["fixed"])
        mark_run_status.assert_not_called()

    def test_get_runs_returns_queued_run_in_normal_listing(self) -> None:
        """Ensure queued runs appear in the normal runs listing."""
        payload = json.dumps(
            {
                "command": "get_runs",
                "token": "good",
                "data": {},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_run_metadata",
            return_value=[
                {
                    "run_id": "run-123",
                    "display_name": "queued-demo",
                    "owner": "alice",
                    "status": "queued",
                    "method": "fed_avg",
                    "dataset_name": "pathmnist",
                    "evaluation_split_mode": "train_val_test",
                    "model_name": "DefaultNet",
                    "stop_reason": "Queued in Job Manager",
                    "best_accuracy": None,
                    "created_at": "",
                }
            ],
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual(1, len(response["runs"]))
        self.assertEqual("run-123", response["runs"][0]["id"])
        self.assertEqual("queued", response["runs"][0]["status"])
        self.assertEqual("Queued in Job Manager", response["runs"][0]["stop_reason"])

    def test_normalized_jobs_snapshot_uses_global_fetch_and_masks_other_users(self) -> None:
        """A regular user's snapshot is built from the global queue so positions
        reflect every job ahead of them, but other users' jobs are masked to
        status+position only (no job_id, run_id, or owner)."""
        jobs_payload = [
            {
                "job_id": "job-bob",
                "run_id": "run-bob",
                "user_id": "bob",
                "status": "running",
                "progress_pct": 99,
            },
            {
                "job_id": "job-alice",
                "run_id": "run-alice",
                "user_id": "alice",
                "status": "queued",
                "progress_pct": 0,
            },
        ]

        with mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "list_jobs_for_user",
        ) as list_jobs_for_user, mock.patch.object(
            self.api_handlers._jm,
            "list_jobs",
            return_value=(True, jobs_payload),
        ) as list_jobs, mock.patch.object(
            self.api_handlers.svc,
            "get_run_status",
            return_value="running",
        ), mock.patch.object(
            self.api_handlers.svc,
            "load_run_config",
            return_value={"method": "fed_avg", "dataset_name": "pathmnist"},
        ), mock.patch.object(
            self.api_handlers.svc,
            "dataset_display_name_from_config",
            return_value="pathmnist",
        ):
            snapshot = self.api_handlers._normalized_jobs_snapshot("alice", is_admin=False)

        # Global list is used; the per-user endpoint is not.
        list_jobs.assert_called_once()
        list_jobs_for_user.assert_not_called()

        by_owner = {row["owner"]: row for row in snapshot}
        # Alice's own job keeps its identity.
        self.assertIn("alice", by_owner)
        self.assertEqual("job-alice", by_owner["alice"]["job_id"])
        self.assertTrue(by_owner["alice"]["is_mine"])

        # Bob's running job (ahead of Alice) is shown but fully masked.
        masked = [row for row in snapshot if not row["is_mine"]]
        self.assertEqual(1, len(masked))
        self.assertEqual("", masked[0]["job_id"])
        self.assertEqual("", masked[0]["run_id"])
        self.assertEqual("", masked[0]["owner"])
        self.assertEqual("running", masked[0]["status"])
        self.assertTrue(masked[0]["locked"])
        self.assertNotIn("config", masked[0])

    def test_normalized_jobs_snapshot_ignores_terminal_runs_even_if_job_manager_is_stale(self) -> None:
        """Ensure a stopped run does not reappear as active just because Job Manager is stale."""
        jobs_payload = [
            {
                "job_id": "job-123",
                "run_id": "run-123",
                "user_id": "alice",
                "status": "running",
                "progress_pct": 42,
            }
        ]

        with mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "list_jobs",
            return_value=(True, jobs_payload),
        ), mock.patch.object(
            self.api_handlers.svc,
            "get_run_status",
            return_value="stopped",
        ) as get_run_status:
            snapshot = self.api_handlers._normalized_jobs_snapshot("alice", is_admin=False)

        self.assertEqual([], snapshot)
        get_run_status.assert_called_once_with("run-123")

    def test_reconcile_running_runs_treats_full_cap_admin_snapshot_as_partial(self) -> None:
        """Ensure capped admin job snapshots are treated as partial reconciliation evidence."""
        with mock.patch.object(
            self.api_handlers.svc,
            "list_runs",
            return_value=["run-123"],
        ), mock.patch.object(
            self.api_handlers.svc,
            "get_run_status",
            return_value="running",
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "_JOB_LIST_LIMIT_MAX",
            2,
            create=True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "list_jobs",
            return_value=(
                True,
                [
                    {"run_id": "other-1", "status": "running"},
                    {"run_id": "other-2", "status": "queued"},
                ],
            ),
        ), mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ) as mark_run_status:
            report = self.api_handlers._reconcile_running_runs(
                "alice",
                is_admin=True,
                include_all=True,
            )

        self.assertEqual(1, report["checked"])
        self.assertEqual(0, report["fixed"])
        self.assertEqual(1, report["skipped"])
        self.assertEqual(0, report["no_evidence"])
        self.assertGreaterEqual(report["stale_or_partial"], 1)
        mark_run_status.assert_not_called()

    def test_system_status_reuses_snapshot_config_for_active_run(self) -> None:
        """Ensure system_status reuses snapshot config instead of reloading run config."""
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
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers,
            "_normalized_jobs_snapshot",
            return_value=[
                {
                    "job_id": "job-123",
                    "run_id": "run-123",
                    "owner": "alice",
                    "status": "running",
                    "kind": "running",
                    "config": {
                        "method": "fed_avg",
                        "dataset_name": "pathmnist",
                        "num_clients": 3,
                        "local_model_epochs": 5,
                        "weights_sending_frequency": 1,
                    },
                }
            ],
        ), mock.patch.object(
            self.api_handlers.svc,
            "load_run_config",
            side_effect=AssertionError("load_run_config should not be called when snapshot config is present"),
        ), mock.patch.object(
            self.api_handlers.svc,
            "dataset_display_name_from_config",
            return_value="pathmnist",
        ), mock.patch.object(
            self.api_handlers.svc,
            "load_run_resource_usage",
            return_value={},
        ), mock.patch.object(
            self.api_handlers.svc,
            "append_run_resource_sample",
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
        self.assertEqual("active", response["user_job"]["status"])
        self.assertEqual("run-123", response["user_job"]["active"]["run_id"])

    def test_get_runs_supports_server_side_history_pagination_and_sorting(self) -> None:
        """Ensure get_runs supports server-side history pagination and sorting."""
        payload = json.dumps(
            {
                "command": "get_runs",
                "token": "good",
                "data": {
                    "search": "fed",
                    "search_field": "method",
                    "page": 2,
                    "page_size": 2,
                    "sort_key": "name",
                    "sort_dir": "asc",
                },
            }
        )

        run_rows = [
            {
                "run_id": "run-c",
                "display_name": "charlie",
                "owner": "alice",
                "status": "completed",
                "method": "fed_nova",
                "dataset_name": "pathmnist",
                "evaluation_split_mode": "train_val_test",
                "model_name": "DefaultNet",
                "stop_reason": "",
                "best_accuracy": 0.7,
                "created_at": "2026-04-01T10:00:00",
            },
            {
                "run_id": "run-a",
                "display_name": "alpha",
                "owner": "alice",
                "status": "completed",
                "method": "fed_avg",
                "dataset_name": "pathmnist",
                "evaluation_split_mode": "train_val_test",
                "model_name": "DefaultNet",
                "stop_reason": "",
                "best_accuracy": 0.8,
                "created_at": "2026-04-03T10:00:00",
            },
            {
                "run_id": "run-b",
                "display_name": "bravo",
                "owner": "alice",
                "status": "completed",
                "method": "fed_prox",
                "dataset_name": "pathmnist",
                "evaluation_split_mode": "train_val_test",
                "model_name": "DefaultNet",
                "stop_reason": "",
                "best_accuracy": 0.9,
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
        self.assertEqual(["charlie"], [item["name"] for item in response["runs"]])
        self.assertEqual(
            {"page": 2, "page_size": 2, "pages": 2, "total": 3},
            response["pagination"],
        )

    def test_get_runs_by_ids_returns_requested_visible_rows_in_order(self) -> None:
        """Ensure get_runs_by_ids preserves request order while removing duplicates."""
        payload = json.dumps(
            {
                "command": "get_runs_by_ids",
                "token": "good",
                "data": {"ids": ["run-2", "run-1", "run-2"]},
            }
        )

        def _meta_for(run_id: str):
            return {
                "run_id": run_id,
                "display_name": f"name-{run_id}",
                "owner": "alice",
                "status": "completed",
                "method": "fed_avg",
                "dataset_name": "pathmnist",
                "evaluation_split_mode": "train_val_test",
                "model_name": "DefaultNet",
                "stop_reason": "",
                "best_accuracy": None,
                "created_at": "2026-04-03T10:00:00",
            }

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
            "get_run_metadata",
            side_effect=_meta_for,
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual(["run-2", "run-1"], [item["id"] for item in response["runs"]])

    def test_admin_platform_overview_paginates_jobs_server_side(self) -> None:
        """Ensure the admin overview paginates job snapshots on the server side."""
        payload = json.dumps(
            {
                "command": "admin_platform_overview",
                "token": "good",
                "data": {"jobs_page": 2, "jobs_page_size": 2},
            }
        )

        jobs = [
            {"job_id": "job-1", "run_id": "run-1", "owner": "alice", "kind": "running", "status": "running", "summary": {"method": "fed_avg", "dataset_name": "pathmnist"}},
            {"job_id": "job-2", "run_id": "run-2", "owner": "alice", "kind": "pending", "status": "pending", "summary": {"method": "fed_prox", "dataset_name": "pathmnist"}},
            {"job_id": "job-3", "run_id": "run-3", "owner": "bob", "kind": "running", "status": "running", "summary": {"method": "fed_nova", "dataset_name": "dermamnist"}},
            {"job_id": "job-4", "run_id": "run-4", "owner": "bob", "kind": "pending", "status": "pending", "summary": {"method": "fed_avgw", "dataset_name": "bloodmnist"}},
            {"job_id": "job-5", "run_id": "run-5", "owner": "bob", "kind": "running", "status": "running", "summary": {"method": "fed_gp", "dataset_name": "organamnist"}},
        ]

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(role="admin"),
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_run_metadata",
            return_value=[],
        ) as list_run_metadata, mock.patch.object(
            self.api_handlers.svc,
            "list_runs",
            return_value=["run-1", "run-2", "run-3"],
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
        self.assertEqual(["job-3", "job-4"], [item["job_id"] for item in response["jobs"]])
        self.assertEqual(
            {"page": 2, "page_size": 2, "pages": 3, "total": 5},
            response["jobs_pagination"],
        )
        list_run_metadata.assert_called_once_with("alice", include_all=True, limit=self.api_handlers.ADMIN_ANALYTICS_MAX_RUNS)
        self.assertEqual(3, response["totals"]["running_jobs"])
        self.assertEqual(2, response["totals"]["pending_jobs"])

    def test_validate_run_names_uses_display_name_limit(self) -> None:
        """Ensure validate_run_names uses the display-name limit."""
        payload = json.dumps(
            {
                "command": "validate_run_names",
                "token": "good",
                "data": {"names": ["a" * 20]},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "find_run_name_conflicts",
            return_value=[],
        ) as find_conflicts:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"success": True, "conflicts": []}, response)
        find_conflicts.assert_called_once_with(["a" * 20], username="alice")

    def test_rename_run_accepts_name_up_to_maximum_length(self) -> None:
        """Ensure rename_run accepts names at the maximum length."""
        payload = json.dumps(
            {
                "command": "rename_run",
                "token": "good",
                "data": {"run_id": "run-123", "new_name": "a" * 20},
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
            "rename_run",
            return_value=(True, "Run display name updated"),
        ) as rename_run:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"success": True, "message": "Run display name updated"}, response)
        rename_run.assert_called_once_with("run-123", "a" * 20)

    def test_rename_run_rejects_name_above_maximum_length(self) -> None:
        """Ensure rename_run rejects names above the maximum length."""
        payload = json.dumps(
            {
                "command": "rename_run",
                "token": "good",
                "data": {"run_id": "run-123", "new_name": "a" * 21},
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
            "rename_run",
        ) as rename_run:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "new_name must be at most 20 characters"}, response)
        rename_run.assert_not_called()

    def test_cancel_queued_job_uses_job_manager_without_local_fallback(self) -> None:
        """Ensure queued-job cancellation uses Job Manager without a local fallback path."""
        payload = json.dumps(
            {
                "command": "cancel_queued_job",
                "token": "good",
                "data": {"job_id": "job-123"},
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
            self.api_handlers,
            "_resolve_job_id_for_cancel",
            return_value=("job-123", None),
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            True,
        ), mock.patch.object(
            self.api_handlers._jm,
            "cancel_job",
            return_value=(True, {"message": "Cancel signal sent"}),
        ), mock.patch.object(
            self.api_handlers.svc,
            "mark_run_status",
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual(
            {
                "success": True,
                "message": "Cancel signal sent",
                "job_manager": {"cancelled": True, "job_id": "job-123"},
            },
            response,
        )

    def test_job_manager_diagnostics_reports_local_runtime_disabled(self) -> None:
        """Ensure diagnostics report the local runtime as disabled when JM is off."""
        payload = json.dumps(
            {
                "command": "job_manager_diagnostics",
                "token": "good",
                "data": {},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers._jm,
            "enabled",
            False,
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual(
            {"enabled": False, "running": 0, "pending": 0},
            response["local_runtime"],
        )

    def test_legacy_upload_custom_dataset_command_is_no_longer_supported(self) -> None:
        """Ensure the removed legacy upload command stays unsupported."""
        payload = json.dumps(
            {
                "command": "upload_custom_dataset",
                "token": "good",
                "data": {
                    "filename": "sample.csv",
                    "content": "f1,label,split\n1,0,train\n",
                    "encoding": "text",
                    "description": "demo",
                },
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "Unknown command: 'upload_custom_dataset'"}, response)

    def test_parse_scenario_yaml_masks_parser_exception_details(self) -> None:
        """Ensure scenario YAML parse errors return a stable masked message."""
        payload = json.dumps(
            {
                "command": "parse_scenario_yaml",
                "token": "good",
                "data": {"yaml_text": "run_name: [unterminated"},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "Invalid YAML content"}, response)

    def test_fl_api_handler_masks_unhandled_exception_details(self) -> None:
        """Ensure unhandled exceptions are masked from API responses."""
        payload = json.dumps(
            {
                "command": "get_defaults",
                "token": "good",
                "data": {},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers,
            "handle_content_command",
            side_effect=RuntimeError("database DSN leaked"),
        ), mock.patch.object(
            self.api_handlers.logger,
            "exception",
        ) as log_exception:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "Internal server error"}, response)
        log_exception.assert_called_once()

    def test_save_scenario_returns_stable_error_when_persistence_fails(self) -> None:
        """Ensure save_scenario returns a stable masked error on persistence failure."""
        payload = json.dumps(
            {
                "command": "save_scenario",
                "token": "good",
                "data": {"name": "demo", "config": {"learning_rate": 0.01}},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "save_user_scenario",
            side_effect=RuntimeError("disk full"),
        ), mock.patch.object(
            self.api_handlers.logger,
            "exception",
        ) as log_exception:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "Failed to save scenario"}, response)
        log_exception.assert_called_once()

    def test_admin_create_user_requires_admin(self) -> None:
        """Ensure admin_create_user rejects non-admin callers."""
        payload = json.dumps(
            {
                "command": "admin_create_user",
                "token": "good",
                "data": {
                    "identifier": "bob",
                    "email": "bob@example.com",
                    "password": "StrongPass1!",
                    "role": "user",
                },
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(role="user"),
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "Forbidden"}, response)

    def test_admin_create_user_calls_auth_backend(self) -> None:
        """Ensure admin_create_user delegates to the auth backend for admins."""
        payload = json.dumps(
            {
                "command": "admin_create_user",
                "token": "good",
                "data": {
                    "identifier": "bob",
                    "email": "bob@example.com",
                    "password": "StrongPass1!",
                    "role": "user",
                },
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(role="admin"),
        ), mock.patch.object(
            self.api_handlers._auth,
            "create_user",
            return_value=(
                True,
                "User created",
                {"identifier": "bob", "email": "bob@example.com", "role": "user"},
            ),
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertTrue(response["success"])
        self.assertEqual("bob", response["user"]["identifier"])
