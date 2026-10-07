"""Integration tests for run lifecycle state transitions in the service layer."""

from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from database.models import RunRow
from tests.helpers import reload_module, valid_run_config


class InMemoryRunsRepo:
    def __init__(self) -> None:
        self.rows: dict[str, RunRow] = {}
        self.configs: dict[str, dict] = {}
        self._next_id = 1

    def create_run(self, payload: dict) -> int:
        row_id = self._next_id
        self._next_id += 1
        now = datetime.now(UTC).replace(tzinfo=None)
        self.rows[payload["run_id"]] = RunRow(
            id=row_id,
            run_id=payload["run_id"],
            owner_identifier=payload["owner_identifier"],
            display_name=payload.get("display_name") or payload["run_id"],
            status=payload.get("status", "running"),
            method=payload.get("method"),
            dataset_name=payload.get("dataset_name"),
            evaluation_split_mode=payload.get("evaluation_split_mode"),
            model_name=payload.get("model_name"),
            stop_reason=payload.get("stop_reason"),
            best_accuracy=payload.get("best_accuracy"),
            created_at=now,
            updated_at=now,
        )
        config = payload.get("config_json")
        if config is not None:
            self.configs[payload["run_id"]] = dict(config)
        return row_id

    def get_run(self, run_id: str) -> RunRow | None:
        return self.rows.get(run_id)

    def list_runs(self, owner_identifier: str | None = None) -> list[RunRow]:
        rows = list(self.rows.values())
        if owner_identifier is not None:
            rows = [row for row in rows if row.owner_identifier == owner_identifier]
        rows.sort(key=lambda row: row.created_at, reverse=True)
        return rows

    def update_run(self, run_id: str, updates: dict) -> bool:
        row = self.rows.get(run_id)
        if row is None:
            return False
        data = {
            "id": row.id,
            "run_id": row.run_id,
            "owner_identifier": row.owner_identifier,
            "display_name": row.display_name,
            "status": row.status,
            "method": row.method,
            "dataset_name": row.dataset_name,
            "evaluation_split_mode": row.evaluation_split_mode,
            "model_name": row.model_name,
            "stop_reason": row.stop_reason,
            "best_accuracy": row.best_accuracy,
            "created_at": row.created_at,
            "updated_at": datetime.now(UTC).replace(tzinfo=None),
        }
        data.update(updates)
        self.rows[run_id] = RunRow(**data)
        return True

    def upsert_run_config(self, run_id: str, config_json: dict) -> bool:
        self.configs[run_id] = dict(config_json)
        return True

    def delete_run(self, run_id: str) -> bool:
        existed = run_id in self.rows
        self.rows.pop(run_id, None)
        self.configs.pop(run_id, None)
        return existed


class ExplodingUpdateRunsRepo(InMemoryRunsRepo):
    def update_run(self, run_id: str, updates: dict) -> bool:
        raise RuntimeError("db write failed")


class InspectingDeleteRunsRepo(InMemoryRunsRepo):
    def __init__(self, path_check) -> None:
        super().__init__()
        self._path_check = path_check
        self.saw_missing_dir = False

    def delete_run(self, run_id: str) -> bool:
        self.saw_missing_dir = not self._path_check(run_id)
        return super().delete_run(run_id)


class ServiceRuntimeStateTests(unittest.TestCase):
    """Cover DB-backed run reservation, lifecycle, and deletion behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.service_mod = reload_module("web_backend.service")

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        self.results_dir = Path(self.tmpdir.name) / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results_patch = mock.patch.object(
            self.service_mod,
            "BASE_RESULTS_PATH",
            str(self.results_dir),
        )
        self.results_patch.start()
        self.addCleanup(self.results_patch.stop)

        self.service = self.service_mod.TrainingServiceLayer()
        self.service._db_enabled = False
        self.service._runs_repo = None

    def test_reserve_run_submission_persists_job_manager_queued_state(self) -> None:
        """Ensure reserved submissions persist queued state before execution begins."""
        repo = InMemoryRunsRepo()
        self.service._db_enabled = True
        self.service._runs_repo = repo

        cfg = valid_run_config()
        cfg["run_name"] = "demo_run"

        ok, msg, run_id, run_path, reserved_cfg = self.service.reserve_run_submission(
            cfg,
            "alice",
        )

        self.assertTrue(ok)
        self.assertEqual("Queued in Job Manager", msg)
        self.assertEqual("demo_run", run_id)
        self.assertEqual("alice", reserved_cfg["owner"])
        self.assertEqual(run_id, reserved_cfg["run_name"])
        self.assertEqual(run_path, reserved_cfg["save_path"])

        cfg_path = Path(run_path) / "config.yaml"
        self.assertTrue(cfg_path.is_file())
        self.assertFalse((Path(run_path) / "run_state.json").exists())
        self.assertEqual("queued", self.service.get_run_status(run_id))
        self.assertEqual("alice", self.service.get_run_owner(run_id))
        self.assertEqual(["demo_run"], self.service.find_run_name_conflicts(["demo_run"], username="alice"))

    def test_reserve_run_submission_persists_db_row_for_queued_run(self) -> None:
        """Ensure queued submissions create the DB row immediately."""
        repo = InMemoryRunsRepo()
        self.service._db_enabled = True
        self.service._runs_repo = repo

        cfg = valid_run_config()
        cfg["run_name"] = "queued_visible"

        ok, msg, run_id, _, reserved_cfg = self.service.reserve_run_submission(cfg, "alice")

        self.assertTrue(ok)
        self.assertEqual("Queued in Job Manager", msg)
        row = repo.get_run(run_id)
        self.assertIsNotNone(row)
        self.assertEqual("queued", row.status)
        self.assertEqual("alice", row.owner_identifier)
        self.assertEqual("queued_visible", row.display_name)
        self.assertEqual("fed_avg", row.method)
        self.assertEqual("pathmnist", row.dataset_name)
        self.assertEqual("Queued in Job Manager", row.stop_reason)
        self.assertEqual(reserved_cfg["run_name"], run_id)
        self.assertEqual(run_id, self.service.list_runs(username="alice")[0])

    def test_list_run_metadata_includes_reserved_queued_run_before_worker_artifacts_exist(self) -> None:
        """Ensure queued runs appear in metadata before worker artifacts are created."""
        repo = InMemoryRunsRepo()
        self.service._db_enabled = True
        self.service._runs_repo = repo

        cfg = valid_run_config()
        cfg["run_name"] = "pre_worker_visible"
        ok, _, run_id, run_path, _ = self.service.reserve_run_submission(cfg, "alice")

        self.assertTrue(ok)
        self.assertFalse((Path(run_path) / "train.log").exists())
        rows = self.service.list_run_metadata(username="alice")

        self.assertEqual(1, len(rows))
        self.assertEqual(run_id, rows[0]["run_id"])
        self.assertEqual("queued", rows[0]["status"])
        self.assertEqual("alice", rows[0]["owner"])
        self.assertEqual("fed_avg", rows[0]["method"])

    def test_list_run_metadata_tracks_transition_from_queued_to_running_to_completed(self) -> None:
        """Ensure list metadata tracks lifecycle transitions from queued to completed."""
        repo = InMemoryRunsRepo()
        self.service._db_enabled = True
        self.service._runs_repo = repo

        cfg = valid_run_config()
        cfg["run_name"] = "lifecycle_visible"
        ok, _, run_id, run_path, _ = self.service.reserve_run_submission(cfg, "alice")

        self.assertTrue(ok)
        queued_rows = self.service.list_run_metadata(username="alice")
        self.assertEqual("queued", queued_rows[0]["status"])

        self.service.mark_run_status(run_id, "running", "")
        running_rows = self.service.list_run_metadata(username="alice")
        self.assertEqual("running", running_rows[0]["status"])

        (Path(run_path) / "train.log").write_text("done\n", encoding="utf-8")
        row = repo.rows[run_id]
        repo.rows[run_id] = RunRow(
            id=row.id,
            run_id=row.run_id,
            owner_identifier=row.owner_identifier,
            display_name=row.display_name,
            status="completed",
            method=row.method,
            dataset_name=row.dataset_name,
            evaluation_split_mode=row.evaluation_split_mode,
            model_name=row.model_name,
            stop_reason="",
            best_accuracy=row.best_accuracy,
            created_at=row.created_at,
            updated_at=datetime.now(UTC).replace(tzinfo=None) + timedelta(seconds=1),
        )
        completed_rows = self.service.list_run_metadata(username="alice")
        self.assertEqual("completed", completed_rows[0]["status"])

    def test_mark_run_status_logs_db_write_failure_without_state_file_fallback(self) -> None:
        """Ensure DB write failures are logged without reviving filesystem state fallbacks."""
        repo = ExplodingUpdateRunsRepo()
        self.service._db_enabled = True
        self.service._runs_repo = repo

        cfg = valid_run_config()
        cfg["run_name"] = "warning_demo"
        ok, _, run_id, run_path, _ = self.service.reserve_run_submission(cfg, "alice")

        self.assertTrue(ok)
        with self.assertLogs("web_backend.service", level="WARNING") as captured:
            self.service.mark_run_status(run_id, "running", "")

        self.assertTrue(
            any("Failed to persist run status 'running' to DB" in message for message in captured.output)
        )
        self.assertFalse((Path(run_path) / "run_state.json").exists())

    def test_get_run_status_ignores_stale_filesystem_state(self) -> None:
        """Ensure DB status remains authoritative over stale filesystem state files."""
        repo = InMemoryRunsRepo()
        self.service._db_enabled = True
        self.service._runs_repo = repo

        cfg = valid_run_config()
        cfg["run_name"] = "db_authoritative"
        ok, _, run_id, run_path, _ = self.service.reserve_run_submission(cfg, "alice")

        self.assertTrue(ok)
        (Path(run_path) / "run_state.json").write_text(
            '{"status": "stopped", "reason": "stale", "updated_at": "2026-01-01T00:00:00"}',
            encoding="utf-8",
        )
        self.assertEqual("queued", self.service.get_run_status(run_id))

    def test_list_run_metadata_includes_terminal_db_row_without_history_artifacts(self) -> None:
        """Ensure terminal DB rows remain visible even without history artifacts."""
        repo = InMemoryRunsRepo()
        self.service._db_enabled = True
        self.service._runs_repo = repo

        cfg = valid_run_config()
        cfg["run_name"] = "terminal_without_artifacts"
        ok, _, run_id, _, _ = self.service.reserve_run_submission(cfg, "alice")
        self.assertTrue(ok)

        row = repo.rows[run_id]
        repo.rows[run_id] = RunRow(
            id=row.id,
            run_id=row.run_id,
            owner_identifier=row.owner_identifier,
            display_name=row.display_name,
            status="completed",
            method=row.method,
            dataset_name=row.dataset_name,
            evaluation_split_mode=row.evaluation_split_mode,
            model_name=row.model_name,
            stop_reason="",
            best_accuracy=row.best_accuracy,
            created_at=row.created_at,
            updated_at=datetime.now(UTC).replace(tzinfo=None) + timedelta(seconds=1),
        )

        rows = self.service.list_run_metadata(username="alice")
        self.assertEqual(1, len(rows))
        self.assertEqual("completed", rows[0]["status"])

    def test_list_runs_ignores_filesystem_only_legacy_history(self) -> None:
        """Ensure legacy filesystem-only runs do not reappear without DB metadata."""
        legacy_run_dir = self.results_dir / "legacy_run"
        legacy_run_dir.mkdir(parents=True, exist_ok=True)
        (legacy_run_dir / "train.log").write_text("legacy\n", encoding="utf-8")
        (legacy_run_dir / "config.yaml").write_text(
            "owner: alice\nrun_name: legacy_run\ndataset_name: pathmnist\nmethod: fed_avg\n",
            encoding="utf-8",
        )

        self.assertEqual([], self.service.list_runs(username="alice"))
        self.assertEqual([], self.service.list_run_metadata(username="alice"))

    def test_delete_run_removes_filesystem_before_database_row(self) -> None:
        """Ensure run deletion removes artifacts before deleting the DB row."""
        repo = InspectingDeleteRunsRepo(lambda run_id: (self.results_dir / run_id).exists())
        self.service._db_enabled = True
        self.service._runs_repo = repo

        cfg = valid_run_config()
        cfg["run_name"] = "delete_order_demo"
        ok, _, run_id, run_path, _ = self.service.reserve_run_submission(cfg, "alice")
        self.assertTrue(ok)
        self.assertTrue(Path(run_path).exists())

        deleted = self.service.delete_run(run_id)

        self.assertTrue(deleted)
        self.assertTrue(repo.saw_missing_dir)
        self.assertFalse(Path(run_path).exists())
        self.assertIsNone(repo.get_run(run_id))

    def test_delete_run_keeps_database_row_when_filesystem_cleanup_fails(self) -> None:
        """Ensure failed filesystem cleanup keeps the DB record for a retry."""
        repo = InMemoryRunsRepo()
        self.service._db_enabled = True
        self.service._runs_repo = repo

        cfg = valid_run_config()
        cfg["run_name"] = "delete_failure_demo"
        ok, _, run_id, run_path, _ = self.service.reserve_run_submission(cfg, "alice")
        self.assertTrue(ok)

        with mock.patch("web_backend.services.run_lifecycle.shutil.rmtree", side_effect=OSError("disk busy")):
            with self.assertLogs("web_backend.service", level="WARNING") as captured:
                deleted = self.service.delete_run(run_id)

        self.assertFalse(deleted)
        self.assertTrue(Path(run_path).exists())
        self.assertIsNotNone(repo.get_run(run_id))
        self.assertTrue(
            any("keeping database record for retry" in message for message in captured.output)
        )

    def test_delete_run_can_finish_database_cleanup_after_artifacts_are_already_missing(self) -> None:
        """Ensure deletion can finish once artifacts have already been removed externally."""
        repo = InMemoryRunsRepo()
        self.service._db_enabled = True
        self.service._runs_repo = repo

        cfg = valid_run_config()
        cfg["run_name"] = "delete_retry_demo"
        ok, _, run_id, run_path, _ = self.service.reserve_run_submission(cfg, "alice")
        self.assertTrue(ok)

        Path(run_path).rename(self.results_dir / f"{run_id}_moved")
        self.assertFalse(Path(run_path).exists())

        deleted = self.service.delete_run(run_id)

        self.assertTrue(deleted)
        self.assertIsNone(repo.get_run(run_id))
