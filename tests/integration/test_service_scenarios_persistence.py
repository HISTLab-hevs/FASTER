"""Integration tests for scenario persistence and repository failure handling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import reload_module


class _ScenarioRow:
    def __init__(self, scenario_yaml: str) -> None:
        self.scenario_yaml = scenario_yaml


class _RepoListFails:
    def list_scenarios(self, username: str):
        raise RuntimeError("db unavailable")


class _ListedScenarioRow:
    def __init__(self, scenario_name: str) -> None:
        self.scenario_name = scenario_name


class _RepoListSucceeds:
    def __init__(self, scenario_names: list[str]) -> None:
        self._scenario_names = list(scenario_names)

    def list_scenarios(self, username: str):
        return [_ListedScenarioRow(name) for name in self._scenario_names]


class _RepoLoadFails:
    def get_scenario(self, username: str, scenario_name: str):
        raise RuntimeError("db unavailable")


class _RepoLoadSucceeds:
    def __init__(self, scenario_yaml: str) -> None:
        self._row = _ScenarioRow(scenario_yaml)

    def get_scenario(self, username: str, scenario_name: str):
        return self._row


class _RepoSaveFails:
    def update_scenario(self, username: str, scenario_name: str, updates: dict):
        raise RuntimeError("db unavailable")

    def create_scenario(self, payload: dict):
        raise RuntimeError("db unavailable")


class _RepoSaveSucceeds:
    def __init__(self) -> None:
        self.updated: list[tuple[str, str, dict]] = []
        self.created: list[dict] = []

    def update_scenario(self, username: str, scenario_name: str, updates: dict):
        self.updated.append((username, scenario_name, dict(updates)))
        return False

    def create_scenario(self, payload: dict):
        self.created.append(dict(payload))
        return 1

    def delete_scenario(self, username: str, scenario_name: str):
        return True


class ScenarioPersistenceTests(unittest.TestCase):
    """Cover DB-backed scenario persistence and filesystem isolation rules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios_mod = reload_module("web_backend.services.scenarios")

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.base_dir = self.root / "scenarios"
        self.base_dir.mkdir()
        self.user_dir = self.base_dir / "users" / "alice"
        self.user_dir.mkdir(parents=True)

    def _service(
        self,
        *,
        repo,
        db_enabled: bool = True,
        load_config_fn=None,
    ):
        if load_config_fn is None:
            def load_config_fn(path: str):
                import yaml
                return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        return self.scenarios_mod.ScenariosService(
            scenarios_dir=str(self.base_dir),
            load_config_fn=load_config_fn,
            db_enabled_getter=lambda: db_enabled,
            scenarios_repo_getter=lambda: repo,
        )

    def test_list_scenarios_raises_when_db_listing_fails(self) -> None:
        """Ensure scenario listing surfaces DB failures instead of falling back silently."""
        (self.user_dir / "saved.yaml").write_text("learning_rate: 0.01\n", encoding="utf-8")
        service = self._service(repo=_RepoListFails())

        with self.assertRaises(RuntimeError):
            service.list_scenarios(username="alice")

    def test_load_scenario_raises_when_db_load_fails(self) -> None:
        """Ensure scenario loading surfaces DB failures for user-owned scenarios."""
        (self.user_dir / "demo.yaml").write_text("batch_size: 32\n", encoding="utf-8")
        service = self._service(repo=_RepoLoadFails())

        with self.assertRaises(RuntimeError):
            service.load_scenario("my/demo.yaml", username="alice")

    def test_load_scenario_uses_persisted_db_copy(self) -> None:
        """Ensure scenario loads prefer the persisted DB copy over filesystem content."""
        (self.user_dir / "demo.yaml").write_text("batch_size: 32\n", encoding="utf-8")
        service = self._service(repo=_RepoLoadSucceeds("batch_size: 64\n"))

        loaded = service.load_scenario("my/demo.yaml", username="alice")

        self.assertEqual({"batch_size": 64}, loaded)

    def test_load_scenario_accepts_dict_shaped_db_rows(self) -> None:
        """Ensure persisted scenarios tolerate dict-shaped repository rows."""
        (self.user_dir / "demo.yaml").write_text("batch_size: 32\n", encoding="utf-8")

        class _Repo:
            def get_scenario(self, username: str, scenario_name: str):
                return {
                    "scenario_yaml": "batch_size: 64\n",
                }

        service = self._service(repo=_Repo())

        loaded = service.load_scenario("my/demo.yaml", username="alice")

        self.assertEqual({"batch_size": 64}, loaded)

    def test_list_scenarios_skips_malformed_rows_without_crashing(self) -> None:
        """Ensure scenario listings tolerate dict rows and skip malformed entries."""
        class _Repo:
            def list_scenarios(self, username: str):
                return [
                    {"scenario_name": "dict_row.yaml"},
                    None,
                    object(),
                ]

        service = self._service(repo=_Repo())

        scenarios = service.list_scenarios(username="alice")

        self.assertEqual(["my/dict_row.yaml"], scenarios)

    def test_save_user_scenario_raises_when_db_persistence_fails(self) -> None:
        """Ensure scenario saves fail fast when DB persistence fails."""
        service = self._service(repo=_RepoSaveFails())

        with self.assertRaises(RuntimeError):
            service.save_user_scenario("alice", {"learning_rate": 0.02}, name="demo")
        self.assertFalse((self.user_dir / "demo.yaml").exists())

    def test_save_user_scenario_persists_to_db_only(self) -> None:
        """Ensure user scenarios persist to the DB without writing a filesystem copy."""
        repo = _RepoSaveSucceeds()
        service = self._service(repo=repo)

        saved = service.save_user_scenario("alice", {"learning_rate": 0.02}, name="demo")

        self.assertEqual("my/demo.yaml", saved)
        self.assertEqual(
            [("alice", "demo.yaml", {"scenario_yaml": "learning_rate: 0.02\n"})],
            repo.updated,
        )
        self.assertEqual(
            [
                {
                    "owner_identifier": "alice",
                    "scenario_name": "demo.yaml",
                    "scenario_yaml": "learning_rate: 0.02\n",
                }
            ],
            repo.created,
        )
        self.assertFalse((self.user_dir / "demo.yaml").exists())

    def test_save_user_scenario_preserves_ui_mode_marker_when_present(self) -> None:
        """Ensure the optional ui_mode marker survives scenario persistence."""
        repo = _RepoSaveSucceeds()
        service = self._service(repo=repo)

        saved = service.save_user_scenario("alice", {"learning_rate": 0.02, "ui_mode": "advanced"}, name="demo")

        self.assertEqual("my/demo.yaml", saved)
        self.assertIn("ui_mode: advanced\n", repo.updated[0][2]["scenario_yaml"])

    def test_list_scenarios_returns_db_user_entries_without_filesystem_user_entries(self) -> None:
        """Ensure DB-backed user scenarios hide filesystem-only user entries."""
        (self.user_dir / "saved.yaml").write_text("learning_rate: 0.01\n", encoding="utf-8")
        (self.user_dir / "local_only.yaml").write_text("learning_rate: 0.02\n", encoding="utf-8")
        service = self._service(repo=_RepoListSucceeds(["saved.yaml", "db_only.yaml"]))

        scenarios = service.list_scenarios(username="alice")

        self.assertEqual(
            ["my/db_only.yaml", "my/saved.yaml"],
            scenarios,
        )

    def test_save_user_scenario_raises_when_db_persistence_fails_without_filesystem_write(self) -> None:
        """Ensure failed saves do not overwrite an existing filesystem copy."""
        (self.user_dir / "demo.yaml").write_text("learning_rate: 0.99\n", encoding="utf-8")
        service = self._service(repo=_RepoSaveFails())

        with self.assertRaises(RuntimeError):
            service.save_user_scenario("alice", {"learning_rate": 0.02}, name="demo")
        self.assertEqual(
            "learning_rate: 0.99\n",
            (self.user_dir / "demo.yaml").read_text(encoding="utf-8"),
        )

    def test_delete_user_scenario_removes_db_copy(self) -> None:
        """Ensure deleting a user scenario targets the DB-backed copy only."""
        (self.user_dir / "demo.yaml").write_text("batch_size: 32\n", encoding="utf-8")
        service = self._service(repo=_RepoSaveSucceeds())

        ok, msg = service.delete_user_scenario("alice", "my/demo.yaml")

        self.assertTrue(ok)
        self.assertIn("Deleted scenario", msg)
        self.assertTrue((self.user_dir / "demo.yaml").exists())

    def test_delete_user_scenario_rejects_builtin_presets(self) -> None:
        """Ensure built-in scenarios cannot be deleted through the user delete path."""
        service = self._service(repo=_RepoSaveSucceeds())

        ok, msg = service.delete_user_scenario("alice", "demo.yaml")

        self.assertFalse(ok)
        self.assertEqual("Only user scenarios can be deleted", msg)
