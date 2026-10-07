"""Integration tests for persistence of saved experiment defaults."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import reload_module


class _FakeAppSettingsRepo:
    def __init__(self) -> None:
        self.saved: dict[str, dict] = {}
        self.updated_by: dict[str, str | None] = {}
        self.rows: dict[str, _FakeSettingRow] = {}

    def get_setting(self, setting_key: str):
        return self.rows.get(setting_key)

    def upsert_setting(self, setting_key: str, setting_json: dict, updated_by: str | None = None) -> bool:
        self.saved[setting_key] = dict(setting_json)
        self.updated_by[setting_key] = updated_by
        self.rows[setting_key] = _FakeSettingRow(setting_json)
        return True


class _FakeSettingRow:
    def __init__(self, payload: dict) -> None:
        self.setting_json = dict(payload)


class ServiceDefaultsPersistenceTests(unittest.TestCase):
    """Cover DB-backed defaults loading and persistence behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.service_mod = reload_module("web_backend.service")

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        self.bootstrap_path = Path(self.tmpdir.name) / "config.yaml"
        self.bootstrap_path.write_text(
            "server_learning_rate: 0.002\nlearning_rate: 0.001\nbatch_size: 32\n",
            encoding="utf-8",
        )

        self.config_patch = mock.patch.object(
            self.service_mod,
            "DEFAULT_CONFIG_FILE",
            str(self.bootstrap_path),
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

        self.service = self.service_mod.TrainingServiceLayer()
        self.repo = _FakeAppSettingsRepo()
        self.service._db_enabled = True
        self.service._app_settings_repo = self.repo

    def test_load_defaults_overlays_persisted_values_on_bootstrap_config(self) -> None:
        """Ensure persisted defaults override matching bootstrap values."""
        self.repo.rows["saved_defaults"] = _FakeSettingRow({"learning_rate": 0.004, "server_learning_rate": 0.005, "batch_size": 64})

        defaults = self.service.load_defaults()

        self.assertEqual(0.004, defaults["learning_rate"])
        self.assertEqual(0.005, defaults["server_learning_rate"])
        self.assertEqual(64, defaults["batch_size"])

    def test_load_defaults_preserves_unspecified_bootstrap_values(self) -> None:
        """Ensure unspecified bootstrap defaults survive persisted overlays."""
        self.repo.rows["saved_defaults"] = _FakeSettingRow({"learning_rate": 0.004})

        defaults = self.service.load_defaults()

        self.assertEqual(0.004, defaults["learning_rate"])
        self.assertEqual(0.002, defaults["server_learning_rate"])
        self.assertEqual(32, defaults["batch_size"])

    def test_load_defaults_falls_back_to_bootstrap_config_when_no_saved_defaults_exist(self) -> None:
        """Ensure bootstrap config is used when no saved defaults exist."""
        self.repo.rows.clear()

        defaults = self.service.load_defaults()

        self.assertEqual(0.001, defaults["learning_rate"])
        self.assertEqual(0.002, defaults["server_learning_rate"])
        self.assertEqual(32, defaults["batch_size"])

    def test_save_defaults_persists_to_repo_without_mutating_bootstrap_config(self) -> None:
        """Ensure saving defaults persists to the repo without rewriting bootstrap config."""
        original = self.bootstrap_path.read_text(encoding="utf-8")

        ok = self.service.save_defaults({"learning_rate": 0.01, "server_learning_rate": 0.02}, updated_by="alice")

        self.assertTrue(ok)
        self.assertEqual({"learning_rate": 0.01, "server_learning_rate": 0.02}, self.repo.saved["saved_defaults"])
        self.assertEqual("alice", self.repo.updated_by["saved_defaults"])
        self.assertEqual(original, self.bootstrap_path.read_text(encoding="utf-8"))

    def test_load_defaults_prefers_user_specific_saved_defaults(self) -> None:
        """Ensure user-specific saved defaults win over shared saved defaults."""
        self.repo.rows["saved_defaults"] = _FakeSettingRow({"learning_rate": 0.004, "server_learning_rate": 0.005})
        self.repo.rows["saved_defaults::user::alice"] = _FakeSettingRow({"learning_rate": 0.02, "server_learning_rate": 0.03, "batch_size": 16})

        defaults = self.service.load_defaults(username="alice")

        self.assertEqual(0.02, defaults["learning_rate"])
        self.assertEqual(0.03, defaults["server_learning_rate"])
        self.assertEqual(16, defaults["batch_size"])

    def test_load_defaults_for_user_falls_back_to_global_saved_defaults(self) -> None:
        """Ensure user lookups fall back to global saved defaults when needed."""
        self.repo.rows["saved_defaults"] = _FakeSettingRow({"learning_rate": 0.004, "server_learning_rate": 0.005})

        defaults = self.service.load_defaults(username="alice")

        self.assertEqual(0.004, defaults["learning_rate"])
        self.assertEqual(0.005, defaults["server_learning_rate"])
        self.assertEqual(32, defaults["batch_size"])

    def test_save_defaults_persists_to_user_specific_key_when_username_provided(self) -> None:
        """Ensure user-specific saves write to the scoped defaults key."""
        ok = self.service.save_defaults({"learning_rate": 0.03, "server_learning_rate": 0.04}, username="alice", updated_by="alice")

        self.assertTrue(ok)
        self.assertEqual({"learning_rate": 0.03, "server_learning_rate": 0.04}, self.repo.saved["saved_defaults::user::alice"])
        self.assertEqual("alice", self.repo.updated_by["saved_defaults::user::alice"])

    def test_load_defaults_round_trips_client_churn_keys(self) -> None:
        """Ensure client-churn keys survive persistence and reloads."""
        self.repo.rows["saved_defaults"] = _FakeSettingRow(
            {
                "initial_eligible_clients": 2,
                "death_prob": 0.25,
                "new_client_prob": 0.5,
                "seed": 7,
            }
        )

        defaults = self.service.load_defaults()

        self.assertEqual(2, defaults["initial_eligible_clients"])
        self.assertEqual(0.25, defaults["death_prob"])
        self.assertEqual(0.5, defaults["new_client_prob"])
        self.assertEqual(7, defaults["seed"])
