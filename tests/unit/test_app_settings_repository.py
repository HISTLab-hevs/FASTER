"""Tests for the MariaDB-backed application settings repository."""

from __future__ import annotations

import unittest
from unittest import mock

from tests.helpers import reload_module


class _FakeSession:
    def __init__(self, factory, calls: list[tuple[str, str, tuple]]) -> None:
        self._factory = factory
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        return None

    def execute(self, query: str, params: tuple = ()) -> int:
        self._calls.append(("execute", query, params))
        return 1

    def fetchone(self, query: str, params: tuple = ()):
        self._calls.append(("fetchone", query, params))
        return None


class AppSettingsRepositoryTests(unittest.TestCase):
    """Cover the MariaDB-backed app settings repository contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_mod = reload_module("database.repositories.app_settings_repository")

    def test_get_setting_queries_without_runtime_ddl_when_schema_exists(self) -> None:
        """Ensure reads use the existing schema instead of issuing runtime DDL."""
        calls: list[tuple[str, str, tuple]] = []

        def factory(_factory):
            return _FakeSession(_factory, calls)

        with mock.patch.object(self.repo_mod, "MariaDBSession", side_effect=factory):
            repo = self.repo_mod.MariaAppSettingsRepository(mock.sentinel.factory)
            row = repo.get_setting("saved_defaults")

        self.assertIsNone(row)
        self.assertEqual(1, len(calls))
        self.assertEqual("fetchone", calls[0][0])
        self.assertIn("SELECT * FROM app_settings", calls[0][1])

    def test_upsert_setting_issues_only_upsert_queries(self) -> None:
        """Ensure writes use idempotent upserts rather than schema-changing queries."""
        calls: list[tuple[str, str, tuple]] = []

        def factory(_factory):
            return _FakeSession(_factory, calls)

        with mock.patch.object(self.repo_mod, "MariaDBSession", side_effect=factory):
            repo = self.repo_mod.MariaAppSettingsRepository(mock.sentinel.factory)
            ok = repo.upsert_setting("saved_defaults", {"learning_rate": 0.01}, updated_by="alice")
            repo.upsert_setting("saved_defaults", {"learning_rate": 0.02}, updated_by="bob")

        self.assertTrue(ok)
        upsert_calls = [entry for entry in calls if "INSERT INTO app_settings" in entry[1]]
        self.assertEqual(2, len(upsert_calls))
        self.assertIn("ON DUPLICATE KEY UPDATE", upsert_calls[0][1])
