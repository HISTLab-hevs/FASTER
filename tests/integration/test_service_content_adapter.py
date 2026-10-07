"""Integration tests for the service-layer content and scenario adapter."""

from __future__ import annotations

import unittest

from tests.helpers import reload_module


class _FakeDefaultsService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def load_defaults(self, username: str | None = None) -> dict:
        self.calls.append(("load_defaults", username))
        return {"username": username, "source": "defaults"}

    def save_defaults(
        self,
        config: dict,
        username: str | None = None,
        updated_by: str | None = None,
    ) -> bool:
        self.calls.append(("save_defaults", dict(config), username, updated_by))
        return True


class _FakeScenariosService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def list_scenarios(self, username: str | None = None) -> list[str]:
        self.calls.append(("list_scenarios", username))
        return ["my/demo.yaml"]

    def load_scenario(self, name: str, username: str | None = None) -> dict | None:
        self.calls.append(("load_scenario", name, username))
        return {"name": name, "username": username}

    def save_user_scenario(self, username: str, config: dict, name: str = "") -> str:
        self.calls.append(("save_user_scenario", username, dict(config), name))
        return f"my/{name or 'generated.yaml'}"

    def delete_user_scenario(self, username: str, name: str) -> tuple[bool, str]:
        self.calls.append(("delete_user_scenario", username, name))
        return True, f"Deleted scenario '{name}'"


class ServiceContentAdapterTests(unittest.TestCase):
    """Verify the content adapter delegates defaults and scenarios consistently."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.content_mod = reload_module("web_backend.services.content")

    def test_adapter_routes_defaults_and_scenarios_through_one_interface(self) -> None:
        """Ensure the adapter forwards defaults and scenarios through one facade."""
        defaults = _FakeDefaultsService()
        scenarios = _FakeScenariosService()
        service = self.content_mod.TrainingContentService(
            defaults_service=defaults,
            scenarios_service=scenarios,
        )

        self.assertEqual(["my/demo.yaml"], service.list_scenarios(username="alice"))
        self.assertEqual(
            {"name": "my/demo.yaml", "username": "alice"},
            service.load_scenario("my/demo.yaml", username="alice"),
        )
        self.assertEqual(
            "my/demo.yaml",
            service.save_user_scenario("alice", {"learning_rate": 0.02}, name="demo.yaml"),
        )
        self.assertEqual(
            (True, "Deleted scenario 'my/demo.yaml'"),
            service.delete_user_scenario("alice", "my/demo.yaml"),
        )
        self.assertEqual(
            {"username": "alice", "source": "defaults"},
            service.load_defaults(username="alice"),
        )
        self.assertTrue(
            service.save_defaults({"learning_rate": 0.01}, username="alice", updated_by="alice")
        )

        self.assertEqual(
            [
                ("load_defaults", "alice"),
                ("save_defaults", {"learning_rate": 0.01}, "alice", "alice"),
            ],
            defaults.calls,
        )
        self.assertEqual(
            [
                ("list_scenarios", "alice"),
                ("load_scenario", "my/demo.yaml", "alice"),
                ("save_user_scenario", "alice", {"learning_rate": 0.02}, "demo.yaml"),
                ("delete_user_scenario", "alice", "my/demo.yaml"),
            ],
            scenarios.calls,
        )

    def test_adapter_exposes_scenario_name_helper(self) -> None:
        """Ensure the adapter exposes the scenario-name sanitization helper."""
        name = self.content_mod.TrainingContentService.sanitize_scenario_name("demo scenario")

        self.assertEqual("demo_scenario.yaml", name)
