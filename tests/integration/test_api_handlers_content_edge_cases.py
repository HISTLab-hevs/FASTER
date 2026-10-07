"""Integration tests for content and scenario API edge cases."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from tests.helpers import reload_module


class ApiHandlerContentEdgeCaseTests(unittest.TestCase):
    """Cover frontend-visible content and scenario API edge cases."""

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

    def test_parse_scenario_yaml_rejects_non_mapping_root(self) -> None:
        """Ensure uploaded scenario YAML must deserialize to a mapping root."""
        payload = json.dumps(
            {
                "command": "parse_scenario_yaml",
                "token": "good",
                "data": {"yaml_text": "- just\n- a\n- list\n"},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ):
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "YAML root must be a mapping/object"}, response)

    def test_get_scenario_returns_not_found_error_for_missing_entry(self) -> None:
        """Ensure get_scenario returns a stable not-found message for missing entries."""
        payload = json.dumps(
            {
                "command": "get_scenario",
                "token": "good",
                "data": {"name": "my/missing.yaml"},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "load_scenario",
            return_value=None,
        ) as load_scenario:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "Scenario 'my/missing.yaml' not found"}, response)
        load_scenario.assert_called_once_with("my/missing.yaml", "alice")

    def test_get_scenario_rejects_non_string_name_shape_without_crashing(self) -> None:
        """Ensure get_scenario fails closed when the nested name is the wrong container type."""
        payload = json.dumps(
            {
                "command": "get_scenario",
                "token": "good",
                "data": {"name": []},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ) as verify_token_claims:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "Scenario '[]' not found"}, response)
        verify_token_claims.assert_called_once_with("good")

    def test_delete_scenario_surfaces_service_error_message(self) -> None:
        """Ensure delete_scenario forwards the service error to the frontend."""
        payload = json.dumps(
            {
                "command": "delete_scenario",
                "token": "good",
                "data": {"name": "demo.yaml"},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "delete_user_scenario",
            return_value=(False, "Only user scenarios can be deleted"),
        ) as delete_user_scenario:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "Only user scenarios can be deleted"}, response)
        delete_user_scenario.assert_called_once_with("alice", "demo.yaml")

    def test_save_scenario_rejects_non_mapping_config_shape(self) -> None:
        """Ensure save_scenario fails closed when config is the wrong container type."""
        payload = json.dumps(
            {
                "command": "save_scenario",
                "token": "good",
                "data": {"name": "demo", "config": []},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(),
        ), mock.patch.object(
            self.api_handlers.svc,
            "save_user_scenario",
        ) as save_user_scenario:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "Scenario config must be a mapping"}, response)
        save_user_scenario.assert_not_called()

    def test_list_custom_datasets_passes_admin_visibility_flag(self) -> None:
        """Ensure dataset listing forwards admin visibility to the service layer."""
        payload = json.dumps(
            {
                "command": "list_custom_datasets",
                "token": "good",
                "data": {},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=self._protected_claims(role="admin"),
        ), mock.patch.object(
            self.api_handlers.svc,
            "list_custom_datasets",
            return_value=[{"dataset_ref": "demo.csv"}],
        ) as list_custom_datasets:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"success": True, "datasets": [{"dataset_ref": "demo.csv"}]}, response)
        list_custom_datasets.assert_called_once_with("alice", is_admin=True)
