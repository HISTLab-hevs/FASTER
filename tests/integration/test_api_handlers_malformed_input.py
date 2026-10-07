"""Integration tests for malformed request bodies and error masking."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from tests.helpers import reload_module


class ApiHandlerMalformedInputTests(unittest.TestCase):
    """Cover malformed request bodies and handler error masking behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api_handlers = reload_module("web_backend.api_handlers.api")

    def test_fl_api_handler_rejects_empty_and_non_json_payloads(self) -> None:
        """Ensure empty and malformed request bodies are treated as invalid JSON."""
        for payload in ("", "   ", None, "{not-json"):
            with self.subTest(payload=payload):
                response = self.api_handlers.fl_api_handler(payload)
                self.assertEqual({"error": "Invalid JSON request"}, response)

    def test_fl_api_handler_masks_json_array_root(self) -> None:
        """Ensure a valid JSON array body is rejected as an invalid request shape."""
        with mock.patch.object(self.api_handlers.logger, "exception") as log_exception:
            response = self.api_handlers.fl_api_handler("[]")

        self.assertEqual({"error": "Invalid JSON request"}, response)
        log_exception.assert_not_called()

    def test_fl_api_handler_masks_nested_data_shape_mismatch(self) -> None:
        """Ensure nested payload shape mismatches are handled safely."""
        payload = json.dumps(
            {
                "command": "login",
                "data": [],
            }
        )

        with mock.patch.object(self.api_handlers.logger, "exception") as log_exception:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "Invalid JSON request"}, response)
        log_exception.assert_not_called()

    def test_fl_api_handler_masks_downstream_auth_exception(self) -> None:
        """Ensure exceptions from auth token verification are masked from callers."""
        payload = json.dumps(
            {
                "command": "me",
                "token": "good",
                "data": {},
            }
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            side_effect=RuntimeError("token verification exploded"),
        ), mock.patch.object(self.api_handlers.logger, "exception") as log_exception:
            response = self.api_handlers.fl_api_handler(payload)

        self.assertEqual({"error": "Internal server error"}, response)
        log_exception.assert_called_once()
