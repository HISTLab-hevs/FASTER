"""Tests for recursive sanitization of API-visible config payloads."""

from __future__ import annotations

import unittest

from tests.helpers import reload_module


class SharedSanitizerTests(unittest.TestCase):
    """Exercise recursive sanitization of API-visible config payloads."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.shared = reload_module("web_backend.api_handlers.shared")

    def test_sanitize_config_obj_normalizes_nested_strings_and_truncates_code(self) -> None:
        """Ensure control characters are cleaned recursively and code blobs are capped."""
        payload = {
            "title": "  Hello\tWorld\n",
            123: "  key\tvalue  ",
            "nested": [
                "  first\x07 item  ",
                {"emoji": " café  crème "},
            ],
            "custom_model_code": "x" * 250_050,
        }

        sanitized = self.shared.sanitize_config_obj(payload)

        self.assertEqual("Hello World", sanitized["title"])
        self.assertEqual("key value", sanitized["123"])
        self.assertEqual(["first item", {"emoji": "café crème"}], sanitized["nested"])
        self.assertEqual(250_000, len(sanitized["custom_model_code"]))
        self.assertTrue(sanitized["custom_model_code"].startswith("x" * 10))
