"""Tests for the ``auth.user_cli`` administration commands."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from tests.helpers import reload_module


class UserCliTests(unittest.TestCase):
    """Cover the DB-backed user management CLI behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.user_cli = reload_module("auth.user_cli")

    def test_configure_storage_mode_is_noop_for_db_only_runtime(self) -> None:
        """Ensure storage-mode configuration stays a no-op in the DB-only runtime."""
        args = mock.Mock()

        result = self.user_cli._configure_storage_mode(args)

        self.assertIsNone(result)
        self.assertEqual([], args.mock_calls)

    def test_list_users_reports_backend_error_when_db_unavailable(self) -> None:
        """Ensure list-users surfaces backend connectivity errors to the operator."""
        fake_auth = mock.Mock()
        fake_auth._db_ready.return_value = False
        fake_auth.backend_error.return_value = "Cannot connect to MariaDB"

        with mock.patch.object(self.user_cli, "_new_auth", return_value=fake_auth):
            output = io.StringIO()
            with redirect_stdout(output):
                rc = self.user_cli._list_users()

        self.assertEqual(1, rc)
        self.assertIn("Cannot connect to MariaDB", output.getvalue())

    def test_create_user_uses_db_backed_authenticator(self) -> None:
        """Ensure user creation delegates to the DB-backed authenticator with CLI metadata."""
        fake_auth = mock.Mock()
        fake_auth._db_ready.return_value = True
        fake_auth.get_user.return_value = None
        fake_auth.register_user.return_value = True

        with mock.patch.object(self.user_cli, "_new_auth", return_value=fake_auth):
            output = io.StringIO()
            with redirect_stdout(output):
                rc = self.user_cli._create_user(
                    username="alice",
                    password="StrongPass1!",
                    role="user",
                    email="alice@example.com",
                    allow_weak_password=False,
                )

        self.assertEqual(0, rc)
        fake_auth.register_user.assert_called_once_with(
            username="alice",
            password="StrongPass1!",
            role="user",
            email="alice@example.com",
            created_by="cli",
        )
        self.assertIn("created successfully", output.getvalue())
