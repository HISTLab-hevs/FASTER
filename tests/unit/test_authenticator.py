"""Tests for password hashing, login, and user administration in :mod:`auth.authenticator`."""

from __future__ import annotations

import unittest
from unittest import mock

from tests.helpers import reload_module


class _FakeUsersRepo:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {}

    def create_user(self, payload: dict) -> int:
        self.users[payload["identifier"]] = {
            "identifier": payload["identifier"],
            "email": payload.get("email"),
            "password_hash": payload["password_hash"],
            "role": payload.get("role", "user"),
            "active": bool(payload.get("active", True)),
            "created_at": payload.get("created_at"),
            "created_by": payload.get("created_by", "system"),
        }
        return len(self.users)

    def get_user_by_identifier(self, identifier: str):
        user = self.users.get(identifier)
        if not user:
            return None
        return mock.Mock(
            identifier=user["identifier"],
            email=user["email"],
            password_hash=user["password_hash"],
            role=user["role"],
            active=user["active"],
            created_at=user.get("created_at"),
            created_by=user["created_by"],
        )

    def get_user_by_email(self, email: str):
        needle = str(email or "").strip().lower()
        for user in self.users.values():
            if str(user.get("email") or "").strip().lower() != needle:
                continue
            return mock.Mock(
                identifier=user["identifier"],
                email=user["email"],
                password_hash=user["password_hash"],
                role=user["role"],
                active=user["active"],
                created_at=user.get("created_at"),
                created_by=user["created_by"],
            )
        return None

    def list_users(self, include_inactive: bool = True):
        rows = []
        for user in self.users.values():
            if not include_inactive and not user["active"]:
                continue
            rows.append(
                mock.Mock(
                    identifier=user["identifier"],
                    email=user["email"],
                    password_hash=user["password_hash"],
                    role=user["role"],
                    active=user["active"],
                    created_at=user.get("created_at"),
                    created_by=user["created_by"],
                )
            )
        return rows

    def update_user(self, identifier: str, updates: dict) -> bool:
        if identifier not in self.users:
            return False
        self.users[identifier].update(updates)
        return True

    def delete_user(self, identifier: str) -> bool:
        return self.users.pop(identifier, None) is not None


class _ExplodingUsersRepo(_FakeUsersRepo):
    def create_user(self, payload: dict) -> int:
        raise RuntimeError("duplicate key")


class _DictShapedUsersRepo(_FakeUsersRepo):
    def get_user_by_identifier(self, identifier: str):
        user = self.users.get(identifier)
        return dict(user) if user else None

    def get_user_by_email(self, email: str):
        needle = str(email or "").strip().lower()
        for user in self.users.values():
            if str(user.get("email") or "").strip().lower() == needle:
                return dict(user)
        return None


class _ExplodingLookupUsersRepo(_FakeUsersRepo):
    def get_user_by_identifier(self, identifier: str):
        raise RuntimeError("lookup failed")

    def get_user_by_email(self, email: str):
        raise RuntimeError("lookup failed")


class AuthenticatorDbModeTests(unittest.TestCase):
    """Exercise the DB-backed authenticator behavior and failure handling."""

    def setUp(self) -> None:
        self.auth_mod = reload_module("auth.authenticator")
        self.repo = _FakeUsersRepo()
        self.auth = self.auth_mod.Authenticator(
            secret_key="secret-for-tests-with-safe-length-1234567890"
        )
        self.auth._db_enabled = True
        self.auth._users_repo = self.repo

    def test_register_authenticate_and_verify_token_roundtrip(self) -> None:
        """Ensure registration, login, and token verification work end to end."""
        created = self.auth.register_user(
            username="alice",
            password="StrongPass1!",
            email="alice@example.com",
            created_by="test",
        )

        self.assertTrue(created)

        ok, token, user = self.auth.authenticate("alice", "StrongPass1!")
        self.assertTrue(ok)
        self.assertTrue(token)
        self.assertEqual("alice", user["identifier"])

        valid, identifier = self.auth.verify_token(token)
        self.assertTrue(valid)
        self.assertEqual("alice", identifier)

    def test_admin_management_allows_admin_and_blocks_regular_user(self) -> None:
        """Ensure admin-only management operations enforce caller permissions."""
        ok, _, _ = self.auth.create_user(
            "admin",
            "admin@example.com",
            "AdminPass1!",
            "admin",
            created_by="bootstrap",
        )
        self.assertTrue(ok)

        ok, _, _ = self.auth.create_user(
            "member",
            "member@example.com",
            "MemberPass1!",
            "user",
            created_by="bootstrap",
        )
        self.assertTrue(ok)

        blocked, msg = self.auth.admin_delete_user("member", "admin")
        self.assertFalse(blocked)
        self.assertEqual("Forbidden", msg)

        deleted, msg = self.auth.admin_delete_user("admin", "member")
        self.assertTrue(deleted)
        self.assertEqual("User deleted", msg)

        remaining_users = self.auth.list_users()
        identifiers = {item["identifier"] for item in remaining_users}
        self.assertEqual({"admin"}, identifiers)

    def test_authenticated_password_change_updates_password(self) -> None:
        """Ensure authenticated password changes reject bad credentials and persist new ones."""
        created = self.auth.register_user(
            username="alice",
            password="StrongPass1!",
            email="alice@example.com",
            created_by="test",
        )
        self.assertTrue(created)

        blocked, msg = self.auth.change_password_authenticated("alice", "wrong-old", "NewStrongPass2!")
        self.assertFalse(blocked)
        self.assertEqual("Current password is incorrect", msg)

        ok, msg = self.auth.change_password_authenticated("alice", "StrongPass1!", "NewStrongPass2!")
        self.assertTrue(ok)
        self.assertEqual("Password updated", msg)

        ok, token, _ = self.auth.authenticate("alice", "NewStrongPass2!")
        self.assertTrue(ok)
        self.assertTrue(token)

    def test_verify_token_rejects_invalid_payload(self) -> None:
        """Ensure invalid JWT payloads are rejected safely."""
        valid, identifier = self.auth.verify_token("this-is-not-a-jwt")
        self.assertFalse(valid)
        self.assertEqual("", identifier)

    def test_user_listing_does_not_leak_password_hashes(self) -> None:
        """Ensure user listings omit password hashes from outward-facing payloads."""
        created = self.auth.register_user(
            username="alice",
            password="StrongPass1!",
            email="alice@example.com",
            created_by="test",
        )
        self.assertTrue(created)

        users = self.auth.list_users()
        self.assertEqual(1, len(users))
        self.assertNotIn("password_hash", users[0])
        self.assertIn("password_hash", self.repo.users["alice"])

    def test_authenticate_supports_email_lookup_via_repository_method(self) -> None:
        """Ensure authentication can resolve users by normalized email address."""
        created = self.auth.register_user(
            username="alice",
            password="StrongPass1!",
            email="Alice@Example.com",
            created_by="test",
        )
        self.assertTrue(created)

        ok, token, user = self.auth.authenticate("alice@example.com", "StrongPass1!")

        self.assertTrue(ok)
        self.assertTrue(token)
        self.assertEqual("alice", user["identifier"])

    def test_authenticate_rejects_inactive_accounts(self) -> None:
        """Ensure inactive accounts cannot authenticate even with a valid password."""
        created = self.auth.register_user(
            username="alice",
            password="StrongPass1!",
            email="alice@example.com",
            created_by="test",
        )
        self.assertTrue(created)
        self.repo.users["alice"]["active"] = False

        ok, token, user = self.auth.authenticate("alice", "StrongPass1!")

        self.assertFalse(ok)
        self.assertEqual("", token)
        self.assertEqual({}, user)

    def test_repository_lookup_shape_mismatches_are_treated_as_missing_users(self) -> None:
        """Ensure dict-shaped repository rows fail closed instead of leaking exceptions."""
        self.auth._users_repo = _DictShapedUsersRepo()
        self.auth._users_repo.users["alice"] = {
            "identifier": "alice",
            "email": "alice@example.com",
            "password_hash": "$2b$12$not_a_real_hash",
            "role": "user",
            "active": True,
            "created_at": None,
            "created_by": "test",
        }

        with self.assertLogs("auth.authenticator", level="WARNING") as captured:
            user = self.auth.get_user("alice")
            ok, token, payload = self.auth.authenticate("alice", "StrongPass1!")

        self.assertIsNone(user)
        self.assertFalse(ok)
        self.assertEqual("", token)
        self.assertEqual({}, payload)
        self.assertTrue(any("failed db lookup" in line.lower() for line in captured.output))

    def test_repository_lookup_exceptions_are_masked_from_auth_flow(self) -> None:
        """Ensure repository lookup errors fail closed without exposing internal details."""
        self.auth._users_repo = _ExplodingLookupUsersRepo()

        with self.assertLogs("auth.authenticator", level="WARNING") as captured:
            ok, token, payload = self.auth.authenticate("alice", "StrongPass1!")

        self.assertFalse(ok)
        self.assertEqual("", token)
        self.assertEqual({}, payload)
        self.assertTrue(any("lookup failed" in line for line in captured.output))

    def test_register_user_rejects_duplicate_email_case_insensitively(self) -> None:
        """Ensure duplicate email registration is rejected after normalization."""
        created = self.auth.register_user(
            username="alice",
            password="StrongPass1!",
            email="Alice@Example.com",
            created_by="test",
        )
        self.assertTrue(created)

        duplicate = self.auth.register_user(
            username="bob",
            password="StrongPass1!",
            email=" alice@example.com ",
            created_by="test",
        )

        self.assertFalse(duplicate)

    def test_register_user_handles_duplicate_insert_race_gracefully(self) -> None:
        """Ensure registration logs and fails cleanly when the repository insert races."""
        self.auth._users_repo = _ExplodingUsersRepo()

        with self.assertLogs("auth.authenticator", level="WARNING") as captured:
            created = self.auth.register_user(
                username="alice",
                password="StrongPass1!",
                email="alice@example.com",
                created_by="test",
            )

        self.assertFalse(created)
        self.assertTrue(
            any("Authenticator failed to register user 'alice'" in message for message in captured.output)
        )

    def test_create_user_handles_duplicate_insert_race_gracefully(self) -> None:
        """Ensure admin user creation also degrades gracefully on repository insert races."""
        self.auth._users_repo = _ExplodingUsersRepo()

        with self.assertLogs("auth.authenticator", level="WARNING") as captured:
            ok, msg, user = self.auth.create_user(
                "alice",
                "alice@example.com",
                "StrongPass1!",
                "user",
                created_by="admin",
            )

        self.assertFalse(ok)
        self.assertEqual("Could not create user", msg)
        self.assertEqual({}, user)
        self.assertTrue(
            any("Authenticator failed to create user 'alice'" in message for message in captured.output)
        )

    def test_list_users_accepts_dict_rows_and_skips_malformed_rows(self) -> None:
        """Ensure admin user listings tolerate dict-shaped and malformed repository rows."""
        self.auth._users_repo.list_users = mock.Mock(
            return_value=[
                {
                    "identifier": "alice",
                    "email": "Alice@Example.com",
                    "role": "admin",
                    "active": True,
                    "created_at": "2026-04-12T00:00:00",
                    "created_by": "system",
                },
                None,
                object(),
            ]
        )

        users = self.auth.list_users()

        self.assertEqual(1, len(users))
        self.assertEqual("alice", users[0]["identifier"])
        self.assertEqual("Alice@Example.com", users[0]["email"])
        self.assertEqual("admin", users[0]["role"])
        self.assertEqual("2026-04-12T00:00:00", users[0]["created_at"])

    def test_list_users_masks_repository_exceptions(self) -> None:
        """Ensure repository listing failures do not crash admin listing callers."""
        self.auth._users_repo.list_users = mock.Mock(side_effect=RuntimeError("list failed"))

        with self.assertLogs("auth.authenticator", level="WARNING") as captured:
            users = self.auth.list_users()

        self.assertEqual([], users)
        self.assertTrue(any("list failed" in line for line in captured.output))
