"""Domain handlers for auth and account-related API commands."""

from __future__ import annotations

from typing import Any


def handle_public_auth_command(command: str, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Handle unauthenticated auth commands."""
    auth = ctx["auth"]
    is_admin_role = ctx["is_admin_role"]

    if command == "login":
        identifier = str(data.get("identifier", "") or data.get("username", "") or "")
        ok, access_token, user = auth.authenticate(identifier, data.get("password", ""))
        if not ok:
            return {"error": "Invalid credentials"}
        return {
            "success": True,
            "token": access_token,
            "username": user.get("identifier", identifier),
            "email": user.get("email"),
            "role": user.get("role", "user"),
            "is_admin": is_admin_role(user.get("role", "user")),
        }

    if command == "register":
        return {"error": "Account creation is disabled"}

    return None


def handle_account_command(command: str, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Handle authenticated account/profile commands."""
    auth = ctx["auth"]
    username = ctx["username"]
    role = ctx["role"]
    is_admin = ctx["is_admin"]

    if command == "me":
        return {
            "success": True,
            "username": username,
            "email": ctx["claims"].get("email"),
            "role": role,
            "is_admin": is_admin,
        }

    if command == "change_password_authenticated":
        old_password = str(data.get("old_password", "") or "")
        new_password = str(data.get("new_password", "") or "")
        confirm_new_password = str(data.get("confirm_new_password", "") or "")

        if not old_password or not new_password or not confirm_new_password:
            return {
                "error": "Current password, new password and confirmation are required"
            }
        if new_password != confirm_new_password:
            return {"error": "New password and confirmation do not match"}

        ok, msg = auth.change_password_authenticated(
            username,
            old_password,
            new_password,
        )
        if not ok:
            return {"error": msg}
        return {"success": True, "message": msg}

    return None
