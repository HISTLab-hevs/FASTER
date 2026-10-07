#!/usr/bin/env python
"""CLI tool for managing job manager users.

Usage::

    python -m api.manage_users add <username> [--admin]
    python -m api.manage_users delete <username>
    python -m api.manage_users list

The database connection is configured via ``config.yaml`` and the
usual ``DB_*`` environment variable overrides.
"""

import argparse
import getpass
import sys

from api.user_db import create_user, delete_user, init_users_db, list_users


def cmd_add(args: argparse.Namespace) -> None:
    """Create a new user after prompting for a password.

    Args:
        args: Parsed CLI arguments with ``username`` and ``admin`` fields.
    """
    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Error: passwords do not match.")
        sys.exit(1)
    if not password:
        print("Error: password cannot be empty.")
        sys.exit(1)
    if create_user(args.username, password, is_admin=args.admin):
        role = "admin " if args.admin else ""
        print(f"{role}User '{args.username}' created.")
    else:
        print(f"Error: user '{args.username}' already exists.")
        sys.exit(1)


def cmd_delete(args: argparse.Namespace) -> None:
    """Delete an existing user.

    Args:
        args: Parsed CLI arguments with ``username`` field.
    """
    if delete_user(args.username):
        print(f"User '{args.username}' deleted.")
    else:
        print(f"Error: user '{args.username}' not found.")
        sys.exit(1)


def cmd_list(_args: argparse.Namespace) -> None:
    """Print a formatted table of all registered users.

    Args:
        _args: Parsed CLI arguments (unused).
    """
    users = list_users()
    if not users:
        print("No users found.")
        return
    print(f"{'ID':<6} {'Username':<30} {'Admin':<7} {'Created at'}")
    print("-" * 70)
    for u in users:
        admin_flag = "yes" if u.get("is_admin") else "no"
        print(f"{u['id']:<6} {u['username']:<30} {admin_flag:<7} {u['created_at']}")


def main() -> None:
    """Entry point: parse arguments and dispatch to the appropriate sub-command."""
    init_users_db()
    parser = argparse.ArgumentParser(description="Manage job manager users")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Create a new user")
    p_add.add_argument("username")
    p_add.add_argument("--admin", action="store_true", help="Grant admin privileges")
    p_add.set_defaults(func=cmd_add)

    p_del = sub.add_parser("delete", help="Delete a user")
    p_del.add_argument("username")
    p_del.set_defaults(func=cmd_delete)

    p_list = sub.add_parser("list", help="List all users")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
