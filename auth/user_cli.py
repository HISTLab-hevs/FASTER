"""CLI utility to manage FASTER users in shared MariaDB storage.

Examples:
    python3 -m auth.user_cli create --username alice --password 'Str0ng!Pass123'
    python3 -m auth.user_cli create-admin --password 'AdminPass1!'
    python3 -m auth.user_cli list
"""

import argparse
import getpass
import os
import secrets
import sys

from auth.authenticator import Authenticator


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for auth user-management commands.

    Returns:
        Configured top-level argument parser.
    """
    p = argparse.ArgumentParser(description='Manage FASTER users via CLI')

    sub = p.add_subparsers(dest='command', required=True)

    create = sub.add_parser('create', help='Create a new user')
    create.add_argument('--username', required=True, help='Username to create')
    create.add_argument('--password', help='Password (omit to be prompted securely)')
    create.add_argument('--email', help='Optional email')
    create.add_argument('--role', choices=['user', 'admin'], default='user', help='Role to assign')
    create.add_argument(
        '--allow-weak-password',
        action='store_true',
        help='Allow weak passwords for local/dev bootstrap only',
    )

    create_admin = sub.add_parser('create-admin', help='Create or update an admin user')
    create_admin.add_argument('--username', default='admin', help='Admin username (default: admin)')
    create_admin.add_argument('--password', help='Admin password (omit to be prompted securely)')
    create_admin.add_argument('--email', help='Optional admin email')
    create_admin.add_argument(
        '--allow-weak-password',
        action='store_true',
        help='Allow weak passwords for local/dev bootstrap only',
    )

    list_cmd = sub.add_parser('list', help='List users')
    list_cmd.add_argument('--include-inactive', action='store_true', help='Reserved for future use')

    delete = sub.add_parser('delete', help='Delete user by identifier')
    delete.add_argument('--username', required=True, help='Username to delete')

    promote = sub.add_parser('set-role', help='Set role for an existing user')
    promote.add_argument('--username', required=True, help='Username to update')
    promote.add_argument('--role', required=True, choices=['user', 'admin'], help='Target role')

    return p


def _configure_storage_mode(args: argparse.Namespace) -> None:
    """Accept parser args; auth storage is always DB-backed.

    Args:
        args: Parsed CLI arguments.
    """
    _ = args


def _new_auth() -> Authenticator:
    """Construct an authenticator using the configured secret key.

    Returns:
        Authenticator instance for CLI operations.
    """
    secret = os.getenv('FL_SECRET_KEY', secrets.token_urlsafe(16))
    return Authenticator(secret_key=secret)


def _require_db_backend(auth: Authenticator) -> bool:
    """Return whether DB-backed auth is ready for the CLI command.

    Args:
        auth: Authenticator instance to validate.

    Returns:
        ``True`` when the DB backend is available, otherwise ``False`` after
        printing the backend error.
    """
    if getattr(auth, '_db_ready', lambda: False)():
        return True
    print(auth.backend_error())
    return False


def _print_users(users: list[dict]) -> None:
    """Print a tabular summary of user records for CLI output.

    Args:
        users: API-safe user payloads to display.
    """
    if not users:
        print('No users found.')
        return
    print('identifier\trole\tactive\temail')
    for user in users:
        print(
            f"{user.get('identifier','')}\t{user.get('role','user')}\t"
            f"{1 if user.get('active', True) else 0}\t{user.get('email') or ''}"
        )


def _create_user(username: str, password: str, role: str, email: str | None, allow_weak_password: bool) -> int:
    """Create a user account from the CLI.

    Args:
        username: Account identifier to create.
        password: Plain-text password value.
        role: Role to assign to the new account.
        email: Optional email address for the new account.
        allow_weak_password: Whether to bypass password-strength checks.

    Returns:
        Process exit code for the CLI command.
    """
    if allow_weak_password:
        os.environ['FL_AUTH_ALLOW_WEAK_PASSWORDS'] = '1'

    auth = _new_auth()
    if not _require_db_backend(auth):
        return 1
    existing = auth.get_user(username)
    if existing:
        print(f"User '{username}' already exists.")
        return 1

    ok = auth.register_user(
        username=username,
        password=password,
        role=role,
        email=email,
        created_by='cli',
    )
    if not ok:
        valid, errors = auth.validate_password(password)
        if not valid:
            print('Invalid password:')
            for err in errors:
                print(f'- {err}')
            return 1
        print(f"Could not create user '{username}' (already exists or invalid email).")
        return 1
    print(f"User '{username}' created successfully with role '{role}'.")
    return 0


def _delete_user(username: str) -> int:
    """Delete one user account from the CLI.

    Args:
        username: Account identifier to delete.

    Returns:
        Process exit code for the CLI command.
    """
    auth = _new_auth()
    if not _require_db_backend(auth):
        return 1
    user = auth.get_user(username)
    if not user:
        print(f"User '{username}' not found.")
        return 1

    deleted = auth._users_repo.delete_user(username)
    if deleted:
        print(f"User '{username}' deleted.")
        return 0
    print(f"Could not delete user '{username}'.")
    return 1


def _set_role(username: str, role: str) -> int:
    """Update the role for an existing user from the CLI.

    Args:
        username: Account identifier to update.
        role: Role to assign to the existing account.

    Returns:
        Process exit code for the CLI command.
    """
    auth = _new_auth()
    if not _require_db_backend(auth):
        return 1
    user = auth.get_user(username)
    if not user:
        print(f"User '{username}' not found.")
        return 1

    current_role = str(user.get('role') or 'user')
    if current_role == role:
        print(f"User '{username}' already has role '{role}'.")
        return 0

    updated = auth._users_repo.update_user(username, {'role': role})
    if updated:
        print(f"User '{username}' role updated to '{role}'.")
        return 0
    print(f"Could not update role for '{username}'.")
    return 1


def _create_admin(username: str, password: str, email: str | None, allow_weak_password: bool) -> int:
    """Create or update an administrative user from the CLI.

    Args:
        username: Account identifier to create or promote.
        password: Plain-text password value.
        email: Optional email address for the account.
        allow_weak_password: Whether to bypass password-strength checks.

    Returns:
        Process exit code for the CLI command.
    """
    if allow_weak_password:
        os.environ['FL_AUTH_ALLOW_WEAK_PASSWORDS'] = '1'

    auth = _new_auth()
    if not _require_db_backend(auth):
        return 1
    existing = auth.get_user(username)
    if not existing:
        return _create_user(
            username=username,
            password=password,
            role='admin',
            email=email,
            allow_weak_password=allow_weak_password,
        )

    role_code = _set_role(username, 'admin')
    if role_code not in (0, 1):
        return role_code

    ok_pwd, msg_pwd = auth.admin_set_password(username, password)
    if not ok_pwd:
        print(f"Could not update admin password for '{username}': {msg_pwd}")
        return 1

    print(f"Admin '{username}' already existed: role ensured and password updated.")
    return 0


def _list_users() -> int:
    """List known user accounts through the CLI.

    Returns:
        Process exit code for the CLI command.
    """
    auth = _new_auth()
    if not _require_db_backend(auth):
        return 1
    users = auth.list_users()
    _print_users(users)
    return 0


def main() -> int:
    """Run the auth user-management CLI entry point.

    Returns:
        Process exit code for the CLI invocation.
    """
    parser = _build_parser()
    args = parser.parse_args()
    _configure_storage_mode(args)

    if args.command == 'create':
        pwd = args.password or getpass.getpass('Password: ')
        return _create_user(
            username=args.username.strip(),
            password=pwd,
            role=args.role,
            email=(args.email.strip() if args.email else None),
            allow_weak_password=bool(args.allow_weak_password),
        )

    if args.command == 'create-admin':
        pwd = args.password or getpass.getpass('Password: ')
        return _create_admin(
            username=args.username.strip(),
            password=pwd,
            email=(args.email.strip() if args.email else None),
            allow_weak_password=bool(args.allow_weak_password),
        )

    if args.command == 'list':
        return _list_users()

    if args.command == 'delete':
        return _delete_user(args.username.strip())

    if args.command == 'set-role':
        return _set_role(args.username.strip(), args.role)

    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
