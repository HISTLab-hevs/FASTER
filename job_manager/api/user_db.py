"""User database operations for the shared MariaDB users table."""

from datetime import datetime, timezone
import bcrypt
import pymysql

from api.db import connection


def init_users_db() -> None:
    """Verify that the ``users`` table is available.

    Side Effects:
        Executes a lightweight read against ``users`` so startup fails clearly
        when ``database/schema/baseline.sql`` has not initialized MariaDB.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users LIMIT 1")


def _hash_password(password: str) -> str:
    """Hash a plain-text password with bcrypt.

    Args:
        password: The plain-text password to hash.

    Returns:
        The bcrypt hash as a UTF-8 string.
    """
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def create_user(username: str, password: str, is_admin: bool = False) -> bool:
    """Insert a new user with a hashed password.

    Args:
        username: Desired login name (must be unique).
        password: Plain-text password (will be hashed before storage).
        is_admin: Whether to grant admin privileges.

    Returns:
        ``True`` if the user was created, ``False`` if the username
        already exists.
    """
    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (identifier, email, password_hash, role, active, created_at, created_by)
                    VALUES (%s, %s, %s, %s, 1, %s, %s)
                    """,
                    (
                        username,
                        None,
                        _hash_password(password),
                        "admin" if is_admin else "user",
                        datetime.now(timezone.utc),
                        "job_manager",
                    ),
                )
        return True
    except pymysql.err.IntegrityError:
        return False


def verify_user(username: str, password: str) -> bool:
    """Verify credentials against the database.

    Args:
        username: The login name to look up.
        password: The plain-text password to check.

    Returns:
        ``True`` if the credentials are valid, ``False`` otherwise.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash FROM users WHERE identifier = %s AND active = 1",
                (username,),
            )
            row = cur.fetchone()
            if row is None:
                return False
            return bcrypt.checkpw(
                password.encode("utf-8"),
                row["password_hash"].encode("utf-8"),
            )


def is_admin(username: str) -> bool:
    """Check whether a user has admin privileges.

    Args:
        username: The login name to check.

    Returns:
        ``True`` if the user exists and has ``is_admin = 1``.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role FROM users WHERE identifier = %s AND active = 1",
                (username,),
            )
            row = cur.fetchone()
            return bool(row and str(row.get("role") or "") == "admin")


def delete_user(username: str) -> bool:
    """Remove a user by username.

    Args:
        username: The login name to delete.

    Returns:
        ``True`` if a row was deleted, ``False`` if the user was
        not found.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE identifier = %s", (username,))
            return cur.rowcount > 0


def list_users() -> list[dict]:
    """Return all users without password hashes.

    Returns:
        A list of dicts with keys ``id``, ``username``, ``is_admin``,
        and ``created_at``.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    identifier AS username,
                    CASE WHEN role = 'admin' THEN 1 ELSE 0 END AS is_admin,
                    created_at
                FROM users
                WHERE active = 1
                ORDER BY id
                """
            )
            return cur.fetchall()
