"""CRUD repository for users table."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from database.connection import MariaDBConnectionFactory, MariaDBSession
from database.interfaces import UsersRepository
from database.models import UserRow
from database.repositories.base import build_update_assignments


class MariaUsersRepository(UsersRepository):
    """MariaDB-backed repository for user records.

    This repository manages records stored in the ``users`` table and maps
    database rows into :class:`UserRow` instances.
    """

    _UPDATABLE_FIELDS = {"email", "password_hash", "role", "active"}

    def __init__(self, factory: MariaDBConnectionFactory) -> None:
        """Initialize the repository.

        Args:
            factory: Connection factory used to create MariaDB sessions.
        """
        self._factory = factory

    @staticmethod
    def _map(row: dict[str, Any]) -> UserRow:
        """Map a raw database row to a :class:`UserRow`.

        Args:
            row: Database row returned by the MariaDB session layer.

        Returns:
            A populated user row model.
        """
        return UserRow(
            id=int(row["id"]),
            identifier=str(row["identifier"]),
            email=row.get("email"),
            password_hash=str(row["password_hash"]),
            role=str(row["role"]),
            active=bool(row["active"]),
            created_at=row["created_at"],
            created_by=str(row["created_by"]),
        )

    def create_user(self, payload: dict[str, Any]) -> int:
        """Create a new user record.

        Args:
            payload: User payload containing the values required for insertion.

        Returns:
            The database identifier of the newly created user record.
        """
        query = (
            "INSERT INTO users (identifier, email, password_hash, role, active, created_at, created_by) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        now = datetime.utcnow()

        with MariaDBSession(self._factory) as db:
            return db.execute_returning_id(
                query,
                (
                    payload["identifier"],
                    payload.get("email"),
                    payload["password_hash"],
                    payload.get("role", "user"),
                    bool(payload.get("active", True)),
                    now,
                    payload.get("created_by", "system"),
                ),
            )

    def get_user_by_identifier(self, identifier: str) -> UserRow | None:
        """Fetch a single user by identifier.

        Args:
            identifier: Unique user identifier.

        Returns:
            The matching user row if found, otherwise ``None``.
        """
        query = "SELECT * FROM users WHERE identifier = %s LIMIT 1"

        with MariaDBSession(self._factory) as db:
            row = db.fetchone(query, (identifier,))

        return self._map(row) if row else None

    def get_user_by_email(self, email: str) -> UserRow | None:
        """Fetch a single user by email address.

        Args:
            email: Email address used for lookup.

        Returns:
            The matching user row if found, otherwise ``None``.
        """
        query = "SELECT * FROM users WHERE LOWER(email) = LOWER(%s) LIMIT 1"

        with MariaDBSession(self._factory) as db:
            row = db.fetchone(query, (email,))

        return self._map(row) if row else None

    def list_users(self, include_inactive: bool = True) -> list[UserRow]:
        """List users, optionally excluding inactive accounts.

        Args:
            include_inactive: Whether inactive users should be included in the
                results.

        Returns:
            A list of users ordered by role descending, then identifier
            ascending.
        """
        query = "SELECT * FROM users"
        params: tuple[Any, ...] = ()

        if not include_inactive:
            query += " WHERE active = 1"

        query += " ORDER BY role DESC, identifier ASC"

        with MariaDBSession(self._factory) as db:
            rows = db.fetchall(query, params)

        return [self._map(row) for row in rows]

    def update_user(self, identifier: str, updates: dict[str, Any]) -> bool:
        """Update selected fields for a user.

        Args:
            identifier: Unique user identifier.
            updates: Mapping of fields to update.

        Returns:
            ``True`` if at least one row was updated, otherwise ``False``.
        """
        filtered_updates, set_sql = build_update_assignments(updates, self._UPDATABLE_FIELDS)
        if not filtered_updates:
            return False

        query = f"UPDATE users SET {set_sql} WHERE identifier = %s"

        with MariaDBSession(self._factory) as db:
            changed = db.execute(query, tuple(filtered_updates.values()) + (identifier,))

        return changed > 0

    def delete_user(self, identifier: str) -> bool:
        """Delete a user by identifier.

        Args:
            identifier: Unique user identifier.

        Returns:
            ``True`` if at least one row was deleted, otherwise ``False``.
        """
        query = "DELETE FROM users WHERE identifier = %s"

        with MariaDBSession(self._factory) as db:
            changed = db.execute(query, (identifier,))

        return changed > 0
