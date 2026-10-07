"""CRUD repository for alert feed."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from database.connection import MariaDBConnectionFactory, MariaDBSession
from database.interfaces import AlertsRepository
from database.models import AlertRow


class MariaAlertsRepository(AlertsRepository):
    """MariaDB-backed repository for alert records.

    This repository provides basic CRUD-style operations for alerts stored in the
    ``alerts`` table and maps database rows into :class:`AlertRow` instances.
    """

    def __init__(self, factory: MariaDBConnectionFactory) -> None:
        """Initialize the repository.

        Args:
            factory: Connection factory used to create MariaDB sessions.
        """
        self._factory = factory

    @staticmethod
    def _map(row: dict[str, Any]) -> AlertRow:
        """Map a raw database row to an :class:`AlertRow`.

        Args:
            row: Database row returned by the MariaDB session layer.

        Returns:
            A populated alert row model instance.
        """
        return AlertRow(
            id=int(row["id"]),
            owner_identifier=str(row["owner_identifier"]),
            level=str(row["level"]),
            message=str(row["message"]),
            consumed=bool(row["consumed"]),
            created_at=row["created_at"],
        )

    def create_alert(self, payload: dict[str, Any]) -> int:
        """Create a new alert record.

        Args:
            payload: Alert payload containing the fields required for insertion.
                Expected keys include ``owner_identifier`` and ``message``.
                Optional keys include ``level`` and ``consumed``.

        Returns:
            The database identifier of the newly created alert.
        """
        query = (
            "INSERT INTO alerts (owner_identifier, level, message, consumed, created_at) "
            "VALUES (%s, %s, %s, %s, %s)"
        )

        with MariaDBSession(self._factory) as db:
            return db.execute_returning_id(
                query,
                (
                    payload["owner_identifier"],
                    payload.get("level", "info"),
                    payload["message"],
                    bool(payload.get("consumed", False)),
                    datetime.utcnow(),
                ),
            )

    def list_alerts(self, owner_identifier: str, consumed: bool | None = None) -> list[AlertRow]:
        """List alerts for a given owner.

        Args:
            owner_identifier: Identifier of the alert owner.
            consumed: Optional filter for consumed status. When ``None``, alerts
                are returned regardless of consumption state.

        Returns:
            A list of alerts ordered by creation time descending.
        """
        query = "SELECT * FROM alerts WHERE owner_identifier = %s"
        params: list[Any] = [owner_identifier]

        if consumed is not None:
            query += " AND consumed = %s"
            params.append(bool(consumed))

        query += " ORDER BY created_at DESC"

        with MariaDBSession(self._factory) as db:
            rows = db.fetchall(query, tuple(params))

        return [self._map(row) for row in rows]

    def mark_consumed(self, alert_ids: list[int]) -> int:
        """Mark one or more alerts as consumed.

        Args:
            alert_ids: List of alert identifiers to update.

        Returns:
            The number of rows affected by the update operation.
        """
        normalized_ids = [int(alert_id) for alert_id in (alert_ids or [])]
        if not normalized_ids:
            return 0

        placeholders = ",".join(["%s"] * len(normalized_ids))
        query = f"UPDATE alerts SET consumed = 1 WHERE id IN ({placeholders})"

        with MariaDBSession(self._factory) as db:
            return db.execute(query, tuple(normalized_ids))

    def delete_alert(self, alert_id: int) -> bool:
        """Delete a single alert by identifier.

        Args:
            alert_id: Identifier of the alert to delete.

        Returns:
            ``True`` if at least one row was deleted, otherwise ``False``.
        """
        query = "DELETE FROM alerts WHERE id = %s"

        with MariaDBSession(self._factory) as db:
            changed = db.execute(query, (int(alert_id),))

        return changed > 0