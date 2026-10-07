"""CRUD repository for persisted application settings."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from database.connection import MariaDBConnectionFactory, MariaDBSession
from database.interfaces import AppSettingsRepository
from database.models import AppSettingRow
from database.repositories.base import parse_json_object_or_empty


class MariaAppSettingsRepository(AppSettingsRepository):
    """MariaDB-backed repository for persisted application settings."""

    def __init__(self, factory: MariaDBConnectionFactory) -> None:
        """Initialize the repository.

        Args:
            factory: Connection factory used to create MariaDB sessions.
        """
        self._factory = factory

    @staticmethod
    def _map(row: dict[str, Any]) -> AppSettingRow:
        """Map a raw database row to an :class:`AppSettingRow`.

        Args:
            row: Database row returned by the MariaDB session layer.

        Returns:
            A populated application-setting row model. Malformed JSON payloads
            are represented as an empty dictionary.
        """
        return AppSettingRow(
            id=int(row["id"]),
            setting_key=str(row["setting_key"]),
            setting_json=parse_json_object_or_empty(row.get("setting_json")),
            updated_by=str(row["updated_by"]) if row.get("updated_by") is not None else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_setting(self, setting_key: str) -> AppSettingRow | None:
        """Fetch a single application setting by key.

        Args:
            setting_key: Unique application setting key.

        Returns:
            The matching setting row if found, otherwise ``None``.
        """
        query = "SELECT * FROM app_settings WHERE setting_key = %s LIMIT 1"

        with MariaDBSession(self._factory) as db:
            row = db.fetchone(query, (setting_key,))

        return self._map(row) if row else None

    def upsert_setting(
        self,
        setting_key: str,
        setting_json: dict[str, Any],
        updated_by: str | None = None,
    ) -> bool:
        """Insert or update an application setting payload.

        Args:
            setting_key: Unique application setting key.
            setting_json: JSON-serializable payload to persist.
            updated_by: Optional identifier of the user or process changing the
                setting.

        Returns:
            ``True`` if MariaDB reports at least one affected row, otherwise
            ``False``.
        """
        now = datetime.now(UTC).replace(tzinfo=None)
        query = (
            "INSERT INTO app_settings (setting_key, setting_json, updated_by, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE setting_json = VALUES(setting_json), "
            "updated_by = VALUES(updated_by), updated_at = VALUES(updated_at)"
        )

        with MariaDBSession(self._factory) as db:
            changed = db.execute(
                query,
                (
                    setting_key,
                    json.dumps(setting_json, ensure_ascii=False),
                    updated_by,
                    now,
                    now,
                ),
            )

        return changed > 0
