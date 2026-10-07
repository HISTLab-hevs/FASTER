"""MariaDB repository for uploaded custom dataset metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from database.connection import MariaDBConnectionFactory, MariaDBSession
from database.interfaces import CustomDatasetsRepository
from database.models import CustomDatasetRow
from database.repositories.base import build_update_assignments


class MariaCustomDatasetsRepository(CustomDatasetsRepository):
    """MariaDB-backed repository for custom dataset metadata.

    This repository manages metadata for user-uploaded custom datasets stored in
    the ``custom_datasets`` table and maps database rows into
    :class:`CustomDatasetRow` instances.
    """

    _UPDATABLE_FIELDS = {
        "dataset_name",
        "dataset_type",
        "storage_uri",
        "rows_count",
        "features_count",
        "classes_count",
    }

    def __init__(self, factory: MariaDBConnectionFactory) -> None:
        """Initialize the repository.

        Args:
            factory: Connection factory used to create MariaDB sessions.
        """
        self._factory = factory

    @staticmethod
    def _map(row: dict[str, Any]) -> CustomDatasetRow:
        """Map a raw database row to a :class:`CustomDatasetRow`.

        Args:
            row: Database row returned by the MariaDB session layer.

        Returns:
            A populated custom dataset row model.
        """
        return CustomDatasetRow(
            id=int(row["id"]),
            owner_identifier=str(row["owner_identifier"]),
            dataset_ref=str(row["dataset_ref"]),
            dataset_name=str(row["dataset_name"]),
            dataset_type=str(row["dataset_type"]),
            storage_uri=str(row["storage_uri"]),
            rows_count=row.get("rows_count"),
            features_count=row.get("features_count"),
            classes_count=row.get("classes_count"),
            created_at=row["created_at"],
        )

    def create_dataset(self, payload: dict[str, Any]) -> int:
        """Create a new custom dataset metadata record.

        Args:
            payload: Dataset payload containing the values required for
                insertion.

        Returns:
            The database identifier of the newly created dataset record.
        """
        now = datetime.utcnow()
        query = (
            "INSERT INTO custom_datasets "
            "(owner_identifier, dataset_ref, dataset_name, dataset_type, storage_uri, "
            "rows_count, features_count, classes_count, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )

        with MariaDBSession(self._factory) as db:
            return db.execute_returning_id(
                query,
                (
                    payload["owner_identifier"],
                    payload["dataset_ref"],
                    payload["dataset_name"],
                    payload["dataset_type"],
                    payload["storage_uri"],
                    payload.get("rows_count"),
                    payload.get("features_count"),
                    payload.get("classes_count"),
                    now,
                ),
            )

    def get_dataset(self, owner_identifier: str | None, dataset_ref: str) -> CustomDatasetRow | None:
        """Fetch a single dataset by dataset reference and optionally owner.

        Args:
            owner_identifier: Identifier of the dataset owner. If None, ignores owner.
            dataset_ref: External dataset reference key.

        Returns:
            The matching dataset row if found, otherwise ``None``.
        """
        query = "SELECT * FROM custom_datasets WHERE dataset_ref = %s"
        params = [dataset_ref]

        if owner_identifier is not None:
            query += " AND owner_identifier = %s"
            params.append(owner_identifier)

        query += " LIMIT 1"

        with MariaDBSession(self._factory) as db:
            row = db.fetchone(query, tuple(params))

        return self._map(row) if row else None

    def list_datasets(self, owner_identifier: str | None) -> list[CustomDatasetRow]:
        """List datasets, optionally filtered by owner.

        Args:
            owner_identifier: Identifier of the dataset owner, or None for all.

        Returns:
            A list of dataset rows ordered by creation time descending.
        """
        query = "SELECT * FROM custom_datasets"
        params = []

        if owner_identifier is not None:
            query += " WHERE owner_identifier = %s"
            params.append(owner_identifier)

        query += " ORDER BY created_at DESC"

        with MariaDBSession(self._factory) as db:
            rows = db.fetchall(query, tuple(params))

        return [self._map(row) for row in rows]

    def update_dataset(self, owner_identifier: str, dataset_ref: str, updates: dict[str, Any]) -> bool:
        """Update selected fields for a dataset.

        Only fields explicitly allowed by the repository are included in the
        generated update statement.

        Args:
            owner_identifier: Identifier of the dataset owner.
            dataset_ref: External dataset reference key.
            updates: Mapping of fields to update.

        Returns:
            ``True`` if at least one row was updated, otherwise ``False``.
        """
        filtered_updates, set_sql = build_update_assignments(updates, self._UPDATABLE_FIELDS)
        if not filtered_updates:
            return False

        query = (
            f"UPDATE custom_datasets SET {set_sql} "
            "WHERE owner_identifier = %s AND dataset_ref = %s"
        )

        with MariaDBSession(self._factory) as db:
            changed = db.execute(
                query,
                tuple(filtered_updates.values()) + (owner_identifier, dataset_ref),
            )

        return changed > 0

    def delete_dataset(self, owner_identifier: str, dataset_ref: str) -> bool:
        """Delete a dataset by owner and dataset reference.

        Args:
            owner_identifier: Identifier of the dataset owner.
            dataset_ref: External dataset reference key.

        Returns:
            ``True`` if at least one row was deleted, otherwise ``False``.
        """
        query = "DELETE FROM custom_datasets WHERE owner_identifier = %s AND dataset_ref = %s"

        with MariaDBSession(self._factory) as db:
            changed = db.execute(query, (owner_identifier, dataset_ref))

        return changed > 0

    def upsert_dataset(self, payload: dict[str, Any]) -> bool:
        """Insert or update a dataset metadata record.

        If a row with the same unique key already exists, selected fields are
        updated in place using MariaDB's ``ON DUPLICATE KEY UPDATE`` behavior.

        Args:
            payload: Dataset payload containing the values required for insert or
                update.

        Returns:
            ``True`` if the statement affected at least one row, otherwise
            ``False``.
        """
        query = (
            "INSERT INTO custom_datasets "
            "(owner_identifier, dataset_ref, dataset_name, dataset_type, storage_uri, "
            "rows_count, features_count, classes_count, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, UTC_TIMESTAMP(6)) "
            "ON DUPLICATE KEY UPDATE "
            "dataset_name = VALUES(dataset_name), "
            "dataset_type = VALUES(dataset_type), "
            "storage_uri = VALUES(storage_uri), "
            "rows_count = VALUES(rows_count), "
            "features_count = VALUES(features_count), "
            "classes_count = VALUES(classes_count)"
        )

        with MariaDBSession(self._factory) as db:
            changed = db.execute(
                query,
                (
                    payload["owner_identifier"],
                    payload["dataset_ref"],
                    payload["dataset_name"],
                    payload["dataset_type"],
                    payload["storage_uri"],
                    payload.get("rows_count"),
                    payload.get("features_count"),
                    payload.get("classes_count"),
                ),
            )

        return changed > 0
