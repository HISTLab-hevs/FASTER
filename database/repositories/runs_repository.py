"""CRUD repository for runs and run_configs tables."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from database.connection import MariaDBConnectionFactory, MariaDBSession
from database.interfaces import RunsRepository
from database.models import RunRow
from database.repositories.base import build_update_assignments, parse_json_object


class MariaRunsRepository(RunsRepository):
    """MariaDB-backed repository for runs and run configuration records.

    This repository manages run metadata stored in the ``runs`` table and
    associated configuration payloads stored in the ``run_configs`` table.
    """

    _UPDATABLE_RUN_FIELDS = {
        "display_name",
        "status",
        "method",
        "dataset_name",
        "evaluation_split_mode",
        "model_name",
        "stop_reason",
        "best_accuracy",
    }

    def __init__(self, factory: MariaDBConnectionFactory) -> None:
        """Initialize the repository.

        Args:
            factory: Connection factory used to create MariaDB sessions.
        """
        self._factory = factory

    @staticmethod
    def _map(row: dict[str, Any]) -> RunRow:
        """Map a raw database row to a :class:`RunRow`.

        Args:
            row: Database row returned by the MariaDB session layer.

        Returns:
            A populated run row model.
        """
        return RunRow(
            id=int(row["id"]),
            run_id=str(row["run_id"]),
            owner_identifier=str(row["owner_identifier"]),
            display_name=str(row["display_name"]),
            status=str(row["status"]),
            method=row.get("method"),
            dataset_name=row.get("dataset_name"),
            evaluation_split_mode=row.get("evaluation_split_mode"),
            model_name=row.get("model_name"),
            stop_reason=row.get("stop_reason"),
            best_accuracy=float(row["best_accuracy"]) if row.get("best_accuracy") is not None else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_run(self, payload: dict[str, Any]) -> int:
        """Create a new run and optionally store its configuration.

        Args:
            payload: Run payload containing the values required for insertion.
                When ``config_json`` is present, a corresponding row is also
                inserted into ``run_configs``.

        Returns:
            The database identifier of the newly created run row.
        """
        now = datetime.utcnow()
        run_insert_sql = (
            "INSERT INTO runs (run_id, owner_identifier, display_name, status, method, dataset_name, "
            "evaluation_split_mode, model_name, stop_reason, best_accuracy, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        config_insert_sql = (
            "INSERT INTO run_configs (run_id, config_json, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s)"
        )

        with MariaDBSession(self._factory) as db:
            row_id = db.execute_returning_id(
                run_insert_sql,
                (
                    payload["run_id"],
                    payload["owner_identifier"],
                    payload.get("display_name") or payload["run_id"],
                    payload.get("status", "running"),
                    payload.get("method"),
                    payload.get("dataset_name"),
                    payload.get("evaluation_split_mode"),
                    payload.get("model_name"),
                    payload.get("stop_reason"),
                    payload.get("best_accuracy"),
                    now,
                    now,
                ),
            )

            config = payload.get("config_json")
            if config is not None:
                db.execute(
                    config_insert_sql,
                    (payload["run_id"], json.dumps(config, ensure_ascii=False), now, now),
                )

            return row_id

    def get_run(self, run_id: str) -> RunRow | None:
        """Fetch a single run by run identifier.

        Args:
            run_id: External run identifier.

        Returns:
            The matching run row if found, otherwise ``None``.
        """
        query = "SELECT * FROM runs WHERE run_id = %s LIMIT 1"

        with MariaDBSession(self._factory) as db:
            row = db.fetchone(query, (run_id,))

        return self._map(row) if row else None

    def list_runs(self, owner_identifier: str | None = None) -> list[RunRow]:
        """List runs, optionally filtered by owner.

        Args:
            owner_identifier: Optional owner identifier used to filter results.

        Returns:
            A list of run rows ordered by creation time descending.
        """
        query = "SELECT * FROM runs"
        params: tuple[Any, ...] = ()

        if owner_identifier:
            query += " WHERE owner_identifier = %s"
            params = (owner_identifier,)

        query += " ORDER BY created_at DESC"

        with MariaDBSession(self._factory) as db:
            rows = db.fetchall(query, params)

        return [self._map(row) for row in rows]

    def update_run(self, run_id: str, updates: dict[str, Any]) -> bool:
        """Update selected fields for a run.

        The method also updates the ``updated_at`` timestamp whenever at least
        one allowed field is included.

        Args:
            run_id: External run identifier.
            updates: Mapping of fields to update.

        Returns:
            ``True`` if at least one row was updated, otherwise ``False``.
        """
        filtered_updates, set_sql = build_update_assignments(
            updates,
            self._UPDATABLE_RUN_FIELDS,
        )
        if not filtered_updates:
            return False

        now = datetime.utcnow()
        query = f"UPDATE runs SET {set_sql}, updated_at = %s WHERE run_id = %s"

        with MariaDBSession(self._factory) as db:
            changed = db.execute(query, tuple(filtered_updates.values()) + (now, run_id))

        return changed > 0

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and its associated configuration.

        Args:
            run_id: External run identifier.

        Returns:
            ``True`` if at least one row was deleted from ``runs``, otherwise
            ``False``.
        """
        with MariaDBSession(self._factory) as db:
            db.execute("DELETE FROM run_configs WHERE run_id = %s", (run_id,))
            changed = db.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))

        return changed > 0

    def rename_run(self, old_run_id: str, new_run_id: str) -> bool:
        """Rename a run display label in the database.

        Args:
            old_run_id: Stable external run identifier.
            new_run_id: New display label to store in ``runs.display_name``.

        Returns:
            ``True`` if the rename was successful, otherwise ``False``.
        """
        now = datetime.utcnow()
        with MariaDBSession(self._factory) as db:
            changed = db.execute(
                "UPDATE runs SET display_name = %s, updated_at = %s WHERE run_id = %s",
                (new_run_id, now, old_run_id),
            )

        return changed > 0

    def get_run_config(self, run_id: str) -> dict[str, Any] | None:
        """Fetch the configuration payload for a run.

        Args:
            run_id: External run identifier.

        Returns:
            The parsed configuration dictionary if available and decodable,
            otherwise ``None``.
        """
        query = "SELECT config_json FROM run_configs WHERE run_id = %s LIMIT 1"

        with MariaDBSession(self._factory) as db:
            row = db.fetchone(query, (run_id,))

        if not row:
            return None

        return parse_json_object(row.get("config_json"))

    def upsert_run_config(self, run_id: str, config_json: dict[str, Any]) -> bool:
        """Insert or update a run configuration payload.

        Args:
            run_id: External run identifier.
            config_json: Configuration payload to store.

        Returns:
            ``True`` when the statement executes without a negative affected-row
            count, preserving the current repository contract.
        """
        now = datetime.utcnow()
        query = (
            "INSERT INTO run_configs (run_id, config_json, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE config_json = VALUES(config_json), updated_at = VALUES(updated_at)"
        )

        with MariaDBSession(self._factory) as db:
            changed = db.execute(query, (run_id, json.dumps(config_json, ensure_ascii=False), now, now))

        return changed >= 0
