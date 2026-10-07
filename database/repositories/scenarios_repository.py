"""CRUD repository for persisted user scenarios."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from database.connection import MariaDBConnectionFactory, MariaDBSession
from database.interfaces import ScenariosRepository
from database.models import ScenarioRow


class MariaScenariosRepository(ScenariosRepository):
    """MariaDB-backed repository for persisted user scenarios.

    This repository manages scenario records stored in the ``scenarios`` table
    and maps database rows into :class:`ScenarioRow` instances.
    """

    def __init__(self, factory: MariaDBConnectionFactory) -> None:
        """Initialize the repository.

        Args:
            factory: Connection factory used to create MariaDB sessions.
        """
        self._factory = factory

    @staticmethod
    def _map(row: dict[str, Any]) -> ScenarioRow:
        """Map a raw database row to a :class:`ScenarioRow`.

        Args:
            row: Database row returned by the MariaDB session layer.

        Returns:
            A populated scenario row model.
        """
        return ScenarioRow(
            id=int(row["id"]),
            owner_identifier=str(row["owner_identifier"]),
            scenario_name=str(row["scenario_name"]),
            scenario_yaml=str(row["scenario_yaml"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_scenario(self, payload: dict[str, Any]) -> int:
        """Create a new persisted scenario.

        Args:
            payload: Scenario payload containing the values required for
                insertion.

        Returns:
            The database identifier of the newly created scenario record.
        """
        now = datetime.utcnow()
        query = (
            "INSERT INTO scenarios (owner_identifier, scenario_name, scenario_yaml, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s)"
        )

        with MariaDBSession(self._factory) as db:
            return db.execute_returning_id(
                query,
                (
                    payload["owner_identifier"],
                    payload["scenario_name"],
                    payload["scenario_yaml"],
                    now,
                    now,
                ),
            )

    def get_scenario(self, owner_identifier: str, scenario_name: str) -> ScenarioRow | None:
        """Fetch a single scenario by owner and scenario name.

        Args:
            owner_identifier: Identifier of the scenario owner.
            scenario_name: Scenario name.

        Returns:
            The matching scenario row if found, otherwise ``None``.
        """
        query = (
            "SELECT * FROM scenarios "
            "WHERE owner_identifier = %s AND scenario_name = %s "
            "LIMIT 1"
        )

        with MariaDBSession(self._factory) as db:
            row = db.fetchone(query, (owner_identifier, scenario_name))

        return self._map(row) if row else None

    def list_scenarios(self, owner_identifier: str) -> list[ScenarioRow]:
        """List scenarios for a given owner.

        Args:
            owner_identifier: Identifier of the scenario owner.

        Returns:
            A list of scenario rows ordered by most recently updated first.
        """
        query = (
            "SELECT * FROM scenarios "
            "WHERE owner_identifier = %s "
            "ORDER BY updated_at DESC"
        )

        with MariaDBSession(self._factory) as db:
            rows = db.fetchall(query, (owner_identifier,))

        return [self._map(row) for row in rows]

    def update_scenario(self, owner_identifier: str, scenario_name: str, updates: dict[str, Any]) -> bool:
        """Update the YAML content of a scenario.

        This method preserves the current repository behavior by only applying
        updates when ``scenario_yaml`` is present in the update mapping.

        Args:
            owner_identifier: Identifier of the scenario owner.
            scenario_name: Scenario name.
            updates: Mapping of fields to update.

        Returns:
            ``True`` if at least one row was updated, otherwise ``False``.
        """
        if "scenario_yaml" not in updates:
            return False

        query = (
            "UPDATE scenarios SET scenario_yaml = %s, updated_at = %s "
            "WHERE owner_identifier = %s AND scenario_name = %s"
        )

        with MariaDBSession(self._factory) as db:
            changed = db.execute(
                query,
                (updates["scenario_yaml"], datetime.utcnow(), owner_identifier, scenario_name),
            )

        return changed > 0

    def delete_scenario(self, owner_identifier: str, scenario_name: str) -> bool:
        """Delete a scenario by owner and scenario name.

        Args:
            owner_identifier: Identifier of the scenario owner.
            scenario_name: Scenario name.

        Returns:
            ``True`` if at least one row was deleted, otherwise ``False``.
        """
        query = "DELETE FROM scenarios WHERE owner_identifier = %s AND scenario_name = %s"

        with MariaDBSession(self._factory) as db:
            changed = db.execute(query, (owner_identifier, scenario_name))

        return changed > 0