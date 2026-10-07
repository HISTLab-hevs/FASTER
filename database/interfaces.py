"""Abstract repository contracts for backend persistence.

These interfaces define the storage boundary used by the service and auth
layers. They intentionally describe data access only; schema ownership belongs
to the shared MariaDB baseline rather than to repository callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import AppSettingRow, AlertRow, CustomDatasetRow, RunRow, ScenarioRow, UserRow


class AppSettingsRepository(ABC):
    """Persist backend-wide and user-scoped application settings by key."""

    @abstractmethod
    def get_setting(self, setting_key: str) -> AppSettingRow | None:
        """Fetch one persisted setting by key.

        Args:
            setting_key: Unique application setting key.

        Returns:
            The matching application-setting row if found, otherwise ``None``.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert_setting(
        self,
        setting_key: str,
        setting_json: dict[str, Any],
        updated_by: str | None = None,
    ) -> bool:
        """Create or replace one persisted setting payload.

        Args:
            setting_key: Unique application setting key.
            setting_json: JSON-serializable payload to persist.
            updated_by: Optional identifier of the actor changing the setting.

        Returns:
            ``True`` if the payload was inserted or updated, otherwise
            ``False``.
        """
        raise NotImplementedError


class UsersRepository(ABC):
    """Persist and query user/account records for authentication flows."""

    @abstractmethod
    def create_user(self, payload: dict[str, Any]) -> int:
        """Create a new user record.

        Args:
            payload: User payload containing the values required for creation.

        Returns:
            The identifier of the newly created user record.
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_by_identifier(self, identifier: str) -> UserRow | None:
        """Fetch a single user by identifier.

        Args:
            identifier: Unique user identifier.

        Returns:
            The matching user row if found, otherwise ``None``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_by_email(self, email: str) -> UserRow | None:
        """Fetch a single user by normalized email address.

        Args:
            email: Email address used for lookup.

        Returns:
            The matching user row if found, otherwise ``None``.
        """
        raise NotImplementedError

    @abstractmethod
    def list_users(self, include_inactive: bool = True) -> list[UserRow]:
        """List users.

        Args:
            include_inactive: Whether inactive users should be included in the
                result set.

        Returns:
            A list of user rows.
        """
        raise NotImplementedError

    @abstractmethod
    def update_user(self, identifier: str, updates: dict[str, Any]) -> bool:
        """Update selected fields for a user.

        Args:
            identifier: Unique user identifier.
            updates: Mapping of fields to update.

        Returns:
            ``True`` if a user was updated, otherwise ``False``.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_user(self, identifier: str) -> bool:
        """Delete a user by identifier.

        Args:
            identifier: Unique user identifier.

        Returns:
            ``True`` if a user was deleted, otherwise ``False``.
        """
        raise NotImplementedError


class RunsRepository(ABC):
    """Persist run metadata and stored run configuration payloads."""

    @abstractmethod
    def create_run(self, payload: dict[str, Any]) -> int:
        """Create a new run record.

        Args:
            payload: Run payload containing the values required for creation.

        Returns:
            The identifier of the newly created run record.
        """
        raise NotImplementedError

    @abstractmethod
    def get_run(self, run_id: str) -> RunRow | None:
        """Fetch a single run by run identifier.

        Args:
            run_id: Run identifier.

        Returns:
            The matching run row if found, otherwise ``None``.
        """
        raise NotImplementedError

    @abstractmethod
    def list_runs(self, owner_identifier: str | None = None) -> list[RunRow]:
        """List runs, optionally filtered by owner.

        Args:
            owner_identifier: Optional owner user identifier used to filter
                the result set.

        Returns:
            A list of run rows.
        """
        raise NotImplementedError

    @abstractmethod
    def update_run(self, run_id: str, updates: dict[str, Any]) -> bool:
        """Update selected fields for a run.

        Args:
            run_id: Run identifier.
            updates: Mapping of fields to update.

        Returns:
            ``True`` if a run was updated, otherwise ``False``.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_run(self, run_id: str) -> bool:
        """Delete a run by identifier.

        Args:
            run_id: Run identifier.

        Returns:
            ``True`` if a run was deleted, otherwise ``False``.
        """
        raise NotImplementedError

    @abstractmethod
    def rename_run(self, old_run_id: str, new_run_id: str) -> bool:
        """Rename a run by identifier.

        Args:
            old_run_id: Current run identifier.
            new_run_id: New run identifier.

        Returns:
            ``True`` if a run was renamed, otherwise ``False``.
        """
        raise NotImplementedError


class ScenariosRepository(ABC):
    """Abstract interface for persisted scenario data access operations."""

    @abstractmethod
    def create_scenario(self, payload: dict[str, Any]) -> int:
        """Create a new scenario record.

        Args:
            payload: Scenario payload containing the values required for
                creation.

        Returns:
            The identifier of the newly created scenario record.
        """
        raise NotImplementedError

    @abstractmethod
    def get_scenario(self, owner_identifier: str, scenario_name: str) -> ScenarioRow | None:
        """Fetch a single scenario by owner and scenario name.

        Args:
            owner_identifier: Owner user identifier.
            scenario_name: Scenario name.

        Returns:
            The matching scenario row if found, otherwise ``None``.
        """
        raise NotImplementedError

    @abstractmethod
    def list_scenarios(self, owner_identifier: str) -> list[ScenarioRow]:
        """List scenarios for a given owner.

        Args:
            owner_identifier: Owner user identifier.

        Returns:
            A list of scenario rows.
        """
        raise NotImplementedError

    @abstractmethod
    def update_scenario(
        self,
        owner_identifier: str,
        scenario_name: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update selected fields for a scenario.

        Args:
            owner_identifier: Owner user identifier.
            scenario_name: Scenario name.
            updates: Mapping of fields to update.

        Returns:
            ``True`` if a scenario was updated, otherwise ``False``.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_scenario(self, owner_identifier: str, scenario_name: str) -> bool:
        """Delete a scenario by owner and scenario name.

        Args:
            owner_identifier: Owner user identifier.
            scenario_name: Scenario name.

        Returns:
            ``True`` if a scenario was deleted, otherwise ``False``.
        """
        raise NotImplementedError


class AlertsRepository(ABC):
    """Abstract interface for alert data access operations."""

    @abstractmethod
    def create_alert(self, payload: dict[str, Any]) -> int:
        """Create a new alert record.

        Args:
            payload: Alert payload containing the values required for creation.

        Returns:
            The identifier of the newly created alert record.
        """
        raise NotImplementedError

    @abstractmethod
    def list_alerts(self, owner_identifier: str, consumed: bool | None = None) -> list[AlertRow]:
        """List alerts for a given owner.

        Args:
            owner_identifier: Owner user identifier.
            consumed: Optional consumed-state filter. When ``None``, alerts are
                returned regardless of consumption state.

        Returns:
            A list of alert rows.
        """
        raise NotImplementedError

    @abstractmethod
    def mark_consumed(self, alert_ids: list[int]) -> int:
        """Mark one or more alerts as consumed.

        Args:
            alert_ids: Alert identifiers to update.

        Returns:
            The number of affected alert records.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_alert(self, alert_id: int) -> bool:
        """Delete an alert by identifier.

        Args:
            alert_id: Alert identifier.

        Returns:
            ``True`` if an alert was deleted, otherwise ``False``.
        """
        raise NotImplementedError


class CustomDatasetsRepository(ABC):
    """Abstract interface for custom dataset metadata access operations."""

    @abstractmethod
    def create_dataset(self, payload: dict[str, Any]) -> int:
        """Create a new custom dataset metadata record.

        Args:
            payload: Dataset payload containing the values required for
                creation.

        Returns:
            The identifier of the newly created dataset record.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dataset(self, owner_identifier: str | None, dataset_ref: str) -> CustomDatasetRow | None:
        """Fetch a single dataset by dataset reference and optionally owner.

        Args:
            owner_identifier: Owner user identifier. If None, fetch globally.
            dataset_ref: Dataset reference key.

        Returns:
            The matching dataset row if found, otherwise ``None``.
        """
        raise NotImplementedError

    @abstractmethod
    def list_datasets(self, owner_identifier: str | None) -> list[CustomDatasetRow]:
        """List datasets, optionally filtered by owner.

        Args:
            owner_identifier: Owner user identifier. If None, fetch all datasets.

        Returns:
            A list of dataset rows.
        """
        raise NotImplementedError

    @abstractmethod
    def update_dataset(
        self,
        owner_identifier: str,
        dataset_ref: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update selected fields for a dataset.

        Args:
            owner_identifier: Owner user identifier.
            dataset_ref: Dataset reference key.
            updates: Mapping of fields to update.

        Returns:
            ``True`` if a dataset was updated, otherwise ``False``.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_dataset(self, owner_identifier: str, dataset_ref: str) -> bool:
        """Delete a dataset by owner and dataset reference.

        Args:
            owner_identifier: Owner user identifier.
            dataset_ref: Dataset reference key.

        Returns:
            ``True`` if a dataset was deleted, otherwise ``False``.
        """
        raise NotImplementedError
