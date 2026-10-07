"""Content service for user-editable defaults and scenarios.

This adapter groups persisted configuration content behind one narrow
interface so ``TrainingServiceLayer`` does not couple directly to multiple
content-oriented domains.
"""

from __future__ import annotations

from typing import Any

from web_backend.services.defaults import DefaultsService
from web_backend.services.scenarios import ScenariosService


class TrainingContentService:
    """Compose defaults and scenarios behavior behind one small interface."""

    def __init__(
        self,
        *,
        defaults_service: DefaultsService,
        scenarios_service: ScenariosService,
    ) -> None:
        """Initialize the content service adapter.

        Args:
            defaults_service: Service handling saved/default configuration values.
            scenarios_service: Service handling built-in and user scenarios.
        """
        self._defaults_service = defaults_service
        self._scenarios_service = scenarios_service

    def list_scenarios(self, username: str | None = None) -> list[str]:
        """List base and user scenarios.

        Args:
            username: Optional user identifier for user-scoped scenarios.

        Returns:
            Scenario references visible to the caller.
        """
        return self._scenarios_service.list_scenarios(username=username)

    def load_scenario(
        self,
        name: str,
        username: str | None = None,
    ) -> dict[str, Any] | None:
        """Load a scenario configuration.

        Args:
            name: Scenario reference to load.
            username: Optional user identifier for user-scoped lookups.

        Returns:
            Scenario configuration mapping when found, otherwise ``None``.
        """
        return self._scenarios_service.load_scenario(name, username=username)

    def save_user_scenario(
        self,
        username: str,
        config: dict[str, Any],
        name: str = "",
    ) -> str:
        """Save or update a user scenario.

        Args:
            username: Identifier of the scenario owner.
            config: Scenario configuration mapping to persist.
            name: Optional requested scenario name.

        Returns:
            Stored scenario reference.
        """
        return self._scenarios_service.save_user_scenario(username, config, name=name)

    def delete_user_scenario(self, username: str, name: str) -> tuple[bool, str]:
        """Delete one user-owned scenario.

        Args:
            username: Identifier of the scenario owner.
            name: Scenario reference to delete.

        Returns:
            Tuple ``(deleted, message)`` describing the outcome.
        """
        return self._scenarios_service.delete_user_scenario(username, name)

    def load_defaults(self, username: str | None = None) -> dict[str, Any]:
        """Load default configuration values for one user-visible session.

        Args:
            username: Optional user identifier for user-scoped defaults.

        Returns:
            Effective defaults mapping for the caller scope.
        """
        return self._defaults_service.load_defaults(username=username)

    def save_defaults(
        self,
        config: dict[str, Any],
        username: str | None = None,
        updated_by: str | None = None,
    ) -> bool:
        """Persist default configuration values.

        Args:
            config: Default configuration mapping to persist.
            username: Optional user identifier for user-scoped defaults.
            updated_by: Optional actor identifier recorded with the change.

        Returns:
            ``True`` when the defaults are persisted successfully.
        """
        return self._defaults_service.save_defaults(
            config,
            username=username,
            updated_by=updated_by,
        )

    @staticmethod
    def sanitize_scenario_name(name: str) -> str:
        """Normalize a scenario file name.

        Args:
            name: Requested scenario name.

        Returns:
            Filesystem-safe scenario filename.
        """
        return ScenariosService.sanitize_scenario_name(name)
