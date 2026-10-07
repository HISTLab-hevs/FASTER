"""Defaults/settings service for bootstrap config and persisted saved defaults.

This module owns the backend logic for:

- resolving bootstrap defaults from ``config.yaml``
- overlaying persisted saved defaults from app settings storage
- saving user-scoped or shared/global defaults
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from utils.config import normalize_run_config_aliases


logger = logging.getLogger("web_backend.service")


class DefaultsService:
    """Resolve and persist backend default configuration values."""

    def __init__(
        self,
        *,
        bootstrap_config_file: str,
        saved_defaults_key: str,
        load_config_fn: Callable[[str], dict[str, Any]],
        db_enabled_getter: Callable[[], bool],
        app_settings_repo_getter: Callable[[], Any],
    ) -> None:
        """Initialize the defaults service.

        Args:
            bootstrap_config_file: Path to the bootstrap defaults file.
            saved_defaults_key: Base app-settings key for persisted defaults.
            load_config_fn: Callable used to load YAML configuration files.
            db_enabled_getter: Callable returning whether DB-backed storage is enabled.
            app_settings_repo_getter: Callable returning the app-settings repository.
        """
        self._bootstrap_config_file = bootstrap_config_file
        self._saved_defaults_key = saved_defaults_key
        self._load_config = load_config_fn
        self._db_enabled = db_enabled_getter
        self._app_settings_repo = app_settings_repo_getter

    def saved_defaults_key(self, username: str | None = None) -> str:
        """Return the app-settings key for saved defaults."""
        return self.build_saved_defaults_key(self._saved_defaults_key, username)

    @staticmethod
    def build_saved_defaults_key(saved_defaults_key: str, username: str | None = None) -> str:
        """Build one saved-defaults app-settings key from a base key and user."""
        normalized = (username or "").strip()
        if not normalized:
            return saved_defaults_key
        return f"{saved_defaults_key}::user::{normalized}"

    def load_defaults(self, username: str | None = None) -> dict[str, Any]:
        """Load default configuration values for one user-visible session."""
        repo = self._app_settings_repo()
        if self._db_enabled() and repo is not None:
            try:
                row = None
                if username:
                    row = repo.get_setting(self.saved_defaults_key(username))
                if row is None:
                    row = repo.get_setting(self._saved_defaults_key)
                if row and isinstance(row.setting_json, dict):
                    base_defaults = self._load_bootstrap_defaults()
                    merged = dict(base_defaults)
                    merged.update(row.setting_json)
                    return normalize_run_config_aliases(merged)
            except Exception as exc:
                logger.warning(
                    "Failed to load persisted defaults for user '%s'; using bootstrap defaults: %s",
                    username or "",
                    exc,
                )

        return normalize_run_config_aliases(self._load_bootstrap_defaults())

    def save_defaults(
        self,
        config: dict[str, Any],
        username: str | None = None,
        updated_by: str | None = None,
    ) -> bool:
        """Persist default configuration values."""
        if not isinstance(config, dict):
            return False

        repo = self._app_settings_repo()
        if self._db_enabled() and repo is not None:
            try:
                normalize_run_config_aliases(config)
                return repo.upsert_setting(
                    self.saved_defaults_key(username),
                    dict(config),
                    updated_by=updated_by,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to persist saved defaults for user '%s': %s",
                    username or "",
                    exc,
                )
                return False

        return False

    def _load_bootstrap_defaults(self) -> dict[str, Any]:
        """Load bootstrap defaults from the main config file."""
        try:
            loaded = self._load_config(self._bootstrap_config_file)
            if not isinstance(loaded, dict):
                raise RuntimeError(f"Bootstrap defaults file '{self._bootstrap_config_file}' must contain a mapping")
            return normalize_run_config_aliases(loaded)
        except Exception as exc:
            raise RuntimeError(f"Failed to load bootstrap defaults from '{self._bootstrap_config_file}'") from exc
