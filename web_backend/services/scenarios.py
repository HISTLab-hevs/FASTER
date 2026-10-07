"""Scenario persistence service for built-in presets and DB-backed user presets."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Callable

import yaml


logger = logging.getLogger("web_backend.service")


class ScenariosService:
    """Load, list, and save user scenarios behind the service facade."""

    def __init__(
        self,
        *,
        scenarios_dir: str,
        load_config_fn: Callable[[str], dict[str, Any]],
        db_enabled_getter: Callable[[], bool],
        scenarios_repo_getter: Callable[[], Any],
    ) -> None:
        """Initialize the scenarios service.

        Args:
            scenarios_dir: Root directory containing built-in scenario files.
            load_config_fn: Callable used to load YAML scenario files.
            db_enabled_getter: Callable returning whether DB-backed storage is enabled.
            scenarios_repo_getter: Callable returning the scenarios repository.
        """
        self._scenarios_dir = scenarios_dir
        self._load_config = load_config_fn
        self._db_enabled = db_enabled_getter
        self._scenarios_repo = scenarios_repo_getter

    @staticmethod
    def sanitize_scenario_name(name: str) -> str:
        """Normalize a scenario file name.

        Args:
            name: Requested scenario name.

        Returns:
            Filesystem-safe scenario filename.
        """
        base = (name or "").strip()
        if not base:
            base = datetime.now().strftime("scenario_%Y%m%d_%H%M%S")

        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", base)
        if not safe.endswith((".yaml", ".yml")):
            safe += ".yaml"
        return safe

    def list_scenarios(self, username: str | None = None) -> list[str]:
        """List base and user scenarios.

        Args:
            username: Optional user identifier for user-scoped scenarios.

        Returns:
            Scenario references visible to the caller.
        """
        repo = self._scenarios_repo()
        if username and self._db_enabled() and repo is not None:
            try:
                rows = repo.list_scenarios(username)
                user_results = []
                for row in rows or []:
                    try:
                        scenario_name = row.get("scenario_name") if isinstance(row, dict) else row.scenario_name
                    except Exception:
                        continue
                    scenario_name_text = str(scenario_name or "").strip()
                    if scenario_name_text:
                        user_results.append(f"my/{scenario_name_text}")
            except Exception as exc:
                raise RuntimeError(f"Failed to list persisted scenarios for user '{username}'") from exc
        else:
            user_results = []

        if not os.path.exists(self._scenarios_dir):
            return sorted(dict.fromkeys(user_results))

        base_results: list[str] = []
        users_root = os.path.join(self._scenarios_dir, "users")
        for root, _, files in os.walk(self._scenarios_dir):
            if os.path.abspath(root).startswith(os.path.abspath(users_root)):
                continue

            for file_name in files:
                if file_name.endswith((".yaml", ".yml")):
                    rel = os.path.relpath(os.path.join(root, file_name), self._scenarios_dir)
                    base_results.append(rel)

        return sorted(dict.fromkeys(user_results)) + sorted(base_results)

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
        name_text = str(name or "").strip()
        repo = self._scenarios_repo()
        if name_text.startswith("my/") and username:
            rel = name_text[3:]

            if self._db_enabled() and repo is not None:
                try:
                    row = repo.get_scenario(username, rel)
                    if row:
                        scenario_yaml = row.get("scenario_yaml") if isinstance(row, dict) else row.scenario_yaml
                        if scenario_yaml:
                            parsed = yaml.safe_load(scenario_yaml) or {}
                            if isinstance(parsed, dict):
                                return parsed
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load persisted scenario '{rel}' for user '{username}'"
                    ) from exc
            return None
        else:
            path = os.path.join(self._scenarios_dir, name_text)

        try:
            return self._load_config(path) if os.path.exists(path) else None
        except Exception as exc:
            logger.warning(
                "Failed to load scenario file '%s': %s",
                path,
                exc,
            )
            return None

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
        file_name = self.sanitize_scenario_name(name)
        scenario_ref = f"my/{file_name}"
        yaml_blob = yaml.safe_dump(
            config or {},
            default_flow_style=False,
            sort_keys=False,
        )

        repo = self._scenarios_repo()
        if not self._db_enabled() or repo is None:
            raise RuntimeError("DB-backed scenario persistence is required")
        try:
            updated = repo.update_scenario(
                username,
                file_name,
                {"scenario_yaml": yaml_blob},
            )
            if not updated:
                repo.create_scenario(
                    {
                        "owner_identifier": username,
                        "scenario_name": file_name,
                        "scenario_yaml": yaml_blob,
                    }
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to persist scenario '{scenario_ref}'") from exc
        return scenario_ref

    def delete_user_scenario(self, username: str, name: str) -> tuple[bool, str]:
        """Delete one DB-backed user-owned scenario.

        Args:
            username: Identifier of the scenario owner.
            name: Scenario reference to delete.

        Returns:
            Tuple ``(deleted, message)`` describing the outcome.
        """
        scenario_name = str(name or "").strip()
        if not username:
            return False, "Scenario owner is required"
        if not scenario_name.startswith("my/"):
            return False, "Only user scenarios can be deleted"

        rel = scenario_name[3:].strip()
        if not rel or rel.startswith("../") or "/../" in rel:
            return False, "Invalid scenario reference"

        repo = self._scenarios_repo()
        if not self._db_enabled() or repo is None:
            return False, "DB-backed scenario persistence is required"
        try:
            deleted = bool(repo.delete_scenario(username, rel))
        except Exception as exc:
            raise RuntimeError(f"Failed to delete persisted scenario '{rel}' for user '{username}'") from exc

        if deleted:
            return True, f"Deleted scenario 'my/{rel}'"
        return False, "Scenario not found"
