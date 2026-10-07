"""Content-oriented API handlers for defaults, scenarios, and custom datasets.

These commands manage user-editable configuration inputs and stored content.
They do not own run execution; they shape the persisted inputs that later feed
run submission and history views.
"""

from __future__ import annotations

from typing import Any

import yaml

from .shared import error_payload, log_and_mask_exception


def handle_content_command(command: str, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch one defaults/scenario/dataset command or return ``None``."""
    svc = ctx["svc"]
    username = ctx["username"]
    is_admin = ctx["is_admin"]
    logger = ctx["logger"]

    if command == "list_custom_datasets":
        datasets = svc.list_custom_datasets(username, is_admin=is_admin)
        return {"success": True, "datasets": datasets}

    if command == "rename_custom_dataset":
        dataset_ref = str(data.get("dataset_ref") or "")
        new_name = str(data.get("new_name") or "")
        ok, msg = svc.rename_custom_dataset(username, dataset_ref, new_name, is_admin=is_admin)
        if not ok:
            return {"error": msg}
        return {"success": True, "message": msg}

    if command == "update_custom_dataset_description":
        dataset_ref = str(data.get("dataset_ref") or "")
        description = str(data.get("description") or "")
        ok, msg = svc.update_custom_dataset_description(
            username, dataset_ref, description, is_admin=is_admin
        )
        if not ok:
            return {"error": msg}
        return {"success": True, "message": msg}

    if command == "delete_custom_dataset":
        dataset_ref = str(data.get("dataset_ref") or "")
        ok, msg = svc.delete_custom_dataset(username, dataset_ref, is_admin=is_admin)
        if not ok:
            return {"error": msg}
        return {"success": True, "message": msg}

    if command == "get_defaults":
        return {"success": True, "defaults": svc.load_defaults(username=username)}

    if command == "save_defaults":
        ok = svc.save_defaults(data.get("config", {}), username=username, updated_by=username)
        if not ok:
            return {"error": "Failed to save defaults"}
        return {"success": True, "message": "Defaults saved"}

    if command == "list_scenarios":
        return {"success": True, "scenarios": svc.list_scenarios(username)}

    if command == "get_scenario":
        name = data.get("name", "")
        cfg = svc.load_scenario(name, username)
        if cfg is None:
            return {"error": f"Scenario '{name}' not found"}
        return {"success": True, "config": cfg}

    if command == "save_scenario":
        cfg = data.get("config", {})
        name = data.get("name", "")
        if not isinstance(cfg, dict):
            return error_payload("Scenario config must be a mapping")
        try:
            saved = svc.save_user_scenario(username, cfg, name)
        except Exception:
            return log_and_mask_exception(
                logger,
                user_message="Failed to save scenario",
                log_message=f"Unhandled scenario save error for user '{username}'",
            )
        return {
            "success": True,
            "name": saved,
            "message": f"Saved scenario '{saved}'",
        }

    if command == "delete_scenario":
        name = str(data.get("name") or "")
        ok, msg = svc.delete_user_scenario(username, name)
        if not ok:
            return {"error": msg}
        return {"success": True, "message": msg}

    if command == "parse_scenario_yaml":
        raw = data.get("yaml_text", "")
        try:
            parsed = yaml.safe_load(raw) if raw else {}
        except Exception as exc:
            if logger is not None:
                logger.warning("Rejected invalid scenario YAML: %s", exc)
            return error_payload("Invalid YAML content")

        if not isinstance(parsed, dict):
            return error_payload("YAML root must be a mapping/object")

        return {"success": True, "config": parsed}

    return None
