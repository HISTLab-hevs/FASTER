"""Shared helpers for MariaDB repository implementations."""

from __future__ import annotations

import json
from typing import Any


def parse_json_object(value: Any) -> dict[str, Any] | None:
    """Parse a database JSON value into a dictionary.

    Args:
        value: Raw value returned from storage. Supported values are ``None``,
            dictionaries, and JSON-encoded strings.

    Returns:
        A dictionary when ``value`` is already a dictionary or decodes to a JSON
        object, otherwise ``None``.

    Side Effects:
        JSON decoding failures are intentionally swallowed to preserve existing
        repository behavior for malformed persisted payloads.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def parse_json_object_or_empty(value: Any) -> dict[str, Any]:
    """Parse a database JSON value into a dictionary with an empty fallback.

    Args:
        value: Raw value returned from storage. Supported values are ``None``,
            dictionaries, and JSON-encoded strings.

    Returns:
        A parsed dictionary when available, otherwise an empty dictionary.

    Side Effects:
        JSON decoding failures are swallowed and represented as ``{}``.
    """
    return parse_json_object(value) or {}


def build_update_assignments(
    updates: dict[str, Any],
    allowed_fields: set[str],
) -> tuple[dict[str, Any], str]:
    """Return filtered update values and their SQL assignment fragment.

    Args:
        updates: Candidate update mapping supplied by a repository caller.
        allowed_fields: Column names that may be written by the repository.

    Returns:
        A tuple containing the filtered mapping in caller-provided insertion
        order and a comma-separated ``column = %s`` SQL assignment fragment.
    """
    filtered_updates = {
        key: value for key, value in updates.items() if key in allowed_fields
    }
    assignments = ", ".join(f"{column} = %s" for column in filtered_updates)
    return filtered_updates, assignments
