"""Alerts service for transient in-memory buffering and DB-backed delivery."""

from __future__ import annotations

import logging
from typing import Any, Callable


logger = logging.getLogger("web_backend.service")


class AlertsService:
    """Manage user alerts behind the service facade."""

    def __init__(
        self,
        *,
        db_enabled_getter: Callable[[], bool],
        alerts_repo_getter: Callable[[], Any],
        lock_getter: Callable[[], Any],
    ) -> None:
        """Initialize the alerts service.

        Args:
            db_enabled_getter: Callable returning whether DB-backed alerts are enabled.
            alerts_repo_getter: Callable returning the alerts repository instance.
            lock_getter: Callable returning the shared service lock/context manager.
        """
        self._db_enabled = db_enabled_getter
        self._alerts_repo = alerts_repo_getter
        self._lock = lock_getter

    def consume_user_alerts(self, username: str) -> list[dict[str, Any]]:
        """Drain pending alerts for a user from DB-backed storage.

        Args:
            username: Identifier of the user whose alerts should be consumed.

        Returns:
            Deduplicated alert payloads in API-friendly dictionary form.
        """
        with self._lock():
            alerts: list[dict[str, Any]] = []

            repo = self._alerts_repo()
            if self._db_enabled() and repo is not None:
                try:
                    rows = repo.list_alerts(username, consumed=False)
                    if rows:
                        alerts.extend(
                            {
                                "message": row.message,
                                "level": row.level,
                                "ts": row.created_at.isoformat(timespec="seconds"),
                            }
                            for row in rows
                        )
                        repo.mark_consumed([row.id for row in rows])
                except Exception as exc:
                    logger.warning("Failed to consume DB alerts for '%s': %s", username, exc)

            deduped: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()
            for item in alerts:
                key = (
                    str(item.get("message") or ""),
                    str(item.get("level") or ""),
                    str(item.get("ts") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            return deduped
