"""Run lifecycle service for DB-backed run metadata and artifact ownership.

MariaDB is the primary source for product-level run metadata and lifecycle
state. The filesystem is still used for execution artifacts such as per-run
``config.yaml`` snapshots, logs, metrics, result files, and exports.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from typing import Any, Callable

from utils.config import normalize_run_config_aliases


logger = logging.getLogger("web_backend.service")


class RunLifecycleService:
    """Manage persisted run lifecycle metadata and discoverability rules."""

    def __init__(
        self,
        *,
        base_results_path_getter: Callable[[], str],
        dump_config_fn: Callable[[str, dict[str, Any]], None],
        load_run_config_fn: Callable[[str], dict[str, Any]],
        dataset_display_name_from_config_fn: Callable[[dict[str, Any]], str],
        validate_run_display_name_fn: Callable[..., tuple[bool, str, str]],
        db_enabled_getter: Callable[[], bool],
        runs_repo_getter: Callable[[], Any],
        lock_getter: Callable[[], Any],
    ) -> None:
        """Initialize the run-lifecycle service.

        Args:
            base_results_path_getter: Callable returning the base run-results path.
            dump_config_fn: Callable used to persist run configuration files.
            load_run_config_fn: Callable used to load persisted run configuration.
            dataset_display_name_from_config_fn: Callable deriving user-facing dataset labels.
            validate_run_display_name_fn: Callable validating run display names.
            db_enabled_getter: Callable returning whether DB-backed storage is enabled.
            runs_repo_getter: Callable returning the runs repository.
            lock_getter: Callable returning the shared service lock/context manager.
        """
        self._base_results_path = base_results_path_getter
        self._dump_config = dump_config_fn
        self._load_run_config = load_run_config_fn
        self._dataset_display_name_from_config = dataset_display_name_from_config_fn
        self._validate_run_display_name = validate_run_display_name_fn
        self._db_enabled = db_enabled_getter
        self._runs_repo = runs_repo_getter
        self._lock = lock_getter

    def make_unique_run_id_locked(
        self,
        requested_name: str,
        username: str,
    ) -> tuple[bool, str, str]:
        """Build a collision-free run ID while the caller holds the lock."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_label = str(requested_name or "").strip()
        run_id = raw_label if raw_label else f"run_{ts}"

        if raw_label:
            conflicts = self.run_name_conflicts_locked([raw_label], username=username)
            if conflicts:
                return False, f"Run name already exists in your account: {raw_label}", ""
            if self.run_id_exists_locked(run_id):
                base_run_id = f"{raw_label}_{ts}"
                run_id = base_run_id
                suffix = 1
                while self.run_id_exists_locked(run_id):
                    suffix += 1
                    run_id = f"{base_run_id}_{suffix}"
        else:
            base_run_id = run_id
            suffix = 1
            while self.run_id_exists_locked(run_id):
                suffix += 1
                run_id = f"{base_run_id}_{suffix}"

        return True, "", run_id

    def get_run_owner(self, run_id: str) -> str | None:
        """Return the owner identifier for a run when it can be resolved."""
        if not run_id:
            return None

        runs_repo = self._runs_repo()
        if self._db_enabled() and runs_repo is not None:
            try:
                row = runs_repo.get_run(run_id)
                if row:
                    owner = str(row.owner_identifier or "").strip()
                    if owner:
                        return owner
            except Exception as exc:
                logger.warning("Failed to resolve run owner from DB for run '%s': %s", run_id, exc)

        return None

    def run_id_exists_locked(self, run_id: str) -> bool:
        """Check whether a run ID is already reserved or persisted."""
        rid = str(run_id or "").strip()
        if not rid:
            return False

        runs_repo = self._runs_repo()
        if self._db_enabled() and runs_repo is not None:
            try:
                if runs_repo.get_run(rid):
                    return True
            except Exception as exc:
                logger.warning("Failed to check DB run-id existence for '%s': %s", rid, exc)

        return False

    def run_display_name_for_locked(self, run_id: str) -> str:
        """Resolve the preferred display label for a run."""
        row = self.get_db_run_row(run_id, log_context="run_display_name_for_locked")
        if row and row.display_name:
            return str(row.display_name).strip() or run_id

        return run_id

    @staticmethod
    def metadata_from_db_row(row: Any) -> dict[str, Any]:
        """Return the persisted metadata mapping for one DB run row."""
        run_id = str(row.run_id or "")
        return {
            "run_id": run_id,
            "display_name": str(row.display_name or run_id),
            "owner": str(row.owner_identifier or "unknown"),
            "status": str(row.status or "unknown"),
            "method": str(row.method or ""),
            "dataset_name": str(row.dataset_name or ""),
            "evaluation_split_mode": str(row.evaluation_split_mode or "train_val_test"),
            "model_name": str(row.model_name or "DefaultNet"),
            "stop_reason": str(row.stop_reason or ""),
            "best_accuracy": row.best_accuracy,
            "created_at": (
                row.created_at.isoformat(timespec="seconds")
                if getattr(row, "created_at", None)
                else ""
            ),
        }

    @staticmethod
    def db_owner_filter(username: str | None, include_all: bool) -> str | None:
        """Normalize owner filtering for DB-backed list queries."""
        return None if include_all or username == "admin" else username

    def get_db_run_row(self, run_id: str, log_context: str = "") -> Any:
        """Fetch one DB-backed run row when available."""
        runs_repo = self._runs_repo()
        if not (self._db_enabled() and runs_repo is not None and run_id):
            return None
        try:
            return runs_repo.get_run(run_id)
        except Exception as exc:
            context = f" during {log_context}" if log_context else ""
            logger.warning("Failed to load DB run row for '%s'%s: %s", run_id, context, exc)
            return None

    def list_db_run_rows(
        self,
        username: str | None = None,
        include_all: bool = False,
        log_context: str = "",
    ) -> list[Any]:
        """List DB-backed run rows for the requested visibility scope."""
        runs_repo = self._runs_repo()
        if not (self._db_enabled() and runs_repo is not None):
            return []
        owner = self.db_owner_filter(username, include_all)
        try:
            return runs_repo.list_runs(owner)
        except Exception as exc:
            context = f" during {log_context}" if log_context else ""
            logger.warning(
                "Failed to list DB run rows for owner '%s' (include_all=%s)%s: %s",
                username,
                include_all,
                context,
                exc,
            )
            return []

    def run_name_conflicts_locked(
        self,
        names: list[str],
        username: str | None = None,
    ) -> list[str]:
        """Return requested display-name conflicts while the caller holds the lock."""
        requested = [str(name or "").strip() for name in names]
        requested = [name for name in requested if name]
        if not requested:
            return []

        requested_map: dict[str, str] = {}
        for name in requested:
            key = name.lower()
            if key not in requested_map:
                requested_map[key] = name

        existing_labels: set[str] = set()
        for row in self.list_db_run_rows(
            username=username,
            include_all=not bool(username),
            log_context="run_name_conflicts_locked",
        ):
            existing_labels.add(str(row.display_name or row.run_id or "").lower())

        return sorted(
            requested_map[key]
            for key in requested_map.keys()
            if key in existing_labels
        )

    def find_run_name_conflicts(
        self,
        names: list[str],
        username: str | None = None,
    ) -> list[str]:
        """Find conflicting run names under the shared service lock."""
        with self._lock():
            return self.run_name_conflicts_locked(names, username=username)

    def user_can_access_run(
        self,
        username: str,
        run_id: str,
        is_admin: bool = False,
    ) -> bool:
        """Return whether a user may access a run."""
        if is_admin or username == "admin":
            return True
        row = self.get_db_run_row(run_id, log_context="user_can_access_run")
        if row:
            return str(row.owner_identifier or "") == username
        return self.get_run_owner(run_id) == username

    def persist_run_status_locked(self, run_id: str, status: str, reason: str = "") -> bool:
        """Persist lifecycle state to the stored run row.

        Args:
            run_id: Stable run identifier.
            status: Lifecycle status to store.
            reason: Optional stop/error reason.

        Returns:
            ``True`` when the DB row was updated, otherwise ``False``.
        """
        runs_repo = self._runs_repo()
        if not (self._db_enabled() and runs_repo is not None and run_id):
            logger.warning("Cannot persist run status for '%s': DB-backed run storage is unavailable", run_id)
            return False
        if status not in {"queued", "running", "completed", "failed", "stopped"}:
            logger.warning("Cannot persist unsupported run status '%s' for run '%s'", status, run_id)
            return False
        try:
            return bool(runs_repo.update_run(run_id, {"status": status, "stop_reason": reason or None}))
        except Exception as exc:
            logger.warning(
                "Failed to persist run status '%s' to DB for run '%s': %s",
                status,
                run_id,
                exc,
            )
            return False

    def persist_reserved_run_locked(
        self,
        run_id: str,
        config: dict[str, Any],
        username: str,
    ) -> bool:
        """Persist the queued run row during reservation.

        Args:
            run_id: Stable run identifier.
            config: Normalized run configuration to persist.
            username: Owner identifier.

        Returns:
            ``True`` when the run metadata and config were stored in MariaDB.
        """
        runs_repo = self._runs_repo()
        if not (self._db_enabled() and runs_repo is not None and run_id):
            return False

        try:
            existing = runs_repo.get_run(run_id)
        except Exception as exc:
            logger.warning("Failed to query existing reserved run '%s' from DB: %s", run_id, exc)
            existing = None

        display_name = str(config.get("display_run_name") or config.get("run_name") or run_id).strip() or run_id
        payload = {
            "run_id": run_id,
            "owner_identifier": username,
            "display_name": display_name,
            "status": "queued",
            "method": str(config.get("method") or ""),
            "dataset_name": str(self._dataset_display_name_from_config(config) or ""),
            "evaluation_split_mode": str(config.get("evaluation_split_mode") or "train_val_test"),
            "model_name": str(config.get("custom_model_name") or "DefaultNet"),
            "stop_reason": "Queued in Job Manager",
            "best_accuracy": None,
            "config_json": dict(config),
        }

        try:
            if existing:
                runs_repo.update_run(
                    run_id,
                    {
                        "display_name": payload["display_name"],
                        "status": payload["status"],
                        "method": payload["method"],
                        "dataset_name": payload["dataset_name"],
                        "evaluation_split_mode": payload["evaluation_split_mode"],
                        "model_name": payload["model_name"],
                        "stop_reason": payload["stop_reason"],
                        "best_accuracy": None,
                    },
                )
                runs_repo.upsert_run_config(run_id, payload["config_json"])
            else:
                runs_repo.create_run(payload)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to persist queued run '%s' in DB during reservation: %s",
                run_id,
                exc,
            )
            return False

    def reserve_run_submission(
        self,
        config: dict[str, Any],
        username: str,
    ) -> tuple[bool, str, str, str, dict[str, Any]]:
        """Reserve a run ID and persist queued state before submission."""
        with self._lock():
            try:
                cfg = dict(config or {})
                requested_name = str(cfg.get("run_name") or "").strip()

                ok, message, run_id = self.make_unique_run_id_locked(
                    requested_name=requested_name,
                    username=username,
                )
                if not ok:
                    return False, message, "", "", {}

                if requested_name:
                    cfg["display_run_name"] = requested_name
                else:
                    cfg["display_run_name"] = run_id

                normalize_run_config_aliases(cfg)
                cfg["run_name"] = run_id
                cfg["owner"] = username
                run_path = os.path.join(self._base_results_path(), run_id)
                cfg["save_path"] = run_path

                if not self.persist_reserved_run_locked(run_id, cfg, username):
                    return False, "Could not reserve run in the database", "", "", {}

                os.makedirs(run_path, exist_ok=True)
                cfg_file = os.path.join(run_path, "config.yaml")
                self._dump_config(cfg_file, cfg)

                return True, "Queued in Job Manager", run_id, run_path, cfg
            except Exception as exc:
                logger.exception(
                    "Failed to reserve run submission for user '%s' and requested name '%s'",
                    username,
                    str((config or {}).get("run_name") or "").strip(),
                )
                return False, f"Could not reserve run: {exc}", "", "", {}

    def mark_run_status(self, run_id: str, status: str, reason: str = "") -> None:
        """Update lifecycle status through the stored run row.

        Args:
            run_id: Stable run identifier.
            status: Lifecycle status to store.
            reason: Optional stop/error reason.

        Side Effects:
            Updates ``runs.status`` and ``runs.stop_reason`` in MariaDB.
        """
        if not run_id:
            return
        with self._lock():
            self.persist_run_status_locked(run_id, status, reason)

    def mark_run_best_accuracy(self, run_id: str, best_accuracy: float) -> None:
        """Update best accuracy in DB when available."""
        if not run_id:
            return
        runs_repo = self._runs_repo()
        if self._db_enabled() and runs_repo is not None:
            try:
                runs_repo.update_run(run_id, {"best_accuracy": float(best_accuracy)})
            except Exception as exc:
                logger.warning(
                    "Failed to persist best_accuracy to DB for run '%s': %s",
                    run_id,
                    exc,
                )

    def get_run_status(self, run_id: str) -> str:
        """Resolve run status from the stored run row.

        Args:
            run_id: Stable run identifier.

        Returns:
            The stored ``runs.status`` value when available, otherwise
            ``"unknown"``.
        """
        if not run_id:
            return "unknown"
        row = self.get_db_run_row(run_id, log_context="get_run_status")
        if row and row.status:
            return str(row.status)
        return "unknown"

    def get_run_metadata(self, run_id: str) -> dict[str, Any]:
        """Return compact metadata for a single run."""
        out: dict[str, Any] = {
            "run_id": run_id,
            "display_name": run_id,
            "owner": self.get_run_owner(run_id) or "unknown",
            "status": "unknown",
            "method": "",
            "dataset_name": "",
            "evaluation_split_mode": "train_val_test",
            "model_name": "DefaultNet",
            "stop_reason": self.get_run_stop_reason(run_id),
            "best_accuracy": None,
        }

        row = self.get_db_run_row(run_id, log_context="get_run_metadata")
        if row:
            out.update(self.metadata_from_db_row(row))
            if not out.get("stop_reason"):
                out["stop_reason"] = str(self.get_run_stop_reason(run_id) or "")
            return out

        out["status"] = self.get_run_status(run_id)
        return out

    def list_run_metadata(
        self,
        username: str | None = None,
        include_all: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return compact run metadata rows for list/history endpoints."""
        out: list[dict[str, Any]] = []

        for row in self.list_db_run_rows(
            username=username,
            include_all=include_all,
            log_context="list_run_metadata",
        ):
            run_id = str(row.run_id or "")
            if not run_id:
                continue
            out.append(self.metadata_from_db_row(row))

        if limit is not None:
            out = out[: max(0, int(limit))]
        return out

    def get_run_stop_reason(self, run_id: str) -> str | None:
        """Return the stop reason from the stored run row.

        Args:
            run_id: Stable run identifier.

        Returns:
            The stored ``runs.stop_reason`` value when present, otherwise
            ``None``.
        """
        row = self.get_db_run_row(run_id, log_context="get_run_stop_reason")
        if row and row.stop_reason:
            return str(row.stop_reason)
        return None

    def is_running(self, run_id: str = "") -> bool:
        """Check whether any run, or a specific run, is currently running."""
        if run_id:
            return self.get_run_status(run_id) == "running"

        runs_repo = self._runs_repo()
        if self._db_enabled() and runs_repo is not None:
            try:
                rows = runs_repo.list_runs(None)
                for row in rows:
                    if str(row.status or "") == "running":
                        return True
            except Exception:
                pass

        return False

    def list_runs(
        self,
        username: str | None = None,
        include_all: bool = False,
    ) -> list[str]:
        """List available runs."""
        db_runs = [
            str(row.run_id)
            for row in self.list_db_run_rows(
                username=username,
                include_all=include_all,
                log_context="list_runs",
            )
            if str(row.run_id or "")
        ]

        return db_runs

    def delete_run(self, run_name: str) -> bool:
        """Permanently remove a run and its persisted metadata."""
        run_path = os.path.join(self._base_results_path(), run_name)
        runs_repo = self._runs_repo()
        has_dir = os.path.isdir(run_path)

        if has_dir:
            try:
                shutil.rmtree(run_path, ignore_errors=False)
            except FileNotFoundError:
                has_dir = False
            except Exception as exc:
                logger.warning(
                    "Failed to delete run artifacts for '%s'; keeping database record for retry: %s",
                    run_name,
                    exc,
                )
                return False

        db_deleted = False
        if self._db_enabled() and runs_repo is not None:
            try:
                db_deleted = bool(runs_repo.delete_run(run_name))
            except Exception as exc:
                logger.warning("Failed to delete DB run row for '%s': %s", run_name, exc)
                return False

        return has_dir or db_deleted

    def rename_run(self, old_name: str, new_name: str) -> tuple[bool, str]:
        """Rename the display name of an existing run."""
        valid, message, normalized_name = self._validate_run_display_name(
            new_name,
            field_name="new_name",
            required=True,
        )
        if not valid:
            return False, message
        new_name = normalized_name or ""

        run_path = os.path.join(self._base_results_path(), old_name)
        has_dir = os.path.isdir(run_path)
        runs_repo = self._runs_repo()

        db_updated = False
        if self._db_enabled() and runs_repo is not None:
            try:
                db_updated = runs_repo.update_run(old_name, {"display_name": new_name})
            except Exception as exc:
                return False, f"Failed to update database on rename: {str(exc)}"

        file_updated = False
        if has_dir:
            try:
                cfg = self._load_run_config(run_path) or {}
                normalize_run_config_aliases(cfg)
                cfg["display_run_name"] = new_name
                cfg.setdefault("run_name", old_name)
                self._dump_config(os.path.join(run_path, "config.yaml"), cfg)
                file_updated = True
            except Exception:
                file_updated = False

        if not db_updated and not has_dir:
            return False, f"Run '{old_name}' not found"
        if not db_updated and self._db_enabled() and runs_repo is not None and not file_updated:
            return False, "Rename failed"
        return True, f"Run display name updated to '{new_name}'"
