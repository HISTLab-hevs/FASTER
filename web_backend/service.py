"""Facade that composes the backend service domains used by the web API.

``TrainingServiceLayer`` keeps the stable service interface used by API
handlers while delegating domain behavior to focused modules for run lifecycle,
artifacts, defaults, scenarios, datasets, and alerts.

The backend persists and presents run metadata here; job execution itself
remains the responsibility of the Job Manager.
"""

import logging
import os
import pathlib
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from web_backend.services.alerts import AlertsService
from web_backend.services.content import TrainingContentService
from web_backend.services.defaults import DefaultsService
from web_backend.services.datasets import CustomDatasetsService
from web_backend.services.run_artifacts import RunArtifactsService
from web_backend.services.run_lifecycle import RunLifecycleService
from web_backend.services.scenarios import ScenariosService
from utils.config import dump_config, load_config

try:
    from database.config import DatabaseSettings
    from database.connection import MariaDBConnectionFactory
    from database.repositories import (
        MariaAppSettingsRepository,
        MariaAlertsRepository,
        MariaCustomDatasetsRepository,
        MariaRunsRepository,
        MariaScenariosRepository,
    )
except Exception:
    MariaAppSettingsRepository = None
    DatabaseSettings = None
    MariaDBConnectionFactory = None
    MariaAlertsRepository = None
    MariaCustomDatasetsRepository = None
    MariaRunsRepository = None
    MariaScenariosRepository = None

_PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent)

BASE_RESULTS_PATH = "results"
SCENARIOS_DIR = "scenarios"
DEFAULT_CONFIG_FILE = "config.yaml"
SAVED_DEFAULTS_KEY = "saved_defaults"
CUSTOM_DATASETS_ROOT = os.path.join("data", "custom_datasets")
RESOURCE_USAGE_FILE = "resource_usage.json"
MAX_RESOURCE_SAMPLES = 4000


logger = logging.getLogger(__name__)

RUN_DISPLAY_NAME_MAX_LEN = 20
RUN_DISPLAY_NAME_PATTERN = r"[A-Za-z0-9_.\- ]+"


def validate_run_display_name(
    raw_name: Any,
    *,
    field_name: str = "run_name",
    required: bool = False,
) -> tuple[bool, str, str]:
    """Validate one run display-name value against the shared rule.

    Args:
        raw_name: Candidate display name.
        field_name: Error-message field label.
        required: Whether an empty value should be rejected.

    Returns:
        Tuple ``(is_valid, message, normalized_name)``.
    """
    if raw_name is None:
        normalized = ""
    else:
        normalized = str(raw_name).strip()

    if not normalized:
        if required:
            return False, f"{field_name} is required", ""
        return True, "", ""

    if len(normalized) > RUN_DISPLAY_NAME_MAX_LEN:
        return (
            False,
            f"{field_name} must be at most {RUN_DISPLAY_NAME_MAX_LEN} characters",
            "",
        )

    if not re.fullmatch(RUN_DISPLAY_NAME_PATTERN, normalized):
        return False, f"{field_name} contains invalid characters", ""

    return True, "", normalized


class TrainingServiceLayer:
    """Compose backend service domains behind the stable service interface.

    Source-of-truth rules:

    - MariaDB is the primary store for persisted run metadata and lifecycle
      state.
    - Filesystem artifacts remain the source for run-local outputs and
      execution snapshots.
    - Job Manager is the execution authority and may inform status changes
      through explicit reconciliation, but it does not replace the backend's
      persisted run record.
    """

    def __init__(self) -> None:
        """Initialize service state."""
        self._db_enabled = False
        self._app_settings_repo = None
        self._runs_repo = None
        self._scenarios_repo = None
        self._alerts_repo = None
        self._custom_datasets_repo = None

        if DatabaseSettings and MariaDBConnectionFactory:
            try:
                factory = MariaDBConnectionFactory(DatabaseSettings.from_env())
                if MariaAppSettingsRepository:
                    self._app_settings_repo = MariaAppSettingsRepository(factory)
                if MariaRunsRepository:
                    self._runs_repo = MariaRunsRepository(factory)
                if MariaScenariosRepository:
                    self._scenarios_repo = MariaScenariosRepository(factory)
                if MariaAlertsRepository:
                    self._alerts_repo = MariaAlertsRepository(factory)
                if MariaCustomDatasetsRepository:
                    self._custom_datasets_repo = MariaCustomDatasetsRepository(factory)

                self._db_enabled = any(
                    [
                        self._runs_repo,
                        self._scenarios_repo,
                        self._alerts_repo,
                        self._custom_datasets_repo,
                        self._app_settings_repo,
                    ]
                )
                if not self._db_enabled:
                    raise RuntimeError("TrainingServiceLayer requires DB-backed repositories")
            except Exception as exc:
                raise RuntimeError("TrainingServiceLayer requires DB-backed storage") from exc
        else:
            raise RuntimeError("TrainingServiceLayer requires configured MariaDB dependencies")

        self._lock = threading.Lock()
        self._content_service = TrainingContentService(
            defaults_service=DefaultsService(
                bootstrap_config_file=DEFAULT_CONFIG_FILE,
                saved_defaults_key=SAVED_DEFAULTS_KEY,
                load_config_fn=load_config,
                db_enabled_getter=lambda: bool(self._db_enabled),
                app_settings_repo_getter=lambda: self._app_settings_repo,
            ),
            scenarios_service=ScenariosService(
                scenarios_dir=SCENARIOS_DIR,
                load_config_fn=load_config,
                db_enabled_getter=lambda: bool(self._db_enabled),
                scenarios_repo_getter=lambda: self._scenarios_repo,
            ),
        )
        self._datasets_service = CustomDatasetsService(
            project_root_getter=lambda: _PROJECT_ROOT,
            custom_datasets_root=CUSTOM_DATASETS_ROOT,
            base_results_path=BASE_RESULTS_PATH,
            dump_config_fn=dump_config,
            load_run_config_fn=self.load_run_config,
            list_runs_fn=self.list_runs,
            db_enabled_getter=lambda: bool(self._db_enabled),
            custom_datasets_repo_getter=lambda: self._custom_datasets_repo,
            runs_repo_getter=lambda: self._runs_repo,
        )
        self._artifacts_service = RunArtifactsService(
            project_root_getter=lambda: _PROJECT_ROOT,
            base_results_path=BASE_RESULTS_PATH,
            resource_usage_file=RESOURCE_USAGE_FILE,
            max_resource_samples=MAX_RESOURCE_SAMPLES,
            load_config_fn=load_config,
            db_enabled_getter=lambda: bool(self._db_enabled),
            runs_repo_getter=lambda: self._runs_repo,
            lock_getter=lambda: self._lock,
        )
        self._alerts_service = AlertsService(
            db_enabled_getter=lambda: bool(self._db_enabled),
            alerts_repo_getter=lambda: self._alerts_repo,
            lock_getter=lambda: self._lock,
        )
        self._run_lifecycle_service = RunLifecycleService(
            base_results_path_getter=lambda: BASE_RESULTS_PATH,
            dump_config_fn=dump_config,
            load_run_config_fn=self.load_run_config,
            dataset_display_name_from_config_fn=self.dataset_display_name_from_config,
            validate_run_display_name_fn=validate_run_display_name,
            db_enabled_getter=lambda: bool(self._db_enabled),
            runs_repo_getter=lambda: self._runs_repo,
            lock_getter=lambda: self._lock,
        )

    # Run lifecycle, ownership, and discoverability
    def get_run_owner(self, run_id: str) -> Optional[str]:
        """Return the owner identifier for a run when it can be resolved."""
        return self._run_lifecycle_service.get_run_owner(run_id)

    def find_run_name_conflicts(
        self,
        names: List[str],
        username: Optional[str] = None,
    ) -> List[str]:
        """Find conflicting run names under the service lock."""
        return self._run_lifecycle_service.find_run_name_conflicts(names, username=username)

    def user_can_access_run(
        self,
        username: str,
        run_id: str,
        is_admin: bool = False,
    ) -> bool:
        """Return whether a user may access a run."""
        return self._run_lifecycle_service.user_can_access_run(username, run_id, is_admin=is_admin)

    def reserve_run_submission(
        self,
        config: Dict[str, Any],
        username: str,
    ) -> Tuple[bool, str, str, str, Dict[str, Any]]:
        """Reserve a run ID and persist queued state before submission."""
        return self._run_lifecycle_service.reserve_run_submission(config, username)

    def mark_run_status(self, run_id: str, status: str, reason: str = "") -> None:
        """Update the DB-backed run status."""
        self._run_lifecycle_service.mark_run_status(run_id, status, reason)

    def mark_run_best_accuracy(self, run_id: str, best_accuracy: float) -> None:
        """Update best accuracy in DB when available."""
        self._run_lifecycle_service.mark_run_best_accuracy(run_id, best_accuracy)

    def get_run_status(self, run_id: str) -> str:
        """Resolve run status from the DB row."""
        return self._run_lifecycle_service.get_run_status(run_id)

    def get_run_metadata(self, run_id: str) -> Dict[str, Any]:
        """Return compact metadata for a single run."""
        return self._run_lifecycle_service.get_run_metadata(run_id)

    def list_run_metadata(
        self,
        username: Optional[str] = None,
        include_all: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return compact run metadata rows for list/history endpoints."""
        return self._run_lifecycle_service.list_run_metadata(
            username=username,
            include_all=include_all,
            limit=limit,
        )

    # Alerts and lightweight runtime state
    def get_run_stop_reason(self, run_id: str) -> Optional[str]:
        """Return the explicit stop reason from the DB row."""
        return self._run_lifecycle_service.get_run_stop_reason(run_id)

    def consume_user_alerts(self, username: str) -> List[Dict[str, Any]]:
        """Drain and return pending alerts for a user."""
        return self._alerts_service.consume_user_alerts(username)

    def is_running(self, run_id: str = "") -> bool:
        """Check whether any run, or a specific run, is currently running."""
        return self._run_lifecycle_service.is_running(run_id)

    # Run artifacts, logs, metrics, and resource usage
    def read_logs(self, run_path: str, n: int = 100) -> str:
        """Read the tail of a run's training log."""
        return self._artifacts_service.read_logs(run_path, n=n)

    def append_run_resource_sample(self, run_id: str, sample: Dict[str, Any]) -> None:
        """Append a hardware sample to a run-local resource usage file."""
        self._artifacts_service.append_run_resource_sample(run_id, sample)

    def load_run_resource_usage(self, run_id: str) -> Dict[str, Any]:
        """Load summarized and sampled hardware usage for one run."""
        return self._artifacts_service.load_run_resource_usage(run_id)

    def load_metrics(self, run_path: str) -> Dict[str, Any]:
        """Load run metrics from known metrics files."""
        return self._artifacts_service.load_metrics(run_path)

    def load_run_config(self, run_path: str) -> Dict[str, Any]:
        """Load ``config.yaml`` for a specific run directory."""
        return self._artifacts_service.load_run_config(run_path)

    def list_runs(
        self,
        username: Optional[str] = None,
        include_all: bool = False,
    ) -> List[str]:
        """List available runs."""
        return self._run_lifecycle_service.list_runs(username=username, include_all=include_all)

    @staticmethod
    def dataset_display_name_from_config(cfg: Dict[str, Any]) -> str:
        """Return the user-facing dataset label for run summaries."""
        return CustomDatasetsService.dataset_display_name_from_config(cfg)

    def resolve_custom_dataset_path(
        self,
        username: str,
        dataset_ref: str,
        is_admin: bool = False,
    ) -> Optional[str]:
        """Resolve a user-owned custom dataset reference to an absolute path."""
        return self._datasets_service.resolve_custom_dataset_path(
            username,
            dataset_ref,
            is_admin=is_admin,
        )

    def list_custom_datasets(
        self,
        username: str,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """List custom datasets uploaded by the user or all users for admins."""
        return self._datasets_service.list_custom_datasets(username, is_admin=is_admin)

    def _load_custom_dataset_sidecar(self, abs_dataset_path: str) -> Dict[str, Any]:
        """Load optional sidecar metadata for a custom dataset file."""
        return self._datasets_service.load_custom_dataset_sidecar(abs_dataset_path)

    def rename_custom_dataset(
        self,
        username: str,
        dataset_ref: str,
        new_name: str,
        is_admin: bool = False,
    ) -> Tuple[bool, str]:
        """Rename a custom dataset display label without changing dataset_ref."""
        return self._datasets_service.rename_custom_dataset(
            username,
            dataset_ref,
            new_name,
            is_admin=is_admin,
        )

    def _propagate_custom_dataset_rename_to_runs(
        self,
        username: str,
        dataset_ref: str,
        new_name: str,
    ) -> int:
        """Propagate custom dataset display-name changes to run metadata.

        This keeps run history and run configs consistent after dataset rename.

        Args:
            username: Dataset owner identifier.
            dataset_ref: Stable custom dataset reference.
            new_name: New dataset display name.

        Returns:
            Number of runs/queued jobs updated.
        """
        return self._datasets_service._propagate_custom_dataset_rename_to_runs(
            username,
            dataset_ref,
            new_name,
        )

    def update_custom_dataset_description(
        self,
        username: str,
        dataset_ref: str,
        description: str,
        is_admin: bool = False,
    ) -> Tuple[bool, str]:
        """Update free-text description for a custom dataset."""
        return self._datasets_service.update_custom_dataset_description(
            username,
            dataset_ref,
            description,
            is_admin=is_admin,
        )

    def delete_custom_dataset(
        self,
        username: str,
        dataset_ref: str,
        is_admin: bool = False,
    ) -> Tuple[bool, str]:
        """Delete a custom dataset file and metadata for the requesting user."""
        return self._datasets_service.delete_custom_dataset(
            username,
            dataset_ref,
            is_admin=is_admin,
        )

    def validate_custom_dataset_structure(
        self,
        dataset_path: str,
        dataset_name: str,
    ) -> Tuple[bool, str]:
        """Validate a custom dataset before launching training."""
        return self._datasets_service.validate_custom_dataset_structure(dataset_path, dataset_name)

    def save_custom_dataset_file(
        self,
        username: str,
        original_name: str,
        source_path: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Validate, normalize, and persist a user-uploaded custom dataset from a file path."""
        return self._datasets_service.save_custom_dataset_file(
            username,
            original_name,
            source_path,
            description=description,
        )

    # Scenario persistence and saved defaults
    def list_scenarios(self, username: Optional[str] = None) -> List[str]:
        """List base and user scenarios."""
        return self._content_service.list_scenarios(username=username)

    def load_scenario(
        self,
        name: str,
        username: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load a scenario configuration."""
        return self._content_service.load_scenario(name, username=username)

    def save_user_scenario(
        self,
        username: str,
        config: Dict[str, Any],
        name: str = "",
    ) -> str:
        """Save or update a user scenario."""
        return self._content_service.save_user_scenario(username, config, name=name)

    def delete_user_scenario(
        self,
        username: str,
        name: str,
    ) -> tuple[bool, str]:
        """Delete one user-owned scenario."""
        return self._content_service.delete_user_scenario(username, name)

    def load_defaults(self, username: str | None = None) -> Dict[str, Any]:
        """Load default configuration values for one user-visible session."""
        return self._content_service.load_defaults(username=username)

    def save_defaults(
        self,
        config: Dict[str, Any],
        username: str | None = None,
        updated_by: str | None = None,
    ) -> bool:
        """Persist default configuration values."""
        return self._content_service.save_defaults(
            config,
            username=username,
            updated_by=updated_by,
        )

    # Run export and display-name mutations
    def list_dlg_images(self, run_path: str) -> List[Tuple[str, str]]:
        """List DLG image files from a run directory."""
        return self._artifacts_service.list_dlg_images(run_path)

    def export_run_to_zip(self, run_id: str) -> str:
        """Archive a run directory as a ZIP file."""
        return self._artifacts_service.export_run_to_zip(run_id)

    def delete_run(self, run_name: str) -> bool:
        """Permanently remove a run and its persisted metadata."""
        return self._run_lifecycle_service.delete_run(run_name)

    def rename_run(self, old_name: str, new_name: str) -> tuple[bool, str]:
        """Rename the display name of an existing run."""
        return self._run_lifecycle_service.rename_run(old_name, new_name)

# Single server-level instance. Owns a hardware resource (GPU/CPU),
# not user state, so one instance per process is correct.
svc = TrainingServiceLayer()
