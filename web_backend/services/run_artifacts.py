"""Run artifact access service for logs, metrics, resource usage, and export."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import nullcontext
from typing import Any, Callable

from job_manager.api.resource_usage import append_resource_sample_file
from utils.config import normalize_run_config_aliases


class RunArtifactsService:
    """Read and write run-local artifacts behind the service facade."""

    def __init__(
        self,
        *,
        project_root_getter: Callable[[], str],
        base_results_path: str,
        resource_usage_file: str,
        max_resource_samples: int,
        load_config_fn: Callable[[str], dict[str, Any]],
        db_enabled_getter: Callable[[], bool],
        runs_repo_getter: Callable[[], Any],
        lock_getter: Callable[[], Any],
    ) -> None:
        """Initialize the run-artifacts service.

        Args:
            project_root_getter: Callable returning the repository/project root.
            base_results_path: Base directory containing run artifact folders.
            resource_usage_file: Resource-usage filename stored per run.
            max_resource_samples: Maximum resource samples retained per run.
            load_config_fn: Callable used to load YAML configuration files.
            db_enabled_getter: Callable returning whether DB-backed storage is enabled.
            runs_repo_getter: Callable returning the runs repository.
            lock_getter: Callable returning the shared service lock/context manager.
        """
        self._project_root = project_root_getter
        self._base_results_path = base_results_path
        self._resource_usage_file = resource_usage_file
        self._max_resource_samples = max_resource_samples
        self._load_config = load_config_fn
        self._db_enabled = db_enabled_getter
        self._runs_repo = runs_repo_getter
        self._lock = lock_getter

    def run_resource_usage_path(self, run_id: str) -> str:
        """Return the resource-usage file path for a run."""
        return os.path.join(self._base_results_path, run_id, self._resource_usage_file)

    def read_logs(self, run_path: str, n: int = 100) -> str:
        """Read the tail of a run's training log."""
        log_path = os.path.join(run_path, "train.log")
        try:
            with open(log_path) as f:
                return "".join(f.readlines()[-n:])
        except FileNotFoundError:
            return "(log file not yet available)"
        except Exception as exc:
            return f"(log read error: {exc})"

    def append_run_resource_sample(self, run_id: str, sample: dict[str, Any]) -> None:
        """Append a hardware sample to a run-local resource usage file.

        Args:
            run_id: Run identifier whose ``resource_usage.json`` should update.
            sample: Hardware usage sample to append.
        """
        if not run_id or not isinstance(sample, dict):
            return

        run_dir = os.path.join(self._base_results_path, run_id)
        if not os.path.isdir(run_dir):
            return

        path = self.run_resource_usage_path(run_id)
        lock = self._lock() or nullcontext()
        with lock:
            append_resource_sample_file(path, sample, self._max_resource_samples)

    def load_run_resource_usage(self, run_id: str) -> dict[str, Any]:
        """Load summarized and sampled hardware usage for one run."""
        if not run_id:
            return {}
        path = self.run_resource_usage_path(run_id)
        try:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            pass
        return {}

    def load_metrics(self, run_path: str) -> dict[str, Any]:
        """Load run metrics from known metrics files.

        Single-repeat runs write ``metrics.json`` directly under ``run_path``.
        Multi-repeat runs instead write live metrics into ``repeat_<n>/`` and
        only emit a root ``summary_metrics.json`` once every repeat finishes. To
        keep the live monitor populated during a multi-repeat run, fall back to
        the most recently updated ``repeat_*/metrics.json`` when the root has no
        metrics file yet.
        """
        for file_name in ("metrics.json", "summary_metrics.json"):
            path = os.path.join(run_path, file_name)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        payload = json.load(f)
                    return self._normalize_metrics_payload(payload, run_path=run_path)
                except Exception:
                    pass

        live_repeat_path = self._latest_repeat_metrics_path(run_path)
        if live_repeat_path:
            try:
                with open(live_repeat_path) as f:
                    payload = json.load(f)
                return self._normalize_metrics_payload(
                    payload, run_path=os.path.dirname(live_repeat_path)
                )
            except Exception:
                pass
        return {}

    @staticmethod
    def _latest_repeat_metrics_path(run_path: str) -> str:
        """Return the newest ``repeat_*/metrics.json`` under ``run_path``.

        Used to surface live metrics from the in-progress repeat before the
        root summary exists. Returns an empty string when none is found.
        """
        best_path = ""
        best_mtime = -1.0
        try:
            entries = os.listdir(run_path)
        except OSError:
            return ""
        for name in entries:
            if not name.startswith("repeat_"):
                continue
            candidate = os.path.join(run_path, name, "metrics.json")
            try:
                mtime = os.path.getmtime(candidate)
            except OSError:
                continue
            if mtime > best_mtime:
                best_mtime = mtime
                best_path = candidate
        return best_path

    @staticmethod
    def _safe_mean_series(series_list: list[list[Any]]) -> list[Any]:
        """Return one element-wise mean series, preserving ``None`` when undefined."""
        if not series_list:
            return []
        max_len = max((len(series) for series in series_list), default=0)
        out: list[Any] = []
        for idx in range(max_len):
            vals: list[float] = []
            for series in series_list:
                if idx >= len(series):
                    continue
                raw = series[idx]
                if raw is None:
                    continue
                try:
                    vals.append(float(raw))
                except Exception:
                    continue
            out.append(sum(vals) / len(vals) if vals else None)
        return out

    def _normalize_metrics_payload(self, payload: dict[str, Any], *, run_path: str = "") -> dict[str, Any]:
        """Normalize metrics payloads so monitor consumers receive split keys."""
        if not isinstance(payload, dict):
            return {}

        normalized = dict(payload)

        def _normalize_client_bucket(bucket: Any) -> dict[str, Any]:
            if not isinstance(bucket, dict):
                return {}
            client = dict(bucket)
            if "val_loss" not in client and isinstance(client.get("test_loss"), list):
                client["val_loss"] = list(client.get("test_loss") or [])
            if "test_loss" in client:
                client.pop("test_loss", None)
            return client

        def _coerce_split(name: str) -> dict[str, list[Any]]:
            existing = normalized.get(name)
            if isinstance(existing, dict):
                return {
                    "loss": existing.get("loss") if isinstance(existing.get("loss"), list) else [],
                    "accuracy": existing.get("accuracy") if isinstance(existing.get("accuracy"), list) else [],
                    "precision": existing.get("precision") if isinstance(existing.get("precision"), list) else [],
                    "recall": existing.get("recall") if isinstance(existing.get("recall"), list) else [],
                    "f1": existing.get("f1") if isinstance(existing.get("f1"), list) else [],
                    "roc_auc": existing.get("roc_auc") if isinstance(existing.get("roc_auc"), list) else [],
                }
            return {
                "loss": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "roc_auc": [],
            }

        clients = normalized.get("clients")
        if isinstance(clients, list):
            normalized["clients"] = [_normalize_client_bucket(client) for client in clients]

        aggregated_val = _coerce_split("aggregated_val")
        aggregated_test = _coerce_split("aggregated_test")

        # Summary payloads for repeated runs are reconstructed from per-repeat
        # metrics so the split buckets stay consistent with the live
        # runtime schema.
        all_metrics = normalized.get("all_metrics")
        if isinstance(all_metrics, list) and all_metrics:
            normalized["all_metrics"] = [
                self._normalize_metrics_payload(run_metrics, run_path=run_path)
                if isinstance(run_metrics, dict)
                else run_metrics
                for run_metrics in all_metrics
            ]
            for split_name, target in (("aggregated_val", aggregated_val), ("aggregated_test", aggregated_test)):
                for metric_name in ("accuracy", "loss", "precision", "recall", "f1", "roc_auc"):
                    if target.get(metric_name):
                        continue
                    per_run_series = []
                    for run_metrics in all_metrics:
                        if not isinstance(run_metrics, dict):
                            continue
                        split_payload = run_metrics.get(split_name)
                        if not isinstance(split_payload, dict):
                            continue
                        series = split_payload.get(metric_name)
                        if isinstance(series, list) and series:
                            per_run_series.append(series)
                    target[metric_name] = self._safe_mean_series(per_run_series)

        normalized["aggregated_val"] = aggregated_val
        normalized["aggregated_test"] = aggregated_test
        return normalized

    def load_run_config(self, run_path: str) -> dict[str, Any]:
        """Load ``config.yaml`` for a specific run directory."""
        run_id = os.path.basename(os.path.abspath(run_path))
        runs_repo = self._runs_repo()
        if self._db_enabled() and runs_repo is not None and run_id:
            try:
                db_cfg = runs_repo.get_run_config(run_id)
                if isinstance(db_cfg, dict) and db_cfg:
                    return normalize_run_config_aliases(dict(db_cfg))
            except Exception:
                pass

        cfg_path = os.path.join(run_path, "config.yaml")
        try:
            loaded = self._load_config(cfg_path) if os.path.exists(cfg_path) else {}
            return normalize_run_config_aliases(loaded) if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    def list_dlg_images(self, run_path: str) -> list[tuple[str, str]]:
        """List DLG image files from a run directory."""
        images: list[tuple[str, str]] = []
        if os.path.isdir(run_path):
            for file_name in sorted(os.listdir(run_path)):
                if file_name.lower().endswith(".png") and "dlg" in file_name.lower():
                    images.append((os.path.join(run_path, file_name), file_name))
        return images

    def export_run_to_zip(self, run_id: str) -> str:
        """Archive a run directory as a ZIP file."""
        run_path = os.path.join(self._project_root(), self._base_results_path, run_id)
        if not os.path.isdir(run_path):
            raise FileNotFoundError(f"Run directory not found: {run_path}")

        out_dir = os.path.join(tempfile.gettempdir(), "fl_runs")
        os.makedirs(out_dir, exist_ok=True)
        archive_base = os.path.join(out_dir, run_id)
        return shutil.make_archive(
            base_name=archive_base,
            format="zip",
            root_dir=os.path.dirname(run_path),
            base_dir=run_id,
        )
