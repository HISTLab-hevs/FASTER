"""Tests for canonical metric loading and normalization of run artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from web_backend.services.run_artifacts import RunArtifactsService


def _make_service(base_results_path: str) -> RunArtifactsService:
    return RunArtifactsService(
        project_root_getter=lambda: "",
        base_results_path=base_results_path,
        resource_usage_file="resource_usage.json",
        max_resource_samples=10,
        load_config_fn=lambda path: yaml.safe_load(Path(path).read_text(encoding="utf-8")),
        db_enabled_getter=lambda: False,
        runs_repo_getter=lambda: None,
        lock_getter=lambda: None,
    )


def _split_payload(prefix: float) -> dict[str, list[float]]:
    return {
        "loss": [prefix + 0.25, prefix + 0.5],
        "accuracy": [prefix + 0.5, prefix + 1.0],
        "precision": [prefix + 0.75, prefix + 1.25],
        "recall": [prefix + 1.0, prefix + 1.5],
        "f1": [prefix + 1.25, prefix + 1.75],
        "roc_auc": [prefix + 1.5, prefix + 2.0],
    }


class RunArtifactsNormalizationTests(unittest.TestCase):
    """Cover canonical metric loading for new single and repeat runs."""

    def test_single_run_metrics_load_canonical_buckets_without_aliases(self) -> None:
        """Ensure a new single run loads canonical split buckets directly."""
        with tempfile.TemporaryDirectory(prefix="artifact_norm_current_") as tmp:
            run_dir = Path(tmp) / "run_alpha"
            run_dir.mkdir(parents=True)
            (run_dir / "config.yaml").write_text("evaluation_split_mode: train_val_test\n", encoding="utf-8")

            payload = {
                "clients": [
                    {
                        "loss": [0.1, 0.2],
                        "test_loss": [0.3, 0.4],
                        "accuracy": [0.5, 0.6],
                        "precision": [0.7, 0.8],
                        "recall": [0.9, 1.0],
                        "f1": [1.1, 1.2],
                        "roc_auc": [1.3, 1.4],
                    }
                ],
                "aggregated_val": _split_payload(0.0),
                "aggregated_test": _split_payload(1.0),
                "live_state": {
                    "phase": "round_complete",
                    "current_round": 2,
                    "total_rounds": 2,
                    "completed_rounds": 2,
                },
            }
            (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

            loaded = _make_service(tmp).load_metrics(str(run_dir))

        self.assertEqual([0.5, 1.0], loaded["aggregated_val"]["accuracy"])
        self.assertEqual([1.5, 2.0], loaded["aggregated_test"]["accuracy"])
        self.assertEqual([0.3, 0.4], loaded["clients"][0]["val_loss"])
        self.assertNotIn("test_loss", loaded["clients"][0])
        self.assertEqual("round_complete", loaded["live_state"]["phase"])
        self.assertNotIn("aggregated", loaded)
        self.assertNotIn("server", loaded)
        self.assertNotIn("mean_aggregated_accuracy", loaded)

    def test_repeat_run_summary_loads_from_all_metrics_without_aliases(self) -> None:
        """Ensure a new repeat-run summary rebuilds canonical split buckets from all_metrics."""
        with tempfile.TemporaryDirectory(prefix="artifact_norm_summary_") as tmp:
            run_dir = Path(tmp) / "summary_root"
            run_dir.mkdir(parents=True)

            summary_payload = {
                "all_metrics": [
                    {
                        "aggregated_val": _split_payload(0.0),
                        "aggregated_test": _split_payload(1.0),
                    },
                    {
                        "aggregated_val": _split_payload(0.5),
                        "aggregated_test": _split_payload(1.5),
                    },
                ],
                "mean_global_validation_accuracy": [0.25, 0.75],
                "mean_global_test_accuracy": [1.25, 1.75],
            }
            (run_dir / "summary_metrics.json").write_text(json.dumps(summary_payload), encoding="utf-8")

            loaded = _make_service(tmp).load_metrics(str(run_dir))

        self.assertEqual([0.75, 1.25], loaded["aggregated_val"]["accuracy"])
        self.assertEqual([1.75, 2.25], loaded["aggregated_test"]["accuracy"])
        self.assertIn("all_metrics", loaded)
        self.assertEqual(2, len(loaded["all_metrics"]))
        self.assertEqual([0.25, 0.75], loaded["mean_global_validation_accuracy"])
        self.assertEqual([1.25, 1.75], loaded["mean_global_test_accuracy"])
