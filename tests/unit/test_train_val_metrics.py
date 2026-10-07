"""Tests for train/validation/test metric collection during federated rounds."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import torch
import yaml

from tests.helpers import reload_module, valid_run_config
from web_backend.services.run_artifacts import RunArtifactsService


class DummyLoader:
    def __init__(self, name: str, size: int = 4) -> None:
        self.name = name
        self.dataset = list(range(size))

    def __len__(self) -> int:
        return len(self.dataset)


class DummyDataset:
    def __init__(self, *args, round_client_sample_counts: list[int] | None = None, **kwargs) -> None:
        self.n_channels = 1
        self.n_classes = 2
        self.client_datasets_train = [
            DummyLoader("server_train"),
            DummyLoader("client_train_1"),
            DummyLoader("client_train_2"),
        ]
        self.val_data = DummyLoader("val")
        self.test_data = DummyLoader("test")
        self.round_client_sample_counts = round_client_sample_counts or [
            len(self.client_datasets_train[1].dataset),
            len(self.client_datasets_train[2].dataset),
        ]
        self.last_instance = self

    def print_class_distribution(self, *args, **kwargs) -> None:
        return None

    def plot_class_distribution(self, *args, **kwargs) -> None:
        return None

    def federated_client_train_percentages(self) -> list[float]:
        return [50.0, 50.0]

    def build_round_train_subsets(self, resolved_round_train_percentages, seed=0):
        return [
            {
                0: self.client_datasets_train[1],
                1: self.client_datasets_train[2],
            }
            for _ in resolved_round_train_percentages
        ]

    def build_round_client_train_loaders(self, round_subset, round_client_percentages, *, eligible_clients, seed=0):
        return {
            "client_loaders": {
                0: DummyLoader("round_client_0", size=self.round_client_sample_counts[0]),
                1: DummyLoader("round_client_1", size=self.round_client_sample_counts[1]),
            },
            "effective_client_percentages": [50.0, 50.0],
            "client_sample_counts": list(self.round_client_sample_counts),
            "active_client_ids": [
                idx for idx, count in enumerate(self.round_client_sample_counts) if count > 0
            ],
            "zero_sample_client_ids": [
                idx for idx, count in enumerate(self.round_client_sample_counts) if count <= 0
            ],
        }


class TrainValMetricRoundTests(unittest.TestCase):
    """Regression coverage for global validation/test metric accounting."""

    def setUp(self) -> None:
        self.main_module = reload_module("main")

    def _patch_runtime(self, *, eval_mode: str, round_client_sample_counts: list[int] | None = None):
        dataset_holder: dict[str, DummyDataset] = {}

        def dataset_factory(*args, **kwargs):
            dataset = DummyDataset(round_client_sample_counts=round_client_sample_counts)
            dataset_holder["dataset"] = dataset
            return dataset

        def resolve_model_factory(config, n_channels, n_classes):
            return lambda: torch.nn.Linear(int(n_channels), int(n_classes))

        def train_stub(*args, epochs=1, **kwargs):
            return {
                "loss": [0.1 for _ in range(epochs)],
                "val_loss": [0.2 for _ in range(epochs)],
                "accuracy": [0.9 for _ in range(epochs)],
                "precision": [0.8 for _ in range(epochs)],
                "recall": [0.7 for _ in range(epochs)],
                "f1": [0.6 for _ in range(epochs)],
                "roc_auc": [0.5 for _ in range(epochs)],
            }

        def evaluate_stub(model, loader, device="cpu"):
            dataset = dataset_holder["dataset"]
            if loader is dataset.test_data:
                return 0.4, 0.45, 0.55, 0.65, 0.75, 0.85
            return 0.3, 0.35, 0.25, 0.15, 0.05, 0.95

        patches = [
            mock.patch.object(self.main_module, "Dataset", side_effect=dataset_factory),
            mock.patch.object(self.main_module, "gpu_check", return_value="cpu"),
            mock.patch.object(self.main_module, "train", side_effect=train_stub),
            mock.patch.object(self.main_module, "evaluate", side_effect=evaluate_stub),
            mock.patch.object(self.main_module, "fed_avg", side_effect=lambda server_model, participant_state_dicts: server_model),
            mock.patch.object(self.main_module, "fed_avgw", side_effect=lambda server_model, participant_state_dicts, participant_num_samples: server_model),
            mock.patch.object(self.main_module, "fed_gp", side_effect=lambda *args, **kwargs: (args[0], None, None)),
            mock.patch.object(self.main_module, "fed_prox", side_effect=lambda server_model, participant_state_dicts, participant_num_samples=None, weighted=True: server_model),
            mock.patch.object(self.main_module, "fed_nova", side_effect=lambda server_model, participant_state_dicts, participant_num_samples, participant_local_steps, participant_lrs, participant_momentums: server_model),
            mock.patch.object(self.main_module, "_resolve_model_factory", side_effect=resolve_model_factory),
        ]
        return dataset_holder, patches

    def test_train_val_mode_appends_one_validation_point_per_round(self) -> None:
        """Ensure train_val records exactly one validation metric point per aggregation round."""
        cfg = valid_run_config()
        cfg.update(
            {
                "evaluation_split_mode": "train_val",
                "local_model_epochs": 2,
                "weights_sending_frequency": 1,
                "num_clients": 2,
                "server_warmup_epochs": 1,
                "batch_size": 4,
                "server_learning_rate": 0.01,
                "learning_rate": 0.01,
                "momentum": 0.0,
            }
        )

        with tempfile.TemporaryDirectory(prefix="train_val_rounds_") as tmp:
            save_path = Path(tmp) / "run"
            dataset_holder, patches = self._patch_runtime(eval_mode="train_val")
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                metrics = self.main_module.run_experiment(cfg, str(save_path))

        self.assertEqual(2, len(metrics["rounds"]))
        self.assertEqual(2, len(metrics["aggregated_val"]["accuracy"]))
        self.assertEqual(0, len(metrics["aggregated_test"]["accuracy"]))
        self.assertEqual(2, metrics["live_state"]["completed_rounds"])
        self.assertEqual(2, metrics["live_state"]["current_round"])
        self.assertEqual(2, metrics["live_state"]["total_rounds"])
        self.assertIsNotNone(dataset_holder["dataset"])

    def test_train_val_test_mode_records_validation_and_test_once_per_round(self) -> None:
        """Ensure train_val_test keeps one validation and one test point per round."""
        cfg = valid_run_config()
        cfg.update(
            {
                "evaluation_split_mode": "train_val_test",
                "local_model_epochs": 2,
                "weights_sending_frequency": 1,
                "num_clients": 2,
                "server_warmup_epochs": 1,
                "batch_size": 4,
                "server_learning_rate": 0.01,
                "learning_rate": 0.01,
                "momentum": 0.0,
            }
        )

        with tempfile.TemporaryDirectory(prefix="train_val_test_rounds_") as tmp:
            save_path = Path(tmp) / "run"
            dataset_holder, patches = self._patch_runtime(eval_mode="train_val_test")
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                metrics = self.main_module.run_experiment(cfg, str(save_path))

        self.assertEqual(2, len(metrics["rounds"]))
        self.assertEqual(2, len(metrics["aggregated_val"]["accuracy"]))
        self.assertEqual(2, len(metrics["aggregated_test"]["accuracy"]))
        self.assertEqual(2, metrics["live_state"]["completed_rounds"])
        self.assertEqual(2, metrics["live_state"]["current_round"])
        self.assertEqual(2, metrics["live_state"]["total_rounds"])
        self.assertIsNotNone(dataset_holder["dataset"])

    def test_train_val_mode_one_round_records_one_validation_point(self) -> None:
        """Ensure a single train_val round yields one validation point and no test points."""
        cfg = valid_run_config()
        cfg.update(
            {
                "evaluation_split_mode": "train_val",
                "local_model_epochs": 1,
                "weights_sending_frequency": 1,
                "num_clients": 2,
                "server_warmup_epochs": 1,
                "batch_size": 4,
                "server_learning_rate": 0.01,
                "learning_rate": 0.01,
                "momentum": 0.0,
            }
        )

        with tempfile.TemporaryDirectory(prefix="train_val_one_round_") as tmp:
            save_path = Path(tmp) / "run"
            dataset_holder, patches = self._patch_runtime(eval_mode="train_val")
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                metrics = self.main_module.run_experiment(cfg, str(save_path))

        self.assertEqual(1, len(metrics["rounds"]))
        self.assertEqual(1, len(metrics["aggregated_val"]["accuracy"]))
        self.assertEqual(1, len(metrics["aggregated_val"]["loss"]))
        self.assertEqual(0, len(metrics["aggregated_test"]["accuracy"]))
        self.assertEqual(1, metrics["live_state"]["completed_rounds"])
        self.assertEqual(1, metrics["live_state"]["current_round"])
        self.assertEqual(1, metrics["live_state"]["total_rounds"])
        self.assertIsNotNone(dataset_holder["dataset"])

    def test_train_val_test_mode_one_round_records_validation_and_test_points(self) -> None:
        """Ensure a single train_val_test round yields one validation and one test point."""
        cfg = valid_run_config()
        cfg.update(
            {
                "evaluation_split_mode": "train_val_test",
                "local_model_epochs": 1,
                "weights_sending_frequency": 1,
                "num_clients": 2,
                "server_warmup_epochs": 1,
                "batch_size": 4,
                "server_learning_rate": 0.01,
                "learning_rate": 0.01,
                "momentum": 0.0,
            }
        )

        with tempfile.TemporaryDirectory(prefix="train_val_test_one_round_") as tmp:
            save_path = Path(tmp) / "run"
            dataset_holder, patches = self._patch_runtime(eval_mode="train_val_test")
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                metrics = self.main_module.run_experiment(cfg, str(save_path))

        self.assertEqual(1, len(metrics["rounds"]))
        self.assertEqual(1, len(metrics["aggregated_val"]["accuracy"]))
        self.assertEqual(1, len(metrics["aggregated_val"]["loss"]))
        self.assertEqual(1, len(metrics["aggregated_test"]["accuracy"]))
        self.assertEqual(1, len(metrics["aggregated_test"]["loss"]))
        self.assertEqual(1, metrics["live_state"]["completed_rounds"])
        self.assertEqual(1, metrics["live_state"]["current_round"])
        self.assertEqual(1, metrics["live_state"]["total_rounds"])
        self.assertIsNotNone(dataset_holder["dataset"])

    def test_repeat_run_summary_and_zip_export_keep_metric_lengths_coherent(self) -> None:
        """Ensure summary metrics and ZIP exports stay aligned with the real round count."""
        cfg = valid_run_config()
        cfg.update(
            {
                "evaluation_split_mode": "train_val_test",
                "local_model_epochs": 2,
                "weights_sending_frequency": 1,
                "num_clients": 2,
                "server_warmup_epochs": 1,
                "batch_size": 4,
                "server_learning_rate": 0.01,
                "learning_rate": 0.01,
                "momentum": 0.0,
                "repeat": 2,
            }
        )

        with tempfile.TemporaryDirectory(prefix="train_val_repeat_") as tmp:
            tmp_path = Path(tmp)
            save_path = tmp_path / "results"
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

            dataset_holder, patches = self._patch_runtime(eval_mode="train_val_test")
            args = argparse.Namespace(config=str(cfg_path), save_path=str(save_path), display_metrics=False)
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                with ExitStack() as stack:
                    for patcher in patches:
                        stack.enter_context(patcher)
                    stack.enter_context(mock.patch.object(self.main_module, "parse_args", return_value=args))
                    self.main_module.main()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            summary_path = save_path / "summary_metrics.json"
            self.assertTrue(summary_path.is_file())
            summary_metrics = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(2, len(summary_metrics["mean_global_validation_accuracy"]))
            self.assertEqual(2, len(summary_metrics["mean_global_validation_loss"]))
            self.assertEqual(2, len(summary_metrics["mean_global_test_accuracy"]))
            self.assertNotIn("mean_aggregated_accuracy", summary_metrics)
            self.assertNotIn("mean_aggregated_loss", summary_metrics)
            self.assertEqual(2, len(summary_metrics["all_metrics"]))
            self.assertEqual(2, len(summary_metrics["all_metrics"][0]["aggregated_val"]["accuracy"]))
            self.assertEqual(2, len(summary_metrics["all_metrics"][0]["aggregated_test"]["accuracy"]))

            artifacts = RunArtifactsService(
                project_root_getter=lambda: "",
                base_results_path=str(save_path),
                resource_usage_file="resource_usage.json",
                max_resource_samples=100,
                load_config_fn=lambda path: yaml.safe_load(Path(path).read_text(encoding="utf-8")),
                db_enabled_getter=lambda: False,
                runs_repo_getter=lambda: None,
                lock_getter=lambda: None,
            )
            zip_path = artifacts.export_run_to_zip("repeat_1")
            self.assertTrue(Path(zip_path).is_file())

            with zipfile.ZipFile(zip_path) as archive:
                self.assertIn("repeat_1/metrics.json", archive.namelist())
                self.assertIn("repeat_1/config.yaml", archive.namelist())
                metrics = json.loads(archive.read("repeat_1/metrics.json").decode("utf-8"))
                config_yaml = archive.read("repeat_1/config.yaml").decode("utf-8")

            self.assertNotIn("aggregated", metrics)
            self.assertNotIn("server", metrics)
            self.assertEqual(2, len(metrics["aggregated_val"]["accuracy"]))
            self.assertEqual(2, len(metrics["aggregated_val"]["loss"]))
            self.assertEqual(2, len(metrics["aggregated_val"]["precision"]))
            self.assertEqual(2, len(metrics["aggregated_val"]["recall"]))
            self.assertEqual(2, len(metrics["aggregated_val"]["f1"]))
            self.assertEqual(2, len(metrics["aggregated_val"]["roc_auc"]))
            self.assertEqual(2, len(metrics["aggregated_test"]["accuracy"]))
            self.assertEqual(2, len(metrics["aggregated_test"]["loss"]))
            self.assertEqual(2, len(metrics["aggregated_test"]["precision"]))
            self.assertEqual(2, len(metrics["aggregated_test"]["recall"]))
            self.assertEqual(2, len(metrics["aggregated_test"]["f1"]))
            self.assertEqual(2, len(metrics["aggregated_test"]["roc_auc"]))
            self.assertEqual(2, len(metrics["clients"][0]["loss"]))
            self.assertEqual(2, len(metrics["clients"][1]["loss"]))
            self.assertEqual(2, len(metrics["clients"][0]["accuracy"]))
            self.assertEqual(2, len(metrics["clients"][0]["f1"]))
            self.assertEqual(2, len(metrics["clients"][0]["roc_auc"]))
            self.assertIn("server_warmup_epochs:", config_yaml)
            self.assertIn("seed: 0", config_yaml)
            self.assertIn("server_data_percentage: 0.1", config_yaml)
            self.assertIn("imbalance_rate: 0.5", config_yaml)
            self.assertIn("train_val_split: 0.9", config_yaml)
            self.assertIn("iid: false", config_yaml)
            self.assertNotIn("global_model_epochs:", config_yaml)
            loaded_summary = artifacts.load_metrics(str(save_path))
            self.assertEqual(2, len(loaded_summary["aggregated_val"]["accuracy"]))
            self.assertEqual(2, len(loaded_summary["aggregated_test"]["accuracy"]))
            self.assertEqual([0.35, 0.35], loaded_summary["aggregated_val"]["accuracy"])
            self.assertEqual([0.45, 0.45], loaded_summary["aggregated_test"]["accuracy"])
            self.assertEqual(2, metrics["live_state"]["completed_rounds"])
            self.assertEqual(2, metrics["live_state"]["current_round"])
            self.assertEqual(2, metrics["live_state"]["total_rounds"])
            self.assertIsNotNone(dataset_holder["dataset"])

    def test_zero_sample_round_skips_empty_clients_and_keeps_zip_metrics_intact(self) -> None:
        """Ensure zero-sample clients are skipped without losing round metrics in exports."""
        cfg = valid_run_config()
        cfg.update(
            {
                "evaluation_split_mode": "train_val_test",
                "method": "fed_nova",
                "local_model_epochs": 1,
                "weights_sending_frequency": 1,
                "num_clients": 2,
                "server_warmup_epochs": 1,
                "batch_size": 2,
                "server_learning_rate": 0.01,
                "learning_rate": 0.01,
                "momentum": 0.0,
            }
        )

        with tempfile.TemporaryDirectory(prefix="zero_sample_round_") as tmp:
            tmp_path = Path(tmp)
            save_path = tmp_path / "results" / "zero_sample_run"
            dataset_holder, patches = self._patch_runtime(
                eval_mode="train_val_test",
                round_client_sample_counts=[4, 0],
            )

            fed_nova_calls: list[tuple[list[int], list[int], list[float]]] = []

            def fed_nova_stub(
                server_model,
                participant_state_dicts,
                participant_num_samples,
                participant_local_steps,
                participant_lrs,
                participant_momentums,
            ):
                self.assertEqual(1, len(participant_state_dicts))
                self.assertEqual([4], participant_num_samples)
                self.assertEqual([4], participant_local_steps)
                self.assertEqual([0.01], participant_lrs)
                self.assertEqual([0.0], participant_momentums)
                fed_nova_calls.append(
                    (
                        list(participant_num_samples),
                        list(participant_local_steps),
                        list(participant_lrs),
                    )
                )
                return server_model

            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                stack.enter_context(mock.patch.object(self.main_module, "fed_nova", side_effect=fed_nova_stub))
                metrics = self.main_module.run_experiment(cfg, str(save_path))

            self.assertEqual(1, len(fed_nova_calls))
            self.assertEqual(1, len(metrics["rounds"]))
            self.assertEqual([4, 0], metrics["rounds"][0]["train_schedule_client_sample_counts"])
            self.assertEqual(4, metrics["rounds"][0]["train_schedule_total_client_samples"])
            self.assertEqual([1], metrics["rounds"][0]["train_schedule_trainable_client_ids"])
            self.assertEqual(1, metrics["rounds"][0]["train_schedule_trainable_client_count"])
            self.assertEqual([2], metrics["rounds"][0]["train_schedule_zero_sample_client_ids"])
            self.assertEqual(1, metrics["rounds"][0]["train_schedule_zero_sample_client_count"])
            self.assertEqual(1, len(metrics["aggregated_val"]["accuracy"]))
            self.assertEqual(1, len(metrics["aggregated_test"]["accuracy"]))
            self.assertEqual(1, len(metrics["clients"][0]["loss"]))
            self.assertEqual(0, len(metrics["clients"][1]["loss"]))

            artifacts = RunArtifactsService(
                project_root_getter=lambda: "",
                base_results_path=str(tmp_path / "results"),
                resource_usage_file="resource_usage.json",
                max_resource_samples=100,
                load_config_fn=lambda path: yaml.safe_load(Path(path).read_text(encoding="utf-8")),
                db_enabled_getter=lambda: False,
                runs_repo_getter=lambda: None,
                lock_getter=lambda: None,
            )
            zip_path = artifacts.export_run_to_zip("zero_sample_run")
            self.assertTrue(Path(zip_path).is_file())

            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())
                self.assertIn("zero_sample_run/metrics.json", names)
                self.assertIn("zero_sample_run/config.yaml", names)
                self.assertIn("zero_sample_run/global_model_0.pth", names)
                zipped_metrics = json.loads(archive.read("zero_sample_run/metrics.json").decode("utf-8"))

            self.assertNotIn("aggregated", zipped_metrics)
            self.assertNotIn("server", zipped_metrics)
            self.assertEqual([4, 0], zipped_metrics["rounds"][0]["train_schedule_client_sample_counts"])
            self.assertEqual([2], zipped_metrics["rounds"][0]["train_schedule_zero_sample_client_ids"])
            self.assertEqual(1, len(zipped_metrics["aggregated_val"]["accuracy"]))
            self.assertEqual(1, len(zipped_metrics["aggregated_test"]["accuracy"]))
            self.assertEqual(1, len(zipped_metrics["clients"][0]["loss"]))
            self.assertEqual(0, len(zipped_metrics["clients"][1]["loss"]))

    def test_run_artifacts_include_metadata_provenance_fields(self) -> None:
        """Ensure run artifacts include environment and revision provenance metadata."""
        cfg = valid_run_config()
        cfg.update(
            {
                "evaluation_split_mode": "train_val_test",
                "local_model_epochs": 1,
                "weights_sending_frequency": 1,
                "num_clients": 2,
                "server_warmup_epochs": 1,
                "batch_size": 4,
                "server_learning_rate": 0.01,
                "learning_rate": 0.01,
                "momentum": 0.0,
                "seed": 123,
            }
        )

        with tempfile.TemporaryDirectory(prefix="artifact_metadata_") as tmp:
            save_path = Path(tmp) / "run"
            _, patches = self._patch_runtime(eval_mode="train_val_test")
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                self.main_module.run_experiment(cfg, str(save_path))

            metadata_path = save_path / "metadata.json"
            self.assertTrue(metadata_path.is_file())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(123, metadata["resolved_seed"])
        self.assertIsInstance(metadata.get("python_version"), str)
        self.assertTrue(metadata["python_version"])
        self.assertIsInstance(metadata.get("platform"), dict)
        self.assertIn("system", metadata["platform"])

        dependency_versions = metadata.get("dependency_versions")
        self.assertIsInstance(dependency_versions, dict)
        self.assertIn("numpy", dependency_versions)
        self.assertIn("torch", dependency_versions)
        self.assertIn("scikit-learn", dependency_versions)

        git_metadata = metadata.get("git")
        self.assertIsInstance(git_metadata, dict)
        self.assertIn("commit_hash", git_metadata)
        self.assertIn("is_dirty", git_metadata)
