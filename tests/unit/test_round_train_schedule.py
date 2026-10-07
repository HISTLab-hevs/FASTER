"""Tests for round-level train scheduling and per-round client allocation."""

from __future__ import annotations

import unittest

import numpy as np

from tests.helpers import reload_module, valid_run_config


class RoundTrainScheduleTests(unittest.TestCase):
    """Cover two-level train scheduling helpers and runtime partitioning."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule_mod = reload_module("utils.round_schedule")
        cls.dataset_mod = reload_module("utils.dataset")

    def test_normalize_round_train_schedule_config_builds_even_auto_schedule(self) -> None:
        """Ensure auto mode resolves to an even 100%-coverage schedule."""
        cfg = valid_run_config()

        resolved = self.schedule_mod.normalize_round_train_schedule_config(cfg)

        self.assertEqual("auto", cfg["round_train_schedule_mode"])
        self.assertEqual([], cfg["round_train_schedule_percentages"])
        self.assertEqual([50.0, 50.0], resolved)
        self.assertEqual(100.0, cfg["round_train_schedule_total_percentage"])
        self.assertEqual(2, cfg["round_train_schedule_rounds"])

    def test_normalize_round_client_allocation_config_builds_auto_schedule_from_base_percentages(self) -> None:
        """Ensure auto client allocation replicates the resolved base client split every round."""
        cfg = valid_run_config()

        resolved = self.schedule_mod.normalize_round_client_allocation_config(
            cfg,
            auto_client_percentages=[30.0, 70.0],
        )

        self.assertEqual("auto", cfg["round_client_allocation_mode"])
        self.assertEqual([], cfg["round_client_allocation_percentages"])
        self.assertEqual([[30.0, 70.0], [30.0, 70.0]], resolved)
        self.assertEqual(2, cfg["round_client_allocation_rounds"])
        self.assertEqual(2, cfg["round_client_allocation_num_clients"])

    def test_normalize_round_train_schedule_config_rejects_invalid_manual_total(self) -> None:
        """Ensure manual schedules must sum to 100 percent."""
        cfg = valid_run_config()
        cfg["round_train_schedule_mode"] = "manual"
        cfg["round_train_schedule_percentages"] = [55, 30]

        with self.assertRaisesRegex(ValueError, "must sum to 100"):
            self.schedule_mod.normalize_round_train_schedule_config(cfg)

    def test_normalize_round_client_allocation_config_rejects_invalid_shape(self) -> None:
        """Ensure manual client allocation rows match both rounds and clients."""
        cfg = valid_run_config()
        cfg["round_client_allocation_mode"] = "manual"
        cfg["round_client_allocation_percentages"] = [[100.0], [100.0]]

        with self.assertRaisesRegex(ValueError, "count must match num_clients"):
            self.schedule_mod.normalize_round_client_allocation_config(cfg)

    def test_build_round_train_subsets_covers_federated_train_pool_exactly_once(self) -> None:
        """Ensure round subsets partition the full federated train pool exactly once."""
        train_dataset = self.dataset_mod.CustomTabularDataset(
            np.arange(24, dtype=np.float32).reshape(12, 2),
            np.array([0, 1] * 6, dtype=np.int64),
        )
        dataset = self.dataset_mod.Dataset.__new__(self.dataset_mod.Dataset)
        dataset.train_dataset = train_dataset
        dataset.batch_size = 2
        dataset.num_clients = 3
        dataset.client_train_indices = [
            [0, 1],
            [2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ]

        round_subsets = dataset.build_round_train_subsets([25.0, 25.0, 50.0], seed=9)

        self.assertEqual(3, len(round_subsets))
        self.assertEqual(10, sum(len(indices) for indices in round_subsets))
        covered = []
        for indices in round_subsets:
            covered.extend(indices)
        self.assertCountEqual(dataset.federated_train_indices(), covered)

    def test_build_round_train_subsets_reuses_full_pool_for_no_split(self) -> None:
        """Ensure a no-split schedule (>100% total) reuses the full pool each round."""
        train_dataset = self.dataset_mod.CustomTabularDataset(
            np.arange(24, dtype=np.float32).reshape(12, 2),
            np.array([0, 1] * 6, dtype=np.int64),
        )
        dataset = self.dataset_mod.Dataset.__new__(self.dataset_mod.Dataset)
        dataset.train_dataset = train_dataset
        dataset.batch_size = 2
        dataset.num_clients = 3
        dataset.client_train_indices = [
            [0, 1],
            [2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ]

        round_subsets = dataset.build_round_train_subsets([100.0, 100.0, 100.0], seed=9)

        self.assertEqual(3, len(round_subsets))
        full_pool = dataset.federated_train_indices()
        for indices in round_subsets:
            self.assertCountEqual(full_pool, indices)

    def test_build_round_train_subsets_is_repeatable_for_same_seed(self) -> None:
        """Ensure the same seed produces the same round partition every time."""
        train_dataset = self.dataset_mod.CustomTabularDataset(
            np.arange(24, dtype=np.float32).reshape(12, 2),
            np.array([0, 1] * 6, dtype=np.int64),
        )
        dataset_a = self.dataset_mod.Dataset.__new__(self.dataset_mod.Dataset)
        dataset_a.train_dataset = train_dataset
        dataset_a.batch_size = 2
        dataset_a.num_clients = 3
        dataset_a.client_train_indices = [
            [0, 1],
            [2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ]

        dataset_b = self.dataset_mod.Dataset.__new__(self.dataset_mod.Dataset)
        dataset_b.train_dataset = train_dataset
        dataset_b.batch_size = 2
        dataset_b.num_clients = 3
        dataset_b.client_train_indices = [
            [0, 1],
            [2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ]

        round_a = dataset_a.build_round_train_subsets([25.0, 25.0, 50.0], seed=9)
        round_b = dataset_b.build_round_train_subsets([25.0, 25.0, 50.0], seed=9)

        self.assertEqual(round_a, round_b)

    def test_build_round_client_train_loaders_respects_manual_client_percentages(self) -> None:
        """Ensure one round subset is split across clients by the requested percentages."""
        train_dataset = self.dataset_mod.CustomTabularDataset(
            np.arange(40, dtype=np.float32).reshape(20, 2),
            np.array([0, 1] * 10, dtype=np.int64),
        )
        dataset = self.dataset_mod.Dataset.__new__(self.dataset_mod.Dataset)
        dataset.train_dataset = train_dataset
        dataset.batch_size = 2
        dataset.num_clients = 3

        plan = dataset.build_round_client_train_loaders(
            list(range(10)),
            [20.0, 80.0],
            seed=4,
        )

        self.assertEqual([3, 7], plan["client_sample_counts"])
        self.assertEqual([20.0, 80.0], plan["effective_client_percentages"])
        self.assertEqual([0, 1], plan["active_client_ids"])
        self.assertEqual([], plan["zero_sample_client_ids"])
        assigned = []
        for loader in plan["client_loaders"]:
            assigned.extend(list(loader.dataset.indices))
        self.assertCountEqual(list(range(10)), assigned)

    def test_build_round_client_train_loaders_is_repeatable_for_same_seed(self) -> None:
        """Ensure identical round allocations preserve sampler order for the same seed."""
        train_dataset = self.dataset_mod.CustomTabularDataset(
            np.arange(40, dtype=np.float32).reshape(20, 2),
            np.array([0, 1] * 10, dtype=np.int64),
        )
        dataset_a = self.dataset_mod.Dataset.__new__(self.dataset_mod.Dataset)
        dataset_a.train_dataset = train_dataset
        dataset_a.batch_size = 2
        dataset_a.num_clients = 3

        dataset_b = self.dataset_mod.Dataset.__new__(self.dataset_mod.Dataset)
        dataset_b.train_dataset = train_dataset
        dataset_b.batch_size = 2
        dataset_b.num_clients = 3

        plan_a = dataset_a.build_round_client_train_loaders(
            list(range(10)),
            [20.0, 80.0],
            seed=4,
        )
        plan_b = dataset_b.build_round_client_train_loaders(
            list(range(10)),
            [20.0, 80.0],
            seed=4,
        )

        self.assertEqual(plan_a["client_sample_counts"], plan_b["client_sample_counts"])
        self.assertEqual(plan_a["client_indices"], plan_b["client_indices"])
        self.assertEqual(
            [list(loader.sampler) for loader in plan_a["client_loaders"]],
            [list(loader.sampler) for loader in plan_b["client_loaders"]],
        )

    def test_build_round_client_train_loaders_keeps_positive_clients_non_empty_when_possible(self) -> None:
        """Ensure enough round samples are spread so every positive client gets one sample."""
        train_dataset = self.dataset_mod.CustomTabularDataset(
            np.arange(40, dtype=np.float32).reshape(20, 2),
            np.array([0, 1] * 10, dtype=np.int64),
        )
        dataset = self.dataset_mod.Dataset.__new__(self.dataset_mod.Dataset)
        dataset.train_dataset = train_dataset
        dataset.batch_size = 2
        dataset.num_clients = 4

        plan = dataset.build_round_client_train_loaders(
            list(range(5)),
            [1.0, 1.0, 98.0],
            seed=11,
        )

        self.assertEqual([1, 1, 3], plan["client_sample_counts"])
        self.assertEqual([0, 1, 2], plan["active_client_ids"])
        self.assertEqual([], plan["zero_sample_client_ids"])

    def test_build_round_client_train_loaders_renormalizes_when_churn_removes_clients(self) -> None:
        """Ensure inactive clients yield zero allocation and eligible clients receive the full round subset."""
        train_dataset = self.dataset_mod.CustomTabularDataset(
            np.arange(24, dtype=np.float32).reshape(12, 2),
            np.array([0, 1] * 6, dtype=np.int64),
        )
        dataset = self.dataset_mod.Dataset.__new__(self.dataset_mod.Dataset)
        dataset.train_dataset = train_dataset
        dataset.batch_size = 2
        dataset.num_clients = 4

        plan = dataset.build_round_client_train_loaders(
            list(range(9)),
            [20.0, 30.0, 50.0],
            eligible_clients=[0, 2],
            seed=7,
        )

        self.assertEqual([28.571429, 0.0, 71.428571], plan["effective_client_percentages"])
        self.assertEqual([3, 0, 6], plan["client_sample_counts"])
        self.assertEqual([0, 2], plan["active_client_ids"])
        self.assertEqual([1], plan["zero_sample_client_ids"])

    def test_build_round_client_train_loaders_preserves_fixed_validation_loaders(self) -> None:
        """Ensure round scheduling only affects train subsets, not validation holders."""
        val_dataset = self.dataset_mod.CustomTabularDataset(
            np.arange(12, dtype=np.float32).reshape(6, 2),
            np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
        )
        train_dataset = self.dataset_mod.CustomTabularDataset(
            np.arange(24, dtype=np.float32).reshape(12, 2),
            np.array([0, 1] * 6, dtype=np.int64),
        )
        dataset = self.dataset_mod.Dataset.__new__(self.dataset_mod.Dataset)
        dataset.train_dataset = train_dataset
        dataset.val_dataset = val_dataset
        dataset.batch_size = 2
        dataset.num_clients = 3
        # Validation is a single whole-set loader; round scheduling must not touch it.
        dataset.val_data = self.dataset_mod.DataLoader(val_dataset, batch_size=2)

        before_val_data = dataset.val_data
        dataset.build_round_client_train_loaders(list(range(8)), [50.0, 50.0], seed=3)

        self.assertIs(before_val_data, dataset.val_data)
