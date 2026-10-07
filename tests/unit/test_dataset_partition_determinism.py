"""Tests that dataset partitioning is deterministic for a fixed seed."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tests.helpers import reload_module


class DummyFashionMNIST:
    """Tiny deterministic stand-in for FashionMNIST."""

    def __init__(self, root, train, transform, download) -> None:  # noqa: D401, ANN001
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        labels = [idx % 2 for idx in range(20 if train else 10)]
        self.targets = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        image = torch.zeros(1, 28, 28, dtype=torch.float32)
        label = torch.tensor(int(self.targets[idx]), dtype=torch.long)
        return image, label


class DummyCIFAR100:
    """Tiny deterministic stand-in for CIFAR100 (3-channel 32x32 images)."""

    def __init__(self, root, train, transform, download) -> None:  # noqa: D401, ANN001
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        # Spread labels across a few classes so the split stays meaningful.
        labels = [idx % 4 for idx in range(20 if train else 10)]
        self.targets = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        image = torch.zeros(3, 32, 32, dtype=torch.float32)
        label = torch.tensor(int(self.targets[idx]), dtype=torch.long)
        return image, label


class DummyCIFAR10:
    """Tiny deterministic stand-in for CIFAR10 (3-channel 32x32 images)."""

    def __init__(self, root, train, transform, download) -> None:  # noqa: D401, ANN001
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        labels = [idx % 4 for idx in range(20 if train else 10)]
        self.targets = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        image = torch.zeros(3, 32, 32, dtype=torch.float32)
        label = torch.tensor(int(self.targets[idx]), dtype=torch.long)
        return image, label


class DatasetPartitionDeterminismTests(unittest.TestCase):
    """Regression coverage for seeded dataset splitting and partitioning."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_mod = reload_module("utils.dataset")

    def _make_dataset(self, *, seed: int, iid: bool, imbalance_rate: float = 0.5):
        with mock.patch.object(self.dataset_mod, "FashionMNIST", DummyFashionMNIST):
            return self.dataset_mod.Dataset(
                "fashionmnist",
                num_clients=2,
                server_data_percent=0.25,
                batch_size=4,
                iid=iid,
                imbalance_rate=imbalance_rate,
                train_val_split=0.6,
                seed=seed,
            )

    def _make_cifar100_dataset(self, *, seed: int, iid: bool, imbalance_rate: float = 0.5):
        with mock.patch.object(self.dataset_mod, "CIFAR100", DummyCIFAR100):
            return self.dataset_mod.Dataset(
                "cifar100",
                num_clients=2,
                server_data_percent=0.25,
                batch_size=4,
                iid=iid,
                imbalance_rate=imbalance_rate,
                train_val_split=0.6,
                seed=seed,
            )

    def _make_cifar10_dataset(self, *, seed: int, iid: bool, imbalance_rate: float = 0.5):
        with mock.patch.object(self.dataset_mod, "CIFAR10", DummyCIFAR10):
            return self.dataset_mod.Dataset(
                "cifar10",
                num_clients=2,
                server_data_percent=0.25,
                batch_size=4,
                iid=iid,
                imbalance_rate=imbalance_rate,
                train_val_split=0.6,
                seed=seed,
            )

    @staticmethod
    def _snapshot(dataset):
        return {
            "train_indices": list(dataset.train_dataset.indices),
            # Validation is not partitioned across clients; only the whole-set split
            # (random_split of train/val) needs to stay seed-stable.
            "val_indices": list(dataset.val_dataset.indices),
            "client_train_indices": [list(indices) for indices in dataset.client_train_indices],
            "train_sampler_orders": [list(loader.sampler) for loader in dataset.client_datasets_train],
        }

    def test_fashionmnist_iid_split_is_repeatable_for_same_seed(self) -> None:
        """Ensure the FashionMNIST train/val split and IID partitioning stay seed-stable."""
        ds_a = self._make_dataset(seed=7, iid=True)
        ds_b = self._make_dataset(seed=7, iid=True)
        ds_c = self._make_dataset(seed=8, iid=True)

        snap_a = self._snapshot(ds_a)
        snap_b = self._snapshot(ds_b)
        snap_c = self._snapshot(ds_c)

        self.assertEqual(snap_a, snap_b)
        self.assertNotEqual(snap_a["train_indices"], snap_c["train_indices"])
        self.assertNotEqual(snap_a["client_train_indices"], snap_c["client_train_indices"])

    def test_fashionmnist_non_iid_split_is_repeatable_for_same_seed(self) -> None:
        """Ensure the non-IID partitioning path is driven by the configured seed."""
        ds_a = self._make_dataset(seed=11, iid=False, imbalance_rate=0.7)
        ds_b = self._make_dataset(seed=11, iid=False, imbalance_rate=0.7)
        ds_c = self._make_dataset(seed=12, iid=False, imbalance_rate=0.7)

        snap_a = self._snapshot(ds_a)
        snap_b = self._snapshot(ds_b)
        snap_c = self._snapshot(ds_c)

        self.assertEqual(snap_a, snap_b)
        self.assertNotEqual(snap_a["client_train_indices"], snap_c["client_train_indices"])
        self.assertNotEqual(snap_a["val_indices"], snap_c["val_indices"])

    def test_cifar10_iid_split_is_repeatable_for_same_seed(self) -> None:
        """Ensure the CIFAR-10 train/val split and IID partitioning stay seed-stable."""
        ds_a = self._make_cifar10_dataset(seed=7, iid=True)
        ds_b = self._make_cifar10_dataset(seed=7, iid=True)
        ds_c = self._make_cifar10_dataset(seed=8, iid=True)

        snap_a = self._snapshot(ds_a)
        snap_b = self._snapshot(ds_b)
        snap_c = self._snapshot(ds_c)

        self.assertEqual(ds_a.n_channels, 3)
        self.assertEqual(ds_a.n_classes, 10)
        self.assertEqual(snap_a, snap_b)
        self.assertNotEqual(snap_a["train_indices"], snap_c["train_indices"])
        self.assertNotEqual(snap_a["client_train_indices"], snap_c["client_train_indices"])

    def test_cifar100_iid_split_is_repeatable_for_same_seed(self) -> None:
        """Ensure the CIFAR-100 train/val split and IID partitioning stay seed-stable."""
        ds_a = self._make_cifar100_dataset(seed=7, iid=True)
        ds_b = self._make_cifar100_dataset(seed=7, iid=True)
        ds_c = self._make_cifar100_dataset(seed=8, iid=True)

        snap_a = self._snapshot(ds_a)
        snap_b = self._snapshot(ds_b)
        snap_c = self._snapshot(ds_c)

        self.assertEqual(ds_a.n_channels, 3)
        self.assertEqual(ds_a.n_classes, 100)
        self.assertEqual(snap_a, snap_b)
        self.assertNotEqual(snap_a["train_indices"], snap_c["train_indices"])
        self.assertNotEqual(snap_a["client_train_indices"], snap_c["client_train_indices"])

    def test_cifar100_non_iid_split_is_repeatable_for_same_seed(self) -> None:
        """Ensure the CIFAR-100 non-IID partitioning path is driven by the configured seed."""
        ds_a = self._make_cifar100_dataset(seed=11, iid=False, imbalance_rate=0.7)
        ds_b = self._make_cifar100_dataset(seed=11, iid=False, imbalance_rate=0.7)
        ds_c = self._make_cifar100_dataset(seed=12, iid=False, imbalance_rate=0.7)

        snap_a = self._snapshot(ds_a)
        snap_b = self._snapshot(ds_b)
        snap_c = self._snapshot(ds_c)

        self.assertEqual(snap_a, snap_b)
        self.assertNotEqual(snap_a["client_train_indices"], snap_c["client_train_indices"])
        self.assertNotEqual(snap_a["val_indices"], snap_c["val_indices"])
