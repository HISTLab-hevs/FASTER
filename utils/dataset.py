"""Dataset loading, client partitioning, and per-round loader construction."""

import os
import random
import csv
import json
import math
from collections import Counter

import numpy as np
import torch
from matplotlib import pyplot as plt
from medmnist import (
    PathMNIST,
    DermaMNIST,
    OCTMNIST,
    BloodMNIST,
    TissueMNIST,
    OrganAMNIST,
    OrganCMNIST,
    OrganSMNIST,
    PneumoniaMNIST,
)
from torchvision.datasets import FashionMNIST, CIFAR100, CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import INFO
from PIL import Image

from utils.config import load_config


def _standard_dataset_root():
    """Return the shared cache directory for the built-in downloadable datasets.

    Jobs are executed with their own per-job working directory, so a relative
    "data" path would resolve to a fresh empty folder for every run and force a
    re-download on each run and repeat. Anchoring on the repository root (or on
    ``FASTER_DATA_DIR`` when set) keeps one cache shared by every run, so a
    built-in dataset is downloaded once and reused afterwards.
    """
    env_root = os.environ.get("FASTER_DATA_DIR", "").strip()
    if env_root:
        return os.path.abspath(env_root)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class CustomTabularDataset(torch.utils.data.Dataset):
    """Simple tensor-backed tabular dataset with labels metadata."""

    def __init__(self, features, labels):
        """Store tabular features and labels as tensors/arrays."""
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return one ``(features, label)`` sample pair."""
        x = self.features[idx]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class CustomImageDataset(torch.utils.data.Dataset):
    """Tensor-backed image dataset for uploaded NPZ arrays."""

    def __init__(self, images, labels):
        """Normalize image arrays and store them with integer labels."""
        x = np.asarray(images)
        if x.ndim == 3:
            x = x[:, None, :, :]
        elif x.ndim == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
            x = np.transpose(x, (0, 3, 1, 2))
        if x.ndim != 4:
            raise ValueError("Custom image dataset expects rank-3 or rank-4 arrays")
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if x.max() > 1.0:
            x = x / 255.0

        self.images = torch.tensor(x, dtype=torch.float32)
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(self.images) != len(self.labels):
            raise ValueError("Custom image dataset images/labels length mismatch")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return one ``(image, label)`` sample pair."""
        return self.images[idx], torch.tensor(self.labels[idx], dtype=torch.long)


class Dataset:
    """Load one dataset and partition it across the server and federated clients.

    Holds the server, validation, and test loaders plus the per-client index
    lists, and builds the per-round training subsets used by the aggregation
    loop.
    """

    def __init__(
        self, dataset_name, num_clients, server_data_percent, batch_size, iid=True, imbalance_rate=0.5, train_val_split=0.9,
        seed=0,
        custom_dataset_path=None
    ):
        """Load a dataset and partition it into server/client data loaders."""
        self.dataset_name = dataset_name
        self.custom_dataset_path = custom_dataset_path
        if dataset_name == "custom_csv":
            self.info = {"description": "User uploaded custom CSV/TSV dataset"}
            self._load_custom_dataset()
        elif dataset_name == "custom_image_npz":
            self.info = {"description": "User uploaded image NPZ dataset"}
            self._load_custom_image_dataset()
        elif dataset_name == "custom_image_folder":
            self.info = {"description": "User uploaded directory-based image dataset"}
            self._load_custom_image_folder_dataset()
        elif dataset_name in INFO:
            self.info = INFO[dataset_name]
            self.n_channels = self.info["n_channels"]
            self.n_classes = len(self.info["label"])
        elif dataset_name == "fashionmnist":
            self.info = {"description": "Zalando’s Fashion-MNIST dataset"}
            self.n_channels = 1
            self.n_classes = 10
        elif dataset_name == "cifar100":
            self.info = {"description": "CIFAR-100 dataset"}
            self.n_channels = 3
            self.n_classes = 100
        elif dataset_name == "cifar10":
            self.info = {"description": "CIFAR-10 dataset"}
            self.n_channels = 3
            self.n_classes = 10
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        # Index 0 is reserved for the server split; user clients start at index 1.
        self.num_clients = num_clients + 1
        self.server_data_percentage = server_data_percent
        self.batch_size = batch_size
        self.iid = iid
        self.seed = int(seed or 0)
        if not (0 < imbalance_rate < 1):
            raise ValueError(f"imbalance_rate must be between 0 and 1 (exclusive), got {imbalance_rate}")
        self.imbalance_rate = imbalance_rate
        self.train_val_split = train_val_split
        self.client_datasets_train = []
        self.client_train_indices = []

        self.dataset_dict = {
            "pathmnist": PathMNIST,
            "dermamnist": DermaMNIST,
            "octmnist": OCTMNIST,
            "bloodmnist": BloodMNIST,
            "tissuemnist": TissueMNIST,
            "pneumoniamnist": PneumoniaMNIST,
            "organamnist": OrganAMNIST,
            "organcmnist": OrganCMNIST,
            "organsmnist": OrganSMNIST,
            "fashionmnist": FashionMNIST,
            "cifar100": CIFAR100,
            "cifar10": CIFAR10,
            "custom_csv": None,
            "custom_image_npz": None,
            "custom_image_folder": None,
        }

        self.dataset = self.dataset_dict[dataset_name]

        if self.dataset is None and dataset_name not in {"custom_csv", "custom_image_npz", "custom_image_folder"}:
            raise ValueError(f"Dataset {dataset_name} not found")

        # Apply the default tensor conversion and normalization transform.
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        data_dir = _standard_dataset_root()
        os.makedirs(data_dir, exist_ok=True)
        medmnist_root = data_dir
        medmnist_npz = os.path.join(medmnist_root, f"{dataset_name}.npz")

        # Load the selected built-in or user-provided dataset.
        if dataset_name in {"custom_csv", "custom_image_npz", "custom_image_folder"}:
            pass
        elif dataset_name == "fashionmnist":
            full_train = FashionMNIST(root=data_dir, train=True, transform=self.transform, download=True)
            train_size = int(self.train_val_split * len(full_train))
            val_size = len(full_train) - train_size

            split_generator = self._torch_generator(offset=0)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_train,
                [train_size, val_size],
                generator=split_generator,
            )

            # Expose labels on split datasets for metric and distribution helpers.
            full_train_targets = np.asarray(full_train.targets)
            self.train_dataset.labels = full_train_targets[self.train_dataset.indices]
            self.val_dataset.labels = full_train_targets[self.val_dataset.indices]

            self.test_dataset = FashionMNIST(root=data_dir, train=False, transform=self.transform, download=True)
            self.test_dataset.labels = np.asarray(self.test_dataset.targets)
        elif dataset_name in {"cifar100", "cifar10"}:
            # CIFAR-10/100 share FashionMNIST's train=True/False + .targets API.
            cifar_cls = self.dataset
            full_train = cifar_cls(root=data_dir, train=True, transform=self.transform, download=True)
            train_size = int(self.train_val_split * len(full_train))
            val_size = len(full_train) - train_size

            split_generator = self._torch_generator(offset=0)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_train,
                [train_size, val_size],
                generator=split_generator,
            )

            # Expose labels on split datasets for metric and distribution helpers.
            full_train_targets = np.asarray(full_train.targets)
            self.train_dataset.labels = full_train_targets[self.train_dataset.indices]
            self.val_dataset.labels = full_train_targets[self.val_dataset.indices]

            self.test_dataset = cifar_cls(root=data_dir, train=False, transform=self.transform, download=True)
            self.test_dataset.labels = np.asarray(self.test_dataset.targets)
        elif dataset_name not in {"custom_csv", "custom_image_npz", "custom_image_folder"}:
            if not os.path.isfile(medmnist_npz):
                print(f"MedMNIST local file not found: {medmnist_npz}")
                print("Trying automatic download from Zenodo via MedMNIST...")

            try:
                self.train_dataset = self.dataset(
                    split="train", transform=self.transform, download=True, root=data_dir
                )
                self.val_dataset = self.dataset(
                    split="val", transform=self.transform, download=True, root=data_dir
                )
                self.test_dataset = self.dataset(
                    split="test", transform=self.transform, download=True, root=data_dir
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load MedMNIST dataset '{dataset_name}'. Expected local file at '{medmnist_npz}'. "
                    "If the automatic download fails, download the .npz file manually and place it in the data/ folder."
                ) from exc

        # Collect class-specific indices for training partitioning only. IID/non-IID
        # describes how clients see TRAINING data; the whole validation set evaluates
        # the global model and stays unpartitioned.
        class_indices_train = {i: [] for i in range(int(self.n_classes))}
        for idx, target in enumerate(self.train_dataset.labels):
            cls = int(target.item()) if hasattr(target, "item") else int(target)
            if cls in class_indices_train:
                class_indices_train[cls].append(idx)

        # Partition only the training data across server/clients per the IID setting.
        if self.iid:
            self._split_iid_train_data("train", class_indices_train)
        else:
            self._split_non_iid_train_data("train", class_indices_train, self.imbalance_rate)

        # Validation and test stay whole: they evaluate the aggregated global model
        # (and drive FedGP fitness) on the full set.
        self.val_data = DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            generator=self._torch_generator(offset=8000),
        )
        self.test_data = DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=True,
            generator=self._torch_generator(offset=9000),
        )

    def _split_seed(self, offset=0):
        """Return one deterministic seed derived from the run seed."""
        return int(getattr(self, "seed", 0) or 0) + int(offset)

    def _python_rng(self, offset=0):
        """Return a Python RNG scoped to one split or loader."""
        return random.Random(self._split_seed(offset))

    def _torch_generator(self, offset=0):
        """Return one seeded torch generator scoped to the current dataset."""
        generator = torch.Generator()
        generator.manual_seed(self._split_seed(offset))
        return generator

    def _np_rng(self, offset=0):
        """Return one seeded NumPy generator scoped to the current dataset."""
        return np.random.default_rng(self._split_seed(offset))

    def _load_custom_dataset(self):
        """Load a normalized custom CSV/TSV from disk with explicit split column."""
        p = self.custom_dataset_path
        if not p or not os.path.isfile(p):
            raise ValueError("custom_dataset_path is missing or invalid for dataset_name=custom_csv")

        ext = os.path.splitext(p)[1].lower()
        delim = "\t" if ext == ".tsv" else ","
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delim)
            rows = list(reader)

        if len(rows) < 2:
            raise ValueError("Custom dataset file must contain header and rows")

        header = [str(c or "").strip() for c in rows[0]]
        lower = [h.lower() for h in header]
        label_idx = next((i for i, h in enumerate(lower) if h in {"label", "target", "class", "y"}), len(header) - 1)
        split_idx = next((i for i, h in enumerate(lower) if h in {"split", "set", "partition"}), -1)
        if split_idx < 0:
            raise ValueError("Custom tabular dataset must include a split column with train/val/test values")

        feat_idx = [i for i in range(len(header)) if i not in {label_idx, split_idx}]
        if not feat_idx:
            raise ValueError("Custom dataset requires at least one feature column")

        feats = []
        labels_raw = []
        splits_raw = []
        split_alias = {
            "train": "train",
            "training": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "dev": "val",
            "test": "test",
            "testing": "test",
        }
        for line_no, row in enumerate(rows[1:], start=2):
            if len(row) != len(header):
                raise ValueError(f"Row {line_no} has wrong column count")
            try:
                feats.append([float(str(row[i]).strip()) for i in feat_idx])
            except Exception:
                raise ValueError(f"Row {line_no} contains non numeric features")
            lab = str(row[label_idx]).strip()
            if lab == "":
                raise ValueError(f"Row {line_no} has empty label")
            split_raw = str(row[split_idx]).strip().lower()
            split = split_alias.get(split_raw)
            if split is None:
                raise ValueError(f"Row {line_no} has invalid split '{row[split_idx]}'. Use train/val/test")
            labels_raw.append(lab)
            splits_raw.append(split)

        label_map = {}
        labels = []
        for lab in labels_raw:
            if lab not in label_map:
                label_map[lab] = len(label_map)
            labels.append(label_map[lab])

        if len(set(labels)) < 2:
            raise ValueError("Custom dataset must contain at least 2 classes")

        X = np.asarray(feats, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)

        self.n_channels = X.shape[1]
        self.n_classes = int(len(np.unique(y)))

        split_arr = np.asarray(splits_raw)
        train_idx = np.where(split_arr == "train")[0]
        val_idx = np.where(split_arr == "val")[0]
        test_idx = np.where(split_arr == "test")[0]
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise ValueError("Custom dataset split must include non-empty train, val and test subsets")

        self.train_dataset = CustomTabularDataset(X[train_idx], y[train_idx])
        self.val_dataset = CustomTabularDataset(X[val_idx], y[val_idx])
        self.test_dataset = CustomTabularDataset(X[test_idx], y[test_idx])

    def _load_custom_image_dataset(self):
        """Load custom NPZ image dataset with either split keys or flat images/labels."""
        p = self.custom_dataset_path
        if not p or not os.path.isfile(p):
            raise ValueError("custom_dataset_path is missing or invalid for dataset_name=custom_image_npz")

        try:
            npz = np.load(p, allow_pickle=False)
            keys = set(npz.keys())
        except Exception:
            raise ValueError("Invalid npz dataset file")

        def _arr(name):
            """Return one NPZ array as a NumPy array when the key exists."""
            return np.asarray(npz[name]) if name in npz else None

        if {"train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels"}.issubset(keys):
            x_train, y_train = _arr("train_images"), _arr("train_labels")
            x_val, y_val = _arr("val_images"), _arr("val_labels")
            x_test, y_test = _arr("test_images"), _arr("test_labels")
        elif {"x_train", "y_train", "x_val", "y_val", "x_test", "y_test"}.issubset(keys):
            x_train, y_train = _arr("x_train"), _arr("y_train")
            x_val, y_val = _arr("x_val"), _arr("y_val")
            x_test, y_test = _arr("x_test"), _arr("y_test")
        else:
            raise ValueError("NPZ must provide explicit train/val/test splits (train_images/val_images/test_images or x_train/x_val/x_test)")

        y_train = np.asarray(y_train).reshape(-1)
        y_val = np.asarray(y_val).reshape(-1)
        y_test = np.asarray(y_test).reshape(-1)
        if y_train.size == 0 or y_val.size == 0 or y_test.size == 0:
            raise ValueError("Custom image split must include non-empty train, val and test subsets")
        all_labels = np.concatenate([y_train, y_val, y_test], axis=0)
        if np.unique(all_labels).size < 2:
            raise ValueError("Custom image dataset must include at least 2 classes")

        self.train_dataset = CustomImageDataset(x_train, y_train)
        self.val_dataset = CustomImageDataset(x_val, y_val)
        self.test_dataset = CustomImageDataset(x_test, y_test)
        self.n_channels = int(self.train_dataset.images.shape[1])
        self.n_classes = int(np.unique(all_labels).size)

    @staticmethod
    def _custom_dataset_sidecar_path(custom_dataset_path):
        """Return the sidecar metadata path used by the web backend."""
        return f"{custom_dataset_path}.meta.json"

    def _load_custom_image_folder_dataset(self):
        """Load a directory-based custom image dataset from persisted split metadata."""
        dataset_root = self.custom_dataset_path
        if not dataset_root or not os.path.isdir(dataset_root):
            raise ValueError(
                "custom_dataset_path is missing or invalid for dataset_name=custom_image_folder"
            )

        sidecar_path = self._custom_dataset_sidecar_path(dataset_root)
        if not os.path.isfile(sidecar_path):
            raise ValueError("Image-folder dataset metadata sidecar is missing")

        try:
            with open(sidecar_path, "r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
        except Exception as exc:
            raise ValueError(f"Failed to read image-folder dataset metadata: {exc}")

        split_index = sidecar.get("split_index")
        class_names = sidecar.get("class_names")
        if not isinstance(split_index, dict) or not isinstance(class_names, list):
            raise ValueError("Image-folder dataset metadata is incomplete")

        def _load_split(split_name):
            rows = split_index.get(split_name)
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"Image-folder dataset split '{split_name}' is missing or empty")

            images = []
            labels = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"Image-folder dataset split '{split_name}' contains invalid index rows")
                rel_path = str(row.get("path") or "").replace("/", os.sep)
                label = row.get("label")
                abs_path = os.path.join(dataset_root, rel_path)
                if not os.path.isfile(abs_path):
                    raise ValueError(f"Indexed image '{row.get('path')}' is missing from split '{split_name}'")
                try:
                    with Image.open(abs_path) as image:
                        arr = np.asarray(image)
                except Exception as exc:
                    raise ValueError(f"Failed to read indexed image '{row.get('path')}': {exc}")
                images.append(arr)
                labels.append(int(label))
            return np.asarray(images), np.asarray(labels, dtype=np.int64)

        x_train, y_train = _load_split("train")
        x_val, y_val = _load_split("val")
        x_test, y_test = _load_split("test")

        all_labels = np.concatenate([y_train, y_val, y_test], axis=0)
        if np.unique(all_labels).size < 2:
            raise ValueError("Image-folder dataset must include at least 2 classes")

        self.train_dataset = CustomImageDataset(x_train, y_train)
        self.val_dataset = CustomImageDataset(x_val, y_val)
        self.test_dataset = CustomImageDataset(x_test, y_test)
        self.n_channels = int(self.train_dataset.images.shape[1])
        self.n_classes = int(len(class_names))

    def _rebalance_empty_clients(self, client_datasets):
        """Best-effort rebalancing to reduce empty subsets by borrowing from largest subset."""
        total = sum(len(c) for c in client_datasets)
        if total < self.num_clients:
            return client_datasets
        for i in range(self.num_clients):
            if client_datasets[i]:
                continue
            donor = max(range(self.num_clients), key=lambda idx: len(client_datasets[idx]))
            if len(client_datasets[donor]) <= 1:
                continue
            client_datasets[i].append(client_datasets[donor].pop())
        return client_datasets

    def _build_client_loaders(self, dataset_obj, client_datasets):
        """Create DataLoaders safely even when some subsets are empty."""
        out = []
        for client_indices in client_datasets:
            subset = torch.utils.data.Subset(dataset_obj, client_indices)
            loader_kwargs = {
                "batch_size": self.batch_size,
                "shuffle": (len(client_indices) > 0),
            }
            if len(client_indices) > 0:
                loader_kwargs["generator"] = self._torch_generator(offset=1000 + len(out))
            out.append(
                DataLoader(
                    subset,
                    **loader_kwargs,
                )
            )
        return out

    def _split_iid_train_data(self, train_or_val, class_indices):
        """Split one dataset partition into IID server/client subsets."""
        client_datasets = [[] for _ in range(self.num_clients)]

        # server_data_percentage governs the training split: it sets the share of
        # training data kept by the server (index 0, used for the warm-up). Validation
        # and test stay whole for global evaluation. Kept as a single knob in case
        # server-side val/test becomes useful later.
        if self.server_data_percentage is False:
            server_data_percentage = 1 / self.num_clients
        else:
            server_data_percentage = self.server_data_percentage

        rng = self._python_rng(offset=0 if train_or_val == "train" else 1)
        for class_id, indices in class_indices.items():
            rng.shuffle(indices)

            total_data = len(indices)
            server_data_count = int(total_data * server_data_percentage)
            client_data_count = (total_data - server_data_count) // (
                self.num_clients - 1
            )

            client_datasets[0].extend(indices[:server_data_count])

            start_idx = server_data_count
            for i in range(1, self.num_clients):
                end_idx = start_idx + client_data_count
                client_datasets[i].extend(indices[start_idx:end_idx])
                start_idx = end_idx

            # Keep class remainders in the server split so no samples are dropped.
            remaining_data = indices[start_idx:]
            if remaining_data:
                client_datasets[0].extend(remaining_data)

        client_datasets = self._rebalance_empty_clients(client_datasets)
        self.client_train_indices = [list(indices) for indices in client_datasets]
        self.client_datasets_train = self._build_client_loaders(self.train_dataset, client_datasets)

    def _split_non_iid_train_data(self, train_or_val, class_indices, imbalance_rate=0.5):
        """Split one dataset partition into non-IID server/client subsets."""
        if not (0 < imbalance_rate < 1):
            raise ValueError(f"imbalance_rate must be between 0 and 1 (exclusive), got {imbalance_rate}")
        
        client_datasets = [[] for _ in range(self.num_clients)]

        if self.server_data_percentage is False:
            server_data_percentage = 1 / self.num_clients
        else:
            server_data_percentage = self.server_data_percentage

        num_shards_per_client = 3

        shard_indices = []
        rng = self._python_rng(offset=10 if train_or_val == "train" else 11)
        np_rng = self._np_rng(offset=20 if train_or_val == "train" else 21)
        for class_id, indices in class_indices.items():
            rng.shuffle(indices)
            shard_indices += np.array_split(
                indices, num_shards_per_client * self.num_clients
            )

        rng.shuffle(shard_indices)

        # Lower concentration produces stronger client-level class imbalance.
        concentration = 1.0 - imbalance_rate
        weights = np_rng.dirichlet(np.ones(self.num_clients) * concentration, size=1).flatten()

        total_shards = len(shard_indices)
        server_shards_count = int(total_shards * server_data_percentage)

        for i in range(server_shards_count):
            client_datasets[0].extend(shard_indices.pop(0))

        while shard_indices:
            for i in range(1, self.num_clients):
                if shard_indices:
                    client_shards_count = max(
                        1, int(total_shards * weights[i])
                    )
                    for _ in range(client_shards_count):
                        if shard_indices:
                            client_datasets[i].extend(shard_indices.pop(0))

        client_datasets = self._rebalance_empty_clients(client_datasets)
        self.client_train_indices = [list(indices) for indices in client_datasets]
        self.client_datasets_train = self._build_client_loaders(self.train_dataset, client_datasets)

    @staticmethod
    def _round_subset_counts(total_samples, round_percentages):
        """Return one exact sample-count schedule covering the full subset."""
        if total_samples <= 0:
            return [0 for _ in round_percentages]

        exact_counts = [
            float(total_samples) * (float(percentage) / 100.0)
            for percentage in round_percentages
        ]
        base_counts = [int(math.floor(value)) for value in exact_counts]
        remainder = int(total_samples - sum(base_counts))
        fractional = sorted(
            (
                (exact_counts[idx] - base_counts[idx], idx)
                for idx in range(len(exact_counts))
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for _, idx in fractional[:remainder]:
            base_counts[idx] += 1
        return base_counts

    def federated_train_indices(self):
        """Return the full federated train pool, excluding the server warm-up split."""
        indices = []
        for client_indices in self.client_train_indices[1:]:
            indices.extend(int(idx) for idx in client_indices)
        return indices

    def federated_client_train_sample_counts(self):
        """Return the base train sample counts for user clients only."""
        return [int(len(indices)) for indices in self.client_train_indices[1:]]

    def federated_client_train_percentages(self):
        """Return the normalized client percentages implied by the base train split."""
        counts = self.federated_client_train_sample_counts()
        total = sum(counts)
        if total <= 0:
            if not counts:
                return []
            base = round(100.0 / float(len(counts)), 6)
            even = [base for _ in counts]
            even[-1] = round(100.0 - sum(even[:-1]), 6)
            return even

        exact_percentages = [
            float(count) * 100.0 / float(total)
            for count in counts
        ]
        rounded = [round(value, 6) for value in exact_percentages]
        if rounded:
            rounded[-1] = round(100.0 - sum(rounded[:-1]), 6)
        return rounded

    def build_round_train_subsets(self, round_percentages, seed=0):
        """Partition the full federated train pool into deterministic round subsets."""
        normalized_percentages = [float(value) for value in round_percentages]
        if not normalized_percentages:
            raise ValueError("round_percentages is required")

        shuffled_indices = list(self.federated_train_indices())
        rng = random.Random(int(seed))
        rng.shuffle(shuffled_indices)

        # "No split" schedules resolve every round to full (100%) coverage, so the
        # total exceeds 100%. In that case each round reuses the entire pool rather
        # than receiving a disjoint slice.
        if sum(normalized_percentages) > 100.0 + 1e-6:
            return [list(shuffled_indices) for _ in normalized_percentages]

        round_counts = self._round_subset_counts(
            len(shuffled_indices),
            normalized_percentages,
        )
        round_subsets = []
        cursor = 0
        for round_count in round_counts:
            next_cursor = cursor + int(round_count)
            round_subsets.append(shuffled_indices[cursor:next_cursor])
            cursor = next_cursor
        return round_subsets

    def build_round_client_train_loaders(
        self,
        round_indices,
        client_percentages,
        *,
        eligible_clients=None,
        seed=0,
    ):
        """Allocate one round subset across clients and build train loaders."""
        normalized_percentages = [float(value) for value in client_percentages]
        if len(normalized_percentages) != self.num_clients - 1:
            raise ValueError("client_percentages count must match num_clients")

        eligible_set = (
            set(range(self.num_clients - 1))
            if eligible_clients is None
            else {int(client_id) for client_id in eligible_clients if 0 <= int(client_id) < self.num_clients - 1}
        )

        effective_percentages = [0.0 for _ in normalized_percentages]
        eligible_total = sum(
            normalized_percentages[client_id]
            for client_id in eligible_set
            if normalized_percentages[client_id] > 0
        )
        if eligible_set and eligible_total > 0:
            for client_id in sorted(eligible_set):
                base_value = normalized_percentages[client_id]
                if base_value > 0:
                    effective_percentages[client_id] = (
                        base_value * 100.0 / eligible_total
                    )
        elif eligible_set:
            even_share = 100.0 / float(len(eligible_set))
            for client_id in sorted(eligible_set):
                effective_percentages[client_id] = even_share

        rounded_effective = [round(value, 6) for value in effective_percentages]
        positive_clients = [
            idx for idx, value in enumerate(rounded_effective)
            if value > 0
        ]
        if positive_clients:
            last_client = positive_clients[-1]
            rounded_effective[last_client] = round(
                100.0
                - sum(rounded_effective[idx] for idx in positive_clients[:-1]),
                6,
            )

        shuffled_indices = list(round_indices)
        rng = random.Random(int(seed))
        rng.shuffle(shuffled_indices)
        total_round_samples = len(shuffled_indices)
        client_counts = [0 for _ in rounded_effective]
        if positive_clients and total_round_samples >= len(positive_clients):
            # When a round has enough samples, keep every eligible client with a
            # positive configured share on the board by reserving one sample for
            # each of them before applying largest-remainder rounding.
            for client_id in positive_clients:
                client_counts[client_id] = 1

            remaining_samples = total_round_samples - len(positive_clients)
            if remaining_samples > 0:
                remaining_percentages = [
                    rounded_effective[client_id]
                    for client_id in positive_clients
                ]
                remaining_counts = self._round_subset_counts(
                    remaining_samples,
                    remaining_percentages,
                )
                for offset, client_id in enumerate(positive_clients):
                    client_counts[client_id] += int(remaining_counts[offset])
        else:
            client_counts = self._round_subset_counts(
                total_round_samples,
                rounded_effective,
            )

        client_indices = []
        client_loaders = []
        cursor = 0
        for client_id, client_count in enumerate(client_counts):
            next_cursor = cursor + int(client_count)
            assigned_indices = shuffled_indices[cursor:next_cursor]
            cursor = next_cursor
            client_indices.append(assigned_indices)
            subset = torch.utils.data.Subset(self.train_dataset, assigned_indices)
            loader_kwargs = {
                "batch_size": self.batch_size,
                "shuffle": (len(assigned_indices) > 0),
            }
            if len(assigned_indices) > 0:
                loader_kwargs["generator"] = torch.Generator().manual_seed(int(seed) + client_id)
            client_loaders.append(
                DataLoader(
                    subset,
                    **loader_kwargs,
                )
            )

        active_client_ids = [
            client_id for client_id, count in enumerate(client_counts)
            if int(count) > 0
        ]

        return {
            "client_loaders": client_loaders,
            "client_indices": client_indices,
            "configured_client_percentages": [round(value, 6) for value in normalized_percentages],
            "effective_client_percentages": rounded_effective,
            "client_sample_counts": [int(len(indices)) for indices in client_indices],
            "active_client_ids": active_client_ids,
            "zero_sample_client_ids": [
                client_id for client_id, count in enumerate(client_counts)
                if int(count) <= 0
            ],
        }

    def print_class_distribution(self, train_or_val, save_path=None, show=False):
        """Print and optionally persist per-client class counts for the training split."""
        # Only the training split is partitioned across clients; validation stays whole.
        if train_or_val != "train":
            return
        if save_path:
            save_path = os.path.join(save_path, "class_distribution.txt")

        client_datasets = self.client_datasets_train
        dataset = self.train_dataset

        labels = np.array([dataset.labels[i] for i in range(len(dataset))])
        total = 0

        with open(save_path, "a") as f:
            f.write("Train\n")

        for client_id, client_dataset in enumerate(client_datasets):
            client_indices = client_dataset.dataset.indices
            client_targets = labels[client_indices]
            class_count = Counter([int(target.item()) for target in client_targets])

            sorted_class_count = dict(sorted(class_count.items()))

            if save_path:
                with open(save_path, "a") as f:
                    f.write(f"Client {client_id}: {sorted_class_count}\n")
            if show:
                print(f"Client {client_id}: {sorted_class_count}")
            total += sum(sorted_class_count.values())

        if save_path:
            with open(save_path, "a") as f:
                f.write(f"Total: {total}\n")
        if show:
            print(f"Total: {total}")

    def plot_class_distribution(self, train_or_val, save_path=None, show=False):
        """Plot per-client class counts for the training split as a grouped bar chart."""
        # Only the training split is partitioned across clients; validation stays whole.
        if train_or_val != "train":
            return

        # Normalize the output filename to SVG when saving plots.
        if save_path:
            save_path = os.path.join(save_path, "class_distribution_train.svg")

        client_datasets = self.client_datasets_train
        dataset = self.train_dataset

        class_counts_per_client = {i: [] for i in range(self.num_clients)}
        n_classes_plot = int(getattr(self, 'n_classes', 0) or 0)
        if n_classes_plot <= 0:
            n_classes_plot = len(np.unique(self.train_dataset.labels))

        labels = np.array([dataset.labels[i] for i in range(len(dataset))])

        # Count per-class samples for every client subset.
        for client_id, client_dataset in enumerate(client_datasets):
            client_indices = client_dataset.dataset.indices
            client_targets = labels[client_indices]
            class_count = Counter([int(target.item()) for target in client_targets])

            # Store the count for each class, including zeros.
            for class_id in range(n_classes_plot):
                class_counts_per_client[client_id].append(class_count[class_id])

        classes = list(range(n_classes_plot))
        fig, ax = plt.subplots(figsize=(12, 7))

        colors = plt.colormaps.get_cmap("tab20")

        bar_width = 0.8 / self.num_clients
        offsets = np.arange(len(classes))

        for client_id in range(self.num_clients):
            label = "Server" if client_id == 0 else f"Client {client_id}"
            ax.bar(
                offsets + client_id * bar_width,
                class_counts_per_client[client_id],
                width=bar_width,
                label=label,
                color=colors(client_id),
                edgecolor="black",
            )

        ax.set_xlabel("Class", fontsize=22)
        ax.set_ylabel("Number of Samples", fontsize=22)
        ax.set_xticks(offsets + bar_width * (self.num_clients - 1) / 2)
        ax.set_xticklabels(classes, fontsize=22)
        ax.tick_params(axis="y", labelsize=22)
        ax.legend(fontsize=16)

        if save_path:
            fig.savefig(save_path, format="svg", bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def dataset_info(self):
        """Print the current dataset metadata and split sizes."""
        print(f"Dataset: {self.dataset_name}")
        print(f"Dataset description: {self.info['description']}")
        print(f"Task: {self.info['task']}")
        print(f"Classes: {self.info['label']}")
        print(f"Channels: {self.n_channels}")
        print(f"License: {self.info['license']}")

        total = (
            self.info["n_samples"]["train"]
            + self.info["n_samples"]["val"]
            + self.info["n_samples"]["test"]
        )
        print(
            f"Train: {self.info['n_samples']['train']} - {round(self.info['n_samples']['train'] * 100 / total, 2)}"
        )
        print(
            f"Val: {self.info['n_samples']['val']} - {round(self.info['n_samples']['val'] * 100 / total, 2)}"
        )
        print(
            f"Test:{self.info['n_samples']['test']} - {round(self.info['n_samples']['test'] * 100 / total, 2)}"
        )
        print(f"Total: {total}")


if __name__ == "__main__":
    config = load_config("config.yaml")

    dataset = Dataset(
        config["dataset_name"],
        config["num_clients"],
        config["server_data_percentage"],
        config["batch_size"],
        config["iid"],
        config.get("imbalance_rate", 0.5),
        seed=int(config.get("seed", 0) or 0),
    )
    print("Train data:")
    dataset.print_class_distribution("train", show=True)
    print("Validation data:")
    dataset.print_class_distribution("val", show=True)

    dataset.plot_class_distribution("train", show=True)
    dataset.plot_class_distribution("val", show=True)
