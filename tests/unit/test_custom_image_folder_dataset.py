"""Tests for directory-based image dataset ingestion and loaders."""

from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np
from PIL import Image

from tests.helpers import reload_module


def _write_image(path: str, value: int) -> None:
    arr = np.full((6, 6, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _build_image_folder_zip(tmpdir: str, name: str) -> str:
    root_dir = os.path.join(tmpdir, "image_folder_source")
    archive_root = os.path.join(root_dir, "dataset_root")
    os.makedirs(archive_root, exist_ok=True)
    for split in ("train", "val", "test"):
        for class_name, value in (("class_b", 120), ("class_a", 40)):
            class_dir = os.path.join(archive_root, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for idx in range(2):
                _write_image(os.path.join(class_dir, f"{class_name}_{idx}.png"), value + idx)

    zip_path = os.path.join(tmpdir, name)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for base, _, files in os.walk(root_dir):
            for file_name in sorted(files):
                abs_path = os.path.join(base, file_name)
                archive.write(abs_path, os.path.relpath(abs_path, root_dir))
    return zip_path


class CustomImageFolderDatasetTests(unittest.TestCase):
    """Cover image-folder dataset ingestion and loader behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.service_mod = reload_module("web_backend.service")
        cls.datasets_service_mod = reload_module("web_backend.services.datasets")
        cls.dataset_mod = reload_module("utils.dataset")

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        self.project_root_patch = mock.patch.object(
            self.service_mod,
            "_PROJECT_ROOT",
            self.tmpdir.name,
        )
        self.project_root_patch.start()
        self.addCleanup(self.project_root_patch.stop)

        self.service = self.service_mod.svc
        self.service._db_enabled = False
        self.service._custom_datasets_repo = None

    def test_dataset_loader_reuses_persisted_sidecar_index_for_image_folders(self) -> None:
        """Ensure the dataset loader rebuilds splits from the persisted sidecar index."""
        zip_path = _build_image_folder_zip(self.tmpdir.name, "vision_dataset.zip")
        meta = self.service.save_custom_dataset_file(
            username="alice",
            original_name="vision_dataset.zip",
            source_path=zip_path,
        )
        dataset_root = self.service.resolve_custom_dataset_path("alice", meta["dataset_ref"])
        self.assertTrue(dataset_root)

        dataset = self.dataset_mod.Dataset(
            "custom_image_folder",
            num_clients=2,
            server_data_percent=0.25,
            batch_size=2,
            iid=True,
            custom_dataset_path=dataset_root,
        )

        self.assertEqual(3, dataset.n_channels)
        self.assertEqual(2, dataset.n_classes)
        self.assertEqual(4, len(dataset.train_dataset))
        self.assertEqual([0, 0, 1, 1], dataset.train_dataset.labels.tolist())

    def test_image_folder_validation_does_not_require_numpy_array_materialization(self) -> None:
        """Ensure ZIP validation inspects image folders without bulk NumPy materialization."""
        zip_path = _build_image_folder_zip(self.tmpdir.name, "vision_dataset.zip")

        with mock.patch.object(self.datasets_service_mod.np, "asarray", side_effect=AssertionError("np.asarray should not be used")):
            meta = self.service.save_custom_dataset_file(
                username="alice",
                original_name="vision_dataset.zip",
                source_path=zip_path,
            )

        self.assertEqual("image_folder", meta["dataset_type"])
