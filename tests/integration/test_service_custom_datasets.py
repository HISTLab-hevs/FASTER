"""Integration tests for custom dataset upload, validation, and storage."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from tests.helpers import reload_module


def _build_valid_csv() -> str:
    rows = ["feature_a,feature_b,label,split"]
    for idx in range(20):
        split = "train" if idx < 14 else "val" if idx < 17 else "test"
        label = idx % 2
        rows.append(f"{idx},{idx + 0.5},{label},{split}")
    return "\n".join(rows) + "\n"


def _build_valid_tsv() -> str:
    rows = ["feature_a\tfeature_b\tlabel\tsplit"]
    for idx in range(20):
        split = "train" if idx < 14 else "val" if idx < 17 else "test"
        label = idx % 2
        rows.append(f"{idx}\t{idx + 0.5}\t{label}\t{split}")
    return "\n".join(rows) + "\n"


def _write_upload_file(tmpdir: str, name: str, content: str | bytes, *, binary: bool = False) -> str:
    path = os.path.join(tmpdir, name)
    if binary:
        with open(path, "wb") as handle:
            handle.write(content)
    else:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
    return path


def _write_image(path: str, value: int) -> None:
    arr = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _image_bytes(value: int, *, size: tuple[int, int] = (8, 8)) -> bytes:
    """Return one in-memory PNG image for ZIP archive fixtures."""
    arr = np.full((size[0], size[1], 3), value, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _build_image_folder_zip(
    tmpdir: str,
    name: str,
    *,
    wrap_root: bool = True,
    mismatched_classes: bool = False,
    extra_root_file: bool = False,
) -> str:
    root_dir = os.path.join(tmpdir, "image_folder_source")
    archive_root = os.path.join(root_dir, "dataset_root") if wrap_root else root_dir
    classes = ("class_b", "class_a")
    os.makedirs(archive_root, exist_ok=True)
    if extra_root_file:
        with open(os.path.join(root_dir, "README.txt"), "w", encoding="utf-8") as handle:
            handle.write("dataset notes")
    for split in ("train", "val", "test"):
        split_classes = classes
        if mismatched_classes and split == "val":
            split_classes = ("class_a", "class_c")
        for class_name in split_classes:
            class_dir = os.path.join(archive_root, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for idx in range(2):
                _write_image(
                    os.path.join(class_dir, f"{class_name}_{idx}.png"),
                    value=(idx + 1) * (25 if class_name.endswith("a") else 40),
                )

    zip_path = os.path.join(tmpdir, name)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for base, _, files in os.walk(root_dir):
            for file_name in sorted(files):
                abs_path = os.path.join(base, file_name)
                archive.write(abs_path, os.path.relpath(abs_path, root_dir))
    return zip_path


def _build_custom_image_folder_zip(
    tmpdir: str,
    name: str,
    members: list[tuple[str, int | bytes]],
) -> str:
    """Build one ZIP archive from explicit archive member names."""
    zip_path = os.path.join(tmpdir, name)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for member_name, payload in members:
            data = payload if isinstance(payload, bytes) else _image_bytes(payload)
            archive.writestr(member_name, data)
    return zip_path


class ServiceCustomDatasetTests(unittest.TestCase):
    """Cover custom dataset ingestion, validation, and listing behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.service_mod = reload_module("web_backend.service")

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

    def test_save_tabular_dataset_lists_and_resolves_for_owner(self) -> None:
        """Ensure saved CSV datasets resolve, list, and validate for the owner."""
        source_path = _write_upload_file(self.tmpdir.name, "customer_churn.csv", _build_valid_csv())

        meta = self.service.save_custom_dataset_file(
            username="alice",
            original_name="customer_churn.csv",
            source_path=source_path,
            description="Main dataset",
        )

        self.assertEqual("customer_churn", meta["dataset_name"])
        self.assertEqual("tabular_csv", meta["dataset_type"])
        self.assertEqual("csv", meta["dataset_format"])

        resolved = self.service.resolve_custom_dataset_path(
            "alice",
            meta["dataset_ref"],
        )
        self.assertTrue(resolved)
        self.assertTrue(os.path.isfile(resolved))

        listed = self.service.list_custom_datasets("alice")
        matching = [item for item in listed if item["dataset_ref"] == meta["dataset_ref"]]
        self.assertEqual(1, len(matching))
        self.assertEqual("Main dataset", matching[0]["description"])
        self.assertEqual("csv", matching[0]["dataset_format"])

        valid, msg = self.service.validate_custom_dataset_structure(
            resolved,
            "custom_csv",
        )
        self.assertTrue(valid)
        self.assertEqual("", msg)

    def test_save_tsv_dataset_preserves_visible_format_while_normalizing_storage(self) -> None:
        """Ensure TSV uploads keep their visible format while normalizing stored content."""
        source_path = _write_upload_file(self.tmpdir.name, "customer_churn.tsv", _build_valid_tsv())

        meta = self.service.save_custom_dataset_file(
            username="alice",
            original_name="customer_churn.tsv",
            source_path=source_path,
            description="TSV dataset",
        )

        self.assertEqual("customer_churn", meta["dataset_name"])
        self.assertEqual("tabular_csv", meta["dataset_type"])
        self.assertEqual("tsv", meta["dataset_format"])

        resolved = self.service.resolve_custom_dataset_path("alice", meta["dataset_ref"])
        self.assertTrue(resolved)
        self.assertTrue(os.path.isfile(resolved))
        self.assertTrue(str(resolved).endswith(".csv"))

        listed = self.service.list_custom_datasets("alice")
        matching = [item for item in listed if item["dataset_ref"] == meta["dataset_ref"]]
        self.assertEqual(1, len(matching))
        self.assertEqual("tsv", matching[0]["dataset_format"])
        self.assertEqual("TSV dataset", matching[0]["description"])

        valid, msg = self.service.validate_custom_dataset_structure(resolved, "custom_csv")
        self.assertTrue(valid)
        self.assertEqual("", msg)

    def test_save_tabular_dataset_from_uploaded_file_path(self) -> None:
        """Ensure tabular uploads can be saved directly from the staged source path."""
        source_path = _write_upload_file(self.tmpdir.name, "uploaded.csv", _build_valid_csv())

        meta = self.service.save_custom_dataset_file(
            username="alice",
            original_name="uploaded.csv",
            source_path=source_path,
            description="Uploaded path dataset",
        )

        self.assertEqual("uploaded", meta["dataset_name"])
        self.assertEqual("tabular_csv", meta["dataset_type"])
        self.assertEqual("csv", meta["dataset_format"])

        resolved = self.service.resolve_custom_dataset_path("alice", meta["dataset_ref"])
        self.assertTrue(resolved)
        self.assertTrue(os.path.isfile(resolved))

        valid, msg = self.service.validate_custom_dataset_structure(resolved, "custom_csv")
        self.assertTrue(valid)
        self.assertEqual("", msg)

    def test_save_custom_dataset_rejects_invalid_tabular_structure(self) -> None:
        """Ensure invalid tabular uploads are rejected before persistence."""
        bad_csv = "feature_a,label\n1,0\n2,1\n"

        with self.assertRaisesRegex(ValueError, "split column"):
            bad_path = _write_upload_file(self.tmpdir.name, "invalid.csv", bad_csv)
            self.service.save_custom_dataset_file(
                username="alice",
                original_name="invalid.csv",
                source_path=bad_path,
            )

    def test_save_npz_dataset_and_export_run_zip(self) -> None:
        """Ensure NPZ image datasets save cleanly and exported runs include artifacts."""
        train_images = np.zeros((20, 8, 8), dtype=np.float32)
        train_labels = np.array([idx % 2 for idx in range(20)], dtype=np.int64)
        val_images = np.zeros((4, 8, 8), dtype=np.float32)
        val_labels = np.array([0, 1, 0, 1], dtype=np.int64)
        test_images = np.zeros((4, 8, 8), dtype=np.float32)
        test_labels = np.array([0, 1, 0, 1], dtype=np.int64)

        payload = io.BytesIO()
        np.savez(
            payload,
            train_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels,
            test_images=test_images,
            test_labels=test_labels,
        )
        npz_path = _write_upload_file(
            self.tmpdir.name,
            "vision_dataset.npz",
            payload.getvalue(),
            binary=True,
        )

        meta = self.service.save_custom_dataset_file(
            username="alice",
            original_name="vision_dataset.npz",
            source_path=npz_path,
            description="Image set",
        )

        self.assertEqual("image_npz", meta["dataset_type"])
        self.assertEqual("npz", meta["dataset_format"])
        self.assertFalse(os.path.exists(npz_path))

        resolved = self.service.resolve_custom_dataset_path("alice", meta["dataset_ref"])
        self.assertTrue(resolved)
        self.assertTrue(os.path.isfile(resolved))

        listed = self.service.list_custom_datasets("alice")
        matching = [item for item in listed if item["dataset_ref"] == meta["dataset_ref"]]
        self.assertEqual(1, len(matching))
        self.assertEqual("npz", matching[0]["dataset_format"])

        runs_root = os.path.join(self.tmpdir.name, self.service_mod.BASE_RESULTS_PATH)
        run_dir = os.path.join(runs_root, "run_alpha")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            handle.write('{"accuracy": 0.9}')

        zip_path = self.service.export_run_to_zip("run_alpha")
        self.assertTrue(os.path.isfile(zip_path))

        with zipfile.ZipFile(zip_path) as archive:
            self.assertIn("run_alpha/metrics.json", archive.namelist())

    def test_save_directory_image_dataset_zip_persists_manifest_and_validates(self) -> None:
        """Ensure image-folder ZIP uploads persist sidecar metadata and validate cleanly."""
        zip_path = _build_image_folder_zip(self.tmpdir.name, "vision_dataset.zip")

        meta = self.service.save_custom_dataset_file(
            username="alice",
            original_name="vision_dataset.zip",
            source_path=zip_path,
            description="Directory image dataset",
        )

        self.assertEqual("vision_dataset", meta["dataset_name"])
        self.assertEqual("image_folder", meta["dataset_type"])
        self.assertEqual("zip", meta["dataset_format"])
        self.assertEqual(12, meta["rows"])
        self.assertEqual(2, meta["classes"])

        resolved = self.service.resolve_custom_dataset_path("alice", meta["dataset_ref"])
        self.assertTrue(resolved)
        self.assertTrue(os.path.isdir(resolved))

        valid, msg = self.service.validate_custom_dataset_structure(
            resolved,
            "custom_image_folder",
        )
        self.assertTrue(valid)
        self.assertEqual("", msg)

        listed = self.service.list_custom_datasets("alice")
        matching = [item for item in listed if item["dataset_ref"] == meta["dataset_ref"]]
        self.assertEqual(1, len(matching))
        self.assertEqual("zip", matching[0]["dataset_format"])
        self.assertEqual("Directory image dataset", matching[0]["description"])

        sidecar = self.service._load_custom_dataset_sidecar(resolved)
        self.assertEqual("custom_image_folder", sidecar["dataset_type"])
        self.assertEqual(["class_a", "class_b"], sidecar["class_names"])
        self.assertEqual({"class_a": 0, "class_b": 1}, sidecar["class_to_idx"])
        self.assertEqual(
            [
                {"path": "train/class_a/class_a_0.png", "label": 0},
                {"path": "train/class_a/class_a_1.png", "label": 0},
                {"path": "train/class_b/class_b_0.png", "label": 1},
                {"path": "train/class_b/class_b_1.png", "label": 1},
            ],
            sidecar["split_index"]["train"],
        )

    def test_save_directory_image_dataset_zip_rejects_class_mismatch(self) -> None:
        """Ensure image-folder ZIP uploads reject mismatched split class directories."""
        zip_path = _build_image_folder_zip(
            self.tmpdir.name,
            "invalid_vision_dataset.zip",
            mismatched_classes=True,
        )

        with self.assertRaisesRegex(ValueError, "Class subdirectories must match"):
            self.service.save_custom_dataset_file(
                username="alice",
                original_name="invalid_vision_dataset.zip",
                source_path=zip_path,
            )

    def test_save_directory_image_dataset_zip_accepts_wrapper_with_extra_root_file(self) -> None:
        """Ensure wrapped image-folder ZIPs tolerate unrelated extra root files."""
        zip_path = _build_image_folder_zip(
            self.tmpdir.name,
            "vision_dataset_with_notes.zip",
            extra_root_file=True,
        )

        meta = self.service.save_custom_dataset_file(
            username="alice",
            original_name="vision_dataset_with_notes.zip",
            source_path=zip_path,
        )

        resolved = self.service.resolve_custom_dataset_path("alice", meta["dataset_ref"])
        self.assertTrue(resolved)
        self.assertTrue(os.path.isdir(resolved))

    def test_save_directory_image_dataset_zip_rejects_duplicate_normalized_member_paths(self) -> None:
        """Ensure ZIP archives cannot silently overwrite duplicated normalized output paths."""
        zip_path = _build_custom_image_folder_zip(
            self.tmpdir.name,
            "duplicate_vision_dataset.zip",
            [
                ("dataset_root/train/class_a/dup.png", 40),
                ("dataset_root/train/class_a/dup.png", 60),
                ("dataset_root/val/class_a/val.png", 40),
                ("dataset_root/val/class_b/val.png", 60),
                ("dataset_root/test/class_a/test.png", 40),
                ("dataset_root/test/class_b/test.png", 60),
            ],
        )

        with self.assertRaisesRegex(ValueError, "duplicate normalized paths"):
            self.service.save_custom_dataset_file(
                username="alice",
                original_name="duplicate_vision_dataset.zip",
                source_path=zip_path,
            )

    def test_save_directory_image_dataset_zip_rejects_case_collisions_after_normalization(self) -> None:
        """Ensure case-only filename differences are treated as duplicate logical paths."""
        zip_path = _build_custom_image_folder_zip(
            self.tmpdir.name,
            "case_collision_vision_dataset.zip",
            [
                ("dataset_root/train/class_a/Resume.png", 40),
                ("dataset_root/train/class_a/resume.png", 60),
                ("dataset_root/val/class_a/val.png", 40),
                ("dataset_root/val/class_b/val.png", 60),
                ("dataset_root/test/class_a/test.png", 40),
                ("dataset_root/test/class_b/test.png", 60),
            ],
        )

        with self.assertRaisesRegex(ValueError, "duplicate normalized paths"):
            self.service.save_custom_dataset_file(
                username="alice",
                original_name="case_collision_vision_dataset.zip",
                source_path=zip_path,
            )

    def test_save_directory_image_dataset_zip_rejects_unicode_normalization_collisions(self) -> None:
        """Ensure NFC/NFD-equivalent filenames cannot coexist in one dataset archive."""
        zip_path = _build_custom_image_folder_zip(
            self.tmpdir.name,
            "unicode_collision_vision_dataset.zip",
            [
                ("dataset_root/train/class_a/café.png", 40),
                ("dataset_root/train/class_a/cafe\u0301.png", 60),
                ("dataset_root/val/class_a/val.png", 40),
                ("dataset_root/val/class_b/val.png", 60),
                ("dataset_root/test/class_a/test.png", 40),
                ("dataset_root/test/class_b/test.png", 60),
            ],
        )

        with self.assertRaisesRegex(ValueError, "duplicate normalized paths"):
            self.service.save_custom_dataset_file(
                username="alice",
                original_name="unicode_collision_vision_dataset.zip",
                source_path=zip_path,
            )

    def test_save_directory_image_dataset_zip_rejects_trailing_space_and_dot_aliases(self) -> None:
        """Ensure Windows-style trailing space/dot aliases are rejected as unsafe paths."""
        aliases = [
            "dataset_root/train/class_a /alias.png",
            "dataset_root/train/class_a./alias.png",
        ]

        for idx, member_name in enumerate(aliases):
            with self.subTest(member_name=member_name):
                zip_path = _build_custom_image_folder_zip(
                    self.tmpdir.name,
                    f"alias_{idx}.zip",
                    [
                        (member_name, 50),
                        ("dataset_root/val/class_a/val.png", 40),
                        ("dataset_root/val/class_b/val.png", 60),
                        ("dataset_root/test/class_a/test.png", 40),
                        ("dataset_root/test/class_b/test.png", 60),
                    ],
                )

                with self.assertRaisesRegex(ValueError, "unsafe paths"):
                    self.service.save_custom_dataset_file(
                        username="alice",
                        original_name=f"alias_{idx}.zip",
                        source_path=zip_path,
                    )

    def test_save_directory_image_dataset_zip_rejects_windows_reserved_device_names(self) -> None:
        """Ensure Windows reserved device names are rejected in any ZIP path segment."""
        reserved_members = [
            "dataset_root/train/CON/image.png",
            "dataset_root/train/con.txt/image.png",
            "dataset_root/train/class_a/PRN.png",
            "dataset_root/train/class_a/com1.txt",
            "dataset_root/train/LPT9/logo.jpg",
        ]

        for idx, member_name in enumerate(reserved_members):
            with self.subTest(member_name=member_name):
                zip_path = _build_custom_image_folder_zip(
                    self.tmpdir.name,
                    f"reserved_{idx}.zip",
                    [
                        (member_name, 50),
                        ("dataset_root/val/class_a/val.png", 40),
                        ("dataset_root/val/class_b/val.png", 60),
                        ("dataset_root/test/class_a/test.png", 40),
                        ("dataset_root/test/class_b/test.png", 60),
                    ],
                )

                with self.assertRaisesRegex(ValueError, "unsafe paths"):
                    self.service.save_custom_dataset_file(
                        username="alice",
                        original_name=f"reserved_{idx}.zip",
                        source_path=zip_path,
                    )

    def test_save_directory_image_dataset_zip_rejects_overlong_path_segments(self) -> None:
        """Ensure overly long ZIP path segments fail closed before extraction."""
        long_name = "a" * 256
        zip_path = _build_custom_image_folder_zip(
            self.tmpdir.name,
            "overlong_vision_dataset.zip",
            [
                (f"dataset_root/train/{long_name}/image.png", 50),
                ("dataset_root/val/class_a/val.png", 40),
                ("dataset_root/val/class_b/val.png", 60),
                ("dataset_root/test/class_a/test.png", 40),
                ("dataset_root/test/class_b/test.png", 60),
            ],
        )

        with self.assertRaisesRegex(ValueError, "unsafe paths"):
            self.service.save_custom_dataset_file(
                username="alice",
                original_name="overlong_vision_dataset.zip",
                source_path=zip_path,
            )

    def test_save_directory_image_dataset_zip_rejects_deep_nested_image_paths(self) -> None:
        """Ensure extra nested directories do not bypass the expected class/image layout."""
        zip_path = _build_custom_image_folder_zip(
            self.tmpdir.name,
            "deep_vision_dataset.zip",
            [
                ("dataset_root/train/class_a/nested/image.png", 50),
                ("dataset_root/train/class_b/nested/image2.png", 60),
                ("dataset_root/val/class_a/val.png", 40),
                ("dataset_root/val/class_b/val.png", 60),
                ("dataset_root/test/class_a/test.png", 40),
                ("dataset_root/test/class_b/test.png", 60),
            ],
        )

        with self.assertRaisesRegex(ValueError, "supported image file"):
            self.service.save_custom_dataset_file(
                username="alice",
                original_name="deep_vision_dataset.zip",
                source_path=zip_path,
            )

    def test_save_directory_image_dataset_zip_rejects_unsafe_paths(self) -> None:
        """Ensure unsafe ZIP member paths are rejected before extraction writes anything."""
        unsafe_members = [
            "dataset_root/train/class_a/../evil.png",
            "/dataset_root/train/class_a/evil.png",
            "dataset_root\\train\\class_a\\..\\evil.png",
        ]

        for idx, member_name in enumerate(unsafe_members):
            with self.subTest(member_name=member_name):
                zip_path = _build_custom_image_folder_zip(
                    self.tmpdir.name,
                    f"unsafe_{idx}.zip",
                    [(member_name, 50)],
                )

                with self.assertRaisesRegex(ValueError, "unsafe paths"):
                    self.service.save_custom_dataset_file(
                        username="alice",
                        original_name=f"unsafe_{idx}.zip",
                        source_path=zip_path,
                    )

    def test_save_directory_image_dataset_zip_rejects_empty_archive(self) -> None:
        """Ensure empty ZIP archives fail closed before any extraction occurs."""
        zip_path = os.path.join(self.tmpdir.name, "empty_dataset.zip")
        with zipfile.ZipFile(zip_path, "w"):
            pass

        with self.assertRaisesRegex(ValueError, "ZIP dataset archive is empty"):
            self.service.save_custom_dataset_file(
                username="alice",
                original_name="empty_dataset.zip",
                source_path=zip_path,
            )

    def test_save_directory_image_dataset_zip_accepts_unicode_names_and_ignores_hidden_files(self) -> None:
        """Ensure safe Unicode archive names work and hidden junk files are ignored."""
        zip_path = _build_custom_image_folder_zip(
            self.tmpdir.name,
            "unicode_vision_dataset.zip",
            [
                ("dataset_root/train/café/.DS_Store", b"junk"),
                ("dataset_root/train/café/café_0.png", 40),
                ("dataset_root/train/naïve/naïve_0.png", 60),
                ("dataset_root/val/café/café_1.png", 40),
                ("dataset_root/val/naïve/naïve_1.png", 60),
                ("dataset_root/test/café/café_2.png", 40),
                ("dataset_root/test/naïve/naïve_2.png", 60),
            ],
        )

        meta = self.service.save_custom_dataset_file(
            username="alice",
            original_name="unicode_vision_dataset.zip",
            source_path=zip_path,
            description="Unicode dataset",
        )

        self.assertEqual("image_folder", meta["dataset_type"])
        self.assertEqual("zip", meta["dataset_format"])
        self.assertEqual(6, meta["rows"])
        self.assertEqual(2, meta["classes"])

        resolved = self.service.resolve_custom_dataset_path("alice", meta["dataset_ref"])
        self.assertTrue(resolved)
        sidecar = self.service._load_custom_dataset_sidecar(resolved)
        self.assertEqual(["café", "naïve"], sidecar["class_names"])
        self.assertIn("train/café/café_0.png", [item["path"] for item in sidecar["split_index"]["train"]])

    def test_save_directory_image_dataset_zip_cleans_up_after_late_invalid_member(self) -> None:
        """Ensure partial extraction failures do not leave behind a partial dataset."""
        members: list[tuple[str, int | bytes]] = []
        for split in ("train", "val", "test"):
            for class_name, value in (("class_a", 40), ("class_b", 60)):
                for idx in range(5):
                    members.append((f"dataset_root/{split}/{class_name}/{split}_{class_name}_{idx}.png", value + idx))
        members.append(("dataset_root/test/class_b/corrupt.png", b"not-a-real-image"))

        zip_path = _build_custom_image_folder_zip(
            self.tmpdir.name,
            "late_failure_vision_dataset.zip",
            members,
        )
        user_root = self.service._datasets_service.user_custom_datasets_dir("alice")
        before_entries = sorted(os.listdir(user_root)) if os.path.isdir(user_root) else []

        with self.assertRaisesRegex(ValueError, "Failed to read image"):
            self.service.save_custom_dataset_file(
                username="alice",
                original_name="late_failure_vision_dataset.zip",
                source_path=zip_path,
            )

        after_entries = sorted(os.listdir(user_root)) if os.path.isdir(user_root) else []
        self.assertEqual(before_entries, after_entries)

    def test_list_custom_datasets_rehydrates_zip_metadata_from_sidecar_for_db_rows(self) -> None:
        """Ensure DB rows for ZIP datasets are rehydrated from persisted sidecar metadata."""
        zip_path = _build_image_folder_zip(self.tmpdir.name, "vision_dataset.zip")
        meta = self.service.save_custom_dataset_file(
            username="alice",
            original_name="vision_dataset.zip",
            source_path=zip_path,
            description="Directory image dataset",
        )
        dataset_root = self.service.resolve_custom_dataset_path("alice", meta["dataset_ref"])
        self.assertTrue(dataset_root)

        class _Repo:
            def list_datasets(self, owner_identifier):
                return [
                    SimpleNamespace(
                        dataset_ref=meta["dataset_ref"],
                        dataset_name="stale_name",
                        dataset_type="custom_image_folder",
                        storage_uri=dataset_root,
                        owner_identifier="alice",
                        rows_count=None,
                        features_count=None,
                        classes_count=None,
                        created_at=SimpleNamespace(isoformat=lambda timespec="seconds": "2026-01-01T00:00:00"),
                    )
                ]

        self.service._db_enabled = True
        self.service._custom_datasets_repo = _Repo()

        listed = self.service.list_custom_datasets("alice")

        self.assertEqual(1, len(listed))
        self.assertEqual(meta["dataset_ref"], listed[0]["dataset_ref"])
        self.assertEqual("vision_dataset", listed[0]["dataset_name"])
        self.assertEqual("custom_image_folder", listed[0]["dataset_type"])
        self.assertEqual("zip", listed[0]["dataset_format"])
        self.assertEqual(12, listed[0]["rows"])
        self.assertEqual(192, listed[0]["features"])
        self.assertEqual(2, listed[0]["classes"])
        self.assertEqual("Directory image dataset", listed[0]["description"])

    def test_list_custom_datasets_keeps_dict_rows_and_skips_malformed_rows(self) -> None:
        """Ensure DB listing tolerates dict-shaped rows and skips malformed entries."""
        class _Repo:
            def list_datasets(self, owner_identifier):
                return [
                    {
                        "dataset_ref": "dataset_20260101_000000_deadbeef_good.csv",
                        "dataset_name": "dict_dataset",
                        "dataset_type": "custom_csv",
                        "storage_uri": "/tmp/dict_dataset.csv",
                        "owner_identifier": "alice",
                        "rows_count": 9,
                        "features_count": 2,
                        "classes_count": 2,
                        "created_at": SimpleNamespace(
                            isoformat=lambda timespec="seconds": "2026-01-01T00:00:00"
                        ),
                    },
                    None,
                ]

        self.service._db_enabled = True
        self.service._custom_datasets_repo = _Repo()

        with mock.patch.object(
            self.service._datasets_service,
            "load_custom_dataset_sidecar",
            return_value={},
        ):
            listed = self.service.list_custom_datasets("alice")

        self.assertEqual(1, len(listed))
        self.assertEqual("dataset_20260101_000000_deadbeef_good.csv", listed[0]["dataset_ref"])
        self.assertEqual("dict_dataset", listed[0]["dataset_name"])
        self.assertEqual("custom_csv", listed[0]["dataset_type"])
        self.assertEqual("csv", listed[0]["dataset_format"])
        self.assertEqual("alice", listed[0]["owner"])
        self.assertEqual("2026-01-01T00:00:00", listed[0]["created_at"])

    def test_list_custom_datasets_keeps_filesystem_zip_entries_when_db_has_other_rows(self) -> None:
        """Ensure filesystem ZIP datasets remain visible alongside unrelated DB rows."""
        zip_path = _build_image_folder_zip(self.tmpdir.name, "vision_dataset.zip")
        meta = self.service.save_custom_dataset_file(
            username="alice",
            original_name="vision_dataset.zip",
            source_path=zip_path,
            description="Directory image dataset",
        )
        dataset_root = self.service.resolve_custom_dataset_path("alice", meta["dataset_ref"])
        self.assertTrue(dataset_root)

        class _Repo:
            def list_datasets(self, owner_identifier):
                return [
                    SimpleNamespace(
                        dataset_ref="dataset_20260101_000000_deadbeef_existing.csv",
                        dataset_name="existing_dataset",
                        dataset_type="custom_csv",
                        storage_uri="/tmp/nonexistent.csv",
                        owner_identifier="alice",
                        rows_count=42,
                        features_count=3,
                        classes_count=2,
                        created_at=SimpleNamespace(isoformat=lambda timespec="seconds": "2026-01-01T00:00:00"),
                    )
                ]

        self.service._db_enabled = True
        self.service._custom_datasets_repo = _Repo()

        listed = self.service.list_custom_datasets("alice")
        refs = {item["dataset_ref"] for item in listed}

        self.assertIn("dataset_20260101_000000_deadbeef_existing.csv", refs)
        self.assertIn(meta["dataset_ref"], refs)

        zip_item = next(item for item in listed if item["dataset_ref"] == meta["dataset_ref"])
        self.assertEqual("vision_dataset", zip_item["dataset_name"])
        self.assertEqual("custom_image_folder", zip_item["dataset_type"])
        self.assertEqual("zip", zip_item["dataset_format"])
        self.assertEqual(12, zip_item["rows"])
        self.assertEqual(192, zip_item["features"])
        self.assertEqual(2, zip_item["classes"])
        self.assertEqual("Directory image dataset", zip_item["description"])

    def test_resolve_custom_dataset_path_returns_none_for_stale_db_reference(self) -> None:
        """Ensure stale DB metadata does not resolve to a deleted filesystem path."""
        missing_path = os.path.join(self.tmpdir.name, "missing.csv")

        class _Repo:
            def get_dataset(self, owner_identifier, dataset_ref):
                return SimpleNamespace(
                    dataset_ref=dataset_ref,
                    dataset_name="stale_dataset",
                    dataset_type="custom_csv",
                    storage_uri=missing_path,
                    owner_identifier=owner_identifier,
                    rows_count=None,
                    features_count=None,
                    classes_count=None,
                    created_at=SimpleNamespace(isoformat=lambda timespec="seconds": "2026-01-01T00:00:00"),
                )

        self.service._db_enabled = True
        self.service._custom_datasets_repo = _Repo()

        resolved = self.service.resolve_custom_dataset_path("alice", "dataset_01.csv")

        self.assertIsNone(resolved)

    def test_resolve_custom_dataset_path_masks_repository_exception(self) -> None:
        """Ensure repository lookup errors are logged and return a missing-path result."""
        class _Repo:
            def get_dataset(self, owner_identifier, dataset_ref):
                raise RuntimeError("lookup failure")

        self.service._db_enabled = True
        self.service._custom_datasets_repo = _Repo()

        with self.assertLogs("web_backend.service", level="WARNING") as captured:
            resolved = self.service.resolve_custom_dataset_path("alice", "dataset_01.csv")

        self.assertIsNone(resolved)
        self.assertTrue(any("lookup failure" in line for line in captured.output))

    def test_list_custom_datasets_falls_back_to_filesystem_when_db_listing_fails(self) -> None:
        """Ensure filesystem datasets remain visible even if DB listing errors out."""
        source_path = _write_upload_file(self.tmpdir.name, "customer_churn.csv", _build_valid_csv())
        meta = self.service.save_custom_dataset_file(
            username="alice",
            original_name="customer_churn.csv",
            source_path=source_path,
            description="Filesystem dataset",
        )

        class _Repo:
            def list_datasets(self, owner_identifier):
                raise RuntimeError("database offline")

        self.service._db_enabled = True
        self.service._custom_datasets_repo = _Repo()

        listed = self.service.list_custom_datasets("alice")
        refs = {item["dataset_ref"] for item in listed}

        self.assertIn(meta["dataset_ref"], refs)
        self.assertTrue(any(item["description"] == "Filesystem dataset" for item in listed))
