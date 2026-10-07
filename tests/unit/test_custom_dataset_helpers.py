"""Tests for custom-dataset format and name sanitizers."""

from __future__ import annotations

import unittest

from tests.helpers import reload_module


class CustomDatasetHelperTests(unittest.TestCase):
    """Exercise dataset-format and dataset-name sanitizers on boundary inputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.datasets_mod = reload_module("web_backend.services.datasets")
        cls.service_cls = cls.datasets_mod.CustomDatasetsService

    def test_normalize_dataset_format_prefers_explicit_value_over_inferred_suffix(self) -> None:
        """Ensure an explicit visible format wins over conflicting dataset hints."""
        fmt = self.service_cls.normalize_dataset_format(
            "  TSV  ",
            dataset_type="custom_image_folder",
            dataset_ref="dataset.zip",
            storage_uri="/tmp/dataset.csv",
        )

        self.assertEqual("tsv", fmt)

    def test_normalize_dataset_format_falls_back_to_reference_and_dataset_type(self) -> None:
        """Ensure dataset format inference uses file suffixes before dataset type defaults."""
        self.assertEqual(
            "npz",
            self.service_cls.normalize_dataset_format(
                None,
                dataset_type="custom_csv",
                dataset_ref="vision_dataset.npz",
                storage_uri="/tmp/vision_dataset.csv",
            ),
        )
        self.assertEqual(
            "zip",
            self.service_cls.normalize_dataset_format(
                None,
                dataset_type="custom_image_folder",
                dataset_ref="vision_dataset.bin",
                storage_uri="/tmp/vision_dataset.bin",
            ),
        )

    def test_sanitize_custom_dataset_label_rejects_blank_long_and_unicode_names(self) -> None:
        """Ensure dataset display labels stay short, non-empty, and ASCII-safe."""
        self.assertEqual(
            "Demo Dataset",
            self.service_cls.sanitize_custom_dataset_label("  Demo Dataset  "),
        )

        with self.assertRaisesRegex(ValueError, "dataset name is required"):
            self.service_cls.sanitize_custom_dataset_label("   ")

        with self.assertRaisesRegex(ValueError, "at most 20 characters"):
            self.service_cls.sanitize_custom_dataset_label("x" * 21)

        with self.assertRaisesRegex(ValueError, "invalid characters"):
            self.service_cls.sanitize_custom_dataset_label("Dataset 🚀")

    def test_sanitize_custom_dataset_description_collapses_control_characters(self) -> None:
        """Ensure descriptions preserve content while removing control noise."""
        desc = "  Line 1\tLine 2\n\x07Line 3   "

        self.assertEqual(
            "Line 1 Line 2 Line 3",
            self.service_cls.sanitize_custom_dataset_description(desc),
        )

    def test_sanitize_custom_dataset_description_rejects_overlong_payloads(self) -> None:
        """Ensure oversized descriptions fail with a stable error message."""
        with self.assertRaisesRegex(ValueError, "at most 200 characters"):
            self.service_cls.sanitize_custom_dataset_description("a" * 201)

