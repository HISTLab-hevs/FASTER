"""Test save_path validation in the main CLI entrypoint."""

from __future__ import annotations

import sys
import unittest
from unittest import mock

from tests.helpers import reload_module


class MainSavePathValidationTests(unittest.TestCase):
    """Verify that save_path is properly validated before use."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.main_mod = reload_module("main")

    def test_missing_save_path_raises_clear_error(self) -> None:
        """Ensure missing output paths fail fast with a clear message."""
        config = {
            "method": "fed_avg",
            "dataset_name": "pathmnist",
            "num_clients": 1,
            "batch_size": 16,
            "server_warmup_epochs": 1,
            "local_model_epochs": 1,
            "weights_sending_frequency": 1,
            "server_learning_rate": 0.001,
            "learning_rate": 0.001,
        }

        with mock.patch.object(sys, "argv", ["main.py"]), \
            mock.patch.object(self.main_mod, "load_config", return_value=config), \
            mock.patch.object(self.main_mod, "run_experiment") as run_experiment:
            with self.assertRaises(ValueError) as cm:
                self.main_mod.main()

        run_experiment.assert_not_called()
        self.assertIn("save_path is required", str(cm.exception))
        self.assertIn("--save_path", str(cm.exception))
        self.assertIn("configuration file", str(cm.exception))
