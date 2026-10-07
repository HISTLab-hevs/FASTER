"""Tests for reproducible seeding of Python, NumPy, Torch, and GP randomness."""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch


class TestSetGlobalSeed(unittest.TestCase):
    """Test the set_global_seed helper function."""

    def setUp(self) -> None:
        """Import main module for each test."""
        import main as main_module
        self.main_module = main_module

    def test_set_global_seed_seeds_python_random(self) -> None:
        """Verify set_global_seed seeds Python's random module."""
        self.main_module.set_global_seed(42)
        val1 = random.random()

        self.main_module.set_global_seed(42)
        val2 = random.random()

        self.assertEqual(val1, val2)

    def test_set_global_seed_seeds_numpy(self) -> None:
        """Verify set_global_seed seeds NumPy's global RNG."""
        self.main_module.set_global_seed(42)
        val1 = np.random.random()

        self.main_module.set_global_seed(42)
        val2 = np.random.random()

        self.assertEqual(val1, val2)

    def test_set_global_seed_seeds_torch_cpu(self) -> None:
        """Verify set_global_seed seeds Torch CPU RNG."""
        self.main_module.set_global_seed(42)
        val1 = torch.randn(1).item()

        self.main_module.set_global_seed(42)
        val2 = torch.randn(1).item()

        self.assertEqual(val1, val2)

    def test_set_global_seed_with_none_does_not_error(self) -> None:
        """Verify set_global_seed(None) returns without error."""
        # Should not raise
        self.main_module.set_global_seed(None)

    def test_set_global_seed_converts_int_like_seed(self) -> None:
        """Verify set_global_seed accepts and converts int-like seeds."""
        # Should accept strings and other int-like types
        self.main_module.set_global_seed("42")
        val1 = random.random()

        self.main_module.set_global_seed(42)
        val2 = random.random()

        self.assertEqual(val1, val2)

    def test_set_global_seed_different_seeds_produce_different_values(self) -> None:
        """Verify different seeds produce different RNG sequences."""
        self.main_module.set_global_seed(42)
        val1 = random.random()

        self.main_module.set_global_seed(43)
        val2 = random.random()

        self.assertNotEqual(val1, val2)

    def test_set_global_seed_multiple_random_calls_consistent(self) -> None:
        """Verify that multiple random calls are consistent with same seed."""
        self.main_module.set_global_seed(42)
        vals1 = [random.random() for _ in range(5)]

        self.main_module.set_global_seed(42)
        vals2 = [random.random() for _ in range(5)]

        self.assertEqual(vals1, vals2)

    def test_set_global_seed_torch_and_numpy_independence(self) -> None:
        """Verify that torch and numpy use independent seeded streams."""
        self.main_module.set_global_seed(42)
        torch_val1 = torch.randn(1).item()
        np_val1 = np.random.random()

        self.main_module.set_global_seed(42)
        torch_val2 = torch.randn(1).item()
        np_val2 = np.random.random()

        self.assertEqual(torch_val1, torch_val2)
        self.assertEqual(np_val1, np_val2)

