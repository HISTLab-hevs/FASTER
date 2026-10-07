"""Regression tests for the default :class:`utils.model.Net` input-size handling."""

from __future__ import annotations

import unittest

import torch

from utils.model import Net


class NetInputSizeTests(unittest.TestCase):
    """Regression guard: the default Net must accept any image input size."""

    def test_net_accepts_28x28_single_channel(self) -> None:
        """MedMNIST/FashionMNIST 28x28 single-channel inputs stay supported."""
        model = Net(1, 10)
        out = model(torch.zeros(2, 1, 28, 28))
        self.assertEqual(tuple(out.shape), (2, 10))

    def test_net_accepts_32x32_three_channel(self) -> None:
        """CIFAR-100 32x32 RGB inputs no longer crash at the classifier."""
        model = Net(3, 100)
        out = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(out.shape), (2, 100))
