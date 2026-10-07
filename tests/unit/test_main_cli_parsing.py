"""Tests for the ``main.py`` training CLI argument parsing and coercions."""

from __future__ import annotations

import sys
import unittest
from unittest import mock

from tests.helpers import reload_module


class MainCliParsingTests(unittest.TestCase):
    """Cover the main training CLI argument coercions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.main_mod = reload_module("main")

    def test_iid_false_is_parsed_as_a_real_boolean(self) -> None:
        """Ensure ``--iid false`` does not collapse to a truthy string."""
        with mock.patch.object(sys, "argv", ["main.py", "--iid", "false"]):
            args = self.main_mod.parse_args()

        self.assertIs(args.iid, False)

    def test_min_tree_size_false_is_parsed_as_a_boolean_sentinel(self) -> None:
        """Ensure ``--min_tree_size false`` stays as the FedGP sentinel value."""
        with mock.patch.object(sys, "argv", ["main.py", "--min_tree_size", "false"]):
            args = self.main_mod.parse_args()

        self.assertIs(args.min_tree_size, False)
