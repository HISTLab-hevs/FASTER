"""Tests for client-churn normalization and per-round client eligibility."""

from __future__ import annotations

import unittest

import numpy as np

from tests.helpers import reload_module


class ClientChurnRuntimeTests(unittest.TestCase):
    """Cover client-churn normalization and round-to-round eligibility behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.main_mod = reload_module("main")

    def test_zero_probability_churn_preserves_full_participation(self) -> None:
        """Ensure zero churn probabilities keep every eligible client active."""
        cfg = self.main_mod._normalize_client_churn_config(
            {
                "num_clients": 4,
                "initial_eligible_clients": 4,
                "death_prob": 0,
                "new_client_prob": 0,
                "seed": 0,
            }
        )
        rng = np.random.default_rng(cfg["seed"])
        eligible = self.main_mod._initial_eligible_clients(
            cfg["total_clients"],
            cfg["initial_eligible_clients"],
        )

        updated, round_info = self.main_mod._apply_client_churn_round(
            eligible,
            cfg["total_clients"],
            cfg["death_prob"],
            cfg["new_client_prob"],
            rng,
        )

        self.assertEqual({0, 1, 2, 3}, updated)
        self.assertEqual([0, 1, 2, 3], round_info["sampled_clients"])
        self.assertEqual(0, round_info["dropped_to_inactive_count"])
        self.assertEqual(0, round_info["newly_eligible_count"])

    def test_client_churn_is_deterministic_for_same_seed(self) -> None:
        """Ensure the churn simulation is deterministic for a fixed random seed."""
        def run_sequence(seed: int):
            rng = np.random.default_rng(seed)
            eligible = self.main_mod._initial_eligible_clients(5, 3)
            rounds = []
            for _ in range(4):
                eligible, round_info = self.main_mod._apply_client_churn_round(
                    eligible,
                    5,
                    0.4,
                    0.3,
                    rng,
                )
                rounds.append(round_info)
            return rounds

        self.assertEqual(run_sequence(13), run_sequence(13))
        self.assertNotEqual(run_sequence(13), run_sequence(14))

    def test_initial_eligible_helper_clamps_boundaries(self) -> None:
        """Ensure the initial eligible helper clamps requests to the valid pool size."""
        self.assertEqual({0}, self.main_mod._initial_eligible_clients(3, -5))
        self.assertEqual({0, 1, 2}, self.main_mod._initial_eligible_clients(3, 10))

    def test_client_churn_preserves_pool_bounds_and_uniqueness(self) -> None:
        """Ensure churn never samples duplicate or out-of-range client identifiers."""
        rng = np.random.default_rng(9)
        eligible = self.main_mod._initial_eligible_clients(6, 2)

        for _ in range(8):
            eligible, round_info = self.main_mod._apply_client_churn_round(
                eligible,
                6,
                0.8,
                0.6,
                rng,
            )
            sampled = round_info["sampled_clients"]
            self.assertEqual(len(sampled), len(set(sampled)))
            self.assertTrue(all(0 <= client_id < 6 for client_id in sampled))
            self.assertEqual(round_info["eligible_client_count"], len(sampled))

    def test_zero_initial_eligible_clients_can_reactivate_from_inactive_pool(self) -> None:
        """Ensure a fully inactive pool can reactivate clients in later rounds."""
        rng = np.random.default_rng(1)
        eligible = self.main_mod._initial_eligible_clients(3, 0)

        updated, round_info = self.main_mod._apply_client_churn_round(
            eligible,
            3,
            0.0,
            1.0,
            rng,
        )

        self.assertEqual({0, 1, 2}, updated)
        self.assertEqual([1, 2], round_info["newly_eligible_clients"])
        self.assertEqual(3, round_info["sampled_client_count"])

    def test_client_churn_safeguard_keeps_one_eligible_client_when_pool_is_non_empty(self) -> None:
        """Ensure the safeguard keeps one client eligible when churn would empty the pool."""
        rng = np.random.default_rng(5)

        updated, round_info = self.main_mod._apply_client_churn_round(
            {0},
            4,
            0.9,
            0.0,
            rng,
        )

        self.assertEqual({0}, updated)
        self.assertTrue(round_info["eligibility_safeguard_applied"])
        self.assertEqual(0, round_info["eligibility_safeguard_client"])
        self.assertEqual(1, round_info["eligible_client_count"])
