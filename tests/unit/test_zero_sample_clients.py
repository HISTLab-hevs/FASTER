"""Tests for training behavior when a client receives zero samples."""

from __future__ import annotations

import unittest

import torch

from tests.helpers import reload_module


class ZeroSampleAggregationTests(unittest.TestCase):
    """Cover zero-sample client safety in aggregation methods."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.agg_mod = reload_module("utils.aggregation_methods")

    @staticmethod
    def _make_model(weight_value: float) -> torch.nn.Module:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(weight_value)
        return model

    def test_fed_avgw_ignores_zero_sample_clients(self) -> None:
        """Ensure zero-sample clients do not alter weighted averaging."""
        global_model = self._make_model(0.0)
        active_client = self._make_model(2.0)
        zero_client = self._make_model(10.0)

        updated = self.agg_mod.fed_avgw(
            global_model,
            [active_client.state_dict(), zero_client.state_dict()],
            [5, 0],
        )

        self.assertAlmostEqual(2.0, float(updated.weight.item()))

    def test_fed_avgw_accepts_partial_round_participation(self) -> None:
        """Allow aggregating only the clients that contributed this round."""
        global_model = self._make_model(0.0)
        participant_a = self._make_model(1.0)
        participant_b = self._make_model(3.0)

        updated = self.agg_mod.fed_avgw(
            global_model,
            [participant_a.state_dict(), participant_b.state_dict()],
            [2, 6],
        )

        # Weighted mean over two participants: (1*2 + 3*6) / 8 = 2.5
        self.assertAlmostEqual(2.5, float(updated.weight.item()))

    def test_fed_avgw_raises_for_mismatched_metadata_lengths(self) -> None:
        """Reject inconsistent per-client model/sample metadata lengths."""
        with self.assertRaisesRegex(ValueError, "fed_avgw metadata length mismatch"):
            self.agg_mod.fed_avgw(
                self._make_model(0.0),
                [self._make_model(1.0).state_dict(), self._make_model(2.0).state_dict()],
                [5],
            )

    def test_fed_avgw_raises_when_no_valid_clients_remain(self) -> None:
        """Reject rounds where every provided contribution is non-trainable."""
        with self.assertRaisesRegex(ValueError, "no valid client updates"):
            self.agg_mod.fed_avgw(
                self._make_model(3.5),
                [self._make_model(9.0).state_dict()],
                [0],
            )

    def test_fed_nova_ignores_zero_sample_clients(self) -> None:
        """Ensure FedNova remains finite when a client round contains an empty subset."""
        global_model = self._make_model(0.0)
        active_client = self._make_model(1.5)
        zero_client = self._make_model(9.0)

        with_zero = self.agg_mod.fed_nova(
            self._make_model(0.0),
            [active_client.state_dict(), zero_client.state_dict()],
            [5, 0],
            [2, 0],
            [1.0, 1.0],
            [0.0, 0.0],
        )
        without_zero = self.agg_mod.fed_nova(
            self._make_model(0.0),
            [active_client.state_dict()],
            [5],
            [2],
            [1.0],
            [0.0],
        )

        self.assertAlmostEqual(float(without_zero.weight.item()), float(with_zero.weight.item()))
        self.assertTrue(torch.isfinite(with_zero.weight).all())

    def test_fed_nova_accepts_partial_round_participation(self) -> None:
        """Allow FedNova aggregation with only participating clients in a round."""
        global_model = self._make_model(0.0)
        participant_a = self._make_model(2.0)
        participant_b = self._make_model(4.0)

        updated = self.agg_mod.fed_nova(
            global_model,
            [participant_a.state_dict(), participant_b.state_dict()],
            [1, 3],
            [2, 2],
            [1.0, 1.0],
            [0.0, 0.0],
        )

        # Under matching local steps/lrs (and no momentum), this should match weighted averaging.
        self.assertAlmostEqual(3.5, float(updated.weight.item()))

    def test_fed_nova_raises_for_mismatched_metadata_lengths(self) -> None:
        """Reject inconsistent per-client FedNova metadata lengths."""
        with self.assertRaisesRegex(ValueError, "fed_nova metadata length mismatch"):
            self.agg_mod.fed_nova(
                self._make_model(0.0),
                [self._make_model(1.0).state_dict(), self._make_model(2.0).state_dict()],
                [5, 5],
                [1],
                [0.1, 0.1],
                [0.0, 0.0],
            )

    def test_fed_nova_raises_when_no_valid_clients_remain(self) -> None:
        """Reject rounds where every provided FedNova contribution is invalid."""
        with self.assertRaisesRegex(ValueError, "no valid client updates"):
            self.agg_mod.fed_nova(
                self._make_model(4.25),
                [self._make_model(8.0).state_dict()],
                [0],
                [0],
                [1.0],
                [0.0],
            )

    @staticmethod
    def _bn_model(num_batches: int) -> torch.nn.Module:
        model = torch.nn.BatchNorm1d(2)
        with torch.no_grad():
            model.num_batches_tracked.fill_(num_batches)
        return model

    def test_fed_avg_keeps_integer_buffers_integer(self) -> None:
        """fed_avg must preserve integer buffer dtypes (e.g. num_batches_tracked)."""
        a = self._bn_model(100)
        b = self._bn_model(200)
        updated = self.agg_mod.fed_avg(self._bn_model(0), [a.state_dict(), b.state_dict()])
        nbt = updated.state_dict()["num_batches_tracked"]
        self.assertEqual(torch.int64, nbt.dtype)
        self.assertEqual(150, int(nbt.item()))  # round(mean(100, 200))

    def test_fed_avgw_rounds_integer_buffers(self) -> None:
        """Weighted FedAvg rounds (not truncates) integer buffers and keeps their dtype."""
        a = self._bn_model(100)
        b = self._bn_model(101)
        updated = self.agg_mod.fed_avgw(
            self._bn_model(0), [a.state_dict(), b.state_dict()], [1, 2]
        )
        nbt = updated.state_dict()["num_batches_tracked"]
        self.assertEqual(torch.int64, nbt.dtype)
        # weighted = (100*1 + 101*2)/3 = 100.667 -> round -> 101 (truncation would give 100)
        self.assertEqual(101, int(nbt.item()))

    def test_fed_prox_weighted_uses_sample_weighting(self) -> None:
        """FedProx weighted aggregation matches fed_avgw (n_k/n)."""
        a = self._make_model(1.0)
        b = self._make_model(3.0)
        updated = self.agg_mod.fed_prox(
            self._make_model(0.0),
            [a.state_dict(), b.state_dict()],
            [2, 6],
            weighted=True,
        )
        # (1*2 + 3*6) / 8 = 2.5
        self.assertAlmostEqual(2.5, float(updated.weight.item()))

    def test_fed_prox_uniform_ignores_sample_counts(self) -> None:
        """FedProx with weighted=False is a plain uniform average."""
        a = self._make_model(1.0)
        b = self._make_model(3.0)
        updated = self.agg_mod.fed_prox(
            self._make_model(0.0),
            [a.state_dict(), b.state_dict()],
            [2, 6],
            weighted=False,
        )
        # Uniform: (1 + 3) / 2 = 2.0, regardless of sample counts.
        self.assertAlmostEqual(2.0, float(updated.weight.item()))

    def test_fed_prox_weighted_requires_sample_counts(self) -> None:
        """Weighted FedProx needs per-client sample counts to weight by."""
        with self.assertRaisesRegex(ValueError, "requires client_num_samples"):
            self.agg_mod.fed_prox(
                self._make_model(0.0),
                [self._make_model(1.0).state_dict()],
                None,
                weighted=True,
            )

    def test_fednova_local_coefficient_matches_paper_formula(self) -> None:
        """The normalization coefficient follows the FedNova momentum formula."""
        coef = self.agg_mod._fednova_local_coefficient
        # rho = 0 -> vanilla SGD: the coefficient is just the number of local steps.
        self.assertAlmostEqual(5.0, coef(5, 0.0))
        # tau = 1 -> momentum has not accumulated yet, so the coefficient is 1.
        self.assertAlmostEqual(1.0, coef(1, 0.9))
        # tau = 2, rho = 0.9 -> (2 - 0.9*(1 - 0.9**2)/0.1) / 0.1 = 2.9
        self.assertAlmostEqual(2.9, coef(2, 0.9))

    def test_fed_nova_momentum_changes_normalization(self) -> None:
        """Momentum must enter the normalization: it shifts the aggregate vs vanilla SGD."""
        models = [self._make_model(2.0).state_dict(), self._make_model(8.0).state_dict()]
        # Equal samples but different local steps, so the per-client coefficients differ.
        no_momentum = self.agg_mod.fed_nova(
            self._make_model(0.0), models, [4, 4], [1, 5], [1.0, 1.0], [0.0, 0.0]
        )
        with_momentum = self.agg_mod.fed_nova(
            self._make_model(0.0), models, [4, 4], [1, 5], [1.0, 1.0], [0.9, 0.9]
        )
        # Vanilla path is unchanged: tau_eff=3, update = 2*1.5 + 8*0.3 = 5.4.
        self.assertAlmostEqual(5.4, float(no_momentum.weight.item()), places=4)
        # Momentum reweights the clients, so the result must move and stay finite.
        self.assertNotAlmostEqual(
            float(no_momentum.weight.item()), float(with_momentum.weight.item()), places=2
        )
        self.assertTrue(torch.isfinite(with_momentum.weight).all())
