"""Tests for the run-config validator shared by the API and the CLI worker."""

from __future__ import annotations

import unittest

from tests.helpers import valid_run_config
from utils.run_config_validation import validate_core_run_config


class CoreRunConfigValidationTests(unittest.TestCase):
    """Cover the web-independent run-config validator shared by the API and the CLI worker."""

    def test_accepts_minimal_valid_config(self) -> None:
        ok, msg = validate_core_run_config(valid_run_config())
        self.assertTrue(ok)
        self.assertEqual("", msg)
        # The default custom-model branch normalizes the model name in place.
        cfg = valid_run_config()
        validate_core_run_config(cfg)
        self.assertEqual("DefaultNet", cfg["custom_model_name"])

    def test_rejects_unknown_method(self) -> None:
        cfg = valid_run_config()
        cfg["method"] = "fed_unknown"
        ok, msg = validate_core_run_config(cfg)
        self.assertFalse(ok)
        self.assertIn("method must be one of", msg)

    def test_rejects_momentum_at_one(self) -> None:
        cfg = valid_run_config()
        cfg["momentum"] = 1.0
        ok, msg = validate_core_run_config(cfg)
        self.assertFalse(ok)
        self.assertIn("momentum", msg)

    def test_rejects_non_positive_mu_only_for_fed_prox(self) -> None:
        prox = valid_run_config()
        prox["method"] = "fed_prox"
        prox["mu"] = 0
        ok, msg = validate_core_run_config(prox)
        self.assertFalse(ok)
        self.assertIn("mu", msg)

        # The same mu is irrelevant (and must not block) for a non-prox method.
        non_prox = valid_run_config()
        non_prox["mu"] = 0
        ok, _ = validate_core_run_config(non_prox)
        self.assertTrue(ok)

    def test_normalizes_fedprox_weighted_to_bool(self) -> None:
        cfg = valid_run_config()
        cfg["method"] = "fed_prox"
        cfg["mu"] = 0.1
        cfg["fedprox_weighted"] = "false"  # string from some config sources
        ok, _ = validate_core_run_config(cfg)
        self.assertTrue(ok)
        self.assertIs(False, cfg["fedprox_weighted"])

    def test_defaults_fedprox_weighted_to_true(self) -> None:
        cfg = valid_run_config()
        cfg["method"] = "fed_prox"
        cfg["mu"] = 0.1
        cfg.pop("fedprox_weighted", None)
        ok, _ = validate_core_run_config(cfg)
        self.assertTrue(ok)
        self.assertIs(True, cfg["fedprox_weighted"])

    def test_rejects_non_positive_warmup_and_repeat(self) -> None:
        for key in ("server_warmup_epochs", "repeat"):
            cfg = valid_run_config()
            cfg[key] = 0
            ok, msg = validate_core_run_config(cfg)
            self.assertFalse(ok, f"{key}=0 should be rejected")
            self.assertIn(key, msg)


if __name__ == "__main__":
    unittest.main()
