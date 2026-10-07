"""Validation-contract tests shared by the frontend, backend, and run config."""

from __future__ import annotations

import contextlib
import asyncio
import json
import os
import re
import shutil
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import reload_module, valid_run_config


class ApiValidationTests(unittest.TestCase):
    """Verify frontend, backend, and config validation contracts stay aligned."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api_handlers = reload_module("web_backend.api_handlers.api")

    def test_validate_run_config_accepts_valid_minimal_config(self) -> None:
        """Ensure a minimal valid run configuration is accepted and normalized."""
        cfg = valid_run_config()

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok)
        self.assertEqual("", msg)
        self.assertEqual("DefaultNet", cfg["custom_model_name"])
        self.assertEqual("auto", cfg["round_train_schedule_mode"])
        self.assertEqual([], cfg["round_train_schedule_percentages"])
        self.assertEqual([50.0, 50.0], cfg["resolved_round_train_schedule_percentages"])
        self.assertEqual("auto", cfg["round_client_allocation_mode"])
        self.assertEqual([], cfg["round_client_allocation_percentages"])
        self.assertEqual([[50.0, 50.0], [50.0, 50.0]], cfg["resolved_round_client_allocation_percentages"])

    def test_validate_run_config_rejects_unknown_method(self) -> None:
        """An unrecognized method must be rejected early, not fail late in the worker."""
        cfg = valid_run_config()
        cfg["method"] = "fed_bogus"

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertIn("method must be one of", msg)

    def test_validate_run_config_rejects_momentum_out_of_range(self) -> None:
        """Momentum must be in [0, 1): 1.0 would divide by zero in FedNova normalization."""
        for bad in (1.0, 1.5, -0.1):
            cfg = valid_run_config()
            cfg["momentum"] = bad
            ok, msg = self.api_handlers._validate_run_config(cfg)
            self.assertFalse(ok, f"momentum={bad} should be rejected")
            self.assertIn("momentum", msg)

    def test_validate_run_config_accepts_momentum_near_one(self) -> None:
        """Momentum just below 1 stays valid."""
        cfg = valid_run_config()
        cfg["momentum"] = 0.99

        ok, _ = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok)

    def test_validate_run_config_requires_positive_mu_for_fed_prox(self) -> None:
        """A non-positive mu silently disables the FedProx proximal term, so reject it."""
        cfg = valid_run_config()
        cfg["method"] = "fed_prox"
        cfg["mu"] = 0

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertIn("mu", msg)

    def test_validate_run_config_accepts_valid_mu_for_fed_prox(self) -> None:
        """A positive in-range mu is accepted for FedProx."""
        cfg = valid_run_config()
        cfg["method"] = "fed_prox"
        cfg["mu"] = 0.1

        ok, _ = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok)

    def test_validate_run_config_ignores_mu_for_non_prox_methods(self) -> None:
        """mu is irrelevant outside FedProx, so an out-of-range value must not block other methods."""
        cfg = valid_run_config()  # method == fed_avg
        cfg["mu"] = 0

        ok, _ = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok)

    def test_validate_run_config_rejects_split_ratio_out_of_range(self) -> None:
        """Split ratios must stay strictly inside (0, 1)."""
        for key, bad in (("train_val_split", 1.0), ("server_data_percentage", 0.0)):
            cfg = valid_run_config()
            cfg[key] = bad
            ok, msg = self.api_handlers._validate_run_config(cfg)
            self.assertFalse(ok, f"{key}={bad} should be rejected")
            self.assertIn(key, msg)

    def test_validate_run_config_rejects_bad_client_learning_rate(self) -> None:
        """Every provided per-client learning rate must be positive and within bounds."""
        cfg = valid_run_config()
        cfg["client_learning_rates"] = [0.01, -1.0]

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertIn("client_learning_rates", msg)

    def test_validate_run_config_rejects_non_positive_warmup_and_repeat(self) -> None:
        """server_warmup_epochs and repeat must be > 0 (no upper bound)."""
        for key in ("server_warmup_epochs", "repeat"):
            cfg = valid_run_config()
            cfg[key] = 0
            ok, msg = self.api_handlers._validate_run_config(cfg)
            self.assertFalse(ok, f"{key}=0 should be rejected")
            self.assertIn(key, msg)

    def test_validate_run_config_accepts_manual_round_train_schedule(self) -> None:
        """Ensure manual round-level train schedules validate and normalize."""
        cfg = valid_run_config()
        cfg["round_train_schedule_mode"] = "manual"
        cfg["round_train_schedule_percentages"] = [40, 60]

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok)
        self.assertEqual("", msg)
        self.assertEqual([40.0, 60.0], cfg["round_train_schedule_percentages"])
        self.assertEqual([40.0, 60.0], cfg["resolved_round_train_schedule_percentages"])

    def test_validate_run_config_accepts_no_split_round_train_schedule(self) -> None:
        """Ensure no-split mode resolves every round to full (100%) coverage."""
        cfg = valid_run_config()
        cfg["round_train_schedule_mode"] = "no_split"

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok)
        self.assertEqual("", msg)
        self.assertEqual("no_split", cfg["round_train_schedule_mode"])
        self.assertEqual([], cfg["round_train_schedule_percentages"])
        self.assertEqual([100.0, 100.0], cfg["resolved_round_train_schedule_percentages"])

    def test_validate_run_config_rejects_manual_round_schedule_with_wrong_sum(self) -> None:
        """Ensure manual round schedules must sum to 100."""
        cfg = valid_run_config()
        cfg["round_train_schedule_mode"] = "manual"
        cfg["round_train_schedule_percentages"] = [20, 20]

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual("round_train_schedule_percentages must sum to 100", msg)

    def test_validate_run_config_rejects_manual_round_schedule_with_wrong_length(self) -> None:
        """Ensure manual round schedules match the aggregation round count."""
        cfg = valid_run_config()
        cfg["round_train_schedule_mode"] = "manual"
        cfg["round_train_schedule_percentages"] = [100]

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual(
            "round_train_schedule_percentages count must match aggregation rounds",
            msg,
        )

    def test_validate_run_config_accepts_manual_round_client_allocation(self) -> None:
        """Ensure manual per-round client allocation validates and normalizes."""
        cfg = valid_run_config()
        cfg["round_client_allocation_mode"] = "manual"
        cfg["round_client_allocation_percentages"] = [[40, 60], [25, 75]]

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok)
        self.assertEqual("", msg)
        self.assertEqual([[40.0, 60.0], [25.0, 75.0]], cfg["round_client_allocation_percentages"])
        self.assertEqual([[40.0, 60.0], [25.0, 75.0]], cfg["resolved_round_client_allocation_percentages"])

    def test_validate_run_config_rejects_manual_round_client_allocation_with_wrong_sum(self) -> None:
        """Ensure each manual client-allocation round must sum to 100."""
        cfg = valid_run_config()
        cfg["round_client_allocation_mode"] = "manual"
        cfg["round_client_allocation_percentages"] = [[40, 50], [25, 75]]

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual("round_client_allocation_percentages[1] must sum to 100", msg)

    def test_validate_run_config_rejects_manual_round_client_allocation_with_wrong_shape(self) -> None:
        """Ensure each manual client-allocation row matches the client count."""
        cfg = valid_run_config()
        cfg["round_client_allocation_mode"] = "manual"
        cfg["round_client_allocation_percentages"] = [[100], [100]]

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual(
            "round_client_allocation_percentages[1] count must match num_clients",
            msg,
        )

    def test_frontend_dataset_options_remain_aligned_with_backend_contract(self) -> None:
        """Ensure frontend dataset choices remain in sync with backend-supported datasets."""
        ui_data_path = Path(__file__).resolve().parents[2] / "wrapper" / "js" / "ui_data.js"
        source = ui_data_path.read_text(encoding="utf-8")

        block_match = re.search(
            r"datasetOptions:\s*\[(?P<body>.*?)\]\s*,\s*/\*\*\s*@type\s*\{Array<Array<string>>\}\s*\*/\s*evaluationSplitModeOptions:",
            source,
            re.S,
        )
        self.assertIsNotNone(block_match, "Could not locate datasetOptions block in ui_data.js")
        block = block_match.group("body")
        option_values = re.findall(r"\[\s*'([^']+)'\s*,", block)

        contracts = reload_module("web_backend.contracts")
        backend_supported = set(contracts.SUPPORTED_DATASET_NAMES)
        expected_frontend_values = (
            backend_supported - {"custom_csv", "custom_image_npz", "custom_image_folder"}
        ) | {"custom"}
        self.assertEqual(expected_frontend_values, set(option_values))

    def test_frontend_custom_dataset_views_use_precise_format_metadata(self) -> None:
        """Ensure custom-dataset views distinguish dataset type from upload format metadata."""
        repo_root = Path(__file__).resolve().parents[2]
        setup_source = (repo_root / "wrapper" / "js" / "modules" / "setup.module.js").read_text(encoding="utf-8")
        datasets_source = (repo_root / "wrapper" / "js" / "modules" / "datasets.module.js").read_text(encoding="utf-8")
        training_source = (repo_root / "wrapper" / "js" / "modules" / "training.module.js").read_text(encoding="utf-8")

        self.assertIn("dataset_format", setup_source)
        self.assertIn("dataset_format", datasets_source)
        self.assertIn("custom_dataset_format", training_source)
        self.assertNotIn("image npz' : 'tabular'", setup_source)

    def test_server_warmup_epochs_is_wired_through_existing_config_surfaces(self) -> None:
        """Ensure server warm-up epochs remain wired through UI, config, and runtime layers."""
        repo_root = Path(__file__).resolve().parents[2]
        render_source = (repo_root / "wrapper" / "js" / "modules" / "render.module.js").read_text(encoding="utf-8")
        training_source = (repo_root / "wrapper" / "js" / "modules" / "training.module.js").read_text(encoding="utf-8")
        app_source = (repo_root / "wrapper" / "js" / "app.js").read_text(encoding="utf-8")
        ui_data_source = (repo_root / "wrapper" / "js" / "ui_data.js").read_text(encoding="utf-8")
        config_source = (repo_root / "config.yaml").read_text(encoding="utf-8")
        runtime_source = (repo_root / "main.py").read_text(encoding="utf-8")
        docs_source = (repo_root / "wrapper" / "templates" / "docs.html").read_text(encoding="utf-8")
        dashboard_source = (repo_root / "wrapper" / "templates" / "dashboard.html").read_text(encoding="utf-8")

        self.assertIn("Server Warm-up Epochs", render_source)
        self.assertNotIn("Global Epochs", render_source)
        self.assertIn("server_warmup_epochs:", training_source)
        self.assertIn("server_warmup_epochs", app_source)
        self.assertIn("server_warmup_epochs:", ui_data_source)
        self.assertIn("server_warmup_epochs:", config_source)
        self.assertIn('config.get("server_warmup_epochs")', runtime_source)
        self.assertIn("server_warmup_epochs:", docs_source)
        self.assertIn("server_warmup_epochs: null", dashboard_source)

    def test_legacy_global_model_epochs_alias_is_normalized(self) -> None:
        """Ensure legacy global_model_epochs config input still normalizes to the canonical name."""
        cfg = valid_run_config()
        cfg.pop("server_warmup_epochs", None)
        cfg["global_model_epochs"] = 3

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok, msg)
        self.assertEqual(3, cfg["server_warmup_epochs"])
        self.assertNotIn("global_model_epochs", cfg)

    def test_clients_record_train_metrics_and_global_uses_whole_validation(self) -> None:
        """Ensure per-client metrics are evaluated on the client's training data.

        The whole validation set evaluates the aggregated global model; each client is
        evaluated on its own training data (round_train_loader) every epoch.
        """
        repo_root = Path(__file__).resolve().parents[2]
        runtime_source = (repo_root / "main.py").read_text(encoding="utf-8")

        # Per-client metrics (loss + accuracy/precision/recall/f1/roc_auc) come from the
        # client's own training data, so train() receives the round_train_loader to evaluate.
        self.assertIn('metrics["clients"][client]["loss"].append(loss)', runtime_source)
        self.assertIn('metrics["clients"][client]["accuracy"].append(accuracy)', runtime_source)
        self.assertIn('metrics["clients"][client]["roc_auc"].append(roc_auc)', runtime_source)
        self.assertIn("val_loader=round_train_loader", runtime_source)
        self.assertNotIn("client_val_loaders", runtime_source)
        # The aggregated global model is evaluated on the whole validation set.
        self.assertIn("server_val_loader = dataset.val_data", runtime_source)

    def test_frontend_launch_validation_checks_fedgp_against_full_method_set(self) -> None:
        """Ensure launch validation checks the full selected method set for FedGP."""
        repo_root = Path(__file__).resolve().parents[2]
        training_source = (repo_root / "wrapper" / "js" / "modules" / "training.module.js").read_text(encoding="utf-8")

        self.assertIn("_getLaunchMethods(cfg)", training_source)
        self.assertIn("_hasFedGpSelected(cfg)", training_source)
        self.assertIn(
            "return this._getLaunchMethods(cfg).some((method) => String(method || '').toLowerCase() === 'fed_gp');",
            training_source,
        )

    def test_client_churn_fields_are_wired_through_existing_config_surfaces(self) -> None:
        """Ensure client-churn fields remain wired across UI and config surfaces."""
        repo_root = Path(__file__).resolve().parents[2]
        render_source = (repo_root / "wrapper" / "js" / "modules" / "render.module.js").read_text(encoding="utf-8")
        training_source = (repo_root / "wrapper" / "js" / "modules" / "training.module.js").read_text(encoding="utf-8")
        setup_source = (repo_root / "wrapper" / "js" / "modules" / "setup.module.js").read_text(encoding="utf-8")
        app_source = (repo_root / "wrapper" / "js" / "app.js").read_text(encoding="utf-8")
        ui_data_source = (repo_root / "wrapper" / "js" / "ui_data.js").read_text(encoding="utf-8")
        config_source = (repo_root / "config.yaml").read_text(encoding="utf-8")

        for key in ("initial_eligible_clients", "death_prob", "new_client_prob", "seed"):
            self.assertIn(f"'{key}'", render_source)
            self.assertIn(key, training_source)
            self.assertIn(f"'{key}'", app_source)
            self.assertIn(f"{key}:", config_source)
            self.assertIn(f"{key}:", ui_data_source)
        self.assertIn("client-pool-warning", render_source)
        self.assertIn("client-churn-warning", render_source)
        self.assertIn("_updateClientChurnWarnings()", setup_source)
        self.assertIn("_validateClientPoolRules", training_source)
        self.assertIn("_validateChurnProbabilityRules", training_source)

    def test_round_train_schedule_fields_are_wired_through_existing_config_surfaces(self) -> None:
        """Ensure round-level train schedule fields stay aligned across UI and config surfaces."""
        repo_root = Path(__file__).resolve().parents[2]
        render_source = (repo_root / "wrapper" / "js" / "modules" / "render.module.js").read_text(encoding="utf-8")
        training_source = (repo_root / "wrapper" / "js" / "modules" / "training.module.js").read_text(encoding="utf-8")
        setup_source = (repo_root / "wrapper" / "js" / "modules" / "setup.module.js").read_text(encoding="utf-8")
        app_source = (repo_root / "wrapper" / "js" / "app.js").read_text(encoding="utf-8")
        ui_data_source = (repo_root / "wrapper" / "js" / "ui_data.js").read_text(encoding="utf-8")
        config_source = (repo_root / "config.yaml").read_text(encoding="utf-8")
        dashboard_source = (repo_root / "wrapper" / "templates" / "dashboard.html").read_text(encoding="utf-8")
        docs_source = (repo_root / "wrapper" / "templates" / "docs.html").read_text(encoding="utf-8")

        self.assertIn("round_train_schedule_mode", render_source)
        self.assertIn("round_train_schedule_percentages", render_source)
        self.assertIn("round_client_allocation_mode", render_source)
        self.assertIn("round_client_allocation_percentages", render_source)
        self.assertIn("round-train-schedule-simple-note", render_source)
        self.assertIn("round-train-schedule-summary-top", render_source)
        self.assertIn("round-train-schedule-summary-editor", render_source)
        self.assertIn("round-client-allocation-summary-top", render_source)
        self.assertIn("round-client-allocation-summary-editor", render_source)
        self.assertIn("round_train_schedule_mode:", training_source)
        self.assertIn("round_train_schedule_percentages:", training_source)
        self.assertIn("round_client_allocation_mode:", training_source)
        self.assertIn("round_client_allocation_percentages:", training_source)
        self.assertIn("_validateRoundTrainScheduleRules(cfg)", training_source)
        self.assertIn("_validateRoundClientAllocationRules(cfg)", training_source)
        self.assertIn("_toggleRoundTrainScheduleInputs()", setup_source)
        self.assertIn("_syncRoundClientAllocationInputs()", setup_source)
        self.assertIn("'round_train_schedule_mode'", app_source)
        self.assertIn("'round_train_schedule_percentages'", app_source)
        self.assertIn("'round_client_allocation_mode'", app_source)
        self.assertIn("'round_client_allocation_percentages'", app_source)
        self.assertIn("round_train_schedule_mode:", ui_data_source)
        self.assertIn("round_train_schedule_percentages:", ui_data_source)
        self.assertIn("round_client_allocation_mode:", ui_data_source)
        self.assertIn("round_client_allocation_percentages:", ui_data_source)
        self.assertIn("round_train_schedule_mode:", config_source)
        self.assertIn("round_train_schedule_percentages:", config_source)
        self.assertIn("round_client_allocation_mode:", config_source)
        self.assertIn("round_client_allocation_percentages:", config_source)
        self.assertIn("round_train_schedule_mode: auto", dashboard_source)
        self.assertIn("round_train_schedule_percentages: []", dashboard_source)
        self.assertIn("round_client_allocation_mode: auto", dashboard_source)
        self.assertIn("round_client_allocation_percentages: []", dashboard_source)
        self.assertIn("round_train_schedule_mode: auto", docs_source)
        self.assertIn("round_train_schedule_percentages: []", docs_source)
        self.assertIn("round_client_allocation_mode: auto", docs_source)
        self.assertIn("round_client_allocation_percentages: []", docs_source)
        self.assertIn("round-client-allocation-runtime-note", render_source)
        self.assertIn("Allocation percentages describe the intended client plan", render_source)
        self.assertIn("very small shares can round down to 0 samples", render_source)
        self.assertIn("classList.toggle('hidden', isManual || roundCount < 1)", setup_source)
        self.assertIn("classList.toggle('hidden', !isManual || roundCount < 1)", setup_source)
        self.assertIn('step="0.01"', setup_source)
        self.assertIn("_buildRoundedPercentageDistribution", setup_source)
        self.assertIn("_updateRoundClientAllocationRuntimeNote", setup_source)
        self.assertIn("very small shares can still round down to 0 samples", setup_source)

    def test_round_client_allocation_ui_explains_zero_sample_runtime_skips(self) -> None:
        """Ensure the round allocation UI explains discrete runtime skipping for empty allocations."""
        repo_root = Path(__file__).resolve().parents[2]
        render_source = (repo_root / "wrapper" / "js" / "modules" / "render.module.js").read_text(encoding="utf-8")
        setup_source = (repo_root / "wrapper" / "js" / "modules" / "setup.module.js").read_text(encoding="utf-8")
        ui_data_source = (repo_root / "wrapper" / "js" / "ui_data.js").read_text(encoding="utf-8")

        self.assertIn("allocation percentages describe the intended client plan", render_source.lower())
        self.assertIn("some clients can still round down to 0 samples", ui_data_source.lower())
        self.assertIn("clients that land on 0 samples are skipped from training and aggregation", setup_source.lower())

    def test_round_percentage_editors_use_two_decimal_precision_and_exact_defaults(self) -> None:
        """Ensure the round editors cap visible precision and keep default splits summing to 100."""
        repo_root = Path(__file__).resolve().parents[2]
        setup_source = (repo_root / "wrapper" / "js" / "modules" / "setup.module.js").read_text(encoding="utf-8")

        self.assertIn("_buildRoundedPercentageDistribution", setup_source)
        self.assertIn('step="0.01"', setup_source)
        self.assertIn("Number(value).toFixed(2)", setup_source)
        self.assertIn("Number(defaultRow?.[clientIdx] ?? 0)", setup_source)

    def test_simple_mode_keeps_fedgp_available_with_fixed_preset_notice(self) -> None:
        """Ensure simple mode still exposes FedGP with its fixed preset guidance."""
        repo_root = Path(__file__).resolve().parents[2]
        ui_data_source = (repo_root / "wrapper" / "js" / "ui_data.js").read_text(encoding="utf-8")
        render_source = (repo_root / "wrapper" / "js" / "modules" / "render.module.js").read_text(encoding="utf-8")
        setup_source = (repo_root / "wrapper" / "js" / "modules" / "setup.module.js").read_text(encoding="utf-8")
        dashboard_source = (repo_root / "wrapper" / "templates" / "dashboard.html").read_text(encoding="utf-8")

        self.assertIn("['fed_gp', 'FedGP']", ui_data_source)
        self.assertNotIn("['fed_gp', 'FedGP', 'advanced']", ui_data_source)
        self.assertIn("simple-gp-mode-note", render_source)
        self.assertIn("_getSimpleModeFedGpPreset()", setup_source)
        self.assertIn("_applySimpleModeFedGpPreset()", setup_source)
        self.assertIn("_updateSimpleModeFedGpNotice()", setup_source)
        self.assertIn("FedGP with a fixed preset", dashboard_source)

    def test_scenarios_use_lightweight_ui_mode_marker_and_mode_inference(self) -> None:
        """Ensure scenarios persist a lightweight UI mode marker and infer mode on load."""
        repo_root = Path(__file__).resolve().parents[2]
        scenarios_source = (repo_root / "wrapper" / "js" / "modules" / "training_scenarios.module.js").read_text(encoding="utf-8")
        setup_source = (repo_root / "wrapper" / "js" / "modules" / "setup.module.js").read_text(encoding="utf-8")
        dashboard_source = (repo_root / "wrapper" / "templates" / "dashboard.html").read_text(encoding="utf-8")
        docs_source = (repo_root / "wrapper" / "templates" / "docs.html").read_text(encoding="utf-8")

        self.assertIn("_applyLoadedScenarioConfig(cfg, opts = {})", scenarios_source)
        self.assertIn("ui_mode: this._state.setupMode === 'advanced' ? 'advanced' : 'simple'", scenarios_source)
        self.assertIn("_deleteScenario()", scenarios_source)
        self.assertIn("scenario-delete-btn", scenarios_source)
        self.assertIn("_inferSetupModeFromConfig(cfg)", setup_source)
        self.assertIn("ui_mode: null                           # optional: simple | advanced", dashboard_source)
        self.assertIn("ui_mode: simple", docs_source)
        self.assertIn('id="scenario-delete-btn"', dashboard_source)

    def test_setup_wizard_validates_step_local_fields_before_later_steps(self) -> None:
        """Ensure setup-step validation checks local fields before later-stage rules."""
        repo_root = Path(__file__).resolve().parents[2]
        setup_source = (repo_root / "wrapper" / "js" / "modules" / "setup.module.js").read_text(encoding="utf-8")

        self.assertIn("requireNumber('initial_eligible_clients', 'Initial eligible', { integer: true, min: 1 });", setup_source)
        self.assertIn("requireNumber('seed', 'Seed', { integer: true, min: 0 });", setup_source)
        self.assertIn("requireNumber('server_learning_rate', 'Server learning rate', { min: 0 });", setup_source)
        self.assertIn("requireNumber('server_data_percentage', 'Server data percentage', { min: 0 });", setup_source)
        self.assertIn("requireNumber('death_prob', 'Death probability', { min: 0, max: 0.9 });", setup_source)
        self.assertIn("requireNumber('new_client_prob', 'New client probability', { min: 0, max: 0.9 });", setup_source)
        self.assertIn("this._validateEpochFrequencyRule?.(cfg)", setup_source)
        self.assertIn("this._validateGpConfigRules?.(cfg)", setup_source)
        self.assertNotIn("requireNumber('learning_rate', 'Learning rate', { min: 0 });", setup_source)

    def test_learning_rate_ui_uses_single_visible_client_lr_control(self) -> None:
        """Ensure the UI exposes one visible client learning-rate control and syncs backing state."""
        repo_root = Path(__file__).resolve().parents[2]
        render_source = (repo_root / "wrapper" / "js" / "modules" / "render.module.js").read_text(encoding="utf-8")
        client_lr_source = (repo_root / "wrapper" / "js" / "modules" / "client_learning_rate.module.js").read_text(encoding="utf-8")
        events_source = (repo_root / "wrapper" / "js" / "modules" / "events.module.js").read_text(encoding="utf-8")
        ui_data_source = (repo_root / "wrapper" / "js" / "ui_data.js").read_text(encoding="utf-8")
        app_source = (repo_root / "wrapper" / "js" / "app.js").read_text(encoding="utf-8")
        docs_source = (repo_root / "wrapper" / "templates" / "docs.html").read_text(encoding="utf-8")
        dashboard_source = (repo_root / "wrapper" / "templates" / "dashboard.html").read_text(encoding="utf-8")

        self.assertIn("this._field('server_learning_rate', 'Server Learning Rate', 'number', null, 'any')", render_source)
        self.assertIn('<input id="cfg-learning_rate" type="hidden">', render_source)
        self.assertNotIn("this._field('learning_rate', 'Learning Rate', 'number', null, 'any')", render_source)
        self.assertIn("_syncLearningRateBackingField()", client_lr_source)
        self.assertIn("closest('.lr-compact-block')?.addEventListener('input', () => this._syncLearningRateBackingField?.())", events_source)
        self.assertIn("server_learning_rate:", ui_data_source)
        self.assertIn("Learning rate used only for the initial server-side warm-up optimizer", ui_data_source)
        self.assertNotIn("\n        learning_rate:", ui_data_source)
        self.assertNotIn("cfg.server_learning_rate == null && cfg.learning_rate != null", app_source)
        self.assertNotIn("\nlearning_rate: 0.0005\n", docs_source)
        self.assertNotIn("\nlearning_rate: null", dashboard_source)

    def test_monitor_metric_cards_compact_large_values_and_clip_safely(self) -> None:
        """Ensure monitor metric cards format and clip large values safely."""
        repo_root = Path(__file__).resolve().parents[2]
        training_source = (repo_root / "wrapper" / "js" / "modules" / "training.module.js").read_text(encoding="utf-8")
        styles_source = (repo_root / "wrapper" / "css" / "styles.base.css").read_text(encoding="utf-8")

        self.assertIn("num.toExponential(2)", training_source)
        self.assertIn("text-overflow: ellipsis;", styles_source)
        self.assertIn("white-space: nowrap;", styles_source)

    def test_http_wrappers_do_not_return_raw_exception_strings(self) -> None:
        """Ensure masked JSON helpers keep raw exception text out of responses."""
        app_module = reload_module("web_backend.app")

        with mock.patch.object(app_module._LOGGER, "exception") as log_exception:
            response = app_module._masked_json_error(
                "Internal server error",
                status_code=500,
                log_message="Unhandled exception in /api endpoint",
            )

        self.assertEqual(500, response.status_code)
        self.assertEqual({"error": "Internal server error"}, json.loads(response.body))
        log_exception.assert_called_once_with("Unhandled exception in /api endpoint")

    def test_download_flow_uses_header_auth_without_query_token_fallback(self) -> None:
        """Ensure bearer tokens are resolved from Authorization headers only."""
        app_module = reload_module("web_backend.app")
        repo_root = Path(__file__).resolve().parents[2]
        api_source = (repo_root / "wrapper" / "js" / "api.js").read_text(encoding="utf-8")

        self.assertEqual(
            "token-123",
            app_module._resolve_bearer_token(mock.Mock(headers={"authorization": "Bearer token-123"})),
        )
        self.assertEqual("", app_module._resolve_bearer_token(mock.Mock(headers={})))
        self.assertIn("Authorization: `Bearer ${this._token}`", api_source)
        self.assertIn("bindExportLink(linkEl, runId, onError = null)", api_source)
        self.assertNotIn("?token=${tok}", api_source)

    def test_history_run_list_flow_uses_paged_fetch_and_selected_run_lookup(self) -> None:
        """Ensure history uses paged fetching plus selected-run lookup helpers."""
        repo_root = Path(__file__).resolve().parents[2]
        api_source = (repo_root / "wrapper" / "js" / "api.js").read_text(encoding="utf-8")
        history_list_source = (repo_root / "wrapper" / "js" / "modules" / "history_list.module.js").read_text(encoding="utf-8")

        self.assertIn("async getRunsPage(options = {})", api_source)
        self.assertIn("async getRunsByIds(ids = [])", api_source)
        self.assertIn("await api.getRunsPage({", history_list_source)
        self.assertIn("await api.getRunsByIds(tracked)", history_list_source)
        self.assertIn("historyRunLookup", history_list_source)
        self.assertIn("status !== 'running' && status !== 'queued' && status !== 'pending' && status !== 'failed'", history_list_source)

    def test_history_detail_and_comparison_views_use_canonical_metric_buckets(self) -> None:
        """Ensure the history metric views only reference the canonical split buckets."""
        repo_root = Path(__file__).resolve().parents[2]
        dashboard_source = (repo_root / "wrapper" / "templates" / "dashboard.html").read_text(encoding="utf-8")
        history_source = (repo_root / "wrapper" / "js" / "modules" / "history.module.js").read_text(encoding="utf-8")
        charts_source = (repo_root / "wrapper" / "js" / "charts.js").read_text(encoding="utf-8")
        training_source = (repo_root / "wrapper" / "js" / "modules" / "training.module.js").read_text(encoding="utf-8")
        setup_source = (repo_root / "wrapper" / "js" / "modules" / "setup.module.js").read_text(encoding="utf-8")
        app_source = (repo_root / "wrapper" / "js" / "app.js").read_text(encoding="utf-8")

        self.assertIn("aggregated_val", history_source)
        self.assertIn("aggregated_test", history_source)
        self.assertIn("aggregated_val", charts_source)
        self.assertIn("aggregated_test", charts_source)
        self.assertNotIn("mean_aggregated_", history_source)
        self.assertNotIn("mean_aggregated_", charts_source)
        self.assertIn("Latest Key Metrics", dashboard_source)
        self.assertIn("test metrics are shown for train_val_test runs.", dashboard_source)
        for label in (
            '<span class="metric-label">Accuracy</span>',
            '<span class="metric-label">Loss</span>',
            '<span class="metric-label">Precision</span>',
            '<span class="metric-label">Recall</span>',
            '<span class="metric-label">F1</span>',
            '<span class="metric-label label-with-help">ROC AUC',
        ):
            self.assertIn(label, dashboard_source)
        for label in (
            "Latest Validation Accuracy",
            "Latest Validation Loss",
            "Latest Validation Precision",
            "Latest Validation Recall",
            "Latest Validation F1",
            "Latest Validation ROC AUC",
            "Latest Test Accuracy",
            "Latest Test Loss",
            "Latest Test Precision",
            "Latest Test Recall",
            "Latest Test F1",
            "Latest Test ROC AUC",
        ):
            self.assertNotIn(label, dashboard_source)
        for label in (
            "Best Accuracy",
            "Lowest Loss",
            "Best Precision",
            "Best Recall",
            "Best F1",
            "Best ROC AUC",
        ):
            self.assertIn(label, history_source)
        self.assertIn("_updateMonitorLatestMetrics(", training_source)
        self.assertIn("latest-so-far values", training_source)
        self.assertIn("if (!metrics && logs === this._state.lastClientProgressLogs) return;", training_source)
        self.assertIn("this._state.lastClientProgressLogs = null;", training_source)
        self.assertIn("this._state.lastClientProgressLogs = logs;", training_source)
        self.assertIn("this._syncRunNamesHiddenFromInputs?.();", training_source)
        self.assertIn("payload.methods = methods.slice();", training_source)
        self.assertNotIn("delete payload.methods;", training_source)
        self.assertNotIn("delete payload.run_names;", training_source)
        self.assertIn("hidden.value = values.some(Boolean) ? values.join(',') : '';", setup_source)
        self.assertIn("if (runNamesWrap) runNamesWrap.innerHTML = '';", app_source)
        self.assertIn("const effectiveShowTest = mode !== 'train_val';", training_source)
        self.assertIn("Mode: train_val_test. Validation metrics and test metrics are shown as latest-so-far values in separate splits.", training_source)
        self.assertNotIn("Mode: train_val_test. Validation metrics are shown as latest-so-far values and test metrics are not available for this run.", training_source)
        self.assertIn("renderSection('Final', 'Validation', valBucket)", history_source)
        self.assertIn("renderSection('Final', 'Test', testBucket)", history_source)
        self.assertIn("_historyComparisonMetricLabel(spec)", history_source)
        self.assertIn("const labels = {", history_source)
        self.assertIn("table-fixed hist-summary-table", history_source)
        self.assertIn("best-over-series metric values", history_source)
        self.assertIn("hist-metric-best", history_source)
        self.assertIn("Final metric cards show the last recorded round value", history_source)

    def test_history_summary_tables_keep_consistent_column_order_and_alignment_helpers(self) -> None:
        """Ensure history summary tables keep a stable metric order and shared layout helpers."""
        repo_root = Path(__file__).resolve().parents[2]
        history_source = (repo_root / "wrapper" / "js" / "modules" / "history.module.js").read_text(encoding="utf-8")
        styles_source = (repo_root / "wrapper" / "css" / "styles.components.css").read_text(encoding="utf-8")

        self.assertIn("_historyMetricTableColgroup('compare')", history_source)
        self.assertIn("_historyMetricTableColgroup('repeat')", history_source)
        self.assertIn("hist-summary-repeat-col", styles_source)
        self.assertIn("hist-summary-run-col", styles_source)
        self.assertIn("hist-summary-mode-col", styles_source)
        self.assertIn("hist-summary-metric-col", styles_source)
        self.assertIn("table-layout: fixed;", styles_source)
        self.assertIn("white-space: nowrap;", styles_source)
        self.assertIn("hist-summary-table thead th", styles_source)
        self.assertIn("white-space: normal;", styles_source)

        metric_specs_order = (
            "{ key: 'accuracy', label: 'Accuracy', digits: 3, direction: 'max' }",
            "{ key: 'loss', label: 'Loss', digits: 4, direction: 'min' }",
            "{ key: 'precision', label: 'Precision', digits: 3, direction: 'max' }",
            "{ key: 'recall', label: 'Recall', digits: 3, direction: 'max' }",
            "{ key: 'f1', label: 'F1', digits: 3, direction: 'max' }",
            "key: 'roc_auc'",
        )
        cursor = 0
        for token in metric_specs_order:
            next_pos = history_source.find(token, cursor)
            self.assertNotEqual(-1, next_pos, token)
            cursor = next_pos

        comparison_headers = (
            "Run",
            "Mode",
            "Best Accuracy",
            "Lowest Loss",
            "Best Precision",
            "Best Recall",
            "Best F1",
            "Best ROC AUC",
        )
        for header in comparison_headers:
            self.assertIn(header, history_source)

    def test_history_queued_status_uses_distinct_orange_badge_tone(self) -> None:
        """Ensure queued history entries use a dedicated orange badge tone and remain non-comparable."""
        repo_root = Path(__file__).resolve().parents[2]
        history_list_source = (repo_root / "wrapper" / "js" / "modules" / "history_list.module.js").read_text(encoding="utf-8")
        styles_source = (repo_root / "wrapper" / "css" / "styles.components.css").read_text(encoding="utf-8")

        self.assertIn("status === 'queued' || status === 'pending'", history_list_source)
        self.assertIn("statusBadgeTone === 'queued' ? 'ui-badge-queued'", history_list_source)
        self.assertIn(".ui-badge-queued {", styles_source)
        self.assertIn("background: #ffedd5;", styles_source)
        self.assertIn("color: #c2410c;", styles_source)

    def test_simple_mode_hides_advanced_churn_warning_copy(self) -> None:
        """Ensure simple mode hides advanced-only churn warning copy."""
        repo_root = Path(__file__).resolve().parents[2]
        setup_source = (repo_root / "wrapper" / "js" / "modules" / "setup.module.js").read_text(encoding="utf-8")

        self.assertIn("if (this._state.setupMode !== 'advanced') {", setup_source)
        self.assertIn("All clients start eligible", setup_source)

    def test_custom_dataset_upload_prefers_multipart_file_streaming(self) -> None:
        """Ensure custom dataset uploads use streamed multipart handling across the stack."""
        repo_root = Path(__file__).resolve().parents[2]
        api_source = (repo_root / "wrapper" / "js" / "api.js").read_text(encoding="utf-8")
        setup_source = (repo_root / "wrapper" / "js" / "modules" / "setup.module.js").read_text(encoding="utf-8")
        render_source = (repo_root / "wrapper" / "js" / "modules" / "render.module.js").read_text(encoding="utf-8")
        dashboard_source = (repo_root / "wrapper" / "templates" / "dashboard.html").read_text(encoding="utf-8")
        content_source = (repo_root / "web_backend" / "api_handlers" / "content.py").read_text(encoding="utf-8")
        datasets_service_source = (repo_root / "web_backend" / "services" / "datasets.py").read_text(encoding="utf-8")
        service_source = (repo_root / "web_backend" / "service.py").read_text(encoding="utf-8")
        compose_source = (repo_root / "docker-compose.yml").read_text(encoding="utf-8")
        nginx_source = (repo_root / "deploy" / "nginx" / "default.conf.template").read_text(encoding="utf-8")
        app_module = reload_module("web_backend.app")

        boundary = "boundary123"
        multipart_body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="description"\r\n\r\n'
            "uploaded from test\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="demo.csv"\r\n'
            "Content-Type: text/csv\r\n\r\n"
            "col1,col2\n1,2\n"
            f"\r\n--{boundary}--\r\n"
        ).encode("utf-8")

        class _FakeRequest:
            def __init__(self, body: bytes):
                self.headers = {
                    "content-type": f"multipart/form-data; boundary={boundary}",
                    "content-length": str(len(body)),
                }
                self._body = body

            async def stream(self):
                yield self._body

        tmp_request = _FakeRequest(multipart_body)
        tmp_path = ""
        upload_tmp_dir = ""
        try:
            original_name, description, tmp_path, total_bytes, upload_tmp_dir = asyncio.run(
                app_module._stream_custom_dataset_upload_form(tmp_request)
            )
            self.assertEqual("demo.csv", original_name)
            self.assertEqual("uploaded from test", description)
            self.assertEqual(len(multipart_body), total_bytes)
            self.assertTrue(Path(tmp_path).exists())
            self.assertIn(".custom_dataset_upload_", Path(upload_tmp_dir).name)
        finally:
            if tmp_path:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(tmp_path)
            if upload_tmp_dir:
                shutil.rmtree(upload_tmp_dir, ignore_errors=True)

        self.assertIn("uploadCustomDatasetFileWithProgress(file, options = {})", api_source)
        self.assertIn("new FormData()", api_source)
        self.assertIn("xhr.upload.onload = () => {", api_source)
        self.assertIn("Dataset upload is larger than the server limit.", api_source)
        self.assertNotIn("uploadCustomDatasetWithProgress(", api_source)
        self.assertNotIn("upload_custom_dataset", api_source)
        self.assertIn("await api.uploadCustomDatasetFileWithProgress(file, {", setup_source)
        self.assertIn("onUploadComplete: () => {", setup_source)
        self.assertIn("Uploading large ZIP archive to server...", setup_source)
        self.assertNotIn("_readFileAsTextWithProgress(", setup_source)
        self.assertNotIn("_readFileAsBase64WithProgress(", setup_source)
        self.assertIn(".csv,.tsv,.npz,.zip", render_source)
        self.assertIn(".csv,.tsv,.npz,.zip", dashboard_source)
        self.assertIn("application/zip", render_source)
        self.assertIn("application/zip", dashboard_source)
        self.assertNotIn("if command == \"upload_custom_dataset\":", content_source)
        self.assertIn("def save_custom_dataset_file(", datasets_service_source)
        self.assertIn("def _inspect_npz_dataset(", datasets_service_source)
        self.assertIn("def _inspect_image_dataset_archive(", datasets_service_source)
        self.assertIn("image.verify()", datasets_service_source)
        self.assertIn("shutil.move(source_path, out_path)", datasets_service_source)
        self.assertIn("tempfile.mkdtemp(prefix=\".image_folder_upload_\", dir=out_dir)", datasets_service_source)
        self.assertIn("os.replace(dataset_stage_root, out_path)", datasets_service_source)
        self.assertNotIn("archive.extractall(", datasets_service_source)
        self.assertIn("Large archives can take a while...", setup_source)
        self.assertNotIn("def save_custom_dataset(", datasets_service_source)
        self.assertNotIn("def save_custom_dataset(", service_source)
        self.assertIn("NGINX_CLIENT_MAX_BODY_SIZE", compose_source)
        self.assertIn("location = /api/upload-custom-dataset", nginx_source)
        self.assertIn("proxy_request_buffering off;", nginx_source)
        self.assertIn("client_max_body_size ${NGINX_CLIENT_MAX_BODY_SIZE};", nginx_source)

    def test_env_example_covers_required_faster_runtime_keys(self) -> None:
        """Ensure the example environment file documents the required runtime keys."""
        repo_root = Path(__file__).resolve().parents[2]
        env_source = (repo_root / ".env.example").read_text(encoding="utf-8")

        for key in (
            "HTTPS_PORT",
            "NGINX_SERVER_NAME",
            "TLS_CERT_FILENAME",
            "TLS_KEY_FILENAME",
            "NGINX_CLIENT_MAX_BODY_SIZE",
            "MARIADB_ROOT_PASSWORD",
            "MARIADB_DATABASE",
            "MARIADB_USER",
            "MARIADB_PASSWORD",
            "FASTER_DB_HOST",
            "FASTER_DB_PORT",
            "FASTER_DB_USER",
            "FASTER_DB_PASSWORD",
            "FASTER_DB_NAME",
            "FASTER_DB_CONNECT_TIMEOUT",
            "FASTER_DB_POOL_SIZE",
            "FL_API_BODY_MAX_BYTES",
            "FL_CUSTOM_DATASET_UPLOAD_MAX_BYTES",
            "FL_SECRET_KEY",
            "JM_DB_HOST",
            "JM_DB_PORT",
            "JM_DB_USER",
            "JM_DB_PASSWORD",
            "JM_DB_NAME",
            "JM_JWT_SECRET",
            "JM_API_BASE_URL",
            "JM_API_TIMEOUT",
            "JM_SERVICE_USERNAME",
            "JM_SERVICE_PASSWORD",
            "FL_JM_SOURCE_APP",
        ):
            self.assertRegex(env_source, rf"(?m)^{re.escape(key)}=")

        fl_secret = next(
            line.split("=", 1)[1]
            for line in env_source.splitlines()
            if line.startswith("FL_SECRET_KEY=")
        )
        self.assertGreaterEqual(
            len(fl_secret.encode("utf-8")),
            32,
            "FL_SECRET_KEY placeholder should meet HS256 minimum length",
        )

    def test_validate_run_config_rejects_missing_required_field(self) -> None:
        """Ensure validation rejects configurations missing required fields."""
        cfg = valid_run_config()
        cfg.pop("dataset_name")

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertIn("Missing required config keys", msg)

    def test_validate_run_config_rejects_non_mapping_payload(self) -> None:
        """Ensure validation rejects non-dict payloads before any field access."""
        ok, msg = self.api_handlers._validate_run_config(None)

        self.assertFalse(ok)
        self.assertEqual("Invalid config payload", msg)

    def test_validate_run_config_accepts_stringified_numeric_fields_and_trims_run_name(self) -> None:
        """Ensure numeric fields are coerced from strings and run names are normalized."""
        cfg = valid_run_config()
        cfg.update(
            {
                "num_clients": "3",
                "batch_size": "32",
                "local_model_epochs": "6",
                "weights_sending_frequency": "3",
                "learning_rate": "0.001",
                "server_learning_rate": "0.01",
                "run_name": "  Boundary Run \n",
            }
        )

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok)
        self.assertEqual("", msg)
        self.assertEqual("Boundary Run", cfg["run_name"])
        self.assertEqual(3, cfg["initial_eligible_clients"])
        self.assertEqual(0.01, cfg["server_learning_rate"])
        self.assertEqual(0, cfg["seed"])

    def test_validate_run_config_rejects_invalid_custom_dataset_reference(self) -> None:
        """Ensure validation rejects unsafe custom dataset references."""
        cfg = valid_run_config()
        cfg["dataset_name"] = "custom_csv"
        cfg["custom_dataset_ref"] = "../bad.csv"
        cfg["custom_dataset_name"] = "Dataset Demo"

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertIn("custom_dataset_ref", msg)

    def test_validate_run_config_rejects_custom_dataset_name_with_unicode(self) -> None:
        """Ensure custom dataset names stay within the documented ASCII label rule."""
        cfg = valid_run_config()
        cfg["dataset_name"] = "custom_csv"
        cfg["custom_dataset_ref"] = "dataset01.csv"
        cfg["custom_dataset_name"] = "Dataset 🚀"

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual("custom_dataset_name contains invalid characters", msg)

    def test_validate_run_config_normalizes_default_client_churn_values(self) -> None:
        """Ensure validation normalizes default client-churn values when omitted."""
        cfg = valid_run_config()

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok)
        self.assertEqual("", msg)
        self.assertEqual(cfg["num_clients"], cfg["initial_eligible_clients"])
        self.assertEqual(0.0, cfg["death_prob"])
        self.assertEqual(0.0, cfg["new_client_prob"])
        self.assertEqual(0, cfg["seed"])

    def test_validate_run_config_accepts_churn_probability_at_new_upper_bound(self) -> None:
        """Ensure validation accepts churn probabilities at the supported upper bound."""
        cfg = valid_run_config()
        cfg["death_prob"] = 0.9
        cfg["new_client_prob"] = 0.9

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok)
        self.assertEqual("", msg)

    def test_validate_run_config_requires_server_learning_rate(self) -> None:
        """Ensure server_learning_rate remains a required config field."""
        cfg = valid_run_config()
        cfg.pop("server_learning_rate")

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual("Missing required config keys: server_learning_rate", msg)

    def test_validate_run_config_rejects_invalid_server_learning_rate(self) -> None:
        """Ensure invalid server learning rates are rejected with a stable message."""
        cfg = valid_run_config()
        cfg["server_learning_rate"] = 0

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual("server_learning_rate must be > 0 and <= 10", msg)

    def test_validate_run_config_rejects_initial_eligible_above_pool(self) -> None:
        """Ensure validation rejects initial eligible counts above the client pool."""
        cfg = valid_run_config()
        cfg["initial_eligible_clients"] = cfg["num_clients"] + 1

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual("initial_eligible_clients must be in [1, num_clients]", msg)

    def test_validate_run_config_rejects_initial_eligible_below_one(self) -> None:
        """Ensure validation rejects initial eligible counts below one."""
        cfg = valid_run_config()
        cfg["initial_eligible_clients"] = 0

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual("initial_eligible_clients must be in [1, num_clients]", msg)

    def test_validate_run_config_rejects_invalid_client_churn_probability(self) -> None:
        """Ensure validation rejects out-of-range churn probabilities."""
        cfg = valid_run_config()
        cfg["death_prob"] = 1.2

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual("death_prob must be in [0.0, 0.9]", msg)

    def test_validate_run_config_rejects_negative_seed(self) -> None:
        """Ensure validation rejects negative random seeds."""
        cfg = valid_run_config()
        cfg["seed"] = -1

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual("seed must be >= 0", msg)

    def test_validate_run_config_accepts_run_name_up_to_maximum_length(self) -> None:
        """Ensure validation accepts run names at the maximum length."""
        cfg = valid_run_config()
        cfg["run_name"] = "a" * 20

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok)
        self.assertEqual("", msg)

    def test_validate_run_config_rejects_run_name_above_maximum_length(self) -> None:
        """Ensure validation rejects run names above the maximum length."""
        cfg = valid_run_config()
        cfg["run_name"] = "a" * 21

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual("run_name must be at most 20 characters", msg)

    def test_validate_run_config_rejects_blocked_custom_model_import(self) -> None:
        """Ensure blocked imports in custom model code are rejected during validation."""
        cfg = valid_run_config()
        cfg["custom_model_mode"] = "code"
        cfg["custom_model_code"] = (
            "import os\n"
            "MODEL_NAME = 'Unsafe'\n"
            "def build_model(n_channels, n_classes):\n"
            "    return None\n"
        )

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertIn("Blocked import", msg)

    def test_fl_api_handler_rejects_unauthorized_protected_command(self) -> None:
        """Ensure protected commands return Unauthorized when token claims fail."""
        request_payload = json.dumps(
            {"command": "me", "token": "bad-token", "data": {}}
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=(False, {}),
        ):
            response = self.api_handlers.fl_api_handler(request_payload)

        self.assertEqual({"error": "Unauthorized"}, response)

    def test_fl_api_handler_returns_submission_error_when_job_manager_submit_fails(self) -> None:
        """Ensure API submission errors surface the Job Manager failure payload."""
        request_payload = json.dumps(
            {"command": "start_run", "token": "good-token", "data": {"config": valid_run_config()}}
        )

        with mock.patch.object(
            self.api_handlers._auth,
            "verify_token_claims",
            return_value=(True, {"identifier": "alice", "role": "user", "email": "alice@example.com"}),
        ), mock.patch.object(
            self.api_handlers,
            "_submit_run_to_job_manager",
            return_value=(False, {"error": "JM unavailable"}),
        ):
            response = self.api_handlers.fl_api_handler(request_payload)

        self.assertEqual({"error": "JM unavailable"}, response)
