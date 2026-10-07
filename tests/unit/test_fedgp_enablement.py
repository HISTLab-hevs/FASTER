"""Tests for FedGP enablement across custom tabular datasets."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import reload_module, valid_run_config


def _fedgp_config() -> dict:
    cfg = valid_run_config()
    cfg.update(
        {
            "method": "fed_gp",
            "individuals": 2,
            "generations": 1,
            "mutation_rate": 0.0,
            "crossover_rate": 1.0,
            "elitism_size": 1,
            "gp_patience": 1,
            "max_tree_size": 1,
            "min_tree_size": 1,
            "gp_initialization": "genGrow",
            "gp_transfer_learning": "disabled",
            "mutation_type": "mutUniform",
            "selection_type": "selTournament",
            "crossover_type": "cxOnePoint",
            "available_primitives": "torch.add,torch.sub,torch.mul,torch_mean",
            "server_warmup_epochs": 1,
            "local_model_epochs": 1,
            "weights_sending_frequency": 1,
            "momentum": 0.0,
            "server_data_percentage": 0.25,
            "iid": True,
            "imbalance_rate": 0.5,
            "train_val_split": 0.9,
        }
    )
    return cfg


def _write_tabular_csv(root: Path) -> Path:
    rows = ["f1,f2,label,split"]
    for split, count in (("train", 24), ("val", 8), ("test", 8)):
        for idx in range(count):
            label = idx % 2
            rows.append(f"{float(idx % 6)},{float(label) + 0.1 * (idx % 3)},{label},{split}")
    csv_path = root / "fedgp_tabular.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv_path


class FedGpEnablementTests(unittest.TestCase):
    """Cover FedGP support for custom tabular datasets across the stack."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api_handlers = reload_module("web_backend.api_handlers.api")

    def test_backend_validation_accepts_fedgp_custom_csv_default_tabular_model(self) -> None:
        """Ensure backend validation accepts FedGP on custom CSV with the default model."""
        cfg = _fedgp_config()
        cfg.update(
            {
                "dataset_name": "custom_csv",
                "custom_dataset_ref": "fedgp1.csv",
                "custom_dataset_name": "FedGP CSV",
            }
        )

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok, msg)

    def test_backend_validation_checks_mutation_subtree_maxsize(self) -> None:
        """mutation_subtree_maxsize must be an integer in [1, 64] for FedGP."""
        for bad in (0, 65):
            cfg = _fedgp_config()
            cfg["mutation_subtree_maxsize"] = bad
            ok, msg = self.api_handlers._validate_run_config(cfg)
            self.assertFalse(ok, f"mutation_subtree_maxsize={bad} should be rejected")
            self.assertIn("mutation_subtree_maxsize", msg)

        # A valid value passes (decoupled from max_tree_size, which is 1 in this fixture).
        cfg = _fedgp_config()
        cfg["mutation_subtree_maxsize"] = 3
        ok, msg = self.api_handlers._validate_run_config(cfg)
        self.assertTrue(ok, msg)

    def test_batched_eval_metrics_match_sequential_sklearn(self) -> None:
        """The vectorized confusion-matrix metrics (batched path) equal the sklearn ones."""
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        gp = reload_module("utils.gp")

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 3)

            def forward(self, x):  # noqa: D401 - tiny test model
                return self.fc(x)

        torch.manual_seed(0)
        model = _Net()
        sd = model.state_dict()
        clients = [{k: v.clone() for k, v in sd.items()} for _ in range(2)]
        features = torch.randn(60, 4)
        labels = torch.randint(0, 3, (60,))
        loader = DataLoader(TensorDataset(features, labels), batch_size=20)

        toolbox, _ = gp.initialize_deap(
            2, "mutUniform", "selTournament", 1, 3,
            ["torch.add", "torch.mul", "torch_mean"], "cxOnePoint", "genHalfAndHalf",
        )
        toolbox.register("evaluate", gp.eval_individual)
        population = toolbox.population(n=4)

        for metric in ("precision", "recall", "f1"):
            gp._FITNESS_METRIC = metric
            batched = gp.batched_eval(population, clients, loader, toolbox, model, "cpu")
            sequential = [
                gp.eval_individual(ind, clients, loader, toolbox, model, "cpu")
                for ind in population
            ]
            for b, s in zip(batched, sequential):
                self.assertAlmostEqual(b, s, places=5, msg=f"{metric} mismatch")
        gp._FITNESS_METRIC = "accuracy"

    def test_transfer_learning_carries_elitism_size_elites(self) -> None:
        """Elite/hybrid warm-starts carry the top `elitism_size` individuals, not just one."""
        from deap import tools

        gp = reload_module("utils.gp")
        from utils.gp_population import (
            init_elite_random,
            init_elite_mutation,
            init_hybrid_population,
        )

        toolbox, _ = gp.initialize_deap(
            3, "mutUniform", "selTournament", 1, 3,
            ["torch.add", "torch.mul", "torch_mean"], "cxOnePoint", "genHalfAndHalf",
        )
        population = toolbox.population(n=10)
        for i, ind in enumerate(population):
            ind.fitness.values = (float(i),)  # fitness 0..9

        elitism_size, pop_size = 3, 10
        elites = tools.selBest(population, elitism_size)
        self.assertEqual([9.0, 8.0, 7.0], [e.fitness.values[0] for e in elites])

        for init_fn in (init_elite_random, init_elite_mutation, init_hybrid_population):
            new_pop = init_fn(elites, pop_size, toolbox)
            self.assertEqual(pop_size, len(new_pop), init_fn.__name__)
            # The first `elitism_size` individuals are clones of the carried elites.
            for k in range(elitism_size):
                self.assertEqual(str(elites[k]), str(new_pop[k]), init_fn.__name__)

    def test_backend_validation_checks_gp_fitness_metric(self) -> None:
        """gp_fitness_metric must be 'accuracy' or 'loss' for FedGP."""
        cfg = _fedgp_config()
        cfg["gp_fitness_metric"] = "f1_does_not_exist"
        ok, msg = self.api_handlers._validate_run_config(cfg)
        self.assertFalse(ok)
        self.assertIn("gp_fitness_metric", msg)

        for good in ("accuracy", "loss", "precision", "recall", "f1"):
            cfg = _fedgp_config()
            cfg["gp_fitness_metric"] = good
            ok, msg = self.api_handlers._validate_run_config(cfg)
            self.assertTrue(ok, msg)

    def test_backend_validation_rejects_selroulette_with_loss(self) -> None:
        """selRoulette + loss is rejected (roulette needs non-negative fitness)."""
        cfg = _fedgp_config()
        cfg["selection_type"] = "selRoulette"
        cfg["gp_fitness_metric"] = "loss"
        ok, msg = self.api_handlers._validate_run_config(cfg)
        self.assertFalse(ok)
        self.assertIn("selRoulette", msg)

        # selRoulette is fine with a non-negative metric...
        cfg = _fedgp_config()
        cfg["selection_type"] = "selRoulette"
        cfg["gp_fitness_metric"] = "f1"
        ok, msg = self.api_handlers._validate_run_config(cfg)
        self.assertTrue(ok, msg)

        # ...and loss is fine with a non-roulette selection.
        cfg = _fedgp_config()
        cfg["selection_type"] = "selTournament"
        cfg["gp_fitness_metric"] = "loss"
        ok, msg = self.api_handlers._validate_run_config(cfg)
        self.assertTrue(ok, msg)

    def test_backend_validation_requires_build_model_for_fedgp_custom_model(self) -> None:
        """Ensure custom FedGP model code still has to define build_model()."""
        cfg = _fedgp_config()
        cfg["custom_model_mode"] = "code"
        cfg["custom_model_code"] = (
            "import torch\n"
            "import torch.nn as nn\n"
            "MODEL_NAME = 'InlineTabular'\n"
            "class InlineTabular(nn.Module):\n"
            "    def __init__(self, n_channels, n_classes):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(n_channels, n_classes)\n"
            "    def forward(self, x):\n"
            "        return self.fc(x.view(x.size(0), -1))\n"
        )

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual("custom_model_code must define build_model(n_channels, n_classes)", msg)

    def test_frontend_and_docs_do_not_block_fedgp_custom_csv(self) -> None:
        """Ensure the UI and docs do not advertise a false FedGP tabular restriction."""
        repo_root = Path(__file__).resolve().parents[2]
        setup_source = (repo_root / "wrapper" / "js" / "modules" / "setup.module.js").read_text(encoding="utf-8")
        docs_source = (repo_root / "wrapper" / "templates" / "docs.html").read_text(encoding="utf-8")
        dashboard_source = (repo_root / "wrapper" / "templates" / "dashboard.html").read_text(encoding="utf-8")

        self.assertNotIn("FedGP method is not supported for tabular", setup_source)
        self.assertIn("Define build_model(n_channels, n_classes).", dashboard_source)
        self.assertIn("define <code>build_model(n_channels, n_classes)</code>", docs_source)
        self.assertNotIn("CustomModel(nn.Module)", docs_source)

    def test_runtime_runs_fedgp_custom_csv_with_default_tabular_model(self) -> None:
        """Ensure the runtime can execute a minimal FedGP tabular run with the default model."""
        main_module = reload_module("main")
        with tempfile.TemporaryDirectory(prefix="fedgp_runtime_") as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_tabular_csv(tmp_path)
            save_path = tmp_path / "out"
            cfg = _fedgp_config()
            cfg.update(
                {
                    "dataset_name": "custom_csv",
                    "custom_dataset_path": str(csv_path),
                    "batch_size": 4,
                    "num_clients": 2,
                    "initial_eligible_clients": 2,
                }
            )

            metrics = main_module.run_experiment(cfg, str(save_path))

        self.assertEqual(1, len(metrics["aggregated_val"]["accuracy"]))
        self.assertEqual(1, len(metrics["aggregated_test"]["accuracy"]))

    def test_runtime_runs_fedgp_custom_csv_with_inline_custom_model(self) -> None:
        """Ensure the runtime can execute a minimal FedGP tabular run with inline custom code."""
        main_module = reload_module("main")
        with tempfile.TemporaryDirectory(prefix="fedgp_runtime_custom_") as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_tabular_csv(tmp_path)
            save_path = tmp_path / "out"
            cfg = _fedgp_config()
            cfg.update(
                {
                    "dataset_name": "custom_csv",
                    "custom_dataset_path": str(csv_path),
                    "batch_size": 4,
                    "num_clients": 2,
                    "initial_eligible_clients": 2,
                    "custom_model_mode": "code",
                    "custom_model_code": (
                        "import torch\n"
                        "import torch.nn as nn\n"
                        "MODEL_NAME = 'FedGpInlineTabular'\n"
                        "class InlineTabular(nn.Module):\n"
                        "    def __init__(self, n_channels, n_classes):\n"
                        "        super().__init__()\n"
                        "        self.net = nn.Linear(n_channels, n_classes)\n"
                        "    def forward(self, x):\n"
                        "        return self.net(x.view(x.size(0), -1))\n"
                        "def build_model(n_channels, n_classes):\n"
                        "    return InlineTabular(n_channels, n_classes)\n"
                    ),
                }
            )

            metrics = main_module.run_experiment(cfg, str(save_path))

        self.assertEqual(1, len(metrics["aggregated_val"]["accuracy"]))
        self.assertEqual(1, len(metrics["aggregated_test"]["accuracy"]))

    def test_backend_validation_normalizes_list_primitives_and_boolean_transfer_learning_aliases(self) -> None:
        """Ensure FedGP validation accepts list primitives and normalizes boolean aliases."""
        cfg = _fedgp_config()
        cfg.update(
            {
                "available_primitives": [" torch.add ", "torch.sub", "torch.mul"],
                "gp_transfer_learning": True,
                "min_tree_size": False,
                "max_tree_size": 2,
                "individuals": 3,
                "generations": 2,
                "elitism_size": 1,
                "gp_patience": 1,
                "mutation_rate": 0.25,
                "crossover_rate": 0.75,
            }
        )

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertTrue(ok, msg)
        self.assertEqual("torch.add,torch.sub,torch.mul", cfg["available_primitives"])
        self.assertEqual("full_reuse", cfg["gp_transfer_learning"])
        self.assertEqual("fed_gp", cfg["method"])
        self.assertEqual("pathmnist", cfg["dataset_name"])

    def test_backend_validation_rejects_fedgp_zero_mutation_and_crossover_rates(self) -> None:
        """Ensure FedGP rejects configurations that disable both mutation and crossover."""
        cfg = _fedgp_config()
        cfg.update(
            {
                "mutation_rate": 0.0,
                "crossover_rate": 0.0,
                "individuals": 3,
                "generations": 2,
                "elitism_size": 1,
                "gp_patience": 1,
                "max_tree_size": 2,
            }
        )

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual(
            "At least one of mutation_rate or crossover_rate must be > 0 for fed_gp",
            msg,
        )

    def test_backend_validation_rejects_fedgp_min_tree_size_above_max_tree_size(self) -> None:
        """Ensure FedGP rejects tree-size bounds that contradict each other."""
        cfg = _fedgp_config()
        cfg.update(
            {
                "min_tree_size": 3,
                "max_tree_size": 2,
                "individuals": 3,
                "generations": 2,
                "elitism_size": 1,
                "gp_patience": 1,
                "mutation_rate": 0.25,
                "crossover_rate": 0.75,
            }
        )

        ok, msg = self.api_handlers._validate_run_config(cfg)

        self.assertFalse(ok)
        self.assertEqual(
            "min_tree_size must be >= 1 and <= max_tree_size for fed_gp",
            msg,
        )


if __name__ == "__main__":
    unittest.main()
