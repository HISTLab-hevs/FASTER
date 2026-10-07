"""Federated-learning training worker launched as a subprocess per run.

The Job Manager starts this module with ``--config`` and ``--save_path``. It
validates the submitted configuration, builds the dataset partitions and model,
runs the warmup and aggregation rounds for the selected method, and writes
metrics, logs, and run metadata into the run directory.
"""

import copy
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
import argparse
import ast
import numpy as np
from importlib import metadata as importlib_metadata

import torch
from torch import optim
from utils.aggregation_methods import fed_avg, fed_avgw, fed_gp, fed_prox, fed_nova
from utils.config import load_config, dump_config, normalize_run_config_aliases
from utils.dataset import Dataset
from utils.evaluate import evaluate
from utils.gpu_check import gpu_check
from utils.metrics import print_metrics, store_metrics
from utils.model import Net, TabularNet
from utils.run_config_validation import validate_core_run_config
from utils.seed import set_global_seed
from utils.round_schedule import (
    normalize_round_client_allocation_config,
    normalize_round_train_schedule_config,
)
from utils.train import train


def _safe_package_version(package_name):
    """Return an installed package version when available, otherwise ``None``."""
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def _safe_git_metadata(base_path):
    """Collect git commit and dirty-state metadata without raising on failures."""
    git_meta = {
        "commit_hash": None,
        "is_dirty": None,
    }
    try:
        commit_proc = subprocess.run(
            ["git", "-C", base_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if commit_proc.returncode == 0:
            commit = commit_proc.stdout.strip()
            if commit:
                git_meta["commit_hash"] = commit

        status_proc = subprocess.run(
            ["git", "-C", base_path, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if status_proc.returncode == 0:
            git_meta["is_dirty"] = bool(status_proc.stdout.strip())
    except Exception:
        pass
    return git_meta


def _build_run_metadata(config):
    """Build provenance metadata for a run artifact directory."""
    dependency_versions = {
        "numpy": getattr(np, "__version__", None),
        "torch": getattr(torch, "__version__", None),
        "scikit-learn": _safe_package_version("scikit-learn"),
        "pandas": _safe_package_version("pandas"),
        "matplotlib": _safe_package_version("matplotlib"),
    }
    dependency_versions["sklearn"] = dependency_versions["scikit-learn"]

    metadata = {
        "resolved_seed": int(config.get("seed", 0) or 0),
        "python_version": sys.version,
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
        "dependency_versions": dependency_versions,
    }
    metadata["git"] = _safe_git_metadata(os.path.dirname(__file__))
    return metadata


def _write_run_metadata(save_path, config):
    """Persist run provenance metadata as JSON in the run artifact directory."""
    metadata_path = os.path.join(save_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(_build_run_metadata(config), fp, indent=2)


def parse_args():
    """Parse command-line arguments for a federated learning run.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Federated Learning")

    def _parse_bool_arg(value):
        """Parse a human-friendly boolean CLI value."""
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")

    def _parse_int_or_false_arg(value):
        """Parse a FedGP tree-size CLI value that can also disable the lower bound."""
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"false", "none", "off"}:
            return False
        try:
            return int(value)
        except Exception as exc:
            raise argparse.ArgumentTypeError(
                f"Expected an integer or false, got {value!r}"
            ) from exc

    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--method",
        type=str,
        choices=["fed_avg", "fed_avgw", "fed_gp", "fed_prox", "fed_nova"],
        help="Federated learning aggregator method to use",
    )
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use")
    parser.add_argument("--num_clients", type=int, help="Number of clients")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument(
        "--server_data_percentage",
        type=float,
        help="Percentage of data to be used by the server",
    )
    parser.add_argument("--iid", type=_parse_bool_arg, help="Whether the dataset is IID or not")
    parser.add_argument(
        "--imbalance_rate",
        type=float,
        help="Degree of data imbalance among clients (0 = balanced, 1 = highly imbalanced)",
    )
    parser.add_argument(
        "--server_warmup_epochs",
        type=int,
        help="Number of epochs for server warm-up training",
    )
    parser.add_argument(
        "--global_model_epochs",
        dest="server_warmup_epochs",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local_model_epochs",
        type=int,
        help="Number of epochs for local model training",
    )
    parser.add_argument(
        "--weights_sending_frequency",
        type=int,
        help="Frequency of sending weights to the server",
    )
    parser.add_argument("--momentum", type=float, help="Momentum for the optimizer")
    parser.add_argument("--server_learning_rate", type=float, help="Server warm-up learning rate")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument(
        "--client_learning_rates",
        type=float,
        nargs="+",
        help="List of learning rates for each client",
    )
    parser.add_argument("--mu", type=float, help="Regularization parameter for FedProx")
    parser.add_argument(
        "--fedprox_weighted",
        type=_parse_bool_arg,
        help="FedProx: sample-weighted aggregation (True) or uniform averaging (False)",
    )
    parser.add_argument("--save_path", type=str, help="Path to save the results")
    parser.add_argument(
        "--individuals", type=int, help="Number of individuals for genetic programming"
    )
    parser.add_argument(
        "--generations", type=int, help="Number of generations for genetic programming"
    )
    parser.add_argument(
        "--elitism_size", type=int, help="Size of the elitism in genetic programming"
    )
    parser.add_argument(
        "--mutation_rate", type=float, help="Mutation rate for genetic programming"
    )
    parser.add_argument(
        "--crossover_rate", type=float, help="Crossover rate for genetic programming"
    )
    parser.add_argument(
        "--gp_patience",
        type=int,
        help="Patience for early stopping in genetic programming",
    )
    parser.add_argument(
        "--gp_initialization",
        type=str,
        choices=["genHalfAndHalf", "genFull", "genGrow"],
        help="Initialization method for genetic programming",
    )
    parser.add_argument(
        "--mutation_type",
        type=str,
        choices=["mutUniform", "mutNodeReplacement", "mutInsert"],
        help="Type of mutation for genetic programming",
    )
    parser.add_argument(
        "--selection_type",
        type=str,
        choices=["selTournament", "selRoulette", "selRandom"],
        help="Selection method for genetic programming",
    )
    parser.add_argument(
        "--crossover_type",
        type=str,
        choices=["cxOnePoint", "cxOnePointLeafBiased"],
        help="Crossover method for genetic programming",
    )
    parser.add_argument(
        "--min_tree_size",
        type=_parse_int_or_false_arg,
        help="Minimum tree size for genetic programming, int or False (with False, the min tree size is 1)",
    )
    parser.add_argument(
        "--max_tree_size",
        type=int,
        help="Maximum tree size for genetic programming, int or False (with False, 100)",
    )
    parser.add_argument(
        "--mutation_subtree_maxsize",
        type=int,
        help="Max height of the random subtree grafted on mutUniform mutation (default 2)",
    )
    parser.add_argument(
        "--gp_fitness_metric",
        type=str,
        choices=["accuracy", "loss", "precision", "recall", "f1"],
        help="Metric driving the GP fitness (accuracy/precision/recall/f1 maximized, loss minimized)",
    )
    parser.add_argument(
        "--repeat", type=int, help="How many times to repeat the experiment"
    )
    parser.add_argument(
        "--evaluation_split_mode",
        type=str,
        help="Evaluation mode: train_val_test or train_val",
    )
    parser.add_argument(
        "--custom_model_mode",
        type=str,
        help="Model source: default, code, or upload_file",
    )
    parser.add_argument(
        "--custom_model_code",
        type=str,
        help="Python code containing build_model(n_channels, n_classes)",
    )
    parser.add_argument(
        "--display_metrics", action="store_true", help="Display metrics after each run"
    )

    return parser.parse_args()


class Tee:
    """Write the same output to multiple streams."""

    def __init__(self, *streams):
        """Store the target streams.

        Args:
            *streams: Stream objects such as ``sys.stdout`` or open files.
        """
        self.streams = streams

    def write(self, data):
        """Write data to every configured stream."""
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        """Flush every configured stream."""
        for s in self.streams:
            s.flush()


def _resolve_model_factory(config, n_channels, n_classes):
    """Return a zero-argument factory that builds the configured model."""
    mode = str(config.get("custom_model_mode", "default") or "default")
    dataset_name = str(config.get("dataset_name", "") or "")
    if mode in {"loaded_file", "upload_file"}:
        mode = "code"
    if mode == "default":
        if dataset_name == "custom_csv":
            return lambda: TabularNet(int(n_channels), n_classes)
        return lambda: Net(n_channels, n_classes)

    custom_code = str(config.get("custom_model_code", "") or "").strip()
    if not custom_code:
        raise ValueError("custom_model_code is empty while custom_model_mode is not 'default'")

    exec_namespace = {
        "__builtins__": __builtins__,
        "torch": torch,
        "nn": torch.nn,
        "np": np,
    }
    try:
        # Use one shared namespace so helper symbols resolve during execution.
        exec(custom_code, exec_namespace, exec_namespace)
    except Exception as exc:
        raise RuntimeError(
            f"Custom model code execution failed: {exc}. "
            "Check syntax and available libraries in documentation."
        )

    build_fn = exec_namespace.get("build_model")
    if callable(build_fn):
        def _factory():
            model = build_fn(n_channels, n_classes)
            if not isinstance(model, torch.nn.Module):
                raise TypeError("build_model must return a torch.nn.Module instance")
            return model
        return _factory

    raise ValueError("Custom model code must define build_model(n_channels, n_classes).")


def _extract_custom_model_name(custom_code: str):
    """Extract ``MODEL_NAME`` from user-provided model code when available."""
    try:
        tree = ast.parse(custom_code)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_NAME":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        name = node.value.value.strip()
                        return name if name else None
    return None


def _normalize_gp_transfer_learning(value):
    """Normalize GP transfer-learning options to the internal representation."""
    if value is True:
        return "full_reuse"
    if value is False or value is None:
        return False
    txt = str(value).strip()
    if not txt:
        return False
    lower = txt.lower()
    if lower in {"false", "none", "disabled", "off"}:
        return False
    if lower == "true":
        return "full_reuse"
    return txt


def _loader_num_samples(loader):
    """Return sample count for a dataloader-like object, or 0 when unavailable."""
    if loader is None:
        return 0
    try:
        return int(len(loader.dataset))
    except Exception:
        try:
            return int(len(loader))
        except Exception:
            return 0


def _normalize_client_churn_config(config):
    """Return one normalized client-churn configuration snapshot."""
    total_clients = int(config.get("num_clients", 0))
    initial_eligible = config.get("initial_eligible_clients", total_clients)
    if initial_eligible in (None, ""):
        initial_eligible = total_clients

    return {
        "total_clients": total_clients,
        "initial_eligible_clients": int(initial_eligible),
        "death_prob": float(config.get("death_prob", 0) or 0),
        "new_client_prob": float(config.get("new_client_prob", 0) or 0),
        "seed": int(config.get("seed", 0) or 0),
    }


def _normalize_optional_float(value, default):
    """Return a float default only when a config value is missing."""
    if value in (None, ""):
        return float(default)
    return float(value)


def _normalize_server_data_percentage(value):
    """Preserve the ``False`` sentinel while still filling in a default."""
    if value in (None, ""):
        return 0.1
    if value is False:
        return False
    return float(value)


def _initial_eligible_clients(total_clients, initial_eligible_clients):
    """Return the deterministic initial eligible client set."""
    total = max(0, int(total_clients))
    minimum = 1 if total > 0 else 0
    normalized = max(minimum, min(total, int(initial_eligible_clients)))
    return set(range(normalized))


def _apply_client_churn_round(eligible_clients, total_clients, death_prob, new_client_prob, rng):
    """Apply one simultaneous eligibility churn step and return round metadata."""
    total = int(total_clients)
    eligible_set = set(int(client_id) for client_id in eligible_clients)
    ordered_eligible = sorted(eligible_set)
    ordered_inactive = [
        client_id for client_id in range(total)
        if client_id not in eligible_set
    ]

    dropped_clients = [
        client_id for client_id in ordered_eligible
        if float(death_prob) > 0 and float(rng.random()) < float(death_prob)
    ]
    newly_eligible_clients = [
        client_id for client_id in ordered_inactive
        if float(new_client_prob) > 0 and float(rng.random()) < float(new_client_prob)
    ]

    updated_eligible = (eligible_set - set(dropped_clients)) | set(newly_eligible_clients)
    safeguard_client = None
    safeguard_applied = False
    if total > 0 and not updated_eligible:
        safeguard_applied = True
        if ordered_eligible:
            safeguard_client = ordered_eligible[0]
            dropped_clients = [client_id for client_id in dropped_clients if client_id != safeguard_client]
        else:
            safeguard_client = 0
            newly_eligible_clients = [safeguard_client]
        updated_eligible = {safeguard_client}
    sampled_clients = sorted(updated_eligible)

    round_info = {
        "eligible_clients": sampled_clients,
        "sampled_clients": sampled_clients,
        "eligible_client_count": len(sampled_clients),
        "sampled_client_count": len(sampled_clients),
        "dropped_to_inactive_clients": dropped_clients,
        "dropped_to_inactive_count": len(dropped_clients),
        "newly_eligible_clients": newly_eligible_clients,
        "newly_eligible_count": len(newly_eligible_clients),
        "eligibility_safeguard_applied": safeguard_applied,
        "eligibility_safeguard_client": safeguard_client,
    }
    return updated_eligible, round_info


def run_experiment(config, save_path):
    """Run one federated learning experiment.

    Args:
        config: Configuration dictionary containing experiment parameters.
        save_path: Directory used to store artifacts and logs.

    Returns:
        dict: Metrics collected during the experiment.
    """
    start_time = time.time()

    normalize_run_config_aliases(config)
    config["evaluation_split_mode"] = str(config.get("evaluation_split_mode") or "train_val_test")
    config["seed"] = int(config.get("seed", 0) or 0)
    
    # Seed all RNG sources globally to ensure reproducibility
    set_global_seed(config["seed"])
    
    config["iid"] = bool(config.get("iid", False))
    config["server_data_percentage"] = _normalize_server_data_percentage(
        config.get("server_data_percentage", 0.1)
    )
    config["train_val_split"] = _normalize_optional_float(
        config.get("train_val_split", 0.9),
        0.9,
    )
    config["imbalance_rate"] = _normalize_optional_float(
        config.get("imbalance_rate", 0.5),
        0.5,
    )

    print(f"Method: {config.get('method')}")
    print(f"Dataset: {config.get('dataset_name')}")
    print(f"Evaluation split mode: {config.get('evaluation_split_mode', 'train_val_test')}")

    aggregation_rounds = int(
        config.get("local_model_epochs") / config.get("weights_sending_frequency")
    )
    resolved_round_train_percentages = normalize_round_train_schedule_config(
        config,
        aggregation_rounds=aggregation_rounds,
    )

    device = gpu_check()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    custom_mode = str(config.get("custom_model_mode", "default") or "default")
    if custom_mode in {"loaded_file", "upload_file"}:
        custom_mode = "code"
    if custom_mode not in {"default", "code"}:
        custom_mode = "default"
    config["custom_model_mode"] = custom_mode
    if custom_mode == "default":
        config["custom_model_code"] = ""
        config["custom_model_name"] = "DefaultNet"
    else:
        config["custom_model_code"] = str(config.get("custom_model_code", "") or "")
        model_name = _extract_custom_model_name(config["custom_model_code"])
        if not model_name:
            raise ValueError("custom_model_code must define MODEL_NAME as a non-empty string")
        config["custom_model_name"] = model_name

    config["gp_transfer_learning"] = _normalize_gp_transfer_learning(
        config.get("gp_transfer_learning", "full_reuse")
    )

    dataset = Dataset(
        config.get("dataset_name"),
        config.get("num_clients"),
        config.get("server_data_percentage"),
        config.get("batch_size"),
        config.get("iid"),
        config.get("imbalance_rate", 0.5),
        seed=config.get("seed", 0),
        custom_dataset_path=config.get("custom_dataset_path"),
    )

    dataset.print_class_distribution("train", save_path)
    dataset.plot_class_distribution("train", save_path)

    n_channels = dataset.n_channels
    n_classes = dataset.n_classes

    server_train_loader = dataset.client_datasets_train[0]
    # The whole validation set evaluates the global model.
    server_val_loader = dataset.val_data
    test_loader = dataset.test_data
    eval_mode = config.get("evaluation_split_mode", "train_val_test")
    auto_client_percentages = dataset.federated_client_train_percentages()
    resolved_round_client_percentages = normalize_round_client_allocation_config(
        config,
        aggregation_rounds=aggregation_rounds,
        num_clients=config.get("num_clients"),
        auto_client_percentages=auto_client_percentages,
    )
    round_train_subsets = dataset.build_round_train_subsets(
        resolved_round_train_percentages,
        seed=int(config.get("seed", 0) or 0),
    )
    dump_config(os.path.join(save_path, "config.yaml"), config)
    _write_run_metadata(save_path, config)

    # FedGP evaluates on validation by default and only falls back to test if needed.
    # Passed below as the `test_data`/`test_dataset` args of gp.eval_individual/run_evolution,
    # whose parameter names are misleading: they receive this validation loader, not test_loader.
    gp_eval_loader = server_val_loader
    if _loader_num_samples(gp_eval_loader) <= 0:
        if _loader_num_samples(test_loader) > 0:
            print("Warning: validation loader is empty; FedGP fitness falls back to test loader.")
            gp_eval_loader = test_loader
        else:
            raise ValueError("FedGP requires a non-empty evaluation loader (validation or test fallback).")
    if config.get("custom_model_mode") == "code":
        print(
            f"Custom model mode: code (name={config.get('custom_model_name')}, "
            f"len={len(config.get('custom_model_code', ''))} chars)"
        )
    else:
        print("Custom model mode: default")

    model_factory = _resolve_model_factory(config, n_channels, n_classes)

    server_model = model_factory()

    server_optimizer = optim.SGD(
        server_model.parameters(),
        lr=config.get("server_learning_rate"),
        momentum=config.get("momentum"),
    )

    metrics = {
        "clients": [
            {
                "loss": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "roc_auc": [],
            }
            for _ in range(config.get("num_clients"))
        ],
        "aggregated_val": {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []},
        "aggregated_test": {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []},
        "rounds": [],
        "live_state": {
            "phase": "server_warmup",
            "current_round": 0,
            "total_rounds": aggregation_rounds,
            "completed_rounds": 0,
            "current_repeat": int(config.get("_current_repeat", 1) or 1),
            "total_repeats": int(config.get("_total_repeats", 1) or 1),
            "configured_total_clients": int(config.get("num_clients", 0) or 0),
            "participating_clients_in_round": 0,
            "active_client_position": 0,
            "active_client_id": None,
            "client_epoch_completed": 0,
            "client_epoch_total": int(config.get("weights_sending_frequency", 0) or 0),
        },
    }

    def _update_live_state(**updates):
        state = metrics.setdefault("live_state", {})
        state.update(updates)

    # Write an initial runtime snapshot so the monitor can render structured progress immediately.
    store_metrics(os.path.join(save_path, "metrics.json"), metrics)

    print("Server warm-up training")
    train(
        server_model,
        server_optimizer,
        server_train_loader,
        val_loader=server_val_loader,
        epochs=config.get("server_warmup_epochs"),
        device=device,
        eval_split_name="Validation",
    )
    _update_live_state(phase="client_rounds_pending")
    store_metrics(os.path.join(save_path, "metrics.json"), metrics)

    client_models = []
    for client in range(config.get("num_clients")):
        client_models.append(model_factory())

    population = None
    best_individual = None
    churn_cfg = _normalize_client_churn_config(config)
    churn_rng = np.random.default_rng(churn_cfg["seed"])
    eligible_clients = _initial_eligible_clients(
        churn_cfg["total_clients"],
        churn_cfg["initial_eligible_clients"],
    )

    print(
        "Client churn config: "
        f"total_clients={churn_cfg['total_clients']}, "
        f"initial_eligible_clients={churn_cfg['initial_eligible_clients']}, "
        f"death_prob={churn_cfg['death_prob']}, "
        f"new_client_prob={churn_cfg['new_client_prob']}, "
        f"seed={churn_cfg['seed']}"
    )

    for aggregation_round in range(aggregation_rounds):
        eligible_clients, round_info = _apply_client_churn_round(
            eligible_clients,
            churn_cfg["total_clients"],
            churn_cfg["death_prob"],
            churn_cfg["new_client_prob"],
            churn_rng,
        )
        sampled_clients = list(round_info["sampled_clients"])
        metrics["rounds"].append(
            {
                "aggregation_round": aggregation_round + 1,
                "train_schedule_percentage": resolved_round_train_percentages[aggregation_round],
                "configured_client_allocation_percentages": resolved_round_client_percentages[aggregation_round],
                **round_info,
            }
        )
        print(
            f"Client churn, aggregation_round {aggregation_round + 1}: "
            f"eligible={round_info['eligible_client_count']}, "
            f"sampled={round_info['sampled_client_count']}, "
            f"dropped={round_info['dropped_to_inactive_count']}, "
            f"new={round_info['newly_eligible_count']}"
        )

        _update_live_state(
            phase="client_training",
            current_round=aggregation_round + 1,
            total_rounds=aggregation_rounds,
            completed_rounds=len(metrics["aggregated_val"]["accuracy"]),
            participating_clients_in_round=len(sampled_clients),
            active_client_position=0,
            active_client_id=None,
            client_epoch_completed=0,
            client_epoch_total=int(config.get("weights_sending_frequency", 0) or 0),
        )
        store_metrics(os.path.join(save_path, "metrics.json"), metrics)
        if round_info["eligibility_safeguard_applied"]:
            print(
                "Client churn safeguard kept one client eligible: "
                f"client={round_info['eligibility_safeguard_client']}"
            )

        client_lrs = config.get("client_learning_rates", None)
        if isinstance(client_lrs, str):
            client_lrs = [float(x.strip()) for x in client_lrs.split(',') if x.strip()]
            print(f"Parsed client learning rates: {client_lrs}")
        # Empty means "not provided", as validation already treats it: fall back
        # to the global learning_rate instead of indexing an empty list.
        if not client_lrs:
            client_lrs = None
        if client_lrs is not None:
            if len(client_lrs) < config.get("num_clients"):
                client_lrs = (client_lrs * config.get("num_clients"))[
                    : config.get("num_clients")
                ]
                print(f"Adjusted client learning rates: {client_lrs}")
        participant_state_dicts = []
        participant_num_samples = []
        participant_local_steps = []
        participant_lrs = []
        participant_momentums = []
        # The configured client allocation applies to the full client pool.
        # If churn makes some clients inactive, this round plan renormalizes the
        # scheduled percentages across the eligible clients so the round subset
        # still receives full coverage.
        round_client_plan = dataset.build_round_client_train_loaders(
            round_train_subsets[aggregation_round],
            resolved_round_client_percentages[aggregation_round],
            eligible_clients=sampled_clients,
            seed=int(config.get("seed", 0) or 0) + aggregation_round,
        )
        round_client_loaders = round_client_plan["client_loaders"]
        trainable_clients = [
            client for client in sampled_clients
            if int(round_client_plan["client_sample_counts"][client]) > 0
        ]
        zero_sample_clients = [
            client for client in sampled_clients
            if int(round_client_plan["client_sample_counts"][client]) <= 0
        ]
        metrics["rounds"][-1]["train_schedule_trainable_client_ids"] = [
            client + 1 for client in trainable_clients
        ]
        metrics["rounds"][-1]["train_schedule_trainable_client_count"] = len(trainable_clients)
        metrics["rounds"][-1]["train_schedule_zero_sample_client_ids"] = [
            client + 1 for client in zero_sample_clients
        ]
        metrics["rounds"][-1]["train_schedule_zero_sample_client_count"] = len(zero_sample_clients)

        _update_live_state(
            phase="client_training",
            current_round=aggregation_round + 1,
            total_rounds=aggregation_rounds,
            completed_rounds=len(metrics["aggregated_val"]["accuracy"]),
            participating_clients_in_round=len(trainable_clients),
            active_client_position=0,
            active_client_id=None,
            client_epoch_completed=0,
            client_epoch_total=int(config.get("weights_sending_frequency", 0) or 0),
        )

        for participant_idx, client in enumerate(trainable_clients, start=1):
            print(f"Training Client {client + 1}")
            round_train_loader = round_client_loaders[client]

            _update_live_state(
                phase="client_training",
                current_round=aggregation_round + 1,
                total_rounds=aggregation_rounds,
                completed_rounds=len(metrics["aggregated_val"]["accuracy"]),
                participating_clients_in_round=len(trainable_clients),
                active_client_position=participant_idx,
                active_client_id=client + 1,
                client_epoch_completed=0,
                client_epoch_total=int(config.get("weights_sending_frequency", 0) or 0),
            )
            store_metrics(os.path.join(save_path, "metrics.json"), metrics)

            def _on_client_epoch(epoch_now, epoch_total, *, _client_id=client + 1, _participant_idx=participant_idx):
                _update_live_state(
                    phase="client_training",
                    current_round=aggregation_round + 1,
                    total_rounds=aggregation_rounds,
                    completed_rounds=len(metrics["aggregated_val"]["accuracy"]),
                    participating_clients_in_round=len(trainable_clients),
                    active_client_position=_participant_idx,
                    active_client_id=_client_id,
                    client_epoch_completed=int(epoch_now),
                    client_epoch_total=int(epoch_total),
                )
                store_metrics(os.path.join(save_path, "metrics.json"), metrics)

            client_models[client].load_state_dict(
                copy.deepcopy(server_model.state_dict())
            )
            if client_lrs is not None:
                lr = float(client_lrs[client])
            else:
                lr = float(config.get("learning_rate")) 

            momentum = float(config.get("momentum") or 0.0)
            optimizer = optim.SGD(
                client_models[client].parameters(),
                lr=lr,
                momentum=momentum,
            )

            if config.get("method") == "fed_prox":
                global_params = [
                    param.detach().clone() for param in server_model.parameters()
                ]
                m = train(
                    client_models[client],
                    optimizer,
                    round_train_loader,
                    val_loader=round_train_loader,
                    epochs=config.get("weights_sending_frequency"),
                    device=device,
                    global_params=global_params,
                    mu=config.get("mu"),
                    eval_split_name="Train",
                    progress_callback=_on_client_epoch,
                )
            else:
                m = train(
                    client_models[client],
                    optimizer,
                    round_train_loader,
                    val_loader=round_train_loader,
                    epochs=config.get("weights_sending_frequency"),
                    device=device,
                    eval_split_name="Train",
                    progress_callback=_on_client_epoch,
                )

            # Per-client metrics are evaluated on the client's own training data each epoch.
            for loss, accuracy, precision, recall, f1, roc_auc in zip(
                m["loss"],
                m["accuracy"],
                m["precision"],
                m["recall"],
                m["f1"],
                m["roc_auc"],
            ):
                metrics["clients"][client]["loss"].append(loss)
                metrics["clients"][client]["accuracy"].append(accuracy)
                metrics["clients"][client]["precision"].append(precision)
                metrics["clients"][client]["recall"].append(recall)
                metrics["clients"][client]["f1"].append(f1)
                metrics["clients"][client]["roc_auc"].append(roc_auc)
            participant_state_dicts.append(client_models[client].state_dict())
            participant_num_samples.append(len(round_train_loader.dataset))
            participant_local_steps.append(
                len(round_train_loader) * config.get("weights_sending_frequency")
            )
            participant_lrs.append(lr)
            participant_momentums.append(momentum)

        metrics["rounds"][-1]["effective_client_allocation_percentages"] = round_client_plan[
            "effective_client_percentages"
        ]
        metrics["rounds"][-1]["train_schedule_client_sample_counts"] = round_client_plan[
            "client_sample_counts"
        ]
        metrics["rounds"][-1]["train_schedule_total_client_samples"] = int(
            sum(metrics["rounds"][-1]["train_schedule_client_sample_counts"])
        )

        print(f"Aggregating global model, aggregation_round {aggregation_round + 1}")
        _update_live_state(phase="aggregation")
        store_metrics(os.path.join(save_path, "metrics.json"), metrics)
        if participant_state_dicts:
            match config.get("method"):
                case "fed_avg":
                    server_model = fed_avg(server_model, participant_state_dicts)
                case "fed_avgw":
                    server_model = fed_avgw(server_model, participant_state_dicts, participant_num_samples)
                case "fed_gp":
                    server_model, best_individual, population = fed_gp(
                        server_model,
                        participant_state_dicts,
                        gp_eval_loader,
                        config.get("individuals"),
                        config.get("generations"),
                        config.get("elitism_size"),
                        config.get("mutation_rate"),
                        config.get("crossover_rate"),
                        save_path,
                        config.get("gp_patience"),
                        device,
                        config.get("mutation_type"),
                        config.get("selection_type"),
                        config.get("min_tree_size", 1),
                        config.get("max_tree_size", 100),
                        config.get("available_primitives"),
                        config.get("crossover_type"),
                        config.get("gp_initialization"),
                        config.get("gp_transfer_learning"),
                        population,
                        # Derive a per-round seed from the global seed so each round's GP is
                        # reproducible yet distinct (otherwise every round would reseed the
                        # global RNG to the same state and share an identical initial population).
                        seed=int(config.get("seed", 0) or 0) + aggregation_round + 1,
                        ephemeral_const_range=tuple(config.get("ephemeral_const_range", (-2.0, 2.0))),
                        weight_magnitude_limit=config.get("weight_magnitude_limit", 1e6),
                        mutation_subtree_maxsize=config.get("mutation_subtree_maxsize", 2),
                        gp_fitness_metric=config.get("gp_fitness_metric", "accuracy"),
                    )
                case "fed_gp_sgd":
                    print("Not implemented yet")
                    exit()
                case "fed_prox":
                    server_model = fed_prox(
                        server_model,
                        participant_state_dicts,
                        participant_num_samples,
                        weighted=bool(config.get("fedprox_weighted", True)),
                    )
                case "fed_nova":
                    server_model = fed_nova(
                        server_model,
                        participant_state_dicts,
                        participant_num_samples,
                        participant_local_steps,
                        participant_lrs,
                        participant_momentums,
                    )
                case _:
                    raise ValueError(f"Unknown method: {config.get('method')}")
        else:
            if zero_sample_clients:
                print(
                    f"Round allocation produced zero trainable samples for aggregation_round {aggregation_round + 1}; "
                    "skipping local training and keeping the current server model."
                )
            else:
                print(
                    f"No eligible clients were available for aggregation_round {aggregation_round + 1}; "
                    "skipping local training and keeping the current server model."
                )

        torch.save(
            server_model.state_dict(),
            os.path.join(save_path, f"global_model_{aggregation_round}.pth"),
        )
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc = evaluate(
            server_model, server_val_loader, device
        )
        metrics["aggregated_val"]["loss"].append(val_loss)
        metrics["aggregated_val"]["accuracy"].append(val_accuracy)
        metrics["aggregated_val"]["precision"].append(val_precision)
        metrics["aggregated_val"]["recall"].append(val_recall)
        metrics["aggregated_val"]["f1"].append(val_f1)
        metrics["aggregated_val"]["roc_auc"].append(val_roc_auc)

        if eval_mode == "train_val_test":
            test_loss, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc = evaluate(
                server_model, test_loader, device
            )
            metrics["aggregated_test"]["loss"].append(test_loss)
            metrics["aggregated_test"]["accuracy"].append(test_accuracy)
            metrics["aggregated_test"]["precision"].append(test_precision)
            metrics["aggregated_test"]["recall"].append(test_recall)
            metrics["aggregated_test"]["f1"].append(test_f1)
            metrics["aggregated_test"]["roc_auc"].append(test_roc_auc)

        _update_live_state(
            phase="round_complete",
            completed_rounds=len(metrics["aggregated_val"]["accuracy"]),
            latest_global_validation={
                "loss": val_loss,
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1,
                "roc_auc": val_roc_auc,
            },
            latest_global_test={
                "loss": test_loss if eval_mode == "train_val_test" else None,
                "accuracy": test_accuracy if eval_mode == "train_val_test" else None,
                "precision": test_precision if eval_mode == "train_val_test" else None,
                "recall": test_recall if eval_mode == "train_val_test" else None,
                "f1": test_f1 if eval_mode == "train_val_test" else None,
                "roc_auc": test_roc_auc if eval_mode == "train_val_test" else None,
            },
        )

        # Persist partial metrics so the dashboard can display live progress.
        store_metrics(os.path.join(save_path, "metrics.json"), metrics)

    end_time = time.time()
    metrics["execution_time"] = end_time - start_time

    _update_live_state(
        phase="completed",
        current_round=aggregation_rounds,
        completed_rounds=len(metrics["aggregated_val"]["accuracy"]),
        active_client_position=0,
        active_client_id=None,
        client_epoch_completed=0,
    )

    store_metrics(os.path.join(save_path, "metrics.json"), metrics)
    return metrics


def main():
    """Load configuration, run one or more experiments, and persist summaries."""
    args = parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:
            config[key] = value
    normalize_run_config_aliases(config)

    # Manual CLI runs bypass the web layer's validate_run_config, so enforce the same
    # web-independent value checks here (single source of truth in utils). This rejects
    # out-of-range settings up front instead of failing deep inside the experiment.
    ok_config, config_message = validate_core_run_config(config)
    if not ok_config:
        raise ValueError(f"Invalid configuration: {config_message}")

    if args.save_path:
        global_save_path = args.save_path
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        config_save_path = config.get("save_path")
        if not config_save_path:
            raise ValueError(
                "save_path is required. Provide --save_path on the command line "
                "or define 'save_path' in the configuration file."
            )
        global_save_path = os.path.join(config_save_path, timestamp)

    os.makedirs(global_save_path, exist_ok=True)
    log_file_path = os.path.join(global_save_path, "train.log")
    log_file = open(log_file_path, "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    # "utils.gp" configures logging at import time, before this redirect, so its
    # handler still holds the original stderr. Re-point it, or GP lines miss train.log.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
        force=True,
    )

    repeats = config.get("repeat", 1)
    all_metrics = []

    # Base seed for the whole experiment. Each repeat derives a distinct seed
    # as `base_seed + repeat`, so repetitions are statistically independent
    # (different RNG draws) while remaining fully reproducible: re-running with
    # the same base seed reproduces the exact same set of runs.
    base_seed = int(config.get("seed", 0) or 0)

    if repeats == 1:
        print("Running experiment without repeats...")

        save_path = global_save_path
        os.makedirs(save_path, exist_ok=True)
        config["seed"] = base_seed
        config["_current_repeat"] = 1
        config["_total_repeats"] = 1
        metrics = run_experiment(config, save_path)
        all_metrics.append(metrics)
        if args.display_metrics:
            print_metrics(metrics)

    else:
        for repeat in range(repeats):
            repeat_seed = base_seed + repeat
            print(f"\n Starting repeat {repeat + 1}/{repeats} (seed={repeat_seed})...\n")
            save_path = os.path.join(global_save_path, f"repeat_{repeat + 1}")
            os.makedirs(save_path, exist_ok=True)

            config["seed"] = repeat_seed
            config["_current_repeat"] = repeat + 1
            config["_total_repeats"] = repeats
            metrics = run_experiment(config, save_path)
            all_metrics.append(metrics)
            if args.display_metrics:
                print_metrics(metrics)

        mean_execution_time = np.mean([m["execution_time"] for m in all_metrics])
        std_execution_time = np.std([m["execution_time"] for m in all_metrics])

        val_accuracy = [m["aggregated_val"]["accuracy"] for m in all_metrics]
        mean_val_accuracy = np.mean(val_accuracy, axis=0)
        std_val_accuracy = np.std(val_accuracy, axis=0)

        val_loss = [m["aggregated_val"]["loss"] for m in all_metrics]
        mean_val_loss = np.mean(val_loss, axis=0)
        std_val_loss = np.std(val_loss, axis=0)

        val_f1 = [m["aggregated_val"]["f1"] for m in all_metrics]
        mean_val_f1 = np.mean(val_f1, axis=0)
        std_val_f1 = np.std(val_f1, axis=0)

        val_precision = [m["aggregated_val"].get("precision", []) for m in all_metrics]
        mean_val_precision = np.mean(val_precision, axis=0)
        std_val_precision = np.std(val_precision, axis=0)

        val_recall = [m["aggregated_val"].get("recall", []) for m in all_metrics]
        mean_val_recall = np.mean(val_recall, axis=0)
        std_val_recall = np.std(val_recall, axis=0)

        val_roc_auc_raw = [m["aggregated_val"]["roc_auc"] for m in all_metrics if m["aggregated_val"]["roc_auc"]]

        if val_roc_auc_raw:
            val_roc_auc_cleaned = [
                [x if x is not None else np.nan for x in run_auc]
                for run_auc in val_roc_auc_raw
            ]

            # Use NaN-aware statistics so valid ROC AUC values still contribute.
            mean_val_roc_auc = np.nanmean(val_roc_auc_cleaned, axis=0)
            std_val_roc_auc = np.nanstd(val_roc_auc_cleaned, axis=0)

            mean_val_roc_auc = np.where(np.isnan(mean_val_roc_auc), None, mean_val_roc_auc)
            std_val_roc_auc = np.where(np.isnan(std_val_roc_auc), None, std_val_roc_auc)
        else:
            mean_val_roc_auc = None
            std_val_roc_auc = None

        test_accuracy = [m["aggregated_test"]["accuracy"] for m in all_metrics if m["aggregated_test"]["accuracy"]]
        if test_accuracy:
            mean_test_accuracy = np.mean(test_accuracy, axis=0)
            std_test_accuracy = np.std(test_accuracy, axis=0)
        else:
            mean_test_accuracy = None
            std_test_accuracy = None

        test_loss = [m["aggregated_test"]["loss"] for m in all_metrics if m["aggregated_test"]["loss"]]
        if test_loss:
            mean_test_loss = np.mean(test_loss, axis=0)
            std_test_loss = np.std(test_loss, axis=0)
        else:
            mean_test_loss = None
            std_test_loss = None

        test_precision = [m["aggregated_test"]["precision"] for m in all_metrics if m["aggregated_test"]["precision"]]
        if test_precision:
            mean_test_precision = np.mean(test_precision, axis=0)
            std_test_precision = np.std(test_precision, axis=0)
        else:
            mean_test_precision = None
            std_test_precision = None

        test_recall = [m["aggregated_test"]["recall"] for m in all_metrics if m["aggregated_test"]["recall"]]
        if test_recall:
            mean_test_recall = np.mean(test_recall, axis=0)
            std_test_recall = np.std(test_recall, axis=0)
        else:
            mean_test_recall = None
            std_test_recall = None

        test_f1 = [m["aggregated_test"]["f1"] for m in all_metrics if m["aggregated_test"]["f1"]]
        if test_f1:
            mean_test_f1 = np.mean(test_f1, axis=0)
            std_test_f1 = np.std(test_f1, axis=0)
        else:
            mean_test_f1 = None
            std_test_f1 = None

        test_roc_auc_raw = [m["aggregated_test"]["roc_auc"] for m in all_metrics if m["aggregated_test"]["roc_auc"]]
        if test_roc_auc_raw:
            test_roc_auc_cleaned = [
                [x if x is not None else np.nan for x in run_auc]
                for run_auc in test_roc_auc_raw
            ]

            mean_test_roc_auc = np.nanmean(test_roc_auc_cleaned, axis=0)
            std_test_roc_auc = np.nanstd(test_roc_auc_cleaned, axis=0)

            mean_test_roc_auc = np.where(np.isnan(mean_test_roc_auc), None, mean_test_roc_auc)
            std_test_roc_auc = np.where(np.isnan(std_test_roc_auc), None, std_test_roc_auc)
        else:
            mean_test_roc_auc = None
            std_test_roc_auc = None

        def safe_tolist(x):
            """Convert NumPy values to native lists while preserving ``None``."""
            return x.tolist() if x is not None else None

        summary_metrics = {
            "mean_global_validation_accuracy": safe_tolist(mean_val_accuracy),
            "std_global_validation_accuracy": safe_tolist(std_val_accuracy),
            "mean_global_validation_loss": safe_tolist(mean_val_loss),
            "std_global_validation_loss": safe_tolist(std_val_loss),
            "mean_global_validation_f1": safe_tolist(mean_val_f1),
            "std_global_validation_f1": safe_tolist(std_val_f1),
            "mean_global_validation_precision": safe_tolist(mean_val_precision),
            "std_global_validation_precision": safe_tolist(std_val_precision),
            "mean_global_validation_recall": safe_tolist(mean_val_recall),
            "std_global_validation_recall": safe_tolist(std_val_recall),
            "mean_global_validation_roc_auc": safe_tolist(mean_val_roc_auc),
            "std_global_validation_roc_auc": safe_tolist(std_val_roc_auc),
            "mean_global_test_accuracy": safe_tolist(mean_test_accuracy),
            "std_global_test_accuracy": safe_tolist(std_test_accuracy),
            "mean_global_test_loss": safe_tolist(mean_test_loss),
            "std_global_test_loss": safe_tolist(std_test_loss),
            "mean_global_test_precision": safe_tolist(mean_test_precision),
            "std_global_test_precision": safe_tolist(std_test_precision),
            "mean_global_test_recall": safe_tolist(mean_test_recall),
            "std_global_test_recall": safe_tolist(std_test_recall),
            "mean_global_test_f1": safe_tolist(mean_test_f1),
            "std_global_test_f1": safe_tolist(std_test_f1),
            "mean_global_test_roc_auc": safe_tolist(mean_test_roc_auc),
            "std_global_test_roc_auc": safe_tolist(std_test_roc_auc),
            "mean_execution_time": safe_tolist(mean_execution_time),
            "std_execution_time": safe_tolist(std_execution_time),
            "all_metrics": all_metrics,
            # Top-level runtime snapshot so the monitor keeps showing the final
            # repetition/round state once the summary replaces the per-repeat
            # metrics as the newest file (otherwise it would reset to 1/N).
            "live_state": {
                **(all_metrics[-1].get("live_state", {}) if all_metrics else {}),
                "phase": "completed",
                "current_repeat": repeats,
                "total_repeats": repeats,
            },
        }

        store_metrics(
            os.path.join(global_save_path, "summary_metrics.json"), summary_metrics
        )
        if args.display_metrics:
            print("Summary Metrics:")
            print(summary_metrics)

    print("End.")


if __name__ == "__main__":
    main()
