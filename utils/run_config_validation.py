"""Web-independent validation of run-configuration values.

This is the single source of truth for the value-level checks (numeric ranges, allowed
method, FedGP settings, FedProx ``mu``, custom-model code) shared by:

- the web API path (``web_backend.api_handlers.run_workflows.validate_run_config``), and
- the standalone worker (``main.py``), which the Job Manager launches as a subprocess.

Keeping these checks here — with no dependency on the web layer — lets both paths reject
the same out-of-range settings instead of having the worker fail late on bad input. The
genuinely web-specific checks (dataset support/refs, run display name, round-schedule
normalization) stay in ``run_workflows`` since they need web-only collaborators.
"""

from __future__ import annotations

import ast
from typing import Any


ALLOWED_METHODS = {"fed_avg", "fed_avgw", "fed_gp", "fed_prox", "fed_nova"}

CUSTOM_MODEL_CODE_MAX_MB = 5
CUSTOM_MODEL_CODE_MAX_BYTES = CUSTOM_MODEL_CODE_MAX_MB * 1024 * 1024

FED_GP_ALLOWED_INITIALIZATION = {"genHalfAndHalf", "genFull", "genGrow"}
FED_GP_ALLOWED_MUTATION = {"mutUniform", "mutNodeReplacement", "mutInsert"}
FED_GP_ALLOWED_SELECTION = {"selTournament", "selRoulette", "selRandom"}
FED_GP_ALLOWED_CROSSOVER = {"cxOnePoint", "cxOnePointLeafBiased"}
FED_GP_ALLOWED_FITNESS_METRIC = {"accuracy", "loss", "precision", "recall", "f1"}
FED_GP_ALLOWED_TRANSFER_LEARNING = {
    False,
    "full_reuse",
    "elite_reuse",
    "elite_mutation_warm_start",
    "hybrid",
}
FED_GP_ALLOWED_PRIMITIVES = {
    "torch.add",
    "torch.sub",
    "torch.mul",
    "torch_protected_div",
    "torch_mean",
    "torch_median",
    "torch.abs",
    "torch_protected_sqrt",
    "torch_pow",
    "torch.log",
    "torch.sin",
    "torch.cos",
}
BLOCKED_CUSTOM_MODEL_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "importlib",
}
BLOCKED_CUSTOM_MODEL_CALLS = {"eval", "exec", "compile", "__import__"}


def coerce_bool(raw: Any, default: bool = False) -> bool:
    """Coerce a config value to a real bool, accepting YAML/JSON bools, strings, and 0/1."""
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off", ""}:
        return False
    return default


def normalize_gp_transfer_learning(raw: Any) -> str | bool:
    """Normalize boolean and UI values for FedGP transfer learning."""
    if raw is True:
        return "full_reuse"
    if raw is False or raw is None:
        return False
    text = str(raw).strip()
    if not text:
        return False
    lower = text.lower()
    if lower in {"false", "none", "disabled", "off"}:
        return False
    if lower == "true":
        return "full_reuse"
    return text


def _find_model_name_node(tree: ast.Module) -> ast.AST | None:
    """Find the top-level ``MODEL_NAME`` assignment value in custom code."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "MODEL_NAME":
                return node.value
    return None


def _validate_custom_model_ast(tree: ast.Module, cfg: dict[str, Any]) -> tuple[bool, str]:
    """Validate custom model code structure and blocked constructs."""
    build_fn = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_model"
        ),
        None,
    )
    if build_fn is None:
        return (
            False,
            "custom_model_code must define build_model(n_channels, n_classes)",
        )
    if build_fn is not None and len([arg.arg for arg in build_fn.args.args]) < 2:
        return False, "build_model must accept at least (n_channels, n_classes)"

    model_name_node = _find_model_name_node(tree)
    if (
        not isinstance(model_name_node, ast.Constant)
        or not isinstance(model_name_node.value, str)
        or not model_name_node.value.strip()
    ):
        return False, "custom_model_code must define MODEL_NAME = '...'(non-empty string)"
    cfg["custom_model_name"] = model_name_node.value.strip()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_CUSTOM_MODEL_IMPORTS:
                    return False, f"Blocked import in custom_model_code: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module_root = (node.module or "").split(".")[0]
            if module_root in BLOCKED_CUSTOM_MODEL_IMPORTS:
                return False, f"Blocked import in custom_model_code: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CUSTOM_MODEL_CALLS:
                return False, f"Blocked function call in custom_model_code: {node.func.id}"

    return True, ""


def validate_custom_model_config(cfg: dict[str, Any]) -> tuple[bool, str]:
    """Validate custom model settings and normalize the model name."""
    custom_mode = str(cfg.get("custom_model_mode", "default") or "default")
    if custom_mode not in {"default", "code", "upload_file", "loaded_file"}:
        return False, "custom_model_mode must be 'default', 'code', 'upload_file', or 'loaded_file'"

    custom_code = str(cfg.get("custom_model_code", "") or "")
    if custom_mode != "default" and not custom_code.strip():
        return False, "custom_model_code is required when custom_model_mode is not 'default'"
    if len(custom_code.encode("utf-8", errors="ignore")) > CUSTOM_MODEL_CODE_MAX_BYTES:
        return False, f"custom_model_code is too large (max {CUSTOM_MODEL_CODE_MAX_MB} MB)"

    if custom_mode == "default":
        cfg["custom_model_name"] = "DefaultNet"
        return True, ""

    try:
        tree = ast.parse(custom_code)
    except SyntaxError as exc:
        line = getattr(exc, "lineno", "?")
        col = getattr(exc, "offset", "?")
        return False, f"custom_model_code syntax error at line {line}, col {col}: {exc.msg}"

    return _validate_custom_model_ast(tree, cfg)


def _validate_fed_gp_choice(
    value: str,
    allowed_values: set[str],
    field_name: str,
) -> tuple[bool, str]:
    """Validate a FedGP enum-like setting."""
    if value in allowed_values:
        return True, ""
    return False, f"{field_name} must be one of: " + ", ".join(sorted(allowed_values))


def normalize_available_primitives(cfg: dict[str, Any]) -> tuple[bool, str, list[str]]:
    """Normalize and validate the FedGP primitive list payload shape."""
    raw_primitives = cfg.get("available_primitives", "")
    if isinstance(raw_primitives, str):
        primitives = [item.strip() for item in raw_primitives.split(",") if item.strip()]
    elif isinstance(raw_primitives, list):
        primitives = [str(item).strip() for item in raw_primitives if str(item).strip()]
        cfg["available_primitives"] = ",".join(primitives)
    else:
        return False, "available_primitives must be a comma-separated string or list for fed_gp", []

    if not primitives:
        return False, "available_primitives cannot be empty for fed_gp", []

    invalid_primitives = [item for item in primitives if item not in FED_GP_ALLOWED_PRIMITIVES]
    if invalid_primitives:
        return False, f"Unsupported primitives for fed_gp: {', '.join(invalid_primitives)}", []

    return True, "", primitives


def validate_fed_gp_config(cfg: dict[str, Any]) -> tuple[bool, str]:
    """Validate FedGP-only configuration settings."""
    try:
        individuals = int(cfg.get("individuals", 0))
        generations = int(cfg.get("generations", 0))
        elitism_size = int(cfg.get("elitism_size", 0))
        gp_patience = int(cfg.get("gp_patience", 0))
        max_tree_size = int(cfg.get("max_tree_size", 0))
        mutation_subtree_maxsize = int(cfg.get("mutation_subtree_maxsize", 2))
        mutation_rate = float(cfg.get("mutation_rate", -1))
        crossover_rate = float(cfg.get("crossover_rate", -1))
    except Exception:
        return False, "FedGP parameters contain invalid numeric types"

    if not (2 <= individuals <= 500):
        return False, "individuals must be in [2, 500] for fed_gp"
    if not (1 <= generations <= 2000):
        return False, "generations must be in [1, 2000] for fed_gp"
    if not (1 <= elitism_size < individuals):
        return False, "elitism_size must be >= 1 and < individuals for fed_gp"
    if not (0 <= mutation_rate <= 1):
        return False, "mutation_rate must be in [0, 1] for fed_gp"
    if not (0 <= crossover_rate <= 1):
        return False, "crossover_rate must be in [0, 1] for fed_gp"
    if mutation_rate + crossover_rate <= 0:
        return False, "At least one of mutation_rate or crossover_rate must be > 0 for fed_gp"
    if not (1 <= gp_patience <= generations):
        return False, "gp_patience must be in [1, generations] for fed_gp"
    if not (1 <= max_tree_size <= 64):
        return False, "max_tree_size must be in [1, 64] for fed_gp"
    if not (1 <= mutation_subtree_maxsize <= 64):
        return False, "mutation_subtree_maxsize must be in [1, 64] for fed_gp"

    min_tree_raw = cfg.get("min_tree_size", False)
    if min_tree_raw is False:
        min_tree_size = 1
    else:
        try:
            min_tree_size = int(min_tree_raw)
        except Exception:
            return False, "min_tree_size must be an integer or false for fed_gp"
    if not (1 <= min_tree_size <= max_tree_size):
        return False, "min_tree_size must be >= 1 and <= max_tree_size for fed_gp"

    for value, allowed_values, field_name in (
        (
            str(cfg.get("gp_initialization", "") or ""),
            FED_GP_ALLOWED_INITIALIZATION,
            "gp_initialization",
        ),
        (str(cfg.get("mutation_type", "") or ""), FED_GP_ALLOWED_MUTATION, "mutation_type"),
        (str(cfg.get("selection_type", "") or ""), FED_GP_ALLOWED_SELECTION, "selection_type"),
        (str(cfg.get("crossover_type", "") or ""), FED_GP_ALLOWED_CROSSOVER, "crossover_type"),
        (str(cfg.get("gp_fitness_metric", "accuracy") or "accuracy"), FED_GP_ALLOWED_FITNESS_METRIC, "gp_fitness_metric"),
    ):
        ok_choice, choice_message = _validate_fed_gp_choice(value, allowed_values, field_name)
        if not ok_choice:
            return False, choice_message

    # Roulette selection needs non-negative fitness, but the loss metric is represented as
    # -loss (negative), which makes selRoulette collapse. Reject the combination up front.
    selection_type = str(cfg.get("selection_type", "") or "")
    fitness_metric = str(cfg.get("gp_fitness_metric", "accuracy") or "accuracy").strip().lower()
    if selection_type == "selRoulette" and fitness_metric == "loss":
        return (
            False,
            "selRoulette is incompatible with the loss fitness metric (roulette needs "
            "non-negative fitness; loss is maximized as -loss). Pick another selection_type "
            "or fitness metric.",
        )

    ok_primitives, primitives_message, _ = normalize_available_primitives(cfg)
    if not ok_primitives:
        return False, primitives_message

    normalized_gp_tl = normalize_gp_transfer_learning(
        cfg.get("gp_transfer_learning", "full_reuse")
    )
    if normalized_gp_tl not in FED_GP_ALLOWED_TRANSFER_LEARNING:
        return (
            False,
            "gp_transfer_learning must be one of: disabled, full_reuse, "
            "elite_reuse, elite_mutation_warm_start, hybrid",
        )
    cfg["gp_transfer_learning"] = normalized_gp_tl

    return True, ""


def validate_common_numeric_config(cfg: dict[str, Any]) -> tuple[bool, str]:
    """Validate shared numeric run settings."""
    try:
        num_clients = int(cfg.get("num_clients", 0))
        initial_eligible_clients = int(cfg.get("initial_eligible_clients", num_clients))
        batch_size = int(cfg.get("batch_size", 0))
        local_epochs = int(cfg.get("local_model_epochs", 0))
        frequency = int(cfg.get("weights_sending_frequency", 0))
        learning_rate = float(cfg.get("learning_rate", 0))
        server_learning_rate = float(cfg.get("server_learning_rate", 0))
        death_prob = float(cfg.get("death_prob", 0) or 0)
        new_client_prob = float(cfg.get("new_client_prob", 0) or 0)
        seed = int(cfg.get("seed", 0) or 0)
        # NB: no ``or default`` for these — it would turn an invalid 0 into a passing
        # default and hide the very value the ``> 0`` checks below are meant to reject.
        momentum = float(cfg.get("momentum", 0.0))
        server_warmup_epochs = int(cfg.get("server_warmup_epochs", 1))
        repeat = int(cfg.get("repeat", 1))
    except Exception:
        return False, "Config contains invalid numeric types"

    if not (1 <= num_clients <= 100):
        return False, "num_clients must be in [1, 100]"
    if not (1 <= initial_eligible_clients <= num_clients):
        return False, "initial_eligible_clients must be in [1, num_clients]"
    if not (1 <= batch_size <= 4096):
        return False, "batch_size must be in [1, 4096]"
    if not (1 <= local_epochs <= 10000):
        return False, "local_model_epochs must be in [1, 10000]"
    if not (1 <= frequency <= 10000):
        return False, "weights_sending_frequency must be in [1, 10000]"
    if not (0 < learning_rate <= 10):
        return False, "learning_rate must be > 0 and <= 10"
    if not (0 < server_learning_rate <= 10):
        return False, "server_learning_rate must be > 0 and <= 10"
    if not (0 <= death_prob <= 0.9):
        return False, "death_prob must be in [0.0, 0.9]"
    if not (0 <= new_client_prob <= 0.9):
        return False, "new_client_prob must be in [0.0, 0.9]"
    if seed < 0:
        return False, "seed must be >= 0"
    if local_epochs % frequency != 0:
        return False, "local_model_epochs must be divisible by weights_sending_frequency"
    # Momentum feeds both the client SGD optimizer and the FedNova normalization
    # coefficient (which divides by 1 - momentum), so values >= 1 are rejected outright.
    if not (0 <= momentum < 1):
        return False, "momentum must be in [0, 1)"
    if server_warmup_epochs <= 0:
        return False, "server_warmup_epochs must be > 0"
    if repeat <= 0:
        return False, "repeat must be > 0"

    # Optional split ratios: validate only when provided, since some datasets ship with
    # native splits and leave these unset/disabled in the UI.
    for key in ("train_val_split", "server_data_percentage"):
        raw_ratio = cfg.get(key)
        if raw_ratio in (None, ""):
            continue
        try:
            ratio = float(raw_ratio)
        except Exception:
            return False, f"{key} must be a number in (0, 1)"
        if not (0 < ratio < 1):
            return False, f"{key} must be in (0, 1)"

    # Per-client learning rates are optional (otherwise learning_rate is used for all).
    # When provided, every entry must be a valid positive lr. Accept a list or a
    # comma-separated string (both shapes reach this validator depending on the source).
    raw_client_lrs = cfg.get("client_learning_rates")
    if raw_client_lrs not in (None, "", []):
        if isinstance(raw_client_lrs, str):
            items = [item.strip() for item in raw_client_lrs.split(",") if item.strip()]
        elif isinstance(raw_client_lrs, (list, tuple)):
            items = list(raw_client_lrs)
        else:
            return False, "client_learning_rates must be a list or a comma-separated string"
        try:
            client_lrs = [float(item) for item in items]
        except Exception:
            return False, "client_learning_rates must contain only numbers"
        if not client_lrs:
            return False, "client_learning_rates cannot be empty when provided"
        if any(not (0 < lr <= 10) for lr in client_lrs):
            return False, "every client_learning_rates value must be > 0 and <= 10"

    cfg["initial_eligible_clients"] = initial_eligible_clients
    cfg["death_prob"] = death_prob
    cfg["new_client_prob"] = new_client_prob
    cfg["seed"] = seed
    cfg["server_learning_rate"] = server_learning_rate
    return True, ""


def validate_core_run_config(cfg: dict[str, Any]) -> tuple[bool, str]:
    """Run every web-independent value validation, mutating ``cfg`` in place.

    Covers: allowed method, numeric ranges, evaluation split mode, FedGP settings
    (when ``method == fed_gp``), FedProx ``mu`` (when ``method == fed_prox``), and the
    custom-model code. Returns ``(ok, message)``; ``message`` is empty when ``ok`` is True.
    """
    method_name = str(cfg.get("method") or "").strip().lower()
    if method_name not in ALLOWED_METHODS:
        return False, "method must be one of: " + ", ".join(sorted(ALLOWED_METHODS))

    ok_numeric, numeric_message = validate_common_numeric_config(cfg)
    if not ok_numeric:
        return False, numeric_message

    mode = str(cfg.get("evaluation_split_mode", "train_val_test"))
    if mode not in {"train_val_test", "train_val"}:
        return False, "evaluation_split_mode must be 'train_val_test' or 'train_val'"

    if method_name == "fed_gp":
        ok_fed_gp, fed_gp_message = validate_fed_gp_config(cfg)
        if not ok_fed_gp:
            return False, fed_gp_message

    if method_name == "fed_prox":
        # mu only matters for FedProx; a non-positive mu would silently drop the proximal
        # term (see utils/train.py), turning the run into plain FedAvg without warning.
        try:
            mu = float(cfg.get("mu", 0))
        except Exception:
            return False, "mu must be a number for fed_prox"
        if not (0 < mu <= 1000):
            return False, "mu must be > 0 and <= 1000 for fed_prox"
        # Normalize the aggregation choice to a real bool (default = paper's weighted rule).
        cfg["fedprox_weighted"] = coerce_bool(cfg.get("fedprox_weighted"), default=True)

    ok_custom_model, custom_model_message = validate_custom_model_config(cfg)
    if not ok_custom_model:
        return False, custom_model_message

    return True, ""
