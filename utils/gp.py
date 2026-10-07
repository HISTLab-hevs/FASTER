"""Genetic-programming engine used by the FedGP aggregation method."""

import os
import random
import logging
from deap import base, creator, tools, gp
import torch
from torch.func import functional_call, vmap
from utils.evaluate import evaluate
from utils.gp_primitives import (
    torch_protected_div, torch_protected_sqrt, torch_mean, torch_median, torch_pow,
    torch_abs, torch_log, torch_sin, torch_cos,
)
from utils.gp_population import init_elite_random, init_elite_mutation, init_hybrid_population
from utils.seed import set_global_seed
from datetime import datetime
import copy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Inclusive range the ephemeral constants are sampled from. Set by initialize_deap;
# kept as module state (rather than a closure) so _rand_constant stays the SAME object
# across calls — DEAP raises if an ephemeral constant is re-registered with a different
# function object, which a fresh closure would be.
_ephemeral_const_range = (-2.0, 2.0)


def _rand_constant():
    """Sample an ephemeral constant for the GP (e.g. to build ``0.3 * CLIENT1``)."""
    low, high = _ephemeral_const_range
    return round(random.uniform(low, high), 3)


_original_sample = random.sample

def _deap_safe_sample(population, k, *, counts=None):
    """
    Safely wraps `random.sample` to ensure compatibility with Python 3.12+.
    
    Context:
    DEAP's internal functions (specifically `gp.cxOnePointLeafBiased`) 
    pass 'set' objects to `random.sample`. Starting from Python 3.9, sampling 
    directly from sets is deprecated, and in Python 3.12+ it explicitly raises 
    a TypeError ("Population must be a sequence").
    
    This wrapper intercepts the call, safely casts any 'set' to a 'tuple' 
    (which is an ordered sequence), and passes it to the original random.sample.
    """

    if isinstance(population, set):
        population = tuple(population)
    return _original_sample(population, k, counts=counts)

# Patch only DEAP's GP namespace so Python 3.12-compatible sampling is used
# without changing global random.sample behavior.
gp.random.sample = _deap_safe_sample

def initialize_deap(num_clients, mutation_type, selection_type, min_tree_size, max_tree_size, available_primitives, crossover_type, gp_inizialization, ephemeral_const_range=(-2.0, 2.0), mutation_subtree_maxsize=2):
    """Create the DEAP toolbox and primitive set for GP aggregation.

    ``ephemeral_const_range`` is the inclusive (low, high) interval the ephemeral
    constants are sampled from (only used when ``"ephemeral_constant"`` is in
    ``available_primitives``).

    ``mutation_subtree_maxsize`` is the maximum height of the random subtree that
    ``mutUniform`` grafts in place of a mutated node (the min is 0). Keeping it small
    limits tree bloat across generations; the DEAP-conventional default is 2.
    """
    global _ephemeral_const_range
    low, high = float(ephemeral_const_range[0]), float(ephemeral_const_range[1])
    _ephemeral_const_range = (min(low, high), max(low, high))
    try:
        creator.FitnessMax
    except AttributeError:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    try:
        creator.Individual
    except AttributeError:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    pset = gp.PrimitiveSet("MAIN", num_clients)

    # Register only the primitives enabled by configuration.
    if "torch.add" in available_primitives:
        pset.addPrimitive(torch.add, 2)
    if "torch.sub" in available_primitives:
        pset.addPrimitive(torch.sub, 2)
    if "torch.mul" in available_primitives:
        pset.addPrimitive(torch.mul, 2)
    if "torch_protected_div" in available_primitives:
        pset.addPrimitive(torch_protected_div, 2)
    if "torch_mean" in available_primitives:
        for i in range(2, num_clients + 1):
            pset.addPrimitive(torch_mean, i)
    if "torch_median" in available_primitives:
        if num_clients < 3:
            logging.warning(
                "torch_median requires at least 3 clients (the median of 2 values equals "
                "their mean); with num_clients=%d it is ignored.", num_clients
            )
        for i in range(3, num_clients + 1):
            pset.addPrimitive(torch_median, i)
    if "torch.abs" in available_primitives:
        pset.addPrimitive(torch_abs, 1)
    if "torch_protected_sqrt" in available_primitives:
        pset.addPrimitive(torch_protected_sqrt, 1)
    if "torch_pow" in available_primitives:
        # Arity 2: pow(base, exponent). The exponent comes from the tree (an ephemeral
        # constant or another subtree), so the power is fixed per individual.
        pset.addPrimitive(torch_pow, 2)
    if "torch.log" in available_primitives:
        pset.addPrimitive(torch_log, 1)
    if "torch.sin" in available_primitives:
        pset.addPrimitive(torch_sin, 1)
    if "torch.cos" in available_primitives:
        pset.addPrimitive(torch_cos, 1)

    # Ephemeral constants let the GP build scaled aggregations such as 0.3 * CLIENT1.
    # The value is sampled once when a node is created and stored in the tree.
    if "ephemeral_constant" in available_primitives:
        try:
            pset.addEphemeralConstant("randConst", _rand_constant)
        except Exception:
            # initialize_deap runs once per round; drop the stale class and re-register.
            if hasattr(gp, "randConst"):
                delattr(gp, "randConst")
            pset.addEphemeralConstant("randConst", _rand_constant)


    for i in range(num_clients):
        pset.renameArguments(**{f"ARG{i}": f"CLIENT{i + 1}"})


    if gp_inizialization == "genFull":
        toolbox.register("expr", gp.genFull, pset=pset, min_=min_tree_size, max_=max_tree_size)
    elif gp_inizialization == "genGrow":
        toolbox.register("expr", gp.genGrow, pset=pset, min_=min_tree_size, max_=max_tree_size)
    else:
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_tree_size, max_=max_tree_size)

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    if selection_type == "selTournament":
        toolbox.register("select", tools.selTournament, tournsize=3)
    elif selection_type == "selRoulette":
        toolbox.register("select", tools.selRoulette)
    else:
        toolbox.register("select", tools.selRandom)


    if crossover_type == "cxOnePoint":
        print("Crossover type: cxOnePoint")
        toolbox.register("mate", gp.cxOnePoint)
    elif crossover_type == "cxOnePointLeafBiased":
        print("Crossover type: cxOnePointLeafBiased")
        toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
    else:
        print("Wrong crossover type, cxOnePoint is applied.")
        toolbox.register("mate", gp.cxOnePoint)

    # Generator for the subtree grafted on mutation. Registered before "mutate" so the
    # mutUniform branches use it (not toolbox.expr, which builds full init-sized trees and
    # drives bloat). min_=0 lets a mutation collapse to a single terminal.
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=mutation_subtree_maxsize)

    if mutation_type == "mutUniform":
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    elif mutation_type == "mutNodeReplacement":
        toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
    elif mutation_type == "mutInsert":
        toolbox.register("mutate", gp.mutInsert, pset=pset)
    else:
        print("Wrong mutation type, mutUniform is applied.")
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    return toolbox, pset


# Aggregated weights beyond this magnitude (or non-finite) mean the GP program blew up.
# Such an individual yields a meaningless, saturated model, so it is discarded rather
# than evaluated: this skips a wasted forward pass on the sequential path and, more
# importantly, prevents a saturated model from scoring the majority-class baseline and
# being selected over a genuinely competitive one. Configurable via main(); kept as
# module state so the validity helper does not need it threaded through every signature.
_WEIGHT_MAGNITUDE_LIMIT = 1e6


# Fitness metric for the GP search, set by main(). The GP always MAXIMIZES fitness, so the
# loss metric is represented as -loss ("higher is better" everywhere — selBest, elitism, and
# the patience improvement test stay unchanged); accuracy/precision/recall/f1 are already
# higher-is-better. Kept as module state, like the limit above, so the eval helpers don't
# need it threaded through every signature.
_FITNESS_METRIC = "accuracy"
_ALLOWED_FITNESS_METRICS = ("accuracy", "loss", "precision", "recall", "f1")


def _metric_fitness(metrics):
    """Map a dict of evaluation metrics to a 'higher is better' fitness value.

    ``metrics`` holds at least the selected metric's key. Loss is negated (minimizing loss =
    maximizing -loss); every other metric is in [0, 1] and used directly.
    """
    if _FITNESS_METRIC == "loss":
        return -float(metrics["loss"])
    return float(metrics[_FITNESS_METRIC])


def _worst_fitness():
    """Fitness for invalid/degenerate individuals: the worst possible under maximization.

    accuracy/precision/recall/f1 live in [0, 1] so 0.0 is worst. The loss fitness is -loss,
    so a large negative value keeps a degenerate individual below any real one (the
    weight-magnitude guard already caps the realistic loss far above -1e9).
    """
    return -1e9 if _FITNESS_METRIC == "loss" else 0.0


def _weights_are_degenerate(tensor):
    """True if the aggregated weights are non-finite (NaN/inf) or explode in magnitude."""
    max_abs = tensor.abs().max().item()
    return (max_abs != max_abs) or (max_abs > _WEIGHT_MAGNITUDE_LIMIT)  # NaN != NaN


def eval_individual(individual, client_weights, test_data, toolbox, model, device):
    """Evaluate one GP individual by aggregating client weights and scoring its fitness.

    The fitness is the configured metric (accuracy, or -loss when minimizing loss). The
    architecture is identical across individuals, so a single reusable ``model`` instance is
    passed in: only the aggregated weights change. ``load_state_dict`` overwrites every
    parameter each call, so no per-individual deepcopy is needed.

    Despite the parameter name, `test_data` receives the validation loader in normal
    operation (see call site in main.py) so fitness never touches the held-out test set.
    """
    func = toolbox.compile(expr=individual)
    weight_keys = list(client_weights[0].keys())
    reference_state = model.state_dict()
    aggregated_weights = {}

    for key in weight_keys:
        client_tensors = [client_weights[i][key].float() for i in range(len(client_weights))]
        try:
            combined_tensor = func(*client_tensors)
        except Exception:
            # Degenerate individual (e.g. a scalar-only subtree feeding a tensor op).
            return _worst_fitness()
        if not torch.is_tensor(combined_tensor):
            return _worst_fitness()
        ref = reference_state.get(key)
        if ref is not None and combined_tensor.shape != ref.shape:
            return _worst_fitness()
        if _weights_are_degenerate(combined_tensor):
            # Exploding/NaN weights: discard instead of clamping and wasting a forward.
            return _worst_fitness()
        aggregated_weights[key] = combined_tensor

    try:
        model.load_state_dict(aggregated_weights)
    except RuntimeError as e:
        logging.warning(f"Error loading weights for individual {individual}: {e}")
        return _worst_fitness()

    # Score the aggregated model with the shared evaluation helper.
    loss, accuracy, precision, recall, f1, _ = evaluate(model, test_data, device)
    return _metric_fitness(
        {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    )


def _build_client_layout(client_weights, reference_state, device):
    """Precompute the flat per-client weight vectors and the unflatten layout.

    Every GP primitive is elementwise across the client axis (add/sub/mul/div/...,
    and mean/median which reduce over the stacked clients), so aggregating each
    parameter independently is identical to aggregating one flat vector that
    concatenates all parameters. We therefore flatten every client's full state
    into a single vector once — clients are fixed for the whole evolution — and let
    each individual's program run a single op-chain instead of one per parameter key.

    Returns (client_flat, keys, sizes, shapes, dtypes, total): ``client_flat`` is a
    list of 1-D tensors (one per client) on ``device``; the rest describe how to slice
    an aggregated flat vector back into a state_dict.
    """
    keys = list(reference_state.keys())
    sizes = [reference_state[k].numel() for k in keys]
    shapes = [reference_state[k].shape for k in keys]
    dtypes = [reference_state[k].dtype for k in keys]
    total = sum(sizes)

    client_flat = [
        torch.cat([client_weights[i][k].flatten().float() for k in keys]).to(device)
        for i in range(len(client_weights))
    ]
    return client_flat, keys, sizes, shapes, dtypes, total


def _aggregate_individual_flat(individual, client_flat, total, toolbox):
    """Run a GP individual on the flat client vectors and return the aggregated vector.

    Returns the 1-D aggregated tensor, or ``None`` if the individual is invalid: its
    program raised, returned a non-tensor, produced a vector whose length does not match
    the total parameter count, or produced exploding/NaN weights. Discarding the latter
    (rather than clamping) keeps it out of the batched forward and out of selection.
    """
    func = toolbox.compile(expr=individual)
    try:
        combined = func(*client_flat)
    except Exception:
        return None
    if not torch.is_tensor(combined) or combined.numel() != total:
        return None
    if _weights_are_degenerate(combined):
        return None
    return combined


def batched_eval(population, client_weights, test_data, toolbox, model_template, device):
    """Evaluate a whole GP population in a single vectorized forward pass per batch.

    Every individual shares the same architecture and only differs in its aggregated
    weights, so instead of running ``N`` tiny forward passes we stack the ``N`` weight
    sets along a leading dimension and evaluate them all at once with
    ``vmap(functional_call)``. This collapses the per-individual Python/kernel-launch
    overhead into one fused launch — a large win on GPU (CUDA), where this overhead,
    not the math, is the bottleneck for small models.

    The weight aggregation is itself flattened (see ``_build_client_layout``): each
    individual runs a single elementwise op-chain over precomputed flat client vectors
    instead of one chain per parameter key, which removes the aggregation bottleneck
    that dominates once the forward pass is batched.

    Invalid individuals (GP program errors or shape mismatches) are excluded from the
    batch and assigned a fitness of 0.0, preserving population order in the result.

    Returns a list of accuracies as 0-1 fractions (one per individual, same order as
    ``population``), matching the scale of the sequential ``eval_individual`` path.
    """
    fitnesses = [_worst_fitness()] * len(population)
    if test_data is None or len(getattr(test_data, "dataset", [])) == 0:
        return fitnesses

    reference_state = model_template.state_dict()
    client_flat, keys, sizes, shapes, dtypes, total = _build_client_layout(
        client_weights, reference_state, device
    )

    # Aggregate weights per individual (one flat op-chain each) and drop the invalid ones.
    valid_positions = []
    valid_flat = []
    for pos, individual in enumerate(population):
        aggregated = _aggregate_individual_flat(individual, client_flat, total, toolbox)
        if aggregated is None:
            logging.warning(f"Invalid individual excluded from batch: {individual}")
            continue
        valid_positions.append(pos)
        valid_flat.append(aggregated)

    if not valid_flat:
        return fitnesses

    # Stack the flat aggregated vectors -> [N, total], then unflatten once into the
    # per-key stacked params/buffers [N, *param_shape] expected by functional_call.
    stacked_flat = torch.stack(valid_flat, dim=0)  # [N, total]
    stacked = {}
    offset = 0
    for key, size, shape, dtype in zip(keys, sizes, shapes, dtypes):
        column = stacked_flat[:, offset:offset + size]
        stacked[key] = column.reshape(stacked_flat.shape[0], *shape).to(dtype)
        offset += size

    base_model = copy.deepcopy(model_template).to(device).eval()

    def _forward(params_and_buffers, inputs):
        return functional_call(base_model, params_and_buffers, (inputs,))

    batched_forward = vmap(_forward, in_dims=(0, None))

    n_valid = len(valid_flat)
    correct = torch.zeros(n_valid, device=device)
    loss_sum = torch.zeros(n_valid, device=device)
    use_loss = _FITNESS_METRIC == "loss"
    need_conf = _FITNESS_METRIC in ("precision", "recall", "f1")
    # Per-individual flattened confusion matrix [N, C*C] (true*C + pred), built lazily once
    # the class count is known from the first forward. precision/recall/f1 derive from it.
    confmat_flat = None
    num_classes = None
    total_samples = 0
    with torch.no_grad():
        for data, target in test_data:
            data = data.to(device)
            target = target.to(device).view(-1).long()
            output = batched_forward(stacked, data)  # [N, batch, num_classes]
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
            predicted = output.argmax(dim=-1)         # [N, batch]
            correct += (predicted == target.unsqueeze(0)).sum(dim=1).float()
            if use_loss:
                # Per-individual cross-entropy summed over the batch (matches evaluate()).
                log_probs = torch.log_softmax(output, dim=-1)              # [N, batch, classes]
                tgt = target.view(1, -1, 1).expand(output.size(0), -1, 1)  # [N, batch, 1]
                loss_sum += (-log_probs.gather(-1, tgt).squeeze(-1)).sum(dim=1)
            if need_conf:
                if confmat_flat is None:
                    num_classes = output.size(-1)
                    confmat_flat = torch.zeros(n_valid, num_classes * num_classes, device=device)
                true_b = target.unsqueeze(0).expand(n_valid, -1)            # [N, batch]
                idx = true_b * num_classes + predicted                     # [N, batch] (true*C + pred)
                confmat_flat.scatter_add_(1, idx, torch.ones_like(idx, dtype=confmat_flat.dtype))
            total_samples += target.size(0)

    if total_samples:
        # Same scale as the sequential eval_individual path (all in [0, 1] except loss, which
        # is returned as -mean_loss so "higher is better" holds everywhere).
        if use_loss:
            scores = (-(loss_sum / total_samples)).tolist()
        elif need_conf and confmat_flat is not None:
            cm = confmat_flat.view(n_valid, num_classes, num_classes)      # [N, true, pred]
            tp = torch.diagonal(cm, dim1=1, dim2=2)                        # [N, C]
            predicted_pos = cm.sum(dim=1)                                  # [N, C] predicted as c
            actual_pos = cm.sum(dim=2)                                     # [N, C] true c (support)
            precision = tp / predicted_pos.clamp(min=1)                    # zero_division=0
            recall = tp / actual_pos.clamp(min=1)
            f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
            per_class = {"precision": precision, "recall": recall, "f1": f1}[_FITNESS_METRIC]
            # Macro over the labels sklearn uses: those present in targets or predictions.
            present = (actual_pos > 0) | (predicted_pos > 0)               # [N, C]
            scores = (
                (per_class * present).sum(dim=1) / present.sum(dim=1).clamp(min=1)
            ).tolist()
        else:
            scores = (correct / total_samples).tolist()
        for n, pos in enumerate(valid_positions):
            fitnesses[pos] = scores[n]
    return fitnesses


def run_evolution(client_weights, test_dataset, toolbox, model_template, n_individuals=20, n_generations=20, elitism_size=1, mutation_rate=0.2, crossover_rate=0.5, save_path=".", patience=None, device="cpu", population=None, seed=None):
    """Run the GP search loop and return the best individual plus final population.

    Despite the parameter name, `test_dataset` receives the validation loader in normal
    operation (see call site in main.py) so fitness never touches the held-out test set.

    Args:
        seed: Accepted for backward compatibility; the global RNG is already seeded by the
            caller (:func:`main`) before this runs, so this value is not re-applied here.
    """
    if population is None:
        population = toolbox.population(n=n_individuals)

    # Fitness is reused ONLY across generations within this round, never across rounds.
    # A transfer-learning population carries fitness measured against the previous round's
    # client weights, which are now stale, so invalidate every individual here: the first
    # generation re-evaluates the whole population fresh against the current weights.
    for ind in population:
        if ind.fitness.valid:
            del ind.fitness.values

    best_fitness = -float('inf')
    no_improvement_generations = 0
    # A generation counts as "no improvement" unless the best fitness strictly improves
    # by more than this margin. Kept small so any real accuracy gain (granular by 1/N_val)
    # resets the counter, while float noise does not. Accuracy fitness is on a 0-1 scale
    # (see evaluate()); 1e-4 sits well below one sample's worth on any realistic val set.
    improvement_eps = 1e-4

    # Build the working model once: every individual reuses it via load_state_dict,
    # so we avoid deepcopying the architecture on each evaluation. The original
    # model_template is left untouched (the caller overwrites it with the best
    # aggregation afterwards).
    working_model = copy.deepcopy(model_template)

    # On CUDA, evaluate the whole population in one vectorized forward pass (vmap).
    # Elsewhere (CPU/MPS) the fused path gives no benefit, so fall back to the
    # per-individual sequential evaluation.
    use_batched = torch.device(device).type == "cuda"
    if use_batched:
        logging.info("Using batched (vmap) population evaluation on CUDA")

    gen = 0
    while gen < n_generations and (patience is None or no_improvement_generations < patience):
        # Only evaluate individuals whose fitness was invalidated: freshly created ones,
        # and offspring changed by crossover/mutation this generation. Elites and unchanged
        # offspring keep the fitness measured earlier this round, since the client weights
        # (and thus the aggregation each program produces) are fixed for the whole evolution.
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        logging.info(
            f"Generation {gen} - Evaluating {len(invalid_ind)}/{len(population)} individuals "
            f"({len(population) - len(invalid_ind)} reused from previous generation)"
        )

        if invalid_ind:
            fitnesses = None
            if use_batched:
                try:
                    fitnesses = batched_eval(invalid_ind, client_weights, test_dataset, toolbox, working_model, device)
                except Exception as exc:
                    # A custom model can use ops vmap cannot trace; the sequential
                    # path gives the same fitnesses on the same device, just slower.
                    logging.warning(
                        f"Batched (vmap) evaluation failed ({type(exc).__name__}: {exc}). "
                        "Falling back to sequential evaluation for the rest of this run."
                    )
                    use_batched = False
            if fitnesses is None:
                fitnesses = list(map(lambda ind: toolbox.evaluate(ind, client_weights, test_dataset, toolbox, working_model, device), invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                if isinstance(fit, (int, float)):
                    ind.fitness.values = (fit,)
                else:
                    logging.warning(f"Invalid fitness value {fit} for individual {ind}. Assigning worst fitness.")
                    ind.fitness.values = (_worst_fitness(),)

        # Persist each generation's individuals and fitness values for inspection.
        time = datetime.now()
        with open(os.path.join(save_path, f"individuals_gen_{gen}_{time.timestamp()}.txt"), "w") as f:
            for ind in population:
                f.write(f"{ind}, {ind.fitness.values}\n")

        elite_individuals = tools.selBest(population, elitism_size)
        best_individual = elite_individuals[0]
        logging.info(
            f"Generation {gen} - Best Individual: {best_individual}, Fitness: {best_individual.fitness.values[0]}")

        # Increment on a non-improving generation and reset only on a real improvement.
        # Elitism makes the best fitness monotonically non-decreasing, so a plateau (no
        # strict gain) is exactly what should advance the patience counter toward early stop.
        if best_individual.fitness.values[0] > best_fitness + improvement_eps:
            best_fitness = best_individual.fitness.values[0]
            no_improvement_generations = 0
        else:
            no_improvement_generations += 1

        offspring = toolbox.select(population, len(population) - elitism_size)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_rate:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        population[:] = offspring + elite_individuals
        gen += 1

    if patience is not None and no_improvement_generations >= patience:
        logging.info(f"Early stopping at generation {gen}, fitness {best_fitness}")

    logging.info("Evolution completed")
    return tools.selBest(population, 1)[0], population


def main(global_model, client_weights, test_dataset, individuals, generations, elitism_size, mutation_rate, crossover_rate, save_path, gp_patience, device, mutation_type, selection_type, min_tree_size, max_tree_size, available_primitives, crossover_type, gp_inizialization, gp_transfer_learning=False, old_population=None, seed=None, ephemeral_const_range=(-2.0, 2.0), weight_magnitude_limit=1e6, mutation_subtree_maxsize=2, gp_fitness_metric="accuracy"):
    """Run the GP aggregation pipeline and apply the best program to the global model.

    Args:
        seed: Random seed applied here (Python/NumPy/Torch) so the evolution is reproducible.
            ``None`` leaves the ambient global RNG state untouched.
        ephemeral_const_range: Inclusive (low, high) interval for the GP ephemeral constants.
        weight_magnitude_limit: Individuals whose aggregated weights exceed this absolute
            value (or are non-finite) are discarded with fitness 0. Must be positive.
        mutation_subtree_maxsize: Max height of the subtree grafted on mutUniform mutation.
        gp_fitness_metric: one of accuracy/precision/recall/f1 (maximized) or loss (minimized).
            The GP maximizes internally, representing loss as -loss. precision/recall/f1 are
            macro-averaged; on the batched path they come from a vectorized confusion matrix.
    """
    global _WEIGHT_MAGNITUDE_LIMIT, _FITNESS_METRIC
    limit = float(weight_magnitude_limit)
    _WEIGHT_MAGNITUDE_LIMIT = limit if (limit == limit and limit > 0) else 1e6
    metric = str(gp_fitness_metric).strip().lower()
    _FITNESS_METRIC = metric if metric in _ALLOWED_FITNESS_METRICS else "accuracy"

    # Seed the global RNG (DEAP draws from Python's random module; Torch drives evaluation)
    # so this GP run is reproducible. The caller passes a per-round seed derived from the
    # experiment seed, keeping rounds both deterministic and distinct.
    if seed is not None:
        set_global_seed(seed)

    toolbox, pset = initialize_deap(len(client_weights), mutation_type, selection_type, min_tree_size,
                                    max_tree_size,
                                    available_primitives, crossover_type, gp_inizialization,
                                    ephemeral_const_range=ephemeral_const_range,
                                    mutation_subtree_maxsize=mutation_subtree_maxsize)
    toolbox.register("evaluate", eval_individual)

    # Disable transfer learning automatically when there is no previous population.
    if old_population is None:
        logging.info("Cold start detected: disabling transfer learning")
        gp_transfer_learning = None

    # Select the requested population-initialization strategy.
    match gp_transfer_learning:

        case False | None:
            logging.info("FedGP standard (random initialization)")
            population = None

        case "full_reuse":
            logging.info("FedGP-TL-Full: full population reuse")
            population = old_population

        case "elite_reuse":
            logging.info("FedGP-TL-Elite: elite + random")
            elites = tools.selBest(old_population, elitism_size)
            population = init_elite_random(
                elites,
                individuals,
                toolbox
            )

        case "elite_mutation_warm_start":
            logging.info("FedGP-TL-Mut: elite mutation warm-start")
            elites = tools.selBest(old_population, elitism_size)
            population = init_elite_mutation(
                elites,
                individuals,
                toolbox
            )

        case "hybrid":
            logging.info("FedGP-TL-Hybrid: elite + mutation + random")
            elites = tools.selBest(old_population, elitism_size)
            population = init_hybrid_population(
                elites,
                individuals,
                toolbox,
                alpha=0.5
            )

        case _:
            raise ValueError(
                f"Invalid value for gp_transfer_learning: {gp_transfer_learning}"
            )

    # Run GP evolution from the selected initial population.
    best_individual, population = run_evolution(
        client_weights,
        test_dataset,
        toolbox,
        global_model,
        individuals,
        generations,
        elitism_size,
        mutation_rate,
        crossover_rate,
        save_path,
        gp_patience,
        device,
        population=population,
        seed=seed
    )

    func = toolbox.compile(expr=best_individual)
    weight_keys = list(client_weights[0].keys())
    aggregated_weights = {}

    for key in weight_keys:
        client_tensors = [client_weights[i][key].float() for i in range(len(client_weights))]
        combined_tensor = func(*client_tensors)
        aggregated_weights[key] = combined_tensor

    try:
        global_model.load_state_dict(aggregated_weights)
    except RuntimeError as e:
        logging.error(f"Error loading aggregated weights: {e}")

    logging.info(f"Best individual: {best_individual}, Fitness: {best_individual.fitness.values}")
    return global_model, best_individual, population
