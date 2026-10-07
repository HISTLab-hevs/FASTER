"""Federated aggregation strategies (FedAvg, FedAvgW, FedGP, FedProx, FedNova)."""

import torch

from utils import gp


def _match_buffer_dtype(aggregated, reference):
    """Cast a float-aggregated tensor back to ``reference``'s dtype.

    Parameters and float buffers (e.g. BatchNorm running stats) keep the float average.
    Integer buffers such as ``num_batches_tracked`` are rounded and cast back to their
    original dtype, instead of relying on ``load_state_dict``'s implicit truncating cast.
    """
    if torch.is_floating_point(reference):
        return aggregated
    return aggregated.round().to(reference.dtype)


def fed_avg(global_model, client_models_state_dicts):
    """Aggregate client weights with uniform averaging.

    Args:
        global_model: Global model updated in place.
        client_models_state_dicts: Client ``state_dict`` objects to aggregate.

    Returns:
        torch.nn.Module: The updated global model.
    """

    if not client_models_state_dicts:
        return global_model

    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        stacked_tensors = torch.stack(
            [
                client_state_dict[key].float()
                for client_state_dict in client_models_state_dicts
            ],
            dim=0,
        )
        global_dict[key] = _match_buffer_dtype(stacked_tensors.mean(dim=0), global_dict[key])

    global_model.load_state_dict(global_dict)
    return global_model


def fed_avgw(global_model, client_models_state_dicts, client_num_samples):
    """Aggregate client weights with sample-count weighting.

    Args:
        global_model: Global model updated in place.
        client_models_state_dicts: Client ``state_dict`` objects to aggregate.
        client_num_samples: Number of training samples owned by each client.

    Returns:
        torch.nn.Module: The updated global model.
    """

    num_models = len(client_models_state_dicts)
    num_sample_counts = len(client_num_samples)
    if num_models != num_sample_counts:
        raise ValueError(
            "fed_avgw metadata length mismatch: "
            f"received {num_models} client models but {num_sample_counts} sample counts"
        )

    valid_clients = [
        (client_state_dict, int(samples))
        for client_state_dict, samples in zip(client_models_state_dicts, client_num_samples)
        if int(samples) > 0
    ]
    if not valid_clients:
        raise ValueError(
            "fed_avgw received no valid client updates to aggregate "
            "(all sample counts were <= 0 or no participants were provided)"
        )

    client_models_state_dicts = [state_dict for state_dict, _ in valid_clients]
    client_num_samples = [samples for _, samples in valid_clients]
    global_dict = global_model.state_dict()

    total_samples = sum(client_num_samples)
    if total_samples <= 0:
        raise ValueError(
            "fed_avgw received no valid client updates to aggregate "
            "(total sample count after filtering is <= 0)"
        )

    weights = torch.tensor(
        [samples / total_samples for samples in client_num_samples], 
        dtype=torch.float32
    )

    for key in global_dict.keys():
        stacked_tensors = torch.stack(
            [
                client_state_dict[key].float()
                for client_state_dict in client_models_state_dicts
            ],
            dim=0,
        )

        view_shape = [-1] + [1] * (stacked_tensors.dim() - 1)
        w = weights.view(*view_shape).to(stacked_tensors.device)
        
        global_dict[key] = _match_buffer_dtype(torch.sum(stacked_tensors * w, dim=0), global_dict[key])

    global_model.load_state_dict(global_dict)
    return global_model


def fed_gp(
    global_model,
    client_models_state_dicts,
    test_dataset,
    individuals,
    generations,
    elitism_size,
    mutation_rate,
    crossover_rate,
    save_path,
    gp_patience,
    device,
    mutation_type,
    selection_type,
    min_tree_size,
    max_tree_size,
    available_primitives,
    crossover_type,
    gp_inizialization,
    gp_transfer_learning,
    old_population=None,
    seed=None,
    ephemeral_const_range=(-2.0, 2.0),
    weight_magnitude_limit=1e6,
    mutation_subtree_maxsize=2,
    gp_fitness_metric="accuracy",
):
    """Aggregate client weights with the GP-based strategy.

    The returned value is delegated to :func:`utils.gp.main`, which performs the
    actual evolutionary search and applies the best individual to the model.

    Args:
        seed: Random seed for reproducible GP evolution.
        ephemeral_const_range: Inclusive (low, high) interval for the GP ephemeral constants.
        weight_magnitude_limit: Aggregated-weight magnitude above which an individual is discarded.
        mutation_subtree_maxsize: Max height of the subtree grafted on mutUniform mutation.
        gp_fitness_metric: accuracy/precision/recall/f1 (maximized) or loss (minimized).
    """
    return gp.main(global_model, client_models_state_dicts, test_dataset, individuals, generations, elitism_size, mutation_rate, crossover_rate, save_path, gp_patience, device, mutation_type, selection_type, min_tree_size, max_tree_size, available_primitives, crossover_type, gp_inizialization, gp_transfer_learning, old_population, seed, ephemeral_const_range=ephemeral_const_range, weight_magnitude_limit=weight_magnitude_limit, mutation_subtree_maxsize=mutation_subtree_maxsize, gp_fitness_metric=gp_fitness_metric)


def fed_prox(global_model, client_models_state_dicts, client_num_samples=None, weighted=True):
    """Aggregate FedProx client weights.

    FedProx (Li et al., 2020) aggregates exactly like FedAvg — the method's contribution is
    the proximal term in client training (see ``utils/train.py``), not the aggregation rule.
    ``weighted`` selects the paper's sample-weighted average (n_k/n, via :func:`fed_avgw`);
    ``weighted=False`` uses a plain uniform average (:func:`fed_avg`). ``client_num_samples``
    is required when ``weighted`` is True.
    """
    if weighted:
        if client_num_samples is None:
            raise ValueError(
                "fed_prox weighted aggregation requires client_num_samples"
            )
        return fed_avgw(global_model, client_models_state_dicts, client_num_samples)
    return fed_avg(global_model, client_models_state_dicts)




def _fednova_local_coefficient(local_steps, momentum):
    """Effective number of normalized local steps (``||a_i||_1``) for FedNova.

    For vanilla SGD this is simply the number of local steps ``tau``. With SGD momentum
    ``rho`` each accumulated step contributes more, so the effective coefficient is
    ``(tau - rho * (1 - rho**tau) / (1 - rho)) / (1 - rho)``, which reduces to ``tau`` when
    ``rho == 0`` and equals 1 when ``tau == 1`` (FedNova, Wang et al. 2020). Ignoring the
    momentum correction biases the normalization whenever clients train with momentum > 0.
    """
    tau = float(local_steps)
    rho = float(momentum)
    if rho <= 0.0:
        return tau
    return (tau - rho * (1.0 - rho ** tau) / (1.0 - rho)) / (1.0 - rho)


def fed_nova(global_model, client_models_state_dicts, client_num_samples, client_local_steps, client_lrs, client_momentums):
    """Aggregate client updates with the FedNova normalization rule.

    ``client_momentums`` is the SGD momentum each client trained with; it enters the
    per-client normalization coefficient (see :func:`_fednova_local_coefficient`) so the
    normalization stays correct when local training uses momentum. Pass 0 for plain SGD.
    """
    num_models = len(client_models_state_dicts)
    num_sample_counts = len(client_num_samples)
    num_local_steps = len(client_local_steps)
    num_lrs = len(client_lrs)
    num_momentums = len(client_momentums)
    if not (num_models == num_sample_counts == num_local_steps == num_lrs == num_momentums):
        raise ValueError(
            "fed_nova metadata length mismatch: "
            f"models={num_models}, sample_counts={num_sample_counts}, "
            f"local_steps={num_local_steps}, learning_rates={num_lrs}, "
            f"momentums={num_momentums}"
        )

    valid_clients = [
        (
            client_models_state_dicts[k],
            int(client_num_samples[k]),
            float(client_local_steps[k]),
            float(client_lrs[k]),
            float(client_momentums[k]),
        )
        for k in range(num_models)
        if int(client_num_samples[k]) > 0 and float(client_local_steps[k]) > 0 and float(client_lrs[k]) > 0
    ]
    if not valid_clients:
        raise ValueError(
            "fed_nova received no valid client updates to aggregate "
            "(all clients had non-positive samples, local steps, or learning rates)"
        )

    client_models_state_dicts = [item[0] for item in valid_clients]
    client_num_samples = [item[1] for item in valid_clients]
    client_local_steps = [item[2] for item in valid_clients]
    client_lrs = [item[3] for item in valid_clients]
    client_momentums = [item[4] for item in valid_clients]
    global_dict = global_model.state_dict()
    num_clients = len(client_models_state_dicts)
    total_samples = sum(client_num_samples)
    if total_samples <= 0:
        raise ValueError(
            "fed_nova received no valid client updates to aggregate "
            "(total sample count after filtering is <= 0)"
        )

    # tau_i = ||a_i||_1 * lr_i: the momentum-aware step count scaled by the client lr.
    # With homogeneous lr the lr factor cancels between tau_eff and tau_k below (matching
    # the paper); it is kept here so heterogeneous learning rates are still accounted for.
    tau_is = [
        _fednova_local_coefficient(client_local_steps[k], client_momentums[k]) * client_lrs[k]
        for k in range(num_clients)
    ]
    if any(tau <= 0 for tau in tau_is):
        raise ValueError(
            "fed_nova received invalid client metadata after filtering "
            "(effective client step size must be > 0)"
        )

    p = [samples / total_samples for samples in client_num_samples]

    tau_eff = sum([p[k] * tau_is[k] for k in range(num_clients)])
    if tau_eff <= 0:
        raise ValueError(
            "fed_nova received invalid client metadata after filtering "
            "(effective global step size must be > 0)"
        )

    global_update = {
        key: torch.zeros_like(val, dtype=torch.float32) 
        for key, val in global_dict.items()
    }

    for k in range(num_clients):
        client_dict = client_models_state_dicts[k]
        tau_k = tau_is[k]
        p_k = p[k]

        scale_factor = tau_eff * (p_k / tau_k)

        for key in global_dict.keys():
            delta = client_dict[key].float() - global_dict[key].float()
            
            global_update[key] += delta * scale_factor

    for key in global_dict.keys():
        global_dict[key] = _match_buffer_dtype(
            global_dict[key].float() + global_update[key], global_dict[key]
        )

    global_model.load_state_dict(global_dict)
    return global_model
