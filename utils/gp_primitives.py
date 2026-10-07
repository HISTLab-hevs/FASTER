"""Device-safe primitive operations used in FedGP genetic-programming trees."""

import math
import torch

# Ephemeral constants enter the GP tree as plain Python floats. They broadcast fine
# through binary ops (torch.add(tensor, 0.3), ...) and stay device-agnostic, but the
# unary torch.* ops reject a bare float. The wrappers below therefore evaluate scalars
# in Python (keeping them as device-free floats) and only call torch on real tensors,
# so a subtree like mul(torch_abs(-0.43), CLIENT1) == 0.43 * CLIENT1 works on any device.

def torch_protected_div(left, right, epsilon=1e-10):
    """Divide two operands while avoiding division by zero with an epsilon."""
    right = right + epsilon
    return left / right

def torch_protected_sqrt(x):
    """Compute a square root after clamping negative values to zero (scalar-safe)."""
    if torch.is_tensor(x):
        return torch.sqrt(torch.where(x < 0, torch.zeros_like(x), x))
    return math.sqrt(x) if x >= 0 else 0.0

def torch_abs(x):
    """Elementwise absolute value (scalar-safe)."""
    return torch.abs(x) if torch.is_tensor(x) else abs(x)

def torch_log(x):
    """Elementwise natural log (scalar-safe); non-positive scalars map to 0.0."""
    if torch.is_tensor(x):
        return torch.log(x)
    return math.log(x) if x > 0 else 0.0

def torch_sin(x):
    """Elementwise sine (scalar-safe)."""
    return torch.sin(x) if torch.is_tensor(x) else math.sin(x)

def torch_cos(x):
    """Elementwise cosine (scalar-safe)."""
    return torch.cos(x) if torch.is_tensor(x) else math.cos(x)

def _coerce_and_broadcast(tensors):
    """Turn a mix of tensors and scalar constants into broadcast-compatible tensors.

    Ephemeral constants enter the tree as Python floats, so a reduction such as
    ``mean(CLIENT1, CLIENT2, 0.3)`` would otherwise fail in ``torch.stack``. Scalars are
    materialized on the same device/dtype as the tensor operands and broadcast to a
    common shape so they can be stacked.
    """
    ref = next((t for t in tensors if torch.is_tensor(t)), None)
    device = ref.device if ref is not None else "cpu"
    materialized = [
        t.float() if torch.is_tensor(t) else torch.as_tensor(float(t), dtype=torch.float32, device=device)
        for t in tensors
    ]
    return torch.broadcast_tensors(*materialized)

def torch_mean(*tensors):
    """Compute the elementwise mean across one or more tensors (scalar-safe)."""
    stacked_tensors = torch.stack(_coerce_and_broadcast(tensors), dim=0)
    return stacked_tensors.mean(dim=0)

def torch_median(*tensors):
    """Compute the elementwise median across one or more tensors (scalar-safe)."""
    stacked_tensors = torch.stack(_coerce_and_broadcast(tensors), dim=0)
    return stacked_tensors.median(dim=0).values

def torch_pow(base, exponent):
    """Raise ``base`` to ``exponent`` elementwise (deterministic).

    The exponent is supplied by the GP tree itself — an ephemeral constant or another
    subtree — so a given individual always applies the same power. This replaces the
    previous version, which sampled a fresh random exponent on every call: that broke
    reproducibility (the same individual scored differently across evaluations and
    diverged between the GP search and the final application) and made the batched and
    sequential evaluation paths disagree.
    """
    if not torch.is_tensor(base) and not torch.is_tensor(exponent):
        # Both scalar: a fractional power of a negative base would be complex/NaN, so
        # fall back to 0.0 for those degenerate (client-independent) subtrees.
        try:
            result = float(base) ** float(exponent)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return result if isinstance(result, float) and math.isfinite(result) else 0.0
    return torch.pow(base, exponent)
