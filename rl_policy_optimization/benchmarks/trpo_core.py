from __future__ import annotations
"""Low-level optimization utilities for native TRPO.

This module contains reusable helpers used by the native TRPO implementation:
- flattening/unflattening model parameters,
- flattening gradients safely,
- conjugate-gradient solver for Fisher system solves,
- backtracking line search for conservative policy updates.

The goal is to keep TRPO math isolated from environment-specific training code.
"""

import torch


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    """Return all model parameters as one contiguous 1D tensor.

    TRPO works in "parameter vector" space when solving trust-region steps.
    This helper converts a structured parameter set (many tensors of different
    shapes) into a single vector representation.
    """
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def set_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
    """Write a flat parameter vector back into a model in-place.

    The function iterates through model parameters in their defined order and
    slices the incoming vector into matching chunks.
    """
    # Cursor over the flat vector while copying each chunk into each parameter.
    offset = 0
    for param in model.parameters():
        # Number of scalars in the current parameter tensor.
        size = param.numel()
        # Restore original shape and copy into parameter storage.
        param.data.copy_(flat[offset:offset + size].view_as(param))
        # Advance cursor for next parameter tensor.
        offset += size


def flat_grad(grads: tuple[torch.Tensor | None, ...], params: tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
    """Flatten a tuple of per-parameter gradients into a single vector.

    Some parameters may produce `None` gradients in certain autograd calls.
    For TRPO math we still need a dense vector of fixed length, so `None`
    entries are replaced by explicit zeros with matching shape.
    """
    # Gather all gradient blocks as 1D chunks, then concatenate.
    pieces: list[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            # Keep vector dimensionality stable even when gradient is missing.
            pieces.append(torch.zeros_like(param).view(-1))
        else:
            # Ensure contiguous memory before flattening.
            pieces.append(grad.contiguous().view(-1))
    return torch.cat(pieces)


def conjugate_gradients(
    avp_fn,
    b: torch.Tensor,
    nsteps: int,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    """Approximately solve A x = b via conjugate gradients.

    In TRPO, `A` is the Fisher information matrix (or damped approximation)
    and is never materialized explicitly. Instead we provide `avp_fn(v)` that
    computes matrix-vector products A·v.

    This iterative solver avoids expensive dense linear algebra and typically
    converges in a small number of steps.
    """
    # Initial guess x=0.
    x = torch.zeros_like(b)
    # Initial residual r=b-Ax=b since x starts at zero.
    residual = b.clone()
    # Initial search direction p=r.
    direction = b.clone()
    # Squared residual norm used for step scaling and convergence checks.
    residual_dot = torch.dot(residual, residual)

    for _ in range(nsteps):
        # A·p via implicit matrix-vector product callback.
        avp = avp_fn(direction)
        # Stabilized denominator for alpha.
        denom = torch.dot(direction, avp) + 1e-8
        # Optimal step size along current search direction.
        alpha = residual_dot / denom
        # Update solution estimate.
        x = x + alpha * direction
        # Update residual.
        residual = residual - alpha * avp
        new_residual_dot = torch.dot(residual, residual)
        # Stop early if residual norm is sufficiently small.
        if new_residual_dot.item() < residual_tol:
            break
        # Conjugate direction mixing coefficient.
        beta = new_residual_dot / (residual_dot + 1e-12)
        # New conjugate direction.
        direction = residual + beta * direction
        residual_dot = new_residual_dot

    return x


def backtracking_line_search(
    policy: torch.nn.Module,
    prev_params: torch.Tensor,
    full_step: torch.Tensor,
    evaluate,
    max_backtracks: int = 10,
    step_fraction: float = 0.5,
    accept_ratio: float = 0.1,
) -> tuple[bool, torch.Tensor]:
    """Perform backtracking line search from `prev_params` along `full_step`.

    Parameters
    ----------
    policy:
        Model whose parameters are temporarily updated while probing candidates.
    prev_params:
        Original flattened parameters before proposed TRPO step.
    full_step:
        Full candidate step direction and magnitude.
    evaluate:
        Callable returning `(loss, kl)` under current model parameters.

    Returns
    -------
    (success, params):
        Whether a candidate was accepted and the accepted parameter vector.
    """
    # Baseline objective value at original parameters.
    base_loss, _ = evaluate()

    for i in range(max_backtracks):
        # Geometric step shrinkage: 1, 1/2, 1/4, 1/8, ...
        frac = step_fraction**i
        candidate = prev_params + frac * full_step
        # Temporarily move policy parameters to candidate point.
        set_flat_params(policy, candidate)
        new_loss, new_kl = evaluate()

        # Positive value means objective improved.
        actual_improve = (base_loss - new_loss).item()
        # Simple expectation heuristic for quality gate.
        expected_improve = abs(base_loss.item()) * frac
        improve_ratio = actual_improve / (expected_improve + 1e-8)

        # Accept if improvement is positive, sufficiently large relative to
        # expectation, and KL value is finite.
        if actual_improve > 0.0 and improve_ratio > accept_ratio and torch.isfinite(new_kl):
            return True, candidate

    # If all candidates fail, restore original parameters.
    set_flat_params(policy, prev_params)
    return False, prev_params
