"""Exact coefficients for the two-witness latent-evidence objective.

For one question, let ``logits[i] = log u_i`` be the detached log importance
evidence of the i-th independently sampled latent trace.  The two-witness
finite-support objective is

    L_2 = 1/2 log[ choose(S, 2)^-1 sum_{i < j} u_i u_j ].

Its derivative with respect to ``log u_i`` is one half of the posterior mass
of every unordered pair containing i.  These coefficients are nonnegative,
sum to one, and cannot exceed one half.  The implementation keeps independent
draw indices distinct even when two draws decode to the same text.
"""

from __future__ import annotations

import math

import torch


def _finite_pair_indices(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return original-row indices and local unordered pairs for finite logits."""

    if logits.ndim != 1:
        raise ValueError("two-witness logits must be one-dimensional")
    finite = torch.isfinite(logits)
    valid = finite | torch.isneginf(logits)
    if not bool(valid.all()):
        raise ValueError("two-witness logits must be finite or -inf exclusions")
    finite_indices = torch.nonzero(finite, as_tuple=False).flatten()
    if int(finite_indices.numel()) < 2:
        raise ValueError("two-witness evidence requires at least two finite draws")
    pairs = torch.triu_indices(
        int(finite_indices.numel()),
        int(finite_indices.numel()),
        offset=1,
        device=logits.device,
    )
    return finite_indices, pairs


def two_witness_log_objective(logits: torch.Tensor) -> torch.Tensor:
    """Evaluate the normalized two-witness log objective for one support."""

    finite_indices, pairs = _finite_pair_indices(logits)
    local = logits[finite_indices]
    pair_logits = local[pairs[0]] + local[pairs[1]]
    pair_count = int(pair_logits.numel())
    return 0.5 * (
        torch.logsumexp(pair_logits, dim=0) - math.log(pair_count)
    )


def two_witness_responsibilities(logits: torch.Tensor) -> torch.Tensor:
    """Return the exact complete-data coefficients induced by the objective."""

    finite_indices, pairs = _finite_pair_indices(logits)
    local = logits[finite_indices]
    pair_logits = local[pairs[0]] + local[pairs[1]]
    pair_mass = torch.softmax(pair_logits, dim=0)
    local_weights = torch.zeros_like(local)
    half_pair_mass = 0.5 * pair_mass
    local_weights.scatter_add_(0, pairs[0], half_pair_mass)
    local_weights.scatter_add_(0, pairs[1], half_pair_mass)

    weights = torch.zeros_like(logits)
    weights[finite_indices] = local_weights
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("two-witness coefficients became non-finite")
    if bool((weights < 0).any()):
        raise RuntimeError("two-witness coefficients became negative")
    if not torch.allclose(
        weights.sum(),
        torch.ones((), dtype=weights.dtype, device=weights.device),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError("two-witness coefficients do not sum to one")
    if float(weights.max()) > 0.5 + 1e-6:
        raise RuntimeError("two-witness coefficient exceeded its one-half bound")
    return weights
