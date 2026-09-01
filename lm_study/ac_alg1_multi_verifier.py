r"""Selective responsibilities for a latent multi-verifier answer marginal.

The ordinary null-latent update augments a sampled rationale support with one
frozen no-update state.  This module adds source-specific validity variables
without turning verifier outputs into rewards or ad-hoc loss multipliers.

For detached log importance terms ``ell_i = log u_i`` and a posterior
probability ``g_i`` that all active verifier properties hold for trace ``i``,
the sampled marginal is

.. math::

    \widehat Z_E = \pi_0 e^{b_0}
      + \frac{1-\pi_0}{S}\sum_i
        \left[g_i e^{\ell_i} + (1-g_i)e^{b_0}\right].

At the runtime E-step the proposal is the current rationale prior, so its
rationale density cancels and ``ell_i`` is the strict-EOS answer log
likelihood.  The first term is the existing global null state.  In each
trace-specific mixture, the jointly valid route uses the model importance term
and the invalid route uses the same frozen null evidence.  Only the valid route
is trainable.  Therefore its exact generalized-EM coefficient is

.. math::

    c_i = \frac{(1-\pi_0)S^{-1}g_i e^{\ell_i}}{\widehat Z_E}.

The coefficients need not sum to one.  Their missing mass is the posterior
probability of either the global null state or a verifier-invalid route and
receives no M-step gradient.  Setting every ``g_i`` to one recovers the
ordinary null-latent update exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


VERIFIER_OBSERVATIONS = ("pass", "fail", "missing")
MULTI_VERIFIER_POSTERIORS = (
    "verifier_arithmetic",
    "verifier_graph",
    "verifier_joint",
    "verifier_joint_shuffled",
)


@dataclass(frozen=True)
class MultiVerifierResponsibilities:
    """Detached posterior quantities for one sampled question support."""

    m_step_coefficients: torch.Tensor
    conditional_valid_trace_weights: torch.Tensor
    validity_probabilities: torch.Tensor
    real_coverage: float
    null_mass: float
    global_null_mass: float
    verifier_invalid_mass: float
    valid_log_mean_evidence: float | None
    sampled_log_objective: float
    null_log_evidence: float
    null_prior: float


def _as_vector(
    values: Sequence[float] | torch.Tensor,
    *,
    name: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    tensor = (
        torch.as_tensor(values, device=device)
        .detach()
        .to(dtype=torch.float64)
        .clone()
    )
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return tensor


def _validate_inputs(
    real_log_evidence: Sequence[float] | torch.Tensor,
    validity_probabilities: Sequence[float] | torch.Tensor,
    *,
    null_log_evidence: float,
    null_prior: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    logits = _as_vector(real_log_evidence, name="real_log_evidence")
    validity = _as_vector(
        validity_probabilities,
        name="validity_probabilities",
        device=logits.device,
    )
    if logits.shape != validity.shape:
        raise ValueError("evidence logits and validity probabilities must align")
    if int(logits.numel()) == 0:
        raise ValueError("multi-verifier evidence requires at least one trace")
    if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()):
        raise ValueError("real_log_evidence must be finite or -inf")
    if not bool(torch.isfinite(validity).all()):
        raise ValueError("validity probabilities must be finite")
    if bool(((validity < 0.0) | (validity > 1.0)).any()):
        raise ValueError("validity probabilities must lie in [0, 1]")
    baseline = float(null_log_evidence)
    if not math.isfinite(baseline):
        raise ValueError("null_log_evidence must be finite")
    prior = float(null_prior)
    if not math.isfinite(prior) or not 0.0 < prior < 1.0:
        raise ValueError("null_prior must lie strictly between zero and one")
    return logits, validity, baseline, prior


def _log_probability(probability: torch.Tensor) -> torch.Tensor:
    values = torch.full_like(probability, float("-inf"))
    positive = probability > 0.0
    values[positive] = torch.log(probability[positive])
    return values


def _json_safe_valid_log_mean(value: float) -> float | None:
    """Encode only the legitimate negative-infinite evidence boundary."""

    value = float(value)
    if math.isfinite(value):
        return value
    if value == -math.inf:
        return None
    raise ValueError("valid log-mean evidence became NaN or positive infinity")


def _route_log_masses(
    real_log_evidence: Sequence[float] | torch.Tensor,
    validity_probabilities: Sequence[float] | torch.Tensor,
    *,
    null_log_evidence: float,
    null_prior: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    logits, validity, baseline, prior = _validate_inputs(
        real_log_evidence,
        validity_probabilities,
        null_log_evidence=null_log_evidence,
        null_prior=null_prior,
    )
    count = int(logits.numel())
    real_log_prior = math.log1p(-prior) - math.log(count)
    valid_log_mass = real_log_prior + _log_probability(validity) + logits
    invalid_log_mass = (
        real_log_prior
        + _log_probability(1.0 - validity)
        + baseline
    )
    global_null_log_mass = torch.tensor(
        math.log(prior) + baseline,
        dtype=logits.dtype,
        device=logits.device,
    )
    all_masses = torch.cat(
        [valid_log_mass, invalid_log_mass, global_null_log_mass.reshape(1)]
    )
    log_normalizer = torch.logsumexp(all_masses, dim=0)
    return (
        logits,
        validity,
        valid_log_mass,
        invalid_log_mass,
        float(global_null_log_mass.item()),
        float(log_normalizer.item()),
    )


def multi_verifier_log_objective(
    real_log_evidence: torch.Tensor,
    validity_probabilities: Sequence[float] | torch.Tensor,
    *,
    null_log_evidence: float,
    null_prior: float,
) -> torch.Tensor:
    """Return the differentiable sampled log marginal for a gradient audit."""

    if real_log_evidence.ndim != 1:
        raise ValueError("real_log_evidence must be one-dimensional")
    validity = torch.as_tensor(
        validity_probabilities,
        dtype=real_log_evidence.dtype,
        device=real_log_evidence.device,
    ).detach()
    if validity.shape != real_log_evidence.shape:
        raise ValueError("evidence logits and validity probabilities must align")
    if not bool(torch.isfinite(validity).all()) or bool(
        ((validity < 0.0) | (validity > 1.0)).any()
    ):
        raise ValueError("validity probabilities must be finite and in [0, 1]")
    baseline = float(null_log_evidence)
    prior = float(null_prior)
    if not math.isfinite(baseline):
        raise ValueError("null_log_evidence must be finite")
    if not math.isfinite(prior) or not 0.0 < prior < 1.0:
        raise ValueError("null_prior must lie strictly between zero and one")
    count = int(real_log_evidence.numel())
    if count == 0:
        raise ValueError("multi-verifier evidence requires at least one trace")
    log_validity = _log_probability(validity)
    log_invalidity = _log_probability(1.0 - validity)
    real_log_prior = math.log1p(-prior) - math.log(count)
    valid = real_log_prior + log_validity + real_log_evidence
    invalid = real_log_prior + log_invalidity + baseline
    global_null = real_log_evidence.new_tensor(math.log(prior) + baseline)
    return torch.logsumexp(torch.cat([valid, invalid, global_null.reshape(1)]), dim=0)


def multi_verifier_responsibilities(
    real_log_evidence: Sequence[float] | torch.Tensor,
    validity_probabilities: Sequence[float] | torch.Tensor,
    *,
    null_log_evidence: float,
    null_prior: float,
) -> MultiVerifierResponsibilities:
    """Compute exact detached valid-route coefficients and null decomposition."""

    (
        logits,
        validity,
        valid_log_mass,
        invalid_log_mass,
        global_null_log_mass,
        log_normalizer,
    ) = _route_log_masses(
        real_log_evidence,
        validity_probabilities,
        null_log_evidence=null_log_evidence,
        null_prior=null_prior,
    )
    valid_coefficients = torch.exp(valid_log_mass - log_normalizer)
    invalid_mass = float(
        torch.exp(invalid_log_mass - log_normalizer).sum().item()
    )
    global_null_mass = math.exp(global_null_log_mass - log_normalizer)
    real_coverage = float(valid_coefficients.sum().item())
    null_mass = invalid_mass + global_null_mass
    if not math.isclose(real_coverage + null_mass, 1.0, abs_tol=1e-10):
        raise RuntimeError("multi-verifier posterior mass does not sum to one")

    conditional = torch.zeros_like(valid_coefficients)
    if real_coverage > 0.0:
        conditional = valid_coefficients / real_coverage

    valid_terms = _log_probability(validity) + logits
    valid_log_mean = float(
        torch.logsumexp(valid_terms, dim=0).item() - math.log(len(logits))
    )
    # ``-inf`` is the exact value when no sampled trace has a positive valid
    # route.  It is a legitimate posterior diagnostic, not a numerical
    # failure, but strict JSON deliberately rejects non-finite floats.  Store
    # that boundary as null.  The saved trace logits and validity observations
    # make the representation reversible, and the frozen analyser independently
    # reconstructs ``-inf`` before accepting the null value.
    json_valid_log_mean = _json_safe_valid_log_mean(valid_log_mean)
    return MultiVerifierResponsibilities(
        m_step_coefficients=valid_coefficients.detach(),
        conditional_valid_trace_weights=conditional.detach(),
        validity_probabilities=validity.detach(),
        real_coverage=real_coverage,
        null_mass=null_mass,
        global_null_mass=global_null_mass,
        verifier_invalid_mass=invalid_mass,
        valid_log_mean_evidence=json_valid_log_mean,
        sampled_log_objective=log_normalizer,
        null_log_evidence=float(null_log_evidence),
        null_prior=float(null_prior),
    )


def observation_validity_probability(observation: str) -> float:
    """Map an exact verifier observation to its latent-validity probability.

    A supported deterministic pass/fail reveals the operational property.
    ``missing`` means the verifier could not parse the property and is
    marginalised under a maximum-entropy Bernoulli prior.
    """

    if observation == "pass":
        return 1.0
    if observation == "fail":
        return 0.0
    if observation == "missing":
        return 0.5
    raise ValueError(f"unknown verifier observation {observation!r}")


def joint_validity_probability(
    arithmetic_observation: str,
    graph_observation: str,
    *,
    posterior: str,
) -> float:
    """Return the valid-route probability selected by a registered arm."""

    if posterior not in MULTI_VERIFIER_POSTERIORS:
        raise ValueError(f"unknown multi-verifier posterior {posterior!r}")
    arithmetic = observation_validity_probability(arithmetic_observation)
    graph = observation_validity_probability(graph_observation)
    if posterior == "verifier_arithmetic":
        return arithmetic
    if posterior == "verifier_graph":
        return graph
    return arithmetic * graph
