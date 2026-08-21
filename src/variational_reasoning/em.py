"""Finite-support posterior weights used by Q5 and PIS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class NullLatentPosterior:
    """Posterior mass on real traces and one fixed null state."""

    weights: np.ndarray
    conditional_weights: np.ndarray
    null_mass: float

    @property
    def real_coverage(self) -> float:
        return 1.0 - self.null_mass


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _groups(question_ids: Sequence[object], size: int) -> np.ndarray:
    groups = np.asarray(question_ids)
    if groups.ndim != 1 or len(groups) != size:
        raise ValueError("question_ids must be a one-dimensional aligned array")
    return groups


def _mask(active: Sequence[bool] | None, size: int) -> np.ndarray:
    if active is None:
        return np.ones(size, dtype=bool)
    mask = np.asarray(active, dtype=bool)
    if mask.ndim != 1 or len(mask) != size:
        raise ValueError("active must be a one-dimensional aligned array")
    return mask


def _group_softmax(
    logits: Sequence[float],
    question_ids: Sequence[object],
    active: Sequence[bool] | None = None,
) -> np.ndarray:
    """Apply a stable softmax independently inside each question group."""

    values = _vector(logits, "logits")
    groups = _groups(question_ids, len(values))
    keep = _mask(active, len(values))
    if not np.isfinite(values[keep]).all():
        raise ValueError("active logits must be finite")

    weights = np.zeros_like(values)
    for group in dict.fromkeys(groups.tolist()):
        local = (groups == group) & keep
        if not local.any():
            continue
        centred = values[local] - values[local].max()
        mass = np.exp(centred)
        weights[local] = mass / mass.sum()
    return weights


def pis_weights(
    answer_logp: Sequence[float],
    question_ids: Sequence[object],
    active: Sequence[bool] | None = None,
) -> np.ndarray:
    """PIS responsibilities for fresh draws from the current question prior.

    The proposal is the current rationale prior, so its density cancels from
    the importance ratio. Only the strict gold-answer-plus-EOS evidence remains.
    """

    return _group_softmax(answer_logp, question_ids, active)


def joint_weights(
    trace_logp: Sequence[float],
    answer_logp: Sequence[float],
    question_ids: Sequence[object],
    active: Sequence[bool] | None = None,
) -> np.ndarray:
    """Joint finite-support posterior used by Q5, VIN, and VOUT."""

    trace = _vector(trace_logp, "trace_logp")
    answer = _vector(answer_logp, "answer_logp")
    if trace.shape != answer.shape:
        raise ValueError("trace_logp and answer_logp must align")
    return _group_softmax(trace + answer, question_ids, active)


def q5_weights(
    trace_logp: Sequence[float],
    answer_logp: Sequence[float],
    question_ids: Sequence[object],
    active: Sequence[bool] | None = None,
) -> np.ndarray:
    """Q5 name for the joint posterior on its persistent support."""

    return joint_weights(trace_logp, answer_logp, question_ids, active)


def importance_weights(
    trace_logp: Sequence[float],
    answer_logp: Sequence[float],
    proposal_logp: Sequence[float],
    question_ids: Sequence[object],
    active: Sequence[bool] | None = None,
) -> np.ndarray:
    """Self-normalised weights for an off-policy rationale proposal."""

    trace = _vector(trace_logp, "trace_logp")
    answer = _vector(answer_logp, "answer_logp")
    proposal = _vector(proposal_logp, "proposal_logp")
    if not (trace.shape == answer.shape == proposal.shape):
        raise ValueError("trace, answer, and proposal log probabilities must align")
    return _group_softmax(trace + answer - proposal, question_ids, active)


def uniform_weights(
    question_ids: Sequence[object],
    active: Sequence[bool] | None = None,
) -> np.ndarray:
    """Uniform empirical credit within each non-empty question support."""

    groups = np.asarray(question_ids)
    if groups.ndim != 1:
        raise ValueError("question_ids must be one-dimensional")
    return _group_softmax(np.zeros(len(groups)), groups, active)


def weighted_joint_loss(
    trace_logp: Sequence[float],
    answer_logp: Sequence[float],
    weights: Sequence[float],
    question_ids: Sequence[object],
) -> float:
    """Negative detached-responsibility joint objective, averaged by question."""

    trace = _vector(trace_logp, "trace_logp")
    answer = _vector(answer_logp, "answer_logp")
    weight = _vector(weights, "weights")
    if not (trace.shape == answer.shape == weight.shape):
        raise ValueError("log probabilities and weights must align")
    groups = _groups(question_ids, len(trace))
    if (weight < 0).any() or not np.isfinite(weight).all():
        raise ValueError("weights must be finite and non-negative")

    per_question = []
    for group in dict.fromkeys(groups.tolist()):
        local = groups == group
        mass = weight[local].sum()
        if mass == 0:
            continue
        if not np.isclose(mass, 1.0, atol=1e-8):
            raise ValueError("weights must sum to one in each active question")
        per_question.append(np.sum(weight[local] * (trace[local] + answer[local])))
    if not per_question:
        return 0.0
    return -float(np.mean(per_question))


def centered_trace_credit(weights: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return the centred rationale coefficients and ordinary answer weights.

    This is the simple credit ablation used in the focused pilot: rationale
    tokens receive ``q_i - 1/K`` while answer tokens retain ``q_i``.
    """

    posterior = _vector(weights, "weights")
    if len(posterior) == 0:
        raise ValueError("centered credit requires a non-empty support")
    if (posterior < 0).any() or not np.isfinite(posterior).all():
        raise ValueError("weights must be finite and non-negative")
    if not np.isclose(posterior.sum(), 1.0, atol=1e-8):
        raise ValueError("weights must sum to one")
    return posterior - 1.0 / len(posterior), posterior.copy()


def _softmax_allowing_infinities(logits: np.ndarray) -> np.ndarray:
    if np.isnan(logits).any():
        raise ValueError("logits cannot contain NaN")
    positive = np.isposinf(logits)
    if positive.any():
        return positive.astype(np.float64) / positive.sum()
    finite = np.isfinite(logits)
    if not finite.any():
        return np.zeros_like(logits)
    mass = np.zeros_like(logits)
    mass[finite] = np.exp(logits[finite] - logits[finite].max())
    return mass / mass.sum()


def null_latent_weights(
    real_log_evidence: Sequence[float],
    *,
    null_log_evidence: float,
    null_prior: float = 0.5,
    temperature: float = 1.0,
) -> NullLatentPosterior:
    """Add a frozen null state without renormalising away abstention.

    The returned real weights sum to ``1 - null_mass``. They are the M-step
    coefficients; ``conditional_weights`` are diagnostic only.
    """

    evidence = _vector(real_log_evidence, "real_log_evidence")
    if not 0.0 < null_prior < 1.0:
        raise ValueError("null_prior must lie strictly between zero and one")
    if not np.isfinite(null_log_evidence):
        raise ValueError("null_log_evidence must be finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if np.isnan(evidence).any():
        raise ValueError("real_log_evidence cannot contain NaN")
    if len(evidence) == 0:
        return NullLatentPosterior(np.empty(0), np.empty(0), 1.0)

    real_logits = (
        evidence / temperature
        + np.log1p(-null_prior)
        - np.log(len(evidence))
    )
    null_logit = np.log(null_prior) + null_log_evidence / temperature
    posterior = _softmax_allowing_infinities(
        np.concatenate((real_logits, [null_logit]))
    )
    conditional = _softmax_allowing_infinities(evidence / temperature)
    return NullLatentPosterior(
        weights=posterior[:-1],
        conditional_weights=conditional,
        null_mass=float(posterior[-1]),
    )


@dataclass
class UniqueFIFOSupport:
    """Token-unique FIFO support used by the persistent Q5 variants."""

    capacity: int
    items: list[tuple[int, ...]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.capacity, bool) or self.capacity < 1:
            raise ValueError("capacity must be a positive integer")

    def add(self, token_ids: Sequence[int]) -> bool:
        trace = tuple(int(token) for token in token_ids)
        if trace in self.items:
            return False
        self.items.append(trace)
        if len(self.items) > self.capacity:
            self.items.pop(0)
        return True
