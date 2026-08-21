"""Finite-support posterior weights used by Q5 and PIS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


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


def q5_weights(
    trace_logp: Sequence[float],
    answer_logp: Sequence[float],
    question_ids: Sequence[object],
    active: Sequence[bool] | None = None,
) -> np.ndarray:
    """Q5 responsibilities on its persistent finite support."""

    trace = _vector(trace_logp, "trace_logp")
    answer = _vector(answer_logp, "answer_logp")
    if trace.shape != answer.shape:
        raise ValueError("trace_logp and answer_logp must align")
    return _group_softmax(trace + answer, question_ids, active)


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
