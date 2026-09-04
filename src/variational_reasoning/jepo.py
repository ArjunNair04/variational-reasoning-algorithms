"""Numerical kernels for multi-sample Jensen evidence policy optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class JEPOMultisampleTerms:
    """Detached coefficients for one multi-sample JEPO update."""

    answer_weights: np.ndarray
    raw_trace_advantages: np.ndarray
    trace_advantages: np.ndarray
    active: np.ndarray
    active_groups: int


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
    keep = np.asarray(active, dtype=bool)
    if keep.ndim != 1 or len(keep) != size:
        raise ValueError("active must be a one-dimensional aligned array")
    return keep


def _logmeanexp(values: np.ndarray) -> float:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("logmeanexp requires a non-empty vector")
    maximum = float(values.max())
    return maximum + float(np.log(np.exp(values - maximum).mean()))


def standardize_and_clip(
    values: Sequence[float],
    *,
    active: Sequence[bool] | None = None,
    clip: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Divide active coefficients by their population standard deviation and clip."""

    vector = _vector(values, "values")
    keep = _mask(active, len(vector))
    if not np.isfinite(vector[keep]).all():
        raise ValueError("active values must be finite")
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    result = np.zeros_like(vector)
    if not keep.any():
        return result
    scale = float(vector[keep].std(ddof=0))
    if scale <= eps:
        return result
    result[keep] = np.clip(vector[keep] / (scale + eps), -clip, clip)
    return result


def leave_one_out_advantages(
    rewards: Sequence[float],
    question_ids: Sequence[object],
) -> np.ndarray:
    """Return each reward minus the mean reward of the other samples in its group."""

    reward = _vector(rewards, "rewards")
    groups = _groups(question_ids, len(reward))
    if not np.isfinite(reward).all():
        raise ValueError("rewards must be finite")
    advantages = np.zeros_like(reward)
    for group in dict.fromkeys(groups.tolist()):
        local = groups == group
        values = reward[local]
        if len(values) < 2:
            continue
        advantages[local] = values - (values.sum() - values) / (len(values) - 1)
    return advantages


def jepo_multisample_terms(
    answer_logp: Sequence[float],
    question_ids: Sequence[object],
    *,
    active: Sequence[bool] | None = None,
    advantage_clip: float = 1.0,
    eps: float = 1e-8,
) -> JEPOMultisampleTerms:
    """Build the detached multi-sample JEPO trace and answer coefficients.

    For each question, answer weights are the softmax of the gold-answer log
    likelihoods.  A trace's raw score-function credit is the log-average
    evidence with all active samples minus the log-average evidence after that
    trace is left out.  Format-invalid samples can be excluded with ``active``;
    the caller should train their format separately.
    """

    evidence = _vector(answer_logp, "answer_logp")
    groups = _groups(question_ids, len(evidence))
    keep = _mask(active, len(evidence))
    if not np.isfinite(evidence[keep]).all():
        raise ValueError("active answer log probabilities must be finite")

    weights = np.zeros_like(evidence)
    raw_advantages = np.zeros_like(evidence)
    active_groups = 0
    for group in dict.fromkeys(groups.tolist()):
        local_indices = np.flatnonzero((groups == group) & keep)
        if not len(local_indices):
            continue
        active_groups += 1
        values = evidence[local_indices]
        centred = values - values.max()
        mass = np.exp(centred)
        weights[local_indices] = mass / mass.sum()
        if len(local_indices) < 2:
            continue
        total = _logmeanexp(values)
        for position, index in enumerate(local_indices):
            remainder = np.delete(values, position)
            raw_advantages[index] = total - _logmeanexp(remainder)

    advantages = standardize_and_clip(
        raw_advantages,
        active=keep,
        clip=advantage_clip,
        eps=eps,
    )
    return JEPOMultisampleTerms(
        answer_weights=weights,
        raw_trace_advantages=raw_advantages,
        trace_advantages=advantages,
        active=keep,
        active_groups=active_groups,
    )


def fixed_masked_coefficients(
    terms: JEPOMultisampleTerms,
    *,
    question_count: int,
    samples_per_question: int,
    supervised_coefficient: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale masked JEPO terms without renormalizing around invalid samples."""

    if question_count < 1 or samples_per_question < 1:
        raise ValueError("JEPO question and sample counts must be positive")
    if not np.isfinite(supervised_coefficient) or supervised_coefficient < 0:
        raise ValueError("JEPO supervised coefficient must be finite and nonnegative")
    trace = terms.trace_advantages / (question_count * samples_per_question)
    answer = supervised_coefficient * terms.answer_weights / question_count
    return trace, answer
