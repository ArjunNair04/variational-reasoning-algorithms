"""Numerical kernels for the GRPO and RLOO comparison methods."""

from __future__ import annotations

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


def grpo_advantages(
    rewards: Sequence[float],
    question_ids: Sequence[object],
    eps: float = 1e-8,
) -> np.ndarray:
    """Population-standardised reward advantage within each prompt group."""

    reward = _vector(rewards, "rewards")
    groups = _groups(question_ids, len(reward))
    advantage = np.zeros_like(reward)
    for group in dict.fromkeys(groups.tolist()):
        local = groups == group
        values = reward[local]
        std = values.std(ddof=0)
        if std > 0:
            advantage[local] = (values - values.mean()) / (std + eps)
    return advantage


def grpo_loss(
    token_logp: Sequence[Sequence[float]],
    old_token_logp: Sequence[Sequence[float]],
    reference_token_logp: Sequence[Sequence[float]],
    advantages: Sequence[float],
    token_mask: Sequence[Sequence[bool]],
    *,
    clip: float = 0.2,
    kl_coef: float = 0.02,
) -> float:
    """Selected token-level clipped GRPO loss with the k3 KL estimator."""

    current = np.asarray(token_logp, dtype=np.float64)
    old = np.asarray(old_token_logp, dtype=np.float64)
    reference = np.asarray(reference_token_logp, dtype=np.float64)
    mask = np.asarray(token_mask, dtype=bool)
    advantage = _vector(advantages, "advantages")
    if not (current.shape == old.shape == reference.shape == mask.shape):
        raise ValueError("token arrays and token_mask must align")
    if current.ndim != 2 or current.shape[0] != len(advantage):
        raise ValueError("one advantage is required per response")
    if not 0 <= clip < 1 or kl_coef < 0:
        raise ValueError("clip and kl_coef are outside their valid ranges")
    if not mask.any():
        return 0.0

    ratio = np.exp(current - old)
    raw = ratio * advantage[:, None]
    clipped = np.clip(ratio, 1 - clip, 1 + clip) * advantage[:, None]
    surrogate = np.minimum(raw, clipped)
    log_ratio = np.clip(reference - current, -5.0, 5.0)
    kl_k3 = np.exp(log_ratio) - log_ratio - 1.0
    per_token = -(surrogate - kl_coef * kl_k3)
    return float(per_token[mask].mean())


def rloo_advantages(
    rewards: Sequence[float],
    question_ids: Sequence[object],
    *,
    policy_logp: Sequence[float] | None = None,
    reference_logp: Sequence[float] | None = None,
    kl_coef: float = 0.0,
) -> np.ndarray:
    """KL-shaped returns minus a within-prompt leave-one-out baseline."""

    reward = _vector(rewards, "rewards")
    groups = _groups(question_ids, len(reward))
    if kl_coef < 0:
        raise ValueError("kl_coef must be non-negative")
    if (policy_logp is None) != (reference_logp is None):
        raise ValueError("policy_logp and reference_logp must be supplied together")
    returns = reward.copy()
    if policy_logp is not None:
        policy = _vector(policy_logp, "policy_logp")
        reference = _vector(reference_logp, "reference_logp")
        if policy.shape != reward.shape or reference.shape != reward.shape:
            raise ValueError("sequence log probabilities must align with rewards")
        returns -= kl_coef * (policy - reference)

    advantage = np.empty_like(returns)
    for group in dict.fromkeys(groups.tolist()):
        local = groups == group
        values = returns[local]
        denominator = max(len(values) - 1, 1)
        advantage[local] = values - (values.sum() - values) / denominator
    return advantage


def rloo_loss(sequence_logp: Sequence[float], advantages: Sequence[float]) -> float:
    """Negative mean trajectory score weighted by detached RLOO advantage."""

    logp = _vector(sequence_logp, "sequence_logp")
    advantage = _vector(advantages, "advantages")
    if logp.shape != advantage.shape:
        raise ValueError("sequence_logp and advantages must align")
    return -float(np.mean(advantage * logp))
