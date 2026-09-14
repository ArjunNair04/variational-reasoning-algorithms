"""Position-only trace-prior masks and Q5 E-step factors; no M-step changes."""

import math

import torch


def validate_segment_exponents(head, tail):
    if any(not math.isfinite(value) or value < 0 for value in (head, tail)):
        raise ValueError("segment prior exponents must be finite and nonnegative")
    return head != 1.0 or tail != 1.0


def split_rationale_mask(trace_mask):
    """Split scored rationale positions, not prompt/padding/answer positions."""
    if trace_mask.ndim != 2 or trace_mask.dtype != torch.bool:
        raise ValueError("rationale mask must be a two-dimensional boolean tensor")
    if trace_mask.shape[1] and trace_mask[:, 0].any():
        raise ValueError("position zero cannot be scored autoregressively")
    half = trace_mask.sum(1, keepdim=True) // 2
    head = trace_mask & (trace_mask.long().cumsum(1) <= half)
    return head, trace_mask & ~head


def segment_prior_logits(answer, prior, head, head_exponent, tail_exponent):
    """Preserve uniform-prior arithmetic; otherwise attenuate two disjoint parts."""
    validate_segment_exponents(head_exponent, tail_exponent)
    if answer.shape != prior.shape or head.shape != prior.shape:
        raise ValueError("answer and segment factors must have identical shapes")
    if not all(torch.isfinite(value).all() for value in (answer, prior, head)):
        raise ValueError("nonfinite segment factors")
    if head_exponent == tail_exponent:
        return answer + head_exponent * prior
    return answer + head_exponent * head + tail_exponent * (prior - head)
