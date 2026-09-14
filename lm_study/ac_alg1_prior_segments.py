"""Position-only trace-prior masks and Q5 E-step factors; no M-step changes."""

import math

import torch

SEGMENT_SCOPE = "reasoning_marker_fixed"


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


def split_reasoning_marker_mask(trace_mask, reasoning_counts):
    """Use the stored token boundary; never tokenize or change the prefix."""
    split_rationale_mask(trace_mask)  # Validate scored positions before splitting.
    if len(reasoning_counts) != trace_mask.shape[0] or any(
        type(n) is not int or n < 0 for n in reasoning_counts
    ):
        raise ValueError("one nonnegative integer reasoning count is required per trace")
    counts = torch.tensor(reasoning_counts, device=trace_mask.device).unsqueeze(1)
    if (counts >= trace_mask.sum(1, keepdim=True)).any():
        raise ValueError("each trace requires a nonempty marker suffix outside reasoning")
    reasoning = trace_mask & (trace_mask.long().cumsum(1) <= counts)
    head, tail = split_rationale_mask(reasoning)
    return head, tail, trace_mask & ~reasoning


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
