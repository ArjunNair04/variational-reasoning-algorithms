"""Pure diagnostics shared by AC-ALG1 and L2R training analysis.

The trainer owns model execution.  This module only reduces detached tensors
and scalar records, which keeps diagnostic calculations testable and prevents
them from changing the scientific objective.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


DIAGNOSTIC_LEVELS = ("standard", "deep")
_EPSILON = 1e-12


def validate_diagnostic_level(level: str) -> str:
    """Validate and return a supported training-diagnostic level."""

    if level not in DIAGNOSTIC_LEVELS:
        raise ValueError(
            f"unknown diagnostics_level {level!r}; "
            f"expected one of {DIAGNOSTIC_LEVELS}"
        )
    return level


def run_diagnostic_probe(model, probe_fn) -> float:
    """Evaluate a fixed probe without perturbing training RNG or model mode."""

    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    was_training = bool(model.training)
    try:
        value = float(probe_fn(model))
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        model.train(was_training)
    if not math.isfinite(value):
        raise FloatingPointError("diagnostic probe returned a non-finite metric")
    return value


def finite_or_none(value: Any) -> float | None:
    """Convert a scalar to a JSON-safe finite float."""

    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def tensor_list_norm(values: Sequence[torch.Tensor | None]) -> float:
    """Stable float32 L2 norm over an aligned tensor sequence."""

    squared = sum(
        float(torch.sum(value.detach().float().square()))
        for value in values
        if value is not None
    )
    if not math.isfinite(squared):
        raise FloatingPointError("non-finite diagnostic tensor norm")
    return math.sqrt(max(squared, 0.0))


def tensor_list_cosine(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> float | None:
    """Cosine between two parameter-aligned tensor sequences."""

    if len(left) != len(right):
        raise ValueError("diagnostic tensor lists must have equal lengths")
    left_norm = tensor_list_norm(left)
    right_norm = tensor_list_norm(right)
    if left_norm <= _EPSILON or right_norm <= _EPSILON:
        return None
    dot = sum(
        float(torch.sum(a.detach().float() * b.detach().float()))
        for a, b in zip(left, right)
        if a is not None and b is not None
    )
    if not math.isfinite(dot):
        raise FloatingPointError("non-finite diagnostic tensor dot product")
    return min(max(dot / (left_norm * right_norm), -1.0), 1.0)


def parameter_delta_norm(
    before: Sequence[torch.Tensor],
    after: Sequence[torch.Tensor],
) -> float:
    """L2 norm of an applied parameter update."""

    if len(before) != len(after):
        raise ValueError("parameter snapshots must have equal lengths")
    deltas = []
    for old, new in zip(before, after):
        if old.shape != new.shape:
            raise ValueError("parameter snapshots contain a mismatched shape")
        deltas.append(new.detach().float() - old.detach().float())
    return tensor_list_norm(deltas)


def optimizer_moment_diagnostics(optimizer) -> dict[str, float | int | None]:
    """Summarise Adam-like state without assuming every parameter has state."""

    first_moments: list[torch.Tensor] = []
    second_moments: list[torch.Tensor] = []
    steps: list[float] = []
    for state in optimizer.state.values():
        first = state.get("exp_avg")
        second = state.get("exp_avg_sq")
        step = state.get("step")
        if isinstance(first, torch.Tensor):
            first_moments.append(first)
        if isinstance(second, torch.Tensor):
            second_moments.append(second)
        if step is not None:
            steps.append(float(step.item() if isinstance(step, torch.Tensor) else step))
    learning_rates = [
        float(group["lr"])
        for group in optimizer.param_groups
        if "lr" in group
    ]
    return {
        "state_parameter_tensors": len(optimizer.state),
        "first_moment_l2_norm": (
            tensor_list_norm(first_moments) if first_moments else None
        ),
        "second_moment_l2_norm": (
            tensor_list_norm(second_moments) if second_moments else None
        ),
        "minimum_step": min(steps) if steps else None,
        "maximum_step": max(steps) if steps else None,
        "learning_rate_min": min(learning_rates) if learning_rates else None,
        "learning_rate_max": max(learning_rates) if learning_rates else None,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks with deterministic tie handling."""

    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * ((start + 1) + end)
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def pearson_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    """Finite Pearson correlation, or None when either side is constant."""

    if len(left) != len(right):
        raise ValueError("correlation vectors must have equal lengths")
    if len(left) < 2:
        return None
    if any(not math.isfinite(value) for value in [*left, *right]):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_sq = sum(value * value for value in left_centered)
    right_sq = sum(value * value for value in right_centered)
    denominator = math.sqrt(left_sq * right_sq)
    if denominator <= _EPSILON:
        return None
    correlation = sum(
        a * b for a, b in zip(left_centered, right_centered)
    ) / denominator
    return min(max(correlation, -1.0), 1.0)


def spearman_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    """Spearman rank correlation with average ranks for ties."""

    if len(left) != len(right):
        raise ValueError("correlation vectors must have equal lengths")
    if len(left) < 2:
        return None
    return pearson_correlation(_average_ranks(left), _average_ranks(right))


def responsibility_gini(weights: Sequence[float]) -> float | None:
    """Gini coefficient of a non-negative responsibility vector."""

    if not weights:
        return None
    if any(not math.isfinite(value) or value < 0 for value in weights):
        return None
    total = sum(weights)
    if total <= _EPSILON:
        return None
    ordered = sorted(weights)
    count = len(ordered)
    weighted_sum = sum(
        (index + 1) * value for index, value in enumerate(ordered)
    )
    gini = 2.0 * weighted_sum / (count * total) - (count + 1.0) / count
    return min(max(gini, 0.0), 1.0)


def responsibility_margin(weights: Sequence[float]) -> float | None:
    """Top-one minus top-two responsibility mass."""

    if not weights or any(not math.isfinite(value) for value in weights):
        return None
    ordered = sorted(weights, reverse=True)
    return ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)


def posterior_churn(
    before: Mapping[int, torch.Tensor],
    after: Mapping[int, torch.Tensor],
    trace_ids: Mapping[int, Sequence[str | None]] | None = None,
) -> dict[str, Any]:
    """Measure question-local posterior movement across one E-step refresh."""

    if set(before) != set(after):
        raise RuntimeError("posterior refresh changed the question support")
    questions = []
    for pid in sorted(before):
        old = [float(value) for value in before[pid].detach().float().cpu()]
        new = [float(value) for value in after[pid].detach().float().cpu()]
        if len(old) != len(new):
            raise RuntimeError(
                f"posterior refresh changed support width for pid {pid}"
            )
        if not old:
            continue
        if any(
            not math.isfinite(value) or value < 0
            for value in [*old, *new]
        ):
            raise FloatingPointError(
                f"non-finite posterior churn input for pid {pid}"
            )
        old_total = sum(old)
        new_total = sum(new)
        if old_total <= 0 or new_total <= 0:
            raise FloatingPointError(
                f"zero-mass posterior churn input for pid {pid}"
            )
        old = [value / old_total for value in old]
        new = [value / new_total for value in new]
        top_before = max(range(len(old)), key=old.__getitem__)
        top_after = max(range(len(new)), key=new.__getitem__)
        ids = list(trace_ids.get(pid, ())) if trace_ids is not None else []
        top_before_id = ids[top_before] if len(ids) == len(old) else None
        top_after_id = ids[top_after] if len(ids) == len(new) else None
        questions.append({
            "pid": int(pid),
            "trace_count": len(old),
            "total_variation": 0.5 * sum(
                abs(left - right) for left, right in zip(old, new)
            ),
            "forward_kl": sum(
                left * math.log(max(left, _EPSILON) / max(right, _EPSILON))
                for left, right in zip(old, new)
            ),
            "reverse_kl": sum(
                right * math.log(max(right, _EPSILON) / max(left, _EPSILON))
                for left, right in zip(old, new)
            ),
            "rank_correlation": spearman_correlation(old, new),
            "top_trace_replaced": top_before != top_after,
            "top_trace_before": top_before_id,
            "top_trace_after": top_after_id,
        })

    def mean(key: str) -> float | None:
        values = [
            float(question[key])
            for question in questions
            if question[key] is not None
        ]
        return sum(values) / len(values) if values else None

    return {
        "questions": questions,
        "summary": {
            "question_count": len(questions),
            "mean_total_variation": mean("total_variation"),
            "mean_forward_kl": mean("forward_kl"),
            "mean_reverse_kl": mean("reverse_kl"),
            "mean_rank_correlation": mean("rank_correlation"),
            "top_trace_replacement_fraction": mean("top_trace_replaced"),
        },
    }


def binary_score_calibration(
    log_probabilities: Sequence[float],
    outcomes: Sequence[bool],
    *,
    bins: int = 10,
) -> dict[str, float | int | None]:
    """Calibration and ranking metrics for sequence log probabilities."""

    if len(log_probabilities) != len(outcomes):
        raise ValueError("reader scores and outcomes must have equal lengths")
    pairs = [
        (float(score), bool(outcome))
        for score, outcome in zip(log_probabilities, outcomes)
        if math.isfinite(float(score))
    ]
    if not pairs:
        return {
            "count": 0,
            "positive_count": 0,
            "auroc": None,
            "auprc": None,
            "brier": None,
            "ece": None,
        }
    scores = [score for score, _outcome in pairs]
    labels = [outcome for _score, outcome in pairs]
    probabilities = [math.exp(min(score, 0.0)) for score in scores]
    positives = sum(labels)
    negatives = len(labels) - positives

    auroc = None
    if positives and negatives:
        ranks = _average_ranks(scores)
        positive_rank_sum = sum(
            rank for rank, label in zip(ranks, labels) if label
        )
        auroc = (
            positive_rank_sum - positives * (positives + 1) / 2
        ) / (positives * negatives)

    auprc = None
    if positives:
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        true_positives = 0
        precision_at_positive = []
        for rank, index in enumerate(order, start=1):
            if labels[index]:
                true_positives += 1
                precision_at_positive.append(true_positives / rank)
        auprc = sum(precision_at_positive) / positives

    brier = sum(
        (probability - float(label)) ** 2
        for probability, label in zip(probabilities, labels)
    ) / len(labels)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper
            or (bin_index == bins - 1 and probability == 1.0)
        ]
        if not selected:
            continue
        confidence = sum(probabilities[index] for index in selected) / len(selected)
        accuracy = sum(labels[index] for index in selected) / len(selected)
        ece += len(selected) / len(labels) * abs(confidence - accuracy)

    return {
        "count": len(labels),
        "positive_count": positives,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
    }


def cuda_memory_diagnostics() -> dict[str, float | None]:
    """Current-process CUDA memory counters in GiB."""

    if not torch.cuda.is_available():
        return {
            "allocated_gib": None,
            "reserved_gib": None,
            "peak_allocated_gib": None,
            "peak_reserved_gib": None,
        }
    divisor = float(2**30)
    return {
        "allocated_gib": torch.cuda.memory_allocated() / divisor,
        "reserved_gib": torch.cuda.memory_reserved() / divisor,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / divisor,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / divisor,
    }
