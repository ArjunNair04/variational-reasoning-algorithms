"""Posterior-only trace allocation for the uncertainty pilot.

This module is deliberately independent of the trainer and of outcome labels.
It consumes detached posterior summaries from a fixed four-trace first stage
and returns an immutable audit record for the remainder of that same round's
trace budget.

For question ``q``, with null mass ``q0`` and conditional real-trace weights
``p_s``, the allocation utility is

    u(q) = q0 + (1 - q0) H(p) / log(S).

The entropy term is zero for fewer than two real support points.  Eight
questions share exactly 64 traces, with a floor of four and a cap of twelve.
The placebo keeps the resulting count multiset fixed and cyclically shifts it
across question ids.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Iterable, Literal


QUESTION_COUNT = 8
TOTAL_TRACE_BUDGET = 64
MIN_TRACES_PER_QUESTION = 4
MAX_TRACES_PER_QUESTION = 12
_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PosteriorUncertainty:
    """Detached posterior quantities used to score one question."""

    null_mass: float
    conditional_real_weights: tuple[float, ...]
    normalized_real_entropy: float
    utility: float


@dataclass(frozen=True)
class QuestionAllocationAudit:
    """All inputs and allocation decisions for one canonical question id."""

    question_id: int
    null_mass: float
    conditional_real_weights: tuple[float, ...]
    normalized_real_entropy: float
    utility: float
    continuous_target: float
    fractional_remainder: float
    aligned_count: int
    placebo_count: int


@dataclass(frozen=True)
class UncertaintyAllocationAudit:
    """Immutable, serializable record of one exact budget allocation."""

    schema_version: int
    total_budget: int
    minimum_per_question: int
    maximum_per_question: int
    placebo_shift: int
    used_zero_utility_fallback: bool
    redistribution_steps: int
    questions: tuple[QuestionAllocationAudit, ...]

    @property
    def question_ids(self) -> tuple[int, ...]:
        return tuple(question.question_id for question in self.questions)

    @property
    def utilities(self) -> tuple[float, ...]:
        return tuple(question.utility for question in self.questions)

    @property
    def aligned_counts(self) -> tuple[int, ...]:
        return tuple(question.aligned_count for question in self.questions)

    @property
    def placebo_counts(self) -> tuple[int, ...]:
        return tuple(question.placebo_count for question in self.questions)

    def counts_by_question(
        self,
        mode: Literal["aligned", "placebo"],
    ) -> dict[int, int]:
        """Return the selected count vector keyed by canonical question id."""

        if mode == "aligned":
            counts = self.aligned_counts
        elif mode == "placebo":
            counts = self.placebo_counts
        else:
            raise ValueError(f"unknown allocation mode {mode!r}")
        return dict(zip(self.question_ids, counts))

    def as_dict(self) -> dict:
        """Return a JSON-compatible representation for runtime diagnostics."""

        return asdict(self)


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a detached real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def detached_posterior_uncertainty(
    null_mass: Real,
    conditional_real_weights: Iterable[Real],
) -> PosteriorUncertainty:
    """Compute posterior-only uncertainty after copying inputs to plain floats.

    Positive real weights are normalized defensively, so callers may pass
    either an already conditional row or an unnormalized proportional row.  A
    row with zero real mass is defined only for a fully-null question.
    """

    q0 = _finite_real(null_mass, name="null_mass")
    if q0 < 0.0 or q0 > 1.0:
        raise ValueError("null_mass must be in [0, 1]")

    try:
        raw_weights = tuple(conditional_real_weights)
    except TypeError as error:
        raise TypeError("conditional_real_weights must be iterable") from error
    weights = tuple(
        _finite_real(value, name=f"conditional_real_weights[{index}]")
        for index, value in enumerate(raw_weights)
    )
    if any(value < 0.0 for value in weights):
        raise ValueError("conditional real weights must be nonnegative")

    total = math.fsum(weights)
    if total == 0.0:
        if q0 != 1.0:
            raise ValueError(
                "zero real support is valid only when null_mass is exactly one"
            )
        normalized = tuple(0.0 for _ in weights)
        normalized_entropy = 0.0
    else:
        normalized = tuple(value / total for value in weights)
        if len(normalized) < 2:
            normalized_entropy = 0.0
        else:
            entropy = -math.fsum(
                probability * math.log(probability)
                for probability in normalized
                if probability > 0.0
            )
            normalized_entropy = entropy / math.log(len(normalized))
            if not -_TOLERANCE <= normalized_entropy <= 1.0 + _TOLERANCE:
                raise RuntimeError("normalized posterior entropy escaped [0, 1]")
            normalized_entropy = min(max(normalized_entropy, 0.0), 1.0)

    utility = q0 + (1.0 - q0) * normalized_entropy
    if not -_TOLERANCE <= utility <= 1.0 + _TOLERANCE:
        raise RuntimeError("posterior uncertainty utility escaped [0, 1]")
    utility = min(max(utility, 0.0), 1.0)
    return PosteriorUncertainty(
        null_mass=q0,
        conditional_real_weights=normalized,
        normalized_real_entropy=normalized_entropy,
        utility=utility,
    )


def _bounded_continuous_targets(
    utilities: tuple[float, ...],
) -> tuple[tuple[float, ...], bool, int]:
    """Water-fill the non-floor budget before largest-remainder rounding."""

    extra_capacity = MAX_TRACES_PER_QUESTION - MIN_TRACES_PER_QUESTION
    extra_budget = TOTAL_TRACE_BUDGET - QUESTION_COUNT * MIN_TRACES_PER_QUESTION
    targets = [0.0] * QUESTION_COUNT
    active = list(range(QUESTION_COUNT))
    remaining = float(extra_budget)
    used_fallback = False
    steps = 0

    while remaining > _TOLERANCE:
        steps += 1
        if not active:
            raise RuntimeError("trace budget remains after all questions reached the cap")
        score_sum = math.fsum(utilities[index] for index in active)
        if score_sum <= _TOLERANCE:
            used_fallback = True
            scores = {index: 1.0 for index in active}
            score_sum = float(len(active))
        else:
            scores = {index: utilities[index] for index in active}

        proposals = {
            index: remaining * scores[index] / score_sum
            for index in active
        }
        saturated = [
            index
            for index in active
            if proposals[index] > extra_capacity - targets[index] + _TOLERANCE
        ]
        if not saturated:
            for index in active:
                targets[index] += proposals[index]
            remaining = 0.0
            break

        for index in saturated:
            grant = extra_capacity - targets[index]
            if grant < -_TOLERANCE:
                raise RuntimeError("continuous allocation exceeded a question cap")
            targets[index] += max(grant, 0.0)
            remaining -= max(grant, 0.0)
        saturated_set = set(saturated)
        active = [index for index in active if index not in saturated_set]

    if not math.isclose(
        math.fsum(targets),
        float(extra_budget),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("continuous allocation did not conserve the trace budget")
    return tuple(targets), used_fallback, steps


def _largest_remainder_counts(
    question_ids: tuple[int, ...],
    utilities: tuple[float, ...],
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], bool, int]:
    extras, used_fallback, steps = _bounded_continuous_targets(utilities)
    floor_extras = [math.floor(value + _TOLERANCE) for value in extras]
    counts = [MIN_TRACES_PER_QUESTION + value for value in floor_extras]
    remainders = tuple(
        max(value - floored, 0.0)
        for value, floored in zip(extras, floor_extras)
    )
    remaining = TOTAL_TRACE_BUDGET - sum(counts)
    eligible = [
        index
        for index, count in enumerate(counts)
        if count < MAX_TRACES_PER_QUESTION
    ]
    order = sorted(
        eligible,
        key=lambda index: (-remainders[index], question_ids[index]),
    )
    if remaining < 0 or remaining > len(order):
        raise RuntimeError("largest-remainder rounding cannot conserve the budget")
    for index in order[:remaining]:
        counts[index] += 1

    if sum(counts) != TOTAL_TRACE_BUDGET:
        raise RuntimeError("integer allocation did not conserve the trace budget")
    if any(
        count < MIN_TRACES_PER_QUESTION or count > MAX_TRACES_PER_QUESTION
        for count in counts
    ):
        raise RuntimeError("integer allocation violated its floor or cap")
    continuous_targets = tuple(
        MIN_TRACES_PER_QUESTION + value for value in extras
    )
    return (
        tuple(counts),
        continuous_targets,
        remainders,
        used_fallback,
        steps,
    )


def allocate_uncertainty_budget(
    question_ids: Iterable[Integral],
    null_masses: Iterable[Real],
    conditional_real_weights: Iterable[Iterable[Real]],
    *,
    placebo_shift: Integral = 1,
) -> UncertaintyAllocationAudit:
    """Allocate the exact eight-question budget and its shifted placebo.

    Inputs are canonicalized by integer question id before scoring and tie
    breaking.  Therefore reordering all three input sequences together cannot
    change the returned audit.
    """

    ids_raw = tuple(question_ids)
    null_raw = tuple(null_masses)
    rows_raw = tuple(conditional_real_weights)
    if not (
        len(ids_raw) == len(null_raw) == len(rows_raw) == QUESTION_COUNT
    ):
        raise ValueError(
            "uncertainty allocation requires exactly eight question ids, "
            "null masses, and real-weight rows"
        )

    ids = []
    for index, value in enumerate(ids_raw):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"question_ids[{index}] must be an integer")
        ids.append(int(value))
    if len(set(ids)) != QUESTION_COUNT:
        raise ValueError("question ids must be unique")

    if isinstance(placebo_shift, bool) or not isinstance(placebo_shift, Integral):
        raise TypeError("placebo_shift must be an integer")
    shift = int(placebo_shift) % QUESTION_COUNT
    if shift == 0:
        raise ValueError("placebo_shift must be non-zero modulo eight")

    canonical = sorted(zip(ids, null_raw, rows_raw), key=lambda item: item[0])
    posterior = tuple(
        detached_posterior_uncertainty(null_mass, row)
        for _question_id, null_mass, row in canonical
    )
    canonical_ids = tuple(question_id for question_id, _null, _row in canonical)
    utilities = tuple(item.utility for item in posterior)
    (
        aligned,
        continuous_targets,
        remainders,
        used_fallback,
        steps,
    ) = _largest_remainder_counts(canonical_ids, utilities)

    placebo = tuple(
        aligned[(index - shift) % QUESTION_COUNT]
        for index in range(QUESTION_COUNT)
    )
    if sorted(placebo) != sorted(aligned):
        raise RuntimeError("placebo allocation did not preserve the count multiset")
    if sum(placebo) != TOTAL_TRACE_BUDGET:
        raise RuntimeError("placebo allocation did not conserve the trace budget")

    questions = tuple(
        QuestionAllocationAudit(
            question_id=question_id,
            null_mass=item.null_mass,
            conditional_real_weights=item.conditional_real_weights,
            normalized_real_entropy=item.normalized_real_entropy,
            utility=item.utility,
            continuous_target=continuous_target,
            fractional_remainder=remainder,
            aligned_count=aligned_count,
            placebo_count=placebo_count,
        )
        for question_id, item, continuous_target, remainder, aligned_count, placebo_count
        in zip(
            canonical_ids,
            posterior,
            continuous_targets,
            remainders,
            aligned,
            placebo,
        )
    )
    return UncertaintyAllocationAudit(
        schema_version=1,
        total_budget=TOTAL_TRACE_BUDGET,
        minimum_per_question=MIN_TRACES_PER_QUESTION,
        maximum_per_question=MAX_TRACES_PER_QUESTION,
        placebo_shift=shift,
        used_zero_utility_fallback=used_fallback,
        redistribution_steps=steps,
        questions=questions,
    )
