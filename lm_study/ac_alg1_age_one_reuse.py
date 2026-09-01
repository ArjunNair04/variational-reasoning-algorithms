r"""Pure selection rules for a bounded, age-one trace-reuse pilot.

The selector deliberately knows nothing about the trainer's ``TraceRow``
type.  Callers provide accessors for the six values needed by the policy.  A
row is reusable only when it:

* was generated freshly in the immediately preceding outer round;
* has not itself come from a cache;
* has finite proposal and current-policy log densities; and
* has an importance log ratio in ``[log(0.5), log(2)]``.

At most four rows are reused.  Eligible rows are ordered by ``trace_id`` so
selection does not depend on cache iteration order.  Every unfilled support
position is explicitly counted as fresh backfill; therefore reuse can reduce
rollouts but can never reduce the requested support size.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
import math
from typing import Generic, TypeVar


RowT = TypeVar("RowT")

MAX_REUSED_ROWS = 4
MIN_LOG_IMPORTANCE_RATIO = math.log(0.5)
MAX_LOG_IMPORTANCE_RATIO = math.log(2.0)
FRESH_SOURCE = "fresh"


class ReuseRejectionReason(str, Enum):
    """Fail-closed reasons for excluding one cache candidate."""

    BOOTSTRAP_ROUND = "bootstrap_round"
    DUPLICATE_TRACE_ID = "duplicate_trace_id"
    NON_FRESH_SOURCE = "non_fresh_source"
    NOT_IMMEDIATELY_PREVIOUS_ROUND = "not_immediately_previous_round"
    NONFINITE_LOG_DENSITY = "nonfinite_log_density"
    LOG_RATIO_BELOW_FLOOR = "log_ratio_below_floor"
    LOG_RATIO_ABOVE_CEILING = "log_ratio_above_ceiling"
    REUSE_CAPACITY_EXCEEDED = "reuse_capacity_exceeded"


@dataclass(frozen=True)
class ReuseDecision:
    """Auditable decision for one candidate row."""

    trace_id: str
    round_added: int
    source: str
    proposal_log_density: float
    current_log_density: float
    log_importance_ratio: float | None
    accepted: bool
    rejection_reason: ReuseRejectionReason | None


@dataclass(frozen=True)
class AgeOneReuseAudit:
    """Complete accounting for one deterministic cache-selection call."""

    current_round: int
    target_support_size: int
    maximum_reused_rows: int
    fresh_source: str
    minimum_log_importance_ratio: float
    maximum_log_importance_ratio: float
    candidates_seen: int
    eligible_before_capacity: int
    selected_trace_ids: tuple[str, ...]
    selected_count: int
    fresh_backfill_count: int
    reuse_shortfall_count: int
    rollouts_saved: int
    rejection_counts: tuple[tuple[str, int], ...]
    decisions: tuple[ReuseDecision, ...]


@dataclass(frozen=True)
class AgeOneReuseSelection(Generic[RowT]):
    """Original selected rows together with immutable selection accounting."""

    selected_rows: tuple[RowT, ...]
    audit: AgeOneReuseAudit


@dataclass(frozen=True)
class _Candidate(Generic[RowT]):
    row: RowT
    trace_id: str
    round_added: int
    source: str
    proposal_log_density: float
    current_log_density: float

    @property
    def log_importance_ratio(self) -> float:
        return self.current_log_density - self.proposal_log_density


def _require_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return value


def _extract_candidate(
    row: RowT,
    *,
    round_added: Callable[[RowT], int],
    trace_id: Callable[[RowT], str],
    source: Callable[[RowT], str],
    proposal_log_density: Callable[[RowT], float],
    current_log_density: Callable[[RowT], float],
) -> _Candidate[RowT]:
    row_trace_id = trace_id(row)
    if not isinstance(row_trace_id, str) or not row_trace_id:
        raise ValueError("trace_id must be a nonempty string")

    row_round = round_added(row)
    if isinstance(row_round, bool) or not isinstance(row_round, int):
        raise ValueError(
            f"round_added for {row_trace_id!r} must be an integer, "
            f"got {row_round!r}"
        )

    row_source = source(row)
    if not isinstance(row_source, str) or not row_source:
        raise ValueError(f"source for {row_trace_id!r} must be a nonempty string")

    return _Candidate(
        row=row,
        trace_id=row_trace_id,
        round_added=row_round,
        source=row_source,
        proposal_log_density=float(proposal_log_density(row)),
        current_log_density=float(current_log_density(row)),
    )


def _float_sort_key(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "+inf"
    if value == -math.inf:
        return "-inf"
    return value.hex()


def _candidate_sort_key(candidate: _Candidate[RowT]) -> tuple[object, ...]:
    return (
        candidate.trace_id,
        candidate.round_added,
        candidate.source,
        _float_sort_key(candidate.proposal_log_density),
        _float_sort_key(candidate.current_log_density),
    )


def _pre_capacity_rejection(
    candidate: _Candidate[RowT],
    *,
    current_round: int,
    fresh_source: str,
    duplicate_trace_ids: frozenset[str],
) -> ReuseRejectionReason | None:
    if current_round == 0:
        return ReuseRejectionReason.BOOTSTRAP_ROUND
    if candidate.trace_id in duplicate_trace_ids:
        return ReuseRejectionReason.DUPLICATE_TRACE_ID
    if candidate.source != fresh_source:
        return ReuseRejectionReason.NON_FRESH_SOURCE
    if candidate.round_added != current_round - 1:
        return ReuseRejectionReason.NOT_IMMEDIATELY_PREVIOUS_ROUND
    if not (
        math.isfinite(candidate.proposal_log_density)
        and math.isfinite(candidate.current_log_density)
    ):
        return ReuseRejectionReason.NONFINITE_LOG_DENSITY

    log_ratio = candidate.log_importance_ratio
    if not math.isfinite(log_ratio):
        return ReuseRejectionReason.NONFINITE_LOG_DENSITY
    if log_ratio < MIN_LOG_IMPORTANCE_RATIO:
        return ReuseRejectionReason.LOG_RATIO_BELOW_FLOOR
    if log_ratio > MAX_LOG_IMPORTANCE_RATIO:
        return ReuseRejectionReason.LOG_RATIO_ABOVE_CEILING
    return None


def validate_age_one_reuse_audit(audit: AgeOneReuseAudit) -> None:
    """Raise ``AssertionError`` if any accounting or safety invariant fails."""

    decisions = audit.decisions
    accepted = tuple(decision for decision in decisions if decision.accepted)
    rejected = tuple(decision for decision in decisions if not decision.accepted)
    reuse_limit = min(audit.maximum_reused_rows, audit.target_support_size)

    assert 0 <= audit.maximum_reused_rows <= MAX_REUSED_ROWS
    assert audit.current_round >= 0
    assert audit.target_support_size >= 0
    assert audit.candidates_seen == len(decisions)
    assert audit.selected_count == len(accepted)
    assert audit.selected_count <= reuse_limit
    assert audit.selected_trace_ids == tuple(
        decision.trace_id for decision in accepted
    )
    assert audit.selected_trace_ids == tuple(sorted(audit.selected_trace_ids))
    assert len(set(audit.selected_trace_ids)) == audit.selected_count
    assert audit.fresh_backfill_count == (
        audit.target_support_size - audit.selected_count
    )
    assert audit.reuse_shortfall_count == reuse_limit - audit.selected_count
    assert audit.rollouts_saved == audit.selected_count
    assert audit.rollouts_saved + audit.fresh_backfill_count == (
        audit.target_support_size
    )

    for decision in accepted:
        assert decision.rejection_reason is None
        assert audit.current_round > 0
        assert decision.round_added == audit.current_round - 1
        assert decision.source == audit.fresh_source
        assert decision.log_importance_ratio is not None
        assert math.isfinite(decision.log_importance_ratio)
        assert (
            audit.minimum_log_importance_ratio
            <= decision.log_importance_ratio
            <= audit.maximum_log_importance_ratio
        )
    for decision in rejected:
        assert decision.rejection_reason is not None

    capacity_rejections = sum(
        decision.rejection_reason
        is ReuseRejectionReason.REUSE_CAPACITY_EXCEEDED
        for decision in rejected
    )
    assert audit.eligible_before_capacity == audit.selected_count + capacity_rejections
    expected_counts = Counter(
        decision.rejection_reason.value
        for decision in rejected
        if decision.rejection_reason is not None
    )
    assert audit.rejection_counts == tuple(sorted(expected_counts.items()))
    if audit.current_round == 0:
        assert audit.selected_count == 0


def select_age_one_reuse(
    rows: Iterable[RowT],
    *,
    current_round: int,
    target_support_size: int,
    round_added: Callable[[RowT], int],
    trace_id: Callable[[RowT], str],
    source: Callable[[RowT], str],
    proposal_log_density: Callable[[RowT], float],
    current_log_density: Callable[[RowT], float],
    maximum_reused_rows: int = MAX_REUSED_ROWS,
    fresh_source: str = FRESH_SOURCE,
) -> AgeOneReuseSelection[RowT]:
    """Select bounded age-one rows and account for mandatory fresh backfill.

    The importance log ratio is ``current_log_density -
    proposal_log_density``.  Both endpoints are inclusive.  Duplicate trace
    identifiers are rejected in full instead of selecting an input-order-
    dependent representative.
    """

    round_index = _require_nonnegative_integer(current_round, name="current_round")
    support_size = _require_nonnegative_integer(
        target_support_size,
        name="target_support_size",
    )
    reuse_maximum = _require_nonnegative_integer(
        maximum_reused_rows,
        name="maximum_reused_rows",
    )
    if reuse_maximum > MAX_REUSED_ROWS:
        raise ValueError(
            f"maximum_reused_rows cannot exceed {MAX_REUSED_ROWS}, "
            f"got {reuse_maximum}"
        )
    if not isinstance(fresh_source, str) or not fresh_source:
        raise ValueError("fresh_source must be a nonempty string")

    candidates = tuple(
        sorted(
            (
                _extract_candidate(
                    row,
                    round_added=round_added,
                    trace_id=trace_id,
                    source=source,
                    proposal_log_density=proposal_log_density,
                    current_log_density=current_log_density,
                )
                for row in rows
            ),
            key=_candidate_sort_key,
        )
    )
    trace_id_counts = Counter(candidate.trace_id for candidate in candidates)
    duplicate_trace_ids = frozenset(
        row_id for row_id, count in trace_id_counts.items() if count > 1
    )

    preliminary = {
        candidate.trace_id: _pre_capacity_rejection(
            candidate,
            current_round=round_index,
            fresh_source=fresh_source,
            duplicate_trace_ids=duplicate_trace_ids,
        )
        for candidate in candidates
        if candidate.trace_id not in duplicate_trace_ids
    }
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.trace_id not in duplicate_trace_ids
        and preliminary[candidate.trace_id] is None
    )
    reuse_limit = min(reuse_maximum, support_size)
    selected = eligible[:reuse_limit]
    selected_ids = frozenset(candidate.trace_id for candidate in selected)

    decisions: list[ReuseDecision] = []
    for candidate in candidates:
        if candidate.trace_id in duplicate_trace_ids:
            reason = ReuseRejectionReason.DUPLICATE_TRACE_ID
        else:
            reason = preliminary[candidate.trace_id]
        accepted = reason is None and candidate.trace_id in selected_ids
        if reason is None and not accepted:
            reason = ReuseRejectionReason.REUSE_CAPACITY_EXCEEDED
        ratio = candidate.log_importance_ratio
        decisions.append(
            ReuseDecision(
                trace_id=candidate.trace_id,
                round_added=candidate.round_added,
                source=candidate.source,
                proposal_log_density=candidate.proposal_log_density,
                current_log_density=candidate.current_log_density,
                log_importance_ratio=ratio if math.isfinite(ratio) else None,
                accepted=accepted,
                rejection_reason=reason,
            )
        )

    rejection_counts = Counter(
        decision.rejection_reason.value
        for decision in decisions
        if decision.rejection_reason is not None
    )
    selected_trace_ids = tuple(candidate.trace_id for candidate in selected)
    selected_count = len(selected)
    audit = AgeOneReuseAudit(
        current_round=round_index,
        target_support_size=support_size,
        maximum_reused_rows=reuse_maximum,
        fresh_source=fresh_source,
        minimum_log_importance_ratio=MIN_LOG_IMPORTANCE_RATIO,
        maximum_log_importance_ratio=MAX_LOG_IMPORTANCE_RATIO,
        candidates_seen=len(candidates),
        eligible_before_capacity=len(eligible),
        selected_trace_ids=selected_trace_ids,
        selected_count=selected_count,
        fresh_backfill_count=support_size - selected_count,
        reuse_shortfall_count=reuse_limit - selected_count,
        rollouts_saved=selected_count,
        rejection_counts=tuple(sorted(rejection_counts.items())),
        decisions=tuple(decisions),
    )
    validate_age_one_reuse_audit(audit)
    return AgeOneReuseSelection(
        selected_rows=tuple(candidate.row for candidate in selected),
        audit=audit,
    )

