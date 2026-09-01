r"""Null-latent responsibilities for selective AC-ALG1 updates.

This module implements a finite-support E-step with one additional, frozen
``null`` latent state.  For ``K`` sampled real traces with detached log-evidence
scores ``ell_i``, temperature ``tau``, null prior ``pi_0``, and fixed null
log-evidence baseline ``b_0``, the augmented posterior is

.. math::

    q_i = \frac{(1-\pi_0)K^{-1}\exp(\ell_i/\tau)}{Z},\qquad
    q_0 = \frac{\pi_0\exp(b_0/\tau)}{Z}.

The ``1 / K`` empirical-support prior makes total real coverage invariant to
duplicating every sampled candidate.  The null score is deliberately a fixed
log-evidence baseline: it is detached, is not a model output, and receives no
M-step gradient.

Most importantly, ``m_step_coefficients`` are the *unconditional* ``q_i``.
They sum to ``1 - q_0``.  The conditional real weights ``q_i / (1 - q_0)`` are
reported only as diagnostics and must not be substituted into the M-step,
because doing so would silently discard abstention and force a full-strength
update even when all real traces have weak evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Union

import torch


LogEvidence = Union[Sequence[float], torch.Tensor]


@dataclass(frozen=True)
class NullLatentResponsibilities:
    """Detached augmented-posterior outputs for one question.

    ``real_coverage`` and ``null_mass`` sum to one (up to floating-point
    rounding).  ``conditional_real_weights`` sum to one only when at least one
    real trace has nonzero evidence; they are diagnostic-only.
    """

    m_step_coefficients: torch.Tensor
    conditional_real_weights: torch.Tensor
    real_coverage: float
    null_mass: float
    conditional_ess: float
    normalized_conditional_ess: float
    real_log_mean_evidence: float
    null_log_evidence: float
    null_prior: float
    temperature: float


@dataclass(frozen=True)
class NullCalibration:
    """Controller-only calibration result for a frozen null parameter."""

    calibrated_parameter: str
    null_log_evidence: float
    null_prior: float
    target_mean_real_coverage: float
    achieved_mean_real_coverage: float
    controller_questions: int
    iterations: int


@dataclass(frozen=True)
class ThresholdRejectionResponsibilities:
    """Hard-rejection comparator using the same real-trace evidence statistic."""

    m_step_coefficients: torch.Tensor
    conditional_real_weights: torch.Tensor
    accepted: bool
    real_coverage: float
    rejection_mass: float
    conditional_ess: float
    normalized_conditional_ess: float
    real_log_mean_evidence: float
    threshold: float
    temperature: float


@dataclass(frozen=True)
class ThresholdCalibration:
    """Outcome-free controller calibration for a hard rejection threshold."""

    threshold: float
    target_real_coverage: float
    achieved_real_coverage: float
    controller_questions: int


@dataclass(frozen=True)
class CoverageRiskCurveInputs:
    """Tie-aware inputs for a selective-prediction coverage--risk curve.

    Each entry represents selecting every question whose selection score is at
    least ``thresholds[index]``.  Equal scores are kept together, since a
    threshold cannot select an arbitrary subset of a tie.
    """

    thresholds: tuple[float, ...]
    retained_coverage: tuple[float, ...]
    selective_risk: tuple[float, ...]
    retained_counts: tuple[int, ...]
    total_questions: int


def _validate_temperature(temperature: float) -> float:
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"temperature must be finite and positive, got {temperature!r}"
        )
    return value


def _validate_null_prior(null_prior: float) -> float:
    value = float(null_prior)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            f"null_prior must be finite and strictly between zero and one, "
            f"got {null_prior!r}"
        )
    return value


def _validate_null_log_evidence(null_log_evidence: float) -> float:
    value = float(null_log_evidence)
    if not math.isfinite(value):
        raise ValueError(
            "null_log_evidence is a fixed finite log-evidence baseline, "
            f"got {null_log_evidence!r}"
        )
    return value


def _detached_log_evidence(real_log_evidence: LogEvidence) -> torch.Tensor:
    """Copy evidence into a detached float64 vector without changing device."""

    values = torch.as_tensor(real_log_evidence)
    if values.ndim != 1:
        raise ValueError(
            "real_log_evidence must be a one-dimensional finite-support vector"
        )
    values = values.detach().to(dtype=torch.float64).clone()
    if bool(torch.any(torch.isnan(values))):
        raise ValueError("real_log_evidence cannot contain NaN")
    return values


def _sigmoid(value: float) -> float:
    """Overflow-safe scalar sigmoid."""

    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _conditional_real_distribution(
    real_log_evidence: LogEvidence,
    *,
    temperature: float,
) -> tuple[torch.Tensor, float]:
    """Return detached conditional trace weights and log-mean evidence."""

    tau = _validate_temperature(temperature)
    values = _detached_log_evidence(real_log_evidence)
    count = int(values.numel())
    weights = torch.zeros_like(values)
    if count == 0:
        return weights, float("-inf")

    scaled = values / tau
    positive_infinity = torch.isposinf(scaled)
    positive_infinity_count = int(positive_infinity.sum().item())
    if positive_infinity_count:
        weights[positive_infinity] = 1.0 / positive_infinity_count
        return weights, float("inf")

    finite = torch.isfinite(scaled)
    if not bool(torch.any(finite)):
        return weights, float("-inf")

    log_sum_evidence = torch.logsumexp(scaled, dim=0)
    weights[finite] = torch.exp(scaled[finite] - log_sum_evidence)
    log_mean_evidence = tau * (
        float(log_sum_evidence.item()) - math.log(count)
    )
    return weights, log_mean_evidence


def real_log_mean_evidence(
    real_log_evidence: LogEvidence,
    *,
    temperature: float = 1.0,
) -> float:
    """Return ``tau * log(mean_i(exp(ell_i / tau)))`` stably.

    This is the question-level selection score used by both the soft null
    posterior and the hard threshold comparator.
    """

    _weights, aggregate = _conditional_real_distribution(
        real_log_evidence,
        temperature=temperature,
    )
    return aggregate


def _conditional_ess(weights: torch.Tensor) -> tuple[float, float]:
    count = int(weights.numel())
    squared_mass = float(torch.sum(weights.square()).item())
    if count == 0 or squared_mass <= 0.0:
        return 0.0, 0.0
    ess = 1.0 / squared_mass
    return ess, ess / count


def _real_coverage_from_aggregate(
    aggregate_log_evidence: float,
    *,
    null_log_evidence: float,
    null_prior: float,
    temperature: float,
) -> float:
    if aggregate_log_evidence == float("-inf"):
        return 0.0
    if aggregate_log_evidence == float("inf"):
        return 1.0
    prior_log_odds = math.log1p(-null_prior) - math.log(null_prior)
    coverage_logit = prior_log_odds + (
        aggregate_log_evidence - null_log_evidence
    ) / temperature
    return _sigmoid(coverage_logit)


def null_latent_responsibilities(
    real_log_evidence: LogEvidence,
    *,
    null_log_evidence: float = 0.0,
    null_prior: float = 0.5,
    temperature: float = 1.0,
) -> NullLatentResponsibilities:
    """Compute a stable, detached posterior over real traces plus a null state.

    ``null_log_evidence`` is a fixed baseline in the same units as each real
    log-evidence score.  This function intentionally detaches *all* evidence:
    responsibilities are E-step constants, not a differentiable path through
    the M-step.
    """

    tau = _validate_temperature(temperature)
    prior = _validate_null_prior(null_prior)
    baseline = _validate_null_log_evidence(null_log_evidence)
    conditional, aggregate = _conditional_real_distribution(
        real_log_evidence,
        temperature=tau,
    )
    coverage = _real_coverage_from_aggregate(
        aggregate,
        null_log_evidence=baseline,
        null_prior=prior,
        temperature=tau,
    )
    # These are q_i, not q_i / (1 - q_0).  Retaining the coverage multiplier is
    # the defining no-forced-update property of the null latent.
    coefficients = (conditional * coverage).detach()
    conditional = conditional.detach()
    ess, normalized_ess = _conditional_ess(conditional)
    return NullLatentResponsibilities(
        m_step_coefficients=coefficients,
        conditional_real_weights=conditional,
        real_coverage=coverage,
        null_mass=1.0 - coverage,
        conditional_ess=ess,
        normalized_conditional_ess=normalized_ess,
        real_log_mean_evidence=aggregate,
        null_log_evidence=baseline,
        null_prior=prior,
        temperature=tau,
    )


def _controller_aggregates(
    controller_real_log_evidence: Iterable[LogEvidence],
    *,
    temperature: float,
) -> list[float]:
    """Validate controller evidence for finite, identifiable calibration."""

    aggregates = [
        real_log_mean_evidence(values, temperature=temperature)
        for values in controller_real_log_evidence
    ]
    if not aggregates:
        raise ValueError("controller split must contain at least one question")
    if not all(math.isfinite(value) for value in aggregates):
        raise ValueError(
            "every controller question must have finite real evidence for "
            "finite null calibration"
        )
    return aggregates


def _validate_target_coverage(target_real_coverage: float) -> float:
    target = float(target_real_coverage)
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError(
            "target_real_coverage must be finite and strictly between zero "
            f"and one, got {target_real_coverage!r}"
        )
    return target


def _mean_soft_coverage(
    aggregates: Sequence[float],
    *,
    null_log_evidence: float,
    null_prior: float,
    temperature: float,
) -> float:
    return sum(
        _real_coverage_from_aggregate(
            aggregate,
            null_log_evidence=null_log_evidence,
            null_prior=null_prior,
            temperature=temperature,
        )
        for aggregate in aggregates
    ) / len(aggregates)


def calibrate_null_log_evidence(
    controller_real_log_evidence: Iterable[LogEvidence],
    *,
    target_real_coverage: float,
    null_prior: float = 0.5,
    temperature: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> NullCalibration:
    """Select a fixed null baseline from controller evidence only.

    No correctness labels, losses, rewards, or validation outcomes are accepted
    by this API.  The caller should supply a designated controller split,
    calibrate once, and freeze the returned baseline before evaluation.
    Bisection matches the requested *mean posterior real coverage*.
    """

    target = _validate_target_coverage(target_real_coverage)
    tau = _validate_temperature(temperature)
    prior = _validate_null_prior(null_prior)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    aggregates = _controller_aggregates(
        controller_real_log_evidence,
        temperature=tau,
    )

    prior_log_odds = math.log1p(-prior) - math.log(prior)
    # At these brackets, every controller coverage is approximately one or
    # zero respectively.  The 50-logit margin is far beyond calibration
    # tolerance while avoiding infinities.
    lower = min(aggregates) + tau * (prior_log_odds - 50.0)
    upper = max(aggregates) + tau * (prior_log_odds + 50.0)
    midpoint = (lower + upper) / 2.0
    achieved = _mean_soft_coverage(
        aggregates,
        null_log_evidence=midpoint,
        null_prior=prior,
        temperature=tau,
    )
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        midpoint = (lower + upper) / 2.0
        achieved = _mean_soft_coverage(
            aggregates,
            null_log_evidence=midpoint,
            null_prior=prior,
            temperature=tau,
        )
        if abs(achieved - target) <= tolerance:
            break
        # Larger null evidence monotonically reduces real coverage.
        if achieved > target:
            lower = midpoint
        else:
            upper = midpoint

    return NullCalibration(
        calibrated_parameter="null_log_evidence",
        null_log_evidence=midpoint,
        null_prior=prior,
        target_mean_real_coverage=target,
        achieved_mean_real_coverage=achieved,
        controller_questions=len(aggregates),
        iterations=iterations,
    )


def calibrate_null_prior(
    controller_real_log_evidence: Iterable[LogEvidence],
    *,
    target_real_coverage: float,
    null_log_evidence: float = 0.0,
    temperature: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> NullCalibration:
    """Select a frozen null prior from controller evidence only.

    As with :func:`calibrate_null_log_evidence`, this outcome-free interface
    cannot inspect correctness or validation risk.
    """

    target = _validate_target_coverage(target_real_coverage)
    tau = _validate_temperature(temperature)
    baseline = _validate_null_log_evidence(null_log_evidence)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    aggregates = _controller_aggregates(
        controller_real_log_evidence,
        temperature=tau,
    )
    offsets = [(value - baseline) / tau for value in aggregates]

    # Work in eta = log((1-pi_0)/pi_0), where coverage is sigmoid(eta+offset).
    lower_eta = -50.0 - max(offsets)
    upper_eta = 50.0 - min(offsets)
    eta = (lower_eta + upper_eta) / 2.0
    achieved = sum(_sigmoid(eta + value) for value in offsets) / len(offsets)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        eta = (lower_eta + upper_eta) / 2.0
        achieved = sum(_sigmoid(eta + value) for value in offsets) / len(offsets)
        if abs(achieved - target) <= tolerance:
            break
        if achieved < target:
            lower_eta = eta
        else:
            upper_eta = eta

    prior = _sigmoid(-eta)
    return NullCalibration(
        calibrated_parameter="null_prior",
        null_log_evidence=baseline,
        null_prior=prior,
        target_mean_real_coverage=target,
        achieved_mean_real_coverage=achieved,
        controller_questions=len(aggregates),
        iterations=iterations,
    )


def threshold_rejection_responsibilities(
    real_log_evidence: LogEvidence,
    *,
    threshold: float,
    temperature: float = 1.0,
) -> ThresholdRejectionResponsibilities:
    """Hard accept/reject comparator on the aggregate real log evidence.

    Accepted questions receive the ordinary conditional trace posterior;
    rejected questions receive exactly zero M-step coefficients.
    """

    tau = _validate_temperature(temperature)
    threshold_value = float(threshold)
    if math.isnan(threshold_value):
        raise ValueError("threshold cannot be NaN")
    conditional, aggregate = _conditional_real_distribution(
        real_log_evidence,
        temperature=tau,
    )
    accepted = aggregate >= threshold_value
    coverage = float(accepted)
    coefficients = (conditional * coverage).detach()
    conditional = conditional.detach()
    ess, normalized_ess = _conditional_ess(conditional)
    return ThresholdRejectionResponsibilities(
        m_step_coefficients=coefficients,
        conditional_real_weights=conditional,
        accepted=accepted,
        real_coverage=coverage,
        rejection_mass=1.0 - coverage,
        conditional_ess=ess,
        normalized_conditional_ess=normalized_ess,
        real_log_mean_evidence=aggregate,
        threshold=threshold_value,
        temperature=tau,
    )


def calibrate_rejection_threshold(
    controller_real_log_evidence: Iterable[LogEvidence],
    *,
    target_real_coverage: float,
    temperature: float = 1.0,
) -> ThresholdCalibration:
    """Choose the closest attainable hard coverage using controller scores only.

    Ties are indivisible.  If two attainable coverages are equally close, the
    lower (more conservative) coverage is selected.
    """

    target = _validate_target_coverage(target_real_coverage)
    tau = _validate_temperature(temperature)
    aggregates = _controller_aggregates(
        controller_real_log_evidence,
        temperature=tau,
    )
    unique_scores = sorted(set(aggregates), reverse=True)
    candidates: list[tuple[float, float]] = [
        (math.nextafter(unique_scores[0], math.inf), 0.0)
    ]
    for score in unique_scores:
        coverage = sum(value >= score for value in aggregates) / len(aggregates)
        candidates.append((score, coverage))
    threshold, achieved = min(
        candidates,
        key=lambda item: (
            abs(item[1] - target),
            item[1],  # conservative tie-break
        ),
    )
    return ThresholdCalibration(
        threshold=threshold,
        target_real_coverage=target,
        achieved_real_coverage=achieved,
        controller_questions=len(aggregates),
    )


def coverage_risk_curve_inputs(
    selection_scores: Union[Sequence[float], torch.Tensor],
    risks: Union[Sequence[float], torch.Tensor],
) -> CoverageRiskCurveInputs:
    """Build tie-aware held-out coverage--risk curve points.

    This evaluation helper is intentionally separate from calibration.  Risks
    may be held-out error indicators or nonnegative losses, but they never
    influence a null baseline, null prior, or rejection threshold.
    """

    score_values = _detached_log_evidence(selection_scores).cpu().tolist()
    risk_tensor = torch.as_tensor(risks)
    if risk_tensor.ndim != 1:
        raise ValueError("risks must be a one-dimensional vector")
    risk_values = risk_tensor.detach().to(dtype=torch.float64).cpu().tolist()
    if len(score_values) != len(risk_values):
        raise ValueError("selection_scores and risks must have the same length")
    if not all(math.isfinite(value) for value in score_values):
        raise ValueError("selection_scores must be finite")
    if not all(math.isfinite(value) and value >= 0.0 for value in risk_values):
        raise ValueError("risks must be finite and nonnegative")
    total = len(score_values)
    if total == 0:
        return CoverageRiskCurveInputs((), (), (), (), 0)

    order = sorted(range(total), key=lambda index: score_values[index], reverse=True)
    thresholds: list[float] = []
    retained_coverage: list[float] = []
    selective_risk: list[float] = []
    retained_counts: list[int] = []
    cumulative_risk = 0.0
    retained = 0
    position = 0
    while position < total:
        threshold = score_values[order[position]]
        while (
            position < total
            and score_values[order[position]] == threshold
        ):
            cumulative_risk += risk_values[order[position]]
            retained += 1
            position += 1
        thresholds.append(threshold)
        retained_counts.append(retained)
        retained_coverage.append(retained / total)
        selective_risk.append(cumulative_risk / retained)

    return CoverageRiskCurveInputs(
        thresholds=tuple(thresholds),
        retained_coverage=tuple(retained_coverage),
        selective_risk=tuple(selective_risk),
        retained_counts=tuple(retained_counts),
        total_questions=total,
    )


__all__ = [
    "CoverageRiskCurveInputs",
    "NullCalibration",
    "NullLatentResponsibilities",
    "ThresholdCalibration",
    "ThresholdRejectionResponsibilities",
    "calibrate_null_log_evidence",
    "calibrate_null_prior",
    "calibrate_rejection_threshold",
    "coverage_risk_curve_inputs",
    "null_latent_responsibilities",
    "real_log_mean_evidence",
    "threshold_rejection_responsibilities",
]
