r"""Calibrated Bayesian fusion of discrete verifier observations.

This module is deliberately independent of the model runtime.  A frozen
calibration bank supplies ``(correct, observation)`` pairs for each verifier,
where an observation is ``pass``, ``fail`` or ``missing``.  For the definitive
observations, a Jeffreys ``Beta(1/2, 1/2)`` prior gives finite posterior-
predictive probabilities even when a calibration cell is empty.

For verifier ``k`` and observation ``o`` the raw evidence is

.. math::

    b_k(o) = \log \frac{P(o \mid C=1)}{P(o \mid C=0)}.

The experiment shrinks this by the amount of definitive calibration evidence,
``n_k / (n_k + 64)``, and caps the resulting contribution at ``+/- log(4)``.
Missing evidence is exactly neutral.  Independent source contributions are
added to a prior validity logit and transformed back to a probability.

The count-only JSON representation is canonical and deterministic.  Derived
probabilities and Bayes factors are always recomputed on load, so a calibration
artifact has one small, auditable source of truth.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any


OBSERVATIONS = ("pass", "fail", "missing")
CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_METHOD = "bayesian_verifier_fusion"
DEFAULT_BETA_ALPHA = 0.5
DEFAULT_PRECISION_PRIOR = 64.0
DEFAULT_MAX_ABS_LOG_BAYES_FACTOR = math.log(4.0)


def _validate_count(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validate_positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_observation(observation: str) -> str:
    if observation not in OBSERVATIONS:
        raise ValueError(
            f"observation must be one of {OBSERVATIONS}, got {observation!r}"
        )
    return observation


def _validate_source_name(source: str) -> str:
    if not isinstance(source, str) or not source or source.strip() != source:
        raise ValueError("source must be a nonempty, whitespace-trimmed string")
    return source


def _validate_probability(probability: float, *, name: str) -> float:
    result = float(probability)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


@dataclass(frozen=True)
class CalibrationCounts:
    """Sufficient calibration counts for one verifier source.

    Missing rows are retained for provenance and coverage reporting, but they
    do not enter the definitive Beta-Bernoulli fit and have zero evidence at
    application time.
    """

    correct_pass: int = 0
    correct_fail: int = 0
    correct_missing: int = 0
    incorrect_pass: int = 0
    incorrect_fail: int = 0
    incorrect_missing: int = 0

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            _validate_count(getattr(self, field_name), name=field_name)

    @property
    def correct_definitive(self) -> int:
        return self.correct_pass + self.correct_fail

    @property
    def incorrect_definitive(self) -> int:
        return self.incorrect_pass + self.incorrect_fail

    @property
    def definitive(self) -> int:
        return self.correct_definitive + self.incorrect_definitive

    @property
    def missing(self) -> int:
        return self.correct_missing + self.incorrect_missing

    @property
    def total(self) -> int:
        return self.definitive + self.missing

    @classmethod
    def from_labelled(
        cls,
        examples: Iterable[tuple[bool, str]],
    ) -> "CalibrationCounts":
        """Count an iterable of ``(strictly_correct, observation)`` pairs."""

        counts = {field_name: 0 for field_name in cls.__dataclass_fields__}
        for correct, observation in examples:
            if not isinstance(correct, bool):
                raise ValueError("calibration correctness labels must be bool")
            state = _validate_observation(observation)
            prefix = "correct" if correct else "incorrect"
            counts[f"{prefix}_{state}"] += 1
        return cls(**counts)

    def to_dict(self) -> dict[str, int]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationCounts":
        if not isinstance(payload, Mapping):
            raise ValueError("calibration counts must be a mapping")
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError(
                "calibration count keys must be exactly "
                f"{sorted(expected)}, got {sorted(payload)}"
            )
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SourceCalibration:
    """Derived finite evidence for one verifier source."""

    source: str
    counts: CalibrationCounts
    precision_weight: float
    pass_probability_if_correct: float
    pass_probability_if_incorrect: float
    raw_pass_log_bayes_factor: float
    raw_fail_log_bayes_factor: float
    pass_log_bayes_factor: float
    fail_log_bayes_factor: float

    def log_bayes_factor(self, observation: str) -> float:
        state = _validate_observation(observation)
        if state == "pass":
            return self.pass_log_bayes_factor
        if state == "fail":
            return self.fail_log_bayes_factor
        return 0.0


@dataclass(frozen=True)
class FusionCalibration:
    """Frozen calibration shared by every paired training seed."""

    sources: tuple[SourceCalibration, ...]
    beta_alpha: float = DEFAULT_BETA_ALPHA
    precision_prior: float = DEFAULT_PRECISION_PRIOR
    max_abs_log_bayes_factor: float = DEFAULT_MAX_ABS_LOG_BAYES_FACTOR

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("fusion calibration requires at least one source")
        names = tuple(source.source for source in self.sources)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("calibration sources must be unique and sorted")
        _validate_positive_finite(self.beta_alpha, name="beta_alpha")
        _validate_positive_finite(self.precision_prior, name="precision_prior")
        _validate_positive_finite(
            self.max_abs_log_bayes_factor,
            name="max_abs_log_bayes_factor",
        )

    def source(self, name: str) -> SourceCalibration:
        wanted = _validate_source_name(name)
        for source in self.sources:
            if source.source == wanted:
                return source
        raise KeyError(f"unknown verifier source {wanted!r}")


@dataclass(frozen=True)
class FusionResult:
    """Trace-level fusion result with auditable source contributions."""

    prior_probability: float
    prior_logit: float
    source_log_evidence: tuple[tuple[str, float], ...]
    total_log_evidence: float
    posterior_logit: float
    validity_probability: float


def fit_source_calibration(
    source: str,
    counts: CalibrationCounts,
    *,
    beta_alpha: float = DEFAULT_BETA_ALPHA,
    precision_prior: float = DEFAULT_PRECISION_PRIOR,
    max_abs_log_bayes_factor: float = DEFAULT_MAX_ABS_LOG_BAYES_FACTOR,
) -> SourceCalibration:
    """Fit finite, shrunk pass/fail evidence from labelled counts."""

    source_name = _validate_source_name(source)
    if not isinstance(counts, CalibrationCounts):
        raise TypeError("counts must be CalibrationCounts")
    alpha = _validate_positive_finite(beta_alpha, name="beta_alpha")
    precision_scale = _validate_positive_finite(
        precision_prior,
        name="precision_prior",
    )
    evidence_cap = _validate_positive_finite(
        max_abs_log_bayes_factor,
        name="max_abs_log_bayes_factor",
    )

    pass_if_correct = (counts.correct_pass + alpha) / (
        counts.correct_definitive + 2.0 * alpha
    )
    pass_if_incorrect = (counts.incorrect_pass + alpha) / (
        counts.incorrect_definitive + 2.0 * alpha
    )
    raw_pass = math.log(pass_if_correct) - math.log(pass_if_incorrect)
    raw_fail = math.log1p(-pass_if_correct) - math.log1p(-pass_if_incorrect)
    precision_weight = counts.definitive / (counts.definitive + precision_scale)

    def shrink_and_cap(value: float) -> float:
        weighted = precision_weight * value
        return min(evidence_cap, max(-evidence_cap, weighted))

    result = SourceCalibration(
        source=source_name,
        counts=counts,
        precision_weight=precision_weight,
        pass_probability_if_correct=pass_if_correct,
        pass_probability_if_incorrect=pass_if_incorrect,
        raw_pass_log_bayes_factor=raw_pass,
        raw_fail_log_bayes_factor=raw_fail,
        pass_log_bayes_factor=shrink_and_cap(raw_pass),
        fail_log_bayes_factor=shrink_and_cap(raw_fail),
    )
    numeric = (
        result.precision_weight,
        result.pass_probability_if_correct,
        result.pass_probability_if_incorrect,
        result.raw_pass_log_bayes_factor,
        result.raw_fail_log_bayes_factor,
        result.pass_log_bayes_factor,
        result.fail_log_bayes_factor,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError("calibration produced nonfinite verifier evidence")
    return result


def fit_fusion_calibration(
    counts_by_source: Mapping[str, CalibrationCounts],
    *,
    beta_alpha: float = DEFAULT_BETA_ALPHA,
    precision_prior: float = DEFAULT_PRECISION_PRIOR,
    max_abs_log_bayes_factor: float = DEFAULT_MAX_ABS_LOG_BAYES_FACTOR,
) -> FusionCalibration:
    """Fit every source in deterministic lexical order."""

    if not isinstance(counts_by_source, Mapping) or not counts_by_source:
        raise ValueError("counts_by_source must be a nonempty mapping")
    sources = tuple(
        fit_source_calibration(
            source,
            counts_by_source[source],
            beta_alpha=beta_alpha,
            precision_prior=precision_prior,
            max_abs_log_bayes_factor=max_abs_log_bayes_factor,
        )
        for source in sorted(counts_by_source)
    )
    return FusionCalibration(
        sources=sources,
        beta_alpha=float(beta_alpha),
        precision_prior=float(precision_prior),
        max_abs_log_bayes_factor=float(max_abs_log_bayes_factor),
    )


def _logit(probability: float) -> float:
    value = _validate_probability(probability, name="prior_probability")
    return math.log(value) - math.log1p(-value)


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def fuse_validity(
    prior_probability: float,
    observations: Mapping[str, str],
    calibration: FusionCalibration,
) -> FusionResult:
    """Fuse verifier observations into a calibrated validity probability.

    Omitted calibrated sources are treated as missing and therefore neutral.
    Unknown supplied sources fail closed to catch configuration mistakes.
    """

    if not isinstance(calibration, FusionCalibration):
        raise TypeError("calibration must be FusionCalibration")
    if not isinstance(observations, Mapping):
        raise TypeError("observations must be a mapping")
    known = {source.source for source in calibration.sources}
    unknown = set(observations) - known
    if unknown:
        raise ValueError(f"observations contain unknown sources: {sorted(unknown)}")

    prior = _validate_probability(prior_probability, name="prior_probability")
    prior_logit = _logit(prior)
    contributions: list[tuple[str, float]] = []
    for source in calibration.sources:
        observation = observations.get(source.source, "missing")
        contribution = source.log_bayes_factor(observation)
        contributions.append((source.source, contribution))
    total_evidence = math.fsum(value for _source, value in contributions)
    posterior_logit = prior_logit + total_evidence
    probability = _sigmoid(posterior_logit)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise RuntimeError("Bayesian verifier fusion produced an invalid probability")
    return FusionResult(
        prior_probability=prior,
        prior_logit=prior_logit,
        source_log_evidence=tuple(contributions),
        total_log_evidence=total_evidence,
        posterior_logit=posterior_logit,
        validity_probability=probability,
    )


def fused_validity_probability(
    prior_probability: float,
    observations: Mapping[str, str],
    calibration: FusionCalibration,
) -> float:
    """Convenience wrapper returning only the posterior probability."""

    return fuse_validity(
        prior_probability,
        observations,
        calibration,
    ).validity_probability


def calibration_to_payload(calibration: FusionCalibration) -> dict[str, Any]:
    """Return the count-only schema-1 calibration payload."""

    if not isinstance(calibration, FusionCalibration):
        raise TypeError("calibration must be FusionCalibration")
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "method": CALIBRATION_METHOD,
        "beta_alpha": calibration.beta_alpha,
        "precision_prior": calibration.precision_prior,
        "max_abs_log_bayes_factor": calibration.max_abs_log_bayes_factor,
        "sources": [
            {"source": source.source, "counts": source.counts.to_dict()}
            for source in calibration.sources
        ],
    }


def calibration_from_payload(payload: Mapping[str, Any]) -> FusionCalibration:
    """Validate a schema-1 payload exactly and recompute all derived values."""

    if not isinstance(payload, Mapping):
        raise ValueError("calibration payload must be a mapping")
    expected = {
        "schema_version",
        "method",
        "beta_alpha",
        "precision_prior",
        "max_abs_log_bayes_factor",
        "sources",
    }
    if set(payload) != expected:
        raise ValueError(
            f"calibration payload keys must be exactly {sorted(expected)}"
        )
    if payload["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported Bayesian-fusion calibration schema")
    if payload["method"] != CALIBRATION_METHOD:
        raise ValueError("unexpected calibration method")
    source_rows = payload["sources"]
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("calibration sources must be a nonempty list")
    counts_by_source: dict[str, CalibrationCounts] = {}
    for row in source_rows:
        if not isinstance(row, Mapping) or set(row) != {"source", "counts"}:
            raise ValueError("each calibration source needs source and counts")
        source = _validate_source_name(row["source"])
        if source in counts_by_source:
            raise ValueError(f"duplicate calibration source {source!r}")
        counts_by_source[source] = CalibrationCounts.from_dict(row["counts"])
    if [row["source"] for row in source_rows] != sorted(counts_by_source):
        raise ValueError("calibration source rows must be lexically sorted")
    return fit_fusion_calibration(
        counts_by_source,
        beta_alpha=payload["beta_alpha"],
        precision_prior=payload["precision_prior"],
        max_abs_log_bayes_factor=payload["max_abs_log_bayes_factor"],
    )


def canonical_calibration_json(calibration: FusionCalibration) -> bytes:
    """Serialize deterministically as UTF-8 JSON terminated by one newline."""

    encoded = json.dumps(
        calibration_to_payload(calibration),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return (encoded + "\n").encode("utf-8")


def calibration_sha256(calibration: FusionCalibration) -> str:
    """Hash the exact canonical calibration bytes."""

    return hashlib.sha256(canonical_calibration_json(calibration)).hexdigest()


@lru_cache(maxsize=8)
def load_fusion_calibration(path: str | Path) -> FusionCalibration:
    """Load the canonical count artifact and recompute every derived value."""

    calibration_path = Path(path).expanduser()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration = calibration_from_payload(payload)
    if calibration_path.read_bytes() != canonical_calibration_json(calibration):
        raise ValueError("Bayesian-fusion calibration is not canonical JSON")
    return calibration
