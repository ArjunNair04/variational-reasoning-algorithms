"""Weight concentration and the one-sided temperature intervention.

These counts describe a distribution over candidate traces. They do not count
independent optimizer updates or assess whether the written calculations are valid.
The original float32 PyTorch implementation is in each relevant source snapshot.
"""
from __future__ import annotations
import numpy as np


def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('weights must be a finite nonnegative vector')
    if not np.isclose(weights.sum(), 1):
        raise ValueError('weights must sum to one')
    return float(1 / np.sum(weights ** 2))


def entropy(weights):
    effective_sample_size(weights)  # same probability-vector contract
    weights = np.asarray(weights, dtype=float)
    positive = weights > 0
    return float(-np.sum(weights[positive] * np.log(weights[positive])))


def temperature_weights(logits, *, base_temperature=1.0, ess_floor_fraction=0.0):
    """Raise temperature only if needed; preserve negative-infinity exclusions.

    Returns (weights, effective temperature). The same 32 doublings and 40
    bisections as the executed controller are used. At a floor of one, exact
    equality may require infinite temperature; the bounded controller returns
    its largest tested temperature if it cannot bracket the requested floor.
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 1 or np.isnan(logits).any() or np.isposinf(logits).any():
        raise ValueError('logits must be finite or negative infinity')
    finite = np.isfinite(logits)
    if not finite.any():
        raise ValueError('at least one finite logit is needed')
    if not np.isfinite(base_temperature) or base_temperature <= 0:
        raise ValueError('base_temperature must be finite and positive')
    if not np.isfinite(ess_floor_fraction) or not 0 <= ess_floor_fraction <= 1:
        raise ValueError('ess_floor_fraction must be in [0, 1]')

    def at(temperature):
        mass = np.zeros_like(logits)
        mass[finite] = np.exp((logits[finite] - logits[finite].max()) / temperature)
        return mass / mass.sum()

    low = high = float(base_temperature)
    target = max(1, ess_floor_fraction * finite.sum())
    weights = at(low)
    if effective_sample_size(weights) >= target:
        return weights, low
    for _ in range(32):
        high *= 2
        if effective_sample_size(at(high)) >= target:
            break
    else:
        return at(high), high
    for _ in range(40):
        middle = (low + high) / 2
        if effective_sample_size(at(middle)) >= target:
            high = middle
        else:
            low = middle
    return at(high), high
