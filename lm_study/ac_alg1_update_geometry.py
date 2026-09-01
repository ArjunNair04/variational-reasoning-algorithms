"""Pure gradient-geometry and rollback rules for safeguarded AC-ALG1 M-steps.

The AC-ALG1 objectives are maximised, but the trainer backpropagates their
negatives.  Every gradient handled here is therefore a *loss* gradient.  An
optimizer step along ``-g`` improves a component to first order when its loss
gradient has positive inner product with ``g``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import torch


COMPONENT_NAMES = ("B_sup", "B_prime_unsup", "B_unsup")
UPDATE_GEOMETRIES = ("sum", "mgda", "normalized_mgda", "answer_primary")
STEP_ACCEPTANCE_MODES = ("none", "total", "componentwise", "answer_primary")


def _subtract_gradients(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    if len(left) != len(right):
        raise ValueError("gradient snapshots must contain the same parameters")
    result = []
    for left_tensor, right_tensor in zip(left, right):
        if left_tensor is None and right_tensor is None:
            result.append(None)
        elif left_tensor is None:
            result.append(-right_tensor)
        elif right_tensor is None:
            result.append(left_tensor.clone())
        else:
            result.append(left_tensor - right_tensor)
    return result


def component_gradients_from_cumulative(
    supervised: Sequence[torch.Tensor | None],
    after_labelled: Sequence[torch.Tensor | None],
    total: Sequence[torch.Tensor | None],
) -> dict[str, list[torch.Tensor | None]]:
    """Recover the three component loss gradients from cumulative snapshots."""

    if not (len(supervised) == len(after_labelled) == len(total)):
        raise ValueError("gradient snapshots must contain the same parameters")
    return {
        "B_sup": [
            None if tensor is None else tensor.clone()
            for tensor in supervised
        ],
        "B_prime_unsup": _subtract_gradients(after_labelled, supervised),
        "B_unsup": _subtract_gradients(total, after_labelled),
    }


def gradient_dot(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> float:
    """Stable float32 inner product across a parameter-aligned gradient list."""

    if len(left) != len(right):
        raise ValueError("gradient lists must contain the same parameters")
    value = 0.0
    for left_tensor, right_tensor in zip(left, right):
        if left_tensor is None or right_tensor is None:
            continue
        value += float(torch.sum(left_tensor.detach().float() * right_tensor.detach().float()))
    if not math.isfinite(value):
        raise FloatingPointError("non-finite gradient inner product")
    return value


def gradient_norm(gradients: Sequence[torch.Tensor | None]) -> float:
    """Stable float32 L2 norm across a parameter-aligned gradient list."""

    return math.sqrt(max(gradient_dot(gradients, gradients), 0.0))


def _scale_gradients(
    gradients: Sequence[torch.Tensor | None],
    scale: float,
) -> list[torch.Tensor | None]:
    return [
        None if tensor is None else tensor * scale
        for tensor in gradients
    ]


def _add_scaled_gradients(
    terms: Sequence[tuple[Sequence[torch.Tensor | None], float]],
) -> list[torch.Tensor | None]:
    if not terms:
        return []
    width = len(terms[0][0])
    if any(len(gradients) != width for gradients, _scale in terms):
        raise ValueError("gradient lists must contain the same parameters")

    combined: list[torch.Tensor | None] = []
    for parameter_index in range(width):
        value = None
        for gradients, scale in terms:
            tensor = gradients[parameter_index]
            if tensor is None or scale == 0.0:
                continue
            contribution = tensor * scale
            value = contribution if value is None else value + contribution
        combined.append(value)
    return combined


def minimum_norm_weights(
    gradients: Sequence[Sequence[torch.Tensor | None]],
    tolerance: float = 1e-12,
) -> list[float]:
    """Solve the MGDA minimum-norm convex-combination problem exactly for <=3 terms.

    The active-set enumeration solves

        min_alpha ||sum_i alpha_i g_i||^2
        subject to alpha_i >= 0 and sum_i alpha_i = 1.

    AC-ALG1 has exactly three objective components, so enumerating every face of
    the simplex is both simpler and more reliable than an iterative QP solver.
    """

    count = len(gradients)
    if count == 0:
        return []
    if count > 3:
        raise ValueError("the exact active-set solver supports at most three gradients")
    if any(len(gradient) != len(gradients[0]) for gradient in gradients):
        raise ValueError("gradient lists must contain the same parameters")

    gram = torch.empty((count, count), dtype=torch.float64)
    for left in range(count):
        for right in range(left, count):
            dot = gradient_dot(gradients[left], gradients[right])
            gram[left, right] = dot
            gram[right, left] = dot

    best_weights = None
    best_value = math.inf
    indices = range(count)
    for active_count in range(1, count + 1):
        for active in itertools.combinations(indices, active_count):
            submatrix = gram[list(active)][:, list(active)]
            # Solve the equality-constrained quadratic KKT system with a
            # pseudoinverse.  The augmented system, unlike G^-1 1, correctly
            # handles singular faces whose convex hull contains zero (for
            # example two exactly opposing gradients).
            kkt = torch.zeros(
                (active_count + 1, active_count + 1),
                dtype=torch.float64,
            )
            kkt[:active_count, :active_count] = 2.0 * submatrix
            kkt[:active_count, active_count] = 1.0
            kkt[active_count, :active_count] = 1.0
            rhs = torch.zeros(active_count + 1, dtype=torch.float64)
            rhs[active_count] = 1.0
            solution = torch.linalg.pinv(kkt) @ rhs
            active_weights = solution[:active_count]
            if (
                not bool(torch.all(torch.isfinite(active_weights)))
                or abs(float(active_weights.sum()) - 1.0) > 1e-8
            ):
                continue
            if bool(torch.any(active_weights < -tolerance)):
                continue
            active_weights = active_weights.clamp_min(0.0)
            active_weights /= active_weights.sum()
            weights = torch.zeros(count, dtype=torch.float64)
            weights[list(active)] = active_weights
            value = float(weights @ gram @ weights)
            if math.isfinite(value) and value < best_value:
                best_value = value
                best_weights = weights

    if best_weights is None:
        # Every simplex vertex is feasible, so this is reachable only after a
        # severe numerical failure.  Make it explicit rather than silently
        # returning an arbitrary direction.
        raise FloatingPointError("could not solve the MGDA convex-combination problem")
    return [float(weight) for weight in best_weights]


def combine_component_gradients(
    components: Mapping[str, Sequence[torch.Tensor | None]],
    mode: str,
    epsilon: float = 1e-12,
) -> tuple[list[torch.Tensor | None], dict[str, object]]:
    """Construct the requested loss-gradient direction and its audit metadata."""

    if mode not in UPDATE_GEOMETRIES:
        raise ValueError(f"unknown update geometry {mode!r}")
    unknown = set(components) - set(COMPONENT_NAMES)
    if unknown:
        raise ValueError(f"unknown objective components: {sorted(unknown)}")

    norms = {
        name: gradient_norm(components[name])
        for name in COMPONENT_NAMES
        if name in components
    }
    active = [
        name for name in COMPONENT_NAMES
        if name in components and norms[name] > epsilon
    ]
    width = len(next(iter(components.values()))) if components else 0
    zero_direction = [None] * width
    if not active:
        return zero_direction, {
            "mode": mode,
            "active_components": [],
            "component_norms": norms,
            "coefficients": {},
            "direction_norm": 0.0,
        }

    if mode == "sum":
        coefficients = {name: 1.0 for name in active}
        direction = _add_scaled_gradients([
            (components[name], coefficients[name]) for name in active
        ])
    elif mode in ("mgda", "normalized_mgda"):
        normalized = mode == "normalized_mgda"
        search_gradients = [
            (
                _scale_gradients(components[name], 1.0 / norms[name])
                if normalized else list(components[name])
            )
            for name in active
        ]
        simplex_weights = minimum_norm_weights(search_gradients)
        coefficients = {
            name: (
                simplex_weight / norms[name]
                if normalized else simplex_weight
            )
            for name, simplex_weight in zip(active, simplex_weights)
        }
        direction = _add_scaled_gradients([
            (components[name], coefficients[name]) for name in active
        ])
    else:
        primary_name = "B_unsup"
        if primary_name not in active:
            return zero_direction, {
                "mode": mode,
                "active_components": active,
                "component_norms": norms,
                "coefficients": {},
                "direction_norm": 0.0,
                "skipped_reason": "answer-primary gradient unavailable",
            }

        primary = list(components[primary_name])
        primary_norm = norms[primary_name]
        adjusted_auxiliaries = []
        projection_coefficients = {}
        for name in active:
            if name == primary_name:
                continue
            auxiliary = list(components[name])
            dot = gradient_dot(auxiliary, primary)
            projection = min(dot / (primary_norm * primary_norm), 0.0)
            adjusted = _add_scaled_gradients([
                (auxiliary, 1.0),
                (primary, -projection),
            ])
            adjusted_auxiliaries.append((name, adjusted))
            projection_coefficients[name] = projection

        auxiliary_norm_sum = sum(
            gradient_norm(gradients) for _name, gradients in adjusted_auxiliaries
        )
        auxiliary_scale = (
            min(1.0, primary_norm / auxiliary_norm_sum)
            if auxiliary_norm_sum > epsilon else 0.0
        )
        coefficients = {primary_name: 1.0}
        coefficients.update({
            name: auxiliary_scale for name, _gradients in adjusted_auxiliaries
        })
        direction = _add_scaled_gradients(
            [(primary, 1.0)]
            + [
                (gradients, auxiliary_scale)
                for _name, gradients in adjusted_auxiliaries
            ]
        )
        metadata = {
            "projection_coefficients": projection_coefficients,
            "auxiliary_scale": auxiliary_scale,
        }

    result = {
        "mode": mode,
        "active_components": active,
        "component_norms": norms,
        "coefficients": coefficients,
        "direction_norm": gradient_norm(direction),
    }
    if mode == "answer_primary":
        result.update(metadata)
    return direction, result


def assign_trainable_gradients(
    model,
    gradients: Sequence[torch.Tensor | None],
) -> None:
    """Replace trainable parameter gradients with a computed direction."""

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(parameters) != len(gradients):
        raise ValueError("gradient direction must match trainable model parameters")
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                parameter.grad = None
            elif parameter.grad is None:
                parameter.grad = gradient.detach().clone()
            else:
                parameter.grad.copy_(gradient)


def fixed_surrogate_acceptance(
    before: Mapping[str, float],
    after: Mapping[str, float],
    mode: str,
    active_components: Sequence[str],
    tolerance: float,
) -> tuple[bool, dict[str, object]]:
    """Apply a post-step no-harm rule to the fixed-responsibility surrogate."""

    if mode not in STEP_ACCEPTANCE_MODES:
        raise ValueError(f"unknown step acceptance mode {mode!r}")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("rollback tolerance must be finite and nonnegative")
    if set(before) != set(after):
        raise ValueError("before/after surrogate values must have identical keys")

    deltas = {
        name: float(after[name]) - float(before[name])
        for name in before
    }
    finite = all(
        math.isfinite(float(value))
        for values in (before, after)
        for value in values.values()
    )
    total_delta = sum(deltas[name] for name in COMPONENT_NAMES)
    checks: dict[str, bool] = {"finite": finite}
    if mode == "none":
        accepted = finite
    elif mode == "total":
        checks["total"] = total_delta >= -tolerance
        accepted = finite and checks["total"]
    elif mode == "componentwise":
        for name in active_components:
            checks[name] = deltas[name] >= -tolerance
        accepted = finite and all(checks.values())
    else:
        checks["total"] = total_delta >= -tolerance
        checks["B_unsup"] = deltas["B_unsup"] >= -tolerance
        accepted = finite and checks["total"] and checks["B_unsup"]

    return accepted, {
        "mode": mode,
        "accepted": accepted,
        "deltas": deltas,
        "total_delta": total_delta,
        "checks": checks,
    }
