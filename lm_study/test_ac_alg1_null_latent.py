"""CPU tests for the AC-ALG1 null-latent E-step."""

import math

import pytest
import torch

from ac_alg1_null_latent import (
    calibrate_null_log_evidence,
    calibrate_null_prior,
    calibrate_rejection_threshold,
    coverage_risk_curve_inputs,
    null_latent_responsibilities,
    threshold_rejection_responsibilities,
)


def test_augmented_posterior_conserves_mass_without_renormalizing_m_step():
    posterior = null_latent_responsibilities(
        torch.tensor([0.0, 0.0]),
        null_log_evidence=0.0,
        null_prior=0.5,
    )

    assert posterior.real_coverage == pytest.approx(0.5)
    assert posterior.null_mass == pytest.approx(0.5)
    assert posterior.m_step_coefficients.tolist() == pytest.approx([0.25, 0.25])
    assert posterior.conditional_real_weights.tolist() == pytest.approx([0.5, 0.5])
    assert float(posterior.m_step_coefficients.sum()) == pytest.approx(
        posterior.real_coverage
    )
    assert posterior.conditional_ess == pytest.approx(2.0)
    assert posterior.normalized_conditional_ess == pytest.approx(1.0)


def test_weak_real_evidence_abstains_instead_of_forcing_an_update():
    posterior = null_latent_responsibilities(
        [-1000.0, -1001.0],
        null_log_evidence=0.0,
        null_prior=0.5,
    )

    assert posterior.null_mass == pytest.approx(1.0)
    assert posterior.real_coverage < 1e-300
    assert float(posterior.m_step_coefficients.sum()) < 1e-300
    # Conditional weights still exist for diagnostics, but are not the M-step
    # coefficients and therefore cannot restore unit update mass.
    assert float(posterior.conditional_real_weights.sum()) == pytest.approx(1.0)


def test_extreme_evidence_and_empty_support_are_stable():
    strong = null_latent_responsibilities(
        [10000.0, -10000.0],
        null_log_evidence=0.0,
    )
    empty = null_latent_responsibilities([], null_log_evidence=0.0)
    impossible = null_latent_responsibilities(
        [float("-inf"), float("-inf")],
        null_log_evidence=0.0,
    )
    positive_infinities = null_latent_responsibilities(
        [float("inf"), 0.0, float("inf")],
        null_log_evidence=0.0,
    )

    assert strong.real_coverage == pytest.approx(1.0)
    assert strong.m_step_coefficients.tolist() == pytest.approx([1.0, 0.0])
    for posterior in (empty, impossible):
        assert posterior.real_coverage == 0.0
        assert posterior.null_mass == 1.0
        assert float(posterior.m_step_coefficients.sum()) == 0.0
        assert posterior.conditional_ess == 0.0
    assert positive_infinities.real_coverage == 1.0
    assert positive_infinities.m_step_coefficients.tolist() == pytest.approx(
        [0.5, 0.0, 0.5]
    )


def test_augmented_posterior_is_shift_permutation_and_replication_invariant():
    original = null_latent_responsibilities(
        [-2.0, 1.0],
        null_log_evidence=0.25,
        null_prior=0.3,
        temperature=1.7,
    )
    shifted = null_latent_responsibilities(
        [8.0, 11.0],
        null_log_evidence=10.25,
        null_prior=0.3,
        temperature=1.7,
    )
    permuted = null_latent_responsibilities(
        [1.0, -2.0],
        null_log_evidence=0.25,
        null_prior=0.3,
        temperature=1.7,
    )
    replicated = null_latent_responsibilities(
        [-2.0, 1.0, -2.0, 1.0],
        null_log_evidence=0.25,
        null_prior=0.3,
        temperature=1.7,
    )

    assert shifted.real_coverage == pytest.approx(original.real_coverage)
    assert shifted.m_step_coefficients.tolist() == pytest.approx(
        original.m_step_coefficients.tolist()
    )
    assert permuted.real_coverage == pytest.approx(original.real_coverage)
    assert permuted.m_step_coefficients.tolist() == pytest.approx(
        list(reversed(original.m_step_coefficients.tolist()))
    )
    assert replicated.real_coverage == pytest.approx(original.real_coverage)
    assert replicated.m_step_coefficients.tolist() == pytest.approx(
        [
            original.m_step_coefficients[0].item() / 2,
            original.m_step_coefficients[1].item() / 2,
            original.m_step_coefficients[0].item() / 2,
            original.m_step_coefficients[1].item() / 2,
        ]
    )


def test_responsibilities_are_detached_e_step_constants():
    evidence = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)

    posterior = null_latent_responsibilities(evidence)

    assert not posterior.m_step_coefficients.requires_grad
    assert posterior.m_step_coefficients.grad_fn is None
    assert not posterior.conditional_real_weights.requires_grad
    assert evidence.grad is None


def test_controller_calibration_matches_mean_coverage_without_outcomes():
    controller = [
        torch.tensor([-2.0, -1.0]),
        torch.tensor([0.0, 0.5]),
        torch.tensor([2.0, 3.0]),
    ]

    baseline_fit = calibrate_null_log_evidence(
        controller,
        target_real_coverage=0.4,
        null_prior=0.35,
        temperature=1.3,
    )
    prior_fit = calibrate_null_prior(
        controller,
        target_real_coverage=0.7,
        null_log_evidence=0.25,
        temperature=1.3,
    )

    assert baseline_fit.calibrated_parameter == "null_log_evidence"
    assert baseline_fit.achieved_mean_real_coverage == pytest.approx(
        0.4, abs=1e-9
    )
    assert baseline_fit.controller_questions == 3
    assert prior_fit.calibrated_parameter == "null_prior"
    assert 0.0 < prior_fit.null_prior < 1.0
    assert prior_fit.achieved_mean_real_coverage == pytest.approx(
        0.7, abs=1e-9
    )


def test_null_baseline_calibration_is_translation_equivariant():
    controller = [[-2.0, 0.0], [1.0, 3.0]]
    shifted = [[value + 17.0 for value in row] for row in controller]

    original = calibrate_null_log_evidence(
        controller,
        target_real_coverage=0.6,
    )
    translated = calibrate_null_log_evidence(
        shifted,
        target_real_coverage=0.6,
    )

    assert translated.null_log_evidence == pytest.approx(
        original.null_log_evidence + 17.0,
        abs=1e-9,
    )


def test_hard_threshold_comparator_and_controller_calibration():
    rejected = threshold_rejection_responsibilities(
        [-3.0, -2.0],
        threshold=-1.0,
    )
    accepted = threshold_rejection_responsibilities(
        [-3.0, -2.0],
        threshold=-3.0,
    )
    calibration = calibrate_rejection_threshold(
        [[3.0], [2.0], [2.0], [-1.0]],
        target_real_coverage=0.5,
    )

    assert not rejected.accepted
    assert rejected.rejection_mass == 1.0
    assert rejected.m_step_coefficients.tolist() == pytest.approx([0.0, 0.0])
    assert accepted.accepted
    assert accepted.real_coverage == 1.0
    assert float(accepted.m_step_coefficients.sum()) == pytest.approx(1.0)
    # The tied score 2.0 is indivisible.  The two closest coverages are 1/4
    # and 3/4, so the documented conservative tie-break chooses 1/4.
    assert calibration.threshold == pytest.approx(3.0)
    assert calibration.achieved_real_coverage == pytest.approx(0.25)


def test_coverage_risk_inputs_group_ties_and_never_calibrate_with_risk():
    curve = coverage_risk_curve_inputs(
        selection_scores=[0.9, 0.9, 0.2],
        risks=[0.0, 1.0, 1.0],
    )

    assert curve.thresholds == pytest.approx((0.9, 0.2))
    assert curve.retained_counts == (2, 3)
    assert curve.retained_coverage == pytest.approx((2 / 3, 1.0))
    assert curve.selective_risk == pytest.approx((0.5, 2 / 3))
    assert curve.total_questions == 3


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"null_prior": 0.0}, "null_prior"),
        ({"null_prior": 1.0}, "null_prior"),
        ({"null_log_evidence": math.inf}, "null_log_evidence"),
        ({"temperature": 0.0}, "temperature"),
    ],
)
def test_invalid_posterior_controls_fail_closed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        null_latent_responsibilities([0.0], **kwargs)


def test_nan_real_evidence_fails_closed():
    with pytest.raises(ValueError, match="NaN"):
        null_latent_responsibilities([0.0, math.nan])
