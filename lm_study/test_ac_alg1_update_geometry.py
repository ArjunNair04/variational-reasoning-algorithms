"""CPU-only tests for safeguarded AC-ALG1 update geometry."""

import math

import pytest
import torch

import ac_alg1
from ac_alg1_update_geometry import (
    combine_component_gradients,
    component_gradients_from_cumulative,
    fixed_surrogate_acceptance,
    gradient_dot,
    minimum_norm_weights,
)


def _gradient(*values):
    return [torch.tensor(values, dtype=torch.float32)]


def test_component_gradients_are_recovered_from_cumulative_snapshots():
    components = component_gradients_from_cumulative(
        _gradient(1.0, 0.0),
        _gradient(3.0, 1.0),
        _gradient(2.0, 5.0),
    )

    assert torch.equal(components["B_sup"][0], torch.tensor([1.0, 0.0]))
    assert torch.equal(
        components["B_prime_unsup"][0], torch.tensor([2.0, 1.0])
    )
    assert torch.equal(components["B_unsup"][0], torch.tensor([-1.0, 4.0]))


def test_minimum_norm_weights_find_the_simplex_interior():
    weights = minimum_norm_weights([
        _gradient(1.0, 0.0),
        _gradient(-1.0, 0.0),
    ])

    assert weights == pytest.approx([0.5, 0.5], abs=1e-8)


@pytest.mark.parametrize("mode", ["mgda", "normalized_mgda"])
def test_mgda_direction_is_common_descent_when_not_pareto_stationary(mode):
    components = {
        "B_sup": _gradient(1.0, 0.0),
        "B_prime_unsup": _gradient(1.0, 1.0),
        "B_unsup": _gradient(2.0, -1.0),
    }

    direction, metadata = combine_component_gradients(components, mode)

    assert metadata["direction_norm"] > 0
    for gradients in components.values():
        assert gradient_dot(direction, gradients) > 0


def test_normalized_mgda_removes_gradient_scale_domination():
    components = {
        "B_sup": _gradient(100.0, 0.0),
        "B_unsup": _gradient(0.0, 1.0),
    }

    direction, metadata = combine_component_gradients(
        components, "normalized_mgda"
    )

    assert direction[0].tolist() == pytest.approx([0.5, 0.5], abs=1e-6)
    assert metadata["coefficients"]["B_sup"] == pytest.approx(0.005)
    assert metadata["coefficients"]["B_unsup"] == pytest.approx(0.5)


def test_answer_primary_projects_conflicts_and_caps_auxiliary_mass():
    components = {
        "B_sup": _gradient(-1.0, 1.0),
        "B_prime_unsup": _gradient(0.0, 2.0),
        "B_unsup": _gradient(1.0, 0.0),
    }

    direction, metadata = combine_component_gradients(
        components, "answer_primary"
    )

    assert gradient_dot(direction, components["B_unsup"]) > 0
    assert metadata["projection_coefficients"]["B_sup"] == pytest.approx(-1.0)
    auxiliary_part = direction[0] - components["B_unsup"][0]
    assert float(torch.linalg.vector_norm(auxiliary_part)) <= 1.0 + 1e-6


def test_answer_primary_skips_when_answer_only_gradient_is_unavailable():
    direction, metadata = combine_component_gradients(
        {"B_sup": _gradient(1.0, 0.0)},
        "answer_primary",
    )

    assert direction == [None]
    assert metadata["direction_norm"] == 0.0
    assert metadata["skipped_reason"] == "answer-primary gradient unavailable"


def test_fixed_surrogate_acceptance_rules_are_distinct():
    before = {"B_sup": -3.0, "B_prime_unsup": -2.0, "B_unsup": -1.0}
    after = {"B_sup": -3.2, "B_prime_unsup": -1.7, "B_unsup": -0.9}

    total, _ = fixed_surrogate_acceptance(
        before,
        after,
        mode="total",
        active_components=("B_sup", "B_prime_unsup", "B_unsup"),
        tolerance=0.0,
    )
    componentwise, _ = fixed_surrogate_acceptance(
        before,
        after,
        mode="componentwise",
        active_components=("B_sup", "B_prime_unsup", "B_unsup"),
        tolerance=0.0,
    )
    answer_primary, _ = fixed_surrogate_acceptance(
        before,
        after,
        mode="answer_primary",
        active_components=("B_sup", "B_prime_unsup", "B_unsup"),
        tolerance=0.0,
    )

    assert total
    assert not componentwise
    assert answer_primary


def test_fixed_surrogate_rejects_nonfinite_candidate():
    before = {"B_sup": -3.0, "B_prime_unsup": -2.0, "B_unsup": -1.0}
    after = dict(before, B_unsup=math.nan)

    accepted, diagnostics = fixed_surrogate_acceptance(
        before,
        after,
        mode="total",
        active_components=("B_unsup",),
        tolerance=1e-6,
    )

    assert not accepted
    assert not diagnostics["checks"]["finite"]


def test_safeguard_restores_adam_and_accepts_a_backtracked_step(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

    def concave_objective(current_model, *_args, **_kwargs):
        weight = current_model.weight.reshape(())
        return -(weight - 0.1).square()

    monkeypatch.setattr(ac_alg1, "_B_sup", concave_objective)
    monkeypatch.setattr(
        ac_alg1,
        "_refresh_minibatch_weights",
        lambda _model, _tok, _buffers, _labelled_pids, _answer_only_pids,
        **_kwargs: ({}, {}),
    )

    _labelled, _answer_only, stats = ac_alg1._inner_weighted_em_steps(
        model,
        tok=object(),
        opt=optimizer,
        task=object(),
        buffers={},
        labelled_pids=[0],
        answer_only_pids=[],
        labelled_weights={},
        answer_only_weights={},
        inner_steps=1,
        labelled_em_weight=0.0,
        answer_only_em_weight=0.0,
        supervised_weight=1.0,
        update_geometry="sum",
        step_acceptance="total",
        rollback_tolerance=0.0,
        rollback_max_backtracks=3,
        rollback_shrink=0.5,
    )

    assert float(model.weight.detach()) == pytest.approx(0.125, abs=1e-5)
    assert stats["steps"] == 1
    assert stats["candidate_steps"] == 4
    assert stats["rolled_back_candidates"] == 3
    assert stats["accepted_step_scale"] == pytest.approx(0.125)
    assert stats["accepted_surrogate_total_delta"] > 0


def test_safeguard_restores_parameters_when_every_candidate_is_harmful(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

    def concave_objective(current_model, *_args, **_kwargs):
        weight = current_model.weight.reshape(())
        return -(weight - 0.01).square()

    monkeypatch.setattr(ac_alg1, "_B_sup", concave_objective)
    monkeypatch.setattr(
        ac_alg1,
        "_refresh_minibatch_weights",
        lambda _model, _tok, _buffers, _labelled_pids, _answer_only_pids,
        **_kwargs: ({}, {}),
    )

    _labelled, _answer_only, stats = ac_alg1._inner_weighted_em_steps(
        model,
        tok=object(),
        opt=optimizer,
        task=object(),
        buffers={},
        labelled_pids=[0],
        answer_only_pids=[],
        labelled_weights={},
        answer_only_weights={},
        inner_steps=1,
        labelled_em_weight=0.0,
        answer_only_em_weight=0.0,
        supervised_weight=1.0,
        update_geometry="sum",
        step_acceptance="total",
        rollback_tolerance=0.0,
        rollback_max_backtracks=2,
        rollback_shrink=0.5,
    )

    assert float(model.weight.detach()) == pytest.approx(0.0, abs=1e-8)
    assert stats["steps"] == 0
    assert stats["candidate_steps"] == 3
    assert stats["rolled_back_candidates"] == 3
    assert optimizer.state_dict()["state"] == {}
