"""Regression tests for pure AC-ALG1 diagnostic reductions."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

from ac_alg1 import _run_diagnostic_probe
from ac_alg1_diagnostics import (
    binary_score_calibration,
    optimizer_moment_diagnostics,
    posterior_churn,
    responsibility_gini,
    responsibility_margin,
    spearman_correlation,
    tensor_list_cosine,
)


def test_responsibility_concentration_metrics():
    assert responsibility_gini([0.25] * 4) == pytest.approx(0.0)
    assert responsibility_gini([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.75)
    assert responsibility_margin([0.6, 0.3, 0.1]) == pytest.approx(0.3)
    assert responsibility_gini([]) is None


def test_posterior_churn_records_rank_and_top_trace_changes():
    before = {7: torch.tensor([0.8, 0.2])}
    after = {7: torch.tensor([0.1, 0.9])}
    result = posterior_churn(
        before,
        after,
        trace_ids={7: ["old-top", "new-top"]},
    )

    question = result["questions"][0]
    assert question["total_variation"] == pytest.approx(0.7)
    assert question["rank_correlation"] == pytest.approx(-1.0)
    assert question["top_trace_replaced"] is True
    assert question["top_trace_before"] == "old-top"
    assert question["top_trace_after"] == "new-top"
    assert result["summary"]["top_trace_replacement_fraction"] == pytest.approx(
        1.0
    )


def test_reader_calibration_metrics_reward_correct_ranking():
    result = binary_score_calibration(
        [math.log(0.9), math.log(0.8), math.log(0.2), math.log(0.1)],
        [True, True, False, False],
        bins=2,
    )

    assert result["count"] == 4
    assert result["positive_count"] == 2
    assert result["auroc"] == pytest.approx(1.0)
    assert result["auprc"] == pytest.approx(1.0)
    assert result["brier"] == pytest.approx(0.025)
    assert result["ece"] == pytest.approx(0.15)


def test_tensor_cosine_and_rank_correlation_handle_degenerate_inputs():
    assert tensor_list_cosine(
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([0.0, 1.0])],
    ) == pytest.approx(0.0)
    assert tensor_list_cosine(
        [torch.zeros(2)],
        [torch.ones(2)],
    ) is None
    assert spearman_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == (
        pytest.approx(-1.0)
    )
    assert spearman_correlation([1.0, 1.0], [0.0, 1.0]) is None


def test_optimizer_moment_diagnostics_after_adam_step():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()
    optimizer.step()

    result = optimizer_moment_diagnostics(optimizer)
    assert result["state_parameter_tensors"] == 2
    assert result["first_moment_l2_norm"] > 0
    assert result["second_moment_l2_norm"] > 0
    assert result["minimum_step"] == pytest.approx(1.0)
    assert result["maximum_step"] == pytest.approx(1.0)
    assert result["learning_rate_min"] == pytest.approx(1e-3)
    assert result["learning_rate_max"] == pytest.approx(1e-3)


def test_fixed_probe_restores_rng_and_training_mode():
    model = torch.nn.Linear(1, 1)
    model.train()
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    expected_python_state = random.getstate()
    expected_numpy_state = np.random.get_state()
    expected_state = torch.random.get_rng_state()

    def probe(current_model):
        current_model.eval()
        random.random()
        np.random.random()
        torch.rand(5)
        return 0.625

    assert _run_diagnostic_probe(model, probe) == pytest.approx(0.625)
    assert model.training is True
    assert random.getstate() == expected_python_state
    actual_numpy_state = np.random.get_state()
    assert actual_numpy_state[0] == expected_numpy_state[0]
    assert np.array_equal(actual_numpy_state[1], expected_numpy_state[1])
    assert actual_numpy_state[2:] == expected_numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), expected_state)
