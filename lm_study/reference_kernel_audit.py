#!/usr/bin/env python3
"""Verify frozen training weights against the compact reference package."""

from __future__ import annotations

import numpy as np
import torch

from ac_alg1 import _barber_variational_logits, _posterior_weights
from l2r import l2r_responsibilities
from variational_reasoning import importance_weights, pis_weights, q5_weights


TRACE = np.array([-0.2, -2.3, -1.4, -0.7, -3.1], dtype=np.float64)
ANSWER = np.array([-2.0, -0.4, -1.1, -2.2, -0.3], dtype=np.float64)
PROPOSAL = np.array([-1.1, -2.8, -0.9, -1.7, -2.4], dtype=np.float64)
QUESTION = np.array([0, 0, 1, 1, 1], dtype=np.int64)


def _tensor(values: np.ndarray) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _ac_weights(estimator: str) -> np.ndarray:
    trace = _tensor(TRACE)
    answer = _tensor(ANSWER)
    proposal = _tensor(PROPOSAL) if estimator != "delta_joint" else None
    logits = _barber_variational_logits(
        estimator,
        trace,
        answer,
        proposal_trace_logprobs=proposal,
    )
    weights = torch.zeros_like(logits)
    for question_id in np.unique(QUESTION):
        local = torch.tensor(QUESTION == question_id)
        weights[local], _ = _posterior_weights(
            logits[local],
            "softmax_entropy",
            temperature=1.0,
            ess_floor_fraction=0.0,
        )
    return weights.numpy()


def run_audit() -> None:
    ones = torch.ones(len(TRACE), dtype=torch.long)
    pids = torch.tensor(QUESTION)

    l2r_joint = l2r_responsibilities(
        _tensor(TRACE), _tensor(ANSWER), ones, ones, pids, score="joint"
    ).numpy()
    l2r_pis = l2r_responsibilities(
        _tensor(TRACE),
        _tensor(ANSWER),
        ones,
        ones,
        pids,
        score="prior_corrected",
    ).numpy()

    expected_q5 = q5_weights(TRACE, ANSWER, QUESTION)
    expected_pis = pis_weights(ANSWER, QUESTION)
    expected_off_policy = importance_weights(TRACE, ANSWER, PROPOSAL, QUESTION)

    np.testing.assert_allclose(l2r_joint, expected_q5, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(l2r_pis, expected_pis, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        _ac_weights("delta_joint"), expected_q5, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        _ac_weights("prior_importance"), expected_pis, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        _ac_weights("answer_conditioned_importance"),
        expected_off_policy,
        rtol=0.0,
        atol=1e-12,
    )


if __name__ == "__main__":
    run_audit()
    print("reference_kernel_audit=pass")
