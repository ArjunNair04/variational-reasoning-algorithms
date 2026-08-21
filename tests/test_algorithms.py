import numpy as np
import pytest

from variational_reasoning.em import (
    UniqueFIFOSupport,
    pis_weights,
    q5_weights,
    uniform_weights,
    weighted_joint_loss,
)
from variational_reasoning.policy_gradient import (
    grpo_advantages,
    grpo_loss,
    rloo_advantages,
    rloo_loss,
)
from variational_reasoning.settings import (
    DEVELOPMENT_SETTINGS,
    SELECTED_SETTINGS,
)
from variational_reasoning.trice import (
    Chain,
    Proposal,
    control_variate_terms,
    trice_step,
)


def test_pis_and_q5_normalise_by_question():
    question = np.array([0, 0, 1, 1])
    answer = np.array([-3.0, -1.0, -2.0, -2.0])
    trace = np.array([-0.1, -3.0, -4.0, -1.0])

    pis = pis_weights(answer, question)
    q5 = q5_weights(trace, answer, question)

    assert pis[:2].sum() == pytest.approx(1.0)
    assert pis[2:].sum() == pytest.approx(1.0)
    assert q5[:2].sum() == pytest.approx(1.0)
    assert q5[2:].sum() == pytest.approx(1.0)
    assert pis[1] > pis[0]
    assert q5[0] > q5[1]


def test_inactive_rows_receive_no_mass():
    weights = pis_weights(
        [-1.0, -2.0, -3.0, -4.0],
        [0, 0, 1, 1],
        active=[True, False, False, False],
    )
    np.testing.assert_allclose(weights, [1.0, 0.0, 0.0, 0.0])


def test_uniform_weights_and_joint_loss():
    weights = uniform_weights([0, 0, 1, 1])
    np.testing.assert_allclose(weights, [0.5, 0.5, 0.5, 0.5])
    loss = weighted_joint_loss(
        trace_logp=[-1.0, -3.0, -2.0, -4.0],
        answer_logp=[-1.0, -1.0, -2.0, -2.0],
        weights=weights,
        question_ids=[0, 0, 1, 1],
    )
    assert loss == pytest.approx(4.0)


def test_unique_fifo_support_deduplicates_and_evicts_oldest():
    support = UniqueFIFOSupport(capacity=2)
    assert support.add([1, 2])
    assert not support.add([1, 2])
    assert support.add([3])
    assert support.add([4])
    assert support.items == [(3,), (4,)]


def test_grpo_group_standardisation_and_dead_group():
    advantage = grpo_advantages(
        rewards=[0.0, 1.0, 1.0, 1.0],
        question_ids=[0, 0, 1, 1],
    )
    np.testing.assert_allclose(advantage[:2], [-1.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(advantage[2:], [0.0, 0.0])


def test_grpo_clipping_uses_active_tokens_only():
    loss = grpo_loss(
        token_logp=[[np.log(2.0), 99.0], [np.log(0.5), 99.0]],
        old_token_logp=[[0.0, 0.0], [0.0, 0.0]],
        reference_token_logp=[[np.log(2.0), 0.0], [np.log(0.5), 0.0]],
        advantages=[1.0, -1.0],
        token_mask=[[True, False], [True, False]],
        clip=0.2,
        kl_coef=0.0,
    )
    assert loss == pytest.approx(-0.2)


def test_rloo_leave_one_out_and_loss():
    advantage = rloo_advantages(
        rewards=[1.0, 0.0, 0.5, 0.5],
        question_ids=[0, 0, 1, 1],
    )
    np.testing.assert_allclose(advantage, [1.0, -1.0, 0.0, 0.0])
    assert rloo_loss([-2.0, -1.0, -4.0, -4.0], advantage) == pytest.approx(0.25)


def test_rloo_kl_shaping_precedes_leave_one_out():
    advantage = rloo_advantages(
        rewards=[1.0, 0.0],
        question_ids=[0, 0],
        policy_logp=[-1.0, -2.0],
        reference_logp=[-2.0, -2.0],
        kl_coef=0.1,
    )
    np.testing.assert_allclose(advantage, [0.9, -0.9])


def test_rloo_singleton_matches_the_executed_fallback():
    advantage = rloo_advantages(rewards=[0.75], question_ids=[0])
    np.testing.assert_allclose(advantage, [0.75])


def test_trice_accepts_correct_and_rejects_incorrect_proposals():
    chain = Chain(0, trace="old", trace_id="old", correct=True)
    retained, rejected = trice_step(
        chain, Proposal(0, trace="bad", trace_id="bad", correct=False)
    )
    assert retained == chain
    assert not rejected.accepted

    updated, accepted = trice_step(
        chain, Proposal(0, trace="new", trace_id="new", correct=True)
    )
    assert updated.trace == "new"
    assert accepted.accepted


def test_trice_control_variate_coefficients():
    chains = [
        Chain(0, "a", "a", True),
        Chain(1, "b", "b", True),
        Chain(2, "c", "c", True),
    ]
    proposals = [
        Proposal(0, "a2", "a2", True),
        Proposal(1, "bad", "bad", False),
        Proposal(2, "c2", "c2", True),
    ]
    transitions = [
        trice_step(chain, proposal)[1]
        for chain, proposal in zip(chains, proposals)
    ]
    terms, beta = control_variate_terms(transitions)

    assert beta[0] == pytest.approx(0.5)
    assert beta[1] == pytest.approx(1.0)
    assert beta[2] == pytest.approx(0.5)
    state_terms = [term for term in terms if term.role == "accepted_state"]
    assert len(state_terms) == 3
    assert all(term.coefficient == pytest.approx(1 / 3) for term in state_terms)
    rejected = [term for term in terms if term.role == "rejected_proposal_control"]
    assert len(rejected) == 1
    assert rejected[0].coefficient == pytest.approx(-1 / 3)


def test_selected_settings_are_the_executed_coordinates():
    assert SELECTED_SETTINGS["Q5"]["support_size"] == 16
    assert SELECTED_SETTINGS["PIS"]["updates_per_round"] == 4
    assert SELECTED_SETTINGS["GRPO"]["optimizer_step_scope"] == "batch"
    assert SELECTED_SETTINGS["RLOO"]["kl_coef"] == pytest.approx(0.03)
    assert DEVELOPMENT_SETTINGS["Q5-MORE"]["proposals_per_question"] == 32
    assert DEVELOPMENT_SETTINGS["Q5-MORE"]["support_size"] == 16
