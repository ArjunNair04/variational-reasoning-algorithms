"""CPU-only tests for constant-budget multi-particle Q5."""

from types import SimpleNamespace

import pytest
import torch

import ac_alg1_q5_multichain as q5mp
from ac_alg1_q5_multichain import (
    build_q5_particle_posterior_plan,
    initialize_particle_memory,
)
from ac_alg1_trice import (
    TriceProposal,
    TriceTrace,
    build_estimator_plan,
    initialize_chain,
    independence_step,
)
from run_sweep_lm import _run_q5_multichain_sweep


def _trace(question_id: int, token: int) -> TriceTrace:
    return TriceTrace(
        ids=torch.tensor([token, 99]),
        span=torch.tensor([True, True]),
        text=f"trace-{question_id}-{token}",
        question_id=question_id,
        source="test",
    )


def _valid_particle(question_id: int, particle: int, token: int):
    trace = _trace(question_id, token)
    return initialize_chain(
        (question_id, particle),
        trace,
        state_id=f"state:{question_id}:{particle}:{token}",
        state_correct=True,
    )


def test_k1_posterior_is_exactly_the_basic_single_chain_estimator() -> None:
    memory = {
        (0, 0): _valid_particle(0, 0, 10),
        (1, 0): _valid_particle(1, 0, 20),
    }
    q5_plan, diagnostics = build_q5_particle_posterior_plan(
        memory,
        [0, 1],
        chains_per_question=1,
        log_scores={(0, 0): -100.0, (1, 0): 7.0},
    )

    transitions = []
    for question_id in (0, 1):
        trace = memory[(question_id, 0)].state
        assert trace is not None
        original = initialize_chain(
            question_id,
            trace,
            state_id=f"single:{question_id}",
            state_correct=True,
        )
        _after, transition = independence_step(
            original,
            TriceProposal(
                question_id=question_id,
                payload=trace,
                state_id=f"proposal:{question_id}",
                correct=False,
            ),
        )
        transitions.append(transition)
    basic_plan = build_estimator_plan(transitions, estimator="basic")

    assert [term.coefficient for term in q5_plan.terms] == pytest.approx(
        [term.coefficient for term in basic_plan.terms]
    )
    assert diagnostics.valid_questions == 2
    assert diagnostics.valid_particles == 2
    assert diagnostics.mean_effective_sample_size == pytest.approx(1.0)


def test_soft_posterior_normalizes_within_question_then_across_questions() -> None:
    memory = {
        (0, 0): _valid_particle(0, 0, 10),
        (0, 1): _valid_particle(0, 1, 11),
        (1, 0): _valid_particle(1, 0, 20),
        (1, 1): _valid_particle(1, 1, 21),
    }
    plan, diagnostics = build_q5_particle_posterior_plan(
        memory,
        [0, 1],
        chains_per_question=2,
        log_scores={(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 2.0},
    )
    coefficient_by_key = {
        term.question_id: term.coefficient for term in plan.terms
    }

    assert coefficient_by_key[(0, 0)] == pytest.approx(0.25)
    assert coefficient_by_key[(0, 1)] == pytest.approx(0.25)
    assert coefficient_by_key[(1, 0)] + coefficient_by_key[(1, 1)] == pytest.approx(0.5)
    assert coefficient_by_key[(1, 1)] > coefficient_by_key[(1, 0)]
    assert sum(coefficient_by_key.values()) == pytest.approx(1.0)
    assert diagnostics.valid_questions == 2
    assert diagnostics.valid_particles == 4


def test_invalid_particles_are_excluded_without_dropping_valid_question() -> None:
    invalid_trace = _trace(0, 12)
    memory = {
        (0, 0): _valid_particle(0, 0, 10),
        (0, 1): initialize_chain(
            (0, 1),
            invalid_trace,
            state_id="invalid",
            state_correct=False,
        ),
        (1, 0): initialize_chain((1, 0)),
        (1, 1): initialize_chain((1, 1)),
    }
    plan, diagnostics = build_q5_particle_posterior_plan(
        memory,
        [0, 1],
        chains_per_question=2,
        log_scores={(0, 0): 4.0},
    )

    assert len(plan.terms) == 1
    assert plan.terms[0].question_id == (0, 0)
    assert plan.terms[0].coefficient == pytest.approx(1.0)
    assert diagnostics.valid_questions == 1
    assert diagnostics.valid_particles == 1


def test_initialization_draws_one_independent_state_per_particle(monkeypatch) -> None:
    calls = []

    def sample(
        _model,
        _tok,
        _task,
        question_ids,
        *,
        chains_per_question,
        prompt_mode,
        source,
        reward_requires_eos,
    ):
        calls.append(
            (
                tuple(question_ids),
                chains_per_question,
                prompt_mode,
                source,
                reward_requires_eos,
            )
        )
        return tuple(
            TriceProposal(
                question_id=(question_id, particle),
                payload=_trace(question_id, 10 + particle),
                state_id=f"{question_id}:{particle}",
                correct=True,
            )
            for particle in range(chains_per_question)
            for question_id in question_ids
        )

    monkeypatch.setattr(q5mp, "_sample_particle_proposals", sample)
    memory = initialize_particle_memory(
        object(),
        object(),
        object(),
        [3, 5],
        chains_per_question=4,
        initializer_prompt="answer_derive",
        reward_requires_eos=True,
    )

    assert len(memory) == 8
    assert set(memory) == {(question_id, particle) for question_id in (3, 5) for particle in range(4)}
    assert calls == [((3, 5), 4, "answer_derive", "answer_guided_initializer", True)]


@pytest.mark.parametrize("batch,particles", [(64, 1), (32, 2), (16, 4)])
def test_sweep_wrapper_preserves_sixty_four_current_proposals(
    monkeypatch,
    batch,
    particles,
) -> None:
    observed = {}

    def run(_task, **kwargs):
        observed.update(kwargs)
        return [{"acceptance_fraction": 0.5, "llm_gen": 1}]

    monkeypatch.setattr("run_sweep_lm.run_q5_multichain", run)
    records = _run_q5_multichain_sweep(
        object(),
        B=batch,
        G=particles,
        proposal_prompt="question",
        reward_requires_eos=True,
        answer_target_termination="eos",
    )

    assert observed["questions_per_round"] == batch
    assert observed["chains_per_question"] == particles
    assert observed["questions_per_round"] * observed["chains_per_question"] == 64
    assert records[0]["mean_reward"] == 0.5


def test_sweep_wrapper_rejects_nonconstant_budget() -> None:
    with pytest.raises(ValueError, match="exactly 64"):
        _run_q5_multichain_sweep(object(), B=64, G=2)


def test_runner_emits_complete_constant_budget_diagnostics(monkeypatch) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    tokenizer = SimpleNamespace(eos_token_id=99)
    pool = list(range(64))
    diagnostics = []

    monkeypatch.setattr(
        q5mp,
        "_labelled_answer_only_pools",
        lambda _task, labelled_frac: ([], pool),
    )
    monkeypatch.setattr(
        q5mp,
        "initialize_particle_memory",
        lambda _model, _tok, _task, question_ids, **_kwargs: {
            (question_id, 0): _valid_particle(question_id, 0, 10 + question_id)
            for question_id in question_ids
        },
    )

    def proposals(_model, _tok, _task, question_ids, **_kwargs):
        return tuple(
            TriceProposal(
                question_id=(question_id, 0),
                payload=_trace(question_id, 20 + question_id),
                state_id=f"new:{question_id}",
                correct=True,
            )
            for question_id in question_ids
        )

    monkeypatch.setattr(q5mp, "_sample_particle_proposals", proposals)

    def score(current_model, _tok, traces, *, grad=True):
        values = current_model.weight.sum().expand(len(traces))
        return values if grad else values.detach()

    monkeypatch.setattr(q5mp, "_score_trice_traces", score)
    monkeypatch.setattr(q5mp, "maybe_eval", lambda *_args, **_kwargs: 0.5)

    records = q5mp.run_q5_multichain(
        object(),
        rounds=1,
        questions_per_round=64,
        chains_per_question=1,
        seed=7,
        lr=0.01,
        model_tok=(model, tokenizer),
        diagnostics_fn=diagnostics.append,
        log=lambda _record: None,
    )

    assert records[0]["particles_this_round"] == 64
    assert records[0]["guide_gen"] == 64
    assert records[0]["gen"] == 64
    assert records[0]["gsteps"] == 1
    assert diagnostics[0]["design"]["constant_current_proposals_per_round"] == 64
    assert diagnostics[0]["posterior"]["mean_effective_sample_size"] == 1.0
    assert diagnostics[0]["reward"]["requires_natural_eos"] is True
