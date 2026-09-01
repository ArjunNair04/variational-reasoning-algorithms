"""CPU-only tests for the persistent-chain AC-ALG1/TRICE variant."""

from types import SimpleNamespace

import pytest
import torch

import ac_alg1_trice
from ac_alg1_trice import (
    _gold_trace,
    TriceProposal,
    advance_chains,
    backward_estimator_plan,
    build_estimator_plan,
    chain_memory_diagnostics,
    collect_frozen_proposals,
    evaluate_estimator_plan,
    independence_step,
    initialize_chain,
    leave_one_out_acceptance_scales,
    run_ac_alg1_trice,
)


def test_gold_trace_uses_contextual_leading_space_and_scores_terminal_eos():
    class Tokenizer:
        eos_token_id = 99

        def __init__(self):
            self.calls = []

        def __call__(
            self,
            text,
            *,
            return_tensors=None,
            add_special_tokens=True,
        ):
            self.calls.append((text, return_tensors, add_special_tokens))
            if return_tensors == "pt":
                ids = torch.tensor([[10, 11]], dtype=torch.long)
            else:
                assert text == " worked solution #### 4"
                ids = [20, 21, 22]
            return type("Tokenized", (), {"input_ids": ids})()

    tokenizer = Tokenizer()
    task = type(
        "Task",
        (),
        {
            "prompts": ["question prompt"],
            "gold_solution": ["worked solution #### 4"],
        },
    )()

    trace = _gold_trace(tokenizer, task, 0)

    assert tokenizer.calls == [
        ("question prompt", "pt", True),
        (" worked solution #### 4", None, False),
    ]
    assert trace.ids.tolist() == [10, 11, 20, 21, 22, 99]
    assert trace.span.tolist() == [False, False, True, True, True, True]
    assert trace.ids[trace.span].tolist()[-1] == tokenizer.eos_token_id
    assert trace.text == "worked solution #### 4"
    assert trace.source == "gold_initializer"


def test_gold_trace_requires_terminal_eos_token():
    class Tokenizer:
        eos_token_id = None

        def __call__(
            self,
            text,
            *,
            return_tensors=None,
            add_special_tokens=True,
        ):
            ids = (
                torch.tensor([[10]], dtype=torch.long)
                if return_tensors == "pt"
                else [20]
            )
            return type("Tokenized", (), {"input_ids": ids})()

    task = type(
        "Task",
        (),
        {"prompts": ["question"], "gold_solution": ["solution"]},
    )()

    with pytest.raises(ValueError, match="requires an EOS token"):
        _gold_trace(Tokenizer(), task, 0)


def _proposal(question_id, payload, correct):
    return TriceProposal(
        question_id=question_id,
        payload=payload,
        state_id=f"state:{payload}",
        correct=correct,
    )


def test_correct_proposal_is_accepted_and_resets_chain_age():
    chain = initialize_chain(
        "q",
        "old",
        state_id="state:old",
        state_correct=True,
    )
    chain, _ = independence_step(chain, _proposal("q", "wrong", False))
    chain, transition = independence_step(chain, _proposal("q", "new", True))

    assert transition.accepted
    assert chain.state == "new"
    assert chain.state_correct
    assert chain.age == 0
    assert chain.proposals == 2
    assert chain.acceptances == 1
    assert chain.rejections == 1
    assert chain.unique_accepted_state_ids == {
        "state:old",
        "state:new",
    }


def test_incorrect_proposal_is_rejected_even_from_invalid_state():
    chain = initialize_chain(
        "q",
        "bad initializer",
        state_id="state:init",
        state_correct=False,
    )

    after, transition = independence_step(
        chain,
        _proposal("q", "also bad", False),
    )

    assert not transition.accepted
    assert after.state == "bad initializer"
    assert not after.state_correct
    assert after.age == 1
    assert after.unique_accepted_state_ids == set()


def test_advance_chains_is_pure_and_order_independent_across_questions():
    original = {
        "a": initialize_chain(
            "a", "a0", state_id="state:a0", state_correct=True
        ),
        "b": initialize_chain(
            "b", "b0", state_id="state:b0", state_correct=True
        ),
    }

    updated, transitions = advance_chains(
        original,
        [_proposal("b", "b1", False), _proposal("a", "a1", True)],
    )

    assert original["a"].state == "a0"
    assert original["a"].proposals == 0
    assert updated["a"].state == "a1"
    assert updated["b"].state == "b0"
    assert [transition.after.question_id for transition in transitions] == [
        "b",
        "a",
    ]


def test_macrocycle_forbids_two_updates_to_the_same_chain():
    memory = {"q": initialize_chain("q")}

    with pytest.raises(ValueError, match="one proposal per question"):
        advance_chains(
            memory,
            [_proposal("q", "first", False), _proposal("q", "second", True)],
        )


def _three_valid_transitions():
    memory = {
        index: initialize_chain(
            index,
            f"old-{index}",
            state_id=f"state:old-{index}",
            state_correct=True,
        )
        for index in range(3)
    }
    _, transitions = advance_chains(
        memory,
        [
            _proposal(0, "accepted-0", True),
            _proposal(1, "rejected-1", False),
            _proposal(2, "accepted-2", True),
        ],
    )
    return transitions


def test_leave_one_out_beta_excludes_each_examples_own_proposal():
    scales = leave_one_out_acceptance_scales(_three_valid_transitions())

    assert scales == pytest.approx({0: 0.5, 1: 1.0, 2: 0.5})


def test_basic_estimator_is_mean_score_of_updated_valid_states():
    transitions = _three_valid_transitions()
    plan = build_estimator_plan(transitions, estimator="basic")
    scores = {
        "accepted-0": torch.tensor(2.0),
        "old-1": torch.tensor(3.0),
        "accepted-2": torch.tensor(5.0),
    }

    objective = evaluate_estimator_plan(
        plan,
        lambda payloads: torch.stack([scores[payload] for payload in payloads]),
    )

    assert float(objective) == pytest.approx(10.0 / 3.0)
    assert plan.valid_chains == 3
    assert plan.rejected_proposals_used == 0


def test_control_variate_uses_rejected_proposal_with_negative_coefficient():
    transitions = _three_valid_transitions()
    plan = build_estimator_plan(transitions, estimator="control_variate")
    values = {
        "accepted-0": torch.tensor(2.0),
        "old-1": torch.tensor(3.0),
        "rejected-1": torch.tensor(-7.0),
        "accepted-2": torch.tensor(5.0),
    }

    objective = evaluate_estimator_plan(
        plan,
        lambda payloads: torch.stack([values[payload] for payload in payloads]),
    )

    # (2 + 3 + 5 - .5*2 - 1*(-7) - .5*5) / 3
    assert float(objective) == pytest.approx(4.5)
    rejected_terms = [
        term for term in plan.terms if term.role == "rejected_proposal_control"
    ]
    assert len(rejected_terms) == 1
    assert rejected_terms[0].payload == "rejected-1"
    assert rejected_terms[0].coefficient == pytest.approx(-1.0 / 3.0)
    assert plan.rejected_proposals_used == 1
    assert plan.rejection_absolute_weight == pytest.approx(1.0 / 3.0)


def test_control_variate_retains_autograd_signs():
    transitions = _three_valid_transitions()
    plan = build_estimator_plan(transitions, estimator="control_variate")
    parameters = {
        payload: torch.nn.Parameter(torch.tensor(float(index + 1)))
        for index, payload in enumerate(
            ("accepted-0", "old-1", "rejected-1", "accepted-2")
        )
    }

    objective = evaluate_estimator_plan(
        plan,
        lambda payloads: torch.stack([parameters[payload] for payload in payloads]),
    )
    objective.backward()

    assert float(parameters["old-1"].grad) == pytest.approx(1.0 / 3.0)
    assert float(parameters["rejected-1"].grad) == pytest.approx(-1.0 / 3.0)
    # The accepted proposal is both z' and tilde-z: (1 - beta) / 3.
    assert float(parameters["accepted-0"].grad) == pytest.approx(1.0 / 6.0)
    assert float(parameters["accepted-2"].grad) == pytest.approx(1.0 / 6.0)


def test_streamed_backward_matches_full_plan_and_bounds_score_batches():
    transitions = _three_valid_transitions()
    plan = build_estimator_plan(transitions, estimator="control_variate")
    payloads = tuple(term.payload for term in plan.terms)
    full_parameters = {
        payload: torch.nn.Parameter(torch.tensor(float(index + 1)))
        for index, payload in enumerate(payloads)
    }
    streamed_parameters = {
        payload: torch.nn.Parameter(parameter.detach().clone())
        for payload, parameter in full_parameters.items()
    }

    full_objective = 0.25 * evaluate_estimator_plan(
        plan,
        lambda batch: torch.stack([full_parameters[payload] for payload in batch]),
    )
    (-full_objective).backward()

    batch_sizes = []

    def score_streamed(batch):
        batch_sizes.append(len(batch))
        return torch.stack([streamed_parameters[payload] for payload in batch])

    streamed_objective, did_backward = backward_estimator_plan(
        plan,
        score_streamed,
        weight=0.25,
        chunk_size=2,
    )

    assert did_backward
    assert float(streamed_objective) == pytest.approx(float(full_objective.detach()))
    assert max(batch_sizes) == 2
    assert len(batch_sizes) > 1
    for payload in payloads:
        assert float(streamed_parameters[payload].grad) == pytest.approx(
            float(full_parameters[payload].grad)
        )


def test_streamed_backward_rejects_nonpositive_chunk_size():
    plan = build_estimator_plan(_three_valid_transitions(), estimator="basic")

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        backward_estimator_plan(
            plan,
            lambda batch: torch.ones(len(batch)),
            chunk_size=0,
        )


def test_invalid_updated_chain_is_omitted_from_both_estimators():
    memory = {
        "valid": initialize_chain(
            "valid", "kept", state_id="state:kept", state_correct=True
        ),
        "invalid": initialize_chain("invalid"),
    }
    _, transitions = advance_chains(
        memory,
        [
            _proposal("valid", "bad", False),
            _proposal("invalid", "also bad", False),
        ],
    )

    plan = build_estimator_plan(transitions, estimator="control_variate")

    assert plan.valid_chains == 1
    assert {term.question_id for term in plan.terms} == {"valid"}
    # No other valid example exists, so the leave-one-out beta is safely zero.
    assert plan.beta_by_question == pytest.approx(
        {"valid": 0.0, "invalid": 0.0}
    )


def test_parameter_version_contract_detects_sampler_mutation():
    model = torch.nn.Linear(1, 1, bias=False)

    def invalid_sampler():
        with torch.no_grad():
            model.weight.add_(1.0)
        return [_proposal("q", "sample", True)]

    with pytest.raises(RuntimeError, match="changed inside"):
        collect_frozen_proposals(model, invalid_sampler)


def test_parameter_version_contract_allows_read_only_sampling_then_detects_update():
    model = torch.nn.Linear(1, 1, bias=False)
    batch = collect_frozen_proposals(
        model,
        lambda: [_proposal("q", "sample", True)],
    )
    batch.assert_model_unchanged(model)

    with torch.no_grad():
        model.weight.mul_(2.0)

    with pytest.raises(RuntimeError, match="changed inside"):
        batch.assert_model_unchanged(model)


def test_chain_diagnostics_report_acceptance_age_and_unique_states():
    memory = {
        "a": initialize_chain(
            "a", "a0", state_id="state:a0", state_correct=True
        ),
        "b": initialize_chain("b"),
    }
    memory, _ = advance_chains(
        memory,
        [_proposal("a", "a1", False), _proposal("b", "b1", True)],
    )

    diagnostics = chain_memory_diagnostics(memory)

    assert diagnostics["chains"] == 2
    assert diagnostics["valid_chains"] == 2
    assert diagnostics["acceptance_fraction"] == pytest.approx(0.5)
    assert diagnostics["mean_chain_age"] == pytest.approx(0.5)
    assert diagnostics["max_chain_age"] == 1
    assert diagnostics["unique_accepted_states"] == 2


def test_single_valid_example_disables_control_variate_safely():
    memory = {
        "q": initialize_chain(
            "q", "old", state_id="state:old", state_correct=True
        )
    }
    memory, transitions = advance_chains(
        memory,
        [_proposal("q", "bad", False)],
    )

    plan = build_estimator_plan(transitions, estimator="control_variate")

    assert plan.beta_by_question == {"q": 0.0}
    assert len(plan.terms) == 1
    assert plan.terms[0].role == "accepted_state"


def test_runner_initializes_every_chain_before_the_first_optimizer_step(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    tokenizer = SimpleNamespace(eos_token_id=99)
    initial_weight = model.weight.detach().clone()
    initialization_calls = []

    monkeypatch.setattr(
        ac_alg1_trice,
        "_labelled_answer_only_pools",
        lambda task, labelled_frac: ([], [0, 1]),
    )

    def initialize_all(
        current_model,
        _tok,
        _task,
        memory,
        labelled_ids,
        answer_only_ids,
        **_kwargs,
    ):
        initialization_calls.append(
            (
                tuple(labelled_ids),
                tuple(answer_only_ids),
                current_model.weight.detach().clone(),
            )
        )
        return {
            **memory,
            **{
                question_id: initialize_chain(
                    question_id,
                    SimpleNamespace(
                        ids=torch.tensor([99]),
                        span=torch.tensor([True]),
                    ),
                    state_id=f"guide:{question_id}",
                    state_correct=True,
                )
                for question_id in answer_only_ids
            },
        }

    monkeypatch.setattr(
        ac_alg1_trice,
        "_initialize_selected_chains",
        initialize_all,
    )
    monkeypatch.setattr(
        ac_alg1_trice,
        "_sample_trace_proposals",
        lambda _model, _tok, _task, question_ids, **_kwargs: tuple(
            _proposal(
                question_id,
                SimpleNamespace(
                    ids=torch.tensor([99]),
                    span=torch.tensor([True]),
                ),
                True,
            )
            for question_id in question_ids
        ),
    )
    monkeypatch.setattr(
        ac_alg1_trice,
        "_score_trice_traces",
        lambda current_model, _tok, traces: current_model.weight.sum().expand(
            len(traces)
        ),
    )
    monkeypatch.setattr(ac_alg1_trice, "maybe_eval", lambda *args, **kwargs: float("nan"))

    records = run_ac_alg1_trice(
        object(),
        rounds=2,
        L_batch=0,
        U_batch=1,
        seed=3,
        lr=0.1,
        model_tok=(model, tokenizer),
        labelled_frac=0.0,
        supervised_weight=0.0,
        labelled_trice_weight=0.0,
        answer_only_trice_weight=1.0,
        question_sampling="epoch_shuffle",
        log=lambda *_args: None,
    )

    assert len(initialization_calls) == 1
    assert initialization_calls[0][:2] == ((), (0, 1))
    assert torch.equal(initialization_calls[0][2], initial_weight)
    assert not torch.equal(model.weight.detach(), initial_weight)
    assert records[-1]["guide_gen"] == 2
    assert records[-1]["unique_questions_seen"] == 2
    assert records[-1]["backward_chunk_size"] == 4
