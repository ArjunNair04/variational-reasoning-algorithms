"""Constant-budget multi-particle Q5 training.

This module extends the persistent strict-correct state used by TRICE from one
state per question to ``K`` independently evolving particles.  Every selected
question receives one current-policy proposal per particle.  Correct proposals
replace only their own particle; incorrect proposals leave that particle
unchanged.  The M-step then uses a detached soft posterior over the valid
particles belonging to each question.

The implementation is deliberately separate from ``run_ac_alg1_trice`` so the
paper-faithful TRICE path and all historical Q5 paths remain behaviour-locked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from ac_alg1 import _labelled_answer_only_pools
from ac_alg1_trice import (
    TRICE_BACKWARD_CHUNK_SIZE,
    TorchParameterVersion,
    TriceChainState,
    TriceEstimatorPlan,
    TriceProposal,
    TriceScoreTerm,
    TriceTrace,
    _sample_trace_proposals,
    _score_trice_traces,
    _trace_emitted_eos,
    _trace_state_id,
    advance_chains,
    backward_estimator_plan,
    chain_memory_diagnostics,
    collect_frozen_proposals,
    initialize_chain,
)
from common import MODEL_NAME, QuestionSampler, load_model, maybe_eval


ParticleKey = tuple[int, int]
Q5_MULTICHAIN_COUNTS = (1, 2, 4)


@dataclass(frozen=True)
class Q5ParticlePosteriorDiagnostics:
    """Detached concentration summary for one multi-particle E-step."""

    selected_questions: int
    valid_questions: int
    valid_particles: int
    mean_valid_particles: float | None
    mean_effective_sample_size: float | None
    mean_effective_sample_size_fraction: float | None
    mean_max_responsibility: float | None


def _particle_key(question_id: int, particle_index: int) -> ParticleKey:
    return int(question_id), int(particle_index)


def _retarget_proposal(
    proposal: TriceProposal[TriceTrace],
    particle_index: int,
) -> TriceProposal[TriceTrace]:
    return TriceProposal(
        question_id=_particle_key(int(proposal.question_id), particle_index),
        payload=proposal.payload,
        state_id=proposal.state_id,
        correct=proposal.correct,
    )


def _sample_particle_proposals(
    model,
    tok,
    task,
    question_ids: Sequence[int],
    *,
    chains_per_question: int,
    prompt_mode: str,
    source: str,
    reward_requires_eos: bool,
) -> tuple[TriceProposal[TriceTrace], ...]:
    proposals: list[TriceProposal[TriceTrace]] = []
    for particle_index in range(chains_per_question):
        sampled = _sample_trace_proposals(
            model,
            tok,
            task,
            question_ids,
            prompt_mode=prompt_mode,
            source=f"{source}_p{particle_index}",
            reward_requires_eos=reward_requires_eos,
        )
        proposals.extend(
            _retarget_proposal(proposal, particle_index)
            for proposal in sampled
        )
    return tuple(proposals)


def initialize_particle_memory(
    model,
    tok,
    task,
    question_ids: Sequence[int],
    *,
    chains_per_question: int,
    initializer_prompt: str,
    reward_requires_eos: bool,
) -> dict[ParticleKey, TriceChainState[TriceTrace]]:
    """Initialize every particle before the first optimizer update."""

    proposals = _sample_particle_proposals(
        model,
        tok,
        task,
        question_ids,
        chains_per_question=chains_per_question,
        prompt_mode=initializer_prompt,
        source="answer_guided_initializer",
        reward_requires_eos=reward_requires_eos,
    )
    expected = len(question_ids) * chains_per_question
    if len(proposals) != expected:
        raise AssertionError(
            f"expected {expected} particle initializers, found {len(proposals)}"
        )
    return {
        proposal.question_id: initialize_chain(
            proposal.question_id,
            proposal.payload,
            state_id=proposal.state_id,
            state_correct=bool(proposal.correct),
        )
        for proposal in proposals
    }


def build_q5_particle_posterior_plan(
    memory: Mapping[ParticleKey, TriceChainState[TriceTrace]],
    question_ids: Sequence[int],
    *,
    chains_per_question: int,
    log_scores: Mapping[ParticleKey, float],
) -> tuple[TriceEstimatorPlan[TriceTrace], Q5ParticlePosteriorDiagnostics]:
    """Form a detached per-question soft posterior over valid particles.

    For every question with at least one strict-correct particle,

    ``w_qk = softmax_k(log p_theta(h_qk, a* | q))``.

    The resulting objective averages questions first and particles second, so
    questions with more valid particles do not receive more total weight.
    """

    selected = [int(question_id) for question_id in question_ids]
    if len(set(selected)) != len(selected):
        raise ValueError("Q5 multi-particle questions must be distinct")
    if chains_per_question <= 0:
        raise ValueError("chains_per_question must be positive")

    per_question: list[
        tuple[int, list[tuple[ParticleKey, TriceChainState[TriceTrace], float]]]
    ] = []
    for question_id in selected:
        valid: list[tuple[ParticleKey, TriceChainState[TriceTrace], float]] = []
        for particle_index in range(chains_per_question):
            key = _particle_key(question_id, particle_index)
            if key not in memory:
                raise KeyError(f"missing Q5 particle {key!r}")
            state = memory[key]
            if not state.state_correct:
                continue
            if state.state is None or state.state_id is None:
                raise AssertionError("valid Q5 particle is missing its state")
            if key not in log_scores:
                raise KeyError(f"missing Q5 posterior score for {key!r}")
            score = float(log_scores[key])
            if not math.isfinite(score):
                raise ValueError(f"non-finite Q5 posterior score for {key!r}")
            valid.append((key, state, score))
        if valid:
            per_question.append((question_id, valid))

    valid_question_count = len(per_question)
    terms: list[TriceScoreTerm[TriceTrace]] = []
    ess_values: list[float] = []
    ess_fractions: list[float] = []
    max_weights: list[float] = []
    valid_particles = 0
    if valid_question_count:
        question_normalizer = 1.0 / valid_question_count
        for _question_id, valid in per_question:
            scores = np.asarray([row[2] for row in valid], dtype=np.float64)
            unnormalized = np.exp(scores - float(np.max(scores)))
            weights = unnormalized / float(unnormalized.sum())
            ess = float(1.0 / np.square(weights).sum())
            ess_values.append(ess)
            ess_fractions.append(ess / len(valid))
            max_weights.append(float(weights.max()))
            valid_particles += len(valid)
            for (key, state, _score), weight in zip(valid, weights):
                assert state.state is not None and state.state_id is not None
                terms.append(
                    TriceScoreTerm(
                        question_id=key,
                        payload=state.state,
                        state_id=state.state_id,
                        coefficient=question_normalizer * float(weight),
                        role="posterior_particle",
                    )
                )

    plan = TriceEstimatorPlan(
        estimator="q5_particle_posterior",
        terms=tuple(terms),
        valid_chains=valid_particles,
        beta_by_question={},
        rejected_proposals_used=0,
        rejection_absolute_weight=0.0,
    )
    diagnostics = Q5ParticlePosteriorDiagnostics(
        selected_questions=len(selected),
        valid_questions=valid_question_count,
        valid_particles=valid_particles,
        mean_valid_particles=(
            float(valid_particles / valid_question_count)
            if valid_question_count
            else None
        ),
        mean_effective_sample_size=(
            float(np.mean(ess_values)) if ess_values else None
        ),
        mean_effective_sample_size_fraction=(
            float(np.mean(ess_fractions)) if ess_fractions else None
        ),
        mean_max_responsibility=(
            float(np.mean(max_weights)) if max_weights else None
        ),
    )
    return plan, diagnostics


def run_q5_multichain(
    task,
    rounds: int = 32,
    questions_per_round: int = 64,
    chains_per_question: int = 1,
    seed: int = 0,
    lr: float = 1e-4,
    model_name: str = MODEL_NAME,
    model_tok=None,
    initializer_prompt: str = "answer_derive",
    reward_requires_eos: bool = True,
    question_sampling: str = "epoch_shuffle",
    eval_every: int = 0,
    eval_rounds=None,
    eval_fn=None,
    diagnostics_fn=None,
    checkpoint_every: int = 0,
    checkpoint_fn=None,
    log=print,
) -> list[dict]:
    """Train Q5 with ``K`` persistent strict-correct particles per question."""

    if rounds < 0 or questions_per_round <= 0:
        raise ValueError("rounds must be nonnegative and questions_per_round positive")
    if chains_per_question not in Q5_MULTICHAIN_COUNTS:
        raise ValueError(
            f"chains_per_question must be one of {Q5_MULTICHAIN_COUNTS}"
        )
    if questions_per_round * chains_per_question != 64:
        raise ValueError("Q5 multi-particle current proposal budget must equal 64")
    if initializer_prompt != "answer_derive":
        raise ValueError("Q5 multi-particle initialization must use answer_derive")
    if not reward_requires_eos:
        raise ValueError("Q5 multi-particle states must be strict-correct with EOS")
    if question_sampling not in {"random", "epoch_shuffle"}:
        raise ValueError(f"unknown Q5 question schedule {question_sampling!r}")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if not math.isfinite(lr) or lr < 0:
        raise ValueError("lr must be finite and nonnegative")

    model, tok = (
        model_tok
        if model_tok is not None
        else load_model(seed=seed, model=model_name)
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Q5 multi-particle training requires trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=lr)
    rng = np.random.default_rng(seed)
    labelled_pool, answer_only_pool = _labelled_answer_only_pools(
        task,
        labelled_frac=0.0,
    )
    if labelled_pool:
        raise AssertionError("Q5 multi-particle screen is answer-only")
    sampler = QuestionSampler(
        answer_only_pool,
        np.random.default_rng(int(seed) * 1013 + 37),
        mode=question_sampling,
    )

    initialization_version = TorchParameterVersion.capture(model)
    memory = initialize_particle_memory(
        model,
        tok,
        task,
        answer_only_pool,
        chains_per_question=chains_per_question,
        initializer_prompt=initializer_prompt,
        reward_requires_eos=reward_requires_eos,
    )
    initialization_version.assert_unchanged(model)

    total_initializers = len(answer_only_pool) * chains_per_question
    total_current_proposals = 0
    total_generated_tokens = sum(
        int(state.state.span.sum())
        for state in memory.values()
        if state.state is not None
    )
    total_backward_tokens = 0
    total_backward_eos_tokens = 0
    total_steps = 0
    unique_questions_seen: set[int] = set()
    records: list[dict] = []

    for round_index in range(rounds):
        if question_sampling == "random":
            size = min(questions_per_round, len(answer_only_pool))
            question_ids = [
                int(value)
                for value in rng.choice(answer_only_pool, size=size, replace=False)
            ]
        else:
            question_ids = [
                int(value) for value in sampler.sample(questions_per_round)
            ]
        unique_questions_seen.update(question_ids)

        proposal_batch = collect_frozen_proposals(
            model,
            lambda: _sample_particle_proposals(
                model,
                tok,
                task,
                question_ids,
                chains_per_question=chains_per_question,
                prompt_mode="question",
                source="current_prior_proposal",
                reward_requires_eos=reward_requires_eos,
            ),
        )
        memory, transitions = advance_chains(memory, proposal_batch.proposals)

        valid_keys: list[ParticleKey] = []
        valid_traces: list[TriceTrace] = []
        for question_id in question_ids:
            for particle_index in range(chains_per_question):
                key = _particle_key(question_id, particle_index)
                state = memory[key]
                if state.state_correct:
                    assert state.state is not None
                    valid_keys.append(key)
                    valid_traces.append(state.state)
        if valid_traces:
            detached_scores = _score_trice_traces(
                model,
                tok,
                valid_traces,
                grad=False,
            )
            score_by_key = {
                key: float(score)
                for key, score in zip(valid_keys, detached_scores.detach().cpu())
            }
        else:
            score_by_key = {}
        plan, posterior = build_q5_particle_posterior_plan(
            memory,
            question_ids,
            chains_per_question=chains_per_question,
            log_scores=score_by_key,
        )

        model.train()
        optimizer.zero_grad()
        objective = torch.tensor(0.0)
        did_backward = False
        backward_traces: list[TriceTrace] = []
        if plan.terms:
            objective, did_backward = backward_estimator_plan(
                plan,
                lambda traces: _score_trice_traces(model, tok, traces),
                chunk_size=TRICE_BACKWARD_CHUNK_SIZE,
            )
            backward_traces = [term.payload for term in plan.terms]
        proposal_batch.assert_model_unchanged(model)
        if did_backward:
            optimizer.step()
            total_steps += 1
            total_backward_tokens += sum(
                int(trace.span.sum()) for trace in backward_traces
            )
            total_backward_eos_tokens += sum(
                int((trace.ids[trace.span] == tok.eos_token_id).sum())
                for trace in backward_traces
            )

        total_current_proposals += len(proposal_batch.proposals)
        total_generated_tokens += sum(
            int(proposal.payload.span.sum())
            for proposal in proposal_batch.proposals
        )
        evaluation = maybe_eval(
            model,
            round_index,
            rounds,
            eval_every,
            eval_fn,
            eval_rounds=eval_rounds,
        )
        accepted = sum(int(transition.accepted) for transition in transitions)
        memory_summary = chain_memory_diagnostics(memory)
        proposal_eos = [
            _trace_emitted_eos(proposal.payload, tok.eos_token_id)
            for proposal in proposal_batch.proposals
        ]
        retained_eos = [
            _trace_emitted_eos(term.payload, tok.eos_token_id)
            for term in plan.terms
        ]
        natural_eos_fraction = float(np.mean(proposal_eos)) if proposal_eos else 0.0
        retained_eos_fraction = float(np.mean(retained_eos)) if retained_eos else None

        record = {
            "round": round_index,
            "method": "q5_multichain",
            "chains_per_question": chains_per_question,
            "questions_per_round": len(question_ids),
            "particles_this_round": len(proposal_batch.proposals),
            "oracle": total_initializers + total_current_proposals,
            "verifier_calls": total_initializers + total_current_proposals,
            "gen": total_current_proposals,
            "llm_gen": total_initializers + total_current_proposals,
            "guide_gen": total_initializers,
            "generated_tokens": total_generated_tokens,
            "backward_tokens": total_backward_tokens,
            "backward_eos_tokens": total_backward_eos_tokens,
            "question_exposures": (round_index + 1) * len(question_ids),
            "unique_questions_seen": len(unique_questions_seen),
            "gsteps": total_steps,
            "valid_questions": posterior.valid_questions,
            "valid_particles": posterior.valid_particles,
            "mean_valid_particles": posterior.mean_valid_particles,
            "posterior_ess": posterior.mean_effective_sample_size,
            "posterior_ess_fraction": posterior.mean_effective_sample_size_fraction,
            "max_responsibility": posterior.mean_max_responsibility,
            "acceptance_fraction": (
                accepted / len(transitions) if transitions else None
            ),
            "mean_chain_age": memory_summary["mean_chain_age"],
            "unique_accepted_states": memory_summary["unique_accepted_states"],
            "reward_requires_eos": True,
            "natural_eos_fraction": natural_eos_fraction,
            "accepted_state_eos_fraction": retained_eos_fraction,
            "B_q5_multichain": float(objective.detach()),
            "F": float(objective.detach()),
            "test_acc": evaluation,
        }
        records.append(record)
        if diagnostics_fn is not None:
            diagnostics_fn(
                {
                    "schema_version": 1,
                    "method_family": "q5_multichain",
                    "answer_target_termination": "eos",
                    "round": round_index,
                    "completed_rounds": round_index + 1,
                    "design": {
                        "chains_per_question": chains_per_question,
                        "questions_per_round": len(question_ids),
                        "constant_current_proposals_per_round": len(
                            proposal_batch.proposals
                        ),
                        "initializer_prompt": initializer_prompt,
                        "current_proposal_prompt": "question",
                        "question_first_normalization": True,
                    },
                    "posterior": {
                        "selected_questions": posterior.selected_questions,
                        "valid_questions": posterior.valid_questions,
                        "valid_particles": posterior.valid_particles,
                        "mean_valid_particles": posterior.mean_valid_particles,
                        "mean_effective_sample_size": (
                            posterior.mean_effective_sample_size
                        ),
                        "mean_effective_sample_size_fraction": (
                            posterior.mean_effective_sample_size_fraction
                        ),
                        "mean_max_responsibility": (
                            posterior.mean_max_responsibility
                        ),
                    },
                    "transition": {
                        "proposals_this_round": len(transitions),
                        "accepted_this_round": accepted,
                        "acceptance_fraction_this_round": (
                            accepted / len(transitions) if transitions else None
                        ),
                    },
                    "memory": memory_summary,
                    "generation": {
                        "initializers_cumulative": total_initializers,
                        "current_proposals_cumulative": total_current_proposals,
                        "llm_generations_cumulative": (
                            total_initializers + total_current_proposals
                        ),
                        "generated_tokens_cumulative": total_generated_tokens,
                        "natural_eos_fraction": natural_eos_fraction,
                        "accepted_state_eos_fraction": retained_eos_fraction,
                    },
                    "reward": {
                        "requires_natural_eos": True,
                    },
                    "optimizer": {
                        "gradient_steps_cumulative": total_steps,
                        "backward_tokens_cumulative": total_backward_tokens,
                        "backward_eos_tokens_cumulative": total_backward_eos_tokens,
                    },
                    "objective": {
                        "B_q5_multichain": record["B_q5_multichain"],
                        "F": record["F"],
                    },
                    "parameter_contract": {
                        "frozen_until_optimizer_step": True,
                        "optimizer_updates_per_round": int(did_backward),
                        "backward_chunk_size": TRICE_BACKWARD_CHUNK_SIZE,
                    },
                }
            )
        if (
            checkpoint_fn is not None
            and checkpoint_every > 0
            and (round_index + 1) % checkpoint_every == 0
            and (round_index + 1) < rounds
        ):
            checkpoint_fn(model, round_index + 1)
        log(record)

    return records


__all__ = [
    "Q5_MULTICHAIN_COUNTS",
    "Q5ParticlePosteriorDiagnostics",
    "build_q5_particle_posterior_plan",
    "initialize_particle_memory",
    "run_q5_multichain",
]
