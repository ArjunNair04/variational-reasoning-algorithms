"""Persistent-chain TRICE variant for AC-ALG1.

This module implements the deterministic-likelihood form of TRICE from:

    Phan et al., "Training Chain-of-Thought via Latent-Variable Inference",
    NeurIPS 2023, arXiv:2312.02179.

For a question ``x`` and known answer ``y``, the latent state is a complete
sample ``z`` from ``p_theta(z | x)``.  The external verifier implements the
binary likelihood ``c(z, y)``.  Proposing from the model prior makes the
independence-Metropolis rule especially simple: accept a proposal exactly when
it is correct, otherwise retain the persistent state.

The core below is deliberately model-agnostic and immutable.  It exposes:

* one persistent chain state per question;
* the exact deterministic independence transition;
* the accepted-state (basic) score estimator; and
* the paper's leave-one-out rejected-proposal control variate.

``run_ac_alg1_trice`` is a thin language-model integration.  It keeps the
three-term AC-ALG1 shape (gold supervision, labelled latent term, answer-only
latent term) but replaces each trace buffer by one persistent TRICE state.  A
macrocycle contains exactly one frozen-theta proposal per selected question
and exactly one subsequent optimizer update.  This ordering matters: changing
parameters between proposal generation and construction of the score
estimator invalidates the score-function control variate.

The implementation intentionally does not include the paper's gradient
subsampling estimator.  The full, non-subsampled control-variate estimator is
both simpler and less error-prone for the diagnostic-scale experiments for
which this variant is intended.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Callable, Generic, Hashable, Mapping, Sequence, TypeVar

import numpy as np
import torch

from ac_alg1 import (
    _B_sup,
    _build_proposal_prompt,
    _labelled_answer_only_pools,
)
from common import (
    DEV,
    MODEL_NAME,
    QuestionSampler,
    encode_task_prompt,
    load_model,
    maybe_eval,
    natural_eos_mask,
    sample_multi,
    seq_logprobs,
    task_pad_token_id,
)


PayloadT = TypeVar("PayloadT")
QuestionId = Hashable

TRICE_ESTIMATORS = ("basic", "control_variate")
TRICE_BACKWARD_CHUNK_SIZE = 4


def _require_hashable(value: Hashable, name: str) -> None:
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{name} must be hashable") from error


@dataclass(frozen=True)
class TriceChainState(Generic[PayloadT]):
    """Immutable state of one persistent rationale chain.

    ``age`` is the number of consecutive rejected proposals since the most
    recent accepted transition.  A correct initializer is included in
    ``unique_accepted_state_ids`` but not in ``acceptances``, which counts only
    independence-chain transitions.
    """

    question_id: QuestionId
    state: PayloadT | None = None
    state_id: Hashable | None = None
    state_correct: bool = False
    age: int = 0
    proposals: int = 0
    acceptances: int = 0
    rejections: int = 0
    unique_accepted_state_ids: frozenset[Hashable] = frozenset()

    def __post_init__(self) -> None:
        _require_hashable(self.question_id, "question_id")
        if (self.state is None) != (self.state_id is None):
            raise ValueError("state and state_id must either both be set or both be None")
        if self.state_id is not None:
            _require_hashable(self.state_id, "state_id")
        if self.state_correct and self.state is None:
            raise ValueError("a correct chain state must contain a payload")
        if min(self.age, self.proposals, self.acceptances, self.rejections) < 0:
            raise ValueError("chain counters must be nonnegative")
        if self.acceptances + self.rejections != self.proposals:
            raise ValueError("acceptances + rejections must equal proposals")
        if self.state_correct and self.state_id not in self.unique_accepted_state_ids:
            raise ValueError(
                "a correct current state must appear in unique_accepted_state_ids"
            )


def initialize_chain(
    question_id: QuestionId,
    state: PayloadT | None = None,
    *,
    state_id: Hashable | None = None,
    state_correct: bool = False,
) -> TriceChainState[PayloadT]:
    """Create a chain, optionally from a hinted-guide or gold initializer."""

    unique_ids = (
        frozenset((state_id,))
        if state_correct and state_id is not None
        else frozenset()
    )
    return TriceChainState(
        question_id=question_id,
        state=state,
        state_id=state_id,
        state_correct=bool(state_correct),
        unique_accepted_state_ids=unique_ids,
    )


@dataclass(frozen=True)
class TriceProposal(Generic[PayloadT]):
    """One exact prior proposal plus externally supplied correctness."""

    question_id: QuestionId
    payload: PayloadT
    state_id: Hashable
    correct: bool

    def __post_init__(self) -> None:
        _require_hashable(self.question_id, "question_id")
        _require_hashable(self.state_id, "state_id")
        if not isinstance(self.correct, (bool, np.bool_)):
            raise TypeError("proposal correctness must be an external boolean")


@dataclass(frozen=True)
class TriceTransition(Generic[PayloadT]):
    """Before/proposal/after record for one independence-chain transition."""

    before: TriceChainState[PayloadT]
    proposal: TriceProposal[PayloadT]
    after: TriceChainState[PayloadT]
    accepted: bool


def independence_step(
    chain: TriceChainState[PayloadT],
    proposal: TriceProposal[PayloadT],
) -> tuple[TriceChainState[PayloadT], TriceTransition[PayloadT]]:
    """Take the deterministic-likelihood TRICE independence-MH step.

    With proposal distribution ``p_theta(z | x)`` and target
    ``p_theta(z | x, y) ∝ c(z, y) p_theta(z | x)``, the model probabilities
    cancel from the Metropolis ratio.  A correct proposal is accepted with
    probability one.  An incorrect proposal is rejected, including the
    formally ``0/0`` case in which the previous state is also incorrect.
    """

    if chain.question_id != proposal.question_id:
        raise ValueError("proposal question_id does not match its chain")

    accepted = bool(proposal.correct)
    if accepted:
        unique_ids = chain.unique_accepted_state_ids | frozenset((proposal.state_id,))
        after = replace(
            chain,
            state=proposal.payload,
            state_id=proposal.state_id,
            state_correct=True,
            age=0,
            proposals=chain.proposals + 1,
            acceptances=chain.acceptances + 1,
            unique_accepted_state_ids=unique_ids,
        )
    else:
        after = replace(
            chain,
            age=chain.age + 1,
            proposals=chain.proposals + 1,
            rejections=chain.rejections + 1,
        )
    return after, TriceTransition(
        before=chain,
        proposal=proposal,
        after=after,
        accepted=accepted,
    )


def advance_chains(
    memory: Mapping[QuestionId, TriceChainState[PayloadT]],
    proposals: Sequence[TriceProposal[PayloadT]],
) -> tuple[
    dict[QuestionId, TriceChainState[PayloadT]],
    tuple[TriceTransition[PayloadT], ...],
]:
    """Purely advance several distinct chains in parallel.

    At most one proposal per question is allowed in a macrocycle.  This makes
    the transition batch independent of iteration order and mirrors Algorithm
    1's parallel MCMC step.
    """

    updated = dict(memory)
    transitions: list[TriceTransition[PayloadT]] = []
    seen: set[QuestionId] = set()
    for proposal in proposals:
        if proposal.question_id in seen:
            raise ValueError("a TRICE macrocycle permits one proposal per question")
        seen.add(proposal.question_id)
        if proposal.question_id not in memory:
            raise KeyError(f"missing chain for question {proposal.question_id!r}")
        state, transition = independence_step(
            memory[proposal.question_id],
            proposal,
        )
        updated[proposal.question_id] = state
        transitions.append(transition)
    return updated, tuple(transitions)


@dataclass(frozen=True)
class TriceScoreTerm(Generic[PayloadT]):
    """One scalar-weighted score ``log p_theta(z | x)`` in an estimator."""

    question_id: QuestionId
    payload: PayloadT
    state_id: Hashable
    coefficient: float
    role: str


@dataclass(frozen=True)
class TriceEstimatorPlan(Generic[PayloadT]):
    """A detached description of a basic or control-variate score estimator."""

    estimator: str
    terms: tuple[TriceScoreTerm[PayloadT], ...]
    valid_chains: int
    beta_by_question: Mapping[QuestionId, float]
    rejected_proposals_used: int
    rejection_absolute_weight: float


def leave_one_out_acceptance_scales(
    transitions: Sequence[TriceTransition[PayloadT]],
) -> dict[QuestionId, float]:
    """Compute the paper's leave-one-out ``beta_m`` values.

    For updated-state correctness ``c'_m`` and proposal correctness
    ``tilde c_m``,

    ``beta_m = sum_{j != m} c'_j tilde_c_j / sum_{j != m} c'_j``.

    Excluding both indicators for example ``m`` is what keeps ``beta_m``
    independent of that example's proposal score.  If there is no other valid
    chain, the only safe scale is zero.
    """

    question_ids = [transition.after.question_id for transition in transitions]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("transitions must contain distinct question ids")

    valid = [int(transition.after.state_correct) for transition in transitions]
    accepted_valid = [
        int(transition.after.state_correct and transition.proposal.correct)
        for transition in transitions
    ]
    total_valid = sum(valid)
    total_accepted_valid = sum(accepted_valid)

    scales: dict[QuestionId, float] = {}
    for transition, own_valid, own_accepted in zip(
        transitions, valid, accepted_valid
    ):
        denominator = total_valid - own_valid
        numerator = total_accepted_valid - own_accepted
        scales[transition.after.question_id] = (
            float(numerator / denominator) if denominator > 0 else 0.0
        )
    return scales


def build_estimator_plan(
    transitions: Sequence[TriceTransition[PayloadT]],
    estimator: str = "basic",
) -> TriceEstimatorPlan[PayloadT]:
    """Build the accepted-state or full TRICE control-variate estimator.

    The basic plan estimates the posterior score with the updated persistent
    states:

    ``(1 / sum c'_m) sum_m c'_m grad log p_theta(z'_m | x_m)``.

    The control-variate plan subtracts
    ``beta_m grad log p_theta(tilde z_m | x_m)`` from every valid-chain
    contribution.  For a rejected proposal this explicitly combines a
    positive score for the retained correct state with a negative score for
    the incorrect proposal.  All coefficients are detached Python floats.
    """

    if estimator not in TRICE_ESTIMATORS:
        raise ValueError(f"unknown TRICE estimator {estimator!r}")

    question_ids = [transition.after.question_id for transition in transitions]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("transitions must contain distinct question ids")

    valid_transitions = [
        transition for transition in transitions if transition.after.state_correct
    ]
    valid_count = len(valid_transitions)
    betas = (
        leave_one_out_acceptance_scales(transitions)
        if estimator == "control_variate"
        else {question_id: 0.0 for question_id in question_ids}
    )
    if valid_count == 0:
        return TriceEstimatorPlan(
            estimator=estimator,
            terms=(),
            valid_chains=0,
            beta_by_question=betas,
            rejected_proposals_used=0,
            rejection_absolute_weight=0.0,
        )

    normalizer = 1.0 / valid_count
    terms: list[TriceScoreTerm[PayloadT]] = []
    rejected_used = 0
    rejection_absolute_weight = 0.0
    for transition in valid_transitions:
        after = transition.after
        if after.state is None or after.state_id is None:
            raise AssertionError("valid TRICE chain is missing its state")
        terms.append(
            TriceScoreTerm(
                question_id=after.question_id,
                payload=after.state,
                state_id=after.state_id,
                coefficient=normalizer,
                role="accepted_state",
            )
        )

        if estimator != "control_variate":
            continue
        beta = float(betas[after.question_id])
        coefficient = -beta * normalizer
        if coefficient == 0.0:
            continue
        rejected = not bool(transition.proposal.correct)
        terms.append(
            TriceScoreTerm(
                question_id=after.question_id,
                payload=transition.proposal.payload,
                state_id=transition.proposal.state_id,
                coefficient=coefficient,
                role=(
                    "rejected_proposal_control"
                    if rejected
                    else "accepted_proposal_control"
                ),
            )
        )
        if rejected:
            rejected_used += 1
            rejection_absolute_weight += abs(coefficient)

    return TriceEstimatorPlan(
        estimator=estimator,
        terms=tuple(terms),
        valid_chains=valid_count,
        beta_by_question=betas,
        rejected_proposals_used=rejected_used,
        rejection_absolute_weight=rejection_absolute_weight,
    )


def evaluate_estimator_plan(
    plan: TriceEstimatorPlan[PayloadT],
    score_batch_fn: Callable[[Sequence[PayloadT]], torch.Tensor],
) -> torch.Tensor:
    """Evaluate a plan without detaching the model scores or their gradients."""

    if not plan.terms:
        raise ValueError("cannot evaluate a TRICE plan with no valid chains")
    scores = score_batch_fn([term.payload for term in plan.terms])
    if not isinstance(scores, torch.Tensor):
        raise TypeError("score_batch_fn must return a torch.Tensor")
    if scores.ndim != 1 or scores.shape[0] != len(plan.terms):
        raise ValueError(
            "score_batch_fn must return one scalar score per estimator term"
        )
    coefficients = scores.new_tensor(
        [term.coefficient for term in plan.terms]
    )
    return torch.sum(coefficients * scores)


def backward_estimator_plan(
    plan: TriceEstimatorPlan[PayloadT],
    score_batch_fn: Callable[[Sequence[PayloadT]], torch.Tensor],
    *,
    weight: float = 1.0,
    chunk_size: int = TRICE_BACKWARD_CHUNK_SIZE,
) -> tuple[torch.Tensor, bool]:
    """Accumulate the negative plan gradient without retaining every score graph.

    The estimator is a linear sum over terms, so backpropagating each bounded
    chunk before scoring the next one gives the same gradient as one monolithic
    backward pass. Parameters are not stepped here; the caller still performs
    the single TRICE M-step after every component has accumulated its gradient.
    """

    if not plan.terms:
        raise ValueError("cannot evaluate a TRICE plan with no valid chains")
    if chunk_size <= 0:
        raise ValueError("TRICE backward chunk_size must be positive")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("TRICE estimator weight must be finite and nonnegative")

    detached_total: torch.Tensor | None = None
    did_backward = False
    for start in range(0, len(plan.terms), chunk_size):
        terms = plan.terms[start : start + chunk_size]
        scores = score_batch_fn([term.payload for term in terms])
        if not isinstance(scores, torch.Tensor):
            raise TypeError("score_batch_fn must return a torch.Tensor")
        if scores.ndim != 1 or scores.shape[0] != len(terms):
            raise ValueError(
                "score_batch_fn must return one scalar score per estimator term"
            )
        coefficients = scores.new_tensor(
            [term.coefficient for term in terms]
        )
        objective = float(weight) * torch.sum(coefficients * scores)
        value = objective.detach()
        detached_total = (
            value if detached_total is None else detached_total + value
        )
        if objective.requires_grad:
            (-objective).backward()
            did_backward = True
        del objective, coefficients, scores

    assert detached_total is not None
    return detached_total, did_backward


@dataclass(frozen=True)
class TorchParameterVersion:
    """Lightweight token proving that model parameters were not mutated."""

    parameter_ids: tuple[int, ...]
    versions: tuple[int, ...]

    @classmethod
    def capture(cls, model) -> "TorchParameterVersion":
        parameters = tuple(model.parameters())
        return cls(
            parameter_ids=tuple(id(parameter) for parameter in parameters),
            versions=tuple(int(parameter._version) for parameter in parameters),
        )

    def assert_unchanged(self, model) -> None:
        parameters = tuple(model.parameters())
        current_ids = tuple(id(parameter) for parameter in parameters)
        current_versions = tuple(int(parameter._version) for parameter in parameters)
        if current_ids != self.parameter_ids or current_versions != self.versions:
            raise RuntimeError(
                "model parameters changed inside a frozen TRICE E-step macrocycle"
            )


@dataclass(frozen=True)
class FrozenTriceProposalBatch(Generic[PayloadT]):
    """Proposal batch tied to the exact model-parameter version that sampled it."""

    proposals: tuple[TriceProposal[PayloadT], ...]
    parameter_version: TorchParameterVersion

    def assert_model_unchanged(self, model) -> None:
        self.parameter_version.assert_unchanged(model)


def collect_frozen_proposals(
    model,
    proposal_fn: Callable[[], Sequence[TriceProposal[PayloadT]]],
) -> FrozenTriceProposalBatch[PayloadT]:
    """Call a sampler once and seal its outputs to the current parameter version."""

    version = TorchParameterVersion.capture(model)
    proposals = tuple(proposal_fn())
    version.assert_unchanged(model)
    question_ids = [proposal.question_id for proposal in proposals]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("a frozen proposal batch must contain distinct questions")
    return FrozenTriceProposalBatch(
        proposals=proposals,
        parameter_version=version,
    )


def chain_memory_diagnostics(
    memory: Mapping[QuestionId, TriceChainState[PayloadT]],
) -> dict[str, float | int | None]:
    """Summarize acceptance, age, validity, and unique-state exploration."""

    states = list(memory.values())
    proposals = sum(state.proposals for state in states)
    acceptances = sum(state.acceptances for state in states)
    ages = [state.age for state in states]
    unique_states = sum(len(state.unique_accepted_state_ids) for state in states)
    return {
        "chains": len(states),
        "valid_chains": sum(int(state.state_correct) for state in states),
        "proposals": proposals,
        "acceptances": acceptances,
        "rejections": proposals - acceptances,
        "acceptance_fraction": (
            acceptances / proposals if proposals else None
        ),
        "mean_chain_age": (
            float(sum(ages) / len(ages)) if ages else None
        ),
        "max_chain_age": max(ages) if ages else None,
        "unique_accepted_states": unique_states,
        "mean_unique_accepted_states": (
            float(unique_states / len(states)) if states else None
        ),
    }


def macrocycle_diagnostics(
    transitions: Sequence[TriceTransition[PayloadT]],
    plan: TriceEstimatorPlan[PayloadT],
    memory: Mapping[QuestionId, TriceChainState[PayloadT]],
) -> dict[str, object]:
    """Combine transition-local and persistent-memory TRICE diagnostics."""

    accepted = sum(int(transition.accepted) for transition in transitions)
    betas = list(plan.beta_by_question.values())
    return {
        "estimator": plan.estimator,
        "proposals_this_macrocycle": len(transitions),
        "accepted_this_macrocycle": accepted,
        "rejected_this_macrocycle": len(transitions) - accepted,
        "acceptance_fraction_this_macrocycle": (
            accepted / len(transitions) if transitions else None
        ),
        "valid_chains_in_estimator": plan.valid_chains,
        "mean_leave_one_out_beta": (
            float(sum(betas) / len(betas)) if betas else None
        ),
        "rejected_proposals_used": plan.rejected_proposals_used,
        "rejection_control_absolute_weight": plan.rejection_absolute_weight,
        "memory": chain_memory_diagnostics(memory),
    }


@dataclass(frozen=True)
class TriceTrace:
    """Exact question-conditioned completion scored by the LM M-step."""

    ids: torch.Tensor
    span: torch.Tensor
    text: str
    question_id: int
    source: str

    def __post_init__(self) -> None:
        if self.ids.ndim != 1 or self.span.ndim != 1:
            raise ValueError("TRICE trace ids and span must be one-dimensional")
        if self.ids.shape != self.span.shape:
            raise ValueError("TRICE trace ids and span must have matching shapes")
        if self.span.dtype != torch.bool:
            raise TypeError("TRICE trace span must be boolean")
        if not bool(self.span.any()):
            raise ValueError("TRICE trace must contain at least one scored token")


def _trace_state_id(trace: TriceTrace) -> str:
    scored_ids = trace.ids[trace.span].detach().cpu().tolist()
    payload = ",".join(str(int(token_id)) for token_id in scored_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _trace_emitted_eos(trace: TriceTrace, eos_token_id: int) -> bool:
    active = trace.ids[trace.span]
    return bool(active.numel() and int(active[-1]) == int(eos_token_id))


def _completion_trace(
    tok,
    task,
    question_id: int,
    generated_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    text: str,
    source: str,
) -> TriceTrace:
    """Reanchor an exact sampled completion under the original question prompt."""

    completion_ids = generated_ids[completion_mask].detach().cpu()
    if completion_ids.numel() == 0:
        # A properly terminated categorical sequence still has an EOS score.
        completion_ids = torch.tensor([tok.eos_token_id], dtype=torch.long)
    prompt_ids = encode_task_prompt(
        tok,
        task,
        int(question_id),
        return_tensors="pt",
    ).input_ids[0].detach().cpu()
    ids = torch.cat((prompt_ids, completion_ids.to(dtype=prompt_ids.dtype)))
    span = torch.zeros(ids.shape[0], dtype=torch.bool)
    span[prompt_ids.shape[0]:] = True
    return TriceTrace(
        ids=ids,
        span=span,
        text=text,
        question_id=int(question_id),
        source=source,
    )


def _gold_trace(tok, task, question_id: int) -> TriceTrace:
    solution = task.gold_solution[int(question_id)]
    prompt_ids = encode_task_prompt(
        tok,
        task,
        int(question_id),
        return_tensors="pt",
    ).input_ids[0].detach().cpu()
    solution_ids = torch.tensor(
        tok(" " + solution, add_special_tokens=False).input_ids,
        dtype=prompt_ids.dtype,
    )
    if solution_ids.numel() == 0:
        raise ValueError("gold TRICE initializer cannot be empty")
    if tok.eos_token_id is None:
        raise ValueError("gold TRICE initializer requires an EOS token")
    eos_id = torch.tensor([tok.eos_token_id], dtype=prompt_ids.dtype)
    ids = torch.cat((prompt_ids, solution_ids, eos_id))
    span = torch.zeros(ids.shape[0], dtype=torch.bool)
    span[prompt_ids.shape[0]:] = True
    return TriceTrace(
        ids=ids,
        span=span,
        text=solution,
        question_id=int(question_id),
        source="gold_initializer",
    )


def _sample_trace_proposals(
    model,
    tok,
    task,
    question_ids: Sequence[int],
    *,
    prompt_mode: str,
    source: str,
    reward_requires_eos: bool = False,
) -> tuple[TriceProposal[TriceTrace], ...]:
    if not question_ids:
        return ()
    task_builder = getattr(task, "build_proposal_prompt", None)
    prompts = [
        (
            task_builder(int(question_id), prompt_mode)
            if task_builder is not None
            else _build_proposal_prompt(
                task.prompts[int(question_id)],
                task.gold_answer[int(question_id)],
                prompt_mode,
                question=(
                    task.questions[int(question_id)]
                    if hasattr(task, "questions")
                    else None
                ),
            )
        )
        for question_id in question_ids
    ]
    ids, completion_masks, texts = sample_multi(
        model,
        tok,
        prompts,
        temperature=1.0,
        max_new=getattr(task, "max_new", 40),
    )
    rewards = task.reward(texts, pids=list(question_ids))
    natural_eos = natural_eos_mask(
        ids,
        completion_masks,
        tok.eos_token_id,
    )
    proposals: list[TriceProposal[TriceTrace]] = []
    for row_index, question_id in enumerate(question_ids):
        trace = _completion_trace(
            tok,
            task,
            int(question_id),
            ids[row_index],
            completion_masks[row_index],
            texts[row_index],
            source,
        )
        proposals.append(
            TriceProposal(
                question_id=int(question_id),
                payload=trace,
                state_id=_trace_state_id(trace),
                correct=bool(
                    float(rewards[row_index]) > 0.5
                    and (
                        not reward_requires_eos
                        or bool(natural_eos[row_index])
                    )
                ),
            )
        )
    return tuple(proposals)


def _pad_trice_traces(
    tok,
    traces: Sequence[TriceTrace],
) -> tuple[torch.Tensor, torch.Tensor]:
    pad = torch.nn.utils.rnn.pad_sequence
    ids = pad(
        [trace.ids for trace in traces],
        batch_first=True,
        padding_value=task_pad_token_id(tok),
    ).to(DEV)
    spans = pad(
        [trace.span for trace in traces],
        batch_first=True,
        padding_value=False,
    ).to(DEV)
    return ids, spans


def _score_trice_traces(
    model,
    tok,
    traces: Sequence[TriceTrace],
    *,
    grad: bool = True,
) -> torch.Tensor:
    ids, spans = _pad_trice_traces(tok, traces)
    return seq_logprobs(
        model,
        ids,
        spans,
        grad=grad,
        length_norm=False,
    )


def _sample_unique_minibatch(
    pool: Sequence[int],
    rng: np.random.Generator,
    batch_size: int,
) -> list[int]:
    if not pool or batch_size <= 0:
        return []
    size = min(int(batch_size), len(pool))
    return [int(value) for value in rng.choice(pool, size=size, replace=False)]


def _initialize_selected_chains(
    model,
    tok,
    task,
    memory: Mapping[int, TriceChainState[TriceTrace]],
    labelled_ids: Sequence[int],
    answer_only_ids: Sequence[int],
    *,
    initializer_prompt: str,
    reward_requires_eos: bool = False,
) -> dict[int, TriceChainState[TriceTrace]]:
    updated = dict(memory)
    for question_id in labelled_ids:
        if int(question_id) in updated:
            continue
        trace = _gold_trace(tok, task, int(question_id))
        updated[int(question_id)] = initialize_chain(
            int(question_id),
            trace,
            state_id=_trace_state_id(trace),
            state_correct=True,
        )

    missing_answer_only = [
        int(question_id)
        for question_id in answer_only_ids
        if int(question_id) not in updated
    ]
    guide_proposals = _sample_trace_proposals(
        model,
        tok,
        task,
        missing_answer_only,
        prompt_mode=initializer_prompt,
        source="hinted_guide_initializer",
        reward_requires_eos=reward_requires_eos,
    )
    for proposal in guide_proposals:
        updated[int(proposal.question_id)] = initialize_chain(
            int(proposal.question_id),
            proposal.payload,
            state_id=proposal.state_id,
            state_correct=bool(proposal.correct),
        )
    return updated


def run_ac_alg1_trice(
    task,
    rounds: int = 40,
    L_batch: int = 32,
    U_batch: int = 32,
    seed: int = 0,
    lr: float = 1e-4,
    model_name: str = MODEL_NAME,
    model_tok=None,
    labelled_frac: float = 0.5,
    estimator: str = "basic",
    initializer_prompt: str = "answer_derive",
    supervised_weight: float = 1.0,
    labelled_trice_weight: float = 1.0,
    answer_only_trice_weight: float = 1.0,
    reward_requires_eos: bool = False,
    question_sampling: str = "random",
    eval_every: int = 0,
    eval_rounds=None,
    eval_fn=None,
    diagnostics_fn=None,
    checkpoint_every: int = 0,
    checkpoint_fn=None,
    log=print,
) -> list[dict]:
    """Train a persistent-chain variant with the AC-ALG1 three-term shape.

    The prior proposal is intentionally fixed to the unmodified question
    prompt, temperature 1, and one proposal per selected question.  These are
    not cosmetic restrictions: the TRICE control variate uses the identity
    ``E_{p_theta(z|x)}[grad log p_theta(z|x)] = 0``.  An answer-conditioned
    hinted guide is used only to initialize previously unseen answer-only
    chains, as in the paper.

    Unlike ``run_ac_alg1``, this runner performs one optimizer update per
    macrocycle.  Reusing a frozen proposal for multiple parameter updates would
    no longer be the estimator derived by Phan et al.
    """

    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    if L_batch < 0 or U_batch < 0:
        raise ValueError("TRICE batch sizes must be nonnegative")
    if estimator not in TRICE_ESTIMATORS:
        raise ValueError(f"unknown TRICE estimator {estimator!r}")
    if initializer_prompt not in ("answer_hint", "answer_derive"):
        raise ValueError(
            "TRICE initializer_prompt must be an answer-conditioned hinted guide"
        )
    if question_sampling not in {"random", "epoch_shuffle"}:
        raise ValueError(f"unknown TRICE question schedule {question_sampling!r}")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    for name, value in (
        ("lr", lr),
        ("supervised_weight", supervised_weight),
        ("labelled_trice_weight", labelled_trice_weight),
        ("answer_only_trice_weight", answer_only_trice_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    model, tok = (
        model_tok
        if model_tok is not None
        else load_model(seed=seed, model=model_name)
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("TRICE requires at least one trainable model parameter")
    optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
    rng = np.random.default_rng(seed)
    labelled_pool, answer_only_pool = _labelled_answer_only_pools(
        task,
        labelled_frac=labelled_frac,
    )
    labelled_sampler = QuestionSampler(
        labelled_pool,
        np.random.default_rng(int(seed) * 1013 + 31),
        mode=question_sampling,
    )
    answer_only_sampler = QuestionSampler(
        answer_only_pool,
        np.random.default_rng(int(seed) * 1013 + 37),
        mode=question_sampling,
    )

    memory: dict[int, TriceChainState[TriceTrace]] = {}
    records: list[dict] = []
    total_prior_generated = 0
    total_guide_generated = len(answer_only_pool)
    total_generated_tokens = 0
    total_backward_tokens = 0
    total_backward_eos_tokens = 0
    total_steps = 0
    unique_questions_seen: set[int] = set()

    # Algorithm 1 initializes every persistent chain before the first M-step.
    # Doing this lazily would make later guide states depend on earlier updates.
    initialization_version = TorchParameterVersion.capture(model)
    memory = _initialize_selected_chains(
        model,
        tok,
        task,
        memory,
        labelled_pool,
        answer_only_pool,
        initializer_prompt=initializer_prompt,
        reward_requires_eos=reward_requires_eos,
    )
    initialization_version.assert_unchanged(model)
    total_generated_tokens += sum(
        int(memory[question_id].state.span.sum())
        for question_id in answer_only_pool
        if memory[question_id].state is not None
    )

    for round_index in range(rounds):
        if question_sampling == "random":
            labelled_ids = _sample_unique_minibatch(
                labelled_pool, rng, L_batch
            )
            answer_only_ids = _sample_unique_minibatch(
                answer_only_pool, rng, U_batch
            )
        else:
            labelled_ids = [
                int(value) for value in labelled_sampler.sample(L_batch)
            ]
            answer_only_ids = [
                int(value) for value in answer_only_sampler.sample(U_batch)
            ]
        selected_ids = labelled_ids + answer_only_ids
        unique_questions_seen.update(selected_ids)

        parameter_version = TorchParameterVersion.capture(model)
        proposal_batch = collect_frozen_proposals(
            model,
            lambda: _sample_trace_proposals(
                model,
                tok,
                task,
                selected_ids,
                prompt_mode="question",
                source="prior_proposal",
                reward_requires_eos=reward_requires_eos,
            ),
        )
        # Initializer and proposal generation must both use the same theta.
        parameter_version.assert_unchanged(model)
        proposal_batch.assert_model_unchanged(model)
        memory, transitions = advance_chains(memory, proposal_batch.proposals)
        transition_by_id = {
            int(transition.after.question_id): transition
            for transition in transitions
        }
        labelled_transitions = [
            transition_by_id[question_id] for question_id in labelled_ids
        ]
        answer_only_transitions = [
            transition_by_id[question_id] for question_id in answer_only_ids
        ]
        labelled_plan = build_estimator_plan(
            labelled_transitions,
            estimator=estimator,
        )
        answer_only_plan = build_estimator_plan(
            answer_only_transitions,
            estimator=estimator,
        )

        model.train()
        optimizer.zero_grad()
        objective_terms: dict[str, torch.Tensor] = {}
        did_backward = False
        backward_traces: list[TriceTrace] = []
        if supervised_weight and labelled_ids:
            supervised_objective = supervised_weight * _B_sup(
                model,
                tok,
                task,
                labelled_ids,
            )
            objective_terms["B_sup"] = supervised_objective.detach()
            if supervised_objective.requires_grad:
                (-supervised_objective).backward()
                did_backward = True
            backward_traces.extend(
                _gold_trace(tok, task, question_id)
                for question_id in labelled_ids
            )
        if labelled_trice_weight and labelled_plan.terms:
            objective_terms["B_prime_trice"], component_backward = (
                backward_estimator_plan(
                    labelled_plan,
                    lambda traces: _score_trice_traces(model, tok, traces),
                    weight=labelled_trice_weight,
                )
            )
            did_backward = did_backward or component_backward
            backward_traces.extend(
                term.payload for term in labelled_plan.terms
            )
        if answer_only_trice_weight and answer_only_plan.terms:
            objective_terms["B_trice"], component_backward = (
                backward_estimator_plan(
                    answer_only_plan,
                    lambda traces: _score_trice_traces(model, tok, traces),
                    weight=answer_only_trice_weight,
                )
            )
            did_backward = did_backward or component_backward
            backward_traces.extend(
                term.payload for term in answer_only_plan.terms
            )

        # Forward/backward may create gradients, but theta must remain the
        # proposal-generating theta until the one and only M-step update.
        proposal_batch.assert_model_unchanged(model)
        if did_backward:
            proposal_batch.assert_model_unchanged(model)
            optimizer.step()
            total_steps += 1
            total_backward_tokens += sum(
                int(trace.span.sum()) for trace in backward_traces
            )
            total_backward_eos_tokens += sum(
                int(
                    (
                        trace.ids[trace.span]
                        == tok.eos_token_id
                    ).sum()
                )
                for trace in backward_traces
            )

        total_prior_generated += len(selected_ids)
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
        combined_plan = build_estimator_plan(
            transitions,
            estimator=estimator,
        )
        diagnostics = macrocycle_diagnostics(
            transitions,
            combined_plan,
            memory,
        )
        diagnostics["labelled"] = macrocycle_diagnostics(
            labelled_transitions,
            labelled_plan,
            {question_id: memory[question_id] for question_id in labelled_ids},
        )
        diagnostics["answer_only"] = macrocycle_diagnostics(
            answer_only_transitions,
            answer_only_plan,
            {
                question_id: memory[question_id]
                for question_id in answer_only_ids
            },
        )
        diagnostics["parameter_contract"] = {
            "proposal_temperature": 1.0,
            "proposal_prompt": "question",
            "reward_requires_eos": bool(reward_requires_eos),
            "optimizer_updates_per_macrocycle": int(bool(objective_terms)),
            "backward_chunk_size": TRICE_BACKWARD_CHUNK_SIZE,
            "frozen_until_optimizer_step": True,
        }
        prior_eos = [
            _trace_emitted_eos(proposal.payload, tok.eos_token_id)
            for proposal in proposal_batch.proposals
        ]
        accepted_state_eos = [
            _trace_emitted_eos(term.payload, tok.eos_token_id)
            for term in combined_plan.terms
            if term.role == "accepted_state"
        ]
        natural_eos_fraction = (
            float(np.mean(prior_eos)) if prior_eos else 0.0
        )
        accepted_state_eos_fraction = (
            float(np.mean(accepted_state_eos))
            if accepted_state_eos
            else None
        )

        record = {
            "round": round_index,
            "method": "ac_alg1_trice",
            "estimator": estimator,
            "oracle": total_prior_generated + total_guide_generated,
            "verifier_calls": total_prior_generated + total_guide_generated,
            "gen": total_prior_generated,
            "llm_gen": total_prior_generated + total_guide_generated,
            "guide_gen": total_guide_generated,
            "generated_tokens": total_generated_tokens,
            "backward_tokens": total_backward_tokens,
            "backward_eos_tokens": total_backward_eos_tokens,
            "question_exposures": total_prior_generated,
            "unique_questions_seen": len(unique_questions_seen),
            "questions_this_round": len(selected_ids),
            "gsteps": total_steps,
            "backward_chunk_size": TRICE_BACKWARD_CHUNK_SIZE,
            "labelled_questions": len(labelled_ids),
            "answer_only_questions": len(answer_only_ids),
            "valid_chains": diagnostics["memory"]["valid_chains"],
            "acceptance_fraction": diagnostics[
                "acceptance_fraction_this_macrocycle"
            ],
            "mean_chain_age": diagnostics["memory"]["mean_chain_age"],
            "unique_accepted_states": diagnostics["memory"][
                "unique_accepted_states"
            ],
            "rejected_proposals_used": diagnostics[
                "rejected_proposals_used"
            ],
            "reward_requires_eos": bool(reward_requires_eos),
            "natural_eos_fraction": natural_eos_fraction,
            "accepted_state_eos_fraction": accepted_state_eos_fraction,
            "test_acc": evaluation,
        }
        for name in ("B_sup", "B_prime_trice", "B_trice"):
            value = objective_terms.get(name)
            record[name] = float(value.detach()) if value is not None else 0.0
        record["F"] = sum(record[name] for name in ("B_sup", "B_prime_trice", "B_trice"))
        records.append(record)
        if diagnostics_fn is not None:
            diagnostics_fn(
                {
                    "schema_version": 1,
                    "method_family": "trice",
                    "answer_target_termination": (
                        "eos" if reward_requires_eos else "none"
                    ),
                    "round": round_index,
                    "completed_rounds": round_index + 1,
                    "trice": diagnostics,
                    "generation": {
                        "prior_generations_cumulative": total_prior_generated,
                        "guide_generations_cumulative": total_guide_generated,
                        "llm_generations_cumulative": (
                            total_prior_generated + total_guide_generated
                        ),
                        "generated_tokens_cumulative": total_generated_tokens,
                        "questions_this_round": len(selected_ids),
                        "unique_questions_seen": len(unique_questions_seen),
                        "natural_eos_fraction": natural_eos_fraction,
                        "accepted_state_eos_fraction": (
                            accepted_state_eos_fraction
                        ),
                    },
                    "reward": {
                        "requires_natural_eos": bool(reward_requires_eos),
                    },
                    "optimizer": {
                        "gradient_steps_cumulative": total_steps,
                        "backward_tokens_cumulative": total_backward_tokens,
                        "backward_eos_tokens_cumulative": (
                            total_backward_eos_tokens
                        ),
                    },
                    "objective": {
                        name: record[name]
                        for name in ("B_sup", "B_prime_trice", "B_trice", "F")
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
    "FrozenTriceProposalBatch",
    "TRICE_ESTIMATORS",
    "TorchParameterVersion",
    "TriceChainState",
    "TriceEstimatorPlan",
    "TriceProposal",
    "TriceScoreTerm",
    "TriceTrace",
    "backward_estimator_plan",
    "TriceTransition",
    "advance_chains",
    "build_estimator_plan",
    "chain_memory_diagnostics",
    "collect_frozen_proposals",
    "evaluate_estimator_plan",
    "independence_step",
    "initialize_chain",
    "leave_one_out_acceptance_scales",
    "macrocycle_diagnostics",
    "run_ac_alg1_trice",
]
