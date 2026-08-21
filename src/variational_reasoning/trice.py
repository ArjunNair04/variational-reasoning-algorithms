"""Persistent-chain and control-variate parts of TRICE."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Hashable


@dataclass(frozen=True)
class Chain:
    question_id: Hashable
    trace: object | None = None
    trace_id: Hashable | None = None
    correct: bool = False


@dataclass(frozen=True)
class Proposal:
    question_id: Hashable
    trace: object
    trace_id: Hashable
    correct: bool


@dataclass(frozen=True)
class Transition:
    before: Chain
    proposal: Proposal
    after: Chain
    accepted: bool


@dataclass(frozen=True)
class ScoreTerm:
    question_id: Hashable
    trace: object
    trace_id: Hashable
    coefficient: float
    role: str


def trice_step(chain: Chain, proposal: Proposal) -> tuple[Chain, Transition]:
    """Accept a correct proposal; otherwise retain the persistent state."""

    if chain.question_id != proposal.question_id:
        raise ValueError("proposal and chain must belong to the same question")
    accepted = bool(proposal.correct)
    after = (
        replace(
            chain,
            trace=proposal.trace,
            trace_id=proposal.trace_id,
            correct=True,
        )
        if accepted
        else chain
    )
    return after, Transition(chain, proposal, after, accepted)


def control_variate_terms(
    transitions: list[Transition] | tuple[Transition, ...],
) -> tuple[list[ScoreTerm], dict[Hashable, float]]:
    """Build the detached TRICE score terms and leave-one-out beta values."""

    question_ids = [transition.after.question_id for transition in transitions]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("transitions must contain distinct question ids")

    valid = [int(transition.after.correct) for transition in transitions]
    accepted_valid = [
        int(transition.after.correct and transition.proposal.correct)
        for transition in transitions
    ]
    total_valid = sum(valid)
    total_accepted = sum(accepted_valid)
    beta = {}
    for transition, own_valid, own_accepted in zip(
        transitions, valid, accepted_valid
    ):
        denominator = total_valid - own_valid
        numerator = total_accepted - own_accepted
        beta[transition.after.question_id] = (
            numerator / denominator if denominator > 0 else 0.0
        )

    if total_valid == 0:
        return [], beta

    normalizer = 1.0 / total_valid
    terms = []
    for transition in transitions:
        state = transition.after
        if not state.correct:
            continue
        if state.trace is None or state.trace_id is None:
            raise ValueError("a valid chain must contain a trace")
        terms.append(
            ScoreTerm(
                state.question_id,
                state.trace,
                state.trace_id,
                normalizer,
                "accepted_state",
            )
        )
        coefficient = -beta[state.question_id] * normalizer
        if coefficient != 0:
            role = (
                "accepted_proposal_control"
                if transition.proposal.correct
                else "rejected_proposal_control"
            )
            terms.append(
                ScoreTerm(
                    state.question_id,
                    transition.proposal.trace,
                    transition.proposal.trace_id,
                    coefficient,
                    role,
                )
            )
    return terms, beta
