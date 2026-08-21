"""Data-selection rules shared by the self-training baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence


@dataclass(frozen=True)
class Candidate:
    question_id: Hashable
    completion: object
    correct: bool
    has_eos: bool = True
    source: str = "sampled"


def select_correct(
    candidates: Sequence[Candidate],
    per_question_limit: int,
) -> tuple[Candidate, ...]:
    """Keep correct, naturally terminated samples up to a per-question cap."""

    if isinstance(per_question_limit, bool) or per_question_limit < 1:
        raise ValueError("per_question_limit must be a positive integer")
    counts: dict[Hashable, int] = {}
    selected = []
    for candidate in candidates:
        used = counts.get(candidate.question_id, 0)
        if not candidate.correct or not candidate.has_eos or used >= per_question_limit:
            continue
        selected.append(candidate)
        counts[candidate.question_id] = used + 1
    return tuple(selected)


def star_examples(
    direct: Sequence[Candidate],
    rationalized: Sequence[Candidate],
) -> tuple[Candidate, ...]:
    """Choose a direct STaR trace, or a valid answer-rationalized fallback.

    The full trainer removes the answer hint and reanchors every selected
    completion under the ordinary question prompt before maximum-likelihood
    training.
    """

    direct_by_question = {}
    for candidate in direct:
        if candidate.question_id in direct_by_question:
            raise ValueError("STaR expects one direct draw per question")
        direct_by_question[candidate.question_id] = candidate
    rationalized_by_question = {}
    for candidate in rationalized:
        if candidate.question_id in rationalized_by_question:
            raise ValueError("STaR expects at most one rationalized draw per question")
        rationalized_by_question[candidate.question_id] = candidate

    selected = []
    for question_id, candidate in direct_by_question.items():
        if candidate.correct and candidate.has_eos:
            selected.append(candidate)
            continue
        fallback = rationalized_by_question.get(question_id)
        if fallback is not None and fallback.correct and fallback.has_eos:
            selected.append(fallback)
    return tuple(selected)
