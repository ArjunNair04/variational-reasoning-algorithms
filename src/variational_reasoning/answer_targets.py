"""Shared answer-target termination semantics for supervised LM factors."""

from __future__ import annotations


ANSWER_TARGET_TERMINATIONS = {"none", "eos"}


def terminated_answer_ids(tok, answer_ids, *, termination: str) -> list[int]:
    """Return answer-factor token ids with optional explicit EOS supervision."""

    if termination not in ANSWER_TARGET_TERMINATIONS:
        raise ValueError(
            f"unknown answer target termination {termination!r}; expected one of "
            f"{sorted(ANSWER_TARGET_TERMINATIONS)}"
        )
    ids = [int(token) for token in answer_ids]
    if termination == "none":
        return ids
    eos_token_id = getattr(tok, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("answer_target_termination='eos' requires tokenizer.eos_token_id")
    if ids and ids[-1] == int(eos_token_id):
        return ids
    return [*ids, int(eos_token_id)]
