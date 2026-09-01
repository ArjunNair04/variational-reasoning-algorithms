"""Free-decoding verifier evidence for AC-ALG1 responsibilities.

The verifier estimates whether a stored rationale makes the known answer
recoverable under either the moving adapter or the frozen pretrained model.
It is deliberately separate from proposal filtering: every candidate remains
in the finite support and the detached rollout value changes only its E-step
responsibility.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
import torch

from common import DEV


VERIFIER_POLICIES = ("current", "frozen_base")


@dataclass(frozen=True)
class VerifierTraceScore:
    """Detached Monte Carlo value estimate for one stored trace."""

    successes: int
    trials: int
    raw_rate: float
    smoothed_value: float
    log_value: float
    generated_tokens: int
    outputs: tuple[str, ...]
    correct: tuple[bool, ...]


def smoothed_verifier_value(
    successes: int,
    trials: int,
    alpha: float,
) -> float:
    """Return a symmetric Beta(alpha, alpha) posterior-mean value."""

    if trials < 1:
        raise ValueError(f"verifier trials must be positive, got {trials}")
    if successes < 0 or successes > trials:
        raise ValueError(
            "verifier successes must lie in [0, trials], got "
            f"{successes}/{trials}"
        )
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            "verifier smoothing alpha must be finite and positive, got "
            f"{alpha}"
        )
    return (successes + alpha) / (trials + 2.0 * alpha)


def verifier_log_values(
    successes: Iterable[int],
    *,
    trials: int,
    alpha: float,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return finite detached log values for a vector of trace outcomes."""

    values = [
        smoothed_verifier_value(int(success), int(trials), float(alpha))
        for success in successes
    ]
    return torch.tensor(
        [math.log(value) for value in values],
        dtype=dtype,
        device=device,
    ).detach()


def verifier_posterior_logits(
    variational_estimator: str,
    trace_logprobs: torch.Tensor,
    verifier_logs: torch.Tensor,
) -> torch.Tensor:
    """Return logits for the delta-set or current-prior IS posterior.

    For a finite delta support the target is proportional to
    ``p_theta(h | q) * v(h, q)``. When traces are current-prior draws, the
    proposal density cancels and the empirical importance weight is ``v``.
    """

    if trace_logprobs.shape != verifier_logs.shape:
        raise ValueError("trace log probabilities and verifier values must align")
    if variational_estimator == "delta_joint":
        return trace_logprobs + verifier_logs
    if variational_estimator == "prior_importance":
        return verifier_logs
    raise ValueError(
        "verifier posterior supports only delta_joint or prior_importance, got "
        f"{variational_estimator!r}"
    )


def _adapter_context(model, policy: str):
    if policy not in VERIFIER_POLICIES:
        raise ValueError(f"unknown verifier policy {policy!r}")
    if policy == "current":
        return nullcontext()
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        raise ValueError(
            "frozen_base verifier requires a model with disable_adapter()"
        )
    return disable_adapter()


def _answer_prefix(row) -> torch.Tensor:
    positions = row.ans.nonzero()
    if len(positions) == 0:
        raise ValueError("verifier trace has no answer span")
    stop = int(positions[0].item())
    if stop < 1:
        raise ValueError("verifier trace has an empty q+h prefix")
    return row.ids[:stop].detach().cpu()


def _trace_text(tok, row) -> str:
    trace_ids = row.ids[row.span & ~row.ans].detach().cpu().tolist()
    return tok.decode(trace_ids)


def _cuda_fork_devices() -> list[int]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.current_device()]


@torch.no_grad()
def score_trace_continuations(
    model,
    tok,
    task,
    rows: list[Any],
    *,
    policy: str,
    repeats: int,
    temperature: float,
    max_new_tokens: int,
    batch_size: int,
    smoothing_alpha: float,
    generation_seed: int,
) -> tuple[list[VerifierTraceScore], dict[str, int | float | str]]:
    """Estimate p(a* | q,h) by independent free-decoding continuations.

    The model receives each exact stored ``q+h`` token prefix. Correctness is
    evaluated on ``h+continuation`` so few-shot answer markers in ``q`` cannot
    contaminate the strict terminal-answer parser.
    """

    if policy not in VERIFIER_POLICIES:
        raise ValueError(f"unknown verifier policy {policy!r}")
    for name, value in (
        ("repeats", repeats),
        ("max_new_tokens", max_new_tokens),
        ("batch_size", batch_size),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"verifier temperature must be finite and positive, got {temperature}"
        )
    smoothed_verifier_value(0, int(repeats), float(smoothing_alpha))
    if not rows:
        return [], {
            "policy": policy,
            "calls": 0,
            "generated_tokens": 0,
            "successes": 0,
            "traces": 0,
        }

    prefixes = [_answer_prefix(row) for row in rows]
    trace_texts = [_trace_text(tok, row) for row in rows]
    flat_indices = np.repeat(np.arange(len(rows), dtype=int), int(repeats))
    outputs: list[list[str]] = [[] for _ in rows]
    correct: list[list[bool]] = [[] for _ in rows]
    generated_tokens = [0 for _ in rows]
    eos = getattr(tok, "eos_token_id", None)
    if eos is None:
        raise ValueError("verifier tokenizer must define eos_token_id")
    pad_token_id = getattr(tok, "pad_token_id", None)
    pad = pad_token_id if pad_token_id is not None else eos
    floor = float(getattr(task, "floor", 0.0))
    was_training = bool(getattr(model, "training", False))

    try:
        model.eval()
        with torch.random.fork_rng(devices=_cuda_fork_devices()):
            torch.manual_seed(int(generation_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(generation_seed))
            with _adapter_context(model, policy):
                for start in range(0, len(flat_indices), int(batch_size)):
                    local_indices = flat_indices[start:start + int(batch_size)]
                    chunk = [prefixes[int(index)] for index in local_indices]
                    width = max(len(prefix) for prefix in chunk)
                    input_ids = torch.full(
                        (len(chunk), width),
                        pad,
                        dtype=torch.long,
                        device=DEV,
                    )
                    attention = torch.zeros_like(input_ids)
                    for row_index, prefix in enumerate(chunk):
                        input_ids[row_index, -len(prefix):] = prefix.to(DEV)
                        attention[row_index, -len(prefix):] = 1
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention,
                        do_sample=True,
                        temperature=float(temperature),
                        top_k=0,
                        top_p=1.0,
                        max_new_tokens=int(max_new_tokens),
                        pad_token_id=pad,
                    )
                    continuation = generated[:, width:]
                    texts = tok.batch_decode(
                        continuation,
                        skip_special_tokens=True,
                    )
                    completions = [
                        trace_texts[int(source_index)] + text
                        for source_index, text in zip(local_indices, texts)
                    ]
                    pids = [
                        int(rows[int(source_index)].pid)
                        for source_index in local_indices
                    ]
                    rewards = task.reward(completions, pids=pids)
                    for row_index, (source_index, text, reward) in enumerate(
                        zip(local_indices, texts, rewards)
                    ):
                        target = int(source_index)
                        outputs[target].append(text)
                        correct[target].append(
                            bool(float(reward) > floor + 0.5)
                        )
                        eos_positions = (
                            continuation[row_index] == eos
                        ).nonzero()
                        generated_tokens[target] += (
                            int(eos_positions[0].item()) + 1
                            if len(eos_positions)
                            else int(continuation.shape[1])
                        )
    finally:
        train = getattr(model, "train", None)
        if train is not None:
            train(was_training)

    scores = []
    for index in range(len(rows)):
        successes = int(sum(correct[index]))
        value = smoothed_verifier_value(
            successes,
            int(repeats),
            float(smoothing_alpha),
        )
        scores.append(
            VerifierTraceScore(
                successes=successes,
                trials=int(repeats),
                raw_rate=successes / int(repeats),
                smoothed_value=value,
                log_value=math.log(value),
                generated_tokens=int(generated_tokens[index]),
                outputs=tuple(outputs[index]),
                correct=tuple(correct[index]),
            )
        )

    return scores, {
        "policy": policy,
        "calls": len(rows) * int(repeats),
        "generated_tokens": sum(generated_tokens),
        "successes": sum(score.successes for score in scores),
        "traces": len(rows),
    }
