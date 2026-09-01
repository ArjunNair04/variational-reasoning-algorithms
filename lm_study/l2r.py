"""Scalable answer-conditioned Learning-to-Reason (L2R) trainer.

The latent-reasoning factorisation is

    p(a* | q) = sum_h p_reader(a* | q, h) p_generator(h | q).

This module isolates the two factors. The LoRA policy proposes and learns only
reasoning tokens. The answer reader can be frozen at the pretrained base model,
so optimisation cannot improve its own E-step reward by changing the answer
head. One official gold rationale may be retained in each per-question archive.

The implementation is intended for scaling studies:

* question breadth and traces per question are explicit through ``B`` and ``G``;
* optional adaptive sampling spends extra traces only on unresolved questions;
* bounded per-question archives avoid quadratic growth over long horizons;
* compact replay preserves gold, high-responsibility, and recent traces;
* every M-step uses the complete detached weighted replay support;
* rollout, generated-token, backward-token, buffer, and posterior diagnostics
  are emitted for matched-compute comparisons with GRPO.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass, replace
import math
import os
import random
import re
import time
from typing import Any, Iterable

import numpy as np
import torch

from prompting import PROPOSAL_PROMPTS, build_proposal_prompt

from answer_events import (
    ANSWER_EVENT_MODES,
    parse_gsm8k_answer_event,
)
from answer_targets import ANSWER_TARGET_TERMINATIONS, terminated_answer_ids
from ac_alg1_diagnostics import (
    finite_or_none,
    optimizer_moment_diagnostics,
    parameter_delta_norm,
    run_diagnostic_probe,
    tensor_list_cosine,
    tensor_list_norm,
    validate_diagnostic_level,
)
from common import (
    DEV,
    MODEL_NAME,
    audit_chat_active_completions,
    encode_task_prompt,
    has_answer_marker,
    load_model,
    maybe_eval,
    sample_multi,
    seq_logprobs,
    task_pad_token_id,
    task_format_rate,
    token_logps,
)
from l2r_structural import (
    GRADIENT_PROJECTION_MODES,
    LORA_TRAINABLE_MODES,
    QUESTION_SCHEDULES,
    GradientProjector,
    QuestionScheduler,
    configure_lora_trainable,
    replicated_responsibilities,
)
from trainer_config import L2RRunConfig


RESPONSIBILITY_SCORES = {
    "joint",
    "token_mean",
    "prior_corrected",
    "mixed_prior_corrected",
}
READER_MODES = {"moving", "frozen"}
L2R_PROPOSAL_MIXTURES = {"single", "question_answer", "question_temperature"}
TRACE_SEGMENTATION_MODES = {"legacy", "validated"}
RESPONSIBILITY_PROJECTIONS = {"none", "safe_set"}
POLICY_ANCHOR_MODES = {"fixed", "grad_ratio"}
L2R_BUFFER_SEMANTICS = {"set_archive", "fresh_multiset"}
MSTEP_OBJECTIVES = {"generator", "joint"}
POLICY_ANCHOR_SCOPES = {"generator", "generator_and_reader"}

_TERMINAL_ANSWER_PATTERNS = (
    (
        "marker",
        re.compile(
            r"(?is)^(?P<reason>.*?)(?:\s*####\s*)"
            r"(?P<answer>[-+]?(?:\d[\d,]*)(?:\.\d+)?)\s*$"
        ),
    ),
    (
        "answer_tag",
        re.compile(
            r"(?is)^(?P<reason>.*?)(?:\s*<answer>\s*)"
            r"(?P<answer>[-+]?(?:\d[\d,]*)(?:\.\d+)?)\s*</answer>\s*$"
        ),
    ),
    (
        "answer_phrase",
        re.compile(
            r"(?is)^(?P<reason>.*?)(?:\s*the\s+answer\s+is\s*)"
            r"(?P<answer>[-+]?(?:\d[\d,]*)(?:\.\d+)?)"
            r"(?:\s*[.!])?\s*$"
        ),
    ),
)


@dataclass
class _L2RRuntimeState:
    model: Any
    tok: Any
    named_trainable: list[tuple[str, torch.nn.Parameter]]
    opt: Any
    gradient_projector: GradientProjector
    rng: np.random.Generator
    prompt_ids: list[torch.Tensor]
    training_pids: list[int]
    safety_pids: list[int]
    candidate_utility_pids: list[int]
    training_question_count: int
    safety_rows: list["L2RTrace"]
    candidate_utility_rows: list["L2RTrace"]
    candidate_utility_probe_fn: Any
    buffers: dict[int, list["L2RTrace"]]
    seen_questions: set[int]
    records: list[dict]
    total_gen: int
    total_generated_tokens: int
    total_backward_tokens: int
    total_steps: int
    total_policy_backward_tokens: int
    total_anchor_backward_tokens: int
    total_reader_decode_tokens: int
    total_evictions: int
    total_duplicates: int
    total_reader_decode: int
    total_current_policy_backward_tokens: int
    total_replay_policy_backward_tokens: int
    question_exposures: int
    scheduler: QuestionScheduler
    policy_anchor_state: dict[str, float]
    cache_stats: dict[str, int]
    training_diagnostic_state: dict[str, Any]


@dataclass
class _L2RRoundOutcome:
    cache_before: dict[str, int]
    pids: list[int]
    sampled_rows: list["L2RTrace"]
    sampled_texts: list[str]
    generation: dict[str, Any]
    schedule_diagnostics: dict[str, Any]
    gold_added: int
    sampled_added: int
    duplicate_rows: int
    evictions: int
    replay_pids: list[int]
    history_monitor_pids: list[int]
    history_rows: list["L2RTrace"]
    current_update_pids: list[int]
    update_pids: list[int]
    rows: list["L2RTrace"]
    ids: torch.Tensor
    h_mask: torch.Tensor
    a_mask: torch.Tensor
    answer_proposal_h: torch.Tensor | None
    policy_h: torch.Tensor | None = None
    reader_a: torch.Tensor | None = None
    trace_pids: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    counterfactual_invalid_mass: float | None = None
    decode_fallback_questions: int = 0
    projection: dict[str, Any] | None = None
    current_policy_backward_tokens: int = 0
    replay_policy_backward_tokens: int = 0
    segmentation: dict[str, Any] | None = None
    mstep: dict[str, Any] | None = None
    inner_step_diagnostics: list[dict] | None = None
    empirical_h_kl: float | None = None
    empirical_h_kl_nonnegative: float | None = None
    posterior: dict[str, Any] | None = None
    top_traces: Any = None
    schedule_after: dict[str, Any] | None = None
    test_acc: float | None = None
    proposal_correct: list[bool] | None = None
    format_diagnostics: dict[str, Any] | None = None
    archive_rows: int = 0
    round_cache: dict[str, int] | None = None
    record: dict[str, Any] | None = None


@dataclass
class L2RTrace:
    """One question-anchored latent rationale followed by a teacher-forced answer."""

    ids: torch.Tensor
    h_mask: torch.Tensor
    a_mask: torch.Tensor
    pid: int
    round_added: int
    completion_key: tuple[int, ...]
    text: str
    replica: int = 0
    is_gold: bool = False
    proposal_correct: bool | None = None
    generated_tokens: int = 0
    proposal_prompt: str = "question"
    proposal_temperature: float = 1.0
    target_mentioned_before_final: bool | None = None
    target_before_equation: bool | None = None
    has_equation: bool | None = None
    last_responsibility: float = 0.0
    segmentation_mode: str = "legacy"
    segmentation_valid: bool = True
    segmentation_answer: str | None = None
    answer_event_mode: str = "legacy"
    answer_event_valid: bool = False
    answer_marker_count: int = 0
    answer_marker_terminal: bool = False
    frozen_reader_logp: float | None = None
    frozen_reader_decode_correct: bool | None = None
    frozen_reader_decode_tokens: int | None = None
    frozen_base_token_logps: torch.Tensor | None = None

    @property
    def h_tokens(self) -> int:
        return int(self.h_mask.sum())


_CACHE_COUNTERS = (
    "reader_score_hits",
    "reader_score_misses",
    "reader_decode_hits",
    "reader_decode_misses",
    "base_token_hits",
    "base_token_misses",
    "safety_nll_hits",
    "safety_nll_misses",
    "saved_forward_rows",
    "saved_forward_tokens",
)


def _reserved_question_partitions(
    question_count: int,
    *,
    safety_questions: int,
    utility_questions: int,
) -> tuple[list[int], list[int], list[int]]:
    """Return ordered optimization, safety, and utility prompt ids."""

    if min(question_count, safety_questions, utility_questions) < 0:
        raise ValueError("L2R partition sizes must be nonnegative")
    training_stop = question_count - safety_questions - utility_questions
    if training_stop < 0:
        raise ValueError(
            "safety and utility reserves exceed the available questions"
        )
    safety_stop = training_stop + safety_questions
    return (
        list(range(training_stop)),
        list(range(training_stop, safety_stop)),
        list(range(safety_stop, question_count)),
    )


def _new_cache_stats() -> dict[str, int]:
    return {key: 0 for key in _CACHE_COUNTERS}


def _cache_delta(
    cumulative: dict[str, int],
    before: dict[str, int],
) -> dict[str, int]:
    return {
        key: int(cumulative.get(key, 0) - before.get(key, 0))
        for key in _CACHE_COUNTERS
    }


def _trace_state(trace: L2RTrace) -> dict:
    return {
        "ids": trace.ids.detach().cpu(),
        "h_mask": trace.h_mask.detach().cpu(),
        "a_mask": trace.a_mask.detach().cpu(),
        "pid": trace.pid,
        "round_added": trace.round_added,
        "completion_key": tuple(trace.completion_key),
        "text": trace.text,
        "replica": trace.replica,
        "is_gold": trace.is_gold,
        "proposal_correct": trace.proposal_correct,
        "generated_tokens": trace.generated_tokens,
        "proposal_prompt": trace.proposal_prompt,
        "proposal_temperature": trace.proposal_temperature,
        "target_mentioned_before_final": trace.target_mentioned_before_final,
        "target_before_equation": trace.target_before_equation,
        "has_equation": trace.has_equation,
        "last_responsibility": trace.last_responsibility,
        "segmentation_mode": trace.segmentation_mode,
        "segmentation_valid": trace.segmentation_valid,
        "segmentation_answer": trace.segmentation_answer,
        "answer_event_mode": trace.answer_event_mode,
        "answer_event_valid": trace.answer_event_valid,
        "answer_marker_count": trace.answer_marker_count,
        "answer_marker_terminal": trace.answer_marker_terminal,
        "frozen_reader_logp": trace.frozen_reader_logp,
        "frozen_reader_decode_correct": trace.frozen_reader_decode_correct,
        "frozen_reader_decode_tokens": trace.frozen_reader_decode_tokens,
        "frozen_base_token_logps": (
            trace.frozen_base_token_logps.detach().cpu()
            if trace.frozen_base_token_logps is not None else None
        ),
    }


def _trace_from_state(payload: dict) -> L2RTrace:
    return L2RTrace(**payload)


def _trainable_parameter_state(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_parameters
    }


def _restore_trainable_parameter_state(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    payload: dict[str, torch.Tensor],
) -> None:
    expected = [name for name, _ in named_parameters]
    if set(payload) != set(expected):
        raise ValueError("resume state trainable-parameter layout does not match the model")
    with torch.no_grad():
        for name, parameter in named_parameters:
            value = payload[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"resume parameter shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _rng_state(rng: np.random.Generator) -> dict:
    return {
        "local_numpy": rng.bit_generator.state,
        "global_numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []
        ),
    }


def _restore_rng_state(rng: np.random.Generator, payload: dict) -> None:
    rng.bit_generator.state = payload["local_numpy"]
    np.random.set_state(payload["global_numpy"])
    random.setstate(payload["python"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and payload.get("torch_cuda"):
        states = payload["torch_cuda"]
        if len(states) != torch.cuda.device_count():
            raise ValueError("resume CUDA RNG state does not match visible device count")
        torch.cuda.set_rng_state_all(states)


def l2r_responsibilities(
    policy_h_logps: torch.Tensor,
    reader_a_logps: torch.Tensor,
    h_lengths: torch.Tensor,
    a_lengths: torch.Tensor,
    pids: torch.Tensor,
    *,
    score: str = "joint",
    temperature: float = 1.0,
    active: torch.Tensor | None = None,
    answer_proposal_h_logps: torch.Tensor | None = None,
    proposal_prior_fraction: float = 1.0,
) -> torch.Tensor:
    """Normalise detached L2R responsibilities independently per question."""

    if score not in RESPONSIBILITY_SCORES:
        raise ValueError(f"unknown L2R responsibility score {score!r}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    shape = policy_h_logps.shape
    if any(values.shape != shape for values in (reader_a_logps, h_lengths, a_lengths, pids)):
        raise ValueError("all L2R responsibility inputs must have the same shape")
    if active is None:
        active = torch.ones_like(policy_h_logps, dtype=torch.bool)
    if active.shape != shape:
        raise ValueError("active mask must align with responsibility inputs")
    if answer_proposal_h_logps is not None and (
        answer_proposal_h_logps.shape != shape
    ):
        raise ValueError("answer proposal scores must align with responsibility inputs")

    if score == "token_mean":
        logits = (
            policy_h_logps / h_lengths.clamp_min(1).to(policy_h_logps.dtype)
            + reader_a_logps / a_lengths.clamp_min(1).to(reader_a_logps.dtype)
        )
    elif score == "prior_corrected":
        logits = reader_a_logps
    elif score == "mixed_prior_corrected":
        if answer_proposal_h_logps is None:
            raise ValueError(
                "mixed proposal correction requires answer proposal log densities"
            )
        if not math.isfinite(proposal_prior_fraction) or not (
            0 < proposal_prior_fraction < 1
        ):
            raise ValueError(
                "mixed proposal correction requires proposal_prior_fraction in (0, 1)"
            )
        mixture_logps = torch.logaddexp(
            policy_h_logps + math.log(proposal_prior_fraction),
            answer_proposal_h_logps
            + math.log1p(-proposal_prior_fraction),
        )
        logits = policy_h_logps + reader_a_logps - mixture_logps
    else:
        logits = policy_h_logps + reader_a_logps

    weights = torch.zeros_like(logits)
    for pid in torch.unique(pids, sorted=True):
        local = (pids == pid) & active
        if bool(local.any()):
            weights[local] = torch.softmax(logits[local] / temperature, dim=0)
    return weights


def _l2r_objective_mask(
    h_mask: torch.Tensor,
    a_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode not in MSTEP_OBJECTIVES:
        raise ValueError(f"unknown L2R M-step objective {mode!r}")
    return h_mask if mode == "generator" else h_mask | a_mask


def _l2r_anchor_mask(
    h_mask: torch.Tensor,
    a_mask: torch.Tensor,
    scope: str,
) -> torch.Tensor:
    if scope not in POLICY_ANCHOR_SCOPES:
        raise ValueError(f"unknown L2R policy anchor scope {scope!r}")
    return h_mask if scope == "generator" else h_mask | a_mask


def _trace_objective_tokens(trace: L2RTrace, mode: str) -> int:
    return int(
        _l2r_objective_mask(trace.h_mask, trace.a_mask, mode).sum()
    )


def project_l2r_responsibilities(
    weights: torch.Tensor,
    pids: torch.Tensor,
    *,
    active: torch.Tensor | None = None,
    ess_floor: float = 0.0,
    max_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Interpolate each question posterior into a registered diffuse safe set.

    This is not the unrestricted KL projection onto the feasible set. It finds
    the smallest coefficient on the line segment between the raw posterior and
    the uniform distribution that satisfies both constraints. The restricted
    operator is deterministic, preserves rank, and leaves already-safe
    posteriors unchanged.
    """

    if weights.shape != pids.shape:
        raise ValueError("weights and pids must have the same shape")
    if active is None:
        active = weights > 0
    if active.shape != weights.shape:
        raise ValueError("active mask must align with responsibility weights")
    if not math.isfinite(ess_floor) or not 0 <= ess_floor <= 1:
        raise ValueError("ess_floor must be finite and in [0, 1]")
    if not math.isfinite(max_weight) or not 0 < max_weight <= 1:
        raise ValueError("max_weight must be finite and in (0, 1]")

    projected = torch.zeros_like(weights)
    alphas = []
    changed = 0
    for pid in torch.unique(pids, sorted=True):
        local = (pids == pid) & active
        count = int(local.sum())
        if count == 0:
            continue
        initial = weights[local]
        initial = initial / initial.sum().clamp_min(
            torch.finfo(initial.dtype).tiny
        )
        uniform = torch.full_like(initial, 1.0 / count)
        effective_cap = max(max_weight, 1.0 / count)

        def safe(alpha: float) -> bool:
            candidate = (1.0 - alpha) * initial + alpha * uniform
            ess_fraction = 1.0 / (
                count * float(torch.square(candidate).sum())
            )
            return (
                float(candidate.max()) <= effective_cap + 1e-8
                and ess_fraction + 1e-8 >= ess_floor
            )

        if safe(0.0):
            alpha = 0.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(48):
                mid = (lo + hi) / 2.0
                if safe(mid):
                    hi = mid
                else:
                    lo = mid
            alpha = hi
            changed += 1
        projected[local] = (1.0 - alpha) * initial + alpha * uniform
        alphas.append(alpha)

    positive = projected[projected > 0]
    return projected, {
        "question_count": len(alphas),
        "changed_questions": changed,
        "changed_fraction": changed / len(alphas) if alphas else None,
        "mean_uniform_mix": float(np.mean(alphas)) if alphas else None,
        "max_uniform_mix": max(alphas, default=None),
        "global_max_weight": float(positive.max()) if len(positive) else None,
    }


def _counterfactual_invalid_mass(
    logits: torch.Tensor,
    pids: torch.Tensor,
    rows: list[L2RTrace],
    *,
    temperature: float,
) -> float | None:
    """Mass invalid traces would receive before fail-closed segmentation."""

    if not rows:
        return None
    invalid = torch.tensor(
        [not row.segmentation_valid for row in rows],
        device=logits.device,
        dtype=torch.bool,
    )
    masses = []
    for pid in torch.unique(pids, sorted=True):
        local = pids == pid
        local_weights = torch.softmax(logits[local] / temperature, dim=0)
        masses.append(float(local_weights[invalid[local]].sum()))
    return float(np.mean(masses)) if masses else None


def select_replay_indices(
    traces: list[L2RTrace],
    logits: torch.Tensor,
    replay_limit: int,
) -> list[int]:
    """Select gold, elite, and recent traces within one question archive."""

    if replay_limit <= 0 or len(traces) <= replay_limit:
        return list(range(len(traces)))
    gold = [index for index, trace in enumerate(traces) if trace.is_gold]
    if len(gold) >= replay_limit:
        return gold[:replay_limit]

    sampled = [index for index, trace in enumerate(traces) if not trace.is_gold]
    slots = replay_limit - len(gold)
    recent_slots = (slots + 1) // 2
    recent = sorted(
        sampled,
        key=lambda index: (traces[index].round_added, index),
        reverse=True,
    )[:recent_slots]
    recent_set = set(recent)
    elite = sorted(
        (index for index in sampled if index not in recent_set),
        key=lambda index: (float(logits[index]), traces[index].round_added, index),
        reverse=True,
    )[: slots - len(recent)]
    return sorted(gold + recent + elite)


def _reason_cut(text: str) -> int:
    cut = len(text)
    for marker in (
        "####",
        "</think>",
        "<answer>",
        "\nQuestion:",
        "\n\nQuestion:",
        "\nQ:",
        "\nUser:",
    ):
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    return cut


def _reasoning_token_prefix(tok, completion_ids: list[int], text: str) -> list[int]:
    if completion_ids and completion_ids[-1] == tok.eos_token_id:
        completion_ids = completion_ids[:-1]
    cut = _reason_cut(text)
    if cut >= len(text):
        return completion_ids
    lo, hi = 0, len(completion_ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok.decode(completion_ids[:mid])) <= cut:
            lo = mid
        else:
            hi = mid - 1
    return completion_ids[:lo]


def _proposal_prompt(base_prompt: str, answer, mode: str) -> str:
    """Build a proposal prompt without changing the question-anchored scorer."""
    return build_proposal_prompt(base_prompt, answer, mode)


def _task_parse_answer_event(task, text: str, *, mode: str):
    parser = getattr(task, "parse_answer_event", parse_gsm8k_answer_event)
    return parser(text, mode=mode)


def _task_answers_equivalent(task, left, right) -> bool:
    comparator = getattr(task, "answers_equivalent", None)
    if comparator is not None:
        return bool(comparator(left, right))
    return left is not None and left == right


_NUMBER_RE = re.compile(r"(?<![\d,])-?\d[\d,]*(?![\d,])")
_EQUATION_RE = re.compile(
    r"-?\d[\d,]*(?:\.\d+)?\s*(?:[+\-*/=×÷]|[xX](?=\s*\d))"
)


def _proposal_text_diagnostics(text: str, answer) -> dict[str, bool]:
    """Measure answer leakage and arithmetic support without affecting training."""

    final_marker = text.find("####")
    reasoning = text if final_marker < 0 else text[:final_marker]
    try:
        target = int(answer) if answer is not None else None
    except (TypeError, ValueError):
        target = None
    target_positions = []
    if target is not None:
        for match in _NUMBER_RE.finditer(reasoning):
            try:
                value = int(match.group(0).replace(",", ""))
            except ValueError:
                continue
            if value == target:
                target_positions.append(match.start())
    equation = _EQUATION_RE.search(reasoning)
    first_target = min(target_positions) if target_positions else None
    return {
        "target_mentioned_before_final": first_target is not None,
        "target_before_equation": (
            first_target is not None
            and (equation is None or first_target < equation.start())
        ),
        "has_equation": equation is not None,
    }


def _validated_reasoning_text(text: str) -> tuple[str, bool, str | None, str | None]:
    """Split one completion only at a unique, strictly terminal answer form."""

    matches = []
    for mode, pattern in _TERMINAL_ANSWER_PATTERNS:
        match = pattern.fullmatch(text)
        if match is not None:
            matches.append(
                (
                    match.group("reason").rstrip(),
                    match.group("answer").replace(",", ""),
                    mode,
                )
            )
    if len(matches) != 1:
        return "", False, None, "ambiguous" if matches else "unparsed"
    reasoning, answer, mode = matches[0]
    if not reasoning.strip():
        return "", False, answer, "empty_reasoning"
    return reasoning, True, answer, mode


def _prefix_ids_for_text(tok, completion_ids: list[int], prefix: str) -> list[int]:
    """Return the largest decoded token prefix contained in ``prefix``."""

    lo, hi = 0, len(completion_ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok.decode(completion_ids[:mid])) <= len(prefix):
            lo = mid
        else:
            hi = mid - 1
    return completion_ids[:lo]


def _prefix_ids_through_marker(
    tok,
    completion_ids: list[int],
    text: str,
    marker_end: int,
) -> tuple[list[int], bool]:
    """Keep sampled tokens through ``####`` only when the boundary is token-aligned."""

    target = text[:marker_end]
    for stop in range(1, len(completion_ids) + 1):
        decoded = tok.decode(completion_ids[:stop])
        if len(decoded) < marker_end:
            continue
        if decoded[:marker_end] != target:
            return [], False
        if decoded[marker_end:].strip():
            return [], False
        return completion_ids[:stop], True
    return [], False


def _extract_reasoning_tokens(
    tok,
    completion_ids: list[int],
    text: str,
    *,
    mode: str,
    answer_event_mode: str = "legacy",
    parse_answer_event=parse_gsm8k_answer_event,
) -> tuple[list[int], bool, str | None, str]:
    if mode not in TRACE_SEGMENTATION_MODES:
        raise ValueError(f"unknown trace segmentation mode {mode!r}")
    if answer_event_mode not in ANSWER_EVENT_MODES:
        raise ValueError(f"unknown answer event mode {answer_event_mode!r}")
    raw_ids = list(completion_ids)
    if raw_ids and raw_ids[-1] == tok.eos_token_id:
        raw_ids = raw_ids[:-1]
    if answer_event_mode == "strict_terminal_marker":
        event = parse_answer_event(
            text,
            mode="strict_terminal_marker",
        )
        if event.marker_end is None:
            return [], False, None, event.parse_mode
        if not event.reasoning.strip():
            return [], False, str(event.answer), "empty_reasoning"
        h_ids, aligned = _prefix_ids_through_marker(
            tok,
            raw_ids,
            text,
            int(event.marker_end),
        )
        if not aligned:
            return [], False, str(event.answer), "marker_not_token_aligned"
        return (
            h_ids,
            True,
            str(event.answer) if event.answer is not None else None,
            event.parse_mode,
        )
    if mode == "legacy":
        return _reasoning_token_prefix(tok, raw_ids, text), True, None, "legacy"
    reasoning, valid, answer, parse_mode = _validated_reasoning_text(text)
    if not valid:
        return [], False, answer, str(parse_mode)
    return (
        _prefix_ids_for_text(tok, raw_ids, reasoning),
        True,
        answer,
        str(parse_mode),
    )


def _validate_strict_answer_event_tokenization(tok) -> None:
    """Fail before training when ``####`` cannot form an exact token boundary."""

    text = "Reasoning\n#### 42"
    completion_ids = tok(
        text,
        add_special_tokens=False,
    ).input_ids
    h_ids, valid, _answer, _mode = _extract_reasoning_tokens(
        tok,
        completion_ids,
        text,
        mode="validated",
        answer_event_mode="strict_terminal_marker",
    )
    if not valid:
        raise ValueError(
            "tokenizer cannot represent the strict L2R #### boundary exactly"
        )
    decoded_h = tok.decode(h_ids)
    separator = "" if decoded_h.endswith((" ", "\n", "\t")) else " "
    answer_ids = tok(
        f"{separator}42",
        add_special_tokens=False,
    ).input_ids
    reconstructed = tok.decode(h_ids + answer_ids)
    event = parse_gsm8k_answer_event(
        reconstructed,
        mode="strict_terminal_marker",
    )
    if not event.strict_valid or event.answer != 42:
        raise ValueError(
            "tokenizer cannot reconstruct the strict L2R #### answer event"
        )


def _make_trace(
    tok,
    prompt_ids: torch.Tensor,
    h_ids: list[int],
    answer,
    *,
    pid: int,
    round_added: int,
    text: str,
    is_gold: bool,
    proposal_correct: bool | None,
    generated_tokens: int,
    replica: int = 0,
    completion_key: tuple[int, ...] | None = None,
    segmentation_mode: str = "legacy",
    segmentation_valid: bool = True,
    segmentation_answer: str | None = None,
    answer_event_mode: str = "legacy",
    answer_event_valid: bool = False,
    answer_marker_count: int = 0,
    answer_marker_terminal: bool = False,
    answer_target_termination: str = "none",
    proposal_prompt: str = "question",
    proposal_temperature: float = 1.0,
    target_mentioned_before_final: bool | None = None,
    target_before_equation: bool | None = None,
    has_equation: bool | None = None,
) -> L2RTrace:
    if answer_event_mode == "strict_terminal_marker":
        decoded_h = tok.decode(h_ids)
        separator = "" if decoded_h.endswith((" ", "\n", "\t")) else " "
        a_ids = tok(
            f"{separator}{answer}",
            add_special_tokens=False,
        ).input_ids
    else:
        a_ids = tok(f"\n#### {answer}", add_special_tokens=False).input_ids
    a_ids = terminated_answer_ids(
        tok,
        a_ids,
        termination=answer_target_termination,
    )
    full = torch.tensor(
        prompt_ids.tolist() + h_ids + a_ids,
        dtype=torch.long,
    )
    h_mask = torch.zeros(len(full), dtype=torch.bool)
    h_mask[len(prompt_ids):len(prompt_ids) + len(h_ids)] = True
    a_mask = torch.zeros(len(full), dtype=torch.bool)
    a_mask[len(prompt_ids) + len(h_ids):] = True
    return L2RTrace(
        ids=full,
        h_mask=h_mask,
        a_mask=a_mask,
        pid=int(pid),
        round_added=int(round_added),
        completion_key=(
            completion_key
            if completion_key is not None
            else tuple(int(token) for token in h_ids)
        ),
        text=text,
        replica=int(replica),
        is_gold=bool(is_gold),
        proposal_correct=proposal_correct,
        generated_tokens=int(generated_tokens),
        proposal_prompt=proposal_prompt,
        proposal_temperature=float(proposal_temperature),
        target_mentioned_before_final=target_mentioned_before_final,
        target_before_equation=target_before_equation,
        has_equation=has_equation,
        segmentation_mode=segmentation_mode,
        segmentation_valid=bool(segmentation_valid),
        segmentation_answer=segmentation_answer,
        answer_event_mode=answer_event_mode,
        answer_event_valid=bool(answer_event_valid),
        answer_marker_count=int(answer_marker_count),
        answer_marker_terminal=bool(answer_marker_terminal),
    )


def _gold_trace(
    tok,
    task,
    prompt_ids: torch.Tensor,
    pid: int,
    round_added: int,
    *,
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
) -> L2RTrace | None:
    if not hasattr(task, "gold_solution") or task.gold_answer[pid] is None:
        return None
    solution = task.gold_solution[pid]
    if not solution:
        return None
    reasoning = solution.split("####", 1)[0].rstrip()
    h_text = (
        " " + reasoning + "\n####"
        if answer_event_mode == "strict_terminal_marker"
        else " " + reasoning
    )
    h_ids = tok(h_text, add_special_tokens=False).input_ids
    strict_event = answer_event_mode == "strict_terminal_marker"
    text = (
        f"{reasoning}\n#### {task.gold_answer[pid]}"
        if strict_event else reasoning
    )
    return _make_trace(
        tok,
        prompt_ids,
        h_ids,
        task.gold_answer[pid],
        pid=pid,
        round_added=round_added,
        text=text,
        is_gold=True,
        proposal_correct=True,
        generated_tokens=0,
        replica=-1,
        answer_event_mode=answer_event_mode,
        answer_event_valid=strict_event,
        answer_marker_count=int(strict_event),
        answer_marker_terminal=strict_event,
        answer_target_termination=answer_target_termination,
    )


def _gold_answer_trace(
    tok,
    task,
    prompt_ids: torch.Tensor,
    pid: int,
    *,
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
) -> L2RTrace | None:
    """Teacher-force only the canonical final answer after the question."""

    if not hasattr(task, "gold_answer") or task.gold_answer[pid] is None:
        return None
    h_ids = (
        tok("\n####", add_special_tokens=False).input_ids
        if answer_event_mode == "strict_terminal_marker"
        else []
    )
    strict_event = answer_event_mode == "strict_terminal_marker"
    return _make_trace(
        tok,
        prompt_ids,
        h_ids,
        task.gold_answer[pid],
        pid=pid,
        round_added=-1,
        text=f"#### {task.gold_answer[pid]}" if strict_event else "",
        is_gold=True,
        proposal_correct=True,
        generated_tokens=0,
        replica=-1,
        answer_event_mode=answer_event_mode,
        answer_event_valid=strict_event,
        answer_marker_count=int(strict_event),
        answer_marker_terminal=strict_event,
        answer_target_termination=answer_target_termination,
    )


def _append_unique(buffers: dict[int, list[L2RTrace]], trace: L2RTrace) -> bool:
    rows = buffers.setdefault(trace.pid, [])
    if any(row.completion_key == trace.completion_key for row in rows):
        return False
    rows.append(trace)
    return True


def _fresh_multiset_buffers(
    pids: Iterable[int],
    sampled_rows: Iterable[L2RTrace],
) -> dict[int, list[L2RTrace]]:
    """Retain every current-round draw and discard stale proposal support."""

    buffers = {int(pid): [] for pid in pids}
    for row in sampled_rows:
        if row.pid not in buffers:
            raise ValueError(
                f"sampled trace pid {row.pid} is outside the selected questions"
            )
        buffers[row.pid].append(row)
    return buffers


def _prune_archive(
    rows: list[L2RTrace],
    archive_limit: int,
    buffer_replicates: int = 1,
) -> int:
    """Bound one archive while preserving gold, prior elites, and recent traces."""

    if archive_limit <= 0 or len(rows) <= archive_limit:
        return 0
    gold = [row for row in rows if row.is_gold]
    sampled = [row for row in rows if not row.is_gold]
    slots = max(archive_limit - len(gold), 0)
    if buffer_replicates > 1:
        keep_rows = list(gold)
        base, remainder = divmod(slots, buffer_replicates)
        for replica in range(buffer_replicates):
            local = [row for row in sampled if row.replica == replica]
            local_slots = base + int(replica < remainder)
            recent_slots = (local_slots + 1) // 2
            recent = sorted(
                local,
                key=lambda row: row.round_added,
                reverse=True,
            )[:recent_slots]
            recent_ids = {id(row) for row in recent}
            elite = sorted(
                (row for row in local if id(row) not in recent_ids),
                key=lambda row: (row.last_responsibility, row.round_added),
                reverse=True,
            )[: local_slots - len(recent)]
            keep_rows.extend(recent + elite)
        keep = {id(row) for row in keep_rows}
        before = len(rows)
        rows[:] = [row for row in rows if id(row) in keep]
        return before - len(rows)
    recent_slots = (slots + 1) // 2
    recent = sorted(
        sampled,
        key=lambda row: row.round_added,
        reverse=True,
    )[:recent_slots]
    recent_ids = {id(row) for row in recent}
    elite = sorted(
        (row for row in sampled if id(row) not in recent_ids),
        key=lambda row: (row.last_responsibility, row.round_added),
        reverse=True,
    )[: slots - len(recent)]
    keep = {id(row) for row in gold + recent + elite}
    before = len(rows)
    rows[:] = [row for row in rows if id(row) in keep]
    return before - len(rows)


def _pad(
    rows: Iterable[L2RTrace],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = list(rows)
    pad = torch.nn.utils.rnn.pad_sequence
    ids = pad([row.ids for row in rows], batch_first=True, padding_value=pad_token_id)
    h_mask = pad([row.h_mask for row in rows], batch_first=True, padding_value=False)
    a_mask = pad([row.a_mask for row in rows], batch_first=True, padding_value=False)
    return ids.to(DEV), h_mask.to(DEV), a_mask.to(DEV)


def _proposal_component_h_logps(
    model,
    tok,
    task,
    rows: list[L2RTrace],
    *,
    proposal_prompt: str,
    micro: int,
) -> torch.Tensor:
    """Score each retained rationale under one alternate proposal prompt."""

    if proposal_prompt == "question":
        raise ValueError("alternate proposal scoring requires a non-prior prompt")
    if proposal_prompt not in PROPOSAL_PROMPTS:
        raise ValueError(f"unknown L2R proposal prompt {proposal_prompt!r}")
    prompt_cache: dict[int, torch.Tensor] = {}
    sequences = []
    masks = []
    for row in rows:
        if row.pid not in prompt_cache:
            task_builder = getattr(task, "build_proposal_prompt", None)
            prompt = (
                task_builder(int(row.pid), proposal_prompt)
                if task_builder is not None
                else _proposal_prompt(
                    task.prompts[row.pid],
                    task.gold_answer[row.pid],
                    proposal_prompt,
                )
            )
            tokenizer_kwargs = {"return_tensors": "pt"}
            if bool(getattr(task, "rendered_chat_prompts", False)):
                tokenizer_kwargs["add_special_tokens"] = False
            prompt_cache[row.pid] = tok(
                prompt,
                **tokenizer_kwargs,
            ).input_ids[0].detach().cpu()
        prompt_ids = prompt_cache[row.pid]
        h_ids = row.ids[row.h_mask].detach().cpu()
        sequences.append(torch.cat((prompt_ids, h_ids)))
        mask = torch.zeros(len(prompt_ids) + len(h_ids), dtype=torch.bool)
        mask[len(prompt_ids):] = True
        masks.append(mask)
    ids = torch.nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=task_pad_token_id(tok),
    ).to(DEV)
    h_mask = torch.nn.utils.rnn.pad_sequence(
        masks,
        batch_first=True,
        padding_value=False,
    ).to(DEV)
    return seq_logprobs(
        model,
        ids,
        h_mask,
        micro=micro,
        length_norm=False,
    )


def _temperature_component_h_logps(
    model,
    ids: torch.Tensor,
    h_mask: torch.Tensor,
    *,
    temperature: float,
    micro: int,
) -> torch.Tensor:
    """Score retained rationales under an autoregressive temperature proposal."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("proposal temperature must be finite and positive")
    values = []
    with torch.no_grad():
        for start in range(0, ids.shape[0], micro):
            mb_ids = ids[start:start + micro]
            mb_mask = h_mask[start:start + micro, 1:].to(torch.float32)
            logits = model(mb_ids).logits[:, :-1] / temperature
            token_logps = (
                logits.gather(-1, mb_ids[:, 1:, None]).squeeze(-1)
                - torch.logsumexp(logits, dim=-1)
            ).to(torch.float32)
            values.append((token_logps * mb_mask).sum(dim=1))
    return torch.cat(values)


def _generate_rows(
    model,
    tok,
    task,
    prompt_ids: list[torch.Tensor],
    pid_row: list[int],
    *,
    round_added: int,
    replica_row: list[int] | None = None,
    trace_segmentation: str = "legacy",
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
    proposal_prompt: str = "question",
    proposal_temperature: float = 1.0,
) -> tuple[list[L2RTrace], list[str], int]:
    if not pid_row:
        return [], [], 0
    if replica_row is None:
        replica_row = [0] * len(pid_row)
    if len(replica_row) != len(pid_row):
        raise ValueError("replica_row must align with pid_row")
    task_builder = getattr(task, "build_proposal_prompt", None)
    prompts = [
        (
            task_builder(int(pid), proposal_prompt)
            if task_builder is not None
            else _proposal_prompt(
                task.prompts[pid],
                task.gold_answer[pid],
                proposal_prompt,
            )
        )
        for pid in pid_row
    ]
    ids, mask, texts = sample_multi(
        model,
        tok,
        prompts,
        temperature=proposal_temperature,
        max_new=getattr(task, "max_new", 40),
    )
    rewards = task.reward(texts, pids=pid_row)
    floor = float(getattr(task, "floor", 0.0))
    rows = []
    generated_tokens = 0
    for index, (pid, replica) in enumerate(zip(pid_row, replica_row)):
        token_count = int(mask[index].sum())
        generated_tokens += token_count
        completion = ids[index][mask[index]].detach().cpu().tolist()
        answer_event = _task_parse_answer_event(
            task,
            texts[index],
            mode=answer_event_mode,
        )
        h_ids, valid, segmented_answer, segmentation_mode = (
            _extract_reasoning_tokens(
                tok,
                completion,
                texts[index],
                mode=trace_segmentation,
                answer_event_mode=answer_event_mode,
                parse_answer_event=lambda text, *, mode: _task_parse_answer_event(
                    task,
                    text,
                    mode=mode,
                ),
            )
        )
        text_diagnostics = _proposal_text_diagnostics(
            texts[index],
            task.gold_answer[pid],
        )
        rows.append(
            _make_trace(
                tok,
                prompt_ids[pid],
                h_ids,
                task.gold_answer[pid],
                pid=pid,
                round_added=round_added,
                text=texts[index],
                is_gold=False,
                proposal_correct=(
                    bool(
                        answer_event.strict_valid
                        and _task_answers_equivalent(
                            task,
                            answer_event.answer,
                            task.gold_answer[pid],
                        )
                    )
                    if answer_event_mode == "strict_terminal_marker"
                    else bool(float(rewards[index]) > floor + 0.5)
                ),
                generated_tokens=token_count,
                replica=replica,
                completion_key=tuple(int(token) for token in completion),
                segmentation_mode=segmentation_mode,
                segmentation_valid=valid,
                segmentation_answer=segmented_answer,
                answer_event_mode=answer_event_mode,
                answer_event_valid=answer_event.strict_valid,
                answer_marker_count=answer_event.marker_count,
                answer_marker_terminal=answer_event.terminal_marker,
                answer_target_termination=answer_target_termination,
                proposal_prompt=proposal_prompt,
                proposal_temperature=proposal_temperature,
                **text_diagnostics,
            )
        )
    return rows, texts, generated_tokens


def _sample_round(
    model,
    tok,
    task,
    prompt_ids: list[torch.Tensor],
    *,
    B: int,
    G: int,
    rng: np.random.Generator,
    round_added: int,
    adaptive_max_g: int,
    adaptive_batch_g: int,
    adaptive_min_correct: int,
    buffer_replicates: int = 1,
    selected_pids: list[int] | None = None,
    trace_segmentation: str = "legacy",
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
    proposal_prompt: str = "question",
) -> tuple[list[int], list[L2RTrace], list[str], dict]:
    n_questions = B // G
    if n_questions > len(task.prompts):
        raise ValueError(
            f"B//G requests {n_questions} unique questions from a pool of {len(task.prompts)}"
        )
    if G % buffer_replicates:
        raise ValueError(
            f"G must be divisible by buffer_replicates, got G={G}, "
            f"buffer_replicates={buffer_replicates}"
        )
    if selected_pids is None:
        pids = [
            int(pid)
            for pid in rng.choice(
                len(task.prompts),
                size=n_questions,
                replace=False,
            )
        ]
    else:
        pids = [int(pid) for pid in selected_pids]
        if len(pids) != n_questions or len(set(pids)) != len(pids):
            raise ValueError("selected_pids must contain B//G unique question ids")
        if any(pid < 0 or pid >= len(task.prompts) for pid in pids):
            raise ValueError("selected_pids contains an out-of-range question id")
    per_replica = G // buffer_replicates
    initial_pid_row = []
    initial_replica_row = []
    for pid in pids:
        for replica in range(buffer_replicates):
            initial_pid_row.extend([pid] * per_replica)
            initial_replica_row.extend([replica] * per_replica)
    rows, texts, generated_tokens = _generate_rows(
        model,
        tok,
        task,
        prompt_ids,
        initial_pid_row,
        round_added=round_added,
        replica_row=initial_replica_row,
        trace_segmentation=trace_segmentation,
        answer_event_mode=answer_event_mode,
        answer_target_termination=answer_target_termination,
        proposal_prompt=proposal_prompt,
    )
    counts = {pid: G for pid in pids}
    correct = {pid: 0 for pid in pids}
    for row in rows:
        correct[row.pid] += int(row.proposal_correct is True)

    extra_rows = []
    extra_texts = []
    extra_tokens = 0
    if adaptive_max_g > G:
        while True:
            pid_row = []
            replica_row = []
            for pid in pids:
                if correct[pid] >= adaptive_min_correct or counts[pid] >= adaptive_max_g:
                    continue
                add = min(adaptive_batch_g, adaptive_max_g - counts[pid])
                for local_index in range(add):
                    pid_row.append(pid)
                    replica_row.append(
                        (counts[pid] + local_index) % buffer_replicates
                    )
                counts[pid] += add
            if not pid_row:
                break
            new_rows, new_texts, new_tokens = _generate_rows(
                model,
                tok,
                task,
                prompt_ids,
                pid_row,
                round_added=round_added,
                replica_row=replica_row,
                trace_segmentation=trace_segmentation,
                answer_event_mode=answer_event_mode,
                answer_target_termination=answer_target_termination,
                proposal_prompt=proposal_prompt,
            )
            extra_rows.extend(new_rows)
            extra_texts.extend(new_texts)
            extra_tokens += new_tokens
            for row in new_rows:
                correct[row.pid] += int(row.proposal_correct is True)

    all_rows = rows + extra_rows
    stats = {
        "questions": len(pids),
        "initial_generations": len(rows),
        "adaptive_generations": len(extra_rows),
        "generations": len(all_rows),
        "generated_tokens": generated_tokens + extra_tokens,
        "mean_g": float(np.mean(list(counts.values()))),
        "max_g": max(counts.values(), default=0),
        "resolved_initial": sum(
            any(row.pid == pid and row.proposal_correct for row in rows) for pid in pids
        ),
        "resolved_final": sum(correct[pid] >= adaptive_min_correct for pid in pids),
        "target_mentioned_before_final_fraction": float(np.mean([
            row.target_mentioned_before_final is True for row in all_rows
        ])),
        "target_before_equation_fraction": float(np.mean([
            row.target_before_equation is True for row in all_rows
        ])),
        "equation_fraction": float(np.mean([
            row.has_equation is True for row in all_rows
        ])),
    }
    return pids, all_rows, texts + extra_texts, stats


def _stratified_proposal_counts(
    G: int,
    proposal_prior_fraction: float,
) -> tuple[int, int]:
    if not math.isfinite(proposal_prior_fraction) or not (
        0 < proposal_prior_fraction < 1
    ):
        raise ValueError("proposal_prior_fraction must be in (0, 1)")
    raw_prior = G * proposal_prior_fraction
    prior = int(round(raw_prior))
    if not math.isclose(raw_prior, prior, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "G * proposal_prior_fraction must be an integer for stratified sampling"
        )
    conditioned = G - prior
    if prior < 1 or conditioned < 1:
        raise ValueError("mixed proposals require at least one draw per component")
    return prior, conditioned


def _sample_mixed_round(
    model,
    tok,
    task,
    prompt_ids: list[torch.Tensor],
    *,
    B: int,
    G: int,
    rng: np.random.Generator,
    round_added: int,
    proposal_prompt: str,
    proposal_prior_fraction: float,
    selected_pids: list[int] | None = None,
    trace_segmentation: str = "legacy",
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
) -> tuple[list[int], list[L2RTrace], list[str], dict]:
    """Draw a fixed stratified prior/answer-conditioned proposal mixture."""

    if proposal_prompt == "question":
        raise ValueError("mixed proposals require an answer-conditioned prompt")
    prior_count, conditioned_count = _stratified_proposal_counts(
        G,
        proposal_prior_fraction,
    )
    n_questions = B // G
    if n_questions > len(task.prompts):
        raise ValueError(
            f"B//G requests {n_questions} unique questions from a pool of "
            f"{len(task.prompts)}"
        )
    if selected_pids is None:
        pids = [
            int(pid)
            for pid in rng.choice(
                len(task.prompts),
                size=n_questions,
                replace=False,
            )
        ]
    else:
        pids = [int(pid) for pid in selected_pids]
        if len(pids) != n_questions or len(set(pids)) != len(pids):
            raise ValueError("selected_pids must contain B//G unique question ids")
        if any(pid < 0 or pid >= len(task.prompts) for pid in pids):
            raise ValueError("selected_pids contains an out-of-range question id")

    prior_pid_row = [
        pid
        for pid in pids
        for _ in range(prior_count)
    ]
    conditioned_pid_row = [
        pid
        for pid in pids
        for _ in range(conditioned_count)
    ]
    prior_rows, prior_texts, prior_tokens = _generate_rows(
        model,
        tok,
        task,
        prompt_ids,
        prior_pid_row,
        round_added=round_added,
        trace_segmentation=trace_segmentation,
        answer_event_mode=answer_event_mode,
        answer_target_termination=answer_target_termination,
        proposal_prompt="question",
    )
    conditioned_rows, conditioned_texts, conditioned_tokens = _generate_rows(
        model,
        tok,
        task,
        prompt_ids,
        conditioned_pid_row,
        round_added=round_added,
        trace_segmentation=trace_segmentation,
        answer_event_mode=answer_event_mode,
        answer_target_termination=answer_target_termination,
        proposal_prompt=proposal_prompt,
    )
    rows = prior_rows + conditioned_rows
    texts = prior_texts + conditioned_texts
    correct = {
        pid: sum(
            row.proposal_correct is True
            for row in rows
            if row.pid == pid
        )
        for pid in pids
    }
    stats = {
        "questions": len(pids),
        "initial_generations": len(rows),
        "adaptive_generations": 0,
        "generations": len(rows),
        "generated_tokens": prior_tokens + conditioned_tokens,
        "mean_g": float(G),
        "max_g": G,
        "resolved_initial": sum(correct[pid] > 0 for pid in pids),
        "resolved_final": sum(correct[pid] > 0 for pid in pids),
        "prior_generations": len(prior_rows),
        "answer_conditioned_generations": len(conditioned_rows),
        "proposal_prior_fraction": proposal_prior_fraction,
        "target_mentioned_before_final_fraction": float(np.mean([
            row.target_mentioned_before_final is True for row in rows
        ])),
        "target_before_equation_fraction": float(np.mean([
            row.target_before_equation is True for row in rows
        ])),
        "equation_fraction": float(np.mean([
            row.has_equation is True for row in rows
        ])),
    }
    return pids, rows, texts, stats


def _sample_temperature_mixed_round(
    model,
    tok,
    task,
    prompt_ids: list[torch.Tensor],
    *,
    B: int,
    G: int,
    rng: np.random.Generator,
    round_added: int,
    proposal_prior_fraction: float,
    proposal_temperature: float,
    selected_pids: list[int] | None = None,
    trace_segmentation: str = "legacy",
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
) -> tuple[list[int], list[L2RTrace], list[str], dict]:
    """Draw a fixed mixture of unit- and hotter-temperature question proposals."""

    if not math.isfinite(proposal_temperature) or proposal_temperature <= 1:
        raise ValueError("temperature mixtures require proposal_temperature > 1")
    prior_count, hotter_count = _stratified_proposal_counts(
        G,
        proposal_prior_fraction,
    )
    n_questions = B // G
    if selected_pids is None:
        pids = [
            int(pid)
            for pid in rng.choice(len(task.prompts), size=n_questions, replace=False)
        ]
    else:
        pids = [int(pid) for pid in selected_pids]
        if len(pids) != n_questions or len(set(pids)) != len(pids):
            raise ValueError("selected_pids must contain B//G unique question ids")
        if any(pid < 0 or pid >= len(task.prompts) for pid in pids):
            raise ValueError("selected_pids contains an out-of-range question id")
    prior_pid_row = [pid for pid in pids for _ in range(prior_count)]
    hotter_pid_row = [pid for pid in pids for _ in range(hotter_count)]
    prior_rows, prior_texts, prior_tokens = _generate_rows(
        model,
        tok,
        task,
        prompt_ids,
        prior_pid_row,
        round_added=round_added,
        trace_segmentation=trace_segmentation,
        answer_event_mode=answer_event_mode,
        answer_target_termination=answer_target_termination,
        proposal_prompt="question",
        proposal_temperature=1.0,
    )
    hotter_rows, hotter_texts, hotter_tokens = _generate_rows(
        model,
        tok,
        task,
        prompt_ids,
        hotter_pid_row,
        round_added=round_added,
        trace_segmentation=trace_segmentation,
        answer_event_mode=answer_event_mode,
        answer_target_termination=answer_target_termination,
        proposal_prompt="question",
        proposal_temperature=proposal_temperature,
    )
    rows = prior_rows + hotter_rows
    correct = {
        pid: sum(row.proposal_correct is True for row in rows if row.pid == pid)
        for pid in pids
    }
    return pids, rows, prior_texts + hotter_texts, {
        "questions": len(pids),
        "initial_generations": len(rows),
        "adaptive_generations": 0,
        "generations": len(rows),
        "generated_tokens": prior_tokens + hotter_tokens,
        "mean_g": float(G),
        "max_g": G,
        "resolved_initial": sum(correct[pid] > 0 for pid in pids),
        "resolved_final": sum(correct[pid] > 0 for pid in pids),
        "prior_generations": len(prior_rows),
        "temperature_generations": len(hotter_rows),
        "proposal_prior_fraction": proposal_prior_fraction,
        "proposal_temperature": proposal_temperature,
        "target_mentioned_before_final_fraction": float(np.mean([
            row.target_mentioned_before_final is True for row in rows
        ])),
        "target_before_equation_fraction": float(np.mean([
            row.target_before_equation is True for row in rows
        ])),
        "equation_fraction": float(np.mean([row.has_equation is True for row in rows])),
    }


def _reader_decode_correct(
    model,
    tok,
    task,
    rows: list[L2RTrace],
    *,
    reader_mode: str,
    max_new: int = 12,
    batch_size: int = 16,
    exact_cache: bool = False,
    cache_stats: dict[str, int] | None = None,
) -> tuple[list[bool], int]:
    """Greedily decode answers from q+h without teacher-forcing a*."""

    cache_stats = cache_stats if cache_stats is not None else _new_cache_stats()
    use_cache = exact_cache and reader_mode == "frozen"
    missing_indices = (
        [
            index
            for index, row in enumerate(rows)
            if row.frozen_reader_decode_correct is None
        ]
        if use_cache else list(range(len(rows)))
    )
    if use_cache:
        hits = len(rows) - len(missing_indices)
        missing_set = set(missing_indices)
        cache_stats["reader_decode_hits"] += hits
        cache_stats["reader_decode_misses"] += len(missing_indices)
        cache_stats["saved_forward_rows"] += hits
        cache_stats["saved_forward_tokens"] += sum(
            len(rows[index].ids)
            for index in range(len(rows))
            if index not in missing_set
        )
        if not missing_indices:
            return (
                [bool(row.frozen_reader_decode_correct) for row in rows],
                0,
            )
    decode_rows = [rows[index] for index in missing_indices]
    prefixes = []
    for row in decode_rows:
        answer_positions = row.a_mask.nonzero()
        stop = int(answer_positions[0]) if len(answer_positions) else len(row.ids)
        prefixes.append(row.ids[:stop])
    outputs = []
    generated_token_counts = []
    context = model.disable_adapter() if reader_mode == "frozen" else nullcontext()
    model.eval()
    with context, torch.no_grad():
        for start in range(0, len(prefixes), batch_size):
            chunk = prefixes[start:start + batch_size]
            width = max(len(prefix) for prefix in chunk)
            chat_runtime = bool(
                getattr(task, "rendered_chat_prompts", False)
            )
            padding_token_id = (
                int(tok.pad_token_id)
                if chat_runtime
                else int(tok.eos_token_id)
            )
            batch_ids = torch.full(
                (len(chunk), width),
                padding_token_id,
                dtype=torch.long,
                device=DEV,
            )
            attention = torch.zeros_like(batch_ids)
            for index, prefix in enumerate(chunk):
                batch_ids[index, -len(prefix):] = prefix.to(DEV)
                attention[index, -len(prefix):] = 1
            generation_kwargs = {
                "input_ids": batch_ids,
                "attention_mask": attention,
                "do_sample": False,
                "max_new_tokens": max_new,
                "pad_token_id": padding_token_id,
            }
            if chat_runtime:
                generation_kwargs["eos_token_id"] = int(tok.eos_token_id)
            generated = model.generate(**generation_kwargs)
            outputs.extend(
                tok.batch_decode(generated[:, width:], skip_special_tokens=True)
            )
            continuation = generated[:, width:]
            active_rows = []
            for row in continuation:
                eos = (row == tok.eos_token_id).nonzero()
                active_count = int(eos[0].item()) + 1 if len(eos) else len(row)
                generated_token_counts.append(active_count)
                active_rows.append(row[:active_count])
            if chat_runtime:
                audit_chat_active_completions(tok, active_rows)
    if getattr(task, "answer_event_mode", "legacy") == "strict_terminal_marker":
        decoded = []
        for output, row in zip(outputs, decode_rows):
            event = _task_parse_answer_event(
                task,
                f"#### {output}",
                mode="strict_terminal_marker",
            )
            decoded.append(
                event.strict_valid
                and _task_answers_equivalent(
                    task,
                    event.answer,
                    task.gold_answer[row.pid],
                )
            )
    else:
        rewards = task.reward(outputs, pids=[row.pid for row in decode_rows])
        floor = float(getattr(task, "floor", 0.0))
        decoded = [bool(float(reward) > floor + 0.5) for reward in rewards]
    if use_cache:
        for index, correct, token_count in zip(
            missing_indices,
            decoded,
            generated_token_counts,
        ):
            rows[index].frozen_reader_decode_correct = correct
            rows[index].frozen_reader_decode_tokens = token_count
        return (
            [bool(row.frozen_reader_decode_correct) for row in rows],
            sum(generated_token_counts),
        )
    return decoded, sum(generated_token_counts)


def _within_question_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _format_failure_diagnostics(
    rows: list[L2RTrace],
    *,
    example_limit: int = 16,
) -> dict:
    """Summarise proposal formatting under each row's declared event contract."""
    failures = [
        row
        for row in rows
        if (
            not row.answer_event_valid
            if row.answer_event_mode == "strict_terminal_marker"
            else not has_answer_marker(row.text)
        )
    ]
    correct = [row for row in failures if row.proposal_correct is True]
    incorrect = [row for row in failures if row.proposal_correct is not True]
    multiple_markers = [
        row for row in rows if row.answer_marker_count > 1
    ]
    nonterminal_markers = [
        row
        for row in rows
        if row.answer_marker_count == 1 and not row.answer_marker_terminal
    ]
    count = len(rows)
    return {
        "sample_count": count,
        "failure_count": len(failures),
        "failure_fraction": len(failures) / count if count else None,
        "correct_without_marker_count": len(correct),
        "correct_without_marker_fraction": len(correct) / count if count else None,
        "incorrect_without_marker_count": len(incorrect),
        "incorrect_without_marker_fraction": len(incorrect) / count if count else None,
        "multiple_marker_count": len(multiple_markers),
        "multiple_marker_fraction": (
            len(multiple_markers) / count if count else None
        ),
        "nonterminal_marker_count": len(nonterminal_markers),
        "nonterminal_marker_fraction": (
            len(nonterminal_markers) / count if count else None
        ),
        "examples_truncated": max(0, len(failures) - example_limit),
        "examples": [
            {
                "pid": row.pid,
                "text": row.text,
                "proposal_correct": row.proposal_correct,
                "generated_tokens": row.generated_tokens,
                "answer_event_mode": row.answer_event_mode,
                "answer_marker_count": row.answer_marker_count,
                "answer_marker_terminal": row.answer_marker_terminal,
            }
            for row in failures[:example_limit]
        ],
    }


def _segmentation_diagnostics(rows: list[L2RTrace]) -> dict:
    sampled = [row for row in rows if not row.is_gold]
    valid = [row for row in sampled if row.segmentation_valid]
    modes: dict[str, int] = {}
    for row in sampled:
        modes[row.segmentation_mode] = modes.get(row.segmentation_mode, 0) + 1
    return {
        "sample_count": len(sampled),
        "valid_count": len(valid),
        "invalid_count": len(sampled) - len(valid),
        "valid_fraction": len(valid) / len(sampled) if sampled else None,
        "modes": modes,
    }


def _posterior_diagnostics(
    rows: list[L2RTrace],
    weights: torch.Tensor,
    pids: torch.Tensor,
    policy_h: torch.Tensor,
    reader_a: torch.Tensor,
    *,
    answer_proposal_h: torch.Tensor | None = None,
    proposal_prior_fraction: float | None = None,
    top_k: int = 3,
) -> tuple[dict, list[dict]]:
    weights_np = weights.detach().double().cpu().numpy()
    pids_np = pids.detach().cpu().numpy()
    h_lengths = np.asarray([row.h_tokens for row in rows], dtype=np.float64)
    policy_np = policy_h.detach().double().cpu().numpy()
    reader_np = reader_a.detach().double().cpu().numpy()
    answer_proposal_np = None
    mixture_np = None
    correction_np = None
    if answer_proposal_h is not None:
        if answer_proposal_h.shape != policy_h.shape:
            raise ValueError("answer proposal scores must align with posterior rows")
        if proposal_prior_fraction is None or not (
            0 < proposal_prior_fraction < 1
        ):
            raise ValueError(
                "mixed posterior diagnostics require proposal_prior_fraction in (0, 1)"
            )
        answer_proposal_np = (
            answer_proposal_h.detach().double().cpu().numpy()
        )
        mixture_np = np.logaddexp(
            policy_np + math.log(proposal_prior_fraction),
            answer_proposal_np + math.log1p(-proposal_prior_fraction),
        )
        correction_np = policy_np - mixture_np
    prompt_stats = []
    top = []
    for pid in sorted(set(int(value) for value in pids_np)):
        local = np.flatnonzero(pids_np == pid)
        active = local[weights_np[local] > 0]
        if not len(active):
            continue
        local_w = weights_np[active]
        local_w = local_w / local_w.sum()
        gold = np.asarray([rows[index].is_gold for index in active], dtype=bool)
        model_correct = np.asarray(
            [rows[index].proposal_correct is True and not rows[index].is_gold for index in active],
            dtype=bool,
        )
        formatted = np.asarray(
            [
                rows[index].is_gold
                or (
                    rows[index].answer_event_valid
                    if rows[index].answer_event_mode
                    == "strict_terminal_marker"
                    else has_answer_marker(rows[index].text)
                )
                for index in active
            ],
            dtype=bool,
        )
        segmented = np.asarray(
            [rows[index].segmentation_valid for index in active],
            dtype=bool,
        )
        prior_draw = np.asarray(
            [
                rows[index].proposal_prompt == "question"
                and math.isclose(rows[index].proposal_temperature, 1.0)
                for index in active
            ],
            dtype=bool,
        )
        draw_ess = 1.0 / np.square(local_w).sum()
        ess = draw_ess / len(active)
        unique_mass: dict[tuple[int, ...], float] = {}
        for index, weight in zip(active, local_w):
            row = rows[int(index)]
            rationale_key = tuple(
                int(token) for token in row.ids[row.h_mask].tolist()
            )
            unique_mass[rationale_key] = (
                unique_mass.get(rationale_key, 0.0) + float(weight)
            )
        unique_weights = np.asarray(
            list(unique_mass.values()),
            dtype=np.float64,
        )
        unique_ess = 1.0 / np.square(unique_weights).sum()
        unique_ess_fraction = unique_ess / len(unique_weights)
        entropy = -(local_w * np.log(np.clip(local_w, 1e-300, None))).sum()
        corr = None
        if len(active) > 1 and np.std(local_w) > 0 and np.std(h_lengths[active]) > 0:
            corr = float(np.corrcoef(local_w, h_lengths[active])[0, 1])
        prompt_stats.append({
            "pid": pid,
            "active": int(len(active)),
            "unique_rationales": int(len(unique_weights)),
            "draw_ess": float(draw_ess),
            "ess_fraction": float(ess),
            "unique_ess": float(unique_ess),
            "unique_ess_fraction": float(unique_ess_fraction),
            "entropy": float(entropy),
            "max_weight": float(local_w.max()),
            "gold_mass": float(local_w[gold].sum()),
            "model_correct_mass": float(local_w[model_correct].sum()),
            "format_failure_mass": float(local_w[~formatted].sum()),
            "invalid_segmentation_mass": float(local_w[~segmented].sum()),
            "weighted_h_tokens": float(np.dot(local_w, h_lengths[active])),
            "weight_length_corr": corr,
            "proposal_prior_draw_fraction": (
                float(prior_draw.mean())
                if answer_proposal_np is not None else None
            ),
            "proposal_prior_posterior_mass": (
                float(local_w[prior_draw].sum())
                if answer_proposal_np is not None else None
            ),
            "proposal_answer_minus_prior_logp": (
                float(np.mean(
                    answer_proposal_np[active] - policy_np[active]
                ))
                if answer_proposal_np is not None else None
            ),
            "proposal_log_importance_correction": (
                float(np.mean(correction_np[active]))
                if correction_np is not None else None
            ),
        })
        for index in active[np.argsort(-local_w)[:top_k]]:
            row = rows[int(index)]
            top.append({
                "pid": pid,
                "responsibility": float(weights_np[index]),
                "is_gold": row.is_gold,
                "proposal_correct": row.proposal_correct,
                "format_applicable": not row.is_gold,
                "has_answer_marker": (
                    None if row.is_gold else has_answer_marker(row.text)
                ),
                "segmentation_mode": row.segmentation_mode,
                "segmentation_valid": row.segmentation_valid,
                "segmentation_answer": row.segmentation_answer,
                "h_tokens": row.h_tokens,
                "round_added": row.round_added,
                "proposal_prompt": row.proposal_prompt,
                "proposal_temperature": row.proposal_temperature,
                "target_mentioned_before_final": (
                    row.target_mentioned_before_final
                ),
                "target_before_equation": row.target_before_equation,
                "has_equation": row.has_equation,
                "policy_h_logp": float(policy_np[index]),
                "reader_a_logp": float(reader_np[index]),
                "answer_proposal_h_logp": (
                    float(answer_proposal_np[index])
                    if answer_proposal_np is not None else None
                ),
                "proposal_mixture_logp": (
                    float(mixture_np[index])
                    if mixture_np is not None else None
                ),
                "proposal_log_importance_correction": (
                    float(correction_np[index])
                    if correction_np is not None else None
                ),
                "text": row.text,
            })
    summary = {
        "question_count": len(prompt_stats),
        "active_rows": int((weights > 0).sum()),
        "unique_rationales": _within_question_mean(
            [float(row["unique_rationales"]) for row in prompt_stats]
        ),
        "draw_ess": _within_question_mean(
            [row["draw_ess"] for row in prompt_stats]
        ),
        "ess_fraction": _within_question_mean(
            [row["ess_fraction"] for row in prompt_stats]
        ),
        "unique_ess": _within_question_mean(
            [row["unique_ess"] for row in prompt_stats]
        ),
        "unique_ess_fraction": _within_question_mean(
            [row["unique_ess_fraction"] for row in prompt_stats]
        ),
        "entropy": _within_question_mean([row["entropy"] for row in prompt_stats]),
        "max_weight": _within_question_mean([row["max_weight"] for row in prompt_stats]),
        "gold_mass": _within_question_mean([row["gold_mass"] for row in prompt_stats]),
        "model_correct_mass": _within_question_mean(
            [row["model_correct_mass"] for row in prompt_stats]
        ),
        "format_failure_mass": _within_question_mean(
            [row["format_failure_mass"] for row in prompt_stats]
        ),
        "invalid_segmentation_mass": _within_question_mean(
            [row["invalid_segmentation_mass"] for row in prompt_stats]
        ),
        "weighted_h_tokens": _within_question_mean(
            [row["weighted_h_tokens"] for row in prompt_stats]
        ),
        "weight_length_corr": _within_question_mean(
            [
                row["weight_length_corr"]
                for row in prompt_stats
                if row["weight_length_corr"] is not None
            ]
        ),
    }
    if answer_proposal_np is not None:
        prior_draw_fraction = _within_question_mean([
            row["proposal_prior_draw_fraction"] for row in prompt_stats
        ])
        prior_posterior_mass = _within_question_mean([
            row["proposal_prior_posterior_mass"] for row in prompt_stats
        ])
        summary.update({
            "proposal_prior_fraction": proposal_prior_fraction,
            "proposal_prior_draw_fraction": prior_draw_fraction,
            "proposal_conditioned_draw_fraction": (
                None
                if prior_draw_fraction is None
                else 1.0 - prior_draw_fraction
            ),
            "proposal_prior_posterior_mass": prior_posterior_mass,
            "proposal_conditioned_posterior_mass": (
                None
                if prior_posterior_mass is None
                else 1.0 - prior_posterior_mass
            ),
            "proposal_answer_minus_prior_logp": _within_question_mean([
                row["proposal_answer_minus_prior_logp"]
                for row in prompt_stats
            ]),
            "proposal_log_importance_correction": _within_question_mean([
                row["proposal_log_importance_correction"]
                for row in prompt_stats
            ]),
        })
    return summary, top


def _snapshot_gradients(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> list[torch.Tensor | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for _, parameter in named_parameters
    ]


def _assign_gradients(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    gradients: list[torch.Tensor | None],
) -> None:
    if len(named_parameters) != len(gradients):
        raise ValueError("L2R gradient snapshot has the wrong length")
    with torch.no_grad():
        for (_, parameter), gradient in zip(named_parameters, gradients):
            if gradient is None:
                parameter.grad = None
            elif parameter.grad is None:
                parameter.grad = gradient.detach().clone()
            else:
                parameter.grad.copy_(gradient)


def _gradient_norm(gradients: list[torch.Tensor | None]) -> float:
    squared = sum(
        float(torch.sum(gradient.detach().float().square()))
        for gradient in gradients
        if gradient is not None
    )
    if not math.isfinite(squared):
        raise FloatingPointError("non-finite L2R gradient norm")
    return math.sqrt(max(squared, 0.0))


def _mstep_support_diagnostics(
    rows: list[L2RTrace],
    weights: torch.Tensor,
    *,
    mstep_objective: str,
) -> dict[str, float | int | None]:
    values = [float(value) for value in weights.detach().cpu()]
    questions = sorted({row.pid for row in rows})
    question_count = len(questions)
    correct_mass = sum(
        weight
        for row, weight in zip(rows, values)
        if row.is_gold or row.proposal_correct is True
    )
    valid_mass = sum(
        weight
        for row, weight in zip(rows, values)
        if row.segmentation_valid
    )
    ess_fractions = []
    max_weights = []
    for pid in questions:
        local = [
            weight
            for row, weight in zip(rows, values)
            if row.pid == pid
        ]
        total = sum(local)
        if total <= 0:
            continue
        normalized = [weight / total for weight in local]
        ess_fractions.append(
            1.0 / (len(normalized) * sum(weight * weight for weight in normalized))
        )
        max_weights.append(max(normalized))
    return {
        "active_questions": question_count,
        "active_traces": len(rows),
        "backward_tokens": sum(
            _trace_objective_tokens(row, mstep_objective)
            for row in rows
        ),
        "correct_trace_mass": (
            correct_mass / question_count if question_count else None
        ),
        "segmentation_valid_mass": (
            valid_mass / question_count if question_count else None
        ),
        "mean_ess_fraction": (
            float(np.mean(ess_fractions)) if ess_fractions else None
        ),
        "mean_max_responsibility": (
            float(np.mean(max_weights)) if max_weights else None
        ),
        "weighted_h_tokens": (
            sum(weight * row.h_tokens for row, weight in zip(rows, values))
            / question_count
            if question_count else None
        ),
    }


def _weighted_policy_loss(
    model,
    ids: torch.Tensor,
    objective_mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    question_count: int,
    micro: int,
    length_norm: bool,
) -> float:
    """Evaluate the fixed-responsibility M-step surrogate without gradients."""

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    was_training = bool(model.training)
    try:
        model.eval()
        total = 0.0
        for start in range(0, len(ids), micro):
            stop = min(start + micro, len(ids))
            logp = seq_logprobs(
                model,
                ids[start:stop],
                objective_mask[start:stop],
                micro=max(1, stop - start),
                grad=False,
                length_norm=length_norm,
            )
            total -= float((weights[start:stop] * logp).sum())
        return total / max(question_count, 1)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        model.train(was_training)


def _diagnostic_safety_gradient(
    model,
    tok,
    rows: list[L2RTrace],
    micro: int,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    gradients_to_restore: list[torch.Tensor | None],
) -> list[torch.Tensor | None] | None:
    if not rows:
        return None
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        model.zero_grad(set_to_none=True)
        ids, h_mask, _ = _pad(rows, tok.eos_token_id)
        row_count = max(len(ids), 1)
        for start in range(0, len(ids), micro):
            stop = min(start + micro, len(ids))
            loss = -seq_logprobs(
                model,
                ids[start:stop],
                h_mask[start:stop],
                micro=max(1, stop - start),
                grad=True,
                length_norm=True,
            ).sum() / row_count
            loss.backward()
        return _snapshot_gradients(named_parameters)
    finally:
        _assign_gradients(named_parameters, gradients_to_restore)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def _question_gradient_attribution(
    model,
    tok,
    rows: list[L2RTrace],
    weights: torch.Tensor,
    *,
    micro: int,
    length_norm: bool,
    mstep_objective: str,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    aggregate_gradients: list[torch.Tensor | None],
    gradients_to_restore: list[torch.Tensor | None],
    safety_gradients: list[torch.Tensor | None] | None,
    limit: int,
) -> dict[str, object]:
    if limit <= 0:
        return {
            "enabled": False,
            "selection": "disabled",
            "question_limit": 0,
            "questions": [],
        }

    values = weights.detach().cpu().tolist()
    candidates = []
    for pid in sorted({row.pid for row in rows}):
        local = [
            float(weight)
            for row, weight in zip(rows, values)
            if row.pid == pid
        ]
        if local:
            candidates.append((-max(local), int(pid)))
    selected = sorted(candidates)[:limit]
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    records = []
    try:
        for _negative_max_weight, pid in selected:
            indices = [
                index for index, row in enumerate(rows) if row.pid == pid
            ]
            local_rows = [rows[index] for index in indices]
            local_weights = weights[indices].to(DEV)
            model.zero_grad(set_to_none=True)
            ids, h_mask, a_mask = _pad(local_rows, tok.eos_token_id)
            objective_mask = _l2r_objective_mask(
                h_mask,
                a_mask,
                mstep_objective,
            )
            logp = seq_logprobs(
                model,
                ids,
                objective_mask,
                micro=micro,
                grad=True,
                length_norm=length_norm,
            )
            loss = -(local_weights * logp).sum()
            loss.backward()
            question_gradients = _snapshot_gradients(named_parameters)
            local_values = [
                float(value) for value in local_weights.detach().cpu()
            ]
            records.append({
                "pid": pid,
                "trace_count": len(local_rows),
                "max_responsibility": max(local_values),
                "correct_trace_mass": sum(
                    value
                    for row, value in zip(local_rows, local_values)
                    if row.is_gold or row.proposal_correct is True
                ),
                "gradient_l2_norm": tensor_list_norm(question_gradients),
                "cosine_with_objective": tensor_list_cosine(
                    question_gradients,
                    aggregate_gradients,
                ),
                "cosine_with_safety": (
                    tensor_list_cosine(question_gradients, safety_gradients)
                    if safety_gradients is not None else None
                ),
            })
    finally:
        _assign_gradients(named_parameters, gradients_to_restore)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    return {
        "enabled": True,
        "selection": "highest_max_responsibility",
        "question_limit": int(limit),
        "questions": records,
    }


def _apply_policy_anchor(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    objective_gradients: list[torch.Tensor | None],
    *,
    mode: str,
    fixed_beta: float,
    state: dict[str, float],
    target_ratio: float | None,
    beta_min: float,
    beta_max: float,
    ema: float,
    epsilon: float = 1e-12,
) -> dict:
    """Replace ``g_objective + g_KL`` by the registered anchored gradient."""

    raw_anchor = []
    for (_, parameter), objective in zip(named_parameters, objective_gradients):
        combined = parameter.grad
        if combined is None:
            raw_anchor.append(None)
        elif objective is None:
            raw_anchor.append(combined.detach().clone())
        else:
            raw_anchor.append(combined.detach() - objective)
    objective_norm = _gradient_norm(objective_gradients)
    raw_anchor_norm = _gradient_norm(raw_anchor)

    previous_objective = state.get("ema_objective_grad_norm")
    previous_anchor = state.get("ema_raw_anchor_grad_norm")
    ema_objective = (
        objective_norm
        if previous_objective is None
        else ema * previous_objective + (1.0 - ema) * objective_norm
    )
    ema_anchor = (
        raw_anchor_norm
        if previous_anchor is None
        else ema * previous_anchor + (1.0 - ema) * raw_anchor_norm
    )
    state["ema_objective_grad_norm"] = ema_objective
    state["ema_raw_anchor_grad_norm"] = ema_anchor

    if mode == "grad_ratio":
        if target_ratio is None:
            raise ValueError("grad_ratio anchoring requires a target ratio")
        if target_ratio == 0 or ema_objective <= epsilon or ema_anchor <= epsilon:
            beta_unclipped = 0.0
        else:
            beta_unclipped = target_ratio * ema_objective / max(
                ema_anchor, epsilon
            )
        beta = min(max(beta_unclipped, beta_min), beta_max)
    else:
        beta_unclipped = fixed_beta
        beta = fixed_beta

    with torch.no_grad():
        for (_, parameter), objective, anchor in zip(
            named_parameters,
            objective_gradients,
            raw_anchor,
        ):
            if objective is None and anchor is None:
                parameter.grad = None
                continue
            combined = (
                torch.zeros_like(anchor)
                if objective is None
                else objective.clone()
            )
            if anchor is not None and beta:
                combined.add_(anchor, alpha=beta)
            if parameter.grad is None:
                parameter.grad = combined
            else:
                parameter.grad.copy_(combined)

    dot = sum(
        float(torch.sum(objective.float() * anchor.float()))
        for objective, anchor in zip(objective_gradients, raw_anchor)
        if objective is not None and anchor is not None
    )
    denominator = objective_norm * raw_anchor_norm
    return {
        "beta": beta,
        "beta_unclipped": beta_unclipped,
        "beta_clipped": float(beta != beta_unclipped),
        "objective_grad_norm": objective_norm,
        "raw_anchor_grad_norm": raw_anchor_norm,
        "applied_anchor_grad_norm": beta * raw_anchor_norm,
        "achieved_ratio": (
            beta * raw_anchor_norm / objective_norm
            if objective_norm > epsilon
            else 0.0
        ),
        "objective_anchor_cosine": (
            min(max(dot / denominator, -1.0), 1.0)
            if denominator > 0 and math.isfinite(dot)
            else None
        ),
        "ema_objective_grad_norm": ema_objective,
        "ema_raw_anchor_grad_norm": ema_anchor,
    }


def _snapshot_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> list[torch.Tensor]:
    return [parameter.detach().clone() for _, parameter in named_parameters]


def _restore_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    values: list[torch.Tensor],
) -> None:
    with torch.no_grad():
        for (_, parameter), value in zip(named_parameters, values):
            parameter.copy_(value)


def _mean_h_nll(model, tok, rows: list[L2RTrace], micro: int) -> float:
    if not rows:
        return float("nan")
    ids, h_mask, _ = _pad(rows, tok.eos_token_id)
    logp = seq_logprobs(
        model,
        ids,
        h_mask,
        micro=micro,
        length_norm=True,
    )
    return -float(logp.mean())


def _mean_answer_nll(model, tok, rows: list[L2RTrace], micro: int) -> float:
    """Mean token-normalized NLL of canonical gold-answer suffixes."""

    if not rows:
        return float("nan")
    was_training = bool(model.training)
    try:
        model.eval()
        ids, _, answer_mask = _pad(rows, tok.eos_token_id)
        logp = seq_logprobs(
            model,
            ids,
            answer_mask,
            micro=micro,
            length_norm=True,
        )
        return -float(logp.mean())
    finally:
        model.train(was_training)


def _diagnostic_gold_answer_gradient(
    model,
    tok,
    rows: list[L2RTrace],
    micro: int,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    gradients_to_restore: list[torch.Tensor | None],
) -> tuple[float, list[torch.Tensor | None]] | None:
    """Return held-out direct-answer NLL and its gradient without side effects."""

    if not rows:
        return None
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    was_training = bool(model.training)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        ids, _, answer_mask = _pad(rows, tok.eos_token_id)
        row_count = max(len(ids), 1)
        total_loss = 0.0
        for start in range(0, len(ids), micro):
            stop = min(start + micro, len(ids))
            loss = -seq_logprobs(
                model,
                ids[start:stop],
                answer_mask[start:stop],
                micro=max(1, stop - start),
                grad=True,
                length_norm=True,
            ).sum() / row_count
            loss.backward()
            total_loss += float(loss.detach())
        return total_loss, _snapshot_gradients(named_parameters)
    finally:
        _assign_gradients(named_parameters, gradients_to_restore)
        model.train(was_training)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def _parameter_delta_tensors(
    before: list[torch.Tensor],
    after: list[torch.Tensor],
) -> list[torch.Tensor]:
    if len(before) != len(after):
        raise ValueError("parameter snapshots must have equal lengths")
    deltas = []
    for old, new in zip(before, after):
        if old.shape != new.shape:
            raise ValueError("parameter snapshots contain a mismatched shape")
        deltas.append(new.detach() - old.detach())
    return deltas


@torch.no_grad()
def _candidate_utility_probe(
    model,
    tok,
    task,
    pids: list[int],
    *,
    batch: int,
) -> dict[str, object]:
    """Greedily decode one fixed completion per reserved tuning question."""

    if not pids:
        raise ValueError("candidate utility probe requires at least one question")
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    was_training = bool(model.training)
    had_padding_side = hasattr(tok, "padding_side")
    padding_side = getattr(tok, "padding_side", None)
    texts: list[str] = []
    generated_tokens = 0
    try:
        model.eval()
        tok.padding_side = "left"
        prompts = [task.prompts[pid] for pid in pids]
        for start in range(0, len(prompts), batch):
            chunk = prompts[start:start + batch]
            chat_runtime = bool(getattr(task, "rendered_chat_prompts", False))
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "padding": True,
            }
            if chat_runtime:
                tokenizer_kwargs["add_special_tokens"] = False
            encoded = tok(chunk, **tokenizer_kwargs).to(DEV)
            generation_kwargs = {
                **encoded,
                "do_sample": False,
                "max_new_tokens": getattr(task, "max_new", 256),
                "pad_token_id": (
                    tok.pad_token_id
                    if chat_runtime and tok.pad_token_id is not None
                    else tok.eos_token_id
                ),
            }
            if chat_runtime:
                generation_kwargs["eos_token_id"] = int(tok.eos_token_id)
            generated = model.generate(**generation_kwargs)
            continuation = generated[:, encoded.input_ids.shape[1]:]
            active_rows = []
            for row in continuation:
                eos_positions = (row == tok.eos_token_id).nonzero(
                    as_tuple=False
                )
                active_count = (
                    int(eos_positions[0].item()) + 1
                    if len(eos_positions) else len(row)
                )
                generated_tokens += active_count
                active_rows.append(row[:active_count])
            if chat_runtime:
                audit_chat_active_completions(tok, active_rows)
            texts.extend(
                tok.batch_decode(
                    continuation,
                    skip_special_tokens=True,
                )
            )
    finally:
        if had_padding_side:
            tok.padding_side = padding_side
        elif hasattr(tok, "padding_side"):
            delattr(tok, "padding_side")
        model.train(was_training)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    answer_event_mode = getattr(task, "answer_event_mode", "legacy")
    events = [
        _task_parse_answer_event(task, text, mode=answer_event_mode)
        for text in texts
    ]
    predictions = [event.answer for event in events]
    gold = [task.gold_answer[pid] for pid in pids]
    correct = [
        _task_answers_equivalent(task, prediction, answer)
        for prediction, answer in zip(predictions, gold)
    ]
    format_failures = [
        (
            not event.strict_valid
            if answer_event_mode == "strict_terminal_marker"
            else not has_answer_marker(text)
        )
        for text, event in zip(texts, events)
    ]
    question_ids = [
        (
            int(task.train_qi[pid])
            if hasattr(task, "train_qi")
            else int(pid)
        )
        for pid in pids
    ]
    return {
        "decode_mode": "greedy",
        "common_randomness": "identical_questions_and_deterministic_decode",
        "question_count": len(pids),
        "generated_tokens": generated_tokens,
        "pids": [int(pid) for pid in pids],
        "question_ids": question_ids,
        "gold_answers": gold,
        "predictions": predictions,
        "correct": correct,
        "format_failures": format_failures,
        "answer_event_mode": answer_event_mode,
        "accuracy": float(np.mean(correct)),
        "format_failure_fraction": float(np.mean(format_failures)),
    }


def _paired_candidate_utility(
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    """Construct paired per-question outcome changes for one candidate step."""

    identity_fields = ("pids", "question_ids", "gold_answers")
    for field in identity_fields:
        if before[field] != after[field]:
            raise ValueError(
                f"candidate utility probe changed paired field {field!r}"
            )
    before_correct = list(before["correct"])
    after_correct = list(after["correct"])
    if len(before_correct) != len(after_correct):
        raise ValueError("candidate utility probe changed question count")
    improved = sum(
        not old and new for old, new in zip(before_correct, after_correct)
    )
    worsened = sum(
        old and not new for old, new in zip(before_correct, after_correct)
    )
    paired_outcomes = [
        {
            "pid": int(pid),
            "question_id": int(question_id),
            "gold_answer": gold_answer,
            "prediction_before": prediction_before,
            "prediction_after": prediction_after,
            "correct_before": bool(correct_before),
            "correct_after": bool(correct_after),
            "format_failure_before": bool(format_before),
            "format_failure_after": bool(format_after),
        }
        for (
            pid,
            question_id,
            gold_answer,
            prediction_before,
            prediction_after,
            correct_before,
            correct_after,
            format_before,
            format_after,
        ) in zip(
            before["pids"],
            before["question_ids"],
            before["gold_answers"],
            before["predictions"],
            after["predictions"],
            before_correct,
            after_correct,
            before["format_failures"],
            after["format_failures"],
        )
    ]
    return {
        "decode_mode": before["decode_mode"],
        "common_randomness": before["common_randomness"],
        "question_count": len(before_correct),
        "accuracy_before": float(before["accuracy"]),
        "accuracy_after": float(after["accuracy"]),
        "accuracy_delta": float(after["accuracy"]) - float(before["accuracy"]),
        "format_failure_fraction_before": float(
            before["format_failure_fraction"]
        ),
        "format_failure_fraction_after": float(
            after["format_failure_fraction"]
        ),
        "improved_questions": improved,
        "worsened_questions": worsened,
        "unchanged_questions": len(before_correct) - improved - worsened,
        "paired_outcomes": paired_outcomes,
    }


def _question_balanced_weights(rows: list[L2RTrace], device) -> torch.Tensor:
    """Give every question equal mass and every trace equal within-question mass."""

    if not rows:
        return torch.empty(0, device=device)
    counts: dict[int, int] = {}
    for row in rows:
        counts[row.pid] = counts.get(row.pid, 0) + 1
    question_count = len(counts)
    return torch.tensor(
        [1.0 / (question_count * counts[row.pid]) for row in rows],
        device=device,
        dtype=torch.float32,
    )


def _token_logprob_matrix(model, ids: torch.Tensor, micro: int, *, grad: bool) -> torch.Tensor:
    """Compute per-token log probabilities without materialising all rows at once."""

    return torch.cat(
        [
            token_logps(model, ids[start:start + micro], grad=grad)
            for start in range(0, len(ids), micro)
        ],
        dim=0,
    )


def _reader_answer_logps(
    model,
    tok,
    rows: list[L2RTrace],
    ids: torch.Tensor,
    a_mask: torch.Tensor,
    *,
    micro: int,
    reader_mode: str,
    exact_cache: bool,
    cache_stats: dict[str, int],
) -> torch.Tensor:
    """Score answers, reusing values only when the reader is immutable."""

    if not exact_cache or reader_mode != "frozen":
        context = (
            model.disable_adapter() if reader_mode == "frozen" else nullcontext()
        )
        with context:
            return seq_logprobs(
                model,
                ids,
                a_mask,
                micro=micro,
                length_norm=False,
            )

    missing = [
        index for index, row in enumerate(rows)
        if row.frozen_reader_logp is None
    ]
    hits = len(rows) - len(missing)
    cache_stats["reader_score_hits"] += hits
    cache_stats["reader_score_misses"] += len(missing)
    cache_stats["saved_forward_rows"] += hits
    cache_stats["saved_forward_tokens"] += sum(
        int(rows[index].a_mask.sum())
        for index in range(len(rows))
        if rows[index].frozen_reader_logp is not None
    )
    if missing:
        missing_rows = [rows[index] for index in missing]
        missing_ids, _, missing_answer_mask = _pad(
            missing_rows,
            tok.eos_token_id,
        )
        with model.disable_adapter():
            values = seq_logprobs(
                model,
                missing_ids,
                missing_answer_mask,
                micro=micro,
                length_norm=False,
            )
        for index, value in zip(missing, values.detach().cpu().tolist()):
            rows[index].frozen_reader_logp = float(value)
    return torch.tensor(
        [float(row.frozen_reader_logp) for row in rows],
        device=DEV,
        dtype=torch.float32,
    )


def _frozen_base_token_logps(
    model,
    tok,
    rows: list[L2RTrace],
    ids: torch.Tensor,
    *,
    micro: int,
    exact_cache: bool,
    cache_stats: dict[str, int],
) -> torch.Tensor:
    """Return immutable base-policy token scores aligned to a padded batch."""

    if not exact_cache:
        with torch.no_grad(), model.disable_adapter():
            return _token_logprob_matrix(model, ids, micro, grad=False)

    missing = [
        index for index, row in enumerate(rows)
        if row.frozen_base_token_logps is None
    ]
    hits = len(rows) - len(missing)
    cache_stats["base_token_hits"] += hits
    cache_stats["base_token_misses"] += len(missing)
    cache_stats["saved_forward_rows"] += hits
    cache_stats["saved_forward_tokens"] += sum(
        max(len(rows[index].ids) - 1, 0)
        for index in range(len(rows))
        if rows[index].frozen_base_token_logps is not None
    )
    if missing:
        missing_rows = [rows[index] for index in missing]
        missing_ids, _, _ = _pad(missing_rows, tok.eos_token_id)
        with torch.no_grad(), model.disable_adapter():
            values = _token_logprob_matrix(
                model,
                missing_ids,
                micro,
                grad=False,
            )
        for index, value in zip(missing, values.detach().cpu()):
            token_count = max(len(rows[index].ids) - 1, 0)
            rows[index].frozen_base_token_logps = (
                value[:token_count].to(torch.float32).clone()
            )

    matrix = torch.zeros(
        (len(rows), max(ids.shape[1] - 1, 0)),
        device=DEV,
        dtype=torch.float32,
    )
    for index, row in enumerate(rows):
        cached = row.frozen_base_token_logps
        if cached is None:
            raise RuntimeError("frozen base-token cache was not populated")
        matrix[index, :len(cached)] = cached.to(DEV)
    return matrix


def _tokenwise_h_k3_statistics(
    model,
    ids: torch.Tensor,
    h_mask: torch.Tensor,
    micro: int,
    *,
    grad: bool,
    base_token_logps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-trace token-mean k3 and log-ratio clipping fractions."""

    if base_token_logps is None:
        with torch.no_grad(), model.disable_adapter():
            base_token_logps = _token_logprob_matrix(
                model,
                ids,
                micro,
                grad=False,
            )
    current = _token_logprob_matrix(model, ids, micro, grad=grad)
    scored = h_mask[:, 1:].to(current.dtype)
    raw_log_ratio = base_token_logps - current
    log_ratio = raw_log_ratio.clamp(-5, 5)
    token_k3 = log_ratio.exp() - log_ratio - 1
    denominator = scored.sum(1).clamp_min(1)
    k3_rows = (token_k3 * scored).sum(1) / denominator
    clipped_rows = (
        ((raw_log_ratio < -5) | (raw_log_ratio > 5)).to(scored.dtype)
        * scored
    ).sum(1) / denominator
    return k3_rows, clipped_rows


def _tokenwise_h_k3_rows(
    model,
    ids: torch.Tensor,
    h_mask: torch.Tensor,
    micro: int,
    *,
    grad: bool,
    base_token_logps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the mean token-level k3 movement surrogate for each trace."""

    rows, _ = _tokenwise_h_k3_statistics(
        model,
        ids,
        h_mask,
        micro,
        grad=grad,
        base_token_logps=base_token_logps,
    )
    return rows


def _history_monitor_rows(
    buffers: dict[int, list[L2RTrace]],
    pids: list[int],
) -> list[L2RTrace]:
    """Choose one stable valid target per historical question."""

    selected = []
    for pid in pids:
        rows = buffers.get(pid, [])
        gold = next(
            (
                row
                for row in rows
                if row.is_gold and row.segmentation_valid
            ),
            None,
        )
        if gold is not None:
            selected.append(gold)
            continue
        valid = [row for row in rows if row.segmentation_valid]
        if valid:
            selected.append(
                max(
                    valid,
                    key=lambda row: (
                        row.last_responsibility,
                        -row.round_added,
                    ),
                )
            )
    return selected


def _mstep(
    model,
    tok,
    opt,
    rows: list[L2RTrace],
    weights: torch.Tensor,
    *,
    iters: int,
    micro: int,
    length_norm: bool,
    mstep_objective: str = "generator",
    kl_coef: float,
    gradient_projector: GradientProjector | None = None,
    named_parameters: list[tuple[str, torch.nn.Parameter]] | None = None,
    policy_anchor_mode: str = "fixed",
    policy_anchor_target_ratio: float | None = None,
    policy_anchor_beta_min: float = 0.0,
    policy_anchor_beta_max: float = 10.0,
    policy_anchor_ema: float = 0.9,
    policy_anchor_scope: str = "generator",
    policy_anchor_state: dict[str, float] | None = None,
    trust_kl_budget: float | None = None,
    safety_rows: list[L2RTrace] | None = None,
    trust_safety_tolerance: float = 0.0,
    boundary_failure_fraction: float = 0.0,
    trust_boundary_failure_ceiling: float = 1.0,
    trust_max_backtracks: int = 0,
    trust_backtrack_shrink: float = 0.5,
    history_rows: list[L2RTrace] | None = None,
    trust_probe_rows: list[L2RTrace] | None = None,
    diagnostics_enabled: bool = False,
    diagnostics_level: str = "standard",
    diagnostics_gradient_questions: int = 0,
    diagnostics_probe_fn=None,
    candidate_utility_rows: list[L2RTrace] | None = None,
    candidate_utility_probe_fn=None,
    diagnostic_state: dict[str, object] | None = None,
    exact_cache: bool = False,
    cache_stats: dict[str, int] | None = None,
) -> dict:
    diagnostics_level = validate_diagnostic_level(diagnostics_level)
    if mstep_objective not in MSTEP_OBJECTIVES:
        raise ValueError(f"unknown L2R M-step objective {mstep_objective!r}")
    if policy_anchor_scope not in POLICY_ANCHOR_SCOPES:
        raise ValueError(
            f"unknown L2R policy anchor scope {policy_anchor_scope!r}"
        )
    if diagnostics_gradient_questions < 0:
        raise ValueError("diagnostics_gradient_questions must be nonnegative")
    if diagnostics_gradient_questions and diagnostics_level != "deep":
        raise ValueError(
            "diagnostics_gradient_questions requires diagnostics_level='deep'"
        )
    if diagnostics_probe_fn is not None and diagnostics_level != "deep":
        raise ValueError("diagnostics_probe_fn requires diagnostics_level='deep'")
    candidate_utility_enabled = bool(
        candidate_utility_rows or candidate_utility_probe_fn is not None
    )
    if bool(candidate_utility_rows) != (
        candidate_utility_probe_fn is not None
    ):
        raise ValueError(
            "candidate utility diagnostics require both reserved rows and a "
            "probe function"
        )
    if candidate_utility_enabled and diagnostics_level != "deep":
        raise ValueError(
            "candidate utility diagnostics require diagnostics_level='deep'"
        )
    if not diagnostics_enabled and (
        diagnostics_level != "standard"
        or diagnostics_gradient_questions
        or diagnostics_probe_fn is not None
        or candidate_utility_enabled
    ):
        raise ValueError("deep L2R diagnostics require diagnostics_enabled=True")
    diagnostic_state = diagnostic_state if diagnostic_state is not None else {}
    diagnostic_state.setdefault("accepted_steps", 0)
    diagnostic_state.setdefault("consecutive_rejections", 0)
    active = torch.nonzero(weights > 0, as_tuple=False).flatten()
    if not len(active):
        return {
            "loss": float("nan"),
            "policy_loss": float("nan"),
            "kl_penalty": float("nan"),
            "gradient_steps": 0,
            "backward_tokens": 0,
            "policy_backward_tokens": 0,
            "anchor_backward_tokens": 0,
            "gradient_norm_raw": float("nan"),
            "optimizer_update_norm_raw": float("nan"),
            "optimizer_update_norm_projected": float("nan"),
            "optimizer_update_norm_applied": float("nan"),
            "projection_retained_fraction": float("nan"),
            "policy_anchor_beta": None,
            "policy_anchor_beta_unclipped": None,
            "policy_anchor_beta_clipped": None,
            "policy_anchor_objective_grad_norm": None,
            "policy_anchor_raw_grad_norm": None,
            "policy_anchor_applied_grad_norm": None,
            "policy_anchor_achieved_ratio": None,
            "policy_anchor_objective_cosine": None,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "backtracks": 0,
            "safety_loss_before": None,
            "safety_loss_after": None,
            "safety_loss_reference": None,
            "realized_kl": None,
            "realized_kl_k1": None,
            "trust_log_ratio_clip_fraction": None,
            "history_loss_before": None,
            "history_loss_after": None,
            "trust_probe_rows": 0,
            "trust_probe_questions": 0,
            "boundary_gate_passed": (
                boundary_failure_fraction <= trust_boundary_failure_ceiling
            ),
            "inner_step_diagnostics": [],
            "diagnostics_level": diagnostics_level,
            "diagnostic_probe_elapsed_seconds": 0.0,
            "candidate_utility_elapsed_seconds": 0.0,
        }
    if named_parameters is None:
        raise RuntimeError("L2R M-step requires named trainable parameters")
    if policy_anchor_state is None:
        policy_anchor_state = {}
    cache_stats = cache_stats if cache_stats is not None else _new_cache_stats()
    selected_rows = [rows[int(index)] for index in active.cpu().tolist()]
    has_safety = bool(safety_rows)
    has_history = bool(history_rows)
    probe_rows = list(trust_probe_rows or [])
    trust_metric_rows = (
        probe_rows if policy_anchor_mode == "grad_ratio" else selected_rows
    )
    ids, h_mask, a_mask = _pad(selected_rows, tok.eos_token_id)
    objective_mask = _l2r_objective_mask(
        h_mask,
        a_mask,
        mstep_objective,
    )
    local_weights = weights[active].to(DEV)
    question_count = len({row.pid for row in selected_rows})
    support_diagnostics = (
        _mstep_support_diagnostics(
            selected_rows,
            local_weights,
            mstep_objective=mstep_objective,
        )
        if diagnostics_enabled else {}
    )
    inner_step_diagnostics: list[dict[str, object]] = []
    diagnostic_probe_elapsed_seconds = 0.0
    candidate_utility_elapsed_seconds = 0.0
    anchor_requested = kl_coef > 0 or policy_anchor_mode == "grad_ratio"
    anchor_rows = (
        probe_rows if policy_anchor_mode == "grad_ratio" else selected_rows
    )
    anchor_enabled = anchor_requested and bool(anchor_rows)
    anchor_ids = anchor_mask = anchor_weights = base_token_matrix = None
    if anchor_enabled:
        anchor_ids, anchor_h_mask, anchor_a_mask = _pad(
            anchor_rows,
            tok.eos_token_id,
        )
        anchor_mask = _l2r_anchor_mask(
            anchor_h_mask,
            anchor_a_mask,
            policy_anchor_scope,
        )
        anchor_weights = (
            _question_balanced_weights(anchor_rows, anchor_ids.device)
            if policy_anchor_mode == "grad_ratio"
            else local_weights / max(question_count, 1)
        )
        base_token_matrix = _frozen_base_token_logps(
            model,
            tok,
            anchor_rows,
            anchor_ids,
            micro=micro,
            exact_cache=exact_cache,
            cache_stats=cache_stats,
        )

    total_loss = total_policy = total_kl = 0.0
    total_gradient_norm = 0.0
    total_update_raw = total_update_projected = total_update_applied = 0.0
    total_retained = 0.0
    anchor_sums = {
        "beta": 0.0,
        "beta_unclipped": 0.0,
        "beta_clipped": 0.0,
        "objective_grad_norm": 0.0,
        "raw_anchor_grad_norm": 0.0,
        "applied_anchor_grad_norm": 0.0,
        "achieved_ratio": 0.0,
        "objective_anchor_cosine": 0.0,
        "ema_objective_grad_norm": 0.0,
        "ema_raw_anchor_grad_norm": 0.0,
    }
    anchor_cosine_count = 0
    accepted_steps = rejected_steps = backtracks = 0
    safety_before_sum = safety_after_sum = 0.0
    realized_kl_sum = realized_kl_k1_sum = 0.0
    log_ratio_clip_sum = 0.0
    policy_backward_tokens = anchor_backward_tokens = 0
    safety_reference = None
    safety_current = None
    if has_safety:
        safety_reference = policy_anchor_state.get("safety_loss_reference")
        created_safety_reference = safety_reference is None
        if safety_reference is None:
            safety_reference = _mean_h_nll(
                model,
                tok,
                safety_rows or [],
                micro,
            )
            policy_anchor_state["safety_loss_reference"] = safety_reference
            if exact_cache:
                cache_stats["safety_nll_misses"] += 1
        if exact_cache:
            safety_current = policy_anchor_state.get("safety_loss_current")
            if safety_current is None:
                safety_current = (
                    safety_reference
                    if created_safety_reference
                    else _mean_h_nll(model, tok, safety_rows or [], micro)
                )
                if not created_safety_reference:
                    cache_stats["safety_nll_misses"] += 1
                policy_anchor_state["safety_loss_current"] = safety_current
    history_before = (
        _mean_h_nll(model, tok, history_rows or [], micro)
        if has_history else None
    )
    for inner_index in range(iters):
        step_started = time.perf_counter()
        parameter_before_step = (
            _snapshot_parameters(named_parameters)
            if diagnostics_level == "deep" else None
        )
        optimizer_before_step = (
            optimizer_moment_diagnostics(opt)
            if diagnostics_level == "deep" else None
        )
        step_attempts: list[dict[str, object]] = []
        opt.zero_grad(set_to_none=True)
        step_loss = step_policy = step_kl = 0.0
        for start in range(0, len(selected_rows), micro):
            stop = min(start + micro, len(selected_rows))
            logp = seq_logprobs(
                model,
                ids[start:stop],
                objective_mask[start:stop],
                micro=max(1, stop - start),
                grad=True,
                length_norm=length_norm,
            )
            policy_loss = -(
                local_weights[start:stop] * logp
            ).sum() / max(question_count, 1)
            policy_loss.backward()
            step_policy += float(policy_loss.detach())
            policy_backward_tokens += int(objective_mask[start:stop].sum())
        objective_gradients = _snapshot_gradients(named_parameters)
        anchor_metrics = None
        step_log_ratio_clip = 0.0
        if base_token_matrix is not None:
            for start in range(0, len(anchor_rows), micro):
                stop = min(start + micro, len(anchor_rows))
                kl_rows, clipped_rows = _tokenwise_h_k3_statistics(
                    model,
                    anchor_ids[start:stop],
                    anchor_mask[start:stop],
                    max(1, stop - start),
                    grad=True,
                    base_token_logps=base_token_matrix[start:stop],
                )
                kl_penalty = (
                    anchor_weights[start:stop] * kl_rows
                ).sum()
                step_log_ratio_clip += float(
                    (
                        anchor_weights[start:stop] * clipped_rows
                    ).sum().detach()
                )
                kl_penalty.backward()
                step_kl += float(kl_penalty.detach())
                anchor_backward_tokens += int(anchor_mask[start:stop].sum())
            anchor_metrics = _apply_policy_anchor(
                named_parameters,
                objective_gradients,
                mode=policy_anchor_mode,
                fixed_beta=kl_coef,
                state=policy_anchor_state,
                target_ratio=policy_anchor_target_ratio,
                beta_min=policy_anchor_beta_min,
                beta_max=policy_anchor_beta_max,
                ema=policy_anchor_ema,
            )
            for key in anchor_sums:
                value = anchor_metrics[key]
                if value is not None:
                    anchor_sums[key] += float(value)
            anchor_cosine_count += int(
                anchor_metrics["objective_anchor_cosine"] is not None
            )
        effective_beta = (
            anchor_metrics["beta"] if anchor_metrics is not None else 0.0
        )
        step_loss = step_policy + effective_beta * step_kl
        combined_gradients = (
            _snapshot_gradients(named_parameters)
            if diagnostics_enabled else []
        )
        diagnostic_policy_loss_before = (
            _weighted_policy_loss(
                model,
                ids,
                objective_mask,
                local_weights,
                question_count=question_count,
                micro=micro,
                length_norm=length_norm,
            )
            if diagnostics_level == "deep"
            else step_policy
        )
        safety_gradients = (
            _diagnostic_safety_gradient(
                model,
                tok,
                safety_rows or [],
                micro,
                named_parameters,
                combined_gradients,
            )
            if diagnostics_enabled
            and diagnostics_level == "deep"
            and has_safety
            else None
        )
        question_gradient_attribution = (
            _question_gradient_attribution(
                model,
                tok,
                selected_rows,
                local_weights,
                micro=micro,
                length_norm=length_norm,
                mstep_objective=mstep_objective,
                named_parameters=named_parameters,
                aggregate_gradients=objective_gradients,
                gradients_to_restore=combined_gradients,
                safety_gradients=safety_gradients,
                limit=diagnostics_gradient_questions,
            )
            if diagnostics_enabled
            and diagnostics_level == "deep"
            and diagnostics_gradient_questions > 0
            else {
                "enabled": False,
                "selection": "disabled",
                "question_limit": int(diagnostics_gradient_questions),
                "questions": [],
            }
        )
        candidate_utility = {
            "enabled": candidate_utility_enabled,
            "evaluated": False,
            "reason": (
                "awaiting_terminal_candidate"
                if candidate_utility_enabled else "disabled"
            ),
            "candidate_status": None,
            "terminal_attempt": None,
            "gold_answer": None,
            "free_decode": None,
            "alignment": None,
            "elapsed_seconds": 0.0,
        }
        capability_gradients = None
        candidate_utility_probe_before = None
        candidate_utility_started = None
        if candidate_utility_enabled:
            candidate_utility_started = time.perf_counter()
            candidate_utility_probe_before = diagnostic_state.get(
                "candidate_utility_previous_probe"
            )
            if candidate_utility_probe_before is None:
                candidate_utility_probe_before = candidate_utility_probe_fn(
                    model
                )
                diagnostic_state["candidate_utility_baseline_probe"] = (
                    candidate_utility_probe_before
                )
                diagnostic_state["candidate_utility_previous_probe"] = (
                    candidate_utility_probe_before
                )
            capability = _diagnostic_gold_answer_gradient(
                model,
                tok,
                candidate_utility_rows or [],
                micro,
                named_parameters,
                combined_gradients,
            )
            if capability is None:
                raise RuntimeError(
                    "candidate utility audit has no gold-answer rows"
                )
            candidate_utility_loss_before, capability_gradients = capability
            candidate_utility.update({
                "gold_answer": {
                    "definition": (
                        "token_mean_nll_of_direct_gold_answer_suffix_given_question"
                    ),
                    "loss_before": candidate_utility_loss_before,
                    "loss_after": None,
                    "loss_delta": None,
                    "gradient_l2_norm": tensor_list_norm(
                        capability_gradients
                    ),
                },
                "alignment": {
                    "positive_means_capability_descent": True,
                    "objective_gradient_cosine": tensor_list_cosine(
                        objective_gradients,
                        capability_gradients,
                    ),
                    "combined_gradient_cosine": tensor_list_cosine(
                        combined_gradients,
                        capability_gradients,
                    ),
                    "candidate_parameter_delta_cosine": None,
                },
            })
        if gradient_projector is None:
            raise RuntimeError("L2R M-step requires an update projector")

        enforce_trust = policy_anchor_mode == "grad_ratio"
        boundary_ok = (
            boundary_failure_fraction <= trust_boundary_failure_ceiling
        )
        if enforce_trust and not boundary_ok:
            accepted = False
            rejection_reason = "trace_boundary"
            projection = {
                "gradient_norm_raw": _gradient_norm(
                    _snapshot_gradients(named_parameters)
                ),
                "optimizer_update_norm_raw": 0.0,
                "optimizer_update_norm_projected": 0.0,
                "optimizer_update_norm_applied": 0.0,
                "projection_retained_fraction": 0.0,
            }
            rejected_steps += 1
            if has_safety and exact_cache:
                safety_before = float(safety_current)
                cache_stats["safety_nll_hits"] += 1
                cache_stats["saved_forward_rows"] += len(safety_rows or [])
                cache_stats["saved_forward_tokens"] += sum(
                    row.h_tokens for row in (safety_rows or [])
                )
            else:
                safety_before = (
                    _mean_h_nll(model, tok, safety_rows or [], micro)
                    if has_safety else 0.0
                )
            safety_after = safety_before
            realized_kl_k1 = _empirical_h_kl(
                model,
                tok,
                trust_metric_rows,
                micro,
                scope=policy_anchor_scope,
            )
            realized_kl = _empirical_h_kl_nonnegative(
                model,
                tok,
                trust_metric_rows,
                micro,
                scope=policy_anchor_scope,
            )
            if diagnostics_enabled:
                step_attempts.append({
                    "attempt": 0,
                    "learning_rate_scale": 0.0,
                    "accepted": False,
                    "failed_gates": ["trace_boundary"],
                    "policy_loss_after": None,
                    "policy_loss_delta": None,
                    "parameter_update_l2_norm": 0.0,
                    "realized_kl": finite_or_none(realized_kl),
                    "realized_kl_k1": finite_or_none(realized_kl_k1),
                    "safety_loss_after": finite_or_none(safety_after),
                    "elapsed_seconds": 0.0,
                })
        else:
            parameter_snapshot = _snapshot_parameters(named_parameters)
            optimizer_snapshot = copy.deepcopy(opt.state_dict())
            base_learning_rates = [
                float(group["lr"]) for group in opt.param_groups
            ]
            if has_safety and exact_cache:
                safety_before = float(safety_current)
                cache_stats["safety_nll_hits"] += 1
                cache_stats["saved_forward_rows"] += len(safety_rows or [])
                cache_stats["saved_forward_tokens"] += sum(
                    row.h_tokens for row in (safety_rows or [])
                )
            else:
                safety_before = (
                    _mean_h_nll(model, tok, safety_rows or [], micro)
                    if has_safety else 0.0
                )
            accepted = False
            projection = {}
            realized_kl = float("nan")
            realized_kl_k1 = float("nan")
            safety_after = float("nan")
            attempts = trust_max_backtracks + 1 if enforce_trust else 1
            for attempt in range(attempts):
                attempt_started = time.perf_counter()
                if attempt:
                    _restore_parameters(named_parameters, parameter_snapshot)
                    opt.load_state_dict(optimizer_snapshot)
                    backtracks += 1
                scale = trust_backtrack_shrink ** attempt
                for group, base_lr in zip(opt.param_groups, base_learning_rates):
                    group["lr"] = base_lr * scale
                projection = gradient_projector.step(opt)
                realized_kl_k1 = _empirical_h_kl(
                    model,
                    tok,
                    trust_metric_rows,
                    micro,
                    scope=policy_anchor_scope,
                )
                realized_kl = _empirical_h_kl_nonnegative(
                    model,
                    tok,
                    trust_metric_rows,
                    micro,
                    scope=policy_anchor_scope,
                )
                safety_after = (
                    _mean_h_nll(model, tok, safety_rows or [], micro)
                    if has_safety else 0.0
                )
                if has_safety and exact_cache:
                    cache_stats["safety_nll_misses"] += 1
                kl_ok = (
                    trust_kl_budget is None
                    or max(realized_kl, 0.0) <= trust_kl_budget
                )
                safety_ok = (
                    not has_safety
                    or safety_after
                    <= float(safety_reference) + trust_safety_tolerance
                )
                attempt_accepted = not enforce_trust or (kl_ok and safety_ok)
                policy_loss_after = (
                    _weighted_policy_loss(
                        model,
                        ids,
                        objective_mask,
                        local_weights,
                        question_count=question_count,
                        micro=micro,
                        length_norm=length_norm,
                    )
                    if diagnostics_level == "deep"
                    else None
                )
                parameter_update_norm = (
                    parameter_delta_norm(
                        parameter_snapshot,
                        _snapshot_parameters(named_parameters),
                    )
                    if diagnostics_level == "deep"
                    else None
                )
                failed_gates = []
                if enforce_trust and not kl_ok:
                    failed_gates.append("realized_kl")
                if enforce_trust and not safety_ok:
                    failed_gates.append("safety_nll")
                terminal_attempt = attempt_accepted or attempt == attempts - 1
                if candidate_utility_enabled and terminal_attempt:
                    utility_after_parameters = _snapshot_parameters(
                        named_parameters
                    )
                    utility_after_loss = _mean_answer_nll(
                        model,
                        tok,
                        candidate_utility_rows or [],
                        micro,
                    )
                    utility_after_probe = candidate_utility_probe_fn(model)
                    applied_delta = _parameter_delta_tensors(
                        parameter_before_step or [],
                        utility_after_parameters,
                    )
                    capability_descent = [
                        -gradient if gradient is not None else None
                        for gradient in capability_gradients or []
                    ]
                    candidate_utility.update({
                        "evaluated": True,
                        "reason": None,
                        "candidate_status": (
                            "accepted" if attempt_accepted
                            else "rejected_by_trust_region"
                        ),
                        "terminal_attempt": attempt,
                        "gold_answer": {
                            **candidate_utility["gold_answer"],
                            "loss_after": utility_after_loss,
                            "loss_delta": (
                                utility_after_loss
                                - candidate_utility["gold_answer"]["loss_before"]
                            ),
                        },
                        "free_decode": _paired_candidate_utility(
                            candidate_utility_probe_before,
                            utility_after_probe,
                        ),
                        "alignment": {
                            **candidate_utility["alignment"],
                            "candidate_parameter_delta_cosine": tensor_list_cosine(
                                applied_delta,
                                capability_descent,
                            ),
                        },
                    })
                    if attempt_accepted:
                        diagnostic_state[
                            "candidate_utility_previous_probe"
                        ] = utility_after_probe
                if diagnostics_enabled:
                    step_attempts.append({
                        "attempt": attempt,
                        "learning_rate_scale": scale,
                        "accepted": attempt_accepted,
                        "failed_gates": failed_gates,
                        "policy_loss_after": policy_loss_after,
                        "policy_loss_delta": (
                            policy_loss_after - diagnostic_policy_loss_before
                            if policy_loss_after is not None else None
                        ),
                        "parameter_update_l2_norm": parameter_update_norm,
                        "realized_kl": finite_or_none(realized_kl),
                        "realized_kl_k1": finite_or_none(realized_kl_k1),
                        "safety_loss_after": finite_or_none(safety_after),
                        "elapsed_seconds": time.perf_counter() - attempt_started,
                    })
                if attempt_accepted:
                    accepted = True
                    break
            for group, base_lr in zip(opt.param_groups, base_learning_rates):
                group["lr"] = base_lr
            if accepted:
                accepted_steps += 1
                rejection_reason = None
                if has_safety and exact_cache:
                    safety_current = float(safety_after)
                    policy_anchor_state["safety_loss_current"] = safety_current
            else:
                _restore_parameters(named_parameters, parameter_snapshot)
                opt.load_state_dict(optimizer_snapshot)
                projection["optimizer_update_norm_applied"] = 0.0
                rejected_steps += 1
                failed = {
                    gate
                    for attempt_record in step_attempts
                    for gate in attempt_record["failed_gates"]
                }
                rejection_reason = "+".join(sorted(failed)) or "trust_region"

        if candidate_utility_enabled:
            if candidate_utility["reason"] == "awaiting_terminal_candidate":
                candidate_utility["reason"] = "trace_boundary_no_update"
            candidate_utility["elapsed_seconds"] = (
                time.perf_counter() - candidate_utility_started
            )
            candidate_utility_elapsed_seconds += float(
                candidate_utility["elapsed_seconds"]
            )

        if diagnostics_enabled:
            if accepted:
                diagnostic_state["accepted_steps"] = (
                    int(diagnostic_state["accepted_steps"]) + 1
                )
                diagnostic_state["consecutive_rejections"] = 0
            else:
                diagnostic_state["consecutive_rejections"] = (
                    int(diagnostic_state["consecutive_rejections"]) + 1
                )
        behavioural_probe = {
            "evaluated": False,
            "accuracy": None,
            "delta_from_baseline": None,
            "delta_from_previous": None,
            "elapsed_seconds": 0.0,
        }
        if (
            accepted
            and diagnostics_level == "deep"
            and diagnostics_probe_fn is not None
        ):
            probe_started = time.perf_counter()
            probe_accuracy = run_diagnostic_probe(model, diagnostics_probe_fn)
            probe_elapsed = time.perf_counter() - probe_started
            diagnostic_probe_elapsed_seconds += probe_elapsed
            baseline = diagnostic_state.get("probe_baseline_accuracy")
            previous = diagnostic_state.get("probe_previous_accuracy")
            behavioural_probe = {
                "evaluated": True,
                "accuracy": probe_accuracy,
                "delta_from_baseline": (
                    probe_accuracy - float(baseline)
                    if baseline is not None else None
                ),
                "delta_from_previous": (
                    probe_accuracy - float(previous)
                    if previous is not None else None
                ),
                "elapsed_seconds": probe_elapsed,
            }
            diagnostic_state["probe_previous_accuracy"] = probe_accuracy
        parameter_after_step = (
            _snapshot_parameters(named_parameters)
            if diagnostics_enabled and diagnostics_level == "deep"
            else None
        )
        applied_parameter_norm = (
            parameter_delta_norm(parameter_before_step, parameter_after_step)
            if parameter_before_step is not None
            and parameter_after_step is not None
            else None
        )
        if diagnostics_enabled:
            accepted_attempt = next(
                (
                    attempt_record
                    for attempt_record in reversed(step_attempts)
                    if attempt_record["accepted"]
                ),
                None,
            )
            rejected_attempt_elapsed = sum(
                float(attempt_record["elapsed_seconds"])
                for attempt_record in step_attempts
                if not attempt_record["accepted"]
            )
            inner_step_diagnostics.append({
                "inner_step": inner_index,
                "status": "accepted" if accepted else "rejected",
                "rejection_reason": rejection_reason,
                "objective": {
                    "policy_loss_before": diagnostic_policy_loss_before,
                    "policy_loss_after": (
                        accepted_attempt["policy_loss_after"]
                        if accepted_attempt is not None else None
                    ),
                    "policy_loss_delta": (
                        accepted_attempt["policy_loss_delta"]
                        if accepted_attempt is not None else None
                    ),
                },
                "trust": {
                    "enforced": enforce_trust,
                    "boundary_ok": boundary_ok,
                    "kl_budget": trust_kl_budget,
                    "safety_tolerance": trust_safety_tolerance,
                    "safety_loss_reference": finite_or_none(safety_reference),
                    "safety_loss_before": finite_or_none(safety_before),
                    "safety_loss_after": finite_or_none(safety_after),
                },
                "policy_kl": {
                    "realized_k3": finite_or_none(realized_kl),
                    "realized_k1": finite_or_none(realized_kl_k1),
                    "log_ratio_clip_fraction": finite_or_none(
                        step_log_ratio_clip
                    ),
                },
                "gradient": {
                    "objective_l2_norm": tensor_list_norm(objective_gradients),
                    "combined_l2_norm": tensor_list_norm(combined_gradients),
                    "safety_l2_norm": (
                        tensor_list_norm(safety_gradients)
                        if safety_gradients is not None else None
                    ),
                    "objective_anchor_cosine": (
                        anchor_metrics["objective_anchor_cosine"]
                        if anchor_metrics is not None else None
                    ),
                    "objective_safety_cosine": (
                        tensor_list_cosine(
                            objective_gradients,
                            safety_gradients,
                        )
                        if safety_gradients is not None else None
                    ),
                    "combined_safety_cosine": (
                        tensor_list_cosine(
                            combined_gradients,
                            safety_gradients,
                        )
                        if safety_gradients is not None else None
                    ),
                },
                "update": {
                    "applied_parameter_l2_norm": applied_parameter_norm,
                    "optimizer_before": optimizer_before_step,
                    "optimizer_after": (
                        optimizer_moment_diagnostics(opt)
                        if diagnostics_level == "deep" else None
                    ),
                },
                "anchor": anchor_metrics,
                "support": support_diagnostics,
                "gradient_attribution": question_gradient_attribution,
                "behavioural_probe": behavioural_probe,
                "candidate_utility": candidate_utility,
                "attempts": step_attempts,
                "elapsed_seconds": time.perf_counter() - step_started,
                "rejected_attempt_elapsed_seconds": rejected_attempt_elapsed,
                "consecutive_rejections": int(
                    diagnostic_state["consecutive_rejections"]
                ),
                "cumulative_accepted_steps": int(
                    diagnostic_state["accepted_steps"]
                ),
            })
        total_loss += step_loss
        total_policy += step_policy
        total_kl += step_kl
        total_gradient_norm += projection["gradient_norm_raw"]
        total_update_raw += projection["optimizer_update_norm_raw"]
        total_update_projected += projection["optimizer_update_norm_projected"]
        total_update_applied += projection["optimizer_update_norm_applied"]
        total_retained += projection["projection_retained_fraction"]
        safety_before_sum += safety_before
        safety_after_sum += safety_after
        realized_kl_sum += realized_kl
        realized_kl_k1_sum += realized_kl_k1
        log_ratio_clip_sum += step_log_ratio_clip
    history_after = (
        _mean_h_nll(model, tok, history_rows or [], micro)
        if has_history else None
    )
    return {
        "loss": total_loss / iters,
        "policy_loss": total_policy / iters,
        "kl_penalty": total_kl / iters,
        "gradient_steps": iters,
        "backward_tokens": policy_backward_tokens + anchor_backward_tokens,
        "policy_backward_tokens": policy_backward_tokens,
        "anchor_backward_tokens": anchor_backward_tokens,
        "gradient_norm_raw": total_gradient_norm / iters,
        "optimizer_update_norm_raw": total_update_raw / iters,
        "optimizer_update_norm_projected": total_update_projected / iters,
        "optimizer_update_norm_applied": total_update_applied / iters,
        "projection_retained_fraction": total_retained / iters,
        "policy_anchor_mode": policy_anchor_mode,
        "policy_anchor_beta": (
            anchor_sums["beta"] / iters if anchor_enabled else None
        ),
        "policy_anchor_beta_unclipped": (
            anchor_sums["beta_unclipped"] / iters if anchor_enabled else None
        ),
        "policy_anchor_beta_clipped": (
            anchor_sums["beta_clipped"] / iters if anchor_enabled else None
        ),
        "policy_anchor_objective_grad_norm": (
            anchor_sums["objective_grad_norm"] / iters
            if anchor_enabled else None
        ),
        "policy_anchor_raw_grad_norm": (
            anchor_sums["raw_anchor_grad_norm"] / iters
            if anchor_enabled else None
        ),
        "policy_anchor_applied_grad_norm": (
            anchor_sums["applied_anchor_grad_norm"] / iters
            if anchor_enabled else None
        ),
        "policy_anchor_achieved_ratio": (
            anchor_sums["achieved_ratio"] / iters
            if anchor_enabled else None
        ),
        "policy_anchor_objective_cosine": (
            anchor_sums["objective_anchor_cosine"] / anchor_cosine_count
            if anchor_cosine_count else None
        ),
        "policy_anchor_ema_objective_grad_norm": (
            anchor_sums["ema_objective_grad_norm"] / iters
            if anchor_enabled else None
        ),
        "policy_anchor_ema_raw_grad_norm": (
            anchor_sums["ema_raw_anchor_grad_norm"] / iters
            if anchor_enabled else None
        ),
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "backtracks": backtracks,
        "safety_loss_before": (
            safety_before_sum / iters if has_safety else None
        ),
        "safety_loss_after": (
            safety_after_sum / iters if has_safety else None
        ),
        "safety_loss_reference": safety_reference,
        "realized_kl": realized_kl_sum / iters,
        "realized_kl_k1": realized_kl_k1_sum / iters,
        "trust_log_ratio_clip_fraction": (
            log_ratio_clip_sum / iters if anchor_enabled else None
        ),
        "history_loss_before": history_before,
        "history_loss_after": history_after,
        "trust_probe_rows": len(probe_rows),
        "trust_probe_questions": len({row.pid for row in probe_rows}),
        "boundary_gate_passed": (
            boundary_failure_fraction <= trust_boundary_failure_ceiling
        ),
        "inner_step_diagnostics": inner_step_diagnostics,
        "diagnostics_level": diagnostics_level,
        "diagnostic_probe_elapsed_seconds": diagnostic_probe_elapsed_seconds,
        "candidate_utility_elapsed_seconds": (
            candidate_utility_elapsed_seconds
        ),
    }


def _empirical_h_kl(
    model,
    tok,
    rows: list[L2RTrace],
    micro: int,
    *,
    scope: str = "generator",
) -> float:
    """Question-balanced signed k1 movement diagnostic on the supplied support."""

    if not rows:
        return float("nan")
    ids, h_mask, a_mask = _pad(rows, tok.eos_token_id)
    mask = _l2r_anchor_mask(h_mask, a_mask, scope)
    current = seq_logprobs(model, ids, mask, micro=micro, length_norm=True)
    with model.disable_adapter():
        base = seq_logprobs(model, ids, mask, micro=micro, length_norm=True)
    weights = _question_balanced_weights(rows, current.device)
    return float((weights * (current - base)).sum())


def _empirical_h_kl_nonnegative(
    model,
    tok,
    rows: list[L2RTrace],
    micro: int,
    *,
    scope: str = "generator",
) -> float:
    """Question-balanced token-level k3 surrogate on the supplied support."""

    if not rows:
        return float("nan")
    ids, h_mask, a_mask = _pad(rows, tok.eos_token_id)
    mask = _l2r_anchor_mask(h_mask, a_mask, scope)
    k3_rows = _tokenwise_h_k3_rows(
        model,
        ids,
        mask,
        micro,
        grad=False,
    )
    weights = _question_balanced_weights(rows, k3_rows.device)
    return float((weights * k3_rows).sum())


def _validate_l2r_run_config(
    config: L2RRunConfig,
    *,
    task,
    diagnostics_fn,
    diagnostics_probe_fn,
    state_checkpoint_fn,
    resume_state,
) -> L2RRunConfig:
    rounds = config.rounds
    B = config.B
    G = config.G
    iters = config.iters
    micro = config.micro
    reader_mode = config.reader_mode
    gold_in_buffer = config.gold_in_buffer
    l2r_buffer_semantics = config.l2r_buffer_semantics
    proposal_prompt = config.proposal_prompt
    proposal_mixture = config.proposal_mixture
    proposal_prior_fraction = config.proposal_prior_fraction
    proposal_temperature = config.proposal_temperature
    trace_segmentation = config.trace_segmentation
    answer_event_mode = config.answer_event_mode
    answer_target_termination = config.answer_target_termination
    responsibility_score = config.responsibility_score
    responsibility_temperature = config.responsibility_temperature
    responsibility_projection = config.responsibility_projection
    responsibility_ess_floor = config.responsibility_ess_floor
    responsibility_max_weight = config.responsibility_max_weight
    mstep_objective = config.mstep_objective
    archive_limit = config.archive_limit
    replay_limit = config.replay_limit
    adaptive_max_g = config.adaptive_max_g
    adaptive_batch_g = config.adaptive_batch_g
    adaptive_min_correct = config.adaptive_min_correct
    reader_decode_filter = config.reader_decode_filter
    kl_coef = config.kl_coef
    policy_anchor_mode = config.policy_anchor_mode
    policy_anchor_target_ratio = config.policy_anchor_target_ratio
    policy_anchor_beta_min = config.policy_anchor_beta_min
    policy_anchor_beta_max = config.policy_anchor_beta_max
    policy_anchor_ema = config.policy_anchor_ema
    policy_anchor_scope = config.policy_anchor_scope
    trust_kl_budget = config.trust_kl_budget
    trust_safety_questions = config.trust_safety_questions
    trust_safety_tolerance = config.trust_safety_tolerance
    trust_boundary_failure_ceiling = config.trust_boundary_failure_ceiling
    trust_max_backtracks = config.trust_max_backtracks
    trust_backtrack_shrink = config.trust_backtrack_shrink
    historical_replay_fraction = config.historical_replay_fraction
    lora_r = config.lora_r
    lora_alpha = config.lora_alpha
    lora_trainable = config.lora_trainable
    gradient_projection = config.gradient_projection
    gradient_projection_rank = config.gradient_projection_rank
    buffer_replicates = config.buffer_replicates
    question_schedule = config.question_schedule
    schedule_exploration = config.schedule_exploration
    eval_every = config.eval_every
    eval_rounds = config.eval_rounds
    diagnostics_level = config.diagnostics_level
    diagnostics_gradient_questions = config.diagnostics_gradient_questions
    candidate_utility_questions = config.candidate_utility_questions
    candidate_utility_batch = config.candidate_utility_batch
    checkpoint_every = config.checkpoint_every
    state_checkpoint_every = config.state_checkpoint_every
    resume_fingerprint = config.resume_fingerprint

    if rounds < 1:
        raise ValueError(f"rounds must be positive, got {rounds}")
    if B < 1 or G < 1 or B % G:
        raise ValueError(f"B must be positive and divisible by G, got B={B}, G={G}")
    if iters < 1 or micro < 1:
        raise ValueError("iters and micro must be positive")
    if reader_mode not in READER_MODES:
        raise ValueError(f"unknown L2R reader mode {reader_mode!r}")
    if l2r_buffer_semantics not in L2R_BUFFER_SEMANTICS:
        raise ValueError(
            f"unknown L2R buffer semantics {l2r_buffer_semantics!r}"
        )
    if proposal_prompt not in PROPOSAL_PROMPTS:
        raise ValueError(
            f"unknown L2R proposal prompt {proposal_prompt!r}; expected one of "
            f"{sorted(PROPOSAL_PROMPTS)}"
        )
    if proposal_mixture not in L2R_PROPOSAL_MIXTURES:
        raise ValueError(
            f"unknown L2R proposal mixture {proposal_mixture!r}; expected one "
            f"of {sorted(L2R_PROPOSAL_MIXTURES)}"
        )
    if not math.isfinite(proposal_prior_fraction) or not (
        0 < proposal_prior_fraction <= 1
    ):
        raise ValueError("proposal_prior_fraction must be in (0, 1]")
    if not math.isfinite(proposal_temperature) or proposal_temperature <= 0:
        raise ValueError("proposal_temperature must be finite and positive")
    if proposal_mixture == "question_answer":
        _stratified_proposal_counts(G, proposal_prior_fraction)
        if proposal_prompt == "question":
            raise ValueError(
                "question_answer proposal mixtures require an "
                "answer-conditioned proposal_prompt"
            )
    elif proposal_mixture == "question_temperature":
        _stratified_proposal_counts(G, proposal_prior_fraction)
        if proposal_prompt != "question":
            raise ValueError(
                "question_temperature mixtures require proposal_prompt='question'"
            )
        if proposal_temperature <= 1:
            raise ValueError(
                "question_temperature mixtures require proposal_temperature > 1"
            )
    elif proposal_temperature != 1.0:
        raise ValueError(
            "proposal_temperature differs from one only for question_temperature mixtures"
        )
    if trace_segmentation not in TRACE_SEGMENTATION_MODES:
        raise ValueError(f"unknown trace segmentation mode {trace_segmentation!r}")
    if answer_event_mode not in ANSWER_EVENT_MODES:
        raise ValueError(f"unknown answer event mode {answer_event_mode!r}")
    if answer_target_termination not in ANSWER_TARGET_TERMINATIONS:
        raise ValueError(
            "unknown answer target termination "
            f"{answer_target_termination!r}"
        )
    task_answer_event_mode = getattr(task, "answer_event_mode", answer_event_mode)
    if task_answer_event_mode != answer_event_mode:
        raise ValueError(
            "L2R and task answer-event modes must match, got "
            f"{answer_event_mode!r} and {task_answer_event_mode!r}"
        )
    if responsibility_score not in RESPONSIBILITY_SCORES:
        raise ValueError(f"unknown L2R responsibility score {responsibility_score!r}")
    if mstep_objective not in MSTEP_OBJECTIVES:
        raise ValueError(f"unknown L2R M-step objective {mstep_objective!r}")
    if responsibility_projection not in RESPONSIBILITY_PROJECTIONS:
        raise ValueError(
            f"unknown responsibility projection {responsibility_projection!r}"
        )
    if not math.isfinite(responsibility_temperature) or responsibility_temperature <= 0:
        raise ValueError("responsibility_temperature must be finite and positive")
    if not math.isfinite(responsibility_ess_floor) or not (
        0 <= responsibility_ess_floor <= 1
    ):
        raise ValueError("responsibility_ess_floor must be in [0, 1]")
    if not math.isfinite(responsibility_max_weight) or not (
        0 < responsibility_max_weight <= 1
    ):
        raise ValueError("responsibility_max_weight must be in (0, 1]")
    if responsibility_projection == "none" and (
        responsibility_ess_floor > 0 or responsibility_max_weight < 1
    ):
        raise ValueError(
            "responsibility constraints require responsibility_projection='safe_set'"
        )
    if archive_limit < 0 or replay_limit < 0:
        raise ValueError("archive_limit and replay_limit must be nonnegative")
    if archive_limit > 0 and replay_limit > archive_limit:
        raise ValueError("replay_limit cannot exceed archive_limit")
    if adaptive_max_g == 0:
        adaptive_max_g = G
    if adaptive_max_g < G:
        raise ValueError("adaptive_max_g must be zero or at least G")
    if adaptive_batch_g < 1 or adaptive_min_correct < 1:
        raise ValueError("adaptive_batch_g and adaptive_min_correct must be positive")
    if not math.isfinite(kl_coef) or kl_coef < 0:
        raise ValueError("kl_coef must be finite and nonnegative")
    if policy_anchor_mode not in POLICY_ANCHOR_MODES:
        raise ValueError(f"unknown policy anchor mode {policy_anchor_mode!r}")
    if policy_anchor_scope not in POLICY_ANCHOR_SCOPES:
        raise ValueError(
            f"unknown L2R policy anchor scope {policy_anchor_scope!r}"
        )
    if policy_anchor_mode == "grad_ratio":
        if kl_coef:
            raise ValueError("grad_ratio anchoring and kl_coef are mutually exclusive")
        if policy_anchor_target_ratio is None or not math.isfinite(
            policy_anchor_target_ratio
        ) or policy_anchor_target_ratio < 0:
            raise ValueError(
                "grad_ratio anchoring requires a finite nonnegative target ratio"
            )
    elif policy_anchor_target_ratio is not None:
        raise ValueError(
            "policy_anchor_target_ratio requires policy_anchor_mode='grad_ratio'"
        )
    for name, value in (
        ("policy_anchor_beta_min", policy_anchor_beta_min),
        ("policy_anchor_beta_max", policy_anchor_beta_max),
        ("trust_safety_tolerance", trust_safety_tolerance),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if policy_anchor_beta_min > policy_anchor_beta_max:
        raise ValueError("policy_anchor_beta_min cannot exceed policy_anchor_beta_max")
    if not math.isfinite(policy_anchor_ema) or not 0 <= policy_anchor_ema < 1:
        raise ValueError("policy_anchor_ema must be in [0, 1)")
    if trust_kl_budget is not None and (
        not math.isfinite(trust_kl_budget) or trust_kl_budget < 0
    ):
        raise ValueError("trust_kl_budget must be finite and nonnegative")
    if trust_safety_questions < 0:
        raise ValueError("trust_safety_questions must be nonnegative")
    if not math.isfinite(trust_boundary_failure_ceiling) or not (
        0 <= trust_boundary_failure_ceiling <= 1
    ):
        raise ValueError("trust_boundary_failure_ceiling must be in [0, 1]")
    if trust_max_backtracks < 0:
        raise ValueError("trust_max_backtracks must be nonnegative")
    if not math.isfinite(trust_backtrack_shrink) or not (
        0 < trust_backtrack_shrink < 1
    ):
        raise ValueError("trust_backtrack_shrink must be in (0, 1)")
    if not math.isfinite(historical_replay_fraction) or not (
        0 <= historical_replay_fraction < 1
    ):
        raise ValueError("historical_replay_fraction must be in [0, 1)")
    if policy_anchor_mode == "grad_ratio" and trust_safety_questions < 1:
        raise ValueError("grad_ratio trust control requires a non-empty safety set")
    if lora_r < 1:
        raise ValueError("lora_r must be positive")
    if lora_alpha < 1:
        raise ValueError("lora_alpha must be positive")
    if lora_trainable not in LORA_TRAINABLE_MODES:
        raise ValueError(f"unknown LoRA trainable mode {lora_trainable!r}")
    if gradient_projection not in GRADIENT_PROJECTION_MODES:
        raise ValueError(f"unknown gradient projection mode {gradient_projection!r}")
    if gradient_projection == "none" and gradient_projection_rank != 0:
        raise ValueError("gradient_projection_rank must be zero when projection is disabled")
    if gradient_projection != "none" and gradient_projection_rank < 1:
        raise ValueError("projected gradients require a positive projection rank")
    if buffer_replicates < 1 or G % buffer_replicates:
        raise ValueError("buffer_replicates must be positive and divide G")
    if (
        buffer_replicates > 1
        and not gold_in_buffer
        and replay_limit > 0
        and replay_limit < buffer_replicates
    ):
        raise ValueError(
            "without a shared gold trace, replay_limit must retain at least "
            "one sampled trace per buffer replica"
        )
    if adaptive_max_g % buffer_replicates:
        raise ValueError("adaptive_max_g must be divisible by buffer_replicates")
    if responsibility_score in {
        "prior_corrected",
        "mixed_prior_corrected",
    }:
        expected_objective = (
            "joint" if reader_mode == "moving" else "generator"
        )
        if mstep_objective != expected_objective:
            raise ValueError(
                "prior-corrected L2R requires a joint M-step for the moving "
                "reader and a generator-only M-step for the frozen reader"
            )
        if l2r_buffer_semantics != "fresh_multiset":
            raise ValueError(
                "prior-corrected L2R requires fresh_multiset buffer semantics"
            )
        if gold_in_buffer:
            raise ValueError(
                "prior-corrected L2R cannot insert a deterministic gold trace"
            )
        if responsibility_score == "prior_corrected":
            if proposal_mixture != "single":
                raise ValueError(
                    "prior_corrected L2R requires proposal_mixture='single'"
                )
            if proposal_prompt != "question":
                raise ValueError(
                    "prior-corrected L2R currently requires the canonical "
                    "answer-blind prior proposal"
                )
            if proposal_prior_fraction != 1.0:
                raise ValueError(
                    "single prior-corrected proposals require "
                    "proposal_prior_fraction=1"
                )
        else:
            if proposal_mixture not in {"question_answer", "question_temperature"}:
                raise ValueError(
                    "mixed_prior_corrected L2R requires "
                    "a supported two-component proposal mixture"
                )
            if (
                proposal_mixture == "question_answer"
                and proposal_prompt not in {"answer_derive", "answer_derive_first"}
            ):
                raise ValueError(
                    "mixed_prior_corrected L2R requires an "
                    "answer-conditioned proposal prompt"
                )
        if archive_limit != 0 or replay_limit != 0:
            raise ValueError(
                "fresh prior-corrected draws require archive_limit=0 and "
                "replay_limit=0"
            )
        if historical_replay_fraction:
            raise ValueError(
                "prior-corrected L2R cannot replay traces sampled by an "
                "earlier policy"
            )
        if adaptive_max_g != G:
            raise ValueError(
                "prior-corrected L2R requires a fixed proposal count per question"
            )
        if reader_decode_filter:
            raise ValueError(
                "prior-corrected L2R cannot filter proposal draws with reader decoding"
            )
        if buffer_replicates != 1:
            raise ValueError(
                "prior-corrected L2R currently requires one draw-level multiset"
            )
        anchor_requested = (
            kl_coef > 0 or policy_anchor_mode == "grad_ratio"
        )
        if anchor_requested and policy_anchor_scope != "generator_and_reader":
            raise ValueError(
                "anchored prior-corrected L2R must anchor generation and "
                "answer-reader contexts"
            )
    if question_schedule not in QUESTION_SCHEDULES:
        raise ValueError(f"unknown question schedule {question_schedule!r}")
    if not math.isfinite(schedule_exploration) or not 0 <= schedule_exploration <= 1:
        raise ValueError("schedule_exploration must be in [0, 1]")
    if eval_every and eval_rounds:
        raise ValueError("eval_every and eval_rounds are mutually exclusive")
    diagnostics_level = validate_diagnostic_level(diagnostics_level)
    if diagnostics_gradient_questions < 0:
        raise ValueError("diagnostics_gradient_questions must be nonnegative")
    if diagnostics_gradient_questions and diagnostics_level != "deep":
        raise ValueError(
            "diagnostics_gradient_questions requires diagnostics_level='deep'"
        )
    if diagnostics_probe_fn is not None and diagnostics_level != "deep":
        raise ValueError("diagnostics_probe_fn requires diagnostics_level='deep'")
    if candidate_utility_questions < 0:
        raise ValueError("candidate_utility_questions must be nonnegative")
    if candidate_utility_batch < 1:
        raise ValueError("candidate_utility_batch must be positive")
    if candidate_utility_questions and diagnostics_level != "deep":
        raise ValueError(
            "candidate utility audit requires diagnostics_level='deep'"
        )
    if (
        candidate_utility_questions
        and hasattr(task, "train_partition")
        and task.train_partition != "train"
    ):
        raise ValueError(
            "candidate utility audit requires train_partition='train' so its "
            "reserve is disjoint from GSM8K validation"
        )
    if diagnostics_fn is None and (
        diagnostics_level != "standard"
        or diagnostics_gradient_questions
        or diagnostics_probe_fn is not None
        or candidate_utility_questions
    ):
        raise ValueError("deep L2R diagnostics require diagnostics_fn")
    if checkpoint_every < 0 or state_checkpoint_every < 0:
        raise ValueError("checkpoint intervals must be nonnegative")
    if state_checkpoint_every and state_checkpoint_fn is None:
        raise ValueError(
            "state_checkpoint_every requires a state_checkpoint_fn"
        )
    if state_checkpoint_fn is not None and not resume_fingerprint:
        raise ValueError(
            "state checkpointing requires a non-empty resume_fingerprint"
        )
    if resume_state is not None and not resume_fingerprint:
        raise ValueError("resume_state requires a non-empty resume_fingerprint")

    return replace(
        config,
        adaptive_max_g=adaptive_max_g,
        diagnostics_level=diagnostics_level,
    )


def _prepare_l2r_round_support(
    *,
    config: L2RRunConfig,
    state: _L2RRuntimeState,
    task,
    round_index: int,
) -> _L2RRoundOutcome:
    B = config.B
    G = config.G
    seed = config.seed
    gold_in_buffer = config.gold_in_buffer
    l2r_buffer_semantics = config.l2r_buffer_semantics
    proposal_prompt = config.proposal_prompt
    proposal_mixture = config.proposal_mixture
    proposal_prior_fraction = config.proposal_prior_fraction
    proposal_temperature = config.proposal_temperature
    trace_segmentation = config.trace_segmentation
    answer_event_mode = config.answer_event_mode
    answer_target_termination = config.answer_target_termination
    archive_limit = config.archive_limit
    adaptive_max_g = config.adaptive_max_g
    adaptive_batch_g = config.adaptive_batch_g
    adaptive_min_correct = config.adaptive_min_correct
    historical_replay_fraction = config.historical_replay_fraction
    buffer_replicates = config.buffer_replicates
    model = state.model
    tok = state.tok
    rng = state.rng
    prompt_ids = state.prompt_ids
    buffers = state.buffers
    seen_questions = state.seen_questions
    total_gen = state.total_gen
    total_generated_tokens = state.total_generated_tokens
    total_evictions = state.total_evictions
    total_duplicates = state.total_duplicates
    question_exposures = state.question_exposures
    scheduler = state.scheduler
    cache_stats = state.cache_stats

    cache_before = dict(cache_stats)
    n_questions = B // G
    selected_pids = scheduler.select(n_questions, round_index)
    schedule_diagnostics = scheduler.diagnostics(selected_pids, round_index)
    if proposal_mixture == "question_answer":
        pids, sampled_rows, sampled_texts, generation = (
            _sample_mixed_round(
                model,
                tok,
                task,
                prompt_ids,
                B=B,
                G=G,
                rng=rng,
                round_added=round_index,
                proposal_prompt=proposal_prompt,
                proposal_prior_fraction=proposal_prior_fraction,
                selected_pids=selected_pids,
                trace_segmentation=trace_segmentation,
                answer_event_mode=answer_event_mode,
                answer_target_termination=answer_target_termination,
            )
        )
    elif proposal_mixture == "question_temperature":
        pids, sampled_rows, sampled_texts, generation = (
            _sample_temperature_mixed_round(
                model,
                tok,
                task,
                prompt_ids,
                B=B,
                G=G,
                rng=rng,
                round_added=round_index,
                proposal_prior_fraction=proposal_prior_fraction,
                proposal_temperature=proposal_temperature,
                selected_pids=selected_pids,
                trace_segmentation=trace_segmentation,
                answer_event_mode=answer_event_mode,
                answer_target_termination=answer_target_termination,
            )
        )
    else:
        pids, sampled_rows, sampled_texts, generation = _sample_round(
            model,
            tok,
            task,
            prompt_ids,
            B=B,
            G=G,
            rng=rng,
            round_added=round_index,
            adaptive_max_g=adaptive_max_g,
            adaptive_batch_g=adaptive_batch_g,
            adaptive_min_correct=adaptive_min_correct,
            buffer_replicates=buffer_replicates,
            selected_pids=selected_pids,
            trace_segmentation=trace_segmentation,
            answer_event_mode=answer_event_mode,
            answer_target_termination=answer_target_termination,
            proposal_prompt=proposal_prompt,
        )
    total_gen += generation["generations"]
    total_generated_tokens += generation["generated_tokens"]
    seen_questions.update(pids)
    question_exposures += len(pids)

    gold_added = sampled_added = duplicate_rows = evictions = 0
    if l2r_buffer_semantics == "fresh_multiset":
        buffers = _fresh_multiset_buffers(pids, sampled_rows)
        sampled_added = len(sampled_rows)
    else:
        if gold_in_buffer:
            for pid in pids:
                if any(row.is_gold for row in buffers.get(pid, [])):
                    continue
                row = _gold_trace(
                    tok,
                    task,
                    prompt_ids[pid],
                    pid,
                    round_index,
                    answer_event_mode=answer_event_mode,
                    answer_target_termination=answer_target_termination,
                )
                if row is not None and _append_unique(buffers, row):
                    gold_added += 1
        for row in sampled_rows:
            if _append_unique(buffers, row):
                sampled_added += 1
            else:
                duplicate_rows += 1
        for pid in pids:
            evictions += _prune_archive(
                buffers.get(pid, []),
                archive_limit,
                buffer_replicates,
            )
    total_duplicates += duplicate_rows
    total_evictions += evictions

    replay_target = int(round(len(pids) * historical_replay_fraction))
    historical_candidates = sorted(set(buffers) - set(pids))
    replay_count = min(replay_target, len(historical_candidates))
    replay_pids = (
        [
            int(pid)
            for pid in rng.choice(
                historical_candidates,
                size=replay_count,
                replace=False,
            )
        ]
        if replay_count
        else []
    )
    monitor_count = min(len(pids), len(historical_candidates))
    monitor_rng = np.random.default_rng(
        seed * 1_000_003 + round_index * 9_176 + 41
    )
    history_monitor_pids = (
        [
            int(pid)
            for pid in monitor_rng.choice(
                historical_candidates,
                size=monitor_count,
                replace=False,
            )
        ]
        if monitor_count else []
    )
    history_rows = _history_monitor_rows(
        buffers,
        history_monitor_pids,
    )
    current_update_count = len(pids) - replay_count
    current_update_pids = (
        [
            int(pid)
            for pid in rng.choice(
                pids,
                size=current_update_count,
                replace=False,
            )
        ]
        if current_update_count < len(pids)
        else list(pids)
    )
    update_pids = current_update_pids + replay_pids
    rows = [row for pid in update_pids for row in buffers.get(pid, [])]
    ids, h_mask, a_mask = _pad(rows, tok.eos_token_id)
    answer_proposal_h = None

    state.buffers = buffers
    state.total_gen = total_gen
    state.total_generated_tokens = total_generated_tokens
    state.total_evictions = total_evictions
    state.total_duplicates = total_duplicates
    state.question_exposures = question_exposures
    return _L2RRoundOutcome(
        cache_before=cache_before,
        pids=pids,
        sampled_rows=sampled_rows,
        sampled_texts=sampled_texts,
        generation=generation,
        schedule_diagnostics=schedule_diagnostics,
        gold_added=gold_added,
        sampled_added=sampled_added,
        duplicate_rows=duplicate_rows,
        evictions=evictions,
        replay_pids=replay_pids,
        history_monitor_pids=history_monitor_pids,
        history_rows=history_rows,
        current_update_pids=current_update_pids,
        update_pids=update_pids,
        rows=rows,
        ids=ids,
        h_mask=h_mask,
        a_mask=a_mask,
        answer_proposal_h=answer_proposal_h,
    )


def _score_l2r_round_support(
    *,
    config: L2RRunConfig,
    state: _L2RRuntimeState,
    outcome: _L2RRoundOutcome,
    task,
) -> None:
    iters = config.iters
    micro = config.micro
    reader_mode = config.reader_mode
    proposal_prompt = config.proposal_prompt
    proposal_mixture = config.proposal_mixture
    proposal_prior_fraction = config.proposal_prior_fraction
    proposal_temperature = config.proposal_temperature
    responsibility_score = config.responsibility_score
    responsibility_temperature = config.responsibility_temperature
    responsibility_projection = config.responsibility_projection
    responsibility_ess_floor = config.responsibility_ess_floor
    responsibility_max_weight = config.responsibility_max_weight
    mstep_objective = config.mstep_objective
    replay_limit = config.replay_limit
    reader_decode_filter = config.reader_decode_filter
    buffer_replicates = config.buffer_replicates
    exact_cache = config.exact_cache
    model = state.model
    tok = state.tok
    buffers = state.buffers
    cache_stats = state.cache_stats
    total_reader_decode = state.total_reader_decode
    total_reader_decode_tokens = state.total_reader_decode_tokens
    replay_pids = outcome.replay_pids
    current_update_pids = outcome.current_update_pids
    update_pids = outcome.update_pids
    rows = outcome.rows
    ids = outcome.ids
    h_mask = outcome.h_mask
    a_mask = outcome.a_mask
    answer_proposal_h = outcome.answer_proposal_h

    with torch.no_grad():
        policy_h = seq_logprobs(
            model,
            ids,
            h_mask,
            micro=micro,
            length_norm=False,
        )
        reader_a = _reader_answer_logps(
            model,
            tok,
            rows,
            ids,
            a_mask,
            micro=micro,
            reader_mode=reader_mode,
            exact_cache=exact_cache,
            cache_stats=cache_stats,
        )
        if responsibility_score == "mixed_prior_corrected":
            if proposal_mixture == "question_answer":
                answer_proposal_h = _proposal_component_h_logps(
                    model,
                    tok,
                    task,
                    rows,
                    proposal_prompt=proposal_prompt,
                    micro=micro,
                )
            else:
                answer_proposal_h = _temperature_component_h_logps(
                    model,
                    ids,
                    h_mask,
                    temperature=proposal_temperature,
                    micro=micro,
                )
    h_lengths = h_mask.sum(1)
    a_lengths = a_mask.sum(1)
    trace_pids = torch.tensor([row.pid for row in rows], device=DEV, dtype=torch.long)
    if responsibility_score == "joint":
        raw_logits = policy_h + reader_a
    elif responsibility_score == "token_mean":
        raw_logits = (
            policy_h / h_lengths.clamp_min(1)
            + reader_a / a_lengths.clamp_min(1)
        )
    elif responsibility_score == "mixed_prior_corrected":
        mixture_logps = torch.logaddexp(
            policy_h + math.log(proposal_prior_fraction),
            answer_proposal_h + math.log1p(-proposal_prior_fraction),
        )
        raw_logits = policy_h + reader_a - mixture_logps
    else:
        raw_logits = reader_a
    counterfactual_invalid_mass = _counterfactual_invalid_mass(
        raw_logits,
        trace_pids,
        rows,
        temperature=responsibility_temperature,
    )

    active = torch.zeros(len(rows), device=DEV, dtype=torch.bool)
    offset = 0
    for pid in update_pids:
        local_rows = buffers.get(pid, [])
        local_logits = raw_logits[offset:offset + len(local_rows)]
        if buffer_replicates == 1:
            valid_indices = [
                index
                for index, row in enumerate(local_rows)
                if row.segmentation_valid
            ]
            valid_rows = [local_rows[index] for index in valid_indices]
            valid_logits = local_logits[
                torch.tensor(valid_indices, device=DEV, dtype=torch.long)
            ]
            selected = [
                valid_indices[index]
                for index in select_replay_indices(
                    valid_rows,
                    valid_logits,
                    replay_limit,
                )
            ]
        else:
            gold_indices = [
                index
                for index, row in enumerate(local_rows)
                if row.is_gold and row.segmentation_valid
            ]
            selected_set = set(gold_indices)
            sampled_budget = max(replay_limit - len(gold_indices), 0)
            base_limit, remainder = (
                (0, 0)
                if replay_limit <= 0
                else divmod(sampled_budget, buffer_replicates)
            )
            for replica in range(buffer_replicates):
                replica_indices = gold_indices + [
                    index
                    for index, row in enumerate(local_rows)
                    if (
                        not row.is_gold
                        and row.replica == replica
                        and row.segmentation_valid
                    )
                ]
                replica_rows = [local_rows[index] for index in replica_indices]
                replica_logits = local_logits[
                    torch.tensor(replica_indices, device=DEV)
                ]
                local_limit = (
                    0
                    if replay_limit <= 0
                    else (
                        len(gold_indices)
                        + base_limit
                        + int(replica < remainder)
                    )
                )
                selected_set.update(
                    replica_indices[index]
                    for index in select_replay_indices(
                        replica_rows,
                        replica_logits,
                        local_limit,
                    )
                )
            selected = sorted(selected_set)
        if selected:
            active[
                offset + torch.tensor(selected, device=DEV)
            ] = True
        offset += len(local_rows)

    decode_pass = None
    decode_fallback_questions = 0
    if reader_decode_filter:
        active_indices = torch.nonzero(active, as_tuple=False).flatten().cpu().tolist()
        decode_rows = [rows[index] for index in active_indices]
        decode_misses_before = cache_stats["reader_decode_misses"]
        decode_values, decode_tokens = _reader_decode_correct(
            model,
            tok,
            task,
            decode_rows,
            reader_mode=reader_mode,
            exact_cache=exact_cache,
            cache_stats=cache_stats,
        )
        total_reader_decode += (
            cache_stats["reader_decode_misses"] - decode_misses_before
            if exact_cache and reader_mode == "frozen"
            else len(decode_rows)
        )
        total_reader_decode_tokens += decode_tokens
        decode_pass = torch.zeros(len(rows), device=DEV, dtype=torch.bool)
        for index, passed in zip(active_indices, decode_values):
            decode_pass[index] = passed
        for pid in update_pids:
            local = (trace_pids == pid) & active
            if bool(local.any()) and not bool((local & decode_pass).any()):
                decode_pass[local] = True
                decode_fallback_questions += 1
        active &= decode_pass

    if buffer_replicates == 1:
        weights = l2r_responsibilities(
            policy_h,
            reader_a,
            h_lengths,
            a_lengths,
            trace_pids,
            score=responsibility_score,
            temperature=responsibility_temperature,
            active=active,
            answer_proposal_h_logps=answer_proposal_h,
            proposal_prior_fraction=proposal_prior_fraction,
        ).detach()
    else:
        replicas = torch.tensor(
            [row.replica for row in rows],
            device=DEV,
            dtype=torch.long,
        )
        is_gold = torch.tensor(
            [row.is_gold for row in rows],
            device=DEV,
            dtype=torch.bool,
        )
        weights = replicated_responsibilities(
            raw_logits,
            trace_pids,
            replicas,
            is_gold,
            active,
            replicate_count=buffer_replicates,
            temperature=responsibility_temperature,
        ).detach()
    projection = {
        "question_count": len(update_pids),
        "changed_questions": 0,
        "changed_fraction": 0.0,
        "mean_uniform_mix": 0.0,
        "max_uniform_mix": 0.0,
        "global_max_weight": (
            float(weights.max()) if len(weights) else None
        ),
    }
    if responsibility_projection == "safe_set":
        weights, projection = project_l2r_responsibilities(
            weights,
            trace_pids,
            active=active,
            ess_floor=responsibility_ess_floor,
            max_weight=responsibility_max_weight,
        )
        weights = weights.detach()
    for row, weight in zip(rows, weights.detach().cpu().tolist()):
        row.last_responsibility = float(weight)
    current_policy_backward_tokens = iters * sum(
        _trace_objective_tokens(row, mstep_objective)
        for row, weight in zip(rows, weights.detach().cpu().tolist())
        if weight > 0 and row.pid in current_update_pids
    )
    replay_policy_backward_tokens = iters * sum(
        _trace_objective_tokens(row, mstep_objective)
        for row, weight in zip(rows, weights.detach().cpu().tolist())
        if weight > 0 and row.pid in replay_pids
    )


    outcome.answer_proposal_h = answer_proposal_h
    outcome.policy_h = policy_h
    outcome.reader_a = reader_a
    outcome.trace_pids = trace_pids
    outcome.weights = weights
    outcome.counterfactual_invalid_mass = counterfactual_invalid_mass
    outcome.decode_fallback_questions = decode_fallback_questions
    outcome.projection = projection
    outcome.current_policy_backward_tokens = current_policy_backward_tokens
    outcome.replay_policy_backward_tokens = replay_policy_backward_tokens
    state.total_reader_decode = total_reader_decode
    state.total_reader_decode_tokens = total_reader_decode_tokens


def _run_l2r_round(
    *,
    config: L2RRunConfig,
    state: _L2RRuntimeState,
    task,
    round_index: int,
    eval_fn,
    diagnostics_fn,
    diagnostics_probe_fn,
    checkpoint_fn,
    state_checkpoint_fn,
    log,
) -> None:
    rounds = config.rounds
    B = config.B
    G = config.G
    seed = config.seed
    iters = config.iters
    micro = config.micro
    reader_mode = config.reader_mode
    gold_in_buffer = config.gold_in_buffer
    l2r_buffer_semantics = config.l2r_buffer_semantics
    proposal_prompt = config.proposal_prompt
    proposal_mixture = config.proposal_mixture
    proposal_prior_fraction = config.proposal_prior_fraction
    proposal_temperature = config.proposal_temperature
    trace_segmentation = config.trace_segmentation
    answer_target_termination = config.answer_target_termination
    responsibility_score = config.responsibility_score
    responsibility_temperature = config.responsibility_temperature
    responsibility_projection = config.responsibility_projection
    responsibility_ess_floor = config.responsibility_ess_floor
    responsibility_max_weight = config.responsibility_max_weight
    length_norm = config.length_norm
    mstep_objective = config.mstep_objective
    archive_limit = config.archive_limit
    replay_limit = config.replay_limit
    adaptive_max_g = config.adaptive_max_g
    adaptive_batch_g = config.adaptive_batch_g
    adaptive_min_correct = config.adaptive_min_correct
    reader_decode_filter = config.reader_decode_filter
    kl_coef = config.kl_coef
    policy_anchor_mode = config.policy_anchor_mode
    policy_anchor_target_ratio = config.policy_anchor_target_ratio
    policy_anchor_beta_min = config.policy_anchor_beta_min
    policy_anchor_beta_max = config.policy_anchor_beta_max
    policy_anchor_ema = config.policy_anchor_ema
    policy_anchor_scope = config.policy_anchor_scope
    trust_kl_budget = config.trust_kl_budget
    trust_safety_questions = config.trust_safety_questions
    trust_safety_tolerance = config.trust_safety_tolerance
    trust_boundary_failure_ceiling = config.trust_boundary_failure_ceiling
    trust_max_backtracks = config.trust_max_backtracks
    trust_backtrack_shrink = config.trust_backtrack_shrink
    historical_replay_fraction = config.historical_replay_fraction
    lora_r = config.lora_r
    lora_alpha = config.lora_alpha
    lora_seed = config.lora_seed
    lora_trainable = config.lora_trainable
    gradient_projection = config.gradient_projection
    gradient_projection_rank = config.gradient_projection_rank
    gradient_basis_path = config.gradient_basis_path
    gradient_projection_preserve_norm = config.gradient_projection_preserve_norm
    buffer_replicates = config.buffer_replicates
    question_schedule = config.question_schedule
    schedule_exploration = config.schedule_exploration
    eval_every = config.eval_every
    eval_rounds = config.eval_rounds
    diagnostics_level = config.diagnostics_level
    diagnostics_gradient_questions = config.diagnostics_gradient_questions
    candidate_utility_questions = config.candidate_utility_questions
    candidate_utility_batch = config.candidate_utility_batch
    checkpoint_every = config.checkpoint_every
    exact_cache = config.exact_cache
    state_checkpoint_every = config.state_checkpoint_every
    resume_fingerprint = config.resume_fingerprint
    model = state.model
    tok = state.tok
    named_trainable = state.named_trainable
    opt = state.opt
    gradient_projector = state.gradient_projector
    rng = state.rng
    prompt_ids = state.prompt_ids
    training_pids = state.training_pids
    safety_pids = state.safety_pids
    candidate_utility_pids = state.candidate_utility_pids
    training_question_count = state.training_question_count
    safety_rows = state.safety_rows
    candidate_utility_rows = state.candidate_utility_rows
    candidate_utility_probe_fn = state.candidate_utility_probe_fn
    buffers = state.buffers
    seen_questions = state.seen_questions
    records = state.records
    total_gen = state.total_gen
    total_generated_tokens = state.total_generated_tokens
    total_backward_tokens = state.total_backward_tokens
    total_steps = state.total_steps
    total_policy_backward_tokens = state.total_policy_backward_tokens
    total_anchor_backward_tokens = state.total_anchor_backward_tokens
    total_reader_decode_tokens = state.total_reader_decode_tokens
    total_evictions = state.total_evictions
    total_duplicates = state.total_duplicates
    total_reader_decode = state.total_reader_decode
    total_current_policy_backward_tokens = state.total_current_policy_backward_tokens
    total_replay_policy_backward_tokens = state.total_replay_policy_backward_tokens
    question_exposures = state.question_exposures
    scheduler = state.scheduler
    policy_anchor_state = state.policy_anchor_state
    cache_stats = state.cache_stats
    training_diagnostic_state = state.training_diagnostic_state

    outcome = _prepare_l2r_round_support(
        config=config,
        state=state,
        task=task,
        round_index=round_index,
    )
    cache_before = outcome.cache_before
    pids = outcome.pids
    sampled_rows = outcome.sampled_rows
    sampled_texts = outcome.sampled_texts
    generation = outcome.generation
    schedule_diagnostics = outcome.schedule_diagnostics
    gold_added = outcome.gold_added
    sampled_added = outcome.sampled_added
    duplicate_rows = outcome.duplicate_rows
    evictions = outcome.evictions
    replay_pids = outcome.replay_pids
    history_monitor_pids = outcome.history_monitor_pids
    history_rows = outcome.history_rows
    current_update_pids = outcome.current_update_pids
    update_pids = outcome.update_pids
    rows = outcome.rows
    ids = outcome.ids
    h_mask = outcome.h_mask
    a_mask = outcome.a_mask
    answer_proposal_h = outcome.answer_proposal_h
    buffers = state.buffers
    total_gen = state.total_gen
    total_generated_tokens = state.total_generated_tokens
    total_evictions = state.total_evictions
    total_duplicates = state.total_duplicates
    question_exposures = state.question_exposures

    _score_l2r_round_support(
        config=config,
        state=state,
        outcome=outcome,
        task=task,
    )
    answer_proposal_h = outcome.answer_proposal_h
    policy_h = outcome.policy_h
    reader_a = outcome.reader_a
    trace_pids = outcome.trace_pids
    weights = outcome.weights
    counterfactual_invalid_mass = outcome.counterfactual_invalid_mass
    decode_fallback_questions = outcome.decode_fallback_questions
    projection = outcome.projection
    current_policy_backward_tokens = outcome.current_policy_backward_tokens
    replay_policy_backward_tokens = outcome.replay_policy_backward_tokens
    total_reader_decode = state.total_reader_decode
    total_reader_decode_tokens = state.total_reader_decode_tokens

    segmentation = _segmentation_diagnostics(sampled_rows)
    boundary_failure_fraction = (
        1.0 - segmentation["valid_fraction"]
        if segmentation["valid_fraction"] is not None
        else 1.0
    )
    trust_probe_rows = [
        row for row in sampled_rows if row.segmentation_valid
    ]
    mstep = _mstep(
        model,
        tok,
        opt,
        rows,
        weights,
        iters=iters,
        micro=micro,
        length_norm=length_norm,
        mstep_objective=mstep_objective,
        kl_coef=kl_coef,
        gradient_projector=gradient_projector,
        named_parameters=named_trainable,
        policy_anchor_mode=policy_anchor_mode,
        policy_anchor_target_ratio=policy_anchor_target_ratio,
        policy_anchor_beta_min=policy_anchor_beta_min,
        policy_anchor_beta_max=policy_anchor_beta_max,
        policy_anchor_ema=policy_anchor_ema,
        policy_anchor_scope=policy_anchor_scope,
        policy_anchor_state=policy_anchor_state,
        trust_kl_budget=trust_kl_budget,
        safety_rows=(
            safety_rows if policy_anchor_mode == "grad_ratio" else None
        ),
        trust_safety_tolerance=trust_safety_tolerance,
        boundary_failure_fraction=boundary_failure_fraction,
        trust_boundary_failure_ceiling=trust_boundary_failure_ceiling,
        trust_max_backtracks=trust_max_backtracks,
        trust_backtrack_shrink=trust_backtrack_shrink,
        history_rows=history_rows,
        trust_probe_rows=(
            trust_probe_rows
            if policy_anchor_mode == "grad_ratio"
            else None
        ),
        diagnostics_enabled=diagnostics_fn is not None,
        diagnostics_level=diagnostics_level,
        diagnostics_gradient_questions=diagnostics_gradient_questions,
        diagnostics_probe_fn=diagnostics_probe_fn,
        candidate_utility_rows=(
            candidate_utility_rows
            if candidate_utility_questions else None
        ),
        candidate_utility_probe_fn=candidate_utility_probe_fn,
        diagnostic_state=training_diagnostic_state,
        exact_cache=exact_cache,
        cache_stats=cache_stats,
    )
    inner_step_diagnostics = mstep["inner_step_diagnostics"]
    total_steps += mstep["gradient_steps"]
    total_backward_tokens += mstep["backward_tokens"]
    total_policy_backward_tokens += mstep["policy_backward_tokens"]
    total_anchor_backward_tokens += mstep["anchor_backward_tokens"]
    total_current_policy_backward_tokens += current_policy_backward_tokens
    total_replay_policy_backward_tokens += replay_policy_backward_tokens
    active_rows = [
        row for row, weight in zip(rows, weights.detach().cpu().tolist()) if weight > 0
    ]
    empirical_h_kl = _empirical_h_kl(
        model,
        tok,
        active_rows,
        micro,
        scope=policy_anchor_scope,
    )
    empirical_h_kl_nonnegative = _empirical_h_kl_nonnegative(
        model,
        tok,
        active_rows,
        micro,
        scope=policy_anchor_scope,
    )
    posterior, top_traces = _posterior_diagnostics(
        rows,
        weights,
        trace_pids,
        policy_h,
        reader_a,
        answer_proposal_h=answer_proposal_h,
        proposal_prior_fraction=(
            proposal_prior_fraction
            if answer_proposal_h is not None else None
        ),
    )
    for pid in pids:
        sampled_for_pid = [row for row in sampled_rows if row.pid == pid]
        correct_rate = (
            float(np.mean([row.proposal_correct is True for row in sampled_for_pid]))
            if sampled_for_pid
            else 0.0
        )
        local_weights = weights[trace_pids == pid]
        positive = local_weights[local_weights > 0]
        if len(positive) <= 1:
            uncertainty = 1.0 if pid not in update_pids else 0.0
        else:
            local = positive / positive.sum()
            uncertainty = float(
                -(local * local.clamp_min(1e-30).log()).sum()
                / math.log(len(local))
            )
        scheduler.observe(
            pid,
            correct_rate=correct_rate,
            uncertainty=min(max(uncertainty, 0.0), 1.0),
            round_index=round_index,
        )
    schedule_after = scheduler.diagnostics(pids, round_index)
    test_acc = maybe_eval(
        model,
        round_index,
        rounds,
        eval_every,
        eval_fn,
        eval_rounds=eval_rounds,
    )

    proposal_correct = [row.proposal_correct is True for row in sampled_rows]
    format_diagnostics = _format_failure_diagnostics(sampled_rows)
    archive_rows = sum(len(buffer) for buffer in buffers.values())
    round_cache = _cache_delta(cache_stats, cache_before)
    record = {
        "round": round_index,
        "oracle": 0,
        "verifier_calls": (
            (total_gen if adaptive_max_g > G else 0) + total_reader_decode
        ),
        "diagnostic_verifier_calls": (
            (0 if adaptive_max_g > G else total_gen)
            + int(
                training_diagnostic_state[
                    "candidate_utility_probe_generations"
                ]
            )
        ),
        "gen": total_gen,
        "llm_gen": (
            total_gen
            + total_reader_decode
            + int(
                training_diagnostic_state[
                    "candidate_utility_probe_generations"
                ]
            )
        ),
        "rollout_generated_tokens": total_generated_tokens,
        "generated_tokens": (
            total_generated_tokens
            + total_reader_decode_tokens
            + int(
                training_diagnostic_state[
                    "candidate_utility_probe_generated_tokens"
                ]
            )
        ),
        "reader_decode_generations": total_reader_decode,
        "reader_decode_tokens": total_reader_decode_tokens,
        "backward_tokens": total_backward_tokens,
        "policy_backward_tokens": total_policy_backward_tokens,
        "anchor_backward_tokens": total_anchor_backward_tokens,
        "current_policy_backward_tokens": (
            total_current_policy_backward_tokens
        ),
        "replay_policy_backward_tokens": (
            total_replay_policy_backward_tokens
        ),
        "gsteps": total_steps,
        "question_exposures": question_exposures,
        "unique_questions_seen": len(seen_questions),
        "questions_this_round": len(pids),
        "update_questions_this_round": len(update_pids),
        "replay_questions_this_round": len(replay_pids),
        "replay_question_fraction": (
            len(replay_pids) / len(update_pids) if update_pids else 0.0
        ),
        "mean_g": generation["mean_g"],
        "adaptive_generations": generation["adaptive_generations"],
        "resolved_initial": generation["resolved_initial"],
        "resolved_final": generation["resolved_final"],
        "frac_correct": float(np.mean(proposal_correct)) if proposal_correct else float("nan"),
        "proposal_prompt": proposal_prompt,
        "proposal_mixture": proposal_mixture,
        "proposal_prior_fraction": proposal_prior_fraction,
        "proposal_temperature": proposal_temperature,
        "proposal_prior_generations": generation.get(
            "prior_generations",
            generation["generations"],
        ),
        "proposal_answer_conditioned_generations": generation.get(
            "answer_conditioned_generations",
            0,
        ),
        "proposal_temperature_generations": generation.get(
            "temperature_generations",
            0,
        ),
        "proposal_prior_posterior_mass": posterior.get(
            "proposal_prior_posterior_mass"
        ),
        "proposal_conditioned_posterior_mass": posterior.get(
            "proposal_conditioned_posterior_mass"
        ),
        "proposal_answer_minus_prior_logp": posterior.get(
            "proposal_answer_minus_prior_logp"
        ),
        "proposal_log_importance_correction": posterior.get(
            "proposal_log_importance_correction"
        ),
        "target_mentioned_before_final": (
            generation["target_mentioned_before_final_fraction"]
        ),
        "target_before_equation": (
            generation["target_before_equation_fraction"]
        ),
        "equation_fraction": generation["equation_fraction"],
        "fmt": task_format_rate(task, sampled_texts),
        "format_failure_count": format_diagnostics["failure_count"],
        "correct_without_marker_count": (
            format_diagnostics["correct_without_marker_count"]
        ),
        "incorrect_without_marker_count": (
            format_diagnostics["incorrect_without_marker_count"]
        ),
        "multiple_marker_count": format_diagnostics["multiple_marker_count"],
        "nonterminal_marker_count": (
            format_diagnostics["nonterminal_marker_count"]
        ),
        "segmentation_valid_fraction": segmentation["valid_fraction"],
        "segmentation_invalid_count": segmentation["invalid_count"],
        "segmentation_counterfactual_invalid_mass": (
            counterfactual_invalid_mass
        ),
        "gen_len": float(np.mean([row.generated_tokens for row in sampled_rows])),
        "archive_rows": archive_rows,
        "active_rows": posterior["active_rows"],
        "unique_rationales": posterior["unique_rationales"],
        "buffer_evictions": total_evictions,
        "duplicate_rows": total_duplicates,
        "gold_mass": posterior["gold_mass"],
        "model_correct_mass": posterior["model_correct_mass"],
        "draw_ess": posterior["draw_ess"],
        "ess": posterior["ess_fraction"],
        "unique_ess": posterior["unique_ess"],
        "unique_ess_fraction": posterior["unique_ess_fraction"],
        "responsibility_entropy": posterior["entropy"],
        "responsibility_max": posterior["max_weight"],
        "weight_length_corr": posterior["weight_length_corr"],
        "weighted_h_tokens": posterior["weighted_h_tokens"],
        "reader_mode": reader_mode,
        "l2r_buffer_semantics": l2r_buffer_semantics,
        "exact_cache": exact_cache,
        "cache_saved_forward_rows": round_cache["saved_forward_rows"],
        "cache_saved_forward_tokens": round_cache["saved_forward_tokens"],
        "cache_reader_score_hits": round_cache["reader_score_hits"],
        "cache_reader_score_misses": round_cache["reader_score_misses"],
        "cache_reader_decode_hits": round_cache["reader_decode_hits"],
        "cache_reader_decode_misses": round_cache["reader_decode_misses"],
        "cache_base_token_hits": round_cache["base_token_hits"],
        "cache_base_token_misses": round_cache["base_token_misses"],
        "cache_safety_nll_hits": round_cache["safety_nll_hits"],
        "cache_safety_nll_misses": round_cache["safety_nll_misses"],
        "responsibility_score": responsibility_score,
        "responsibility_projection": responsibility_projection,
        "responsibility_ess_floor": responsibility_ess_floor,
        "responsibility_max_weight": responsibility_max_weight,
        "responsibility_projection_changed_fraction": projection[
            "changed_fraction"
        ],
        "responsibility_uniform_mix": projection["mean_uniform_mix"],
        "length_norm": length_norm,
        "mstep_objective": mstep_objective,
        "trace_segmentation": trace_segmentation,
        "answer_event_mode": config.answer_event_mode,
        "archive_limit": archive_limit,
        "replay_limit": replay_limit,
        "buffer_replicates": buffer_replicates,
        "question_schedule": question_schedule,
        "schedule_priority_mean": schedule_diagnostics["selected_priority_mean"],
        "schedule_selected_unseen": schedule_diagnostics["selected_unseen"],
        "schedule_pool_unseen": schedule_diagnostics["pool_unseen"],
        "schedule_max_exposures": schedule_diagnostics["max_exposures"],
        "schedule_min_exposures": schedule_diagnostics["min_exposures"],
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_seed": lora_seed,
        "lora_trainable": lora_trainable,
        "gradient_projection": gradient_projection,
        "gradient_projection_rank": gradient_projection_rank,
        "gradient_projection_preserve_norm": gradient_projection_preserve_norm,
        "reader_decode_filter": reader_decode_filter,
        "decode_fallback_questions": decode_fallback_questions,
        "loss": mstep["loss"],
        "policy_loss": mstep["policy_loss"],
        "kl_penalty": mstep["kl_penalty"],
        "policy_anchor_mode": policy_anchor_mode,
        "policy_anchor_scope": policy_anchor_scope,
        "policy_anchor_target_ratio": policy_anchor_target_ratio,
        "policy_anchor_beta": mstep["policy_anchor_beta"],
        "policy_anchor_beta_unclipped": mstep[
            "policy_anchor_beta_unclipped"
        ],
        "policy_anchor_beta_clipped": mstep[
            "policy_anchor_beta_clipped"
        ],
        "policy_anchor_objective_grad_norm": mstep[
            "policy_anchor_objective_grad_norm"
        ],
        "policy_anchor_raw_grad_norm": mstep[
            "policy_anchor_raw_grad_norm"
        ],
        "policy_anchor_applied_grad_norm": mstep[
            "policy_anchor_applied_grad_norm"
        ],
        "policy_anchor_achieved_ratio": mstep[
            "policy_anchor_achieved_ratio"
        ],
        "policy_anchor_objective_cosine": mstep[
            "policy_anchor_objective_cosine"
        ],
        "trust_accepted_steps": mstep["accepted_steps"],
        "trust_rejected_steps": mstep["rejected_steps"],
        "trust_backtracks": mstep["backtracks"],
        "trust_log_ratio_clip_fraction": mstep[
            "trust_log_ratio_clip_fraction"
        ],
        "trust_safety_loss_before": mstep["safety_loss_before"],
        "trust_safety_loss_after": mstep["safety_loss_after"],
        "trust_safety_loss_reference": mstep[
            "safety_loss_reference"
        ],
        "trust_realized_kl": mstep["realized_kl"],
        "trust_realized_kl_k1": mstep["realized_kl_k1"],
        "trust_probe_rows": mstep["trust_probe_rows"],
        "trust_probe_questions": mstep["trust_probe_questions"],
        "candidate_utility_questions": candidate_utility_questions,
        "candidate_utility_probe_calls": int(
            training_diagnostic_state[
                "candidate_utility_probe_calls"
            ]
        ),
        "candidate_utility_probe_generations": int(
            training_diagnostic_state[
                "candidate_utility_probe_generations"
            ]
        ),
        "candidate_utility_probe_generated_tokens": int(
            training_diagnostic_state[
                "candidate_utility_probe_generated_tokens"
            ]
        ),
        "candidate_utility_evaluations": sum(
            bool(step["candidate_utility"]["evaluated"])
            for step in inner_step_diagnostics
        ),
        "candidate_utility_elapsed_seconds": mstep[
            "candidate_utility_elapsed_seconds"
        ],
        "history_monitor_loss_before": mstep[
            "history_loss_before"
        ],
        "history_monitor_loss_after": mstep[
            "history_loss_after"
        ],
        "trust_boundary_gate_passed": mstep["boundary_gate_passed"],
        "gradient_norm_raw": mstep["gradient_norm_raw"],
        "optimizer_update_norm_raw": mstep["optimizer_update_norm_raw"],
        "optimizer_update_norm_projected": (
            mstep["optimizer_update_norm_projected"]
        ),
        "optimizer_update_norm_applied": (
            mstep["optimizer_update_norm_applied"]
        ),
        "projection_retained_fraction": mstep["projection_retained_fraction"],
        "kl": empirical_h_kl,
        "kl_nonnegative": empirical_h_kl_nonnegative,
        "test_acc": test_acc,
    }
    records.append(record)

    if diagnostics_fn is not None:
        diagnostics_fn({
            "schema_version": 2,
            "method_family": "l2r",
            "diagnostics_level": diagnostics_level,
            "round": round_index,
            "completed_rounds": round_index + 1,
            "configuration": {
                "reader_mode": reader_mode,
                "gold_in_buffer": gold_in_buffer,
                "l2r_buffer_semantics": l2r_buffer_semantics,
                "proposal_prompt": proposal_prompt,
                "proposal_mixture": proposal_mixture,
                "proposal_prior_fraction": proposal_prior_fraction,
                "proposal_temperature": proposal_temperature,
                "trace_segmentation": trace_segmentation,
                "answer_event_mode": config.answer_event_mode,
                "answer_target_termination": answer_target_termination,
                "responsibility_score": responsibility_score,
                "responsibility_temperature": responsibility_temperature,
                "responsibility_projection": responsibility_projection,
                "responsibility_ess_floor": responsibility_ess_floor,
                "responsibility_max_weight": responsibility_max_weight,
                "length_norm": length_norm,
                "mstep_objective": mstep_objective,
                "archive_limit": archive_limit,
                "replay_limit": replay_limit,
                "adaptive_max_g": adaptive_max_g,
                "adaptive_batch_g": adaptive_batch_g,
                "adaptive_min_correct": adaptive_min_correct,
                "reader_decode_filter": reader_decode_filter,
                "kl_coef": kl_coef,
                "policy_anchor_mode": policy_anchor_mode,
                "policy_anchor_scope": policy_anchor_scope,
                "policy_anchor_target_ratio": policy_anchor_target_ratio,
                "policy_anchor_beta_min": policy_anchor_beta_min,
                "policy_anchor_beta_max": policy_anchor_beta_max,
                "policy_anchor_ema": policy_anchor_ema,
                "trust_kl_budget": trust_kl_budget,
                "trust_safety_questions": trust_safety_questions,
                "trust_safety_tolerance": trust_safety_tolerance,
                "trust_boundary_failure_ceiling": (
                    trust_boundary_failure_ceiling
                ),
                "trust_max_backtracks": trust_max_backtracks,
                "trust_backtrack_shrink": trust_backtrack_shrink,
                "historical_replay_fraction": historical_replay_fraction,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_seed": lora_seed,
                "lora_trainable": lora_trainable,
                "gradient_projection": gradient_projection,
                "gradient_projection_rank": gradient_projection_rank,
                "gradient_basis_path": gradient_basis_path,
                "gradient_projection_preserve_norm": (
                    gradient_projection_preserve_norm
                ),
                "buffer_replicates": buffer_replicates,
                "question_schedule": question_schedule,
                "schedule_exploration": schedule_exploration,
                "diagnostics_level": diagnostics_level,
                "diagnostics_gradient_questions": (
                    diagnostics_gradient_questions
                ),
                "candidate_utility_questions": (
                    candidate_utility_questions
                ),
                "candidate_utility_batch": candidate_utility_batch,
                "optimization_questions": training_question_count,
                "trust_safety_question_ids": [
                    (
                        int(task.train_qi[pid])
                        if hasattr(task, "train_qi")
                        else int(pid)
                    )
                    for pid in safety_pids
                ],
                "candidate_utility_question_ids": [
                    (
                        int(task.train_qi[pid])
                        if hasattr(task, "train_qi")
                        else int(pid)
                    )
                    for pid in candidate_utility_pids
                ],
                "exact_cache": exact_cache,
            },
            "cache": {
                "enabled": exact_cache,
                "round": round_cache,
                "cumulative": dict(cache_stats),
            },
            "generation": {
                **generation,
                "cumulative_generations": total_gen,
                "cumulative_generated_tokens": total_generated_tokens,
                "proposal_correct_fraction": (
                    float(np.mean(proposal_correct)) if proposal_correct else None
                ),
                "format_fraction": (
                    task_format_rate(task, sampled_texts)
                    if sampled_texts
                    else None
                ),
            },
            "sampled_traces": [
                {
                    "pid": row.pid,
                    "replica": row.replica,
                    "proposal_prompt": row.proposal_prompt,
                    "proposal_temperature": row.proposal_temperature,
                    "proposal_correct": row.proposal_correct,
                    "generated_tokens": row.generated_tokens,
                    "h_tokens": row.h_tokens,
                    "target_mentioned_before_final": (
                        row.target_mentioned_before_final
                    ),
                    "target_before_equation": row.target_before_equation,
                    "has_equation": row.has_equation,
                    "text": row.text,
                }
                for row in sampled_rows
            ],
            "format": format_diagnostics,
            "segmentation": segmentation,
            "segmentation_counterfactual_invalid_mass": (
                counterfactual_invalid_mass
            ),
            "responsibility_projection": projection,
            "trust_region": {
                "mode": policy_anchor_mode,
                "target_ratio": policy_anchor_target_ratio,
                "beta": mstep["policy_anchor_beta"],
                "beta_unclipped": mstep[
                    "policy_anchor_beta_unclipped"
                ],
                "beta_clipped": mstep["policy_anchor_beta_clipped"],
                "objective_grad_norm": mstep[
                    "policy_anchor_objective_grad_norm"
                ],
                "raw_anchor_grad_norm": mstep[
                    "policy_anchor_raw_grad_norm"
                ],
                "applied_anchor_grad_norm": mstep[
                    "policy_anchor_applied_grad_norm"
                ],
                "achieved_ratio": mstep[
                    "policy_anchor_achieved_ratio"
                ],
                "objective_anchor_cosine": mstep[
                    "policy_anchor_objective_cosine"
                ],
                "accepted_steps": mstep["accepted_steps"],
                "rejected_steps": mstep["rejected_steps"],
                "backtracks": mstep["backtracks"],
                "log_ratio_clip_fraction": mstep[
                    "trust_log_ratio_clip_fraction"
                ],
                "safety_loss_before": mstep["safety_loss_before"],
                "safety_loss_after": mstep["safety_loss_after"],
                "safety_loss_reference": mstep[
                    "safety_loss_reference"
                ],
                "realized_kl": mstep["realized_kl"],
                "realized_kl_k1": mstep["realized_kl_k1"],
                "probe_rows": mstep["trust_probe_rows"],
                "probe_questions": mstep["trust_probe_questions"],
                "boundary_gate_passed": mstep[
                    "boundary_gate_passed"
                ],
            },
            "inner_m_step": {
                "steps": inner_step_diagnostics,
                "attempted_steps": len(inner_step_diagnostics),
                "accepted_steps": sum(
                    step["status"] == "accepted"
                    for step in inner_step_diagnostics
                ),
                "rejected_steps": sum(
                    step["status"] == "rejected"
                    for step in inner_step_diagnostics
                ),
                "cumulative_accepted_steps": int(
                    training_diagnostic_state["accepted_steps"]
                ),
                "consecutive_rejections": int(
                    training_diagnostic_state["consecutive_rejections"]
                ),
                "rejected_attempt_elapsed_seconds": sum(
                    float(step["rejected_attempt_elapsed_seconds"])
                    for step in inner_step_diagnostics
                ),
            },
            "behavioural_utility": {
                "fixed_probe_baseline_accuracy": finite_or_none(
                    training_diagnostic_state["probe_baseline_accuracy"]
                ),
                "fixed_probe_latest_accuracy": finite_or_none(
                    training_diagnostic_state["probe_previous_accuracy"]
                ),
                "fixed_probe_evaluations_this_round": sum(
                    bool(step["behavioural_probe"]["evaluated"])
                    for step in inner_step_diagnostics
                ),
                "full_validation_accuracy": finite_or_none(test_acc),
                "full_validation_evaluated": finite_or_none(test_acc) is not None,
                "full_validation_policy": (
                    "configured_checkpoint_schedule"
                    if eval_fn is not None else "disabled"
                ),
            },
            "candidate_utility": {
                "enabled": bool(candidate_utility_questions),
                "partition": (
                    "training_derived_tuning_reserve"
                    if candidate_utility_questions else "disabled"
                ),
                "disjoint_from_optimization": bool(
                    candidate_utility_questions
                ),
                "disjoint_from_trust_safety": bool(
                    candidate_utility_questions
                ),
                "disjoint_from_validation": bool(
                    candidate_utility_questions
                    and getattr(task, "train_partition", None) == "train"
                ),
                "question_count": candidate_utility_questions,
                "question_ids": [
                    (
                        int(task.train_qi[pid])
                        if hasattr(task, "train_qi")
                        else int(pid)
                    )
                    for pid in candidate_utility_pids
                ],
                "baseline_accuracy": finite_or_none(
                    (
                        training_diagnostic_state.get(
                            "candidate_utility_baseline_probe"
                        )
                        or {}
                    ).get("accuracy")
                ),
                "latest_accuracy": finite_or_none(
                    (
                        training_diagnostic_state.get(
                            "candidate_utility_previous_probe"
                        )
                        or {}
                    ).get("accuracy")
                ),
                "evaluations_this_round": sum(
                    bool(step["candidate_utility"]["evaluated"])
                    for step in inner_step_diagnostics
                ),
                "probe_calls_cumulative": int(
                    training_diagnostic_state[
                        "candidate_utility_probe_calls"
                    ]
                ),
                "probe_generations_cumulative": int(
                    training_diagnostic_state[
                        "candidate_utility_probe_generations"
                    ]
                ),
                "probe_generated_tokens_cumulative": int(
                    training_diagnostic_state[
                        "candidate_utility_probe_generated_tokens"
                    ]
                ),
                "elapsed_seconds_this_round": mstep[
                    "candidate_utility_elapsed_seconds"
                ],
            },
            "historical_replay": {
                "current_pids": current_update_pids,
                "replay_pids": replay_pids,
                "monitor_pids": history_monitor_pids,
                "monitor_exposure_ages": [
                    round_index
                    - min(
                        row.round_added
                        for row in buffers.get(pid, [])
                    )
                    for pid in history_monitor_pids
                    if buffers.get(pid)
                ],
                "monitor_loss_before": mstep[
                    "history_loss_before"
                ],
                "monitor_loss_after": mstep[
                    "history_loss_after"
                ],
                "monitor_loss_delta": (
                    mstep["history_loss_after"]
                    - mstep["history_loss_before"]
                    if mstep["history_loss_before"] is not None
                    and mstep["history_loss_after"] is not None
                    else None
                ),
                "target_fraction": historical_replay_fraction,
                "realized_fraction": (
                    len(replay_pids) / len(update_pids)
                    if update_pids else 0.0
                ),
                "current_policy_backward_tokens": (
                    current_policy_backward_tokens
                ),
                "replay_policy_backward_tokens": (
                    replay_policy_backward_tokens
                ),
            },
            "buffer": {
                "gold_added": gold_added,
                "sampled_added": sampled_added,
                "duplicates": duplicate_rows,
                "evictions": evictions,
                "archive_rows": archive_rows,
                "active_rows": posterior["active_rows"],
            },
            "posterior": posterior,
            "reader_decode": {
                "enabled": reader_decode_filter,
                "calls_cumulative": total_reader_decode,
                "generated_tokens_cumulative": total_reader_decode_tokens,
                "fallback_questions": decode_fallback_questions,
            },
            "optimizer": {
                **{
                    key: value
                    for key, value in mstep.items()
                    if key not in {
                        "inner_step_diagnostics",
                        "diagnostics_level",
                        "diagnostic_probe_elapsed_seconds",
                    }
                },
                "gradient_steps_cumulative": total_steps,
                "backward_tokens_cumulative": total_backward_tokens,
                "policy_backward_tokens_cumulative": (
                    total_policy_backward_tokens
                ),
                "anchor_backward_tokens_cumulative": (
                    total_anchor_backward_tokens
                ),
                "empirical_h_kl": empirical_h_kl,
                "empirical_h_kl_nonnegative": (
                    empirical_h_kl_nonnegative
                ),
                "diagnostic_probe_elapsed_seconds": mstep[
                    "diagnostic_probe_elapsed_seconds"
                ],
            },
            "scheduler": {
                **schedule_diagnostics,
                "pool_unseen_after": schedule_after["pool_unseen"],
                "max_exposures_after": schedule_after["max_exposures"],
                "min_exposures_after": schedule_after["min_exposures"],
            },
            "top_traces": top_traces,
            "test_acc": None if not math.isfinite(test_acc) else test_acc,
        })

    if (
        state_checkpoint_fn is not None
        and state_checkpoint_every > 0
        and (round_index + 1) % state_checkpoint_every == 0
    ):
        state_checkpoint_fn({
            "schema_version": 1,
            "fingerprint": resume_fingerprint,
            "completed_rounds": round_index + 1,
            "exact_cache": bool(exact_cache),
            "model_training": bool(model.training),
            "trainable_parameters": _trainable_parameter_state(
                named_trainable
            ),
            "optimizer": copy.deepcopy(opt.state_dict()),
            "buffers": {
                int(pid): [_trace_state(row) for row in buffer]
                for pid, buffer in buffers.items()
            },
            "seen_questions": sorted(seen_questions),
            "records": list(records),
            "counters": {
                "total_gen": total_gen,
                "total_generated_tokens": total_generated_tokens,
                "total_backward_tokens": total_backward_tokens,
                "total_steps": total_steps,
                "total_policy_backward_tokens": (
                    total_policy_backward_tokens
                ),
                "total_anchor_backward_tokens": (
                    total_anchor_backward_tokens
                ),
                "total_reader_decode_tokens": (
                    total_reader_decode_tokens
                ),
                "total_evictions": total_evictions,
                "total_duplicates": total_duplicates,
                "total_reader_decode": total_reader_decode,
                "total_current_policy_backward_tokens": (
                    total_current_policy_backward_tokens
                ),
                "total_replay_policy_backward_tokens": (
                    total_replay_policy_backward_tokens
                ),
                "question_exposures": question_exposures,
            },
            "scheduler": scheduler.state_dict(),
            "policy_anchor_state": dict(policy_anchor_state),
            "training_diagnostic_state": dict(
                training_diagnostic_state
            ),
            "cache_stats": dict(cache_stats),
            "rng": _rng_state(rng),
        })

    if (
        checkpoint_fn is not None
        and checkpoint_every > 0
        and (round_index + 1) % checkpoint_every == 0
        and (round_index + 1) < rounds
    ):
        checkpoint_fn(model, round_index + 1)

    log(
        f"  [L2R-{reader_mode} r{round_index:>3}] "
        f"gen={total_gen:>6} tok={total_generated_tokens:>8} "
        f"correct={record['frac_correct']:.3f} "
        f"gold={record['gold_mass']:.3f} ESS={record['ess']:.3f} "
        f"KLh={empirical_h_kl:+.3f} H={posterior['active_rows']}/{archive_rows}"
    )


    state.buffers = buffers
    state.total_gen = total_gen
    state.total_generated_tokens = total_generated_tokens
    state.total_backward_tokens = total_backward_tokens
    state.total_steps = total_steps
    state.total_policy_backward_tokens = total_policy_backward_tokens
    state.total_anchor_backward_tokens = total_anchor_backward_tokens
    state.total_reader_decode_tokens = total_reader_decode_tokens
    state.total_evictions = total_evictions
    state.total_duplicates = total_duplicates
    state.total_reader_decode = total_reader_decode
    state.total_current_policy_backward_tokens = total_current_policy_backward_tokens
    state.total_replay_policy_backward_tokens = total_replay_policy_backward_tokens
    state.question_exposures = question_exposures


def run_l2r(
    task,
    rounds: int = 40,
    B: int = 64,
    G: int = 4,
    seed: int = 0,
    lr: float = 5e-5,
    iters: int = 4,
    model_name: str = MODEL_NAME,
    model_tok=None,
    micro: int = 4,
    reader_mode: str = "frozen",
    gold_in_buffer: bool = True,
    l2r_buffer_semantics: str = "set_archive",
    proposal_prompt: str = "question",
    proposal_mixture: str = "single",
    proposal_prior_fraction: float = 1.0,
    proposal_temperature: float = 1.0,
    trace_segmentation: str = "legacy",
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
    responsibility_score: str = "joint",
    responsibility_temperature: float = 1.0,
    responsibility_projection: str = "none",
    responsibility_ess_floor: float = 0.0,
    responsibility_max_weight: float = 1.0,
    length_norm: bool = False,
    mstep_objective: str = "generator",
    archive_limit: int = 64,
    replay_limit: int = 16,
    adaptive_max_g: int = 0,
    adaptive_batch_g: int = 2,
    adaptive_min_correct: int = 1,
    reader_decode_filter: bool = False,
    kl_coef: float = 0.0,
    policy_anchor_mode: str = "fixed",
    policy_anchor_target_ratio: float | None = None,
    policy_anchor_beta_min: float = 0.0,
    policy_anchor_beta_max: float = 10.0,
    policy_anchor_ema: float = 0.9,
    policy_anchor_scope: str = "generator",
    trust_kl_budget: float | None = None,
    trust_safety_questions: int = 0,
    trust_safety_tolerance: float = 0.0,
    trust_boundary_failure_ceiling: float = 1.0,
    trust_max_backtracks: int = 0,
    trust_backtrack_shrink: float = 0.5,
    historical_replay_fraction: float = 0.0,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_seed: int | None = None,
    lora_trainable: str = "all",
    gradient_projection: str = "none",
    gradient_projection_rank: int = 0,
    gradient_basis_path: str | None = None,
    gradient_projection_preserve_norm: bool = True,
    buffer_replicates: int = 1,
    question_schedule: str = "uniform",
    schedule_exploration: float = 0.1,
    eval_every: int = 0,
    eval_rounds: tuple[int, ...] | list[int] | None = None,
    eval_fn=None,
    diagnostics_fn=None,
    diagnostics_level: str = "standard",
    diagnostics_gradient_questions: int = 0,
    diagnostics_probe_fn=None,
    candidate_utility_questions: int = 0,
    candidate_utility_batch: int = 16,
    checkpoint_every: int = 0,
    checkpoint_fn=None,
    exact_cache: bool = False,
    state_checkpoint_every: int = 0,
    state_checkpoint_fn=None,
    resume_state: dict | None = None,
    resume_fingerprint: str | None = None,
    log=print,
) -> list[dict]:
    """Train an isolated latent-reasoning generator with answer-conditioned EM."""

    run_config = L2RRunConfig.from_call(locals())

    run_config = _validate_l2r_run_config(
        run_config,
        task=task,
        diagnostics_fn=diagnostics_fn,
        diagnostics_probe_fn=diagnostics_probe_fn,
        state_checkpoint_fn=state_checkpoint_fn,
        resume_state=resume_state,
    )
    adaptive_max_g = run_config.adaptive_max_g
    diagnostics_level = run_config.diagnostics_level

    model, tok = (
        model_tok
        if model_tok is not None
        else load_model(
            seed=seed,
            model=model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_seed=lora_seed,
        )
    )
    if answer_event_mode == "strict_terminal_marker":
        _validate_strict_answer_event_tokenization(tok)
    named_trainable = configure_lora_trainable(model, lora_trainable)
    opt = torch.optim.Adam(
        [parameter for _, parameter in named_trainable],
        lr=lr,
    )
    gradient_projector = GradientProjector(
        named_trainable,
        mode=gradient_projection,
        rank=gradient_projection_rank,
        basis_path=(
            gradient_basis_path
            or os.environ.get("L2R_GRADIENT_BASIS_PATH")
        ),
        seed=(lora_seed if lora_seed is not None else seed) * 1009 + 17,
        preserve_norm=gradient_projection_preserve_norm,
    )
    rng = np.random.default_rng(seed)
    prompt_ids = [
        encode_task_prompt(
            tok,
            task,
            pid,
            return_tensors="pt",
        ).input_ids[0].detach().cpu()
        for pid in range(len(task.prompts))
    ]
    training_pids, safety_pids, candidate_utility_pids = (
        _reserved_question_partitions(
            len(task.prompts),
            safety_questions=trust_safety_questions,
            utility_questions=candidate_utility_questions,
        )
    )
    training_question_count = len(training_pids)
    if training_question_count < B // G:
        raise ValueError(
            "training questions remaining after safety and utility reserves "
            "must cover B//G"
        )
    safety_rows = []
    for pid in safety_pids:
        row = _gold_trace(
            tok,
            task,
            prompt_ids[pid],
            pid,
            -1,
            answer_event_mode=answer_event_mode,
            answer_target_termination=answer_target_termination,
        )
        if row is None:
            raise ValueError("trust safety questions require gold rationales")
        safety_rows.append(row)
    candidate_utility_rows = []
    for pid in candidate_utility_pids:
        row = _gold_answer_trace(
            tok,
            task,
            prompt_ids[pid],
            pid,
            answer_event_mode=answer_event_mode,
            answer_target_termination=answer_target_termination,
        )
        if row is None:
            raise ValueError(
                "candidate utility questions require gold answers"
            )
        candidate_utility_rows.append(row)
    candidate_utility_probe_fn = None
    buffers: dict[int, list[L2RTrace]] = {}
    seen_questions: set[int] = set()
    records = []
    total_gen = total_generated_tokens = total_backward_tokens = total_steps = 0
    total_policy_backward_tokens = total_anchor_backward_tokens = 0
    total_reader_decode_tokens = 0
    total_evictions = total_duplicates = total_reader_decode = 0
    total_current_policy_backward_tokens = 0
    total_replay_policy_backward_tokens = 0
    question_exposures = 0
    scheduler = QuestionScheduler(
        training_question_count,
        seed=seed * 1013 + 23,
        mode=question_schedule,
        exploration=schedule_exploration,
    )
    policy_anchor_state: dict[str, float] = {}
    cache_stats = _new_cache_stats()
    training_diagnostic_state: dict[str, object] = {
        "accepted_steps": 0,
        "consecutive_rejections": 0,
        "probe_baseline_accuracy": None,
        "probe_previous_accuracy": None,
        "probe_baseline_elapsed_seconds": 0.0,
        "candidate_utility_baseline_probe": None,
        "candidate_utility_previous_probe": None,
        "candidate_utility_probe_calls": 0,
        "candidate_utility_probe_generations": 0,
        "candidate_utility_probe_generated_tokens": 0,
    }
    if candidate_utility_pids:
        def candidate_utility_probe_fn(current_model):
            result = _candidate_utility_probe(
                current_model,
                tok,
                task,
                candidate_utility_pids,
                batch=candidate_utility_batch,
            )
            training_diagnostic_state["candidate_utility_probe_calls"] = (
                int(
                    training_diagnostic_state.get(
                        "candidate_utility_probe_calls",
                        0,
                    )
                )
                + 1
            )
            training_diagnostic_state[
                "candidate_utility_probe_generations"
            ] = (
                int(
                    training_diagnostic_state.get(
                        "candidate_utility_probe_generations",
                        0,
                    )
                )
                + int(result["question_count"])
            )
            training_diagnostic_state[
                "candidate_utility_probe_generated_tokens"
            ] = (
                int(
                    training_diagnostic_state.get(
                        "candidate_utility_probe_generated_tokens",
                        0,
                    )
                )
                + int(result["generated_tokens"])
            )
            return result
    start_round = 0
    if resume_state is not None:
        if resume_state.get("schema_version") != 1:
            raise ValueError("unsupported L2R round-state schema")
        if resume_state.get("fingerprint") != resume_fingerprint:
            raise ValueError("L2R resume-state fingerprint does not match this cell")
        start_round = int(resume_state.get("completed_rounds", -1))
        if start_round < 0 or start_round > rounds:
            raise ValueError(
                f"invalid completed-round count in resume state: {start_round}"
            )
        if bool(resume_state.get("exact_cache")) != bool(exact_cache):
            raise ValueError("L2R resume-state cache mode does not match this cell")
        _restore_trainable_parameter_state(
            named_trainable,
            resume_state["trainable_parameters"],
        )
        model.train(bool(resume_state["model_training"]))
        opt.load_state_dict(resume_state["optimizer"])
        buffers = {
            int(pid): [_trace_from_state(row) for row in rows]
            for pid, rows in resume_state["buffers"].items()
        }
        seen_questions = {
            int(pid) for pid in resume_state["seen_questions"]
        }
        records = list(resume_state["records"])
        counters = resume_state["counters"]
        total_gen = int(counters["total_gen"])
        total_generated_tokens = int(counters["total_generated_tokens"])
        total_backward_tokens = int(counters["total_backward_tokens"])
        total_steps = int(counters["total_steps"])
        total_policy_backward_tokens = int(
            counters["total_policy_backward_tokens"]
        )
        total_anchor_backward_tokens = int(
            counters["total_anchor_backward_tokens"]
        )
        total_reader_decode_tokens = int(
            counters["total_reader_decode_tokens"]
        )
        total_evictions = int(counters["total_evictions"])
        total_duplicates = int(counters["total_duplicates"])
        total_reader_decode = int(counters["total_reader_decode"])
        total_current_policy_backward_tokens = int(
            counters["total_current_policy_backward_tokens"]
        )
        total_replay_policy_backward_tokens = int(
            counters["total_replay_policy_backward_tokens"]
        )
        question_exposures = int(counters["question_exposures"])
        scheduler.load_state_dict(resume_state["scheduler"])
        policy_anchor_state = dict(resume_state["policy_anchor_state"])
        training_diagnostic_state = dict(
            resume_state["training_diagnostic_state"]
        )
        cache_stats = {
            key: int(resume_state.get("cache_stats", {}).get(key, 0))
            for key in _CACHE_COUNTERS
        }
        _restore_rng_state(rng, resume_state["rng"])
        log(f"  [L2R resume] restored {start_round}/{rounds} completed rounds")
    elif diagnostics_probe_fn is not None:
        probe_started = time.perf_counter()
        baseline_accuracy = run_diagnostic_probe(model, diagnostics_probe_fn)
        training_diagnostic_state["probe_baseline_accuracy"] = baseline_accuracy
        training_diagnostic_state["probe_previous_accuracy"] = baseline_accuracy
        training_diagnostic_state["probe_baseline_elapsed_seconds"] = (
            time.perf_counter() - probe_started
        )
    for key in (
        "candidate_utility_probe_calls",
        "candidate_utility_probe_generations",
        "candidate_utility_probe_generated_tokens",
    ):
        training_diagnostic_state.setdefault(key, 0)
    if (
        candidate_utility_probe_fn is not None
        and training_diagnostic_state.get(
            "candidate_utility_previous_probe"
        ) is None
    ):
        utility_baseline = candidate_utility_probe_fn(model)
        training_diagnostic_state[
            "candidate_utility_baseline_probe"
        ] = utility_baseline
        training_diagnostic_state[
            "candidate_utility_previous_probe"
        ] = utility_baseline

    state = _L2RRuntimeState(
        model=model,
        tok=tok,
        named_trainable=named_trainable,
        opt=opt,
        gradient_projector=gradient_projector,
        rng=rng,
        prompt_ids=prompt_ids,
        training_pids=training_pids,
        safety_pids=safety_pids,
        candidate_utility_pids=candidate_utility_pids,
        training_question_count=training_question_count,
        safety_rows=safety_rows,
        candidate_utility_rows=candidate_utility_rows,
        candidate_utility_probe_fn=candidate_utility_probe_fn,
        buffers=buffers,
        seen_questions=seen_questions,
        records=records,
        total_gen=total_gen,
        total_generated_tokens=total_generated_tokens,
        total_backward_tokens=total_backward_tokens,
        total_steps=total_steps,
        total_policy_backward_tokens=total_policy_backward_tokens,
        total_anchor_backward_tokens=total_anchor_backward_tokens,
        total_reader_decode_tokens=total_reader_decode_tokens,
        total_evictions=total_evictions,
        total_duplicates=total_duplicates,
        total_reader_decode=total_reader_decode,
        total_current_policy_backward_tokens=total_current_policy_backward_tokens,
        total_replay_policy_backward_tokens=total_replay_policy_backward_tokens,
        question_exposures=question_exposures,
        scheduler=scheduler,
        policy_anchor_state=policy_anchor_state,
        cache_stats=cache_stats,
        training_diagnostic_state=training_diagnostic_state,
    )
    for round_index in range(start_round, rounds):
        _run_l2r_round(
            config=run_config,
            state=state,
            task=task,
            round_index=round_index,
            eval_fn=eval_fn,
            diagnostics_fn=diagnostics_fn,
            diagnostics_probe_fn=diagnostics_probe_fn,
            checkpoint_fn=checkpoint_fn,
            state_checkpoint_fn=state_checkpoint_fn,
            log=log,
        )

    return state.records
