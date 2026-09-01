"""Exact, machine-readable prompt provenance for GSM8K experiment cells."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any

from prompting import build_proposal_prompt


PROMPT_CONTRACT_SCHEMA_VERSION = 1

_SOURCE_REJECTION_METHODS = {"RFT-Source", "ReST-EM"}
_AUXILIARY_ANSWER_GUIDE_METHODS = {"STaR", "TRICE"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_gsm8k_prompt_contract(
    task,
    *,
    proposal_prompt: str | None,
    method: str,
    seed: int,
    tag: str,
    answer_event_mode: str,
    answer_target_termination: str,
) -> dict[str, Any]:
    """Render and validate every canonical and proposal prompt for one cell."""

    is_frozen_base = method == "base"
    is_gold_cot = method == "Gold-CoT-SFT"
    if proposal_prompt is None:
        if not is_frozen_base:
            raise ValueError("trained GSM8K cells require an explicit proposal prompt")
        effective_proposal_prompt = "question"
    else:
        effective_proposal_prompt = str(proposal_prompt)

    required = ("prompts", "gold_answer", "train_qi", "shot_qi")
    missing = [name for name in required if not hasattr(task, name)]
    if missing:
        raise ValueError(f"GSM8K prompt contract is missing task fields: {missing}")
    if len(task.prompts) != len(task.gold_answer) or len(task.prompts) != len(task.train_qi):
        raise ValueError("GSM8K prompt contract inputs do not align")

    rows = []
    for pid, (canonical, answer, dataset_index) in enumerate(
        zip(task.prompts, task.gold_answer, task.train_qi, strict=True)
    ):
        if not canonical.endswith("\nAnswer:"):
            raise ValueError(f"canonical prompt {pid} does not end in 'Answer:'")
        proposal = build_proposal_prompt(
            canonical,
            answer,
            effective_proposal_prompt,
        )
        auxiliary_mode = (
            "answer_derive"
            if method in _AUXILIARY_ANSWER_GUIDE_METHODS
            else None
        )
        auxiliary_prompt = (
            build_proposal_prompt(canonical, answer, auxiliary_mode)
            if auxiliary_mode is not None
            else None
        )
        injected_phrase = f"correct final answer is {answer}"
        canonical_has_injected_answer = injected_phrase in canonical
        proposal_has_injected_answer = injected_phrase in proposal
        if canonical_has_injected_answer:
            raise ValueError(f"canonical prompt {pid} contains the proposal-only answer hint")
        if effective_proposal_prompt == "question":
            if proposal != canonical or proposal_has_injected_answer:
                raise ValueError("question-only proposal changed the canonical prompt")
        elif proposal == canonical or not proposal_has_injected_answer:
            raise ValueError("answer-conditioned proposal did not inject the gold answer")
        rows.append(
            {
                "pid": pid,
                "dataset_train_index": int(dataset_index),
                "gold_answer": str(answer),
                "canonical_prompt": canonical,
                "canonical_prompt_sha256": _sha256(canonical),
                "proposal_prompt": proposal,
                "proposal_prompt_sha256": _sha256(proposal),
                "proposal_differs_from_canonical": proposal != canonical,
                "canonical_contains_proposal_answer_hint": canonical_has_injected_answer,
                "proposal_contains_gold_answer_hint": proposal_has_injected_answer,
                "auxiliary_prompt_mode": auxiliary_mode,
                "auxiliary_prompt": auxiliary_prompt,
                "auxiliary_prompt_sha256": (
                    _sha256(auxiliary_prompt)
                    if auxiliary_prompt is not None
                    else None
                ),
                "auxiliary_prompt_contains_gold_answer_hint": (
                    injected_phrase in auxiliary_prompt
                    if auxiliary_prompt is not None
                    else False
                ),
            }
        )

    if is_frozen_base:
        generation_paths = []
        reconstruction = {
            "applied": False,
            "reason": "frozen evaluation-only control",
        }
    elif is_gold_cot:
        generation_paths = []
        reconstruction = {
            "applied": True,
            "source": "human GSM8K rationale",
            "canonical_context_reused": True,
            "loss_span": "gold rationale and final answer followed by tokenizer EOS",
            "generated_trace_used": False,
        }
    elif method in _SOURCE_REJECTION_METHODS:
        generation_paths = [
            {
                "role": "direct_candidate",
                "prompt_mode": "question",
                "gold_answer_visible": False,
                "decoding": "temperature_sampling",
            }
        ]
        reconstruction = {
            "applied": True,
            "proposal_instruction_removed": False,
            "canonical_context_reused": True,
            "loss_span": "answer-correct naturally terminated sampled completion",
            "generated_answer_suffix_removed": False,
            "mstep": "token-mean supervised fine-tuning",
        }
    elif method == "STaR":
        generation_paths = [
            {
                "role": "direct_candidate",
                "prompt_mode": "question",
                "gold_answer_visible": False,
                "decoding": "greedy",
            },
            {
                "role": "failed-question_rationalization",
                "prompt_mode": "answer_derive",
                "gold_answer_visible": True,
                "decoding": "greedy",
            },
        ]
        reconstruction = {
            "applied": True,
            "proposal_instruction_removed": True,
            "canonical_context_reused": True,
            "loss_span": "answer-correct naturally terminated generated completion",
            "generated_answer_suffix_removed": False,
            "mstep": "token-mean supervised fine-tuning from the original adapter",
        }
    elif method == "TRICE":
        generation_paths = [
            {
                "role": "one_time_chain_initializer",
                "prompt_mode": "answer_derive",
                "gold_answer_visible": True,
                "decoding": "temperature_1_sampling",
            },
            {
                "role": "persistent_chain_prior_proposal",
                "prompt_mode": "question",
                "gold_answer_visible": False,
                "decoding": "temperature_1_sampling",
            },
        ]
        reconstruction = {
            "applied": True,
            "proposal_instruction_removed": True,
            "canonical_context_reused": True,
            "loss_span": "complete persistent-chain state and control-variate proposal tokens",
            "generated_answer_suffix_removed": False,
            "mstep": "TRICE score estimator with one update per macrocycle",
        }
    else:
        generation_paths = [
            {
                "role": "latent_candidate",
                "prompt_mode": effective_proposal_prompt,
                "gold_answer_visible": effective_proposal_prompt != "question",
            }
        ]
        reconstruction = {
            "applied": True,
            "proposal_instruction_removed": True,
            "generated_answer_suffix_removed": True,
            "canonical_context_reused": True,
            "latent_factor": "sampled rationale tokens through the terminal #### marker",
            "answer_factor": (
                "teacher-forced gold answer followed by tokenizer EOS"
                if answer_target_termination == "eos"
                else "teacher-forced gold answer"
            ),
            "joint_mstep_row": "canonical_prompt + latent_factor + answer_factor",
        }

    return {
        "schema_version": PROMPT_CONTRACT_SCHEMA_VERSION,
        "method": str(method),
        "tag": str(tag),
        "training_seed": int(seed),
        "task_seed": int(seed),
        "proposal_prompt_requested": proposal_prompt,
        "proposal_prompt_mode": effective_proposal_prompt,
        "proposal_generation_applied": not (is_frozen_base or is_gold_cot),
        "generation_paths": generation_paths,
        "answer_event_mode": str(answer_event_mode),
        "answer_target_termination": str(answer_target_termination),
        "shot_dataset_train_indices": [int(value) for value in task.shot_qi],
        "training_reconstruction": reconstruction,
        "rows": rows,
    }


def write_prompt_contract(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically write a deterministic compressed prompt manifest."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", mtime=0
            ) as stream:
                stream.write(encoded)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
