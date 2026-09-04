#!/usr/bin/env python3
"""Generate the seven-seed common-protocol multi-sample JEPO comparator."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from generate_qwen3_17b_selected_method_posterity import (
    CHECKPOINTS,
    SEEDS,
    build_payload as build_control_payload,
)


RUN_ID = "7452ba96"
CONTROL_RUN_ID = "68078ecc"
CELL_ID = "JEPO-MS4-LR1e-5"


def build_payload() -> dict[str, Any]:
    source = build_control_payload()
    defaults = deepcopy(source["defaults"])
    defaults.update(
        out=("~/po_results/2026-09-04/jepo-comparator/qwen3-jepo-ms4__7452ba96"),
        training_diagnostics_trace_tape=False,
        save_adapter=True,
    )
    cell = {
        "cell_id": CELL_ID,
        "lr": 1.0e-5,
        "batch": 64,
        "G": 4,
        "kl_coef": 1.0e-3,
        "jepo_supervised_coef": 1.0e-2,
        "jepo_format_penalty": 10.0,
        "jepo_advantage_clip": 1.0,
        "proposal_prompt": "question",
        "proposal_temperature": 1.0,
        "question_schedule_rng": "run_seed",
        "reward_requires_eos": True,
        "answer_target_termination": "eos",
    }
    payload = {
        "run_id": RUN_ID,
        "tag_prefix": "q3_jepo_comparator",
        "diagnostic": {
            "stage": "seven_seed_jepo_comparator",
            "evidence_class": "prespecified_external_method_comparator",
            "scientific_question": (
                "Can multi-sample JEPO improve Qwen3-1.7B GSM8K reasoning under "
                "the same training-generation budget and validation protocol as "
                "the selected Q5/PIS comparison panel?"
            ),
            "provenance": {
                "method_source": "Tang et al., Beyond Verifiable Rewards (NeurIPS 2025)",
                "control_run_id": CONTROL_RUN_ID,
                "selection_rule": (
                    "One paper-derived multi-sample cell; no local JEPO hyperparameter search."
                ),
                "protocol_adaptation": (
                    "The paper's K=4 estimator, format penalty, advantage clipping, "
                    "supervised coefficient and KL coefficient are retained. The "
                    "common Qwen LoRA learning rate and fixed 2,048-generation "
                    "budget replace the paper's full-model MATH scale."
                ),
            },
            "design": {
                "cell_order": [CELL_ID],
                "paired_seeds": list(SEEDS),
                "array_tasks": len(SEEDS),
                "historical_controls_reused": True,
                "control_cells": [
                    "CTRL-base",
                    "Q5-AD-M-LR1e-5-U1-K16",
                    "PIS-Q-S8-B8-U4",
                    "TRICE-LR1e-4-CV",
                    "RLOO-S16-B8-U4",
                ],
            },
            "fixed_contract": {
                "model_id": "Qwen/Qwen3-1.7B-Base",
                "dataset_id": "openai/gsm8k",
                "prompt_shots": 3,
                "optimization_questions": 128,
                "development_validation_questions": 400,
                "outer_rounds": 32,
                "generations_per_round": 64,
                "questions_per_round": 16,
                "samples_per_question": 4,
                "training_generations_per_seed": 2048,
                "adapter_surface": "attention_and_mlp_projections",
                "adapter_rank": 16,
                "proposal": "current_policy_question_conditioned",
                "answer_event": "one_terminal_marker_followed_by_tokenizer_eos",
                "official_test_used": False,
            },
            "estimator_contract": {
                "trace_credit": "multi_sample_log_evidence_minus_leave_one_out",
                "answer_term": "gold_answer_log_mean_probability",
                "answer_log_probability_reduction": "sequence_sum",
                "invalid_format_policy": "mask_lower_bound_and_apply_format_advantage",
                "masked_objective_denominator": "fixed_sample_count",
                "advantage_normalization": "population_std_then_clip_minus1_plus1",
                "kl_estimator": "token_level_k3_on_sampled_generation",
            },
            "analysis_contract": [
                "Treat paired training seed as the independent replicate.",
                "Report Final Acc@1 first, strict final accuracy second and trajectory AUC third.",
                "Use final round 32; do not select a checkpoint from outcomes.",
                "Compare JEPO with the same-seed frozen base and report descriptive context for Q5, PIS, TRICE-CV and RLOO.",
                "Report training generations, optimizer steps, generated tokens, backward tokens and accelerator-hours separately.",
                "Audit valid-format coverage, normalized advantage clipping, gold-answer weight ESS and sampled policy KL.",
                "Do not access the official GSM8K test split.",
            ],
        },
        "defaults": defaults,
        "algos": {"JEPO": cell},
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("run_id") != RUN_ID:
        raise ValueError("JEPO comparator run ID changed")
    defaults = payload.get("defaults") or {}
    required_defaults = {
        "rounds": 32,
        "seeds": 7,
        "n_test": 400,
        "train_partition": "train",
        "eval_partition": "validation",
        "answer_event_mode": "strict_terminal_marker",
        "evaluation_prompt": "question",
        "prompts": 128,
        "shots": 3,
        "shot_bank_size": 5,
        "task_seed_from_run_seed": True,
        "question_sampling": "epoch_shuffle",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_target_set": "attention_mlp",
    }
    mismatches = {
        key: {"actual": defaults.get(key), "required": value}
        for key, value in required_defaults.items()
        if defaults.get(key) != value
    }
    if mismatches:
        raise ValueError(f"JEPO common protocol changed: {mismatches}")
    if tuple(defaults.get("seed_values") or ()) != SEEDS:
        raise ValueError("JEPO paired seed family changed")
    if tuple(defaults.get("eval_rounds") or ()) != CHECKPOINTS:
        raise ValueError("JEPO checkpoint schedule changed")
    cell = payload.get("algos", {}).get("JEPO")
    required_cell = {
        "cell_id": CELL_ID,
        "lr": 1.0e-5,
        "batch": 64,
        "G": 4,
        "kl_coef": 1.0e-3,
        "jepo_supervised_coef": 1.0e-2,
        "jepo_format_penalty": 10.0,
        "jepo_advantage_clip": 1.0,
        "proposal_prompt": "question",
        "proposal_temperature": 1.0,
        "question_schedule_rng": "run_seed",
        "reward_requires_eos": True,
        "answer_target_termination": "eos",
    }
    if cell != required_cell:
        raise ValueError(f"JEPO cell changed: {cell}")
    if payload["diagnostic"]["design"].get("array_tasks") != 7:
        raise ValueError("JEPO comparator must contain exactly seven tasks")
    if payload["diagnostic"]["fixed_contract"].get("official_test_used") is not False:
        raise ValueError("official-test prohibition is missing")


def render_payload() -> str:
    header = (
        "# Generated by generate_qwen3_17b_jepo_comparator.py.\n"
        "# Edit the generator, regenerate this file, and run its --check mode.\n"
    )
    return header + yaml.safe_dump(
        build_payload(), sort_keys=False, width=100, allow_unicode=False
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", type=Path, metavar="YAML")
    group.add_argument("--write", type=Path, metavar="YAML")
    args = parser.parse_args()
    expected = render_payload()
    if args.check is not None:
        actual = yaml.safe_load(args.check.read_text(encoding="utf-8"))
        if actual != build_payload():
            raise SystemExit(f"generated configuration drift: {args.check}")
        print(f"verified generated configuration: {args.check}")
        return 0
    if args.write is not None:
        args.write.write_text(expected, encoding="utf-8")
        print(f"wrote generated configuration: {args.write}")
        return 0
    print(expected, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
