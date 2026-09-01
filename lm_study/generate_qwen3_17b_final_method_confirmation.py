#!/usr/bin/env python3
"""Generate the frozen ten-method, seven-seed final confirmation YAML."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import yaml


RUN_ID = "978b99c8"
SEEDS = (1201, 1213, 1217, 1223, 1229, 1231, 1237)
CHECKPOINTS = (1, 2, 4, 8, 16, 24, 32)
SCRIPT_DIR = Path(__file__).resolve().parent
DEVELOPMENT_CONFIG = SCRIPT_DIR / "experiments_qwen3_17b_final_method_development.yaml"
FACTORIAL_CONFIG = SCRIPT_DIR / "experiments_qwen3_17b_l2r_common_protocol_factorial.yaml"

METHOD_ORDER = (
    "base",
    "AC-ALG1",
    "Gold-CoT-SFT",
    "RFT-Source",
    "ReST-EM",
    "STaR",
    "TRICE",
    "GRPO",
    "RLOO",
)
CELL_ORDER = (
    "CTRL-base",
    "Q5-LR1e-5-U1-K16",
    "PIS-S8-B8-U4",
    "GOLD-LR3e-6-E2",
    "RFT-LR1e-5-E2",
    "REST-LR1e-5-E1-I4",
    "STAR-LR3e-6-E2",
    "TRICE-LR1e-4-CV",
    "GRPO-S16-B4-U4",
    "RLOO-S16-B8-U4",
)
SELECTED_SOURCE = {
    "Q5-LR1e-5-U1-K16": DEVELOPMENT_CONFIG,
    "GOLD-LR3e-6-E2": DEVELOPMENT_CONFIG,
    "RFT-LR1e-5-E2": DEVELOPMENT_CONFIG,
    "REST-LR1e-5-E1-I4": DEVELOPMENT_CONFIG,
    "STAR-LR3e-6-E2": DEVELOPMENT_CONFIG,
    "TRICE-LR1e-4-CV": DEVELOPMENT_CONFIG,
    "PIS-S8-B8-U4": FACTORIAL_CONFIG,
    "GRPO-S16-B4-U4": FACTORIAL_CONFIG,
    "RLOO-S16-B8-U4": FACTORIAL_CONFIG,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _variants(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [dict(value) for value in raw]
    if isinstance(raw, dict):
        return [dict(raw)]
    raise ValueError("algorithm declaration must be an object or list")


def _selected_cell(cell_id: str) -> tuple[str, dict[str, Any]]:
    source = SELECTED_SOURCE[cell_id]
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    matches = [
        (str(method), deepcopy(cell))
        for method, raw in (payload.get("algos") or {}).items()
        for cell in _variants(raw)
        if str(cell.get("cell_id")) == cell_id
    ]
    if len(matches) != 1:
        raise ValueError(f"{source.name}: expected one source cell {cell_id!r}")
    return matches[0]


def selected_cells() -> dict[str, tuple[str, dict[str, Any]]]:
    return {cell_id: _selected_cell(cell_id) for cell_id in CELL_ORDER[1:]}


def build_payload() -> dict[str, Any]:
    selected = selected_cells()
    algos: dict[str, Any] = {"base": {"cell_id": "CTRL-base"}}
    for cell_id in CELL_ORDER[1:]:
        method, cell = selected[cell_id]
        if method in algos:
            existing = algos[method]
            if not isinstance(existing, list):
                algos[method] = [existing]
            algos[method].append(cell)
        else:
            algos[method] = cell

    payload = {
        "run_id": RUN_ID,
        "tag_prefix": "q3_final_method_confirmation",
        "diagnostic": {
            "stage": "fresh_seven_seed_final_method_confirmation",
            "evidence_class": (
                "preregistered_confirmation_on_reused_development_validation"
            ),
            "require_protocol_outcome_diagnostics": True,
            "partitions": {
                "optimization_source": "gsm8k_train_optimization_subset",
                "development_validation_source": "gsm8k_train_validation_subset",
                "official_test_used": False,
            },
            "scientific_question": (
                "Under one frozen common protocol, how do the two posterior-SFT "
                "finalists compare with supervised, self-training and online-RL "
                "controls over seven fresh paired training seeds?"
            ),
            "provenance": {
                "model_id": "Qwen/Qwen3-1.7B-Base",
                "model_revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                "dataset_id": "openai/gsm8k",
                "dataset_config": "main",
                "dataset_revision": "740312add88f781978c0658806c59bc2815b9866",
                "protocol": (
                    "docs/experiments/qwen3_final_method_comparison/"
                    "qwen3_final_method_comparison_protocol.tex"
                ),
                "source_config_sha256": {
                    DEVELOPMENT_CONFIG.name: _sha256(DEVELOPMENT_CONFIG),
                    FACTORIAL_CONFIG.name: _sha256(FACTORIAL_CONFIG),
                },
                "q5_retention_evidence": (
                    "qwen3_q5_allocation_update_screen_20260817"
                ),
            },
            "design": {
                "cell_order": list(CELL_ORDER),
                "paired_confirmation_seeds": list(SEEDS),
                "methods": len(CELL_ORDER),
                "seeds_per_method": len(SEEDS),
                "array_tasks": len(CELL_ORDER) * len(SEEDS),
                "checkpoint_rule": "final_round_32",
            },
            "fixed_contract": {
                "prompt_shots": 3,
                "nested_prompt_bank_size": 5,
                "optimization_questions": 128,
                "development_validation_questions": 400,
                "outer_rounds": 32,
                "adapter_surface": "attention_and_mlp_projections",
                "adapter_rank": 16,
                "answer_event": "one_terminal_marker_followed_by_tokenizer_eos",
                "evaluation_prompt": "question_only",
                "trained_adapters_retained": True,
                "official_test_used": False,
            },
            "reporting_order": [
                "final_extracted_answer_accuracy",
                "final_strict_terminal_accuracy",
                "normalized_extracted_trajectory_auc",
            ],
            "analysis_contract": [
                "Treat paired training seed as the independent replicate.",
                "Use a nested paired bootstrap over training seeds and matched evaluation questions.",
                "Report final extracted Acc@1 first, strict final accuracy second and extracted trajectory AUC third.",
                "Report generated tokens, backward tokens, optimizer steps and accelerator-hours separately.",
                "A superiority claim requires a positive lower 95 percent interval bound.",
                "Use final round 32 for the endpoint; do not select a confirmation checkpoint from outcomes.",
                "Select transfer roles deterministically by the registered metric order, while reporting unresolved intervals as unresolved.",
                "Do not access the official GSM8K test split.",
            ],
            "generality_role_contract": {
                "posterior": ["Q5-LR1e-5-U1-K16", "PIS-S8-B8-U4"],
                "non_rl_self_training": [
                    "RFT-LR1e-5-E2",
                    "REST-LR1e-5-E1-I4",
                    "STAR-LR3e-6-E2",
                    "TRICE-LR1e-4-CV",
                ],
                "rl": ["GRPO-S16-B4-U4", "RLOO-S16-B8-U4"],
                "scale_comparator": "higher-ranked of non_rl_self_training and rl",
            },
        },
        "defaults": {
            "task": "gsm8k",
            "model": "qwen3-1.7b-base",
            "rounds": 32,
            "seeds": len(SEEDS),
            "seed_values": list(SEEDS),
            "n_test": 400,
            "train_partition": "train",
            "eval_partition": "validation",
            "answer_event_mode": "strict_terminal_marker",
            "answer_target_termination": "none",
            "evaluation_prompt": "question",
            "batch": 64,
            "eval_batch": 32,
            "G": 16,
            "prompts": 128,
            "shots": 3,
            "shot_bank_size": 5,
            "task_seed_from_run_seed": True,
            "question_sampling": "epoch_shuffle",
            "grad_checkpoint": False,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_target_set": "attention_mlp",
            "out": (
                "~/po_results/2026-08-18/final-comparison/"
                "seven-seed-confirmation__978b99c8"
            ),
            "eval_rounds": list(CHECKPOINTS),
            "dump_completions": 100,
            "save_adapter": True,
            "save_training_diagnostics": True,
            "training_diagnostics_level": "standard",
            "training_diagnostics_trace_tape": False,
            "training_diagnostics_gradient_questions": 0,
            "training_diagnostics_probe_size": 0,
            "checkpoint_every": 0,
            "passk": 8,
            "passk_n": 100,
        },
        "algos": algos,
    }
    validate_payload(payload)
    return payload


def _configured_cells(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (str(method), cell)
        for method, raw in (payload.get("algos") or {}).items()
        for cell in _variants(raw)
    ]


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("run_id") != RUN_ID:
        raise ValueError("final confirmation run ID changed")
    defaults = payload.get("defaults") or {}
    if tuple(defaults.get("seed_values") or ()) != SEEDS:
        raise ValueError("final confirmation seed family changed")
    if tuple(defaults.get("eval_rounds") or ()) != CHECKPOINTS:
        raise ValueError("final confirmation checkpoint schedule changed")
    if defaults.get("eval_partition") != "validation":
        raise ValueError("final confirmation must use validation only")
    if defaults.get("save_adapter") is not True:
        raise ValueError("final confirmation must retain trained adapters")
    if defaults.get("lora_target_set") != "attention_mlp":
        raise ValueError("final confirmation LoRA surface changed")
    cells = _configured_cells(payload)
    observed_ids = tuple(str(cell.get("cell_id")) for _method, cell in cells)
    if observed_ids != CELL_ORDER:
        raise ValueError(f"final confirmation cell order changed: {observed_ids}")
    if tuple(method for method, _cell in cells) != (
        "base",
        "AC-ALG1",
        "AC-ALG1",
        "Gold-CoT-SFT",
        "RFT-Source",
        "ReST-EM",
        "STaR",
        "TRICE",
        "GRPO",
        "RLOO",
    ):
        raise ValueError("final confirmation method mapping changed")
    for method, cell in cells:
        termination = cell.get(
            "answer_target_termination", defaults.get("answer_target_termination")
        )
        if method == "base":
            if termination != "none":
                raise ValueError("frozen base cannot declare a training answer target")
        elif method in {"GRPO", "RLOO"}:
            if termination != "none" or cell.get("reward_requires_eos") is not True:
                raise ValueError(
                    f"{cell['cell_id']}: online-RL cell lost its strict EOS reward"
                )
        elif termination != "eos":
            raise ValueError(f"{cell['cell_id']}: trained answer target lost EOS")
    fixed = (payload.get("diagnostic") or {}).get("fixed_contract") or {}
    if fixed.get("official_test_used") is not False:
        raise ValueError("official-test prohibition is missing")
    if int((payload.get("diagnostic") or {}).get("design", {}).get("array_tasks", -1)) != 70:
        raise ValueError("final confirmation must contain exactly 70 tasks")


def render_payload() -> str:
    header = (
        "# Generated by generate_qwen3_17b_final_method_confirmation.py.\n"
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
