#!/usr/bin/env python3
"""Generate the seven-seed Q5 support-allocation follow-up."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from generate_qwen3_17b_q5_buffer_sampling import (
    CHECKPOINTS,
    RUN_ID as CONTROL_RUN_ID,
    SEEDS,
    build_payload as build_control_payload,
)


RUN_ID = "6f89a2c1"
CELL_ORDER = (
    "Q5-S64-B32-U1-FULL",
    "Q5-S32-B32-U1-TOPRES15",
)
CONTROL_CELLS = (
    "Q5-S32-B32-U1-FULL",
    "Q5-S32-B32-U1-MS16",
)


def _variants(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [deepcopy(value) for value in raw]
    if isinstance(raw, dict):
        return [deepcopy(raw)]
    raise ValueError("algorithm declaration must be an object or list")


def _control_cell(cell_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = build_control_payload()
    cells = {
        str(cell["cell_id"]): cell
        for raw in payload["algos"].values()
        for cell in _variants(raw)
    }
    return payload, deepcopy(cells[cell_id])


def build_payload() -> dict[str, Any]:
    control, full_control = _control_cell(CONTROL_CELLS[0])
    common = deepcopy(full_control)
    common.update(
        algorithm_profile="q5_support_followup",
        buffer_limit=32,
        iters=1,
    )

    broad = deepcopy(common)
    broad.update(
        cell_id=CELL_ORDER[0],
        batch=256,
        G=64,
        mstep_sample_size=0,
        mstep_sampling_strategy="posterior_categorical",
    )
    stratified = deepcopy(common)
    stratified.update(
        cell_id=CELL_ORDER[1],
        batch=128,
        G=32,
        mstep_sample_size=16,
        mstep_sampling_strategy="top_plus_residual",
    )

    defaults = deepcopy(control["defaults"])
    defaults.update(
        out=(
            "~/po_results/2026-09-03/q5-support-followup/"
            "qwen3-q5-support-followup__6f89a2c1"
        ),
        training_diagnostics_trace_tape=True,
        save_adapter=True,
    )
    payload = {
        "run_id": RUN_ID,
        "tag_prefix": "q3_q5_support_followup",
        "diagnostic": {
            "stage": "seven_seed_q5_support_followup",
            "evidence_class": "prespecified_paired_followup",
            "scientific_question": (
                "Can Q5 use a larger candidate set, or preserve its dominant "
                "trace exactly while sampling the posterior tail, to improve "
                "accuracy without an unnecessary full-buffer backward pass?"
            ),
            "control_run_id": CONTROL_RUN_ID,
            "control_cells": list(CONTROL_CELLS),
            "design": {
                "cell_order": list(CELL_ORDER),
                "paired_seeds": list(SEEDS),
                "array_tasks": len(CELL_ORDER) * len(SEEDS),
                "historical_controls_reused": True,
            },
            "fixed_contract": {
                "model_id": "Qwen/Qwen3-1.7B-Base",
                "dataset_id": "openai/gsm8k",
                "prompt_shots": 3,
                "optimization_questions": 128,
                "development_validation_questions": 400,
                "outer_rounds": 32,
                "questions_per_round": 4,
                "retained_support_per_question": 32,
                "adapter_surface": "attention_and_mlp_projections",
                "adapter_rank": 16,
                "answer_event": "one_terminal_marker_followed_by_tokenizer_eos",
                "official_test_used": False,
            },
            "analysis_contract": [
                "Treat paired training seed as the independent replicate.",
                "Report Final Acc@1 first, strict final accuracy second and trajectory AUC third.",
                "Use final round 32; do not select a checkpoint from outcomes.",
                "Compare S64 full support with the existing S32 full-support control.",
                "Compare exact-top plus 15 residual draws with both S32 full support and 16 ordinary posterior draws.",
                "Report proposal draws, unique M-step traces, posterior mass covered, backward tokens, optimizer steps and accelerator-hours separately.",
                "Do not access the official GSM8K test split.",
            ],
        },
        "defaults": defaults,
        "algos": {"AC-ALG1": [broad, stratified]},
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("run_id") != RUN_ID:
        raise ValueError("Q5 support-follow-up run ID changed")
    defaults = payload.get("defaults") or {}
    if tuple(defaults.get("seed_values") or ()) != SEEDS:
        raise ValueError("paired seed family changed")
    if tuple(defaults.get("eval_rounds") or ()) != CHECKPOINTS:
        raise ValueError("checkpoint schedule changed")
    if defaults.get("eval_partition") != "validation":
        raise ValueError("study must use train-derived validation only")
    if defaults.get("lora_target_set") != "attention_mlp" or defaults.get("lora_r") != 16:
        raise ValueError("LoRA contract changed")
    cells = _variants(payload["algos"]["AC-ALG1"])
    if tuple(str(cell["cell_id"]) for cell in cells) != CELL_ORDER:
        raise ValueError("Q5 support-follow-up cell order changed")
    expected = (
        (256, 64, 0, "posterior_categorical"),
        (128, 32, 16, "top_plus_residual"),
    )
    for cell, coordinates in zip(cells, expected, strict=True):
        required = {
            "algorithm_profile": "q5_support_followup",
            "batch": coordinates[0],
            "G": coordinates[1],
            "buffer_limit": 32,
            "iters": 1,
            "mstep_sample_size": coordinates[2],
            "mstep_sampling_strategy": coordinates[3],
        }
        mismatches = {
            key: {"actual": cell.get(key), "required": value}
            for key, value in required.items()
            if cell.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{cell['cell_id']} changed: {mismatches}")
    if payload["diagnostic"]["design"].get("array_tasks") != 14:
        raise ValueError("Q5 support follow-up must contain exactly 14 tasks")
    if payload["diagnostic"]["fixed_contract"].get("official_test_used") is not False:
        raise ValueError("official-test prohibition is missing")


def render_payload() -> str:
    header = (
        "# Generated by generate_qwen3_17b_q5_support_followup.py.\n"
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
