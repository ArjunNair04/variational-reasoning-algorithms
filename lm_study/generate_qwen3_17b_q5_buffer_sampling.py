#!/usr/bin/env python3
"""Generate the seven-seed Q5 large-support M-step sampling study."""

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


RUN_ID = "1d5b0eb4"
CONTROL_RUN_ID = "f20c9e17"
CELL_ORDER = (
    "Q5-S32-B32-U1-FULL",
    "Q5-S32-B32-U1-MS16",
)
CONTROL_CELL = "Q5-MORE-S32-B16-U1"


def _variants(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [deepcopy(value) for value in raw]
    if isinstance(raw, dict):
        return [deepcopy(raw)]
    raise ValueError("algorithm declaration must be an object or list")


def _control_q5() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = build_control_payload()
    cells = {
        str(cell["cell_id"]): cell
        for raw in payload["algos"].values()
        for cell in _variants(raw)
    }
    return payload, deepcopy(cells["Q5-AD-M-LR1e-5-U1-K16"])


def build_payload() -> dict[str, Any]:
    control, q5 = _control_q5()
    common = deepcopy(q5)
    common.update(
        algorithm_profile="q5_buffer_sampling",
        batch=128,
        G=32,
        buffer_limit=32,
    )
    full = deepcopy(common)
    full.update(cell_id=CELL_ORDER[0], mstep_sample_size=0)
    sampled = deepcopy(common)
    sampled.update(cell_id=CELL_ORDER[1], mstep_sample_size=16)

    defaults = deepcopy(control["defaults"])
    defaults.update(
        out=(
            "~/po_results/2026-09-02/q5-buffer-sampling/"
            "qwen3-q5-buffer-sampling__1d5b0eb4"
        ),
        training_diagnostics_trace_tape=True,
        save_adapter=True,
    )
    payload = {
        "run_id": RUN_ID,
        "tag_prefix": "q3_q5_buffer_sampling",
        "diagnostic": {
            "stage": "seven_seed_q5_buffer_sampling",
            "evidence_class": "prespecified_paired_followup",
            "scientific_question": (
                "Does retaining 32 Q5 traces and sampling 16 posterior draws for "
                "the M-step improve the performance-compute trade-off?"
            ),
            "control_run_id": CONTROL_RUN_ID,
            "control_cell": CONTROL_CELL,
            "design": {
                "cell_order": list(CELL_ORDER),
                "paired_seeds": list(SEEDS),
                "array_tasks": len(CELL_ORDER) * len(SEEDS),
                "historical_control_reused": True,
            },
            "fixed_contract": {
                "model_id": "Qwen/Qwen3-1.7B-Base",
                "dataset_id": "openai/gsm8k",
                "prompt_shots": 3,
                "optimization_questions": 128,
                "development_validation_questions": 400,
                "outer_rounds": 32,
                "raw_proposals_per_question": 32,
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
                "Compare full B32 with the existing Q5-MORE S32-B16 control.",
                "Compare posterior-sampled B32 directly with full B32.",
                "Report proposal draws, unique M-step traces, posterior mass covered, backward tokens, optimizer steps and accelerator-hours separately.",
                "Do not access the official GSM8K test split.",
            ],
        },
        "defaults": defaults,
        "algos": {"AC-ALG1": [full, sampled]},
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("run_id") != RUN_ID:
        raise ValueError("Q5 buffer-sampling run ID changed")
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
        raise ValueError("Q5 buffer-sampling cell order changed")
    if payload["diagnostic"]["design"].get("array_tasks") != 14:
        raise ValueError("Q5 buffer-sampling study must contain exactly 14 tasks")
    for cell, sample_size in zip(cells, (0, 16), strict=True):
        required = {
            "algorithm_profile": "q5_buffer_sampling",
            "batch": 128,
            "G": 32,
            "buffer_limit": 32,
            "iters": 1,
            "mstep_sample_size": sample_size,
        }
        mismatches = {
            key: {"actual": cell.get(key), "required": value}
            for key, value in required.items()
            if cell.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{cell['cell_id']} changed: {mismatches}")
    if payload["diagnostic"]["fixed_contract"].get("official_test_used") is not False:
        raise ValueError("official-test prohibition is missing")


def render_payload() -> str:
    header = (
        "# Generated by generate_qwen3_17b_q5_buffer_sampling.py.\n"
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
