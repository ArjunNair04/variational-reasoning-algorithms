#!/usr/bin/env python3
"""Generate the evaluation-only GSM8K panel-robustness protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml


RUN_ID = "ec4c38db"
SOURCE_RUN_ID = "978b99c8"
DATASET_ROWS = 7473
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
VALIDATION_SEED = 20260716
PANEL_SEED = 20260904
PANEL_SIZE = 400
PANEL_COUNT = 3
SEEDS = (1201, 1213, 1217, 1223, 1229, 1231, 1237)
METHODS = {
    "Frozen-base": None,
    "Q5": "AC-ALG1_Q5-LR1e-5-U1-K16",
    "PIS": "AC-ALG1_PIS-S8-B8-U4",
    "TRICE": "TRICE_TRICE-LR1e-4-CV",
    "GRPO": "GRPO_GRPO-S16-B4-U4",
    "RLOO": "RLOO_RLOO-S16-B8-U4",
}


def _partition() -> tuple[
    list[int],
    list[list[int]],
    dict[int, list[int]],
    dict[int, list[int]],
    dict[int, list[int]],
]:
    order = np.random.default_rng(VALIDATION_SEED).permutation(DATASET_ROWS)
    validation = [int(value) for value in order[:400]]
    training_pool = order[400:]
    demonstrations: dict[int, list[int]] = {}
    shot_banks: dict[int, list[int]] = {}
    optimization: dict[int, list[int]] = {}
    excluded = set(validation)
    for seed in SEEDS:
        seeded = np.random.default_rng(seed).permutation(training_pool)
        demonstrations[seed] = [int(value) for value in seeded[:3]]
        shot_banks[seed] = [int(value) for value in seeded[:5]]
        optimization[seed] = [int(value) for value in seeded[5:133]]
        excluded.update(int(value) for value in seeded[:133])
    candidates = np.asarray(
        sorted(set(int(value) for value in training_pool) - excluded),
        dtype=int,
    )
    selected = np.random.default_rng(PANEL_SEED).permutation(candidates)[
        : PANEL_COUNT * PANEL_SIZE
    ]
    panels = [
        [int(value) for value in selected[start : start + PANEL_SIZE]]
        for start in range(0, PANEL_COUNT * PANEL_SIZE, PANEL_SIZE)
    ]
    return validation, panels, demonstrations, shot_banks, optimization


def build_payload() -> dict[str, Any]:
    validation, panels, demonstrations, shot_banks, optimization = _partition()
    payload = {
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "title": "Qwen3 GSM8K fresh-panel ranking robustness",
        "model": {
            "preset": "qwen3-1.7b-base",
            "hf_id": "Qwen/Qwen3-1.7B-Base",
            "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        },
        "dataset": {
            "id": "openai/gsm8k",
            "configuration": "main",
            "revision": DATASET_REVISION,
            "loaded_split": "train",
            "official_test_used": False,
            "row_count": DATASET_ROWS,
        },
        "design": {
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "panel_seed": PANEL_SEED,
            "panel_size": PANEL_SIZE,
            "panel_count": PANEL_COUNT,
            "array_tasks": len(METHODS) * len(SEEDS),
            "evaluation_generations_per_task": PANEL_COUNT * PANEL_SIZE,
            "training_performed": False,
            "source_adapter_root": (
                "~/po_results/2026-08-18/final-comparison/"
                "seven-seed-confirmation__978b99c8"
            ),
            "output_root": (
                "~/po_results/2026-09-04/evaluation/gsm8k-panel-robustness__ec4c38db"
            ),
            "prompt_shots": 3,
            "shot_bank_size": 5,
            "evaluation_prompt": "question_only",
            "decoding": "greedy",
            "max_new_tokens": 256,
            "batch_size": 32,
        },
        "source_contract": {
            "original_validation_indices": validation,
            "demonstration_indices_by_seed": demonstrations,
            "shot_bank_indices_by_seed": shot_banks,
            "optimization_indices_by_seed": optimization,
            "panels": [
                {"panel_id": index + 1, "dataset_train_indices": rows}
                for index, rows in enumerate(panels)
            ],
            "exclusion_rule": (
                "Panels exclude the original 400 validation rows and the first "
                "133 seeded training-pool positions for every source seed, covering "
                "the five-row shot bank and 128 optimization questions."
            ),
        },
        "methods": [
            {
                "method": method,
                "adapter_name_fragment": fragment,
                "trained": fragment is not None,
            }
            for method, fragment in METHODS.items()
        ],
        "reporting": {
            "primary": "final_extracted_acc1",
            "secondary": "strict_terminal_acc1",
            "auc_available": False,
            "contrasts": ["Q5", "PIS", "TRICE", "GRPO", "RLOO"],
            "reference": "Frozen-base",
            "ranking_unit": "paired_seed_by_panel",
        },
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("run_id") != RUN_ID:
        raise ValueError("run ID changed")
    if payload["dataset"].get("official_test_used") is not False:
        raise ValueError("official GSM8K test must remain sealed")
    design = payload["design"]
    if design.get("array_tasks") != 42 or design.get("training_performed") is not False:
        raise ValueError("evaluation-only 42-task contract changed")
    source = payload["source_contract"]
    original = set(source["original_validation_indices"])
    demos = {
        int(value)
        for rows in source["demonstration_indices_by_seed"].values()
        for value in rows
    }
    shot_bank = {
        int(value)
        for rows in source["shot_bank_indices_by_seed"].values()
        for value in rows
    }
    optimization = {
        int(value)
        for rows in source["optimization_indices_by_seed"].values()
        for value in rows
    }
    panels = [row["dataset_train_indices"] for row in source["panels"]]
    flat = [int(value) for panel in panels for value in panel]
    if len(panels) != PANEL_COUNT or any(len(panel) != PANEL_SIZE for panel in panels):
        raise ValueError("panel dimensions changed")
    if len(flat) != len(set(flat)):
        raise ValueError("fresh panels overlap")
    if not demos <= shot_bank:
        raise ValueError("demonstrations are not contained in the shot bank")
    if set(flat) & (original | shot_bank | optimization):
        raise ValueError(
            "fresh panels overlap source validation, demos or optimization"
        )
    if tuple(payload["design"].get("seeds") or ()) != SEEDS:
        raise ValueError("paired seeds changed")
    if [row["method"] for row in payload["methods"]] != list(METHODS):
        raise ValueError("method order changed")


def render_payload() -> str:
    return (
        "# Generated by generate_qwen3_17b_gsm8k_panel_robustness.py.\n"
        "# Edit the generator and regenerate; do not edit this YAML directly.\n"
        + yaml.safe_dump(build_payload(), sort_keys=False, width=100)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", type=Path)
    group.add_argument("--write", type=Path)
    args = parser.parse_args()
    expected = render_payload()
    if args.check:
        if yaml.safe_load(args.check.read_text(encoding="utf-8")) != build_payload():
            raise SystemExit(f"generated configuration drift: {args.check}")
        print(f"verified generated configuration: {args.check}")
    elif args.write:
        args.write.write_text(expected, encoding="utf-8")
        print(f"wrote generated configuration: {args.write}")
    else:
        print(expected, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
