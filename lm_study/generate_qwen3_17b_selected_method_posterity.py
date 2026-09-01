#!/usr/bin/env python3
"""Generate the selected-method seven-seed posterity replay."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import yaml

from generate_qwen3_17b_final_method_confirmation import (
    CHECKPOINTS,
    SEEDS,
    build_payload as build_source_payload,
)
from generate_qwen3_17b_answer_conditioned_pis_addon import (
    build_payload as build_answer_pis_payload,
)


RUN_ID = "68078ecc"
SOURCE_RUN_ID = "978b99c8"
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_CONFIG = SCRIPT_DIR / "experiments_qwen3_17b_final_method_confirmation.yaml"
CELL_ORDER = (
    "CTRL-base",
    "Q5-Q-LR1e-5-U1-K16",
    "Q5-AD-M-LR1e-5-U1-K16",
    "Q5-AD-F-LR1e-5-U1-K16",
    "Q5-AD-M-ESS50-LR1e-5-U1-K16",
    "Q5-AD-M-KL03R-LR1e-5-U1-K16",
    "Q5-AD-M-T1p2-LR1e-5-U1-K16",
    "PIS-Q-S8-B8-U4",
    "PIS-AD-S8-B8-U4",
    "REST-LR1e-5-E1-I4",
    "STAR-LR3e-6-E2",
    "TRICE-LR1e-4-CV",
    "RLOO-S16-B8-U4",
)
METHOD_ORDER = (
    "base",
    "AC-ALG1",
    "AC-ALG1",
    "AC-ALG1",
    "AC-ALG1",
    "AC-ALG1",
    "AC-ALG1",
    "AC-ALG1",
    "AC-ALG1",
    "ReST-EM",
    "STaR",
    "TRICE",
    "RLOO",
)
ANSWER_DERIVED_CELLS = {
    "Q5-AD-M-LR1e-5-U1-K16",
    "Q5-AD-F-LR1e-5-U1-K16",
    "Q5-AD-M-ESS50-LR1e-5-U1-K16",
    "Q5-AD-M-KL03R-LR1e-5-U1-K16",
    "Q5-AD-M-T1p2-LR1e-5-U1-K16",
    "PIS-AD-S8-B8-U4",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _variants(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [deepcopy(value) for value in raw]
    if isinstance(raw, dict):
        return [deepcopy(raw)]
    raise ValueError("algorithm declaration must be an object or list")


def _source_cells() -> dict[str, tuple[str, dict[str, Any]]]:
    payload = build_source_payload()
    cells: dict[str, tuple[str, dict[str, Any]]] = {}
    for method, raw in payload["algos"].items():
        for cell in _variants(raw):
            cell_id = str(cell["cell_id"])
            cells[cell_id] = (str(method), cell)
    return cells


def build_payload() -> dict[str, Any]:
    source = build_source_payload()
    cells = _source_cells()
    base = deepcopy(cells["CTRL-base"][1])
    q5_moving = deepcopy(cells["Q5-LR1e-5-U1-K16"][1])
    q5_moving["cell_id"] = "Q5-AD-M-LR1e-5-U1-K16"
    q5_moving["algorithm_profile"] = "barber_q5_control"
    q5_moving["proposal_temperature"] = 1.0
    q5_moving["policy_anchor_target_ratio"] = None
    q5_moving["policy_anchor_beta_min"] = 0.0
    q5_moving["policy_anchor_beta_max"] = 10.0
    q5_moving["policy_anchor_ema"] = 0.9
    q5_moving["policy_anchor_token_scope"] = "objective"
    q5_frozen = deepcopy(q5_moving)
    q5_frozen["cell_id"] = "Q5-AD-F-LR1e-5-U1-K16"
    q5_frozen["responsibility_answer_policy"] = "frozen_base"
    q5_ess = deepcopy(q5_moving)
    q5_ess["cell_id"] = "Q5-AD-M-ESS50-LR1e-5-U1-K16"
    q5_ess["responsibility_ess_floor"] = 0.5
    q5_kl = deepcopy(q5_moving)
    q5_kl["cell_id"] = "Q5-AD-M-KL03R-LR1e-5-U1-K16"
    q5_kl["policy_anchor_mode"] = "grad_ratio"
    q5_kl["policy_anchor_target_ratio"] = 0.03
    q5_kl["policy_anchor_token_scope"] = "reasoning"
    q5_temperature = deepcopy(q5_moving)
    q5_temperature["cell_id"] = "Q5-AD-M-T1p2-LR1e-5-U1-K16"
    q5_temperature["proposal_temperature"] = 1.2
    q5_question = deepcopy(q5_moving)
    q5_question["cell_id"] = "Q5-Q-LR1e-5-U1-K16"
    q5_question["algorithm_profile"] = "barber_source"
    q5_question["proposal_prompt"] = "question"
    pis_question = deepcopy(cells["PIS-S8-B8-U4"][1])
    pis_question["cell_id"] = "PIS-Q-S8-B8-U4"
    answer_pis = build_answer_pis_payload()
    pis_answer = deepcopy(answer_pis["algos"]["AC-ALG1"])
    pis_answer["cell_id"] = "PIS-AD-S8-B8-U4"

    algos = {
        "base": base,
        "AC-ALG1": [
            q5_question,
            q5_moving,
            q5_frozen,
            q5_ess,
            q5_kl,
            q5_temperature,
            pis_question,
            pis_answer,
        ],
        "ReST-EM": deepcopy(cells["REST-LR1e-5-E1-I4"][1]),
        "STaR": deepcopy(cells["STAR-LR3e-6-E2"][1]),
        "TRICE": deepcopy(cells["TRICE-LR1e-4-CV"][1]),
        "RLOO": deepcopy(cells["RLOO-S16-B8-U4"][1]),
    }
    defaults = deepcopy(source["defaults"])
    defaults["out"] = (
        "~/po_results/2026-08-29/reproducibility/"
        "qwen3-selected-method-posterity__68078ecc"
    )

    payload = {
        "run_id": RUN_ID,
        "tag_prefix": "q3_selected_method_posterity",
        "diagnostic": {
            "stage": "selected_method_seven_seed_posterity_replay",
            "evidence_class": "same_seed_current_code_reproducibility_replay",
            "partitions": deepcopy(source["diagnostic"]["partitions"]),
            "scientific_question": (
                "Do selected Q5, PIS, ReST-EM, STaR, TRICE-CV and RLOO settings "
                "reproduce on the original seven paired seeds, and how do regular "
                "versus answer-derived proposals and a frozen Q5 reader change the "
                "posterior-SFT methods?"
            ),
            "provenance": {
                "source_run_id": SOURCE_RUN_ID,
                "source_config": SOURCE_CONFIG.name,
                "source_config_sha256": _sha256(SOURCE_CONFIG),
                "selection_rule": "best previously frozen setting per method",
                "controlled_interventions": {
                    "ess_floor": (
                        "Repeat the historical 0.50 minimum finite-support ESS as "
                        "one isolated Q5 arm."
                    ),
                    "proposal_temperature": (
                        "Compare canonical Q5 temperature 1.0 with one isolated "
                        "temperature-1.2 answer-derived proposal arm."
                    ),
                    "policy_anchor": (
                        "Apply the established 0.03 adaptive gradient-ratio anchor "
                        "only to rationale tokens in one isolated Q5 arm."
                    ),
                },
            },
            "design": {
                "cell_order": list(CELL_ORDER),
                "paired_confirmation_seeds": list(SEEDS),
                "methods": len(CELL_ORDER),
                "seeds_per_method": len(SEEDS),
                "array_tasks": len(CELL_ORDER) * len(SEEDS),
                "checkpoint_rule": "final_round_32",
                "replay_cells": [
                    "CTRL-base",
                    "Q5-AD-M-LR1e-5-U1-K16",
                    "PIS-Q-S8-B8-U4",
                    "PIS-AD-S8-B8-U4",
                    "REST-LR1e-5-E1-I4",
                    "STAR-LR3e-6-E2",
                    "TRICE-LR1e-4-CV",
                    "RLOO-S16-B8-U4",
                ],
                "extended_prompt_contrast": "Q5-Q-LR1e-5-U1-K16",
                "new_paired_reader_ablation": "Q5-AD-F-LR1e-5-U1-K16",
                "new_q5_stability_ablation": [
                    "Q5-AD-M-ESS50-LR1e-5-U1-K16",
                    "Q5-AD-M-KL03R-LR1e-5-U1-K16",
                    "Q5-AD-M-T1p2-LR1e-5-U1-K16",
                ],
            },
            "fixed_contract": deepcopy(source["diagnostic"]["fixed_contract"]),
            "reporting_order": deepcopy(source["diagnostic"]["reporting_order"]),
            "analysis_contract": [
                "Treat paired training seed as the independent replicate.",
                "Report Final Acc@1 first, strict final accuracy second and trajectory AUC third.",
                "Use final round 32; do not select a checkpoint from outcomes.",
                "Compare regular and answer-derived prompts within Q5 and PIS.",
                "Compare moving and frozen Q5 readers only under answer-derived proposals.",
                "Compare each Q5 ESS, KL and proposal-temperature arm only with canonical moving-reader Q5.",
                "Treat differences from the source run as reproducibility measurements, not hyperparameter selection.",
                "Do not access the official GSM8K test split.",
            ],
        },
        "defaults": defaults,
        "algos": algos,
    }
    validate_payload(payload)
    return payload


def _configured(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (str(method), cell)
        for method, raw in payload["algos"].items()
        for cell in _variants(raw)
    ]


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("run_id") != RUN_ID:
        raise ValueError("posterity run ID changed")
    defaults = payload.get("defaults") or {}
    if tuple(defaults.get("seed_values") or ()) != SEEDS:
        raise ValueError("posterity seed family changed")
    if tuple(defaults.get("eval_rounds") or ()) != CHECKPOINTS:
        raise ValueError("posterity checkpoint schedule changed")
    if defaults.get("eval_partition") != "validation":
        raise ValueError("posterity replay must use train-derived validation only")
    if defaults.get("lora_target_set") != "attention_mlp" or defaults.get("lora_r") != 16:
        raise ValueError("posterity LoRA contract changed")
    configured = _configured(payload)
    ids = tuple(str(cell["cell_id"]) for _method, cell in configured)
    methods = tuple(method for method, _cell in configured)
    if ids != CELL_ORDER or methods != METHOD_ORDER:
        raise ValueError(f"posterity cell mapping changed: {methods} {ids}")
    question = configured[1][1]
    moving = configured[2][1]
    frozen = configured[3][1]
    ess = configured[4][1]
    kl = configured[5][1]
    temperature = configured[6][1]
    moving_axes = {key: value for key, value in moving.items() if key != "cell_id"}
    frozen_axes = {key: value for key, value in frozen.items() if key != "cell_id"}
    if moving_axes.get("responsibility_answer_policy") != "current":
        raise ValueError("moving-reader Q5 changed")
    frozen_axes["responsibility_answer_policy"] = "current"
    if moving_axes != frozen_axes:
        raise ValueError("Q5 reader ablation changes more than the answer reader")
    q5_controls = (
        (ess, {"responsibility_ess_floor": 0.5}),
        (
            kl,
            {
                "policy_anchor_mode": "grad_ratio",
                "policy_anchor_target_ratio": 0.03,
                "policy_anchor_token_scope": "reasoning",
            },
        ),
        (temperature, {"proposal_temperature": 1.2}),
    )
    for control, expected_changes in q5_controls:
        control_axes = {key: value for key, value in control.items() if key != "cell_id"}
        changed = {
            key
            for key in set(moving_axes) | set(control_axes)
            if moving_axes.get(key) != control_axes.get(key)
        }
        if changed != set(expected_changes):
            raise ValueError(f"Q5 control changed unexpected axes: {changed}")
        if any(control_axes[key] != value for key, value in expected_changes.items()):
            raise ValueError(f"Q5 control has incorrect values: {expected_changes}")
    question_axes = {key: value for key, value in question.items() if key != "cell_id"}
    question_axes["algorithm_profile"] = moving_axes["algorithm_profile"]
    question_axes["proposal_prompt"] = moving_axes["proposal_prompt"]
    if question_axes != moving_axes:
        raise ValueError("Q5 prompt contrast changes more than prompt and validation profile")
    pis_question = configured[7][1]
    pis_answer = configured[8][1]
    allowed_pis_changes = {
        "cell_id",
        "algorithm_profile",
        "proposal_prompt",
        "variational_estimator",
    }
    changed = {
        key
        for key in set(pis_question) | set(pis_answer)
        if pis_question.get(key) != pis_answer.get(key)
    }
    if changed != allowed_pis_changes:
        raise ValueError(f"PIS prompt contrast changed unexpected axes: {changed}")
    if pis_question.get("proposal_prompt") != "question":
        raise ValueError("regular PIS proposal prompt changed")
    if pis_answer.get("proposal_prompt") != "answer_derive":
        raise ValueError("answer-derived PIS proposal prompt changed")
    if pis_question.get("variational_estimator") != "prior_importance":
        raise ValueError("regular PIS estimator changed")
    if pis_answer.get("variational_estimator") != "answer_conditioned_importance":
        raise ValueError("answer-derived PIS correction changed")
    if payload["diagnostic"]["fixed_contract"].get("official_test_used") is not False:
        raise ValueError("official-test prohibition is missing")
    if payload["diagnostic"]["design"].get("array_tasks") != 91:
        raise ValueError("posterity replay must contain exactly 91 tasks")


def render_payload() -> str:
    header = (
        "# Generated by generate_qwen3_17b_selected_method_posterity.py.\n"
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
