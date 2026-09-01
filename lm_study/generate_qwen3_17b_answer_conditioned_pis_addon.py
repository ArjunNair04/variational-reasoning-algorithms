#!/usr/bin/env python3
"""Generate the seven-seed answer-conditioned PIS add-on."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from generate_qwen3_17b_final_method_confirmation import (
    CHECKPOINTS,
    RUN_ID as CONTROL_RUN_ID,
    SEEDS,
    build_payload as build_control_payload,
)


RUN_ID = "613ffde2"
CELL_ID = "AC-PIS-S8-B8-U4"
CONTROL_CELL_ID = "PIS-S8-B8-U4"
CONTROL_VALIDATOR_JOB = "7191863"
SCRIPT_DIR = Path(__file__).resolve().parent


def _variants(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [dict(value) for value in raw]
    if isinstance(raw, dict):
        return [dict(raw)]
    raise ValueError("algorithm declaration must be an object or list")


def _control_cell(payload: dict[str, Any]) -> dict[str, Any]:
    matches = [
        deepcopy(cell)
        for raw in (payload.get("algos") or {}).values()
        for cell in _variants(raw)
        if cell.get("cell_id") == CONTROL_CELL_ID
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one native PIS control cell")
    return matches[0]


def build_payload() -> dict[str, Any]:
    control = build_control_payload()
    defaults = deepcopy(control["defaults"])
    defaults["out"] = (
        "~/po_results/2026-08-18/final-comparison/"
        "answer-conditioned-pis-addon__613ffde2"
    )
    cell = _control_cell(control)
    cell.update(
        {
            "cell_id": CELL_ID,
            "algorithm_profile": "l2r_answer_conditioned_importance",
            "proposal_prompt": "answer_derive",
            "variational_estimator": "answer_conditioned_importance",
        }
    )
    payload = {
        "run_id": RUN_ID,
        "tag_prefix": "q3_answer_conditioned_pis_addon",
        "diagnostic": {
            "stage": "answer_conditioned_pis_seven_seed_addon",
            "evidence_class": "preregistered_paired_mechanism_addon",
            "require_protocol_outcome_diagnostics": True,
            "partitions": {
                "optimization_source": "gsm8k_train_optimization_subset",
                "development_validation_source": "gsm8k_train_validation_subset",
                "official_test_used": False,
            },
            "scientific_question": (
                "Does answer-conditioned proposal sampling with exact sample-time "
                "importance correction improve native PIS under the same modern "
                "three-shot protocol?"
            ),
            "matched_control": {
                "registry_id": "qwen3_final_method_confirmation_20260818",
                "run_id": CONTROL_RUN_ID,
                "cell_id": CONTROL_CELL_ID,
                "validator_job": CONTROL_VALIDATOR_JOB,
            },
            "design": {
                "cell_order": [CELL_ID],
                "paired_confirmation_seeds": list(SEEDS),
                "array_tasks": len(SEEDS),
                "checkpoint_rule": "final_round_32",
            },
            "intervention": {
                "proposal_prompt": "question_to_answer_derive",
                "variational_estimator": (
                    "prior_importance_to_answer_conditioned_importance"
                ),
                "unchanged_responsibility_refresh": "outer_round",
                "unchanged_support": "fresh_empirical_multiset_S8",
                "unchanged_m_step": "four_current_policy_joint_updates",
            },
            "reporting_order": [
                "final_extracted_answer_accuracy",
                "final_strict_terminal_accuracy",
                "normalized_extracted_trajectory_auc",
            ],
            "analysis_contract": [
                "Reuse only the contemporaneous native PIS cells from run 978b99c8 as the paired control.",
                "Treat paired training seed as the independent replicate.",
                "Use a nested paired bootstrap over training seeds and matched evaluation questions.",
                "Report final extracted Acc@1 first, strict final accuracy second and extracted trajectory AUC third.",
                "Require finite proposal densities and the exact corrected-logit identity in every round.",
                "Do not interpret the proposal change without its mathematically required importance correction as a separate factor.",
                "Do not access the official GSM8K test split.",
            ],
            "decision_rules": {
                "superiority": (
                    "advance AC-PIS over native PIS only if the paired 95 percent "
                    "interval for final extracted Acc@1 is positive"
                ),
                "endpoint_noninferiority": (
                    "otherwise retain it only as mechanism-positive if the final "
                    "extracted lower bound is above minus one point"
                ),
                "trajectory_safety": (
                    "the extracted-AUC lower bound must exceed minus one point"
                ),
                "strict_safety": (
                    "the strict-final lower bound must exceed minus two points"
                ),
                "seed_consistency": "at least five of seven final effects are positive",
                "official_test_used": False,
            },
        },
        "defaults": defaults,
        "algos": {"AC-ALG1": cell},
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: dict[str, Any]) -> None:
    control = build_control_payload()
    defaults = payload.get("defaults") or {}
    if payload.get("run_id") != RUN_ID:
        raise ValueError("answer-conditioned PIS run ID changed")
    if tuple(defaults.get("seed_values") or ()) != SEEDS:
        raise ValueError("answer-conditioned PIS seed family changed")
    if tuple(defaults.get("eval_rounds") or ()) != CHECKPOINTS:
        raise ValueError("answer-conditioned PIS checkpoint schedule changed")
    if defaults.get("eval_partition") != "validation":
        raise ValueError("answer-conditioned PIS must use validation only")
    if defaults.get("save_adapter") is not True:
        raise ValueError("answer-conditioned PIS must retain trained adapters")
    if defaults.get("lora_target_set") != "attention_mlp":
        raise ValueError("answer-conditioned PIS LoRA surface changed")
    control_defaults = deepcopy(control["defaults"])
    control_defaults.pop("out")
    candidate_defaults = deepcopy(defaults)
    candidate_defaults.pop("out")
    if candidate_defaults != control_defaults:
        raise ValueError("answer-conditioned PIS changed common protocol defaults")
    cell = dict(payload.get("algos", {}).get("AC-ALG1") or {})
    source = _control_cell(control)
    expected_changes = {
        "cell_id": (CONTROL_CELL_ID, CELL_ID),
        "algorithm_profile": (
            "l2r_common_factorial",
            "l2r_answer_conditioned_importance",
        ),
        "proposal_prompt": ("question", "answer_derive"),
        "variational_estimator": (
            "prior_importance",
            "answer_conditioned_importance",
        ),
    }
    changed = {
        key: (source.get(key), cell.get(key))
        for key in set(source) | set(cell)
        if source.get(key) != cell.get(key)
    }
    if changed != expected_changes:
        raise ValueError(f"answer-conditioned PIS axes changed: {changed}")
    if cell.get("responsibility_refresh") != "outer_round":
        raise ValueError("answer-conditioned PIS must preserve outer-round E-step")
    if ((payload.get("diagnostic") or {}).get("partitions") or {}).get(
        "official_test_used"
    ) is not False:
        raise ValueError("official-test prohibition is missing")


def render_payload() -> str:
    header = (
        "# Generated by generate_qwen3_17b_answer_conditioned_pis_addon.py.\n"
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
