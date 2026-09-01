#!/usr/bin/env python3
"""Generate the seven-seed posterior-update follow-up study."""

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


RUN_ID = "f20c9e17"
CONTROL_RUN_ID = "68078ecc"
CELL_ORDER = (
    "Q5-MORE-S32-B16-U1",
    "Q5-TOKENMEAN-S16-B16-U1",
    "PIS-Q-S8-B8-U1",
    "PIS-Q-S8-B8-U4-KL03R",
    "EXACT-Q-S8-B8-U1",
    "EXACT-Q-S8-B8-U4",
)
CONTROL_CELLS = {
    "Q5-MORE-S32-B16-U1": "Q5-AD-M-LR1e-5-U1-K16",
    "Q5-TOKENMEAN-S16-B16-U1": "Q5-AD-M-LR1e-5-U1-K16",
    "PIS-Q-S8-B8-U1": "PIS-Q-S8-B8-U4",
    "PIS-Q-S8-B8-U4-KL03R": "PIS-Q-S8-B8-U4",
    "EXACT-Q-S8-B8-U1": "PIS-Q-S8-B8-U1",
    "EXACT-Q-S8-B8-U4": "PIS-Q-S8-B8-U4",
}


def _variants(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [deepcopy(value) for value in raw]
    if isinstance(raw, dict):
        return [deepcopy(raw)]
    raise ValueError("algorithm declaration must be an object or list")


def _control_cells() -> dict[str, dict[str, Any]]:
    payload = build_control_payload()
    return {
        str(cell["cell_id"]): cell
        for raw in payload["algos"].values()
        for cell in _variants(raw)
    }


def build_payload() -> dict[str, Any]:
    control = build_control_payload()
    cells = _control_cells()
    q5 = deepcopy(cells["Q5-AD-M-LR1e-5-U1-K16"])
    pis = deepcopy(cells["PIS-Q-S8-B8-U4"])

    q5_more = deepcopy(q5)
    q5_more.update(
        cell_id="Q5-MORE-S32-B16-U1",
        algorithm_profile="q5_support_reallocation",
        batch=128,
        G=32,
    )
    q5_token_mean = deepcopy(q5)
    q5_token_mean.update(
        cell_id="Q5-TOKENMEAN-S16-B16-U1",
        algorithm_profile="barber_q5_token_mean_followup",
        responsibility_score="token_mean",
    )
    pis_u1 = deepcopy(pis)
    pis_u1.update(
        cell_id="PIS-Q-S8-B8-U1",
        algorithm_profile="l2r_exact_signed_factorial",
        iters=1,
    )
    pis_kl = deepcopy(pis)
    pis_kl.update(
        cell_id="PIS-Q-S8-B8-U4-KL03R",
        algorithm_profile="l2r_pis_rationale_kl_followup",
        policy_anchor_mode="grad_ratio",
        policy_anchor_target_ratio=0.03,
        policy_anchor_beta_min=0.0,
        policy_anchor_beta_max=10.0,
        policy_anchor_ema=0.9,
        policy_anchor_token_scope="reasoning",
    )
    exact_u1 = deepcopy(pis_u1)
    exact_u1.update(
        cell_id="EXACT-Q-S8-B8-U1",
        responsibility_refresh="inner_step",
        variational_estimator="sampled_support_importance",
        latent_mstep_objective="exact_signed_trace_answer",
    )
    exact_u4 = deepcopy(exact_u1)
    exact_u4.update(cell_id="EXACT-Q-S8-B8-U4", iters=4)

    defaults = deepcopy(control["defaults"])
    defaults.update(
        out=(
            "~/po_results/2026-09-01/reproducibility/"
            "qwen3-posterity-followups__f20c9e17"
        ),
        training_diagnostics_trace_tape=True,
        save_adapter=True,
    )
    payload = {
        "run_id": RUN_ID,
        "tag_prefix": "q3_posterity_followups",
        "diagnostic": {
            "stage": "seven_seed_posterity_followups",
            "evidence_class": "prespecified_paired_followup",
            "scientific_question": (
                "Which previously motivated posterior-update changes improve Q5 or PIS "
                "under the same seeds and protocol as the active posterity controls?"
            ),
            "control_run_id": CONTROL_RUN_ID,
            "design": {
                "cell_order": list(CELL_ORDER),
                "control_cells": deepcopy(CONTROL_CELLS),
                "paired_seeds": list(SEEDS),
                "array_tasks": len(CELL_ORDER) * len(SEEDS),
            },
            "fixed_contract": {
                "model_id": "Qwen/Qwen3-1.7B-Base",
                "dataset_id": "openai/gsm8k",
                "prompt_shots": 3,
                "optimization_questions": 128,
                "development_validation_questions": 400,
                "outer_rounds": 32,
                "adapter_surface": "attention_and_mlp_projections",
                "adapter_rank": 16,
                "answer_event": "one_terminal_marker_followed_by_tokenizer_eos",
                "official_test_used": False,
            },
            "analysis_contract": [
                "Treat paired training seed as the independent replicate.",
                "Report Final Acc@1 first, strict final accuracy second and trajectory AUC third.",
                "Use final round 32; do not select a checkpoint from outcomes.",
                "Compare Q5-MORE and token-mean Q5 only with canonical moving-reader Q5.",
                "Compare PIS-U1 and rationale-only KL only with canonical PIS-U4.",
                "Compare exact signed U1 with PIS-U1 and exact signed U4 with canonical PIS-U4.",
                "Report the registered update-pass interaction for exact signed versus ordinary PIS.",
                "Report proposal calls, backward tokens, optimizer steps and accelerator-hours separately.",
                "Do not access the official GSM8K test split.",
            ],
        },
        "defaults": defaults,
        "algos": {
            "AC-ALG1": [
                q5_more,
                q5_token_mean,
                pis_u1,
                pis_kl,
                exact_u1,
                exact_u4,
            ]
        },
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("run_id") != RUN_ID:
        raise ValueError("follow-up run ID changed")
    defaults = payload.get("defaults") or {}
    if tuple(defaults.get("seed_values") or ()) != SEEDS:
        raise ValueError("paired seed family changed")
    if tuple(defaults.get("eval_rounds") or ()) != CHECKPOINTS:
        raise ValueError("checkpoint schedule changed")
    if defaults.get("eval_partition") != "validation":
        raise ValueError("follow-up must use train-derived validation only")
    if defaults.get("lora_target_set") != "attention_mlp" or defaults.get("lora_r") != 16:
        raise ValueError("LoRA contract changed")
    if defaults.get("training_diagnostics_trace_tape") is not True:
        raise ValueError("exact signed diagnostics require the trace tape")
    configured = _variants(payload["algos"]["AC-ALG1"])
    if tuple(str(cell["cell_id"]) for cell in configured) != CELL_ORDER:
        raise ValueError("follow-up cell order changed")
    if payload["diagnostic"]["design"].get("array_tasks") != 42:
        raise ValueError("follow-up must contain exactly 42 tasks")
    if payload["diagnostic"]["fixed_contract"].get("official_test_used") is not False:
        raise ValueError("official-test prohibition is missing")

    by_id = {str(cell["cell_id"]): cell for cell in configured}
    if (by_id["Q5-MORE-S32-B16-U1"]["batch"], by_id["Q5-MORE-S32-B16-U1"]["G"]) != (128, 32):
        raise ValueError("Q5-MORE support allocation changed")
    if by_id["Q5-TOKENMEAN-S16-B16-U1"]["responsibility_score"] != "token_mean":
        raise ValueError("Q5 token-mean intervention is missing")
    if by_id["PIS-Q-S8-B8-U1"]["iters"] != 1:
        raise ValueError("PIS-U1 update count changed")
    if by_id["PIS-Q-S8-B8-U4-KL03R"].get("policy_anchor_token_scope") != "reasoning":
        raise ValueError("PIS anchor must apply only to rationale tokens")
    for cell_id, steps in (("EXACT-Q-S8-B8-U1", 1), ("EXACT-Q-S8-B8-U4", 4)):
        cell = by_id[cell_id]
        if (
            cell.get("variational_estimator") != "sampled_support_importance"
            or cell.get("latent_mstep_objective") != "exact_signed_trace_answer"
            or cell.get("responsibility_refresh") != "inner_step"
            or cell.get("iters") != steps
        ):
            raise ValueError(f"exact signed contract changed for {cell_id}")


def render_payload() -> str:
    header = (
        "# Generated by generate_qwen3_17b_posterity_followups.py.\n"
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
