"""Freeze an M-step-only length screen, with no E-step or prompt changes."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from generate_qwen3_17b_q5_prior_exponent import (
    CONTROL_COMMIT, CONTROL_RUN_ID, SEEDS, runtime_configs as prepare_runtime_configs,
)
from generate_qwen3_17b_selected_method_posterity import CHECKPOINTS, build_payload as controls

RUN_ID = "d61a0c84"
CONTROL_CELLS = {
    "Q5-MSTEP-TOKENMEAN": "Q5-AD-M-LR1e-5-U1-K16",
    "PIS-MSTEP-TOKENMEAN": "PIS-Q-S8-B8-U4",
}
CELL_ORDER = tuple(CONTROL_CELLS)


def build_payload():
    source = controls()
    originals = {cell["cell_id"]: cell for cell in source["algos"]["AC-ALG1"]}
    cells = []
    for cell_id, original_id in CONTROL_CELLS.items():
        cell = deepcopy(originals[original_id])
        cell.update(cell_id=cell_id, algorithm_profile="mstep_length_normalisation",
                    latent_mstep_objective="joint_token_mean")
        cells.append(cell)
    defaults = deepcopy(source["defaults"])
    defaults.update(seed_values=list(SEEDS), seeds=len(SEEDS),
                    out=f"~/po_results/2026-10-16/mstep-length-normalisation/qwen3-mstep-length__{RUN_ID}")
    return {
        "run_id": RUN_ID, "tag_prefix": "q3_mstep_length",
        "diagnostic": {
            "stage": "three_seed_mstep_length_screen",
            "evidence_class": "development_screen_historical_paired_controls",
            "control_run_id": CONTROL_RUN_ID, "control_commit": CONTROL_COMMIT,
            "control_cells": CONTROL_CELLS,
            "design": {"array_tasks": 6, "cell_order": list(CELL_ORDER), "paired_seeds": list(SEEDS)},
            "intervention": "Replace sum_s w_s log p_theta(h_s,a+|q) with sum_s w_s log p_theta(h_s,a+|q)/T_s; T_s counts shifted objective-mask positions including marker, answer and EOS, excluding prompt and padding.",
            "unchanged": "Posterior formulas, answer/prior policies, refresh schedules, proposals, buffers, question schedule, inner steps, Adam and learning rate, target boundaries and evaluation.",
            "audit": "Receipt-bound audit of 192 historical rounds and 7868 retained rows; see docs/experiments/mstep_length_normalisation. Arithmetic parser flags are not verified errors. Q5 snapshots are post-update.",
            "release_gate": "Audit reviewed and result analyzer prepared; source-control runtime/provenance equivalence and deployment wiring still required before submission. No automatic release.",
            "analysis_contract": [
                "Final extracted Acc@1 first, strict final second, normalized extracted AUC rounds 0-32 third; also strict AUC.",
                "Compare each method only with its own receipt-verified same-seed control; Q5 and PIS retain different question/update budgets.",
                "Use final round 32, never select the best checkpoint; include every seed and paired differences with descriptive 95% t intervals.",
                "Report lengths, posterior concentration, weight-length correlation, gradient norms, applied update norms, formatting, backward/generated tokens and accelerator hours.",
                "Unchanged weights means unchanged E-step at identical parameters/support; subsequent learned policies and their weights may differ.",
                "This changes relative per-trace loss scaling and overall objective scale. Adam need not turn a 100-fold smaller loss into a 100-fold smaller update.",
                "Do not attribute gains solely to length preference without a separate scale-matched follow-up; do not add that follow-up automatically.",
                "No official-test access, no new posterior temperature, no prompt change, no added KL, no automatic confirmation.",
            ],
        },
        "defaults": defaults, "algos": {"AC-ALG1": cells},
    }


def runtime_configs():
    return prepare_runtime_configs(build_payload())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-check", action="store_true")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.runtime_check:
        from ac_alg1 import _validate_ac_alg1_run_config
        for config in runtime_configs():
            _validate_ac_alg1_run_config(config, diagnostics_fn=lambda _: None, diagnostics_probe_fn=None)
    if args.check:
        if yaml.safe_load(args.check.read_text()) != build_payload():
            raise SystemExit("frozen M-step length protocol changed")
    else:
        args.write.write_text("# Generated; edit generate_qwen3_17b_mstep_length_normalisation.py.\n"
                              + yaml.safe_dump(build_payload(), sort_keys=False))


if __name__ == "__main__":
    main()
