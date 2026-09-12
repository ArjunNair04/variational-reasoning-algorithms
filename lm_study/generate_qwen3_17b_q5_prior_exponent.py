"""Freeze the three-seed Q5 prior-exponent by answer-reader screen."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from generate_qwen3_17b_selected_method_posterity import (
    CHECKPOINTS, SEEDS as ALL_SEEDS, build_payload as controls,
)

RUN_ID = "e97c3a20"
SEEDS = ALL_SEEDS[:3]
CONTROL_RUN_ID = "68078ecc"
CONTROL_COMMIT = "4a9a9cafc6b17cc18e7c1da8a422e847daeb4746"
CONTROL_CELLS = {"M": "Q5-AD-M-LR1e-5-U1-K16", "F": "Q5-AD-F-LR1e-5-U1-K16"}
CELL_ORDER = tuple(f"Q5-AD-{reader}-TAU{tau}" for reader in ("M", "F") for tau in ("0p5", "0"))


def build_payload():
    source = controls()
    originals = {c["cell_id"]: c for c in source["algos"]["AC-ALG1"]}
    cells = []
    for reader in ("M", "F"):
        for label, exponent in (("0p5", 0.5), ("0", 0.0)):
            cell = deepcopy(originals[CONTROL_CELLS[reader]])
            cell.update(cell_id=f"Q5-AD-{reader}-TAU{label}",
                        algorithm_profile="q5_prior_exponent",
                        responsibility_prior_exponent=exponent)
            cells.append(cell)
    defaults = deepcopy(source["defaults"])
    defaults.update(seed_values=list(SEEDS), seeds=len(SEEDS),
        out=f"~/po_results/2026-09-12/q5-prior-exponent/qwen3-q5-prior-exponent__{RUN_ID}")
    return {
        "run_id": RUN_ID, "tag_prefix": "q3_q5_prior_exponent",
        "diagnostic": {
            "stage": "three_seed_prior_exponent_reader_screen",
            "evidence_class": "development_screen_historical_paired_controls",
            "control_run_id": CONTROL_RUN_ID, "control_commit": CONTROL_COMMIT,
            "control_cells": CONTROL_CELLS,
            "design": {"array_tasks": 12, "cell_order": list(CELL_ORDER), "paired_seeds": list(SEEDS)},
            "weighting": "softmax(answer_reader_logprob + tau * current_trace_logprob)",
            "fixed_contract": {"official_test_used": False, "answer_target": "answer_plus_eos",
                "proposal": "answer_derive", "mstep": "unchanged_joint_weighted_mle"},
            "analysis_contract": [
                "Final extracted Acc@1, strict final Acc@1, normalized extracted AUC rounds 0-32, then strict AUC.",
                "Pair each new cell with its same-seed same-reader tau=1 historical control.",
                "Compare frozen minus moving at each tau; report per-seed changes and descriptive paired intervals.",
                "Three seeds are screening evidence, not confirmation; no checkpoint selection or automatic extra jobs.",
                "Nominate at most one tau-reader setting with positive mean final change and no mean strict-final loss versus its reader control; AUC is secondary, not a veto.",
                "Report all settings, concentration, weight-length correlation, tau1 top-trace changes on identical support, backward tokens and compute.",
                "Historical controls remain historical: unchanged default code path is tested, but CUDA and execution date may differ.",
            ],
        },
        "defaults": defaults, "algos": {"AC-ALG1": cells},
    }


def runtime_configs():
    from dataclasses import replace
    import inspect
    from ac_alg1 import run_ac_alg1
    from trainer_config import ACAlg1RunConfig
    from experiment_config import ACAlg1BatchAllocation
    from run_yaml import _prepare_cells
    values = {k: v.default for k, v in inspect.signature(run_ac_alg1).parameters.items()}
    base = ACAlg1RunConfig.from_call(values)
    payload = build_payload()
    defaults = payload["defaults"]
    cells = _prepare_cells(payload, only=None, run_id=payload["run_id"], defaults=defaults)
    for cell in cells:
        allocation = ACAlg1BatchAllocation.from_budget(
            batch=cell.axes["batch"], generations=cell.axes["G"], labelled_fraction=cell.axes["labelled_frac"])
        values = dict(rounds=defaults["rounds"], L_batch=allocation.labelled, U_batch=allocation.answer_only,
            G_label=cell.axes["G"], G_answer_only=cell.axes["G"], inner_steps=cell.axes["iters"],
            answer_event_mode=defaults["answer_event_mode"], diagnostics_level=defaults["training_diagnostics_level"],
            diagnostics_trace_tape=defaults["training_diagnostics_trace_tape"], question_sampling=defaults["question_sampling"])
        values.update({k: v for k, v in cell.axes.items() if k in ACAlg1RunConfig.__dataclass_fields__})
        yield replace(base, **values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-check", action="store_true")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write", type=Path)
    modes.add_argument("--check", type=Path)
    args = parser.parse_args()
    payload = build_payload()
    if args.runtime_check:
        from ac_alg1 import _validate_ac_alg1_run_config
        for config in runtime_configs():
            _validate_ac_alg1_run_config(config, diagnostics_fn=lambda _: None, diagnostics_probe_fn=None)
    if args.check:
        if yaml.safe_load(args.check.read_text()) != payload:
            raise SystemExit("frozen prior-exponent configuration changed")
    else:
        args.write.write_text("# Generated; edit generate_qwen3_17b_q5_prior_exponent.py.\n" + yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
