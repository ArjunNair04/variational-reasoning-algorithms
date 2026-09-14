"""Replicate the three positional cells with only the E-step answer reader frozen."""

import argparse
from pathlib import Path

import yaml

import generate_qwen3_17b_q5_prior_segments as moving
from generate_qwen3_17b_q5_prior_exponent import runtime_configs as prepare_runtime_configs

RUN_ID = "7a31c9e2"
SEEDS, CHECKPOINTS = moving.SEEDS, moving.CHECKPOINTS
CONTROL_RUN_ID, CONTROL_COMMIT = moving.CONTROL_RUN_ID, moving.CONTROL_COMMIT
CONTROL_CELL = "Q5-AD-F-LR1e-5-U1-K16"
SETTINGS = {name + "-F": value for name, value in moving.SETTINGS.items()}
CELL_ORDER = tuple(SETTINGS)
MOVING_COMMIT = "037b1d38b484f207e5bff9cf437a3ea421860f53"
MOVING_JOB = "7413571"
MOVING_CONFIG_SHA256 = "c73b7c65c97aff185b3c73ab04019ac2ed1be9f31d1bd30fa17ebdd2c2a13084"


def build_payload():
    payload = moving.build_payload()
    payload.update(run_id=RUN_ID, tag_prefix="q3_q5_prior_segments_frozen")
    payload["defaults"]["out"] = (
        f"~/po_results/2026-09-15/q5-prior-segments-frozen/qwen3-q5-prior-segments-frozen__{RUN_ID}"
    )
    diagnostic = payload["diagnostic"]
    diagnostic.pop("supersedes_unsubmitted_run")
    diagnostic.update(
        stage="three_seed_frozen_reader_positional_prior_screen",
        control_cell=CONTROL_CELL,
        reader_extension_approval="User requested all three positional variants with frozen reader and three seeds on 2026-09-15.",
        moving_comparison=dict(run_id=moving.RUN_ID, execution_commit=MOVING_COMMIT,
            source_job_id=MOVING_JOB, configuration_sha256=MOVING_CONFIG_SHA256,
            cell_mapping=dict(zip(CELL_ORDER, moving.CELL_ORDER, strict=True))),
    )
    diagnostic["design"]["cell_order"] = list(CELL_ORDER)
    diagnostic["fixed_contract"]["reader"] = "frozen_base"
    diagnostic["analysis_contract"][2] = (
        "Pair each new cell with historical frozen-reader Q5; compare early versus late and each positional cell "
        "versus concurrent uniform075. Separately report frozen-minus-moving reader contrasts for each positional "
        "setting and the unattenuated Q5 control, using receipt-bound same-seed runs c7e32a91 and 68078ecc."
    )
    diagnostic["analysis_contract"].append(
        "Only the E-step answer factor uses the frozen base. The rationale prior and marker use the current policy; "
        "generation and the complete joint M-step remain trainable. Historical reader contrasts are not bit-exact replays."
    )
    for cell, name in zip(payload["algos"]["AC-ALG1"], CELL_ORDER, strict=True):
        cell.update(cell_id=name, algorithm_profile="q5_prior_segments_frozen",
                    responsibility_answer_policy="frozen_base")
    return payload


def runtime_configs():
    return prepare_runtime_configs(build_payload())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-check", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", type=Path)
    mode.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.runtime_check:
        from ac_alg1 import _validate_ac_alg1_run_config
        for config in runtime_configs():
            _validate_ac_alg1_run_config(config, diagnostics_fn=lambda _: None, diagnostics_probe_fn=None)
    payload = build_payload()
    if args.check:
        if yaml.safe_load(args.check.read_text()) != payload:
            raise SystemExit("frozen-reader prior-segment configuration changed")
    else:
        args.write.write_text("# Generated; edit generate_qwen3_17b_q5_prior_segments_frozen.py.\n"
                              + yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
