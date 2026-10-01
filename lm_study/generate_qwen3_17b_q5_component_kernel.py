"""Freeze the fixed-Q5-weight continuation-kernel ablation (three new tasks)."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from generate_qwen3_17b_q5_prefix_continuations import build_payload as prefix_payload
from generate_qwen3_17b_q5_prior_exponent import SEEDS, runtime_configs as prepare_runtime_configs

RUN_ID = "b6c4e921"
CELL = "Q5-KERNEL25"
CONTROL_CELL = "Q5-INDEPENDENT16"
CONTROL_RUN = "f812ca40"
CONTROL_JOB = "7483302"
CONTROL_COMMIT = "4900a9f8413e191e951d534f9fa4d179431defc9"
CONTROL_CONFIG_SHA256 = "5cf9f3e9846930de233b56b02d7ded597ad3f983099784b73323ea34ce52c6ea"


def build_payload():
    source = prefix_payload()
    cell = deepcopy(source["algos"]["AC-ALG1"][0])
    cell.update(cell_id=CELL, algorithm_profile="q5_component_kernel", component_kernel_epsilon=.25)
    defaults = deepcopy(source["defaults"])
    defaults["out"] = f"~/po_results/2026-10-01/q5-component-kernel/qwen3-q5-component-kernel__{RUN_ID}"
    return {
        "run_id": RUN_ID, "tag_prefix": "q3_q5_kernel",
        "diagnostic": {
            "stage": "three_seed_fixed_weight_component_kernel",
            "design": {"array_tasks": 3, "cell_order": [CELL], "paired_seeds": list(SEEDS)},
            "control": {"run_id": CONTROL_RUN, "cell_id": CONTROL_CELL, "job_id": CONTROL_JOB,
                "tasks": [1, 2, 3], "execution_commit": CONTROL_COMMIT,
                "config_sha256": CONTROL_CONFIG_SHA256, "out": source["defaults"]["out"],
                "reuse": "Reuse only receipt-validated existing ordinary-Q5 cells. Local disabled-path differential tests check default parity; runtime receipts must match before paired claims. No control reruns."},
            "round_zero": deepcopy(source["diagnostic"]["round_zero"]),
            "distribution": "q_epsilon(h) = sum_s w_s [(1-epsilon) delta_hs(h) + epsilon K_s(h)], epsilon=0.25. Q5 joint weights are computed on the original buffer only.",
            "kernel": "Retain floor(n/2) native reasoning tokens before the first marker. Draw one suffix using the current outer-round model, answer-derived prompt, temperature 1, top_k 0, top_p 1. Cap total prefix+suffix at the original generation budget. Extract the first native marker boundary using the existing Q5 mapping. Invalid-boundary or empty-prefix draws map to the original trace; no rejection/resampling. This defines a normalised pushforward kernel, not the raw suffix density.",
            "mstep": "Parent mass 0.75*w_s and one child mass 0.25*w_s; hold both fixed for the sole M-step. Children are neither rescored nor admitted to the persistent buffer. Train rationale+marker+gold answer+EOS under the question-only prompt, unchanged from Q5.",
            "rng": "Dedicated SHA256-derived seed per training seed and outer round, torch fork_rng restores global CPU/CUDA state. No random draws or changed execution path at epsilon=0.",
            "fixed_contract": {"official_test_used": False, "answer_target": "answer_plus_eos",
                "proposal": "answer_derive", "reader": "current", "inner_steps": 1},
            "analysis_contract": [
                "Final extracted Acc@1 first, strict final second, normalised extracted AUC rounds 0-32 third, then strict AUC.",
                "Validate source commit, YAML, receipts, terminal logs, training-only split, all checkpoints, native-token lineage, inherited mass and absence of kernel rows from persistent buffers.",
                "Show all three paired seed differences and descriptive paired 95% t intervals versus Q5-INDEPENDENT16. Development screen, no automatic follow-up.",
                "Nominate for discussion only if mean final gain is positive without a mean strict-final loss; report AUC, fallback mass and actual changed mass alongside it.",
                "Count kernel continuations in total training generations; copied prefixes are not generated tokens but are backward tokens. Report additional generation, backward tokens, elapsed hours, and parent-weight concentration separately.",
                "This changes variational components at fixed Q5 weights. It is a valid variational choice with a Monte Carlo M-step, not a guaranteed improved E-step or a measured ELBO increase. Kernel draws add compute; this is not a compute-matched comparison.",
            ],
        },
        "defaults": defaults, "algos": {"AC-ALG1": [cell]},
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
    payload = build_payload()
    if args.check:
        if yaml.safe_load(args.check.read_text()) != payload:
            raise SystemExit("frozen component-kernel protocol changed")
    else:
        args.write.write_text("# Generated; edit generate_qwen3_17b_q5_component_kernel.py.\n" + yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
