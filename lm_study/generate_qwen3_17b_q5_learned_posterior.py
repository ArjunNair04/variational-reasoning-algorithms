"""Frozen three-seed learned-distribution replacement, not a proposal ablation."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from generate_qwen3_17b_q5_component_kernel import (
    CONTROL_CELL, CONTROL_RUN, CONTROL_JOB, CONTROL_COMMIT, CONTROL_CONFIG_SHA256,
)
from generate_qwen3_17b_q5_prefix_continuations import build_payload as control_payload
from generate_qwen3_17b_q5_prior_exponent import SEEDS, runtime_configs as prepare_runtime_configs

RUN_ID = "2d7c8a61"
CELL = "Q5-POSTERIOR-U1"


def build_payload():
    source = control_payload()
    cell = deepcopy(source["algos"]["AC-ALG1"][0])
    cell.update(cell_id=CELL, algorithm_profile="q5_learned_posterior")
    defaults = deepcopy(source["defaults"])
    defaults["out"] = f"~/po_results/2026-10-02/q5-learned-posterior/qwen3-q5-learned-posterior__{RUN_ID}"
    return {
        "run_id": RUN_ID, "tag_prefix": "q3_q5_posterior",
        "diagnostic": {
            "stage": "three_seed_learned_distribution_replacement",
            "design": {"array_tasks": 3, "cell_order": [CELL], "paired_seeds": list(SEEDS)},
            "control": {"run_id": CONTROL_RUN, "cell_id": CONTROL_CELL, "job_id": CONTROL_JOB,
                "tasks": [1, 2, 3], "execution_commit": CONTROL_COMMIT,
                "config_sha256": CONTROL_CONFIG_SHA256, "out": source["defaults"]["out"]},
            "round_zero": deepcopy(source["diagnostic"]["round_zero"]),
            "teacher": "Unchanged moving-reader Q5 joint weights on the persistent unique FIFO-16 buffer. Main model supplies 16 answer-derived proposals per question. No gold rationale.",
            "posterior_fit": {"updates_per_round": 1, "learning_rate": 1e-5,
                "optimizer": "persistent Adam", "initialization": "exact copy of initial main LoRA",
                "backbone": "shared frozen Qwen3-1.7B", "adapter": "separate rank-16 attention+MLP",
                "target": "Q5-weighted rationale+marker+gold answer+EOS under the answer-derived prompt",
                "normalization": "sum token log probability, per-question weights sum to one, average over four questions"},
            "distribution": "q_phi(h|q,a) is the pushforward of the learned completion distribution through a fixed first-marker rationale map. It replaces, not reweights, the delta mixture in the main M-step.",
            "sampling": {"draws_per_question": 16, "temperature": 1.0, "top_k": 0, "top_p": 1.0,
                "max_new_tokens": 256, "timing": "after posterior fit, before main update",
                "mapping": "Use the native first-marker prefix when alignable and nonempty. Otherwise take decoded text before the first marker, append newline+#### and retokenize. Empty outputs retain their mass as an empty rationale. Attach gold answer+EOS. No rejection, resampling or correctness filter; log all repairs."},
            "main_update": "One joint weighted-MLE step under question-only prompts, using weight 1/16 for every fresh posterior draw, duplicates included. No Q5 rescoring. Posterior samples never enter the persistent teacher buffer. Posterior parameters are frozen during the main update and evaluation uses only the main adapter.",
            "rng": "Independent SHA256-derived fit and sample streams per round; fork_rng restores CPU/CUDA states. Adapter initialization also isolated. Separate persistent optimizers.",
            "accounting": {"main_steps": 32, "posterior_steps": 32, "total_steps": 64,
                "teacher_proposals": 2048, "posterior_draws": 2048, "total_training_generations": 4096},
            "analysis_contract": [
                "Final extracted Acc@1, strict final, normalized Acc@1 AUC rounds 0-32, then strict AUC.",
                "Require three paired seeds, receipt hashes, exact commit/YAML/runtime, all checkpoints, identical validation support and training-question schedule. Validate both saved adapters.",
                "Require exactly 32 posterior and 32 main updates; validate uniform M-step weights, teacher joint logits, no posterior children in teacher buffers, separate RNG streams and exact generation/token accounting.",
                "Report three per-seed differences and descriptive paired 95% t intervals. Nominate only if mean final improves without mean strict-final loss. No automatic confirmation.",
                "Report teacher concentration, posterior diversity, boundary repairs, main/posterior backward tokens, generation, phase timing and total GPU hours. Extra compute, not a compute-matched comparison.",
                "This is distillation of a Q5 finite-buffer teacher into a learned completion law. It does not optimize the entropy of the induced rationale law, guarantee a better E-step, or establish exact posterior inference.",
                "Official test is not loaded; validation remains the fixed 400-question train-derived pool.",
            ],
        }, "defaults": defaults, "algos": {"AC-ALG1": [cell]},
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
            raise SystemExit("frozen learned-posterior protocol changed")
    else:
        args.write.write_text("# Generated; edit generate_qwen3_17b_q5_learned_posterior.py.\n" + yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
