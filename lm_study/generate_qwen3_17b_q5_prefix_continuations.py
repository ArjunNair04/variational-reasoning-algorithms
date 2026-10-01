"""Freeze a candidate-count-matched Q5 prefix-continuation screen."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from generate_qwen3_17b_q5_prior_exponent import SEEDS, runtime_configs as prepare_runtime_configs
from generate_qwen3_17b_selected_method_posterity import CHECKPOINTS, build_payload as controls

RUN_ID = "f812ca40"
CONTROL_CELL = "Q5-AD-M-LR1e-5-U1-K16"
SETTINGS = {"Q5-INDEPENDENT16": "independent", "Q5-PREFIX8x2": "prefix_half"}
CELL_ORDER = tuple(SETTINGS)
BASELINES = {
    1201: {"final_extracted": .745, "final_strict": .6125,
           "evaluation_sha256": "2cc7ee4905613764c4cf6123ee66cd576175140891f2a74e3ee8ba2ebc5313a0"},
    1213: {"final_extracted": .7525, "final_strict": .72,
           "evaluation_sha256": "0de689cecbd30535557685d89e81139d3e94f788bfc4bf4b519ff3a0b02a11d3"},
    1217: {"final_extracted": .7675, "final_strict": .5125,
           "evaluation_sha256": "c121a8b84f4169de2f0b9d66410fa7bad28806691e91cc0f7fa532e398b884f9"},
}
SUPPORT_SHA256 = "2e79c7dd90a05f47991da744720f7298635feea7cb592fe82d291a6542c2bf97"


def build_payload():
    source = controls()
    original = next(c for c in source["algos"]["AC-ALG1"] if c["cell_id"] == CONTROL_CELL)
    cells = []
    for name, mode in SETTINGS.items():
        cell = deepcopy(original)
        cell.update(cell_id=name, algorithm_profile="q5_prefix_continuations",
                    proposal_continuation_mode=mode)
        cells.append(cell)
    defaults = deepcopy(source["defaults"])
    defaults.update(seed_values=list(SEEDS), seeds=len(SEEDS),
        out=f"~/po_results/2026-10-01/q5-prefix-continuations/qwen3-q5-prefix-continuations__{RUN_ID}")
    return {
        "run_id": RUN_ID, "tag_prefix": "q3_q5_prefix",
        "diagnostic": {
            "stage": "three_seed_prefix_continuation_screen",
            "evidence_class": "development_screen_concurrent_paired_control",
            "design": {"array_tasks": 6, "cell_order": list(CELL_ORDER), "paired_seeds": list(SEEDS)},
            "round_zero": {"run_id": "68078ecc", "cell_id": "CTRL-base",
                "question_ids_sha256": SUPPORT_SHA256, "seeds": deepcopy(BASELINES),
                "provenance": "Recomputed from 400 per-question records per seed and checked against archived completion-receipt SHA256 on 2026-10-01. Historical baseline only; both trained cells are concurrent."},
            "intervention": "Keep 8 independent complete roots per question; draw one new suffix from the first floor(n/2) native reasoning tokens of each root. Keep all roots and children. Control draws 16 independent completions.",
            "boundary": "Reasoning ends before the first #### marker. A token overlapping the marker is excluded. No answer-correctness or reader-score root selection. A missing/empty usable prefix falls back to an independent draw and is counted.",
            "budget": "32 rounds x 4 questions x 16 complete candidates = 2048. Root generation is charged in full. Each child has the original whole-response cap minus copied-prefix length. Report suffix generation separately from copied and backward tokens; not compute matched.",
            "fixed_contract": {"official_test_used": False, "answer_target": "answer_plus_eos",
                "proposal": "answer_derive", "reader": "current", "mstep": "unchanged_joint_weighted_mle"},
            "analysis_contract": [
                "Report Final extracted Acc@1 first, strict final second, normalized extracted AUC rounds 0-32 third, then strict AUC.",
                "Show each same-seed treatment-minus-control difference and descriptive paired 95% t intervals. Three reused development seeds do not establish confirmation.",
                "Use the receipt-verified, same-seed historical frozen-base endpoints only for round-zero AUC; evaluation hashes are frozen in round_zero. Both training cells run concurrently.",
                "Report all outcomes; a positive mean final difference without a mean strict-final loss nominates the treatment for discussion only. AUC and cost remain visible; no automatic follow-up.",
                "Report parent/child trace identities, retained-prefix hashes, fallback and duplicate rates, generated versus reused tokens, backward tokens, duration, ESS and weight concentration.",
                "This changes candidate discovery, not the finite-support variational family. Children share prefixes and are correlated; ESS is a weight-concentration statistic, not a count of independent observations.",
                "The prefix boundary uses the fully generated root. This is a positional continuation experiment, not an identified understanding/solving split.",
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
    payload = build_payload()
    if args.check:
        if yaml.safe_load(args.check.read_text()) != payload:
            raise SystemExit("frozen prefix-continuation protocol changed")
    else:
        args.write.write_text("# Generated; edit generate_qwen3_17b_q5_prefix_continuations.py.\n" + yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
