"""Freeze a positional, not semantic, Q5 rationale-prior screen."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from generate_qwen3_17b_q5_prior_exponent import (
    CHECKPOINTS, SEEDS, CONTROL_COMMIT, CONTROL_RUN_ID,
    runtime_configs as prepare_runtime_configs,
)
from generate_qwen3_17b_selected_method_posterity import build_payload as controls
from ac_alg1_prior_segments import SEGMENT_SCOPE

RUN_ID = "c7e32a91"
SCORING_EVIDENCE = {
    "job": "7412300",
    "execution_commit": "802c174e0e1a20842b757ce8b25aea70d8730b6d",
    "manifest_sha256": "292f80620f6e90e34b5051778e2a1b28f70c7efa30d06572512356f88568caa9",
    "summary_sha256": "0de6cdca7137bc346ac8849f1c678a2f1b8a1dd05332ad750fa7ecdbf7971abc",
    "selected_scope": SEGMENT_SCOPE,
    "decision": "User approved the marker-fixed nine-task training screen after reviewing the completed scoring audit on 2026-09-14. The old partial audit remains unchanged.",
}
CONTROL_CELL = "Q5-AD-M-LR1e-5-U1-K16"
SETTINGS = {
    "Q5-EARLY-HALF": (0.5, 1.0),
    "Q5-LATE-HALF": (1.0, 0.5),
    "Q5-UNIFORM075": (0.75, 0.75),
}
CELL_ORDER = tuple(SETTINGS)


def build_payload():
    source = controls()
    original = next(c for c in source["algos"]["AC-ALG1"] if c["cell_id"] == CONTROL_CELL)
    cells = []
    for name, (head, tail) in SETTINGS.items():
        cell = deepcopy(original)
        cell.update(cell_id=name, algorithm_profile="q5_prior_segments",
                    responsibility_prior_head_exponent=head,
                    responsibility_prior_tail_exponent=tail)
        cells.append(cell)
    defaults = deepcopy(source["defaults"])
    defaults.update(seed_values=list(SEEDS), seeds=len(SEEDS),
        out=f"~/po_results/2026-09-14/q5-prior-segments/qwen3-q5-prior-segments__{RUN_ID}")
    return {
        "run_id": RUN_ID, "tag_prefix": "q3_q5_prior_segments",
        "diagnostic": {
            "stage": "three_seed_positional_prior_screen",
            "evidence_class": "development_screen_historical_paired_control",
            "control_run_id": CONTROL_RUN_ID, "control_commit": CONTROL_COMMIT,
            "control_cell": CONTROL_CELL,
            "design": {"array_tasks": 9, "cell_order": list(CELL_ORDER), "paired_seeds": list(SEEDS)},
            "split": "First floor(n/2) reasoning tokens; remaining ceil(n/2) are late. Keep the retained #### marker, answer digits and EOS contributions unchanged; exclude prompt and padding. Use the stored native token boundary, with no segment retokenization.",
            "segment_scope": SEGMENT_SCOPE,
            "pretraining_evidence": dict(SCORING_EVIDENCE),
            "supersedes_unsubmitted_run": "a2f46c91",
            "weighting": "softmax(answer_logprob + marker_logprob + head_exponent*head_logprob + tail_exponent*tail_logprob)",
            "fixed_contract": {"official_test_used": False, "answer_target": "answer_plus_eos",
                "proposal": "answer_derive", "reader": "current", "mstep": "unchanged_joint_weighted_mle"},
            "analysis_contract": [
                "Final extracted Acc@1 first, strict final second, normalized extracted AUC rounds 0-32 third; strict AUC secondary.",
                "Report all per-seed differences and descriptive paired intervals, not a confirmatory significance claim.",
                "Pair each new cell with historical moving-reader Q5; compare early versus late and each positional cell versus concurrent uniform075.",
                "Three reused development seeds; no checkpoint selection or automatic follow-up. A positive mean final change with no strict-final loss nominates at most one setting for discussion, not deployment.",
                "Uniform075 matches nominal exponent mass at even lengths; odd lengths and realized logprob mass are not exactly matched. Report token counts and factor sums.",
                "Log head/tail/fixed-marker factors, count identities, top-trace changes versus joint and marker-fixed uniform scores, ESS, weight-length rank correlation, backward tokens and compute.",
                "This tests token position, not a demonstrated understanding-versus-solving decomposition. Semantic prompting, M-step masking and KL are excluded.",
            ],
        },
        "defaults": defaults, "algos": {"AC-ALG1": cells},
    }


def runtime_configs():
    return prepare_runtime_configs(build_payload())


def study_for_payload(payload):
    """Resolve only the two immutable, generated positional protocols."""
    import sys
    import generate_qwen3_17b_q5_prior_segments_frozen as frozen
    for study in (sys.modules[__name__], frozen):
        if payload == study.build_payload():
            return study
    raise ValueError("frozen design changed")


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
            raise SystemExit("frozen prior-segment configuration changed")
    else:
        args.write.write_text("# Generated; edit generate_qwen3_17b_q5_prior_segments.py.\n" + yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
