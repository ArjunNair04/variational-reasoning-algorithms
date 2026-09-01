#!/usr/bin/env python3
"""Analyse the selected-method seven-seed posterity replay.

The design is frozen before release. It repeats selected methods on the
original paired seeds, adds matched Q5 and PIS proposal-prompt contrasts, and
changes only Q5's answer reader in the reader ablation. The script makes no
method-selection decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LM_STUDY = REPOSITORY_ROOT / "lm_study"
if str(LM_STUDY) not in sys.path:
    sys.path.insert(0, str(LM_STUDY))
if str(REPOSITORY_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "analysis"))

import analyze_qwen3_final_method_confirmation as shared  # noqa: E402
from generate_qwen3_17b_selected_method_posterity import (  # noqa: E402
    ANSWER_DERIVED_CELLS,
    CELL_ORDER,
    CHECKPOINTS,
    RUN_ID,
    SEEDS,
    build_payload,
)


BASE_CELL = "CTRL-base"
Q5_QUESTION = "Q5-Q-LR1e-5-U1-K16"
Q5_ANSWER_MOVING = "Q5-AD-M-LR1e-5-U1-K16"
Q5_ANSWER_FROZEN = "Q5-AD-F-LR1e-5-U1-K16"
Q5_ANSWER_ESS = "Q5-AD-M-ESS50-LR1e-5-U1-K16"
Q5_ANSWER_KL = "Q5-AD-M-KL03R-LR1e-5-U1-K16"
Q5_ANSWER_TEMPERATURE = "Q5-AD-M-T1p2-LR1e-5-U1-K16"
PIS_QUESTION = "PIS-Q-S8-B8-U4"
PIS_ANSWER = "PIS-AD-S8-B8-U4"
SOURCE_CELL = {
    BASE_CELL: "CTRL-base",
    Q5_ANSWER_MOVING: "Q5-LR1e-5-U1-K16",
    PIS_QUESTION: "PIS-S8-B8-U4",
    "REST-LR1e-5-E1-I4": "REST-LR1e-5-E1-I4",
    "STAR-LR3e-6-E2": "STAR-LR3e-6-E2",
    "TRICE-LR1e-4-CV": "TRICE-LR1e-4-CV",
    "RLOO-S16-B8-U4": "RLOO-S16-B8-U4",
}
METRICS = (
    "final_extracted",
    "final_strict",
    "extracted_auc",
    "strict_auc",
)
CONTRASTS = (
    (Q5_QUESTION, BASE_CELL, "question_q5_vs_base"),
    (Q5_ANSWER_MOVING, BASE_CELL, "answer_q5_vs_base"),
    (Q5_ANSWER_FROZEN, BASE_CELL, "frozen_answer_q5_vs_base"),
    (Q5_ANSWER_MOVING, Q5_QUESTION, "q5_prompt_ablation"),
    (Q5_ANSWER_FROZEN, Q5_ANSWER_MOVING, "q5_reader_ablation"),
    (Q5_ANSWER_ESS, Q5_ANSWER_MOVING, "q5_ess_ablation"),
    (Q5_ANSWER_KL, Q5_ANSWER_MOVING, "q5_kl_ablation"),
    (Q5_ANSWER_TEMPERATURE, Q5_ANSWER_MOVING, "q5_proposal_temperature_ablation"),
    (PIS_QUESTION, BASE_CELL, "question_pis_vs_base"),
    (PIS_ANSWER, BASE_CELL, "answer_pis_vs_base"),
    (PIS_ANSWER, PIS_QUESTION, "pis_prompt_estimator_ablation"),
    ("REST-LR1e-5-E1-I4", BASE_CELL, "rest_vs_base"),
    ("STAR-LR3e-6-E2", BASE_CELL, "star_vs_base"),
    ("TRICE-LR1e-4-CV", BASE_CELL, "trice_vs_base"),
    ("RLOO-S16-B8-U4", BASE_CELL, "rloo_vs_base"),
)


def _configure_shared_loader() -> None:
    shared.RUN_ID = RUN_ID
    shared.CELL_ORDER = CELL_ORDER
    shared.SEEDS = SEEDS
    shared.CHECKPOINTS = CHECKPOINTS
    shared.BASE_CELL = BASE_CELL
    shared.Q5_CELL = Q5_ANSWER_MOVING
    shared.ANSWER_DERIVED_CELLS = ANSWER_DERIVED_CELLS
    shared.build_payload = build_payload


def validate_design(config_path: Path) -> tuple[dict[str, Any], list[Any]]:
    _configure_shared_loader()
    config, cells = shared.load_design(config_path)
    if tuple(cell.cell_id for cell in cells) != CELL_ORDER:
        raise ValueError("posterity cell order changed")
    if tuple(config["defaults"]["seed_values"]) != SEEDS:
        raise ValueError("posterity seed family changed")
    configured = {cell.cell_id: cell for cell in cells}
    question = dict(configured[Q5_QUESTION].axes)
    moving = dict(configured[Q5_ANSWER_MOVING].axes)
    frozen = dict(configured[Q5_ANSWER_FROZEN].axes)
    ess = dict(configured[Q5_ANSWER_ESS].axes)
    kl = dict(configured[Q5_ANSWER_KL].axes)
    temperature = dict(configured[Q5_ANSWER_TEMPERATURE].axes)
    if moving.get("responsibility_answer_policy") != "current":
        raise ValueError("moving-reader Q5 is not moving")
    frozen["responsibility_answer_policy"] = "current"
    if moving != frozen:
        raise ValueError("Q5 reader contrast changes more than one factor")
    expected_controls = (
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
    for candidate, expected in expected_controls:
        changed = {
            key
            for key in set(moving) | set(candidate)
            if moving.get(key) != candidate.get(key)
        }
        if changed != set(expected):
            raise ValueError(f"Q5 stability contrast changed unexpected axes: {changed}")
        if any(candidate[key] != value for key, value in expected.items()):
            raise ValueError(f"Q5 stability contrast has wrong values: {expected}")
    question["algorithm_profile"] = moving["algorithm_profile"]
    question["proposal_prompt"] = moving["proposal_prompt"]
    if question != moving:
        raise ValueError("Q5 prompt contrast changes more than prompt and validation profile")
    pis_question = dict(configured[PIS_QUESTION].axes)
    pis_answer = dict(configured[PIS_ANSWER].axes)
    changed = {
        key
        for key in set(pis_question) | set(pis_answer)
        if pis_question.get(key) != pis_answer.get(key)
    }
    if changed != {
        "algorithm_profile",
        "proposal_prompt",
        "variational_estimator",
    }:
        raise ValueError(f"PIS prompt/estimator contrast changed unexpected axes: {changed}")
    return config, cells


def _exact_sign_flip_p(values: np.ndarray) -> float:
    return shared._exact_sign_flip_p(values)


def build_contrasts(
    seed_metrics: pd.DataFrame,
    bootstrap_means: np.ndarray,
) -> pd.DataFrame:
    cell_index = {cell: index for index, cell in enumerate(CELL_ORDER)}
    metric_index = {metric: index for index, metric in enumerate(shared.BOOTSTRAP_METRICS)}
    rows: list[dict[str, Any]] = []
    for treatment, control, family in CONTRASTS:
        for metric in METRICS:
            treatment_seed = (
                seed_metrics[seed_metrics["cell"] == treatment]
                .set_index("seed")
                .reindex(SEEDS)[metric]
            )
            control_seed = (
                seed_metrics[seed_metrics["cell"] == control]
                .set_index("seed")
                .reindex(SEEDS)[metric]
            )
            seed_delta = (treatment_seed - control_seed).to_numpy(float)
            draws = (
                bootstrap_means[:, cell_index[treatment], metric_index[metric]]
                - bootstrap_means[:, cell_index[control], metric_index[metric]]
            )
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "family": family,
                    "treatment": treatment,
                    "control": control,
                    "metric": metric,
                    "mean_difference": float(seed_delta.mean()),
                    "mean_difference_pp": 100.0 * float(seed_delta.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "ci95_low_pp": 100.0 * float(low),
                    "ci95_high_pp": 100.0 * float(high),
                    "positive_seeds": int((seed_delta > 0).sum()),
                    "exact_sign_flip_p": _exact_sign_flip_p(seed_delta),
                    "interpretation": "descriptive_prespecified_paired_contrast",
                }
            )
    return pd.DataFrame(rows)


def compare_source_summary(
    replay_summary: pd.DataFrame,
    source_path: Path | None,
) -> pd.DataFrame:
    if source_path is None:
        return pd.DataFrame(
            columns=[
                "cell",
                "source_cell",
                "metric",
                "replay_mean",
                "source_mean",
                "replay_minus_source",
            ]
        )
    source = pd.read_csv(source_path)
    required = {"cell", "final_extracted", "final_strict", "extracted_auc", "strict_auc"}
    if not required.issubset(source.columns):
        raise ValueError(f"source summary lacks columns: {sorted(required - set(source.columns))}")
    rows = []
    for cell, source_cell in SOURCE_CELL.items():
        replay_row = replay_summary[replay_summary["cell"] == cell]
        source_row = source[source["cell"] == source_cell]
        if len(replay_row) != 1 or len(source_row) != 1:
            raise ValueError(f"source comparison cannot resolve {cell}/{source_cell}")
        for metric in METRICS:
            replay_value = float(replay_row.iloc[0][metric])
            source_value = float(source_row.iloc[0][metric])
            rows.append(
                {
                    "cell": cell,
                    "source_cell": source_cell,
                    "metric": metric,
                    "replay_mean": replay_value,
                    "source_mean": source_value,
                    "replay_minus_source": replay_value - source_value,
                    "replay_minus_source_pp": 100.0 * (replay_value - source_value),
                    "interpretation": "same_seed_mean_replay_difference",
                }
            )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-design-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-job")
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2_026_082_901)
    args = parser.parse_args()

    config, cells = validate_design(args.config)
    if args.validate_design_only:
        print(json.dumps({"run_id": RUN_ID, "cells": len(cells), "seeds": len(SEEDS)}))
        return 0
    required = {
        "artifact_dir": args.artifact_dir,
        "marker": args.marker,
        "expected_commit": args.expected_commit,
        "expected_source_job": args.expected_source_job,
        "output_dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing result-analysis arguments: {', '.join(missing)}")

    marker = shared.verify_marker(
        args.marker,
        args.config,
        expected_commit=args.expected_commit,
        expected_source_job=args.expected_source_job,
    )
    questions, seeds, resources, _receipts = shared.load_results(
        args.artifact_dir,
        config,
        cells,
        expected_commit=args.expected_commit,
    )
    cube, question_ids = shared.question_metric_cube(questions)
    bootstrap = shared.hierarchical_method_bootstrap(
        cube,
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    summary = shared.method_summary(seeds, resources)
    contrasts = build_contrasts(seeds, bootstrap)
    reproduction = compare_source_summary(summary, args.source_summary)

    analysis = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "analysis_status": "complete",
        "decision": "descriptive_reproducibility_replay_no_method_selection",
        "official_test_used": False,
        "metric_order": [
            "final_extracted_answer_accuracy",
            "final_strict_terminal_accuracy",
            "normalized_extracted_trajectory_auc",
        ],
        "observational_unit": "paired training seed",
        "uncertainty": "nested paired bootstrap over seeds and matched validation questions",
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "question_count": len(question_ids),
        "cell_count": len(CELL_ORDER),
        "seed_count": len(SEEDS),
        "validator_marker": marker,
        "source_summary_compared": args.source_summary is not None,
        "limitations": [
            "The same train-derived validation partition and seeds are intentionally replayed.",
            "This is reproducibility evidence, not a fresh method-selection experiment.",
            "The official GSM8K test remains sealed.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    questions.to_csv(args.output_dir / "question_metrics.csv.gz", index=False)
    seeds.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    resources.to_csv(args.output_dir / "seed_resources.csv", index=False)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "paired_contrasts.csv", index=False)
    reproduction.to_csv(args.output_dir / "source_reproduction_differences.csv", index=False)
    _write_json(args.output_dir / "analysis.json", analysis)
    print(json.dumps(analysis, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
