#!/usr/bin/env python3
"""Analyse the prespecified seven-seed posterior-update follow-ups."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
LM_STUDY = ROOT / "lm_study"
if str(LM_STUDY) not in sys.path:
    sys.path.insert(0, str(LM_STUDY))

import analyze_qwen3_selected_method_posterity as control_analysis  # noqa: E402
from generate_qwen3_17b_posterity_followups import (  # noqa: E402
    CELL_ORDER,
    CHECKPOINTS,
    RUN_ID,
    SEEDS,
    build_payload,
)
from result_contract import (  # noqa: E402
    validate_completion_receipt,
    validate_receipt_identity,
)
from run_yaml import _prepare_cells  # noqa: E402


METRICS = ("final_extracted", "final_strict", "extracted_auc", "strict_auc")
PAIRWISE = (
    ("Q5-MORE-S32-B16-U1", "Q5-AD-M-LR1e-5-U1-K16", "q5_more_vs_q5"),
    ("Q5-TOKENMEAN-S16-B16-U1", "Q5-AD-M-LR1e-5-U1-K16", "q5_token_mean_vs_q5"),
    ("PIS-Q-S8-B8-U1", "PIS-Q-S8-B8-U4", "pis_u1_vs_u4"),
    ("PIS-Q-S8-B8-U4-KL03R", "PIS-Q-S8-B8-U4", "pis_kl_vs_u4"),
    ("EXACT-Q-S8-B8-U1", "PIS-Q-S8-B8-U1", "exact_vs_pis_u1"),
    ("EXACT-Q-S8-B8-U4", "PIS-Q-S8-B8-U4", "exact_vs_pis_u4"),
)


def validate_design(config_path: Path) -> tuple[dict[str, Any], list[Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config != build_payload():
        raise ValueError("configuration differs from the generated frozen payload")
    cells = _prepare_cells(
        config,
        only=None,
        run_id=RUN_ID,
        defaults=config["defaults"],
    )
    configured_ids = tuple(
        str(cell["cell_id"])
        for cell in config["algos"]["AC-ALG1"]
    )
    if configured_ids != CELL_ORDER:
        raise ValueError("follow-up cell order changed")
    if tuple(config["defaults"]["seed_values"]) != SEEDS:
        raise ValueError("paired seed family changed")
    return config, cells


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _artifact(root: Path, receipt: Mapping[str, Any], prefix: str) -> Path:
    matches = [
        root / str(record["path"])
        for record in receipt.get("artifacts") or []
        if Path(str(record.get("path") or "")).name.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {prefix!r} artifact, found {len(matches)}")
    return matches[0]


def _finite(value: Any, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {field}: {value!r}")
    return number


def _normalized_auc(base: float, values: list[float]) -> float:
    rounds = np.asarray((0, *CHECKPOINTS), dtype=float)
    trajectory = np.asarray((base, *values), dtype=float)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(trajectory, rounds) / float(CHECKPOINTS[-1]))


def _verify_followup_marker(
    path: Path,
    config_path: Path,
    *,
    expected_commit: str,
    expected_source_job: str,
) -> dict[str, Any]:
    marker = _read_json(path)
    expected = {
        "schema_version": 1,
        "status": "ok",
        "run_id": RUN_ID,
        "execution_commit": expected_commit,
        "source_job_id": expected_source_job,
        "task_count": 42,
        "trained_adapter_count": 42,
        "official_test_used": False,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"follow-up validator marker mismatch for {key}")
    import hashlib

    expected_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if marker.get("configuration_sha256") != expected_sha:
        raise ValueError("follow-up validator marker has the wrong configuration hash")
    return marker


def load_followup_results(
    artifact_dir: Path,
    cells: list[Any],
    control_seed_metrics: pd.DataFrame,
    *,
    expected_commit: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = control_seed_metrics.set_index(["cell", "seed"])
    rows: list[dict[str, Any]] = []
    mechanisms: list[dict[str, Any]] = []
    for cell_id, cell in zip(CELL_ORDER, cells, strict=True):
        for seed in SEEDS:
            tag = f"{cell.tag}_seed{seed}"
            receipt_path = artifact_dir / f"complete_gsm8k__{tag}__AC-ALG1_s{seed}.json"
            receipt = validate_completion_receipt(
                receipt_path, result_root=artifact_dir, verify_hashes=True
            )
            validate_receipt_identity(
                receipt,
                {
                    "run_id": RUN_ID,
                    "task": "gsm8k",
                    "model": "qwen3-1.7b-base",
                    "method": "AC-ALG1",
                    "seed": seed,
                    "tag": tag,
                },
            )
            evaluation = _read_json(_artifact(artifact_dir, receipt, "eval_"))
            if evaluation.get("eval_official_test_accessed") is not False:
                raise ValueError(f"{cell_id}/{seed}: official-test access is not false")
            final_extracted = _finite(evaluation["test_acc_legacy"], field="final extracted")
            final_strict = _finite(evaluation["test_acc_strict"], field="final strict")

            checkpoints = []
            checkpoint_path = _artifact(artifact_dir, receipt, "checkpoint_eval_")
            with gzip.open(checkpoint_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    checkpoints.append(row)
            if tuple(int(row["completed_rounds"]) for row in checkpoints) != CHECKPOINTS:
                raise ValueError(f"{cell_id}/{seed}: checkpoint schedule changed")
            extracted = [
                _finite(row["metrics"]["test_acc_legacy"], field="checkpoint extracted")
                for row in checkpoints
            ]
            strict = [
                _finite(row["metrics"]["test_acc_strict"], field="checkpoint strict")
                for row in checkpoints
            ]
            if not math.isclose(extracted[-1], final_extracted, abs_tol=1e-12):
                raise ValueError(f"{cell_id}/{seed}: final extracted/checkpoint mismatch")
            if not math.isclose(strict[-1], final_strict, abs_tol=1e-12):
                raise ValueError(f"{cell_id}/{seed}: final strict/checkpoint mismatch")
            base_key = ("CTRL-base", seed)
            if base_key not in controls.index:
                raise ValueError(f"control metrics lack frozen base seed {seed}")
            base = controls.loc[base_key]

            cell_result = _read_json(_artifact(artifact_dir, receipt, "cell_result_"))
            result = dict(cell_result.get("result") or {})
            params = json.loads(str(result.get("params") or "{}"))
            commit = str((params.get("env") or {}).get("commit") or "")
            if len(commit) < 7 or not expected_commit.startswith(commit):
                raise ValueError(f"{cell_id}/{seed}: execution commit mismatch")
            if int(result.get("optimizer_steps", -1)) != 32 * int(cell.axes["iters"]):
                raise ValueError(f"{cell_id}/{seed}: optimizer-step schedule changed")
            diagnostics_path = _artifact(artifact_dir, receipt, "training_diagnostics_")
            with gzip.open(diagnostics_path, "rt", encoding="utf-8") as handle:
                diagnostics = [json.loads(line) for line in handle]
            generated_tokens, backward_tokens, scored_tokens = (
                control_analysis.shared._resource_totals(
                    "AC-ALG1",
                    result,
                    diagnostics,
                )
            )
            rows.append(
                {
                    "cell": cell_id,
                    "seed": seed,
                    "final_extracted": final_extracted,
                    "final_strict": final_strict,
                    "extracted_auc": _normalized_auc(float(base["final_extracted"]), extracted),
                    "strict_auc": _normalized_auc(float(base["final_strict"]), strict),
                    "train_llm_gen": _finite(result["train_llm_gen"], field="train draws"),
                    "optimizer_steps": int(result["optimizer_steps"]),
                    "accelerator_hours": _finite(result["accelerator_hours"], field="H100 hours"),
                    "generated_tokens": generated_tokens,
                    "backward_tokens": backward_tokens,
                    "teacher_forced_scoring_tokens": scored_tokens,
                }
            )
            trajectory = _read_json(_artifact(artifact_dir, receipt, "traj_"))
            for round_row in trajectory:
                mechanisms.append(
                    {
                        "cell": cell_id,
                        "seed": seed,
                        "round": int(round_row["round"]) + 1,
                        "policy_kl": round_row.get("policy_kl"),
                        "policy_anchor_beta": round_row.get("policy_anchor_beta"),
                        "policy_anchor_beta_unclipped": round_row.get("policy_anchor_beta_unclipped"),
                        "policy_anchor_beta_clip_fraction": round_row.get("policy_anchor_beta_clip_fraction"),
                        "policy_anchor_raw_grad_norm": round_row.get("policy_anchor_raw_grad_norm"),
                        "policy_anchor_applied_grad_norm": round_row.get("policy_anchor_applied_grad_norm"),
                        "responsibility_ess": round_row.get("responsibility_ess"),
                        "responsibility_max": round_row.get("responsibility_max"),
                    }
                )
    frame = pd.DataFrame(rows)
    expected = {(cell, seed) for cell in CELL_ORDER for seed in SEEDS}
    if set(zip(frame["cell"], frame["seed"], strict=True)) != expected:
        raise ValueError("follow-up seed coordinate coverage is incomplete")
    return frame, pd.DataFrame(mechanisms)


def paired_contrasts(
    followup: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    draws: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    combined = pd.concat([controls, followup], ignore_index=True)
    indexed = combined.set_index(["cell", "seed"])
    rng = np.random.default_rng(bootstrap_seed)
    seed_draws = rng.integers(0, len(SEEDS), size=(draws, len(SEEDS)))
    rows = []
    for treatment, control, family in PAIRWISE:
        for metric in METRICS:
            deltas = np.asarray(
                [indexed.loc[(treatment, seed), metric] - indexed.loc[(control, seed), metric] for seed in SEEDS],
                dtype=float,
            )
            sampled = deltas[seed_draws].mean(axis=1)
            low, high = np.quantile(sampled, [0.025, 0.975])
            rows.append(
                {
                    "family": family,
                    "treatment": treatment,
                    "control": control,
                    "metric": metric,
                    "mean_difference": float(deltas.mean()),
                    "mean_difference_pp": 100.0 * float(deltas.mean()),
                    "ci95_low_pp": 100.0 * float(low),
                    "ci95_high_pp": 100.0 * float(high),
                    "positive_seeds": int((deltas > 0).sum()),
                }
            )
    for metric in METRICS:
        interaction = np.asarray(
            [
                (indexed.loc[("EXACT-Q-S8-B8-U4", seed), metric] - indexed.loc[("EXACT-Q-S8-B8-U1", seed), metric])
                - (indexed.loc[("PIS-Q-S8-B8-U4", seed), metric] - indexed.loc[("PIS-Q-S8-B8-U1", seed), metric])
                for seed in SEEDS
            ],
            dtype=float,
        )
        sampled = interaction[seed_draws].mean(axis=1)
        low, high = np.quantile(sampled, [0.025, 0.975])
        rows.append(
            {
                "family": "exact_signed_reuse_interaction",
                "treatment": "EXACT_U4_minus_U1",
                "control": "PIS_U4_minus_U1",
                "metric": metric,
                "mean_difference": float(interaction.mean()),
                "mean_difference_pp": 100.0 * float(interaction.mean()),
                "ci95_low_pp": 100.0 * float(low),
                "ci95_high_pp": 100.0 * float(high),
                "positive_seeds": int((interaction > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-design-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-job")
    parser.add_argument("--control-config", type=Path)
    parser.add_argument("--control-marker", type=Path)
    parser.add_argument("--control-expected-commit")
    parser.add_argument("--control-source-job")
    parser.add_argument("--control-seed-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2_026_090_101)
    args = parser.parse_args()

    config, cells = validate_design(args.config)
    if args.validate_design_only:
        print(json.dumps({"run_id": RUN_ID, "cells": len(cells), "seeds": len(SEEDS)}))
        return 0
    required = (
        "artifact_dir",
        "marker",
        "expected_commit",
        "expected_source_job",
        "control_config",
        "control_marker",
        "control_expected_commit",
        "control_source_job",
        "control_seed_metrics",
        "output_dir",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"missing result-analysis arguments: {', '.join(missing)}")

    marker = _verify_followup_marker(
        args.marker,
        args.config,
        expected_commit=args.expected_commit,
        expected_source_job=args.expected_source_job,
    )
    control_analysis.validate_design(args.control_config)
    control_marker = control_analysis.shared.verify_marker(
        args.control_marker,
        args.control_config,
        expected_commit=args.control_expected_commit,
        expected_source_job=args.control_source_job,
    )
    controls = pd.read_csv(args.control_seed_metrics)
    needed_controls = {"CTRL-base", "Q5-AD-M-LR1e-5-U1-K16", "PIS-Q-S8-B8-U4"}
    if not needed_controls.issubset(set(controls["cell"])):
        raise ValueError("control analysis lacks required Q5/PIS/base rows")
    followup, mechanisms = load_followup_results(
        args.artifact_dir,
        cells,
        controls,
        expected_commit=args.expected_commit,
    )
    contrasts = paired_contrasts(
        followup,
        controls,
        draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary = (
        followup.groupby("cell", sort=False)
        .agg(
            final_extracted=("final_extracted", "mean"),
            final_strict=("final_strict", "mean"),
            extracted_auc=("extracted_auc", "mean"),
            strict_auc=("strict_auc", "mean"),
            train_llm_gen=("train_llm_gen", "mean"),
            optimizer_steps=("optimizer_steps", "mean"),
            accelerator_hours=("accelerator_hours", "mean"),
            generated_tokens=("generated_tokens", "mean"),
            backward_tokens=("backward_tokens", "mean"),
            teacher_forced_scoring_tokens=("teacher_forced_scoring_tokens", "mean"),
        )
        .reindex(CELL_ORDER)
        .reset_index()
    )
    analysis = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "analysis_status": "complete",
        "official_test_used": False,
        "metric_order": ["Final Acc@1", "Strict final accuracy", "Normalized trajectory AUC"],
        "independent_unit": "paired training seed",
        "uncertainty": "paired nonparametric bootstrap over seven training seeds",
        "validator_marker": marker,
        "control_validator_marker": control_marker,
        "limitations": [
            "The same train-derived validation partition and seeds are intentionally reused.",
            "Controls come from the dependency-gated selected-method posterity run.",
            "The official GSM8K test remains sealed.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    followup.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "paired_contrasts.csv", index=False)
    mechanisms.to_csv(args.output_dir / "mechanism_rounds.csv.gz", index=False)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
