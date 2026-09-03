#!/usr/bin/env python3
"""Analyse the prespecified seven-seed Q5 support follow-up."""

from __future__ import annotations

import argparse
import gzip
import hashlib
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

from generate_qwen3_17b_q5_support_followup import (  # noqa: E402
    CELL_ORDER,
    CHECKPOINTS,
    CONTROL_CELLS,
    RUN_ID,
    SEEDS,
    build_payload,
)
from result_contract import validate_completion_receipt, validate_receipt_identity  # noqa: E402
from run_yaml import _prepare_cells  # noqa: E402


METRICS = ("final_extracted", "final_strict", "extracted_auc", "strict_auc")
PAIRWISE = (
    ("Q5-S64-B32-U1-FULL", "Q5-S32-B32-U1-FULL", "s64_vs_s32_full"),
    ("Q5-S32-B32-U1-TOPRES15", "Q5-S32-B32-U1-FULL", "topres15_vs_full"),
    ("Q5-S32-B32-U1-TOPRES15", "Q5-S32-B32-U1-MS16", "topres15_vs_categorical16"),
)


def validate_design(config_path: Path) -> tuple[dict[str, Any], list[Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config != build_payload():
        raise ValueError("configuration differs from the generated frozen payload")
    cells = _prepare_cells(
        config, only=None, run_id=RUN_ID, defaults=config["defaults"]
    )
    configured = tuple(cell["cell_id"] for cell in config["algos"]["AC-ALG1"])
    if configured != CELL_ORDER or tuple(config["defaults"]["seed_values"]) != SEEDS:
        raise ValueError("Q5 support-follow-up coordinates changed")
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


def _inner_steps(round_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return persisted inner-step diagnostics from the current schema."""

    value = (round_row.get("inner_m_step") or {}).get("steps") or []
    if not isinstance(value, list):
        raise ValueError("inner_m_step.steps must be a list")
    return value


def _verify_marker(
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
        "task_count": 14,
        "trained_adapter_count": 14,
        "official_test_used": False,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"validator marker mismatch for {key}")
    if marker.get("configuration_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError("validator marker has the wrong configuration hash")
    return marker


def load_results(
    artifact_dir: Path,
    cells: list[Any],
    base_metrics: pd.DataFrame,
    *,
    expected_commit: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bases = base_metrics.set_index(["cell", "seed"])
    rows = []
    mechanisms = []
    for cell_id, cell in zip(CELL_ORDER, cells, strict=True):
        for seed in SEEDS:
            tag = f"{cell.tag}_seed{seed}"
            receipt_path = artifact_dir / f"complete_gsm8k__{tag}__AC-ALG1_s{seed}.json"
            receipt = validate_completion_receipt(
                receipt_path, result_root=artifact_dir, verify_hashes=True
            )
            validate_receipt_identity(receipt, {
                "run_id": RUN_ID,
                "task": "gsm8k",
                "model": "qwen3-1.7b-base",
                "method": "AC-ALG1",
                "seed": seed,
                "tag": tag,
            })
            evaluation = _read_json(_artifact(artifact_dir, receipt, "eval_"))
            if evaluation.get("eval_official_test_accessed") is not False:
                raise ValueError(f"{cell_id}/{seed}: official-test access is not false")
            final_extracted = _finite(evaluation["test_acc_legacy"], field="final extracted")
            final_strict = _finite(evaluation["test_acc_strict"], field="final strict")
            checkpoints = []
            with gzip.open(_artifact(artifact_dir, receipt, "checkpoint_eval_"), "rt", encoding="utf-8") as handle:
                checkpoints = [json.loads(line) for line in handle]
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
                raise ValueError(f"{cell_id}/{seed}: extracted endpoint mismatch")
            if not math.isclose(strict[-1], final_strict, abs_tol=1e-12):
                raise ValueError(f"{cell_id}/{seed}: strict endpoint mismatch")
            base = bases.loc[("CTRL-base", seed)]
            cell_result = _read_json(_artifact(artifact_dir, receipt, "cell_result_"))
            result = dict(cell_result.get("result") or {})
            params = json.loads(str(result.get("params") or "{}"))
            commit = str((params.get("env") or {}).get("commit") or "")
            if len(commit) < 7 or not expected_commit.startswith(commit):
                raise ValueError(f"{cell_id}/{seed}: execution commit mismatch")
            if int(result.get("optimizer_steps", -1)) != 32:
                raise ValueError(f"{cell_id}/{seed}: optimizer-step schedule changed")
            diagnostic_path = _artifact(artifact_dir, receipt, "training_diagnostics_")
            with gzip.open(diagnostic_path, "rt", encoding="utf-8") as handle:
                diagnostics = [json.loads(line) for line in handle]
            sample_records = []
            backward_tokens = 0
            for round_row in diagnostics:
                for inner in _inner_steps(round_row):
                    backward_tokens += int((inner.get("support") or {}).get("backward_tokens") or 0)
                    sampling = inner.get("mstep_sampling") or {}
                    questions = list(sampling.get("questions") or [])
                    if sampling.get("mode") == "full_posterior":
                        questions = [
                            {
                                "pid": int(question["pid"]),
                                "posterior_support_size": int(question["rows"]),
                                "draw_count": 0,
                                "unique_draw_count": int(question["rows"]),
                                "posterior_mass_covered": 1.0,
                                "empirical_ess": None,
                                "selected_indices": None,
                                "multiplicities": None,
                            }
                            for question in (round_row.get("buffer") or {}).get("per_question") or []
                        ]
                    for question in questions:
                        sample_records.append({
                            "completed_rounds": int(round_row["completed_rounds"]),
                            "inner_step": int(inner["inner_step"]),
                            **question,
                        })
            expected_mode = (
                "top_plus_residual"
                if cell_id.endswith("TOPRES15")
                else "full_posterior"
            )
            if diagnostics and any(
                (inner.get("mstep_sampling") or {}).get("mode") != expected_mode
                for round_row in diagnostics
                for inner in _inner_steps(round_row)
            ):
                raise ValueError(f"{cell_id}/{seed}: M-step sampling mode changed")
            if len(diagnostics) != 32 or any(len(_inner_steps(row)) != 1 for row in diagnostics):
                raise ValueError(f"{cell_id}/{seed}: inner-step diagnostic coverage changed")
            if cell_id.endswith("TOPRES15"):
                topres = [
                    (inner.get("mstep_sampling") or {}).get("questions") or []
                    for round_row in diagnostics
                    for inner in _inner_steps(round_row)
                ]
                flat_topres = [question for questions in topres for question in questions]
                if any(
                    int(question.get("exact_trace_count", -1)) != 1
                    or len(question.get("exact_indices") or []) != 1
                    for question in flat_topres
                ):
                    raise ValueError(f"{cell_id}/{seed}: exact-top coverage changed")
            if not sample_records or backward_tokens <= 0:
                raise ValueError(f"{cell_id}/{seed}: M-step mechanism diagnostics are incomplete")
            rows.append({
                "cell": cell_id,
                "seed": seed,
                "final_extracted": final_extracted,
                "final_strict": final_strict,
                "extracted_auc": _normalized_auc(float(base["final_extracted"]), extracted),
                "strict_auc": _normalized_auc(float(base["final_strict"]), strict),
                "train_llm_gen": _finite(result["train_llm_gen"], field="train draws"),
                "optimizer_steps": int(result["optimizer_steps"]),
                "accelerator_hours": _finite(result["accelerator_hours"], field="H100 hours"),
                "backward_tokens": backward_tokens,
                "mean_unique_mstep_traces": (
                    float(np.mean([row["unique_draw_count"] for row in sample_records]))
                    if sample_records else None
                ),
                "mean_posterior_mass_covered": (
                    float(np.mean([row["posterior_mass_covered"] for row in sample_records]))
                    if sample_records else None
                ),
            })
            mechanisms.extend({"cell": cell_id, "seed": seed, **row} for row in sample_records)
    frame = pd.DataFrame(rows)
    expected = {(cell, seed) for cell in CELL_ORDER for seed in SEEDS}
    if set(zip(frame["cell"], frame["seed"], strict=True)) != expected:
        raise ValueError("seed coordinate coverage is incomplete")
    return frame, pd.DataFrame(mechanisms)


def paired_contrasts(new: pd.DataFrame, controls: pd.DataFrame, *, draws: int, seed: int) -> pd.DataFrame:
    indexed = pd.concat([controls, new], ignore_index=True).set_index(["cell", "seed"])
    generator = np.random.default_rng(seed)
    resamples = generator.integers(0, len(SEEDS), size=(draws, len(SEEDS)))
    rows = []
    for treatment, control, family in PAIRWISE:
        for metric in METRICS:
            differences = np.asarray([
                float(indexed.loc[(treatment, value), metric])
                - float(indexed.loc[(control, value), metric])
                for value in SEEDS
            ])
            bootstrap = differences[resamples].mean(axis=1)
            low, high = np.quantile(bootstrap, (0.025, 0.975))
            rows.append({
                "contrast": family,
                "treatment": treatment,
                "control": control,
                "metric": metric,
                "mean_difference_pp": 100.0 * float(differences.mean()),
                "ci95_low_pp": 100.0 * float(low),
                "ci95_high_pp": 100.0 * float(high),
                "positive_seeds": int((differences > 0).sum()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-design-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-job")
    parser.add_argument("--base-seed-metrics", type=Path)
    parser.add_argument("--q5-buffer-seed-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2_026_090_202)
    args = parser.parse_args()

    _config, cells = validate_design(args.config)
    if args.validate_design_only:
        print(json.dumps({"run_id": RUN_ID, "cells": len(cells), "seeds": len(SEEDS)}))
        return 0
    required = (
        "artifact_dir", "marker", "expected_commit", "expected_source_job",
        "base_seed_metrics", "q5_buffer_seed_metrics", "output_dir",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"missing result-analysis arguments: {', '.join(missing)}")
    marker = _verify_marker(
        args.marker,
        args.config,
        expected_commit=args.expected_commit,
        expected_source_job=args.expected_source_job,
    )
    bases = pd.read_csv(args.base_seed_metrics)
    controls = pd.read_csv(args.q5_buffer_seed_metrics)
    if "CTRL-base" not in set(bases["cell"]) or not set(CONTROL_CELLS).issubset(
        set(controls["cell"])
    ):
        raise ValueError("analysis inputs lack the registered base or Q5 controls")
    results, mechanisms = load_results(
        args.artifact_dir, cells, bases, expected_commit=args.expected_commit
    )
    contrasts = paired_contrasts(
        results,
        controls.loc[controls["cell"].isin(CONTROL_CELLS)],
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    summary = (
        results.groupby("cell", sort=False)
        .agg(
            final_extracted=("final_extracted", "mean"),
            final_strict=("final_strict", "mean"),
            extracted_auc=("extracted_auc", "mean"),
            strict_auc=("strict_auc", "mean"),
            train_llm_gen=("train_llm_gen", "mean"),
            optimizer_steps=("optimizer_steps", "mean"),
            accelerator_hours=("accelerator_hours", "mean"),
            backward_tokens=("backward_tokens", "mean"),
            mean_unique_mstep_traces=("mean_unique_mstep_traces", "mean"),
            mean_posterior_mass_covered=("mean_posterior_mass_covered", "mean"),
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
        "historical_controls": list(CONTROL_CELLS),
        "validator_marker": marker,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "paired_contrasts.csv", index=False)
    mechanisms.to_csv(args.output_dir / "mstep_sampling_questions.csv.gz", index=False)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
