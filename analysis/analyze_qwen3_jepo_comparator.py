#!/usr/bin/env python3
"""Analyse the prespecified seven-seed JEPO comparator."""

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

from generate_qwen3_17b_jepo_comparator import (  # noqa: E402
    CELL_ID,
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


BASE_CELL = "CTRL-base"
METRICS = ("final_extracted", "final_strict", "extracted_auc", "strict_auc")
T_CRITICAL_95_DF6 = 2.446911851


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _artifact(root: Path, receipt: Mapping[str, Any], prefix: str) -> Path:
    matches = [
        root / str(record["path"])
        for record in receipt.get("artifacts") or []
        if Path(str(record.get("path") or "")).name.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {prefix!r} artifact, found {len(matches)}")
    return matches[0]


def _finite(value: Any, *, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected a finite number")
    return number


def _require_non_access(payload: Mapping[str, Any], *, context: str) -> None:
    expected = {
        "eval_official_test_accessed": False,
        "eval_source_split": "train",
        "eval_dataset_splits_loaded": ["train"],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{context}: {field}={payload.get(field)!r}")


def _normalized_auc(base: float, values: list[float]) -> float:
    rounds = np.asarray((0, *CHECKPOINTS), dtype=float)
    trajectory = np.asarray((base, *values), dtype=float)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(trajectory, rounds) / float(CHECKPOINTS[-1]))


def validate_design(config_path: Path) -> tuple[dict[str, Any], Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config != build_payload():
        raise ValueError("configuration differs from the generated frozen payload")
    cells = _prepare_cells(
        config, only=None, run_id=RUN_ID, defaults=config["defaults"]
    )
    if len(cells) != 1 or cells[0].method != "JEPO":
        raise ValueError("JEPO comparator must expand to exactly one JEPO cell")
    if not cells[0].tag.endswith(f"_{CELL_ID}"):
        raise ValueError("JEPO cell identity changed")
    return config, cells[0]


def verify_marker(
    marker_path: Path,
    config_path: Path,
    *,
    expected_commit: str,
    expected_source_job: str,
) -> dict[str, Any]:
    marker = _read_json(marker_path)
    expected = {
        "schema_version": 1,
        "status": "ok",
        "run_id": RUN_ID,
        "execution_commit": expected_commit,
        "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_job_id": expected_source_job,
        "task_count": len(SEEDS),
        "trained_adapter_count": len(SEEDS),
        "official_test_used": False,
    }
    mismatches = {
        key: (marker.get(key), value)
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise ValueError(f"validator marker mismatch: {mismatches}")
    return marker


def load_controls(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"cell", "seed", "final_extracted", "final_strict"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"control metrics lack {sorted(required - set(frame.columns))}"
        )
    frame = frame[frame["cell"] == BASE_CELL].copy()
    if tuple(sorted(int(value) for value in frame["seed"])) != tuple(sorted(SEEDS)):
        raise ValueError("same-seed frozen controls are incomplete")
    if frame["seed"].nunique() != len(SEEDS) or len(frame) != len(SEEDS):
        raise ValueError("frozen controls contain duplicate seeds")
    return frame.set_index("seed").reindex(SEEDS)


def _validate_evaluation(path: Path) -> tuple[float, float, tuple[int, ...]]:
    payload = _read_json(path)
    _require_non_access(payload, context=path.name)
    records = payload.get("records") or []
    if len(records) != 400:
        raise ValueError(f"{path}: expected 400 validation records")
    question_ids = tuple(int(row.get("idx", -1)) for row in records)
    if len(set(question_ids)) != 400 or any(value < 0 for value in question_ids):
        raise ValueError(f"{path}: invalid validation question identities")
    for index, row in enumerate(records):
        for field in ("legacy_correct", "strict_correct"):
            if not isinstance(row.get(field), bool):
                raise ValueError(f"{path}:{index}: {field} is not boolean")
        if row.get("official_test_accessed") is not False:
            raise ValueError(f"{path}:{index}: official-test access is not false")
    extracted = float(np.mean([row["legacy_correct"] for row in records]))
    strict = float(np.mean([row["strict_correct"] for row in records]))
    if not math.isclose(
        extracted, _finite(payload["test_acc_legacy"], context=path.name), abs_tol=1e-12
    ):
        raise ValueError(f"{path}: extracted aggregate disagrees with records")
    if not math.isclose(
        strict, _finite(payload["test_acc_strict"], context=path.name), abs_tol=1e-12
    ):
        raise ValueError(f"{path}: strict aggregate disagrees with records")
    return extracted, strict, question_ids


def load_results(
    artifact_dir: Path,
    cell: Any,
    controls: pd.DataFrame,
    *,
    expected_commit: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    validation_support: tuple[int, ...] | None = None
    receipt_paths = sorted(artifact_dir.glob("complete_*.json"))
    if len(receipt_paths) != len(SEEDS):
        raise ValueError(f"expected {len(SEEDS)} receipts, found {len(receipt_paths)}")

    for seed in SEEDS:
        tag = f"{cell.tag}_seed{seed}"
        receipt_path = artifact_dir / f"complete_gsm8k__{tag}__JEPO_s{seed}.json"
        receipt = validate_completion_receipt(
            receipt_path, result_root=artifact_dir, verify_hashes=True
        )
        validate_receipt_identity(
            receipt,
            {
                "run_id": RUN_ID,
                "task": "gsm8k",
                "model": "qwen3-1.7b-base",
                "method": "JEPO",
                "seed": seed,
                "tag": tag,
            },
        )
        evaluation_path = _artifact(artifact_dir, receipt, "eval_")
        final_extracted, final_strict, question_ids = _validate_evaluation(
            evaluation_path
        )
        if validation_support is None:
            validation_support = question_ids
        elif question_ids != validation_support:
            raise ValueError(f"{seed}: validation question support changed")

        checkpoint_path = _artifact(artifact_dir, receipt, "checkpoint_eval_")
        checkpoints = _read_jsonl_gz(checkpoint_path)
        if (
            tuple(int(row.get("completed_rounds", -1)) for row in checkpoints)
            != CHECKPOINTS
        ):
            raise ValueError(f"{checkpoint_path}: checkpoint schedule changed")
        extracted_trajectory = []
        strict_trajectory = []
        for index, row in enumerate(checkpoints):
            metrics = dict(row.get("metrics") or {})
            _require_non_access(metrics, context=f"{checkpoint_path.name}/{index}")
            extracted_trajectory.append(
                _finite(metrics.get("test_acc_legacy"), context=checkpoint_path.name)
            )
            strict_trajectory.append(
                _finite(metrics.get("test_acc_strict"), context=checkpoint_path.name)
            )
        if not math.isclose(extracted_trajectory[-1], final_extracted, abs_tol=1e-12):
            raise ValueError(f"{seed}: extracted endpoint mismatch")
        if not math.isclose(strict_trajectory[-1], final_strict, abs_tol=1e-12):
            raise ValueError(f"{seed}: strict endpoint mismatch")

        diagnostics_path = _artifact(artifact_dir, receipt, "training_diagnostics_")
        diagnostics = _read_jsonl_gz(diagnostics_path)
        if len(diagnostics) != 32:
            raise ValueError(f"{diagnostics_path}: expected 32 rows")
        for round_index, row in enumerate(diagnostics):
            identity = {
                "run_id": RUN_ID,
                "model": "qwen3-1.7b-base",
                "task": "gsm8k",
                "method": "JEPO",
                "seed": seed,
                "tag": tag,
                "method_family": "jepo_multisample",
                "round": round_index,
                "completed_rounds": round_index + 1,
            }
            mismatches = {
                field: (row.get(field), value)
                for field, value in identity.items()
                if row.get(field) != value
            }
            if mismatches:
                raise ValueError(f"{diagnostics_path}: identity mismatch {mismatches}")
            contract = dict(row.get("contract") or {})
            expected_contract = {
                "group_size": 4,
                "supervised_coefficient": 0.01,
                "format_penalty": 10.0,
                "advantage_clip": 1.0,
                "kl_coefficient": 0.001,
                "answer_target_termination": "eos",
                "proposal_prompt": "question",
                "proposal_temperature": 1.0,
                "masked_objective_denominator": "fixed_sample_count",
            }
            if contract != expected_contract:
                raise ValueError(f"{diagnostics_path}: JEPO contract changed")
            signal = dict(row.get("signal") or {})
            optimizer = dict(row.get("optimizer") or {})
            for field in (
                "valid_generation_fraction",
                "raw_trace_advantage_std",
                "normalized_trace_advantage_std",
                "trace_advantage_clip_fraction",
                "answer_weight_ess",
                "logmean_gold_answer_probability",
            ):
                _finite(signal.get(field), context=f"{diagnostics_path}/{field}")
            for field in (
                "loss",
                "lower_bound_loss",
                "format_loss",
                "kl_loss",
                "sampled_policy_kl",
            ):
                _finite(optimizer.get(field), context=f"{diagnostics_path}/{field}")
            mechanism_rows.append(
                {
                    "seed": seed,
                    "round": round_index + 1,
                    **{f"signal_{key}": value for key, value in signal.items()},
                    **{f"optimizer_{key}": value for key, value in optimizer.items()},
                }
            )

        cell_result_path = _artifact(artifact_dir, receipt, "cell_result_")
        result = dict(_read_json(cell_result_path).get("result") or {})
        _require_non_access(result, context=cell_result_path.name)
        params = json.loads(str(result.get("params") or "{}"))
        commit = str((params.get("env") or {}).get("commit") or "")
        if len(commit) < 7 or not expected_commit.startswith(commit):
            raise ValueError(f"{cell_result_path}: execution commit mismatch")
        if int(result.get("optimizer_steps", -1)) != 32:
            raise ValueError(f"{cell_result_path}: optimizer-step schedule changed")
        if int(result.get("train_llm_gen", -1)) != 2048:
            raise ValueError(f"{cell_result_path}: training-generation budget changed")
        adapter_paths = [
            str(record.get("path") or "")
            for record in receipt.get("artifacts") or []
            if str(record.get("path") or "").startswith("adapter_")
        ]
        if not adapter_paths:
            raise ValueError(f"{receipt_path}: trained adapter is absent")

        base = controls.loc[seed]
        seed_rows.append(
            {
                "cell": CELL_ID,
                "seed": seed,
                "final_extracted": final_extracted,
                "final_strict": final_strict,
                "extracted_auc": _normalized_auc(
                    float(base["final_extracted"]), extracted_trajectory
                ),
                "strict_auc": _normalized_auc(
                    float(base["final_strict"]), strict_trajectory
                ),
                "train_llm_gen": int(result["train_llm_gen"]),
                "optimizer_steps": int(result["optimizer_steps"]),
                "generated_tokens": _finite(
                    result["generated_tokens"], context="generated tokens"
                ),
                "backward_tokens": _finite(
                    result["backward_tokens"], context="backward tokens"
                ),
                "accelerator_hours": _finite(
                    result["accelerator_hours"], context="accelerator hours"
                ),
            }
        )
    return pd.DataFrame(seed_rows), pd.DataFrame(mechanism_rows)


def summarize(
    seeds: pd.DataFrame, controls: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = {"cell": CELL_ID, "seeds": len(seeds)}
    for metric in METRICS:
        summary[metric] = float(seeds[metric].mean())
        summary[f"sd_{metric}"] = float(seeds[metric].std(ddof=1))
    for metric in (
        "train_llm_gen",
        "optimizer_steps",
        "generated_tokens",
        "backward_tokens",
        "accelerator_hours",
    ):
        summary[f"mean_{metric}"] = float(seeds[metric].mean())
    contrasts = []
    base = controls.reindex(SEEDS)
    for metric, base_metric in (
        ("final_extracted", "final_extracted"),
        ("final_strict", "final_strict"),
    ):
        differences = seeds.set_index("seed").reindex(SEEDS)[metric] - base[base_metric]
        mean = float(differences.mean())
        half_width = (
            T_CRITICAL_95_DF6 * float(differences.std(ddof=1)) / math.sqrt(len(SEEDS))
        )
        contrasts.append(
            {
                "treatment": CELL_ID,
                "control": BASE_CELL,
                "metric": metric,
                "mean_difference_pp": 100.0 * mean,
                "ci95_low_pp": 100.0 * (mean - half_width),
                "ci95_high_pp": 100.0 * (mean + half_width),
                "positive_seeds": int((differences > 0).sum()),
                "independent_unit": "paired_training_seed",
            }
        )
    return pd.DataFrame([summary]), pd.DataFrame(contrasts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-design-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--validator-marker", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-job")
    parser.add_argument("--control-seed-metrics", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    _config, cell = validate_design(args.config)
    if args.validate_design_only:
        print(json.dumps({"run_id": RUN_ID, "cells": 1, "seeds": len(SEEDS)}))
        return 0
    required = (
        "artifact_dir",
        "validator_marker",
        "expected_commit",
        "expected_source_job",
        "control_seed_metrics",
        "out_dir",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"missing result-analysis arguments: {', '.join(missing)}")

    marker = verify_marker(
        args.validator_marker,
        args.config,
        expected_commit=args.expected_commit,
        expected_source_job=args.expected_source_job,
    )
    controls = load_controls(args.control_seed_metrics)
    seeds, mechanisms = load_results(
        args.artifact_dir,
        cell,
        controls,
        expected_commit=args.expected_commit,
    )
    summary, contrasts = summarize(seeds, controls)
    analysis = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "analysis_status": "complete",
        "metric_order": ["Final Acc@1", "strict final accuracy", "trajectory AUC"],
        "independent_unit": "paired training seed",
        "cell_count": 1,
        "seed_count": len(SEEDS),
        "official_test_used": False,
        "validator_marker": marker,
        "limitations": [
            "JEPO hyperparameters are paper-derived but training scale and LoRA adaptation follow the common Qwen protocol.",
            "Historical same-seed frozen controls are reused rather than regenerated.",
            "The fixed train-derived validation partition is development evidence; the official GSM8K test remains sealed.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    seeds.to_csv(args.out_dir / "seed_metrics.csv", index=False)
    mechanisms.to_csv(args.out_dir / "mechanism_diagnostics.csv", index=False)
    summary.to_csv(args.out_dir / "method_summary.csv", index=False)
    contrasts.to_csv(args.out_dir / "paired_contrasts.csv", index=False)
    (args.out_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
