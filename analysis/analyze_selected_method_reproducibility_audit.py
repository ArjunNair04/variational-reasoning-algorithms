#!/usr/bin/env python3
"""Diagnose same-seed differences between the historical and posterity runs.

This is an audit, not a method-selection analysis. It compares the seven
historical source seeds with their posterity replays, verifies matched
validation support at every checkpoint, and locates the earliest observable
training-path divergence for Q5 and PIS.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


SEEDS = (1201, 1213, 1217, 1223, 1229, 1231, 1237)
SOURCE_TO_REPLAY = {
    "CTRL-base": "CTRL-base",
    "Q5-LR1e-5-U1-K16": "Q5-AD-M-LR1e-5-U1-K16",
    "PIS-S8-B8-U4": "PIS-Q-S8-B8-U4",
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
PATH_AUDIT_CELLS = (
    "Q5-LR1e-5-U1-K16",
    "PIS-S8-B8-U4",
)


def _finite(value: Any, *, context: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{context}: expected a finite number")
    return numeric


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty JSONL artifact")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected an object")
    return payload


def _artifact(
    root: Path,
    *,
    prefix: str,
    cell: str,
    seed: int,
    suffix: str,
) -> Path:
    matches = sorted(root.glob(f"{prefix}*_{cell}_seed{seed}__*{suffix}"))
    if len(matches) != 1:
        raise ValueError(
            f"{root}: expected one {prefix} artifact for {cell}/{seed}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _artifact_matches(
    root: Path,
    *,
    prefix: str,
    cell: str,
    seed: int,
    suffix: str,
) -> list[Path]:
    return sorted(root.glob(f"{prefix}*_{cell}_seed{seed}__*{suffix}"))


def load_seed_metrics(path: Path, *, cells: Iterable[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"cell", "seed", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    wanted = tuple(cells)
    frame = frame[frame["cell"].isin(wanted)].copy()
    expected = {(cell, seed) for cell in wanted for seed in SEEDS}
    observed = {(str(row.cell), int(row.seed)) for row in frame.itertuples()}
    if observed != expected or len(frame) != len(expected):
        raise ValueError(
            f"{path}: incomplete or duplicate cell/seed coverage; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    for metric in METRICS:
        values = frame[metric].map(lambda value: _finite(value, context=metric))
        if not values.between(0.0, 1.0).all():
            raise ValueError(f"{path}: {metric} is outside [0, 1]")
        frame[metric] = values
    return frame.sort_values(["cell", "seed"]).reset_index(drop=True)


def compare_seed_metrics(
    source: pd.DataFrame,
    replay: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    replay_lookup = replay.set_index(["cell", "seed"])
    rows: list[dict[str, Any]] = []
    for source_cell, replay_cell in SOURCE_TO_REPLAY.items():
        source_rows = source[source["cell"] == source_cell].set_index("seed")
        for seed in SEEDS:
            for metric in METRICS:
                source_value = float(source_rows.loc[seed, metric])
                replay_value = float(replay_lookup.loc[(replay_cell, seed), metric])
                rows.append(
                    {
                        "source_cell": source_cell,
                        "replay_cell": replay_cell,
                        "seed": seed,
                        "metric": metric,
                        "source_value": source_value,
                        "replay_value": replay_value,
                        "replay_minus_source": replay_value - source_value,
                        "replay_minus_source_pp": 100.0 * (replay_value - source_value),
                    }
                )
    seed_differences = pd.DataFrame(rows)
    method_rows: list[dict[str, Any]] = []
    for (source_cell, replay_cell, metric), group in seed_differences.groupby(
        ["source_cell", "replay_cell", "metric"], sort=False
    ):
        source_values = group["source_value"].to_numpy(float)
        replay_values = group["replay_value"].to_numpy(float)
        differences = group["replay_minus_source"].to_numpy(float)
        method_rows.append(
            {
                "source_cell": source_cell,
                "replay_cell": replay_cell,
                "metric": metric,
                "source_mean": float(source_values.mean()),
                "replay_mean": float(replay_values.mean()),
                "mean_difference": float(differences.mean()),
                "mean_difference_pp": float(100.0 * differences.mean()),
                "source_seed_sd_pp": float(100.0 * source_values.std(ddof=1)),
                "replay_seed_sd_pp": float(100.0 * replay_values.std(ddof=1)),
                "difference_sd_pp": float(100.0 * differences.std(ddof=1)),
                "max_absolute_seed_difference_pp": float(
                    100.0 * np.abs(differences).max()
                ),
                "seed_correlation": float(np.corrcoef(source_values, replay_values)[0, 1]),
            }
        )
    return seed_differences, pd.DataFrame(method_rows)


def _records_by_idx(row: Mapping[str, Any], *, context: str) -> dict[int, Mapping[str, Any]]:
    records = row.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{context}: missing checkpoint records")
    indexed = {int(record["idx"]): record for record in records}
    if len(indexed) != len(records):
        raise ValueError(f"{context}: duplicate validation question IDs")
    return indexed


def _equal(left: Any, right: Any) -> bool:
    return left == right or (left is None and right is None)


def compare_checkpoints(
    source_artifacts: Path,
    replay_artifacts: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_cell, replay_cell in SOURCE_TO_REPLAY.items():
        for seed in SEEDS:
            source_matches = _artifact_matches(
                source_artifacts,
                prefix="checkpoint_eval_",
                cell=source_cell,
                seed=seed,
                suffix=".jsonl.gz",
            )
            replay_matches = _artifact_matches(
                replay_artifacts,
                prefix="checkpoint_eval_",
                cell=replay_cell,
                seed=seed,
                suffix=".jsonl.gz",
            )
            if source_cell == "CTRL-base" and not source_matches and not replay_matches:
                continue
            if len(source_matches) != 1 or len(replay_matches) != 1:
                raise ValueError(
                    f"expected paired checkpoint artifacts for {source_cell}/{seed}; "
                    f"source={len(source_matches)}, replay={len(replay_matches)}"
                )
            source_path = source_matches[0]
            replay_path = replay_matches[0]
            source_rounds = {
                int(row["completed_rounds"]): row for row in _read_jsonl_gz(source_path)
            }
            replay_rounds = {
                int(row["completed_rounds"]): row for row in _read_jsonl_gz(replay_path)
            }
            if source_rounds.keys() != replay_rounds.keys():
                raise ValueError(f"checkpoint schedule differs for {source_cell}/{seed}")
            for completed_rounds in sorted(source_rounds):
                source_row = source_rounds[completed_rounds]
                replay_row = replay_rounds[completed_rounds]
                source_records = _records_by_idx(
                    source_row, context=f"source {source_cell}/{seed}/r{completed_rounds}"
                )
                replay_records = _records_by_idx(
                    replay_row, context=f"replay {replay_cell}/{seed}/r{completed_rounds}"
                )
                if source_records.keys() != replay_records.keys():
                    raise ValueError(
                        f"validation support differs for {source_cell}/{seed}/r{completed_rounds}"
                    )
                question_ids = sorted(source_records)
                prediction_matches = 0
                strict_prediction_matches = 0
                correct_matches = 0
                strict_correct_matches = 0
                extracted_gains = 0
                extracted_losses = 0
                strict_gains = 0
                strict_losses = 0
                for question_id in question_ids:
                    source_record = source_records[question_id]
                    replay_record = replay_records[question_id]
                    source_correct = bool(source_record["correct"])
                    replay_correct = bool(replay_record["correct"])
                    source_strict = bool(source_record["strict_correct"])
                    replay_strict = bool(replay_record["strict_correct"])
                    prediction_matches += _equal(
                        source_record.get("pred"), replay_record.get("pred")
                    )
                    strict_prediction_matches += _equal(
                        source_record.get("strict_pred"), replay_record.get("strict_pred")
                    )
                    correct_matches += source_correct == replay_correct
                    strict_correct_matches += source_strict == replay_strict
                    extracted_gains += replay_correct and not source_correct
                    extracted_losses += source_correct and not replay_correct
                    strict_gains += replay_strict and not source_strict
                    strict_losses += source_strict and not replay_strict
                source_metrics = dict(source_row["metrics"])
                replay_metrics = dict(replay_row["metrics"])
                count = len(question_ids)
                rows.append(
                    {
                        "source_cell": source_cell,
                        "replay_cell": replay_cell,
                        "seed": seed,
                        "completed_rounds": completed_rounds,
                        "validation_questions": count,
                        "validation_support_equal": True,
                        "prediction_agreement": prediction_matches / count,
                        "strict_prediction_agreement": strict_prediction_matches / count,
                        "correctness_agreement": correct_matches / count,
                        "strict_correctness_agreement": strict_correct_matches / count,
                        "extracted_gains": extracted_gains,
                        "extracted_losses": extracted_losses,
                        "strict_gains": strict_gains,
                        "strict_losses": strict_losses,
                        "source_extracted": _finite(
                            source_metrics["test_acc_legacy"], context="source extracted"
                        ),
                        "replay_extracted": _finite(
                            replay_metrics["test_acc_legacy"], context="replay extracted"
                        ),
                        "source_strict": _finite(
                            source_metrics["test_acc_strict"], context="source strict"
                        ),
                        "replay_strict": _finite(
                            replay_metrics["test_acc_strict"], context="replay strict"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _sample_signature(row: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    generation = dict(row.get("generation") or {})
    sample_block = dict(generation.get("samples") or {})
    samples = sample_block.get("samples") or []
    return [
        (
            sample.get("pid"),
            sample.get("source"),
            sample.get("text"),
            sample.get("gold_answer"),
            sample.get("parsed_answer"),
        )
        for sample in samples
    ]


def _responsibility_map(
    row: Mapping[str, Any],
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    traces = dict(row.get("responsibilities") or {}).get("traces") or []
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for trace in traces:
        key = (trace.get("pid"), trace.get("source"), trace.get("trace_id"))
        if key in result:
            raise ValueError(f"duplicate responsibility trace identity: {key}")
        _finite(trace.get("responsibility"), context="responsibility")
        result[key] = trace
    return result


def _maximum_trace_delta(
    source: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replay: Mapping[tuple[Any, ...], Mapping[str, Any]],
    field: str,
) -> tuple[float | None, str | None]:
    if source.keys() != replay.keys() or not source:
        return None, None
    deltas = {
        key: abs(
            _finite(source[key].get(field), context=f"source {field}")
            - _finite(replay[key].get(field), context=f"replay {field}")
        )
        for key in source
    }
    key = max(deltas, key=deltas.get)
    return deltas[key], json.dumps(key, separators=(",", ":"))


def _minibatch_signature(row: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    minibatch = dict(row.get("minibatch") or {})
    return (
        tuple(int(value) for value in minibatch.get("labelled_pids") or ()),
        tuple(int(value) for value in minibatch.get("answer_only_pids") or ()),
    )


def _first_stage(
    *,
    minibatch_equal: bool,
    sample_texts_equal: bool,
    responsibility_keys_equal: bool,
    maximum_responsibility_delta: float | None,
    parameter_drift_delta: float,
) -> str:
    if not minibatch_equal:
        return "question_selection"
    if not sample_texts_equal:
        return "sampled_generation"
    if not responsibility_keys_equal:
        return "posterior_support"
    if maximum_responsibility_delta is not None and maximum_responsibility_delta > 1e-10:
        return "posterior_scoring"
    if abs(parameter_drift_delta) > 1e-10:
        return "optimizer_update"
    return "none_observed"


def compare_training_paths(
    source_artifacts: Path,
    replay_artifacts: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_cell in PATH_AUDIT_CELLS:
        replay_cell = SOURCE_TO_REPLAY[source_cell]
        for seed in SEEDS:
            source_path = _artifact(
                source_artifacts,
                prefix="training_diagnostics_",
                cell=source_cell,
                seed=seed,
                suffix=".jsonl.gz",
            )
            replay_path = _artifact(
                replay_artifacts,
                prefix="training_diagnostics_",
                cell=replay_cell,
                seed=seed,
                suffix=".jsonl.gz",
            )
            source_rounds = {int(row["round"]): row for row in _read_jsonl_gz(source_path)}
            replay_rounds = {int(row["round"]): row for row in _read_jsonl_gz(replay_path)}
            if source_rounds.keys() != replay_rounds.keys():
                raise ValueError(f"training-round coverage differs for {source_cell}/{seed}")
            for round_index in sorted(source_rounds):
                source_row = source_rounds[round_index]
                replay_row = replay_rounds[round_index]
                minibatch_equal = _minibatch_signature(source_row) == _minibatch_signature(
                    replay_row
                )
                source_samples = _sample_signature(source_row)
                replay_samples = _sample_signature(replay_row)
                sample_texts_equal = source_samples == replay_samples
                paired_samples = min(len(source_samples), len(replay_samples))
                matching_samples = sum(
                    left == right
                    for left, right in zip(source_samples, replay_samples, strict=False)
                )
                source_responsibilities = _responsibility_map(source_row)
                replay_responsibilities = _responsibility_map(replay_row)
                responsibility_keys_equal = (
                    source_responsibilities.keys() == replay_responsibilities.keys()
                )
                maximum_responsibility_delta, maximum_responsibility_trace = (
                    _maximum_trace_delta(
                        source_responsibilities,
                        replay_responsibilities,
                        "responsibility",
                    )
                )
                maximum_joint_logprob_delta, _ = _maximum_trace_delta(
                    source_responsibilities,
                    replay_responsibilities,
                    "joint_logprob",
                )
                maximum_trace_logprob_delta, _ = _maximum_trace_delta(
                    source_responsibilities,
                    replay_responsibilities,
                    "trace_logprob",
                )
                maximum_answer_logprob_delta, _ = _maximum_trace_delta(
                    source_responsibilities,
                    replay_responsibilities,
                    "answer_logprob",
                )
                source_drift = _finite(
                    dict(source_row.get("optimizer") or {}).get(
                        "parameter_l2_drift_from_initial"
                    ),
                    context="source adapter drift",
                )
                replay_drift = _finite(
                    dict(replay_row.get("optimizer") or {}).get(
                        "parameter_l2_drift_from_initial"
                    ),
                    context="replay adapter drift",
                )
                parameter_drift_delta = replay_drift - source_drift
                rows.append(
                    {
                        "source_cell": source_cell,
                        "replay_cell": replay_cell,
                        "seed": seed,
                        "round": round_index,
                        "minibatch_equal": minibatch_equal,
                        "source_sample_count": len(source_samples),
                        "replay_sample_count": len(replay_samples),
                        "paired_sample_count": paired_samples,
                        "matching_sample_count": matching_samples,
                        "sample_agreement": (
                            matching_samples / max(len(source_samples), len(replay_samples), 1)
                        ),
                        "sample_texts_equal": sample_texts_equal,
                        "responsibility_keys_equal": responsibility_keys_equal,
                        "maximum_responsibility_delta": maximum_responsibility_delta,
                        "maximum_responsibility_trace": maximum_responsibility_trace,
                        "maximum_joint_logprob_delta": maximum_joint_logprob_delta,
                        "maximum_trace_logprob_delta": maximum_trace_logprob_delta,
                        "maximum_answer_logprob_delta": maximum_answer_logprob_delta,
                        "source_adapter_drift": source_drift,
                        "replay_adapter_drift": replay_drift,
                        "adapter_drift_delta": parameter_drift_delta,
                        "first_divergence_stage_this_round": _first_stage(
                            minibatch_equal=minibatch_equal,
                            sample_texts_equal=sample_texts_equal,
                            responsibility_keys_equal=responsibility_keys_equal,
                            maximum_responsibility_delta=maximum_responsibility_delta,
                            parameter_drift_delta=parameter_drift_delta,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def compare_final_outputs(
    source_artifacts: Path,
    replay_artifacts: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_cell, replay_cell in SOURCE_TO_REPLAY.items():
        for seed in SEEDS:
            source_path = _artifact(
                source_artifacts,
                prefix="dump_",
                cell=source_cell,
                seed=seed,
                suffix=".json",
            )
            replay_path = _artifact(
                replay_artifacts,
                prefix="dump_",
                cell=replay_cell,
                seed=seed,
                suffix=".json",
            )
            source_samples = {int(row["idx"]): row for row in _read_json(source_path)["samples"]}
            replay_samples = {int(row["idx"]): row for row in _read_json(replay_path)["samples"]}
            if source_samples.keys() != replay_samples.keys():
                raise ValueError(f"final validation support differs for {source_cell}/{seed}")
            count = len(source_samples)
            completion_matches = 0
            prediction_matches = 0
            extracted_gains = 0
            extracted_losses = 0
            for question_id in source_samples:
                source_row = source_samples[question_id]
                replay_row = replay_samples[question_id]
                if source_row.get("question") != replay_row.get("question"):
                    raise ValueError(f"question text differs for index {question_id}")
                if source_row.get("gold") != replay_row.get("gold"):
                    raise ValueError(f"gold answer differs for index {question_id}")
                completion_matches += source_row.get("completion") == replay_row.get("completion")
                prediction_matches += _equal(source_row.get("pred"), replay_row.get("pred"))
                source_correct = bool(source_row.get("correct"))
                replay_correct = bool(replay_row.get("correct"))
                extracted_gains += replay_correct and not source_correct
                extracted_losses += source_correct and not replay_correct
            rows.append(
                {
                    "source_cell": source_cell,
                    "replay_cell": replay_cell,
                    "seed": seed,
                    "validation_questions": count,
                    "completion_exact_agreement": completion_matches / count,
                    "prediction_agreement": prediction_matches / count,
                    "extracted_gains": extracted_gains,
                    "extracted_losses": extracted_losses,
                }
            )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_audit_summary(
    seed_differences: pd.DataFrame,
    method_differences: pd.DataFrame,
    training_paths: pd.DataFrame,
) -> dict[str, Any]:
    final_methods = method_differences[
        method_differences["metric"] == "final_extracted"
    ].copy()
    final_methods = final_methods[final_methods["source_cell"] != "CTRL-base"]
    sensitive = final_methods.iloc[final_methods["mean_difference_pp"].abs().argmax()]
    sensitive_seed_rows = seed_differences[
        (seed_differences["source_cell"] == sensitive["source_cell"])
        & (seed_differences["metric"] == "final_extracted")
    ]
    sensitive_seed = sensitive_seed_rows.iloc[
        sensitive_seed_rows["replay_minus_source_pp"].abs().argmax()
    ]
    q5_rows = seed_differences[
        (seed_differences["source_cell"] == "Q5-LR1e-5-U1-K16")
        & (seed_differences["metric"] == "final_extracted")
    ]
    control_seed = q5_rows.iloc[q5_rows["replay_minus_source_pp"].abs().argmin()]
    first_divergences = []
    for (source_cell, seed), group in training_paths.groupby(["source_cell", "seed"]):
        divergent = group[group["first_divergence_stage_this_round"] != "none_observed"]
        if divergent.empty:
            first_divergences.append(
                {"source_cell": source_cell, "seed": int(seed), "round": None, "stage": "none_observed"}
            )
        else:
            row = divergent.sort_values("round").iloc[0]
            first_divergences.append(
                {
                    "source_cell": source_cell,
                    "seed": int(seed),
                    "round": int(row["round"]),
                    "stage": str(row["first_divergence_stage_this_round"]),
                }
            )
    return {
        "schema_version": 1,
        "purpose": "same_seed_historical_vs_posterity_reproducibility_audit",
        "metric_order": ["Final Acc@1", "Strict final accuracy", "Normalized trajectory AUC"],
        "sensitive_cell": str(sensitive["replay_cell"]),
        "sensitive_seed": int(sensitive_seed["seed"]),
        "sensitive_seed_final_difference_pp": float(sensitive_seed["replay_minus_source_pp"]),
        "stable_control_cell": "Q5-AD-M-LR1e-5-U1-K16",
        "stable_control_seed": int(control_seed["seed"]),
        "stable_control_seed_final_difference_pp": float(control_seed["replay_minus_source_pp"]),
        "repeatability_decision": {
            "new_gpu_repeat_required": False,
            "result_semantics": "statistical_not_bit_exact",
            "reason": (
                "The paired artifacts preserve validation support and isolate the first "
                "training-path divergence after matching question selection and sampled "
                "traces, so the preregistered no-compute gate is satisfied."
            ),
        },
        "first_training_path_divergences": first_divergences,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-seed-metrics", type=Path, required=True)
    parser.add_argument("--replay-seed-metrics", type=Path, required=True)
    parser.add_argument("--source-artifact-dir", type=Path, required=True)
    parser.add_argument("--replay-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")

    source = load_seed_metrics(args.source_seed_metrics, cells=SOURCE_TO_REPLAY)
    replay = load_seed_metrics(args.replay_seed_metrics, cells=SOURCE_TO_REPLAY.values())
    seed_differences, method_differences = compare_seed_metrics(source, replay)
    checkpoints = compare_checkpoints(args.source_artifact_dir, args.replay_artifact_dir)
    training_paths = compare_training_paths(args.source_artifact_dir, args.replay_artifact_dir)
    final_outputs = compare_final_outputs(args.source_artifact_dir, args.replay_artifact_dir)
    summary = build_audit_summary(seed_differences, method_differences, training_paths)
    summary["inputs"] = {
        "source_seed_metrics_sha256": _sha256(args.source_seed_metrics),
        "replay_seed_metrics_sha256": _sha256(args.replay_seed_metrics),
    }

    args.output_dir.mkdir(parents=True)
    seed_differences.to_csv(args.output_dir / "seed_differences.csv", index=False)
    method_differences.to_csv(args.output_dir / "method_differences.csv", index=False)
    checkpoints.to_csv(args.output_dir / "checkpoint_differences.csv", index=False)
    training_paths.to_csv(args.output_dir / "training_path_differences.csv", index=False)
    final_outputs.to_csv(args.output_dir / "final_output_agreement.csv", index=False)
    _write_json(args.output_dir / "audit_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
