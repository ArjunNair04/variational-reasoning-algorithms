#!/usr/bin/env python3
"""Frozen paired analysis for the exact temperature-mixture PIS test."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "5c7a10e2"
EXECUTION_COMMIT = "6683d69f133a34e2ff8816d55754b755e8d7a4d0"
SOURCE_JOB = "7224195"
CONFIG_SHA256 = "0c2c10af13b1bde1636b75b55b0ac7def367243b383444a2c16b10b3667ebdb9"
TAG_PREFIX = "q3_l2r_temperature_mixture"
LOG_STEM = "qwen3_17b_l2r_temperature_mixture"
SEEDS = (1481, 1483, 1487, 1489, 1499, 1511, 1523)
CELLS = ("PIS-T1", "PIS-TMIX1.2")
CONTROL, TREATMENT = CELLS
CHECKPOINTS = (1, 2, 4, 8, 16, 24, 32)
T95_ONE_SIDED_DF6 = 1.943180281
FAILURE_RE = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemoryError|No space left on device|"
    r"Disk quota exceeded|FAILED",
    re.IGNORECASE,
)
SHELL_NON_ACCESS_LOG_MARKER = "official_test_accessed=false"
PYTHON_NON_ACCESS_LOG_MARKERS = (
    "'eval_official_test_accessed': False",
    "'passk_official_test_accessed': False",
)
ACCESS_LOG_MARKERS = (
    "official_test_accessed=true",
    "'eval_official_test_accessed': True",
    "'passk_official_test_accessed': True",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite number")
    return result


def _close(observed: Any, expected: float, context: str, atol: float = 1e-6) -> None:
    if not math.isclose(_finite(observed, context), expected, rel_tol=1e-6, abs_tol=atol):
        raise ValueError(f"{context}: numerical identity changed")


def _resource_counters(result: Mapping[str, Any], context: str) -> dict[str, float]:
    """Require the resource counters registered by the frozen design."""

    counters = {
        field: _finite(result.get(field), f"{context}/{field}")
        for field in (
            "generated_tokens",
            "backward_tokens",
            "optimizer_steps",
            "accelerator_hours",
        )
    }
    for field, value in counters.items():
        if value < 0:
            raise ValueError(f"{context}/{field}: expected a nonnegative number")
    return counters


def load_design(config_path: Path) -> dict[str, Any]:
    raw = config_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("temperature-mixture configuration checksum changed")
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or str(config.get("run_id")) != RUN_ID:
        raise ValueError("temperature-mixture run ID changed")
    defaults = dict(config.get("defaults") or {})
    expected_defaults = {
        "model": "qwen3-1.7b-base",
        "rounds": 32,
        "seeds": 7,
        "seed_values": list(SEEDS),
        "n_test": 400,
        "train_partition": "train",
        "eval_partition": "validation",
        "answer_event_mode": "strict_terminal_marker",
        "evaluation_prompt": "question",
        "batch": 64,
        "G": 8,
        "prompts": 128,
        "shots": 3,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_target_set": "attention_mlp",
        "eval_rounds": list(CHECKPOINTS),
        "passk": 8,
        "passk_n": 100,
    }
    for key, expected in expected_defaults.items():
        if defaults.get(key) != expected:
            raise ValueError(f"default {key} changed")
    diagnostic = dict(config.get("diagnostic") or {})
    design = dict(diagnostic.get("design") or {})
    if design.get("cells") != list(CELLS) or design.get("array_tasks") != 14:
        raise ValueError("registered two-cell design changed")
    if dict(diagnostic.get("fixed_contract") or {}).get("official_test_used") is not False:
        raise ValueError("official-test prohibition is missing")
    variants = [dict(row) for row in (config.get("algos") or {}).get("L2R", [])]
    if [row.get("cell_id") for row in variants] != list(CELLS):
        raise ValueError("temperature-mixture cell order changed")
    expected_axes = {
        CONTROL: ("single", 1.0, 1.0, "prior_corrected"),
        TREATMENT: ("question_temperature", 0.5, 1.2, "mixed_prior_corrected"),
    }
    for variant in variants:
        settings = {**defaults, **variant}
        cell = str(variant["cell_id"])
        observed = (
            settings.get("proposal_mixture"),
            float(settings.get("proposal_prior_fraction")),
            float(settings.get("proposal_temperature")),
            settings.get("responsibility_score"),
        )
        if observed != expected_axes[cell]:
            raise ValueError(f"{cell}: proposal estimator changed")
        for key, expected in {
            "iters": 4,
            "reader_mode": "moving",
            "answer_target_termination": "eos",
            "gold_in_buffer": False,
            "l2r_buffer_semantics": "fresh_multiset",
            "proposal_prompt": "question",
            "trace_segmentation": "validated",
            "responsibility_temperature": 1.0,
            "responsibility_projection": "none",
            "mstep_objective": "joint",
            "archive_limit": 0,
            "replay_limit": 0,
            "reader_decode_filter": False,
            "kl_coef": 0.0,
        }.items():
            if settings.get(key) != expected:
                raise ValueError(f"{cell}: invariant {key} changed")
    return config


def verify_marker(marker_path: Path) -> dict[str, Any]:
    marker = _read_json(marker_path)
    expected = {
        "schema_version": 1,
        "status": "ok",
        "run_id": RUN_ID,
        "execution_commit": EXECUTION_COMMIT,
        "configuration_sha256": CONFIG_SHA256,
        "source_job_id": SOURCE_JOB,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"validator marker {key} mismatch")
    return marker


def verify_logs(log_dir: Path) -> None:
    expected = {
        log_dir / f"{LOG_STEM}.{SOURCE_JOB}.{task}.log"
        for task in range(1, 15)
    }
    observed = set(log_dir.glob(f"{LOG_STEM}.{SOURCE_JOB}.*.log"))
    if observed != expected:
        raise ValueError("payload log set does not contain the exact fourteen tasks")
    for path in expected:
        text = path.read_text(encoding="utf-8", errors="replace")
        if FAILURE_RE.search(text):
            raise ValueError(f"{path}: failure signature found")
        if any(marker in text for marker in ACCESS_LOG_MARKERS):
            raise ValueError(f"{path}: official-test access found")
        has_shell_epilog = SHELL_NON_ACCESS_LOG_MARKER in text
        has_python_epilog = all(
            marker in text for marker in PYTHON_NON_ACCESS_LOG_MARKERS
        )
        if not (has_shell_epilog or has_python_epilog):
            raise ValueError(f"{path}: official-test non-access epilog missing")


def _expected_tag(cell: str, seed: int) -> str:
    return f"{TAG_PREFIX}_{RUN_ID}_L2R_{cell}_seed{seed}"


def _artifact_map(root: Path, receipt: Mapping[str, Any], context: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"{context}: artifacts are missing")
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ValueError(f"{context}: malformed artifact entry")
        relative = str(entry.get("path") or "")
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"{context}: missing or escaped artifact {relative}")
        if int(entry.get("size", -1)) != path.stat().st_size:
            raise ValueError(f"{context}: size mismatch for {relative}")
        if entry.get("sha256") != _sha256(path):
            raise ValueError(f"{context}: checksum mismatch for {relative}")
        result[relative] = path
    return result


def _single_prefix(artifacts: Mapping[str, Path], prefix: str, context: str) -> Path:
    matches = [path for relative, path in artifacts.items() if Path(relative).name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"{context}: expected one {prefix} artifact")
    return matches[0]


def _require_non_access(record: Mapping[str, Any], context: str) -> None:
    if record.get("official_test_accessed") is not False:
        raise ValueError(f"{context}: official-test non-access missing")
    if record.get("eval_source_split") != "train":
        raise ValueError(f"{context}: evaluation source is not train")
    if record.get("dataset_splits_loaded") != ["train"]:
        raise ValueError(f"{context}: unexpected dataset split loaded")


def _metrics(records: Sequence[Mapping[str, Any]], context: str) -> dict[str, float]:
    if len(records) != 400:
        raise ValueError(f"{context}: expected 400 validation records")
    values: dict[str, list[bool]] = {
        "extracted": [], "strict": [], "natural_eos": [], "format_failure": []
    }
    indices: set[int] = set()
    for index, record in enumerate(records):
        _require_non_access(record, f"{context}/{index}")
        idx = record.get("idx")
        if not isinstance(idx, int) or idx in indices:
            raise ValueError(f"{context}: invalid validation identity")
        indices.add(idx)
        fields = {
            "extracted": record.get("legacy_correct", record.get("correct")),
            "strict": record.get("strict_correct", record.get("correct")),
            "natural_eos": record.get("generated_eos"),
            "format_failure": record.get("strict_format_failure", record.get("format_failure")),
        }
        for key, value in fields.items():
            if not isinstance(value, bool):
                raise ValueError(f"{context}/{index}: {key} is not boolean")
            values[key].append(value)
    return {key: float(np.mean(value)) for key, value in values.items()}


def verify_mechanism(cell: str, diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if len(diagnostics) != 32:
        raise ValueError(f"{cell}: expected 32 diagnostic rounds")
    ess: list[float] = []
    correct_coverage: list[float] = []
    prior_mass: list[float] = []
    for round_index, row in enumerate(diagnostics):
        if row.get("method_family") != "l2r" or row.get("round") != round_index:
            raise ValueError(f"{cell}/round{round_index}: diagnostic identity changed")
        config = dict(row.get("configuration") or {})
        expected_mixture = "single" if cell == CONTROL else "question_temperature"
        expected_temperature = 1.0 if cell == CONTROL else 1.2
        if config.get("proposal_mixture") != expected_mixture:
            raise ValueError(f"{cell}/round{round_index}: proposal mixture changed")
        _close(config.get("proposal_temperature"), expected_temperature, f"{cell}/temperature")
        generation = dict(row.get("generation") or {})
        if generation.get("generations") != 64 or generation.get("cumulative_generations") != 64 * (round_index + 1):
            raise ValueError(f"{cell}/round{round_index}: generation budget changed")
        traces = list(row.get("sampled_traces") or [])
        if len(traces) != 64:
            raise ValueError(f"{cell}/round{round_index}: sampled trace count changed")
        grouped: dict[int, list[float]] = {}
        for trace in traces:
            if trace.get("proposal_prompt") != "question":
                raise ValueError(f"{cell}/round{round_index}: proposal prompt changed")
            grouped.setdefault(int(trace["pid"]), []).append(
                _finite(trace.get("proposal_temperature"), "proposal temperature")
            )
        if len(grouped) != 8 or any(len(rows) != 8 for rows in grouped.values()):
            raise ValueError(f"{cell}/round{round_index}: support allocation changed")
        for temperatures in grouped.values():
            expected = ([1.0] * 8) if cell == CONTROL else ([1.0] * 4 + [1.2] * 4)
            if sorted(temperatures) != sorted(expected):
                raise ValueError(f"{cell}/round{round_index}: temperature allocation changed")
        posterior = dict(row.get("posterior") or {})
        question_count = posterior.get("question_count")
        if not isinstance(question_count, int) or not 0 <= question_count <= 8:
            raise ValueError(f"{cell}/round{round_index}: invalid positive-support question count")
        if question_count:
            current_ess = _finite(posterior.get("ess_fraction"), f"{cell}/round{round_index}/ESS")
            if not 0 < current_ess <= 1:
                raise ValueError(f"{cell}/round{round_index}: ESS is outside (0,1]")
            ess.append(current_ess)
        elif posterior.get("active_rows") != 0:
            raise ValueError(f"{cell}/round{round_index}: empty posterior has active rows")
        coverage = _finite(generation.get("resolved_initial"), "resolved questions") / 8.0
        if not 0 <= coverage <= 1:
            raise ValueError(f"{cell}/round{round_index}: invalid correct-trace coverage")
        correct_coverage.append(coverage)
        if cell == TREATMENT:
            _close(generation.get("prior_generations"), 32, f"{cell}/prior generations")
            _close(generation.get("temperature_generations"), 32, f"{cell}/hot generations")
            if question_count:
                component_mass = _finite(
                    posterior.get("proposal_prior_posterior_mass"),
                    f"{cell}/round{round_index}/component mass",
                )
                if not 0 <= component_mass <= 1:
                    raise ValueError(f"{cell}/round{round_index}: component mass outside [0,1]")
                prior_mass.append(component_mass)
            top = list(row.get("top_traces") or [])
            if question_count and not top:
                raise ValueError(f"{cell}/round{round_index}: mixture audit rows missing")
            for trace in top:
                p1 = _finite(trace.get("policy_h_logp"), "temperature-one log density")
                p12 = _finite(trace.get("answer_proposal_h_logp"), "hotter log density")
                expected_mix = float(np.logaddexp(p1 + math.log(0.5), p12 + math.log(0.5)))
                _close(trace.get("proposal_mixture_logp"), expected_mix, "mixture log density", 2e-5)
                _close(
                    trace.get("proposal_log_importance_correction"),
                    p1 - expected_mix,
                    "mixture importance correction",
                    2e-5,
                )
    return {
        "mean_ess_fraction": float(np.mean(ess)) if ess else 0.0,
        "mean_correct_trace_question_coverage": float(np.mean(correct_coverage)),
        "mean_temperature_one_posterior_mass": (
            float(np.mean(prior_mass)) if prior_mass else 1.0
        ),
    }


def load_results(artifact_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mechanism: dict[str, Any] = {}
    paired_support: dict[tuple[int, int], tuple[int, ...]] = {}
    paired_schedules: dict[tuple[int, int], tuple[int, ...]] = {}
    for cell in CELLS:
        cell_mechanism: list[dict[str, float]] = []
        for seed in SEEDS:
            tag = _expected_tag(cell, seed)
            receipt_path = artifact_dir / f"complete_gsm8k__{tag}__L2R_s{seed}.json"
            receipt = _read_json(receipt_path)
            identity = dict(receipt.get("identity") or {})
            expected_identity = {
                "status": "complete",
                "run_id": RUN_ID,
                "method": "L2R",
                "seed": seed,
                "tag": tag,
            }
            if receipt.get("status") != "complete":
                raise ValueError(f"{receipt_path.name}: receipt is incomplete")
            for key, value in expected_identity.items():
                observed = receipt.get(key) if key == "status" else identity.get(key)
                if observed != value:
                    raise ValueError(f"{receipt_path.name}: {key} mismatch")
            artifacts = _artifact_map(artifact_dir, receipt, receipt_path.name)
            result_path = _single_prefix(artifacts, "cell_result_", receipt_path.name)
            eval_path = _single_prefix(artifacts, "eval_gsm8k__", receipt_path.name)
            checkpoint_path = _single_prefix(artifacts, "checkpoint_eval_", receipt_path.name)
            diagnostic_path = _single_prefix(artifacts, "training_diagnostics_", receipt_path.name)
            passk_path = _single_prefix(artifacts, "passk_", receipt_path.name)
            result = dict(_read_json(result_path).get("result") or {})
            if result.get("run_id") != RUN_ID or int(result.get("train_llm_gen", -1)) != 2048:
                raise ValueError(f"{result_path.name}: execution budget changed")
            final_payload = _read_json(eval_path)
            final_records = list(final_payload.get("records") or [])
            final = _metrics(final_records, eval_path.name)
            support = tuple(int(record["idx"]) for record in final_records)
            checkpoints = _read_jsonl_gz(checkpoint_path)
            rounds = tuple(int(row.get("completed_rounds", -1)) for row in checkpoints)
            if rounds != CHECKPOINTS:
                raise ValueError(f"{checkpoint_path.name}: checkpoint schedule changed")
            curve: list[float] = []
            strict_curve: list[float] = []
            for checkpoint in checkpoints:
                records = list(checkpoint.get("records") or [])
                metrics = _metrics(records, checkpoint_path.name)
                indices = tuple(int(record["idx"]) for record in records)
                if indices != support:
                    raise ValueError(f"{checkpoint_path.name}: validation support changed")
                key = (seed, int(checkpoint["completed_rounds"]))
                if key in paired_support and paired_support[key] != indices:
                    raise ValueError(f"seed {seed}: cells use different validation support")
                paired_support[key] = indices
                curve.append(metrics["extracted"])
                strict_curve.append(metrics["strict"])
            if not math.isclose(curve[-1], final["extracted"], abs_tol=1e-9):
                raise ValueError(f"{cell}/{seed}: terminal extracted metric mismatch")
            passk = _read_json(passk_path)
            pass_records = list(passk.get("records") or [])
            if len(pass_records) != 100:
                raise ValueError(f"{passk_path.name}: expected 100 pass@8 records")
            for index, record in enumerate(pass_records):
                _require_non_access(record, f"{passk_path.name}/{index}")
                if int(record.get("k", -1)) != 8 or not isinstance(record.get("n_correct"), int):
                    raise ValueError(f"{passk_path.name}: malformed pass@8 record")
            pass8 = float(np.mean([record["n_correct"] > 0 for record in pass_records]))
            diagnostics = _read_jsonl_gz(diagnostic_path)
            schedule = tuple(
                pid
                for diagnostic in diagnostics
                for pid in dict.fromkeys(
                    int(trace["pid"])
                    for trace in list(diagnostic.get("sampled_traces") or [])
                )
            )
            schedule_key = (seed, 0)
            if schedule_key in paired_schedules and paired_schedules[schedule_key] != schedule:
                raise ValueError(f"seed {seed}: question schedules differ across cells")
            paired_schedules[schedule_key] = schedule
            cell_mechanism.append(verify_mechanism(cell, diagnostics))
            x = np.asarray(CHECKPOINTS, dtype=float)
            auc = float(np.trapezoid(np.asarray(curve), x) / (x[-1] - x[0]))
            late_drop = float(max(curve[index] for index, value in enumerate(CHECKPOINTS) if value >= 8) - curve[-1])
            resources = _resource_counters(result, result_path.name)
            rows.append({
                "cell": cell,
                "seed": seed,
                "final_extracted": final["extracted"],
                "final_strict": final["strict"],
                "final_natural_eos": final["natural_eos"],
                "final_format_failure": final["format_failure"],
                "extracted_auc": auc,
                "late_drop": late_drop,
                "pass8": pass8,
                **resources,
            })
        mechanism[cell] = {
            key: float(np.mean([row[key] for row in cell_mechanism]))
            for key in cell_mechanism[0]
        }
    return pd.DataFrame(rows), mechanism


def _paired_effect(table: pd.DataFrame, metric: str) -> dict[str, Any]:
    pivot = table.pivot(index="seed", columns="cell", values=metric).loc[list(SEEDS)]
    delta = (pivot[TREATMENT] - pivot[CONTROL]).to_numpy(dtype=float)
    mean = float(np.mean(delta))
    sem = float(np.std(delta, ddof=1) / math.sqrt(len(delta)))
    return {
        "metric": metric,
        "mean_delta": mean,
        "one_sided_95_lower": mean - T95_ONE_SIDED_DF6 * sem,
        "positive_seeds": int(np.sum(delta > 0)),
        "seed_deltas": {str(seed): float(value) for seed, value in zip(SEEDS, delta, strict=True)},
    }


def analyse(table: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    effects = [
        _paired_effect(table, metric)
        for metric in (
            "final_extracted", "final_strict", "extracted_auc", "late_drop",
            "final_format_failure", "final_natural_eos", "pass8",
        )
    ]
    by_metric = {row["metric"]: row for row in effects}
    primary = by_metric["final_extracted"]
    decision = {
        "advance": bool(
            primary["mean_delta"] >= 0.01
            and primary["one_sided_95_lower"] > 0
            and primary["positive_seeds"] >= 5
            and by_metric["final_strict"]["one_sided_95_lower"] > -0.01
            and by_metric["extracted_auc"]["one_sided_95_lower"] > -0.01
            and table.loc[table.cell == TREATMENT, "late_drop"].mean() <= 0.03
            and by_metric["final_format_failure"]["mean_delta"] <= 0.03
            and by_metric["final_natural_eos"]["mean_delta"] >= -0.02
        ),
        "primary_metric": "final_extracted",
        "temperature_retuning_permitted": False,
        "official_test_used": False,
    }
    return effects, decision


def main() -> None:
    global RUN_ID, EXECUTION_COMMIT, SOURCE_JOB, CONFIG_SHA256, TAG_PREFIX, LOG_STEM
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "lm_study/experiments_qwen3_17b_l2r_temperature_mixture.yaml",
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--expected-commit", default=EXECUTION_COMMIT)
    parser.add_argument("--source-job", default=SOURCE_JOB)
    parser.add_argument("--config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--tag-prefix", default=TAG_PREFIX)
    parser.add_argument("--log-stem", default=LOG_STEM)
    parser.add_argument("--validate-design-only", action="store_true")
    args = parser.parse_args()
    RUN_ID = args.run_id
    EXECUTION_COMMIT = args.expected_commit
    SOURCE_JOB = args.source_job
    CONFIG_SHA256 = args.config_sha256
    TAG_PREFIX = args.tag_prefix
    LOG_STEM = args.log_stem
    load_design(args.config)
    if args.validate_design_only:
        print(json.dumps({"run_id": RUN_ID, "cells": len(CELLS), "seeds": len(SEEDS)}))
        return
    required = {
        "artifact_dir": args.artifact_dir,
        "marker": args.marker,
        "log_dir": args.log_dir,
        "out_dir": args.out_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing result-analysis arguments: {', '.join(missing)}")
    verify_marker(args.marker)
    verify_logs(args.log_dir)
    table, mechanism = load_results(args.artifact_dir)
    effects, decision = analyse(table)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.sort_values(["cell", "seed"]).to_csv(args.out_dir / "seed_summary.csv", index=False)
    (args.out_dir / "paired_effects.json").write_text(
        json.dumps(effects, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "mechanism_summary.json").write_text(
        json.dumps(mechanism, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
