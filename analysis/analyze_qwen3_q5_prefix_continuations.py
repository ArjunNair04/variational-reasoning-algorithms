"""Pre-outcome validation and paired analysis of Q5 prefix continuations."""

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lm_study"))
from generate_qwen3_17b_q5_prefix_continuations import (
    BASELINES, CELL_ORDER, CHECKPOINTS, RUN_ID, SEEDS, SETTINGS, SUPPORT_SHA256, build_payload,
)
from analyze_qwen3_q5_prior_exponent import load_cell, mechanism_rows as posterior_rows, METRICS
from analyze_qwen3_jepo_comparator import (
    _artifact, _read_jsonl_gz, _normalized_auc, _require_non_access, _finite,
)
from result_contract import atomic_write_json
from run_yaml import _prepare_cells
from validate_yaml_run import validate as validate_yaml_outputs


def validate_design(path):
    payload = yaml.safe_load(path.read_text())
    if payload != build_payload():
        raise ValueError("frozen prefix-continuation design changed")
    cells = _prepare_cells(payload, only=None, run_id=RUN_ID, defaults=payload["defaults"])
    if len(cells) != 2 or any(c.method != "AC-ALG1" for c in cells):
        raise ValueError("expected exactly two AC-ALG1 cells")
    return payload, cells


def continuation_rows(diagnostics, mode):
    if [r["completed_rounds"] for r in diagnostics] != list(range(1, 33)):
        raise ValueError("expected 32 complete diagnostic rounds")
    output = []
    for row in diagnostics:
        gen, compute = row["generation"], row["compute"]
        if (gen["this_round"] != 64 or gen["cumulative"] != 64*row["completed_rounds"]
                or gen["proposal_prompt"] != "answer_derive"
                or gen["proposal_policy"] != "current" or gen["proposal_temperature"] != 1.):
            raise ValueError("proposal contract or budget changed")
        resp = row["responsibilities"]
        if (resp["answer_policy"] != "current" or resp["policy"] != "current"
                or resp["score"] != "joint" or resp["ess_floor_fraction"] != 0.):
            raise ValueError("Q5 scoring changed")
        tokens = compute["tokens"]
        for key in ("generated", "backward", "backward_eos"):
            if not isinstance(tokens.get(key), int) or tokens[key] <= 0:
                raise ValueError(f"invalid {key} token accounting")
        intervention = gen.get("sampling_intervention")
        copied, fallbacks = 0, 0
        if mode == "independent":
            if intervention is not None:
                raise ValueError("control unexpectedly branched")
        elif mode == "prefix_half":
            if not isinstance(intervention, dict) or intervention.get("mode") != mode:
                raise ValueError("missing prefix-continuation diagnostics")
            arrays = [intervention[k] for k in ("parent_indices", "prefix_tokens", "generated_tokens",
                "completion_tokens", "prefix_sha256", "fallback_reasons", "trace_ids", "parent_trace_ids")]
            if any(len(a) != 64 for a in arrays) or len(set(intervention["trace_ids"])) != 64:
                raise ValueError("incomplete or duplicate lineage")
            parents, prefix, generated, completed, hashes, reasons, ids, parent_ids = arrays
            if intervention["root_count"] != 32:
                raise ValueError("expected 32 complete roots per round")
            for i in range(64):
                if any(not isinstance(v[i], int) or isinstance(v[i], bool) for v in (prefix, generated, completed)):
                    raise ValueError("token counts must be integers")
                if (prefix[i] < 0 or generated[i] <= 0 or generated[i]+prefix[i] != completed[i]
                        or completed[i] > intervention["max_new_tokens"]):
                    raise ValueError("copied/generated/total token identity failed")
                if i % 16 < 8:
                    if parents[i] is not None or parent_ids[i] is not None or prefix[i] or hashes[i] is not None or reasons[i] is not None:
                        raise ValueError("root incorrectly labelled as a child")
                else:
                    if parents[i] != i-8 or parent_ids[i] != ids[i-8]:
                        raise ValueError("child parent changed")
                    if not isinstance(hashes[i], str) or not re.fullmatch(r"[0-9a-f]{64}", hashes[i]):
                        raise ValueError("missing native-prefix hash")
                    if bool(prefix[i]) == (reasons[i] is not None):
                        raise ValueError("prefix/fallback contradiction")
            copied = sum(prefix)
            fallbacks = sum(reason is not None for reason in reasons)
            if (intervention["fallback_count"] != fallbacks
                    or intervention["continuation_count"] != 32-fallbacks
                    or sum(generated) != tokens["generated"]):
                raise ValueError("continuation aggregate mismatch")
        else:
            raise ValueError("unknown continuation mode")
        output.append(dict(round=row["completed_rounds"], generated_tokens=tokens["generated"],
            copied_prefix_tokens=copied, fallback_count=fallbacks, backward_tokens=tokens["backward"],
            duplicate_count=row["buffer_set_duplicates_this_round"],
            generation_seconds=_finite(compute["timings_seconds"]["generation"], context="generation timing"),
            mstep_seconds=_finite(compute["timings_seconds"]["m_step"], context="M-step timing")))
    return output


def paired_contrasts(frame):
    if set(zip(frame.cell, frame.seed)) != {(c,s) for c in CELL_ORDER for s in SEEDS} or len(frame) != 6:
        raise ValueError("paired analysis requires six unique coordinates")
    indexed = frame.set_index(["cell", "seed"])
    rows = []
    for metric in METRICS:
        delta = np.array([indexed.loc[(CELL_ORDER[1],s),metric] - indexed.loc[(CELL_ORDER[0],s),metric] for s in SEEDS])
        if not np.isfinite(delta).all():
            raise ValueError("nonfinite paired endpoint")
        # Student t, two degrees of freedom; intentionally wide with three seeds.
        margin = 4.302652729911275 * delta.std(ddof=1) / np.sqrt(3)
        rows.append(dict(metric=metric, mean_difference_pp=100*delta.mean(),
            descriptive_low_pp=100*(delta.mean()-margin), descriptive_high_pp=100*(delta.mean()+margin),
            per_seed_difference_pp=json.dumps((100*delta).tolist())))
    return pd.DataFrame(rows)


def validate_results(config_path, log_glob, commit, config_sha, job):
    payload, cells = validate_design(config_path)
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != config_sha:
        raise ValueError("configuration SHA mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not job.isdigit():
        raise ValueError("full execution commit and source job required")
    logs = sorted(Path(p) for p in glob.glob(str(Path(log_glob).expanduser())))
    if len(logs) != 6 or {p.name for p in logs} != {f"qwen3_q5_prefix.{job}.{i}.log" for i in range(1,7)}:
        raise ValueError("expected six exact payload logs")
    for path in logs:
        text = path.read_text(errors="replace")
        if "=== done in" not in text or re.search(r"Traceback|CUDA out of memory|OutOfMemoryError|No space left|Disk quota exceeded|FAILED|ERROR:", text):
            raise ValueError(f"failed or incomplete payload log: {path}")
    problems = validate_yaml_outputs(config_path, log_glob)
    if problems:
        raise ValueError("; ".join(problems))
    root = Path(payload["defaults"]["out"]).expanduser()
    if len(list(root.glob("complete_*.json"))) != 6:
        raise ValueError("expected six completion receipts")
    metrics, mechanisms, posteriors = [], [], []
    question_schedules = {}
    for cell_id, cell in zip(CELL_ORDER, cells, strict=True):
        for seed in SEEDS:
            receipt, result, values, support = load_cell(root, cell, seed, RUN_ID, commit)
            _require_non_access(result, context="cell result")
            support_hash = hashlib.sha256(json.dumps(support, separators=(",", ":")).encode()).hexdigest()
            if support_hash != SUPPORT_SHA256:
                raise ValueError("validation question support differs from registered baseline")
            checkpoints = _read_jsonl_gz(_artifact(root, receipt, "checkpoint_eval_"))
            if tuple(r["completed_rounds"] for r in checkpoints) != CHECKPOINTS:
                raise ValueError("checkpoint schedule changed")
            for row in checkpoints:
                _require_non_access(row["metrics"], context="checkpoint")
            for name, key, endpoint in (("extracted_auc", "test_acc_legacy", "final_extracted"), ("strict_auc", "test_acc_strict", "final_strict")):
                vals = [_finite(r["metrics"][key], context=key) for r in checkpoints]
                if any(not 0 <= v <= 1 for v in vals) or abs(vals[-1]-values[endpoint]) > 1e-12:
                    raise ValueError("invalid checkpoint metric or endpoint")
                values[name] = _normalized_auc(BASELINES[seed][endpoint], vals)
            ds = _read_jsonl_gz(_artifact(root, receipt, "training_diagnostics_"))
            for r in ds:
                if any(r.get(k) != v for k,v in dict(run_id=RUN_ID, model=cell.model,
                        seed=seed, method=cell.method, tag=f"{cell.tag}_seed{seed}").items()):
                    raise ValueError("diagnostic identity mismatch")
                if (r.get("answer_event_mode") != "strict_terminal_marker"
                        or r.get("answer_target_termination") != "eos"
                        or r.get("algorithm_profile") != "q5_prefix_continuations"):
                    raise ValueError("answer event or algorithm profile changed")
            schedule = [r["minibatch"] for r in ds]
            if seed in question_schedules and question_schedules[seed] != schedule:
                raise ValueError("paired training-question schedules differ")
            question_schedules[seed] = schedule
            mr = continuation_rows(ds, SETTINGS[cell_id])
            pr = posterior_rows(ds, 1.)
            if int(result["optimizer_steps"]) != 32 or int(result["train_llm_gen"]) != 2048:
                raise ValueError("training budget changed")
            metrics.append(dict(cell=cell_id, seed=seed, **values, train_llm_gen=2048, optimizer_steps=32,
                accelerator_hours=_finite(result["accelerator_hours"], context="compute"),
                generated_tokens=sum(r["generated_tokens"] for r in mr),
                copied_prefix_tokens=sum(r["copied_prefix_tokens"] for r in mr),
                backward_tokens=sum(r["backward_tokens"] for r in mr)))
            mechanisms.extend(dict(cell=cell_id, seed=seed, **r) for r in mr)
            posteriors.extend(dict(cell=cell_id, seed=seed, **r) for r in pr)
    frame = pd.DataFrame(metrics)
    contrasts = paired_contrasts(frame)
    marker = dict(schema_version=1, status="ok", run_id=RUN_ID, execution_commit=commit,
        configuration_sha256=config_sha, source_job_id=job, task_count=6, trained_adapter_count=6,
        official_test_used=False, validated_at=datetime.now(timezone.utc).isoformat())
    return marker, frame, pd.DataFrame(mechanisms), pd.DataFrame(posteriors), contrasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-design-only", action="store_true")
    for flag in ("log-glob", "expected-commit", "expected-config-sha256", "source-job-id"):
        parser.add_argument("--"+flag)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    validate_design(args.config)
    if args.validate_design_only:
        print(f"{RUN_ID}: two cells x three seeds; concurrent Q5 control")
        return
    if any(getattr(args, k) is None for k in ("log_glob", "expected_commit", "expected_config_sha256", "source_job_id", "marker", "output_dir")):
        parser.error("logs, immutable identity, marker and analysis output directory required")
    marker, frame, mechanism, posterior, contrasts = validate_results(args.config, args.log_glob,
        args.expected_commit, args.expected_config_sha256, args.source_job_id)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_dir/"seed_metrics.csv", index=False)
    means = frame.groupby("cell", sort=False).mean(numeric_only=True)
    means.drop(columns="seed").to_csv(args.output_dir/"method_summary.csv")
    mechanism.to_csv(args.output_dir/"generation_diagnostics.csv", index=False)
    posterior.to_csv(args.output_dir/"posterior_diagnostics.csv.gz", index=False)
    contrasts.to_csv(args.output_dir/"paired_contrasts.csv", index=False)
    delta = means.loc[CELL_ORDER[1]]-means.loc[CELL_ORDER[0]]
    nominee = CELL_ORDER[1] if delta.final_extracted > 0 and delta.final_strict >= 0 else None
    atomic_write_json(args.output_dir/"analysis.json", dict(marker=marker, nominee=nominee,
        automatic_followup=False, metric_order=METRICS, evidence="Three-seed development screen; descriptive paired intervals; historical round-zero baseline."))
    atomic_write_json(args.marker, marker)
    print(json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
