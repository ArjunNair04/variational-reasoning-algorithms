"""Frozen paired analysis and validation for the six-task M-step length screen."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from generate_qwen3_17b_mstep_length_normalisation import (
    CELL_ORDER, CHECKPOINTS, CONTROL_CELLS, CONTROL_COMMIT, CONTROL_RUN_ID,
    RUN_ID, SEEDS, build_payload,
)
from generate_qwen3_17b_selected_method_posterity import build_payload as control_payload
from analyze_qwen3_q5_prior_exponent import METRICS, load_cell
from analyze_qwen3_q5_learned_posterior import validate_log
from analyze_qwen3_jepo_comparator import (
    _artifact, _finite, _normalized_auc, _read_json, _read_jsonl_gz, _require_non_access,
)
from result_contract import atomic_write_json
from run_yaml import _prepare_cells
from validate_yaml_run import validate as validate_outputs


def validate_design(path):
    payload = yaml.safe_load(path.read_text())
    if payload != build_payload():
        raise ValueError("frozen M-step length design changed")
    return payload, _prepare_cells(payload, only=None, run_id=RUN_ID, defaults=payload["defaults"])


def mechanism_rows(diagnostics, method, mode):
    if method not in ("Q5", "PIS") or mode not in ("joint", "joint_token_mean"):
        raise ValueError("unknown method or M-step objective")
    if [r["completed_rounds"] for r in diagnostics] != list(range(1, 33)):
        raise ValueError("32 complete diagnostic rounds required")
    estimator, refresh, steps = (("delta_joint", "inner_step", 1) if method == "Q5"
                                 else ("prior_importance", "outer_round", 4))
    output = []
    for row in diagnostics:
        resp = row["responsibilities"]
        if (row["objective"]["latent_mstep_objective"] != mode
                or row["answer_target_termination"] != "eos"
                or row["answer_event_mode"] != "strict_terminal_marker"
                or resp["score"] != "joint" or resp["variational_estimator"] != estimator
                or resp["refresh"] != refresh or resp["temperature"] != 1.
                or resp["ess_floor_fraction"] != 0 or resp["policy"] != "current"
                or resp["answer_policy"] != "current"):
            raise ValueError("posterior or M-step contract changed")
        groups = {}
        for trace in resp["traces"]:
            groups.setdefault(trace["pid"], []).append(trace)
        if not groups:
            raise ValueError("missing retained support")
        ess, maximum, length = [], [], []
        for group in groups.values():
            weights = np.array([t["responsibility"] for t in group], dtype=float)
            logits = np.array([t["responsibility_logit"] for t in group], dtype=float)
            score = np.array([t["joint_logprob"] if method == "Q5" else t["answer_logprob"] for t in group], dtype=float)
            if not np.isfinite(np.concatenate((weights, logits, score))).all():
                raise ValueError("nonfinite posterior diagnostics")
            expected = np.exp(score - score.max()); expected /= expected.sum()
            if not np.allclose(logits, score, rtol=0, atol=2e-6) or not np.allclose(weights, expected, rtol=2e-5, atol=2e-6):
                raise ValueError("posterior weights were changed or do not reconstruct")
            ess.append(1 / np.square(weights).sum())
            maximum.append(weights.max())
            length.append(sum(w * t["objective_tokens"] for w, t in zip(weights, group)))
        inner = row["inner_m_step"]["steps"]
        if len(inner) != steps or any(s["status"] != "accepted" for s in inner):
            raise ValueError("M-step schedule changed")
        for step in inner:
            support = step["support"]
            if support["backward_eos_tokens"] != support["active_traces"] or support["backward_tokens"] <= 0:
                raise ValueError("missing EOS or backward support")
        output.append({
            "round": row["completed_rounds"], "ess": float(np.mean(ess)),
            "max_weight": float(np.mean(maximum)), "weighted_target_tokens": float(np.mean(length)),
            "generated_tokens": int(row["compute"]["tokens"]["generated"]),
            "backward_tokens": sum(s["support"]["backward_tokens"] for s in inner),
            "gradient_norm": float(np.mean([_finite(s["gradient"]["raw_objective_l2_norm"], context="gradient") for s in inner])),
            "update_norm": float(np.mean([_finite(s["update"]["applied_parameter_l2_norm"], context="update") for s in inner])),
        })
    return output


def contrasts(frame):
    required = {(cell, seed) for cell in (*CELL_ORDER, *CONTROL_CELLS.values()) for seed in SEEDS}
    if len(frame) != len(required) or set(zip(frame.cell, frame.seed)) != required:
        raise ValueError("twelve unique paired coordinates required")
    indexed = frame.set_index(["cell", "seed"])
    output = []
    for treatment, control in CONTROL_CELLS.items():
        for metric in METRICS:
            delta = np.array([indexed.loc[(treatment, seed), metric] - indexed.loc[(control, seed), metric] for seed in SEEDS])
            if not np.isfinite(delta).all():
                raise ValueError("nonfinite paired metric")
            margin = 4.302652729911275 * delta.std(ddof=1) / np.sqrt(3)
            output.append(dict(treatment=treatment, control=control, metric=metric,
                difference_pp=100 * delta.mean(), descriptive_low_pp=100 * (delta.mean()-margin),
                descriptive_high_pp=100 * (delta.mean()+margin), per_seed_pp=json.dumps((100*delta).tolist())))
    return pd.DataFrame(output)


def validate_results(args):
    payload, cells = validate_design(args.config)
    digest = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if (digest != args.expected_config_sha256 or not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit)
            or not args.source_job_id.isdigit()):
        raise ValueError("immutable execution identity missing or changed")
    control_marker = _read_json(args.control_marker)
    if any(control_marker.get(k) != v for k, v in dict(status="ok", run_id=CONTROL_RUN_ID,
           execution_commit=CONTROL_COMMIT, task_count=91).items()):
        raise ValueError("historical control validator marker mismatch")
    for task in range(1, 7):
        validate_log(args.logs / f"qwen3_mstep_length.{args.source_job_id}.{task}.log")
    problems = validate_outputs(args.config, str(args.logs / f"qwen3_mstep_length.{args.source_job_id}.*.log"))
    if problems:
        raise ValueError("; ".join(problems))
    root = Path(payload["defaults"]["out"]).expanduser()
    if len(list(root.glob("complete_*.json"))) != 6:
        raise ValueError("six new completion receipts required")
    historical = control_payload()
    old = _prepare_cells(historical, only=None, run_id=CONTROL_RUN_ID, defaults=historical["defaults"])
    control_map = {cell.tag.split(f"{CONTROL_RUN_ID}_", 1)[1].removeprefix(f"{cell.method}_"): cell for cell in old}
    metrics, mechanisms = [], []
    for seed in SEEDS:
        _, _, baseline, base_ids = load_cell(args.control_dir, control_map["CTRL-base"], seed, CONTROL_RUN_ID, CONTROL_COMMIT)
        for cell_id, cell in zip(CELL_ORDER, cells):
            schedules, environments = [], []
            method = cell_id.split("-", 1)[0]
            control_id = CONTROL_CELLS[cell_id]
            for name, prepared, source, rid, revision, mode in (
                (control_id, control_map[control_id], args.control_dir, CONTROL_RUN_ID, CONTROL_COMMIT, "joint"),
                (cell_id, cell, root, RUN_ID, args.expected_commit, "joint_token_mean"),
            ):
                receipt, result, values, ids = load_cell(source, prepared, seed, rid, revision)
                _require_non_access(result, context="cell result")
                if ids != base_ids:
                    raise ValueError("paired validation questions differ")
                checkpoints = _read_jsonl_gz(_artifact(source, receipt, "checkpoint_eval_"))
                if tuple(c["completed_rounds"] for c in checkpoints) != CHECKPOINTS:
                    raise ValueError("checkpoint schedule changed")
                for checkpoint in checkpoints:
                    _require_non_access(checkpoint["metrics"], context="checkpoint")
                for metric, key, endpoint in (("extracted_auc", "test_acc_legacy", "final_extracted"), ("strict_auc", "test_acc_strict", "final_strict")):
                    vals = [_finite(c["metrics"][key], context=key) for c in checkpoints]
                    if any(not 0 <= v <= 1 for v in vals) or abs(vals[-1]-values[endpoint]) > 1e-12:
                        raise ValueError("invalid checkpoint metric or endpoint")
                    values[metric] = _normalized_auc(baseline[endpoint], vals)
                diagnostics = _read_jsonl_gz(_artifact(source, receipt, "training_diagnostics_"))
                rows = mechanism_rows(diagnostics, method, mode)
                schedules.append([r["minibatch"] for r in diagnostics])
                env = json.loads(result["params"])["env"]
                environments.append({key: env[key] for key in ("python", "torch", "transformers", "peft", "cuda_runtime", "cudnn", "gpu")})
                updates = 32 if method == "Q5" else 128
                if int(result["optimizer_steps"]) != updates or int(result["train_llm_gen"]) != 2048:
                    raise ValueError("training budget changed")
                metrics.append(dict(cell=name, seed=seed, **values, optimizer_steps=updates,
                    train_llm_gen=2048, accelerator_hours=_finite(result["accelerator_hours"], context="compute"),
                    generated_tokens=sum(r["generated_tokens"] for r in rows), backward_tokens=sum(r["backward_tokens"] for r in rows)))
                mechanisms.extend(dict(cell=name, seed=seed, **row) for row in rows)
            if schedules[0] != schedules[1] or environments[0] != environments[1]:
                raise ValueError("paired schedule or runtime differs; investigate before comparison")
    frame = pd.DataFrame(metrics)
    marker = dict(status="ok", run_id=RUN_ID, task_count=6, execution_commit=args.expected_commit,
                  configuration_sha256=digest, source_job_id=args.source_job_id,
                  official_test_used=False, validated_at=datetime.now(timezone.utc).isoformat())
    return marker, frame, pd.DataFrame(mechanisms), contrasts(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-design-only", action="store_true")
    for name in ("logs", "control-dir", "control-marker", "output-dir", "marker"):
        parser.add_argument("--"+name, type=Path)
    for name in ("expected-commit", "expected-config-sha256", "source-job-id"):
        parser.add_argument("--"+name)
    args = parser.parse_args()
    validate_design(args.config)
    if args.validate_design_only:
        print(f"{RUN_ID}: Q5/PIS token-mean M-step; six tasks; no control reruns")
        return
    if any(getattr(args, key) is None for key in ("logs", "control_dir", "control_marker", "output_dir", "marker", "expected_commit", "expected_config_sha256", "source_job_id")):
        parser.error("logs, control paths and immutable execution identity are required")
    marker, frame, mechanisms, paired = validate_results(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    frame.groupby("cell", sort=False)[list(METRICS)].mean().to_csv(args.output_dir / "method_summary.csv")
    paired.to_csv(args.output_dir / "paired_contrasts.csv", index=False)
    mechanisms.to_csv(args.output_dir / "mechanisms.csv", index=False)
    atomic_write_json(args.output_dir / "analysis.json", dict(marker=marker, metric_order=METRICS,
        automatic_followup=False, evidence="Three-seed development screen with historical controls; per-trace and overall loss scaling both change."))
    atomic_write_json(args.marker, marker)
    print(json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
