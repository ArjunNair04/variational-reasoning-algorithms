"""Frozen paired analysis of positional Q5 rationale-prior weighting."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lm_study"))
from generate_qwen3_17b_q5_prior_segments import (
    CELL_ORDER, SEEDS, CHECKPOINTS, CONTROL_CELL, CONTROL_COMMIT,
    CONTROL_RUN_ID, RUN_ID, SETTINGS, build_payload, study_for_payload,
)
import generate_qwen3_17b_q5_prior_segments as moving_study
from run_yaml import _prepare_cells
from ac_alg1_prior_segments import SEGMENT_SCOPE
from analyze_qwen3_q5_prior_exponent import load_cell, verify_controls, METRICS
from analyze_qwen3_jepo_comparator import (
    _artifact, _read_json, _read_jsonl_gz, _normalized_auc,
)


def validate_design(path):
    payload = yaml.safe_load(path.read_text())
    study = study_for_payload(payload)
    return payload, _prepare_cells(payload, only=None, run_id=study.RUN_ID, defaults=payload["defaults"])


def mechanism_rows(diagnostics, head_exponent, tail_exponent, expected_reader=None):
    if tuple(r["completed_rounds"] for r in diagnostics) != tuple(range(1, 33)):
        raise ValueError("32 complete diagnostic rounds required")
    output = []
    for row in diagnostics:
        resp = row["responsibilities"]
        if expected_reader is not None and (
            resp.get("answer_policy") != expected_reader or resp.get("policy") != "current"
        ):
            raise ValueError("logged answer reader or rationale policy mismatch")
        if (resp.get("prior_head_exponent"), resp.get("prior_tail_exponent")) != (head_exponent, tail_exponent):
            raise ValueError("logged segment exponent mismatch")
        if resp.get("prior_exponent") != 1.0:
            raise ValueError("global prior exponent must remain one")
        if resp.get("prior_segment_scope") != SEGMENT_SCOPE:
            raise ValueError("reasoning-only split with fixed marker required")
        groups = {}
        for trace in resp["traces"]:
            groups.setdefault((trace["partition"], trace["pid"]), []).append(trace)
        if not groups:
            raise ValueError("missing trace factor diagnostics")
        for (_, pid), group in groups.items():
            arrays = {key: np.array([t[key] for t in group], dtype=float) for key in (
                "prior_head_logprob", "prior_tail_logprob", "trace_logprob", "answer_logprob",
                "responsibility_logit", "prior_head_tokens", "prior_tail_tokens", "responsibility",
                "prior_marker_logprob", "prior_marker_tokens", "reasoning_token_count",
            )}
            if not all(np.isfinite(v).all() for v in arrays.values()):
                raise ValueError("nonfinite segment E-step factors")
            h, t = arrays["prior_head_logprob"], arrays["prior_tail_logprob"]
            prior, answer, actual = arrays["trace_logprob"], arrays["answer_logprob"], arrays["responsibility_logit"]
            hn, tn = arrays["prior_head_tokens"], arrays["prior_tail_tokens"]
            marker, mn = arrays["prior_marker_logprob"], arrays["prior_marker_tokens"]
            lengths = arrays["reasoning_token_count"]
            prior_counts = np.array([r["objective_tokens"] - r["answer_tokens"] for r in group])
            if (np.any(lengths < 0) or np.any(mn <= 0)
                    or not np.array_equal(lengths, np.floor(lengths))
                    or not np.array_equal(mn, np.floor(mn))
                    or not np.array_equal(lengths + mn, prior_counts)):
                raise ValueError("reasoning and fixed marker counts do not partition prior")
            if not (np.array_equal(hn, lengths // 2) and np.array_equal(tn, lengths - lengths // 2)):
                raise ValueError("segment token counts do not partition rationale")
            if not np.allclose(h + t + marker, prior, atol=2e-3, rtol=2e-6):
                raise ValueError("segment factor sum does not match rationale prior")
            if not np.allclose(actual, answer + marker + head_exponent*h + tail_exponent*t, atol=2e-3, rtol=2e-6):
                raise ValueError("E-step does not match registered segment exponents")
            weights = np.exp(actual - actual.max())
            weights /= weights.sum()
            if not np.allclose(weights, arrays["responsibility"], atol=2e-6, rtol=2e-5):
                raise ValueError("logged responsibilities do not match segment logits")
            corr = None
            if len(group) > 1 and np.std(weights) > 0 and np.std(lengths) > 0:
                ranks_w = pd.Series(weights).rank(method="average").to_numpy()
                ranks_n = pd.Series(lengths).rank(method="average").to_numpy()
                corr = float(np.corrcoef(ranks_w, ranks_n)[0, 1])
            output.append(dict(round=row["completed_rounds"], pid=pid, support=len(group),
                max_weight=float(weights.max()), ess=float(1/np.square(weights).sum()),
                weight_length_spearman=corr,
                top_changed_vs_joint=bool(np.argmax(actual) != np.argmax(prior + answer)),
                top_changed_vs_uniform075=bool(np.argmax(actual) != np.argmax(.75*(h+t) + marker + answer)),
                mean_head_tokens=float(hn.mean()), mean_tail_tokens=float(tn.mean()),
                mean_head_logprob=float(h.mean()), mean_tail_logprob=float(t.mean()),
                mean_marker_tokens=float(mn.mean()), mean_marker_logprob=float(marker.mean()),
                odd_length_fraction=float(np.mean(lengths % 2)),
                mean_nominal_exponent=(float(np.mean((
                    head_exponent*hn[lengths > 0] + tail_exponent*tn[lengths > 0]
                ) / lengths[lengths > 0])) if np.any(lengths > 0) else None),
            ))
    return output


def paired_contrasts(frame, study=moving_study, include_readers=False):
    indexed = frame.set_index(["cell", "seed"])
    order = study.CELL_ORDER
    pairs = [(c, study.CONTROL_CELL) for c in order]
    pairs += [(order[0], order[1]), (order[0], order[2]), (order[1], order[2])]
    if include_readers:
        pairs += list(zip(order, moving_study.CELL_ORDER, strict=True))
        pairs += [(study.CONTROL_CELL, moving_study.CONTROL_CELL)]
    samples = np.random.default_rng(20260914).integers(0, len(SEEDS), size=(20000, len(SEEDS)))
    out = []
    for treatment, control in pairs:
        for metric in METRICS:
            delta = np.array([indexed.loc[(treatment, s), metric] - indexed.loc[(control, s), metric] for s in SEEDS])
            lo, hi = np.quantile(delta[samples].mean(axis=1), [.025, .975])
            out.append(dict(treatment=treatment, control=control, metric=metric,
                mean_difference_pp=100*delta.mean(), descriptive_low_pp=100*lo,
                descriptive_high_pp=100*hi, per_seed_difference_pp=json.dumps((100*delta).tolist())))
    return pd.DataFrame(out)


def nominate(frame, study=moving_study):
    means = frame.groupby("cell")[list(METRICS)].mean()
    control = means.loc[study.CONTROL_CELL]
    eligible = [c for c in study.CELL_ORDER if means.loc[c, "final_extracted"] > control["final_extracted"]
                and means.loc[c, "final_strict"] >= control["final_strict"]]
    if not eligible:
        return None
    return sorted(eligible, key=lambda c: (-means.loc[c, "final_extracted"],
        -means.loc[c, "final_strict"], -means.loc[c, "extracted_auc"], c))[0]


def verify_moving_marker(path, study):
    marker = _read_json(path)
    expected = dict(status="ok", run_id=moving_study.RUN_ID,
        execution_commit=study.MOVING_COMMIT, source_job_id=study.MOVING_JOB,
        configuration_sha256=study.MOVING_CONFIG_SHA256, task_count=9,
        trained_adapter_count=9, official_test_used=False)
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"moving comparison marker mismatch: {key}")
    return marker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-design-only", action="store_true")
    parser.add_argument("--validate-controls-only", action="store_true")
    for name in ("artifact-dir", "control-dir", "marker", "control-marker", "output-dir", "moving-dir", "moving-marker"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-job")
    args = parser.parse_args()
    payload, new_cells = validate_design(args.config)
    study = study_for_payload(payload)
    frozen = payload["diagnostic"]["fixed_contract"]["reader"] == "frozen_base"
    if args.validate_design_only:
        print(f"{study.RUN_ID}: three cells x three seeds; historical {study.CONTROL_CELL} control")
        return
    if args.validate_controls_only:
        verify_controls(args.control_dir, args.control_marker)
        print("verified 3 seeds x base/moving-Q5/frozen-Q5 receipt identities and hashes")
        return
    if any(getattr(args, k) is None for k in ("artifact_dir", "control_dir", "marker", "control_marker", "output_dir", "expected_commit", "expected_source_job")):
        parser.error("result paths, both markers and execution identity required")
    if frozen and (args.moving_dir is None or args.moving_marker is None):
        parser.error("frozen-reader analysis requires --moving-dir and --moving-marker")
    moving_marker = verify_moving_marker(args.moving_marker, study) if frozen else None
    marker = _read_json(args.marker)
    for key, value in dict(status="ok", run_id=study.RUN_ID, execution_commit=args.expected_commit,
            source_job_id=args.expected_source_job, task_count=9,
            trained_adapter_count=9, official_test_used=False,
            configuration_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest()).items():
        if marker.get(key) != value:
            raise ValueError(f"marker mismatch: {key}")
    controls = verify_controls(args.control_dir, args.control_marker)
    metrics, mechanisms = [], []
    support = None
    for seed in SEEDS:
        _, base_result, base, base_ids = load_cell(args.control_dir, controls["CTRL-base"], seed, CONTROL_RUN_ID, CONTROL_COMMIT)
        metrics.append(dict(cell="CTRL-base", seed=seed, **base,
            extracted_auc=base["final_extracted"], strict_auc=base["final_strict"],
            train_llm_gen=0, optimizer_steps=0, accelerator_hours=base_result["accelerator_hours"]))
        if support is None:
            support = base_ids
        if support != base_ids:
            raise ValueError("baseline validation support changed")
        control_ids = [study.CONTROL_CELL] + ([moving_study.CONTROL_CELL] if frozen else [])
        selected = [(c, controls[c], args.control_dir, CONTROL_RUN_ID, CONTROL_COMMIT) for c in control_ids]
        selected += [(c, p, args.artifact_dir, study.RUN_ID, args.expected_commit)
                     for c, p in zip(study.CELL_ORDER, new_cells, strict=True)]
        if frozen:
            mp = moving_study.build_payload()
            moving_cells = _prepare_cells(mp, only=None, run_id=moving_study.RUN_ID, defaults=mp["defaults"])
            selected += [(c, p, args.moving_dir, moving_study.RUN_ID, study.MOVING_COMMIT)
                         for c, p in zip(moving_study.CELL_ORDER, moving_cells, strict=True)]
        for cell_id, cell, root, run_id, commit in selected:
            receipt, result, values, ids = load_cell(root, cell, seed, run_id, commit)
            if ids != support:
                raise ValueError("validation support mismatch")
            checkpoints = _read_jsonl_gz(_artifact(root, receipt, "checkpoint_eval_"))
            if tuple(r["completed_rounds"] for r in checkpoints) != CHECKPOINTS:
                raise ValueError("checkpoint schedule mismatch")
            for metric, key, baseline in (("extracted_auc", "test_acc_legacy", "final_extracted"), ("strict_auc", "test_acc_strict", "final_strict")):
                vals = [r["metrics"][key] for r in checkpoints]
                if not np.isfinite(vals).all() or abs(vals[-1] - values[baseline]) > 1e-12:
                    raise ValueError("invalid checkpoint metric or endpoint")
                values[metric] = _normalized_auc(base[baseline], vals)
            if int(result["optimizer_steps"]) != 32 or int(result["train_llm_gen"]) != 2048:
                raise ValueError("training budget mismatch")
            metrics.append(dict(cell=cell_id, seed=seed, **values, train_llm_gen=result["train_llm_gen"],
                optimizer_steps=result["optimizer_steps"], accelerator_hours=result["accelerator_hours"]))
            if run_id in {study.RUN_ID, moving_study.RUN_ID}:
                ds = _read_jsonl_gz(_artifact(root, receipt, "training_diagnostics_"))
                backward = sum(int((step.get("support") or {}).get("backward_tokens") or 0)
                    for r in ds for step in (r.get("inner_m_step") or {}).get("steps", []))
                if backward <= 0:
                    raise ValueError("missing backward-token diagnostics")
                metrics[-1]["backward_tokens"] = backward
                settings = study.SETTINGS if run_id == study.RUN_ID else moving_study.SETTINGS
                reader = "frozen_base" if frozen and run_id == study.RUN_ID else "current"
                mechanisms += [dict(cell=cell_id, seed=seed, **r)
                               for r in mechanism_rows(ds, *settings[cell_id], expected_reader=reader)]
    frame = pd.DataFrame(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    frame.groupby("cell", sort=False)[list(METRICS)].mean().to_csv(args.output_dir / "method_summary.csv")
    paired_contrasts(frame, study, include_readers=frozen).to_csv(args.output_dir / "paired_contrasts.csv", index=False)
    pd.DataFrame(mechanisms).to_csv(args.output_dir / "posterior_diagnostics.csv.gz", index=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(dict(run_id=study.RUN_ID, marker=marker,
        moving_comparison_marker=moving_marker,
        nominee=nominate(frame, study), automatic_followup=False,
        evidence="three-seed development screen; historical controls; descriptive intervals", metric_order=METRICS), indent=2))


if __name__ == "__main__":
    main()
