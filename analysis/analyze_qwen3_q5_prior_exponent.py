"""Receipt-bound, paired analysis of Q5 prior exponents and answer readers."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lm_study"))
from generate_qwen3_17b_q5_prior_exponent import (
    CELL_ORDER, SEEDS, CHECKPOINTS, CONTROL_CELLS, CONTROL_COMMIT,
    CONTROL_RUN_ID, RUN_ID, build_payload,
)
from generate_qwen3_17b_selected_method_posterity import build_payload as control_payload
from run_yaml import _prepare_cells
from result_contract import validate_completion_receipt, validate_receipt_identity
from analyze_qwen3_jepo_comparator import (
    _artifact, _read_json, _read_jsonl_gz, _validate_evaluation, _normalized_auc,
)

METRICS = ("final_extracted", "final_strict", "extracted_auc", "strict_auc")


def validate_design(path):
    payload = yaml.safe_load(path.read_text())
    if payload != build_payload():
        raise ValueError("frozen design changed")
    return payload, _prepare_cells(payload, only=None, run_id=RUN_ID, defaults=payload["defaults"])


def load_cell(root, cell, seed, run_id, commit):
    tag = f"{cell.tag}_seed{seed}"
    receipt = validate_completion_receipt(root / f"complete_gsm8k__{tag}__{cell.method}_s{seed}.json", result_root=root)
    validate_receipt_identity(receipt, dict(run_id=run_id, seed=seed, method=cell.method,
        task="gsm8k", model=cell.model, tag=tag))
    final, strict, support = _validate_evaluation(_artifact(root, receipt, "eval_"))
    result = _read_json(_artifact(root, receipt, "cell_result_"))["result"]
    params = json.loads(result["params"])
    observed_commit = params["env"]["commit"]
    if len(observed_commit) < 7 or not commit.startswith(observed_commit):
        raise ValueError("execution commit mismatch")
    return receipt, result, dict(final_extracted=final, final_strict=strict), support


def mechanism_rows(diagnostics, tau):
    if tuple(r["completed_rounds"] for r in diagnostics) != tuple(range(1, 33)):
        raise ValueError("32 complete diagnostic rounds required")
    output = []
    for row in diagnostics:
        resp = row["responsibilities"]
        if resp.get("prior_exponent") != tau:
            raise ValueError("logged prior exponent mismatch")
        traces = resp["traces"]
        groups = {}
        for trace in traces:
            groups.setdefault((trace["partition"], trace["pid"]), []).append(trace)
        if not groups:
            raise ValueError("missing trace factor diagnostics")
        for (_, pid), group in groups.items():
            prior = np.array([t["trace_logprob"] for t in group], dtype=float)
            answer = np.array([t["answer_logprob"] for t in group], dtype=float)
            actual = np.array([t["responsibility_logit"] for t in group], dtype=float)
            if not np.isfinite(np.concatenate([prior, answer, actual])).all():
                raise ValueError("nonfinite E-step factors")
            expected = answer + tau * prior
            if not np.allclose(actual, expected, atol=2e-3, rtol=2e-6):
                raise ValueError("E-step does not match registered exponent")
            weights = np.exp(actual - actual.max()); weights /= weights.sum()
            lengths = np.array([t["objective_tokens"] - t["answer_tokens"] for t in group])
            corr = None
            if len(group) > 1 and np.std(weights) > 0 and np.std(lengths) > 0:
                # Spearman is Pearson correlation of average ranks, including ties.
                # Avoid pandas' optional SciPy import on the cluster runtime.
                weight_ranks = pd.Series(weights).rank(method="average").to_numpy()
                length_ranks = pd.Series(lengths).rank(method="average").to_numpy()
                corr = float(np.corrcoef(weight_ranks, length_ranks)[0, 1])
            output.append(dict(round=row["completed_rounds"], pid=pid, support=len(group),
                max_weight=float(weights.max()), ess=float(1 / np.square(weights).sum()),
                weight_length_spearman=corr,
                top_trace_changed=bool(np.argmax(actual) != np.argmax(prior + answer))))
    return output


def paired_contrasts(frame):
    indexed = frame.set_index(["cell", "seed"])
    pairs = [(c, CONTROL_CELLS[c.split("-")[2]]) for c in CELL_ORDER]
    pairs += [(CELL_ORDER[2], CELL_ORDER[0]), (CELL_ORDER[3], CELL_ORDER[1])]
    samples = np.random.default_rng(20260912).integers(0, 3, size=(20000, 3))
    out = []
    for treatment, control in pairs:
        for metric in METRICS:
            delta = np.array([indexed.loc[(treatment, s), metric] - indexed.loc[(control, s), metric] for s in SEEDS])
            lo, hi = np.quantile(delta[samples].mean(axis=1), [.025, .975])
            out.append(dict(treatment=treatment, control=control, metric=metric,
                mean_difference_pp=100*delta.mean(), descriptive_low_pp=100*lo,
                descriptive_high_pp=100*hi, per_seed_difference_pp=json.dumps((100*delta).tolist())))
    return pd.DataFrame(out)


def verify_controls(root, marker_path):
    marker = _read_json(marker_path)
    for key, value in dict(status="ok", run_id=CONTROL_RUN_ID, execution_commit=CONTROL_COMMIT, task_count=91).items():
        if marker.get(key) != value:
            raise ValueError(f"control marker mismatch: {key}")
    cp = control_payload()
    prepared = _prepare_cells(cp, only=None, run_id=CONTROL_RUN_ID, defaults=cp["defaults"])
    ids = [v["cell_id"] for raw in cp["algos"].values() for v in (raw if isinstance(raw, list) else [raw])]
    controls = dict(zip(ids, prepared, strict=True))
    support = None
    for cell_id in ("CTRL-base", *CONTROL_CELLS.values()):
        for seed in SEEDS:
            _, _, _, ids = load_cell(root, controls[cell_id], seed, CONTROL_RUN_ID, CONTROL_COMMIT)
            if support is None:
                support = ids
            if support != ids:
                raise ValueError("historical control validation support mismatch")
    return controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-design-only", action="store_true")
    parser.add_argument("--validate-controls-only", action="store_true")
    for name in ("artifact-dir", "control-dir", "marker", "control-marker", "output-dir"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-job")
    args = parser.parse_args()
    _, new_cells = validate_design(args.config)
    if args.validate_design_only:
        print(f"{RUN_ID}: four cells x three seeds; no tau=1 reruns")
        return
    if args.validate_controls_only:
        verify_controls(args.control_dir, args.control_marker)
        print("verified 3 seeds x base/moving-Q5/frozen-Q5 receipt identities and hashes")
        return
    if any(getattr(args, k) is None for k in ("artifact_dir", "control_dir", "marker", "control_marker", "output_dir", "expected_commit", "expected_source_job")):
        parser.error("result paths, both markers and execution identity required")
    marker = _read_json(args.marker)
    for key, value in dict(status="ok", run_id=RUN_ID, execution_commit=args.expected_commit,
            source_job_id=args.expected_source_job, task_count=12,
            configuration_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest()).items():
        if marker.get(key) != value:
            raise ValueError(f"marker mismatch: {key}")
    controls = verify_controls(args.control_dir, args.control_marker)
    metrics, mechanisms = [], []
    support = None
    for seed in SEEDS:
        _, _, base, base_ids = load_cell(args.control_dir, controls["CTRL-base"], seed, CONTROL_RUN_ID, CONTROL_COMMIT)
        if support is None:
            support = base_ids
        if support != base_ids:
            raise ValueError("baseline validation support changed")
        selected = [(c, controls[c], args.control_dir, CONTROL_RUN_ID, CONTROL_COMMIT) for c in CONTROL_CELLS.values()]
        selected += [(c, p, args.artifact_dir, RUN_ID, args.expected_commit) for c, p in zip(CELL_ORDER, new_cells, strict=True)]
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
            if run_id == RUN_ID:
                ds = _read_jsonl_gz(_artifact(root, receipt, "training_diagnostics_"))
                backward = sum(int((step.get("support") or {}).get("backward_tokens") or 0)
                    for r in ds for step in (r.get("inner_m_step") or {}).get("steps", []))
                if backward <= 0:
                    raise ValueError("missing backward-token diagnostics")
                metrics[-1]["backward_tokens"] = backward
                mechanisms += [dict(cell=cell_id, seed=seed, **r) for r in mechanism_rows(ds, cell.axes["responsibility_prior_exponent"])]
    frame = pd.DataFrame(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    frame.groupby("cell", sort=False)[list(METRICS)].mean().to_csv(args.output_dir / "method_summary.csv")
    paired_contrasts(frame).to_csv(args.output_dir / "paired_contrasts.csv", index=False)
    pd.DataFrame(mechanisms).to_csv(args.output_dir / "posterior_diagnostics.csv.gz", index=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(dict(run_id=RUN_ID, marker=marker,
        evidence="three-seed development screen; historical controls; descriptive intervals", metric_order=METRICS), indent=2))


if __name__ == "__main__":
    main()
