"""Read-only semantic and fixed-support audit; never infer missing token scores."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = (1201, 1213, 1217)
CELL = "Q5-AD-M-LR1e-5-U1-K16"
RUN_ID = "68078ecc"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_file(root, prefix, seed, suffix):
    paths = list(root.glob(f"{prefix}_*{RUN_ID}*{CELL}_seed{seed}__*{suffix}"))
    if len(paths) != 1:
        raise ValueError(f"expected one {prefix} source for seed {seed}, got {len(paths)}")
    return paths[0]


def receipt_evidence(path, receipt_path, seed):
    receipt = json.loads(receipt_path.read_text())
    identity = receipt["identity"]
    if (receipt.get("status") != "complete" or identity["run_id"] != RUN_ID
            or identity["seed"] != seed or CELL not in identity["tag"]):
        raise ValueError("source receipt identity mismatch")
    entries = [a for a in receipt["artifacts"] if a["path"] == path.name]
    if (len(entries) != 1 or entries[0]["sha256"] != sha256(path)
            or entries[0]["size"] != path.stat().st_size):
        raise ValueError("source file does not match its completion receipt")
    return {"source": str(path.resolve()), "sha256": sha256(path),
            "receipt": str(receipt_path.resolve()), "receipt_sha256": sha256(receipt_path)}


def rationale_lines(text):
    return [line.strip() for line in text.split("####", 1)[0].splitlines() if line.strip()]


def selected_samples(samples, seed):
    if len(samples) != 100 or len({s["idx"] for s in samples}) != 100:
        raise ValueError("expected exactly 100 distinct saved evaluation completions")
    ordered = sorted(samples, key=lambda s: (len(s["completion"].split("####", 1)[0].split()), s["idx"]))
    selected = []
    for start in range(0, 100, 25):
        block = sorted(ordered[start:start + 25], key=lambda s: hashlib.sha256(
            f"q5-semantic-v1:{seed}:{s['idx']}".encode()).hexdigest())[:2]
        selected.extend((start // 25 + 1, sample) for sample in block)
    return selected


def semantic_audit(root, annotations):
    rows, sources, expected_keys = [], [], set()
    labels = {(r["seed"], r["question_id"]): r for r in annotations["annotations"]}
    if len(labels) != len(annotations["annotations"]):
        raise ValueError("duplicate semantic annotation")
    for seed in SEEDS:
        path = source_file(root, "dump", seed, ".json")
        payload = json.loads(path.read_text())
        if (payload["run_id"] != RUN_ID or payload["seed"] != seed
                or payload.get("eval_official_test_accessed") is not False
                or payload.get("eval_dataset_splits_loaded") != ["train"]):
            raise ValueError("evaluation provenance or train-only boundary changed")
        sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
        for quartile, sample in selected_samples(payload["samples"], seed):
            key = seed, sample["idx"]
            expected_keys.add(key)
            label = labels[key]
            lines = rationale_lines(sample["completion"])
            roles = label["roles"]
            if len(lines) != len(roles) or any(not r or set(r) - set("IPCSVO") for r in roles):
                raise ValueError(f"semantic line alignment failed: {key}")
            first_c = next((i for i, r in enumerate(roles) if "C" in r), None)
            if first_c is None or any(set(r) - set("IP") for r in roles[:first_c]):
                raise ValueError(f"unclassified pre-calculation prefix: {key}")
            transition = label.get("within_line_transition")
            if transition and sample["completion"].count(transition) != 1:
                raise ValueError(f"ambiguous within-line transition: {key}")
            word_count = sum(len(line.split()) for line in lines)
            rows.append({
                "seed": seed, "question_id": sample["idx"], "length_quartile": quartile,
                "question": sample.get("question"), "completion": sample["completion"],
                "completion_sha256": hashlib.sha256(sample["completion"].encode()).hexdigest(),
                "lines": [{"text": line, "roles": role} for line, role in zip(lines, roles)],
                "pure_initial_lines": first_c,
                "pure_initial_word_fraction": sum(len(line.split()) for line in lines[:first_c]) / word_count,
                "initial_prefix_includes_interpretation": any("I" in r for r in roles[:first_c]),
                "interpretation_after_calculation": any("I" in r for r in roles[first_c + 1:]),
                "planning_after_calculation": any("P" in r for r in roles[first_c + 1:]),
                "mixed_first_line": "C" in roles[0] and bool(set(roles[0]) & set("IP")),
                "has_independent_check": any("V" in r for r in roles),
                "note": label["note"], "within_line_transition": transition,
            })
    if set(labels) != expected_keys:
        raise ValueError("annotations differ from outcome-blind sample")
    return {
        "scope": "24 length-stratified saved FINAL EVALUATION traces; not a representative estimate of all traces or training buffers",
        "reviewer": annotations["reviewer"], "selection": annotations["selection"],
        "roles": annotations["roles"], "conventions": annotations["conventions"],
        "count": len(rows),
        "initial_plan_or_interpretation_block": sum(r["pure_initial_lines"] > 0 for r in rows),
        "initial_block_with_interpretation": sum(r["initial_prefix_includes_interpretation"] for r in rows),
        "calculation_on_first_line": sum(r["pure_initial_lines"] == 0 for r in rows),
        "mixed_first_line": sum(r["mixed_first_line"] for r in rows),
        "interpretation_reappears_after_calculation": sum(r["interpretation_after_calculation"] for r in rows),
        "planning_reappears_after_calculation": sum(r["planning_after_calculation"] for r in rows),
        "single_line_with_internal_transition": sum(r["within_line_transition"] is not None for r in rows),
        "independent_verification": sum(r["has_independent_check"] for r in rows),
        "sources": sources, "records": rows,
    }


def softmax(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("nonempty finite logits required")
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranked


def rank_correlation(a, b):
    a, b = ranks(a), ranks(b)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def support_comparison(traces):
    if not traces or len({t["trace_id"] for t in traces}) != len(traces):
        raise ValueError("empty or duplicate trace support")
    p = np.array([t["trace_logprob"] for t in traces], dtype=float)
    a = np.array([t["answer_logprob"] for t in traces], dtype=float)
    joint = np.array([t["joint_logprob"] for t in traces], dtype=float)
    logged = np.array([t["responsibility_logit"] for t in traces], dtype=float)
    weights = np.array([t["responsibility"] for t in traces], dtype=float)
    if not all(np.isfinite(x).all() for x in (p, a, joint, logged, weights)):
        raise ValueError("nonfinite saved scores")
    if np.any(p > 1e-8) or np.any(a > 1e-8):
        raise ValueError("log probabilities must be nonpositive")
    if not np.allclose(joint, logged, atol=1e-8, rtol=0):
        raise ValueError("historical control is not joint weighting")
    original = softmax(joint)
    if not np.allclose(weights, original, atol=2e-6, rtol=0):
        raise ValueError("recorded weights do not reconstruct from recorded logits")
    # Joint and factor sums can differ due to BF16 subtraction in the logger.
    # Report that discrepancy and compare uniform weighting with BOTH baselines.
    factor_joint = a + p
    factor_original = softmax(factor_joint)
    uniform_score = a + 0.75 * p
    uniform = softmax(uniform_score)
    old_top = int(np.argmax(joint))
    new_top = int(np.argmax(uniform_score))
    factor_top = int(np.argmax(factor_joint))
    # With only P=H+T and H,T<=0, either half-weighted logit lies in this interval.
    lower, upper = a + p, a + 0.5 * p
    competitors = np.delete(upper, old_top)
    guaranteed = not len(competitors) or lower[old_top] > competitors.max()
    lengths = np.array([t["reasoning_token_count"] for t in traces])
    return {
        "trace_count": len(traces), "top_changed_uniform_vs_logged": old_top != new_top,
        "top_changed_uniform_vs_factor_joint": factor_top != new_top,
        "top_tie_in_logged": int(np.sum(joint == joint.max())) > 1,
        "top_tie_in_uniform": int(np.sum(uniform_score == uniform_score.max())) > 1,
        "top_changed_by_factor_reconstruction": old_top != factor_top,
        "max_factor_sum_error": float(np.max(np.abs(factor_joint - joint))),
        "ess_joint": float(1 / (original @ original)),
        "ess_uniform": float(1 / (uniform @ uniform)),
        "max_weight_joint": float(original.max()), "max_weight_uniform": float(uniform.max()),
        "total_variation_uniform_vs_joint": float(np.abs(uniform - original).sum() / 2),
        "score_length_rank_correlation_joint": rank_correlation(joint, lengths),
        "score_length_rank_correlation_uniform": rank_correlation(uniform_score, lengths),
        "same_top_guaranteed_for_either_half_from_bounds": bool(guaranteed),
        "segment_ranking_identified": False,
        "traces": [
            {"trace_id": t["trace_id"], "prior_logprob": float(p[i]), "answer_logprob": float(a[i]),
             "logged_joint_logit": float(joint[i]), "factor_joint_logit": float(factor_joint[i]),
             "uniform_logit": float(uniform_score[i]), "joint_weight": float(original[i]),
             "factor_joint_weight": float(factor_original[i]), "uniform_weight": float(uniform[i]),
             "possible_half_logit_lower": float(lower[i]), "possible_half_logit_upper": float(upper[i])}
            for i, t in enumerate(traces)
        ],
    }


def ranking_audit(root):
    groups, sources, absent = [], [], []
    for seed in SEEDS:
        path = source_file(root, "training_diagnostics", seed, ".jsonl.gz")
        with gzip.open(path, "rt") as stream:
            rounds = [json.loads(line) for line in stream if line.strip()]
        if [r["round"] for r in rounds] != list(range(32)):
            raise ValueError("expected each of 32 diagnostic rounds exactly once")
        sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
        for row in rounds:
            if row["run_id"] != RUN_ID or row["seed"] != seed or CELL not in row["tag"]:
                raise ValueError("diagnostic identity mismatch")
            if row["answer_target_termination"] != "eos" or row["algorithm_profile"] != "barber_q5_control":
                raise ValueError("not the historical EOS-Q5 control")
            resp = row["responsibilities"]
            for key, value in {"score":"joint", "policy":"current", "answer_policy":"current",
                               "temperature":1.0, "ess_floor_fraction":0.0}.items():
                if resp[key] != value:
                    raise ValueError(f"historical responsibility setting changed: {key}")
            if resp["trace_tape"].get("contains_full_token_ids") is not False:
                raise ValueError("trace-tape availability changed: inspect before using aggregate-only audit")
            expected = row["minibatch"]["answer_only_pids"]
            if row["minibatch"]["labelled_pids"] or len(expected) != 4 or len(set(expected)) != 4:
                raise ValueError("unexpected minibatch")
            supports = defaultdict(list)
            for trace in resp["traces"]:
                if trace["partition"] != "answer_only" or trace["pid"] not in expected:
                    raise ValueError("unexpected trace partition/question")
                if trace.get("segment_responsibility_deltas"):
                    raise ValueError("segment scores unexpectedly available: audit those directly")
                supports[trace["pid"]].append(trace)
            for pid in expected:
                coordinate = {"seed": seed, "round": row["round"], "question_id": pid}
                if not supports[pid]:
                    absent.append(coordinate)
                    continue
                groups.append({**coordinate, **support_comparison(supports[pid])})
    def summarize(rows):
        keys = ("ess_joint", "ess_uniform", "max_weight_joint", "max_weight_uniform",
                "total_variation_uniform_vs_joint", "score_length_rank_correlation_joint", "score_length_rank_correlation_uniform")
        return {
            "question_rounds": len(rows), "trace_rows": sum(r["trace_count"] for r in rows),
            "top_changes_uniform_vs_logged": sum(r["top_changed_uniform_vs_logged"] for r in rows),
            "top_changes_uniform_vs_factor_joint": sum(r["top_changed_uniform_vs_factor_joint"] for r in rows),
            "factor_reconstruction_top_changes": sum(r["top_changed_by_factor_reconstruction"] for r in rows),
            "logged_top_ties": sum(r["top_tie_in_logged"] for r in rows),
            "uniform_top_ties": sum(r["top_tie_in_uniform"] for r in rows),
            "max_factor_sum_error": max(r["max_factor_sum_error"] for r in rows),
            "half_top_guaranteed_unchanged": sum(r["same_top_guaranteed_for_either_half_from_bounds"] for r in rows),
            **{f"mean_{k}": float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys},
            "length_correlation_defined_question_rounds": {
                k: sum(r[k] is not None for r in rows) for k in keys if "correlation" in k},
        }
    return {
        "scope": "Archived moving-Q5 supports from 32 diagnostic rounds per seed; uniform reweighting of saved factors, not a new training outcome or bit-exact BF16 replay",
        "planned_question_rounds": 384, "empty_supports": absent,
        "overall": summarize(groups),
        "per_seed": {str(s): summarize([r for r in groups if r["seed"] == s]) for s in SEEDS},
        "segment_ranking_status": "unavailable: no archived per-token prior scores; do not assume equal token likelihoods",
        "regional_bound": "If P=H+T and H,T<=0, both A+0.5H+T and A+H+0.5T lie in [A+P,A+0.5P]. Overlapping intervals do not establish an actual rank change.",
        "sources": sources, "records": groups,
    }


def require_training_gate(path):
    manifest = json.loads(path.read_text())
    if manifest.get("training_may_start") is not True:
        raise ValueError("training paused: " + manifest.get("reason", "pre-training ranking audit incomplete"))
    if manifest.get("status") != "complete_pretraining_audit":
        raise ValueError("pre-training audit is not complete")
    for name in ("semantic_review.json", "saved_buffer_rankings.json"):
        if sha256(path.parent / name) != manifest["outputs"].get(name):
            raise ValueError("pre-training audit checksum mismatch")
    rankings = json.loads((path.parent / "saved_buffer_rankings.json").read_text())
    if rankings.get("segment_ranking_status") != "complete_token_resolved":
        raise ValueError("token-resolved early/late ranking audit required before training")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--receipt-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-training-gate", type=Path)
    args = parser.parse_args()
    if args.check_training_gate:
        require_training_gate(args.check_training_gate)
        return
    if any(value is None for value in (args.artifact_dir, args.annotations, args.output_dir, args.receipt_dir)):
        parser.error("--artifact-dir, --receipt-dir, --annotations and --output-dir are required for an audit")
    if args.output_dir.exists():
        raise ValueError("output directory already exists; preserve prior audit")
    annotations = json.loads(args.annotations.read_text())
    receipts = [receipt_evidence(source_file(args.artifact_dir, prefix, seed, suffix),
                                 source_file(args.receipt_dir, "complete", seed, ".json"), seed)
                for seed in SEEDS for prefix, suffix in (("dump", ".json"),
                                                         ("training_diagnostics", ".jsonl.gz"))]
    semantic = semantic_audit(args.artifact_dir, annotations)
    ranking = ranking_audit(args.artifact_dir)
    args.output_dir.mkdir(parents=True)
    for name, payload in (("semantic_review.json", semantic), ("saved_buffer_rankings.json", ranking)):
        (args.output_dir / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    manifest = {
        "schema_version": 1, "status": "partial_pretraining_audit",
        "training_may_start": False,
        "reason": "Semantic review and uniform reweighting complete; actual early/late rankings require token-resolved scoring on identical support before training.",
        "script_sha256": sha256(Path(__file__)), "annotations_sha256": sha256(args.annotations),
        "verified_input_receipts": receipts,
        "outputs": {p.name: sha256(p) for p in sorted(args.output_dir.glob("*.json"))},
        "inference_performed": False, "optimizer_steps": 0,
    }
    (args.output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"semantic": {k:v for k,v in semantic.items() if k not in ("records","sources","roles")},
                      "ranking": ranking["overall"], "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
