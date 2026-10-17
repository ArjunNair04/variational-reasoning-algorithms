"""Audit receipt-bound Q5/PIS training traces without inference or rescoring."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from ac_alg1 import _numeric_audit
from answer_events import parse_gsm8k_answer_event
from audit_q5_prior_segments_pretraining import rank_correlation


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(text):
    def nonfinite(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(text, parse_constant=nonfinite)


def verify_source(source):
    path, receipt_path = Path(source["diagnostics"]), Path(source["receipt"])
    receipt = load_json(receipt_path.read_text())
    identity = receipt["identity"]
    if (source["method"] not in ("Q5", "PIS") or receipt.get("status") != "complete"
            or identity["run_id"] != source["run_id"] or identity["seed"] != source["seed"]
            or source["cell_id"] not in identity["tag"]):
        raise ValueError("source completion identity mismatch")
    entries = [item for item in receipt["artifacts"] if Path(item["path"]).name == path.name]
    digest = sha256(path)
    if (len(entries) != 1 or entries[0]["size"] != path.stat().st_size
            or entries[0]["sha256"] != digest):
        raise ValueError("diagnostics do not match their completion receipt")
    return identity, {
        **source, "diagnostics": str(path.resolve()), "diagnostics_sha256": digest,
        "receipt": str(receipt_path.resolve()), "receipt_sha256": sha256(receipt_path),
    }


def inspect_raw(sample):
    if sample is None:
        return {"raw_text": None, "raw_extracted_correct": None, "raw_strict_correct": None,
                "raw_format_failure": None, "arithmetic_parser_flag": None,
                "arithmetic_scope": "unavailable", "parsed_equations": None,
                "equation_mentions": None, "invalid_equations": [], "raw_parsed_answer": None}
    text, gold = sample["text"], sample.get("gold_answer")
    parsed = parse_gsm8k_answer_event(text)
    strict = parse_gsm8k_answer_event(text, mode="strict_terminal_marker")
    # Missing markers do not identify the retained rationale/answer boundary.
    arithmetic = _numeric_audit(text[:parsed.marker_start], "") if parsed.marker_start is not None else None
    return {
        "raw_text": text, "raw_parsed_answer": parsed.answer,
        "raw_extracted_correct": (parsed.answer == int(gold) if gold is not None else None),
        "raw_strict_correct": (strict.strict_valid and strict.answer == int(gold) if gold is not None else None),
        "raw_format_failure": strict.format_failure,
        "arithmetic_parser_flag": (
            bool(arithmetic.invalid_equations) if arithmetic and arithmetic.parsed_equations else None
        ),
        "arithmetic_scope": "raw_prefix_before_first_marker" if arithmetic else "unavailable",
        "parsed_equations": arithmetic.parsed_equations if arithmetic else None,
        "equation_mentions": arithmetic.equation_mentions if arithmetic else None,
        "invalid_equations": list(arithmetic.invalid) if arithmetic else [],
    }


def support_summary(traces):
    weights = [float(row["responsibility"]) for row in traces]
    lengths = [int(row["objective_tokens"]) for row in traces]
    if (not traces or any(not math.isfinite(w) or w < 0 for w in weights)
            or not math.isclose(sum(weights), 1, abs_tol=2e-6)
            or any(n <= 0 for n in lengths)):
        raise ValueError("invalid posterior mass or objective length")
    result = {
        "trace_count": len(traces), "posterior_mass": sum(weights),
        "ess": 1 / sum(w * w for w in weights), "largest_weight": max(weights),
        "mean_objective_tokens": mean(lengths),
        "posterior_mean_objective_tokens": sum(w * n for w, n in zip(weights, lengths)),
        "weight_length_spearman": rank_correlation(weights, lengths),
        "mean_mstep_sum_row_coefficients": sum(w / n for w, n in zip(weights, lengths)),
        "joint_mstep_total_token_coefficients": sum(w * n for w, n in zip(weights, lengths)),
        "mean_mstep_total_token_coefficients": sum(weights),
    }
    for field in ("raw_extracted_correct", "raw_strict_correct", "raw_format_failure", "arithmetic_parser_flag"):
        for label, value in (("true", True), ("false", False), ("unknown", None)):
            result[f"{field}_{label}_mass"] = sum(
                w for row, w in zip(traces, weights) if row[field] is value
            )
    return result


def audit_source(source):
    identity, provenance = verify_source(source)
    path = Path(source["diagnostics"])
    opener = gzip.open if path.suffix == ".gz" else open
    seen_rounds, generations, supports, trace_records, empty = [], {}, [], [], []
    expected_estimator, expected_refresh = (
        ("delta_joint", "inner_step") if source["method"] == "Q5" else ("prior_importance", "outer_round")
    )
    with opener(path, "rt") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = load_json(line)
            if any(record[key] != identity[key] for key in ("run_id", "seed", "tag")):
                raise ValueError("mixed diagnostic identity")
            if (record["round"] != len(seen_rounds) or record["answer_target_termination"] != "eos"
                    or record["answer_event_mode"] != "strict_terminal_marker"):
                raise ValueError("round order or EOS/answer contract mismatch")
            seen_rounds.append(record["round"])
            resp = record["responsibilities"]
            if (resp["variational_estimator"] != expected_estimator or resp["refresh"] != expected_refresh
                    or resp["score"] != "joint" or record["objective"]["latent_mstep_objective"] != "joint"):
                raise ValueError("not an unchanged Q5/PIS joint-objective control")
            for sample in record["generation"]["samples"]["samples"]:
                if sample.get("trace_id") is None:
                    continue
                key = (sample["pid"], sample["trace_id"])
                if key in generations:
                    raise ValueError("duplicate generated trace identity")
                generations[key] = sample
            expected_pids = record["minibatch"]["answer_only_pids"]
            if record["minibatch"]["labelled_pids"] or len(set(expected_pids)) != len(expected_pids):
                raise ValueError("unexpected labelled or duplicate question support")
            grouped = defaultdict(list)
            for trace in resp["traces"]:
                if trace["partition"] != "answer_only" or trace["pid"] not in expected_pids or trace["is_gold"]:
                    raise ValueError("unexpected posterior trace")
                key = (trace["pid"], trace["trace_id"])
                raw = inspect_raw(generations.get(key))
                row = {
                    "method": source["method"], "seed": source["seed"], "round": record["round"],
                    "pid": trace["pid"], "trace_id": trace["trace_id"],
                    "responsibility": trace["responsibility"], "objective_tokens": trace["objective_tokens"],
                    "reasoning_tokens": trace["reasoning_token_count"], "answer_tokens": trace["answer_tokens"],
                    "logged_proposal_correct": trace["proposal_correct"], **raw,
                }
                grouped[trace["pid"]].append(row)
                trace_records.append(row)
            for pid in expected_pids:
                rows = grouped[pid]
                coordinate = {"method": source["method"], "seed": source["seed"], "round": record["round"], "pid": pid}
                if not rows:
                    empty.append(coordinate)
                    continue
                if len({row["trace_id"] for row in rows}) != len(rows) or any(row["trace_id"] is None for row in rows):
                    raise ValueError("posterior trace IDs must be distinct and present")
                supports.append({**coordinate, **support_summary(rows)})
    if seen_rounds != list(range(32)) or not supports:
        raise ValueError("expected 32 complete diagnostic rounds and nonempty support")
    if sha256(path) != provenance["diagnostics_sha256"]:
        raise ValueError("source changed while reading")
    return {
        "source": provenance,
        "weight_timing": ("end_of_round_after_M_step_refresh" if source["method"] == "Q5"
                          else "outer_round_weights_reused_across_M_steps"),
        "empty_supports": empty, "supports": supports, "traces": trace_records,
    }


def summarize(audit):
    rows = audit["supports"]
    keys = [key for key in rows[0] if key not in ("method", "seed", "round", "pid")]
    return {
        "method": audit["source"]["method"], "seed": audit["source"]["seed"],
        "weight_timing": audit["weight_timing"], "question_rounds": len(rows),
        "trace_rows": len(audit["traces"]), "empty_question_rounds": len(audit["empty_supports"]),
        "question_means": {key: mean(vals) if (vals := [row[key] for row in rows if row[key] is not None]) else None for key in keys},
        "defined_weight_length_correlations": sum(row["weight_length_spearman"] is not None for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True, help="JSON with explicit method/seed/run/cell and diagnostics/receipt paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = load_json(args.sources.read_text())["sources"]
    coordinates = [(s["method"], s["seed"]) for s in sources]
    if not sources or len(set(coordinates)) != len(coordinates):
        raise ValueError("one source per method/seed required")
    audits = [audit_source(source) for source in sources]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "audit_complete_for_review", "training_may_start": False,
        "per_seed": [summarize(audit) for audit in audits],
        "scope": [
            "Saved training traces only; no model inference, optimizer steps or dataset loading.",
            "Q5 weights are end-of-round snapshots, not its pre-update training weights.",
            "Arithmetic flags are unverified parser outputs, not measured reasoning-error rates. Units, percentages and rounding can cause false positives.",
            "Arithmetic checks cover explicit equations before the first marker; a false flag value does not establish valid reasoning, and missing equations remain unknown.",
            "Token-coefficient totals describe the objective, not measured gradient norms or Adam update sizes.",
            "Per-seed means average question-round supports equally; trace rows are not independent trials.",
            "Review findings and source/control equivalence before freezing or submitting the training study.",
        ],
    }
    for name, payload in (("summary.json", summary), ("sources.json", [audit["source"] for audit in audits])):
        (args.output_dir / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for name, key in (("supports.jsonl", "supports"), ("traces.jsonl", "traces")):
        with (args.output_dir / name).open("w") as stream:
            for audit in audits:
                for row in audit[key]:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
    manifest = {
        "script_sha256": sha256(Path(__file__)), "source_spec_sha256": sha256(args.sources),
        "outputs": {p.name: sha256(p) for p in sorted(args.output_dir.iterdir())},
        "training_may_start": False, "inference_performed": False, "optimizer_steps": 0,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
