"""Receipt, raw-answer, missing-data and token-coefficient audit contracts."""

from copy import deepcopy
import gzip
import json

import pytest

from audit_mstep_length_normalisation import audit_source, inspect_raw, sha256, support_summary


def fixture(tmp_path, method="Q5"):
    path, receipt = tmp_path / "diagnostics.jsonl.gz", tmp_path / "complete.json"
    identity = {"run_id": "audit-test", "seed": 1201, "tag": "cell-seed1201"}
    records = []
    for round_index in range(32):
        samples = [{"pid": 3, "trace_id": f"r{round_index}:{i}", "text": text, "gold_answer": 7}
                   for i, text in enumerate(("2 + 5 = 8\n#### 8", "2 + 5 = 7\n#### 7"))]
        traces = [{"pid": 3, "partition": "answer_only", "trace_id": s["trace_id"], "is_gold": False,
                   "responsibility": w, "objective_tokens": n, "answer_tokens": 3,
                   "reasoning_token_count": n - 4, "proposal_correct": i == 1}
                  for i, (s, w, n) in enumerate(zip(samples, (.8, .2), (10, 20)))]
        records.append({**identity, "round": round_index, "answer_event_mode": "strict_terminal_marker",
                        "answer_target_termination": "eos", "generation": {"samples": {"samples": samples}},
                        "minibatch": {"labelled_pids": [], "answer_only_pids": [3]},
                        "objective": {"latent_mstep_objective": "joint"}, "responsibilities": {
                            "score": "joint", "variational_estimator": "delta_joint" if method == "Q5" else "prior_importance",
                            "refresh": "inner_step" if method == "Q5" else "outer_round", "traces": traces}})
    source = {"method": method, "run_id": "audit-test", "seed": 1201, "cell_id": "cell",
              "diagnostics": str(path), "receipt": str(receipt)}
    def save(rows):
        with gzip.open(path, "wt") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        receipt.write_text(json.dumps({"status": "complete", "identity": identity, "artifacts": [
            {"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)}]}))
    save(records)
    return source, records, save


@pytest.mark.parametrize("method", ["Q5", "PIS"])
def test_raw_answer_not_replaced_target_and_snapshot_timing(tmp_path, method):
    source, _, _ = fixture(tmp_path, method)
    result = audit_source(source)
    row = result["supports"][0]
    assert len(result["supports"]) == 32
    assert row["raw_extracted_correct_false_mass"] == .8
    assert row["arithmetic_parser_flag_true_mass"] == .8
    assert row["joint_mstep_total_token_coefficients"] == 12
    assert row["mean_mstep_total_token_coefficients"] == 1
    assert row["mean_mstep_sum_row_coefficients"] == pytest.approx(.09)
    assert result["weight_timing"] == ("end_of_round_after_M_step_refresh" if method == "Q5"
                                      else "outer_round_weights_reused_across_M_steps")


def test_unknown_is_not_valid_and_format_is_separate():
    assert inspect_raw(None)["arithmetic_parser_flag"] is None
    plain = inspect_raw({"text": "The answer is 7", "gold_answer": 7})
    assert plain["raw_extracted_correct"] is True and plain["raw_format_failure"] is True
    assert plain["arithmetic_parser_flag"] is None
    no_equations = inspect_raw({"text": "Seven in total.\n#### 7", "gold_answer": 7})
    assert no_equations["arithmetic_parser_flag"] is None
    assert no_equations["parsed_equations"] == 0


def test_persistent_trace_identity_lookup_across_rounds(tmp_path):
    source, records, save = fixture(tmp_path)
    for record in records[1:]:
        record["generation"]["samples"]["samples"] = []
        record["responsibilities"]["traces"] = deepcopy(records[0]["responsibilities"]["traces"])
    save(records)
    result = audit_source(source)
    assert all(row["raw_extracted_correct_unknown_mass"] == 0 for row in result["supports"])


@pytest.mark.parametrize("damage", ["missing_round", "duplicate_round", "identity", "weight", "length", "estep", "mstep", "duplicate_trace", "nan"])
def test_malformed_sources_fail_closed(tmp_path, damage):
    source, records, save = fixture(tmp_path)
    if damage == "missing_round":
        records.pop()
    elif damage == "duplicate_round":
        records[1]["round"] = 0
    elif damage == "identity":
        records[0]["seed"] = 1
    elif damage == "estep":
        records[0]["responsibilities"]["score"] = "token_mean"
    elif damage == "mstep":
        records[0]["objective"]["latent_mstep_objective"] = "joint_token_mean"
    elif damage == "duplicate_trace":
        records[0]["responsibilities"]["traces"][1]["trace_id"] = "r0:0"
    else:
        trace = records[0]["responsibilities"]["traces"][0]
        trace["objective_tokens" if damage == "length" else "responsibility"] = (
            float("nan") if damage == "nan" else 0)
    save(records)
    with pytest.raises(ValueError):
        audit_source(source)


def test_changed_file_and_truncated_gzip_rejected(tmp_path):
    source, _, _ = fixture(tmp_path)
    from pathlib import Path
    path = Path(source["diagnostics"])
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="completion receipt"):
        audit_source(source)
    receipt_path = Path(source["receipt"])
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"][0].update(size=path.stat().st_size, sha256=sha256(path))
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises((EOFError, OSError)):
        audit_source(source)


def test_unknown_mass_is_visible():
    row = {"responsibility": 1., "objective_tokens": 10, **inspect_raw(None)}
    result = support_summary([row])
    assert result["raw_extracted_correct_unknown_mass"] == 1
    assert result["arithmetic_parser_flag_false_mass"] == 0
    assert result["weight_length_spearman"] is None
