import json
from pathlib import Path

import numpy as np
import pytest

from audit_q5_prior_segments_pretraining import (
    CELL, RUN_ID, rank_correlation, receipt_evidence, require_training_gate,
    selected_samples, sha256, softmax, support_comparison,
)


def traces(priors=(-10.0, -10.0), answers=(-1.0, -2.0)):
    logits = np.array(priors) + answers
    return [dict(trace_id=str(i), trace_logprob=p, answer_logprob=a,
                 joint_logprob=float(logits[i]), responsibility_logit=float(logits[i]),
                 responsibility=float(softmax(logits)[i]), reasoning_token_count=10+i)
            for i, (p, a) in enumerate(zip(priors, answers))]


def test_uniform_reweighting_is_reconstructed_without_inference():
    result = support_comparison(traces((-20, -10), (-1, -10)))
    assert result["top_changed_uniform_vs_logged"] is True
    assert result["max_factor_sum_error"] == 0
    assert sum(t["uniform_weight"] for t in result["traces"]) == pytest.approx(1)
    assert result["segment_ranking_identified"] is False


def test_total_prior_does_not_identify_half_rankings():
    p, a = np.array([-10, -10]), np.array([-1, -2])
    h = np.array([-9, -1])
    assert np.argmax(a + 0.5*h + (p-h)) == 0
    assert np.argmax(a + h + 0.5*(p-h)) == 1
    h = h[::-1]
    assert np.argmax(a + 0.5*h + (p-h)) == 1
    assert np.argmax(a + h + 0.5*(p-h)) == 0
    result = support_comparison(traces())
    assert not result["same_top_guaranteed_for_either_half_from_bounds"]
    assert not result["top_changed_uniform_vs_logged"]


def test_bound_can_guarantee_winner_but_not_full_weights():
    result = support_comparison(traces((-1, -30), (-1, -2)))
    assert result["same_top_guaranteed_for_either_half_from_bounds"]
    assert not result["segment_ranking_identified"]


@pytest.mark.parametrize("field,value", [("trace_logprob", float("nan")),
                                           ("trace_logprob", 1.0),
                                           ("responsibility_logit", -100.0),
                                           ("responsibility", 0.1)])
def test_bad_saved_scores_fail_closed(field, value):
    rows = traces()
    rows[0][field] = value
    with pytest.raises(ValueError):
        support_comparison(rows)


def test_selection_ignores_correctness_and_input_order():
    samples = [dict(idx=i, completion="word " * (i+1), correct=i % 2 == 0) for i in range(100)]
    before = [(q, r["idx"]) for q, r in selected_samples(samples, 1201)]
    changed = [{**r, "correct": not r["correct"]} for r in samples[::-1]]
    assert before == [(q, r["idx"]) for q, r in selected_samples(changed, 1201)]
    assert [q for q, _ in before] == [1, 1, 2, 2, 3, 3, 4, 4]


def test_rank_correlation_handles_ties_and_constants():
    assert rank_correlation([1, 1, 2], [3, 3, 1]) == pytest.approx(-1)
    assert rank_correlation([1, 1], [2, 3]) is None


def test_pretraining_gate_rejects_partial_audit(tmp_path):
    path = tmp_path / "audit_manifest.json"
    path.write_text(json.dumps({"training_may_start": False, "reason": "missing per-token scores"}))
    with pytest.raises(ValueError, match="training paused.*missing per-token"):
        require_training_gate(path)


def test_receipt_verification_rejects_changed_source(tmp_path):
    source = tmp_path / "data.json"
    source.write_text("{}")
    receipt = tmp_path / "complete.json"
    receipt.write_text(json.dumps({"status":"complete", "identity":{
        "run_id":RUN_ID, "seed":1201, "tag":CELL}, "artifacts":[{
            "path":source.name, "size":source.stat().st_size, "sha256":sha256(source)}]}))
    assert receipt_evidence(source, receipt, 1201)["sha256"] == sha256(source)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="completion receipt"):
        receipt_evidence(source, receipt, 1201)


def test_submission_checks_gate_before_qsub():
    root = Path(__file__).parents[1]
    shell = (root / "lm_study/submit_qwen3_17b_q5_prior_segments_ucl.sh").read_text()
    assert shell.index("--check-training-gate") < shell.index("payload_output=$(qsub")
    path = root / "docs/experiments/qwen3_q5_prior_segments/pretraining_audit/audit_manifest.json"
    with pytest.raises(ValueError, match="training paused"):
        require_training_gate(path)
