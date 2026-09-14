from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyze_qwen3_q5_prior_segments import mechanism_rows, paired_contrasts, nominate, METRICS
from generate_qwen3_17b_q5_prior_segments import CELL_ORDER, CONTROL_CELL, SEEDS, build_payload
from validate_qwen3_17b_q5_prior_segments import _expected_coordinates, _validate_logs
from ac_alg1_prior_segments import SEGMENT_SCOPE


def diagnostics(head=.5, tail=1.):
    traces = [dict(partition="answer_only", pid=7, prior_head_logprob=h,
        prior_tail_logprob=t, trace_logprob=h+t-.7, answer_logprob=a,
        prior_marker_logprob=-.7, prior_marker_tokens=2, reasoning_token_count=n,
        responsibility_logit=a-.7+head*h+tail*t, prior_head_tokens=n//2,
        prior_tail_tokens=n-n//2, objective_tokens=n+4, answer_tokens=2)
        for h,t,a,n in ((-4., -1., -.1, 10), (-.2, -.8, -2., 3))]
    weights = np.exp([r["responsibility_logit"] for r in traces])
    weights /= weights.sum()
    for trace, weight in zip(traces, weights):
        trace["responsibility"] = float(weight)
    return [dict(completed_rounds=r, responsibilities=dict(prior_exponent=1.,
        prior_segment_scope=SEGMENT_SCOPE, prior_head_exponent=head,
        prior_tail_exponent=tail, traces=deepcopy(traces))) for r in range(1,33)]


def test_mechanism_identities_and_rejections():
    assert len(mechanism_rows(diagnostics(), .5, 1.)) == 32
    for key in ("prior_head_logprob", "prior_tail_tokens", "responsibility_logit", "responsibility",
                "prior_marker_logprob", "prior_marker_tokens", "reasoning_token_count"):
        bad = diagnostics()
        bad[0]["responsibilities"]["traces"][0][key] += 1
        with pytest.raises(ValueError):
            mechanism_rows(bad, .5, 1.)
    with pytest.raises(ValueError):
        mechanism_rows(diagnostics()[:-1], .5, 1.)
    with pytest.raises(ValueError):
        mechanism_rows(diagnostics(), 1., .5)
    wrong_scope = diagnostics()
    wrong_scope[0]["responsibilities"]["prior_segment_scope"] = "historical_prior"
    with pytest.raises(ValueError, match="fixed marker"):
        mechanism_rows(wrong_scope, .5, 1.)


def test_uniform_control_does_not_attenuate_marker():
    rows = diagnostics(.75, .75)
    assert all(not r["top_changed_vs_uniform075"] for r in mechanism_rows(rows, .75, .75))
    for record in rows:
        for trace in record["responsibilities"]["traces"]:
            trace["responsibility_logit"] -= .25 * trace["prior_marker_logprob"]
    with pytest.raises(ValueError, match="exponents"):
        mechanism_rows(rows, .75, .75)


def test_pairing_and_nomination():
    frame = pd.DataFrame([dict(cell=cell, seed=seed, **{m:.8+.01*(cell in CELL_ORDER) for m in METRICS})
        for cell in (*CELL_ORDER, CONTROL_CELL) for seed in SEEDS])
    contrasts = paired_contrasts(frame)
    assert len(contrasts) == 24
    assert np.allclose(contrasts.iloc[:12].mean_difference_pp, 1)
    assert np.allclose(contrasts.iloc[12:].mean_difference_pp, 0)
    assert nominate(frame) == CELL_ORDER[0]
    frame.loc[frame.cell.isin(CELL_ORDER), "final_strict"] = .7
    assert nominate(frame) is None
    with pytest.raises(KeyError):
        paired_contrasts(frame.iloc[1:])


def test_task_mapping_and_log_failures(tmp_path):
    rows = _expected_coordinates(build_payload())
    assert [r["task_id"] for r in rows] == list(range(1,10))
    assert [r["seed"] for r in rows] == list(SEEDS)*3
    assert [r["cell_id"] for r in rows] == [c for c in CELL_ORDER for _ in SEEDS]
    for task in range(1,10):
        (tmp_path/f"payload.123.{task}.log").write_text("=== done in 2s ===")
    assert not _validate_logs(str(tmp_path/"*.log"))
    (tmp_path/"payload.123.5.log").write_text("Traceback: bad data")
    assert _validate_logs(str(tmp_path/"*.log"))


def test_scheduler_contract():
    root = Path(__file__).resolve().parents[1]/"lm_study"
    runner = (root/"run_qwen3_17b_q5_prior_segments_ucl.sh").read_text()
    submit = (root/"submit_qwen3_17b_q5_prior_segments_ucl.sh").read_text()
    assert "#$ -t 1-9" in runner and "cell_count=3" in runner
    assert "-le 9" in runner and "-tc" not in runner
    assert "for cell in 0 1 2;" in submit and "--nshard 3" in submit
    assert 'export PYTHONPATH="$PROJ/src:$SCRIPT_DIR:$PROJ/analysis:' in submit
    assert "--validate-controls-only" in submit and "qsub -h" in submit


def test_frozen_reader_pairings_and_nomination():
    import generate_qwen3_17b_q5_prior_segments_frozen as frozen
    cells = (*CELL_ORDER, CONTROL_CELL, *frozen.CELL_ORDER, frozen.CONTROL_CELL)
    frame = pd.DataFrame([dict(cell=cell, seed=seed, **{m:.8 + .01*(cell in frozen.CELL_ORDER)
        for m in METRICS}) for cell in cells for seed in SEEDS])
    result = paired_contrasts(frame, frozen, include_readers=True)
    assert len(result) == 40  # Six within-reader contrasts plus four reader contrasts.
    assert result.iloc[:12].control.eq(frozen.CONTROL_CELL).all()
    assert np.allclose(result.iloc[24:36].mean_difference_pp, 1)
    assert np.allclose(result.iloc[36:].mean_difference_pp, 0)
    assert nominate(frame, frozen) == frozen.CELL_ORDER[0]
    frame.loc[frame.cell.isin(frozen.CELL_ORDER), "final_strict"] = .7
    assert nominate(frame, frozen) is None


def test_frozen_diagnostics_and_moving_marker_fail_closed(tmp_path):
    import json
    import generate_qwen3_17b_q5_prior_segments_frozen as frozen
    import generate_qwen3_17b_q5_prior_segments as moving
    from analyze_qwen3_q5_prior_segments import verify_moving_marker
    rows = diagnostics()
    for row in rows:
        row["responsibilities"].update(answer_policy="frozen_base", policy="current")
    assert len(mechanism_rows(rows, .5, 1., expected_reader="frozen_base")) == 32
    for field, value in (("answer_policy", "current"), ("policy", "frozen_base")):
        bad = deepcopy(rows)
        bad[0]["responsibilities"][field] = value
        with pytest.raises(ValueError, match="policy mismatch"):
            mechanism_rows(bad, .5, 1., expected_reader="frozen_base")
    marker = dict(status="ok", run_id=moving.RUN_ID, execution_commit=frozen.MOVING_COMMIT,
        source_job_id=frozen.MOVING_JOB, configuration_sha256=frozen.MOVING_CONFIG_SHA256,
        task_count=9, trained_adapter_count=9, official_test_used=False)
    path = tmp_path/"moving.ok"
    path.write_text(json.dumps(marker))
    assert verify_moving_marker(path, frozen) == marker
    for key in marker:
        changed = dict(marker)
        changed[key] = "bad"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="moving comparison marker mismatch"):
            verify_moving_marker(path, frozen)
