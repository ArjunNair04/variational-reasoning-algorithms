from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyze_qwen3_q5_prior_segments import mechanism_rows, paired_contrasts, nominate, METRICS
from generate_qwen3_17b_q5_prior_segments import CELL_ORDER, CONTROL_CELL, SEEDS, build_payload
from validate_qwen3_17b_q5_prior_segments import _expected_coordinates, _validate_logs


def diagnostics(head=.5, tail=1.):
    traces = [dict(partition="answer_only", pid=7, prior_head_logprob=h,
        prior_tail_logprob=t, trace_logprob=h+t, answer_logprob=a,
        responsibility_logit=a+head*h+tail*t, prior_head_tokens=n//2,
        prior_tail_tokens=n-n//2, objective_tokens=n+2, answer_tokens=2)
        for h,t,a,n in ((-4., -1., -.1, 10), (-.2, -.8, -2., 3))]
    weights = np.exp([r["responsibility_logit"] for r in traces])
    weights /= weights.sum()
    for trace, weight in zip(traces, weights):
        trace["responsibility"] = float(weight)
    return [dict(completed_rounds=r, responsibilities=dict(prior_exponent=1.,
        prior_head_exponent=head, prior_tail_exponent=tail, traces=deepcopy(traces))) for r in range(1,33)]


def test_mechanism_identities_and_rejections():
    assert len(mechanism_rows(diagnostics(), .5, 1.)) == 32
    for key in ("prior_head_logprob", "prior_tail_tokens", "responsibility_logit", "responsibility"):
        bad = diagnostics()
        bad[0]["responsibilities"]["traces"][0][key] += 1
        with pytest.raises(ValueError):
            mechanism_rows(bad, .5, 1.)
    with pytest.raises(ValueError):
        mechanism_rows(diagnostics()[:-1], .5, 1.)
    with pytest.raises(ValueError):
        mechanism_rows(diagnostics(), 1., .5)


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
