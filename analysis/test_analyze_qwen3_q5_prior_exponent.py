from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from analyze_qwen3_q5_prior_exponent import mechanism_rows, paired_contrasts, METRICS
from generate_qwen3_17b_q5_prior_exponent import CELL_ORDER, CONTROL_CELLS, SEEDS


def diagnostics(tau):
    traces = [dict(partition="answer_only", pid=7, trace_logprob=p, answer_logprob=a,
        responsibility_logit=a+tau*p, objective_tokens=n+1, answer_tokens=1)
        for p, a, n in ((-5., -.1, 10), (-1., -2., 3))]
    return [dict(completed_rounds=r, responsibilities=dict(prior_exponent=tau, traces=traces)) for r in range(1, 33)]


def test_mechanism_detects_prior_rank_change_and_wrong_factors():
    rows = mechanism_rows(diagnostics(0.), 0.)
    assert len(rows) == 32 and all(r["top_trace_changed"] for r in rows)
    assert all(1 < r["ess"] < 2 for r in rows)
    bad = deepcopy(diagnostics(.5))
    bad[0]["responsibilities"]["traces"][0]["responsibility_logit"] += 1
    with pytest.raises(ValueError, match="registered exponent"):
        mechanism_rows(bad, .5)
    with pytest.raises(ValueError, match="32 complete"):
        mechanism_rows(diagnostics(.5)[:-1], .5)


def test_pairing_preserves_seed_and_reader():
    frame = pd.DataFrame([dict(cell=cell, seed=seed, **{m: .8 + .01*(cell in CELL_ORDER) for m in METRICS})
        for cell in (*CELL_ORDER, *CONTROL_CELLS.values()) for seed in SEEDS])
    contrasts = paired_contrasts(frame)
    assert len(contrasts) == 24
    assert np.allclose(contrasts.iloc[:16].mean_difference_pp, 1)
    assert np.allclose(contrasts.iloc[16:].mean_difference_pp, 0)
    with pytest.raises(KeyError):
        paired_contrasts(frame.iloc[1:])
