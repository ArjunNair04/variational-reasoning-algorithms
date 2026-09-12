from copy import deepcopy
import builtins

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


@pytest.mark.parametrize("lengths", [(3, 3, 9, 12), (12, 9, 6, 3), (3, 6, 9, 12)])
def test_mechanism_spearman_matches_scipy_without_importing_it(monkeypatch, lengths):
    from scipy.stats import spearmanr

    logit = np.array([-1., -1., -2., -3.])
    weights = np.exp(logit - logit.max())
    weights /= weights.sum()
    expected = float(spearmanr(weights, lengths).statistic)
    ds = diagnostics(0.)
    for row in ds:
        row["responsibilities"]["traces"] = [dict(
            partition="answer_only", pid=7, trace_logprob=-5., answer_logprob=float(a),
            responsibility_logit=float(a), objective_tokens=n+1, answer_tokens=1,
        ) for a, n in zip(logit, lengths)]
    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ModuleNotFoundError("SciPy deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    rows = mechanism_rows(ds, 0.)
    assert all(row["weight_length_spearman"] == pytest.approx(expected) for row in rows)
