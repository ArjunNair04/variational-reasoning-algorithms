"""Fail-closed learned distribution accounting and paired contrasts."""

from copy import deepcopy
import pandas as pd
import pytest

from ac_alg1_component_kernel import token_hash
from ac_alg1_learned_posterior import stream_seed
from analyze_qwen3_q5_learned_posterior import posterior_rows, contrasts, CELL, CONTROL_CELL, SEEDS, METRICS


def diagnostics():
    output=[]
    for i in range(32):
        teachers=[dict(pid=p,trace_id=f"teacher:{p}",weight=1.,logit=-3.,trace_logprob=-2.,
            answer_logprob=-1.,h_ids=[20],target_ids=[20,30,0],target_sha256=token_hash([20,30,0])) for p in range(4)]
        samples=[dict(pid=p,trace_id=f"posterior:{i}:{p}:{j}",weight=1/16,
            completion_ids=[25,30,0],generated_tokens=3,h_ids=[25],target_ids=[25,30,0],
            mapping="native_first_marker",raw_extracted_correct=True,raw_strict_correct=True,
            raw_strict_format=True) for p in range(4) for j in range(16)]
        meta=dict(mode="distilled_completion_pushforward",posterior_updates=1,cumulative_posterior_updates=i+1,
            posterior_lr=1e-5,draws_per_question=16,main_adapter="default",posterior_adapter="learned_posterior",
            weights_rescored=False,children_persisted=False,generated_draws=64,max_new_tokens=256,
            fit_rng_seed=stream_seed(1201,i,"fit"),sample_rng_seed=stream_seed(1201,i,"sample"),
            generated_tokens=192,repaired_draws=0,posterior_backward_tokens=12,posterior_backward_eos_tokens=4,
            posterior_fit_seconds=1.,elapsed_seconds=2.,posterior_gradient_norm=.1,
            posterior_parameter_delta_norm=.01,teacher_objective=-4.,
            teachers=teachers,samples=samples)
        output.append(dict(completed_rounds=i+1,seed=1201,minibatch=dict(answer_only_pids=[0,1,2,3]),
            generation=dict(learned_posterior=meta,proposal_prompt="answer_derive",proposal_policy="current",
                proposal_temperature=1.,buffer_proposals_this_round=64,sampling_intervention=None,
                this_round=128,cumulative=128*(i+1)),
            responsibilities=dict(score="joint",policy="current",answer_policy="current",
                ess_floor_fraction=0.,prior_exponent=1.,traces=[dict(trace_id=t["trace_id"],pid=t["pid"]) for t in teachers]),
            inner_m_step=dict(steps=[dict(status="accepted",learned_posterior=deepcopy(meta))]),
            compute=dict(tokens=dict(backward=204,main_backward=192,backward_eos=68,
                main_backward_eos=64,generated=400,llm_generated_total=400))))
    return output


def test_cost_accounting_and_diversity():
    rows=posterior_rows(diagnostics())
    assert len(rows)==32 and rows[0]["train_draws"]==128
    assert rows[0]["posterior_unique_fraction"]==1/16


@pytest.mark.parametrize("mutation",[
    lambda d:d.pop(),
    lambda d:d[0]["generation"].update(cumulative=64),
    lambda d:d[0]["generation"]["learned_posterior"].update(weights_rescored=True),
    lambda d:d[0]["generation"]["learned_posterior"].update(sample_rng_seed=1),
    lambda d:d[0]["generation"]["learned_posterior"].update(cumulative_posterior_updates=2),
    lambda d:d[0]["generation"]["learned_posterior"]["samples"][0].update(weight=.5),
    lambda d:d[0]["generation"]["learned_posterior"]["samples"].pop(),
    lambda d:d[0]["generation"]["learned_posterior"]["teachers"][0].update(weight=.5),
    lambda d:d[0]["generation"]["learned_posterior"]["teachers"][0].update(logit=-7.),
    lambda d:d[0]["generation"]["learned_posterior"]["samples"][0].update(target_ids=[25,30,1]),
    lambda d:d[0]["compute"]["tokens"].update(backward=192),
    lambda d:d[0]["responsibilities"].update(answer_policy="frozen_base"),
    lambda d:d[0]["inner_m_step"]["steps"].append(d[0]["inner_m_step"]["steps"][0]),
])
def test_corruption_rejected(mutation):
    ds=diagnostics(); mutation(ds)
    with pytest.raises((ValueError,KeyError)):
        posterior_rows(ds)


def test_paired_coordinates():
    frame=pd.DataFrame([dict(cell=c,seed=s,**{m:.7+.01*(c==CELL) for m in METRICS})
        for c in (CONTROL_CELL,CELL) for s in SEEDS])
    assert contrasts(frame).mean_difference_pp.tolist()==pytest.approx([1.]*4)
    with pytest.raises(ValueError): contrasts(frame.iloc[:-1])
