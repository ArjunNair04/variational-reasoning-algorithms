"""Kernel analysis rejects broken lineage, mass, semantics and accounting."""

from copy import deepcopy

import pandas as pd
import pytest

from ac_alg1_component_kernel import component_kernel_support
from test_q5_component_kernel import fixtures, Generator
from analyze_qwen3_q5_component_kernel import kernel_rows, contrasts, CELL, CONTROL_CELL, SEEDS, METRICS


def diagnostics():
    args=fixtures()
    _,_,k=component_kernel_support(model=Generator(),**args)
    traces=[dict(trace_id=row.trace_id,pid=0,buffer_index=i,reasoning_token_count=row.reasoning_token_count,source="buffer")
        for i,row in enumerate(args["buffers"][0])]
    output=[]
    for round in range(1,33):
        output.append(dict(completed_rounds=round,generation=dict(component_kernel=deepcopy(k),
            proposal_prompt="answer_derive",proposal_policy="current",proposal_temperature=1.,
            sampling_intervention=None,buffer_proposals_this_round=64,this_round=66,cumulative=66*round),
            responsibilities=dict(traces=deepcopy(traces),answer_policy="current",policy="current",score="joint",ess_floor_fraction=0.,prior_exponent=1.),
            inner_m_step=dict(steps=[dict(status="accepted",component_kernel=deepcopy(k))]),
            compute=dict(tokens=dict(generated=200,backward=300,backward_eos=4,llm_generated_total=200))))
    return output


def test_valid_kernel_accounting():
    rows=kernel_rows(diagnostics())
    assert len(rows)==32 and rows[0]["train_draws"]==66
    assert rows[0]["changed_mass_per_question"]==pytest.approx(.25)


@pytest.mark.parametrize("mutation", [
    lambda d:d[0]["generation"]["component_kernel"].update(weights_rescored=True),
    lambda d:d[0]["generation"]["component_kernel"]["parents"][0].update(kernel_mass=.9),
    lambda d:d[0]["generation"]["component_kernel"]["parents"][0].update(parent_logit=9.),
    lambda d:d[0]["generation"]["component_kernel"]["parents"][0].update(prefix_ids=[99]),
    lambda d:d[0]["generation"]["component_kernel"]["parents"][0].update(parent_trace_id="absent"),
    lambda d:d[0]["generation"]["component_kernel"]["parents"].append(d[0]["generation"]["component_kernel"]["parents"][0]),
    lambda d:d[0]["responsibilities"]["traces"][0].update(source="component_kernel"),
    lambda d:d[0]["generation"].update(cumulative=64),
    lambda d:d[0]["responsibilities"].update(answer_policy="frozen_base"),
    lambda d:d[0]["compute"]["tokens"].update(generated=1),
    lambda d:d[0]["inner_m_step"]["steps"].append(d[0]["inner_m_step"]["steps"][0]),
    lambda d:d.pop(),
])
def test_corruption_rejected(mutation):
    ds=diagnostics(); mutation(ds)
    with pytest.raises((ValueError,KeyError)):
        kernel_rows(ds)


def test_paired_table_requires_exact_coordinates():
    rows=[dict(cell=c,seed=s,**{m:.7+.01*(c==CELL) for m in METRICS}) for c in (CONTROL_CELL,CELL) for s in SEEDS]
    frame=pd.DataFrame(rows)
    assert contrasts(frame).mean_difference_pp.tolist()==pytest.approx([1.]*4)
    with pytest.raises(ValueError):
        contrasts(frame.iloc[:-1])
    with pytest.raises(ValueError):
        contrasts(pd.concat([frame.iloc[:-1],frame.iloc[[0]]]))
