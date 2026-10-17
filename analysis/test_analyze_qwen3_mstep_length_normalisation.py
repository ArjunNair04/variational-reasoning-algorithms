from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

from analyze_qwen3_mstep_length_normalisation import contrasts, mechanism_rows, validate_design
from generate_qwen3_17b_mstep_length_normalisation import CONTROL_CELLS, SEEDS, build_payload


def diagnostics(method):
    score = np.array([-1., -2.])
    weights = np.exp(score-score.max()); weights /= weights.sum()
    traces = [dict(pid=0, responsibility=float(w), responsibility_logit=float(s),
                   joint_logprob=float(s), answer_logprob=float(s), objective_tokens=n)
              for w,s,n in zip(weights,score,(10,20))]
    step = dict(status="accepted", support=dict(backward_eos_tokens=2, active_traces=2, backward_tokens=30),
                gradient=dict(raw_objective_l2_norm=1.), update=dict(applied_parameter_l2_norm=.1))
    row = dict(objective=dict(latent_mstep_objective="joint_token_mean"), answer_target_termination="eos",
               answer_event_mode="strict_terminal_marker", responsibilities=dict(score="joint", temperature=1.,
               ess_floor_fraction=0., policy="current", answer_policy="current", traces=traces,
               variational_estimator="delta_joint" if method=="Q5" else "prior_importance",
               refresh="inner_step" if method=="Q5" else "outer_round"),
               inner_m_step=dict(steps=[deepcopy(step) for _ in range(1 if method=="Q5" else 4)]),
               compute=dict(tokens=dict(generated=30)))
    return [dict(deepcopy(row), completed_rounds=i+1) for i in range(32)]


@pytest.mark.parametrize("method", ["Q5", "PIS"])
def test_new_mechanism_and_unchanged_estep(method):
    rows = diagnostics(method)
    result = mechanism_rows(rows,method,"joint_token_mean")
    assert len(result)==32
    assert sum(r["backward_tokens"] for r in result)==30*32*(1 if method=="Q5" else 4)
    rows[0]["responsibilities"]["traces"][0]["responsibility"]=.5
    with pytest.raises(ValueError,match="weights"):
        mechanism_rows(rows,method,"joint_token_mean")


@pytest.mark.parametrize("damage", ["score", "reader", "eos", "mode", "missing_step"])
def test_changed_protocol_rejected(damage):
    rows = diagnostics("PIS")
    if damage == "score": rows[0]["responsibilities"]["score"]="token_mean"
    if damage == "reader": rows[0]["responsibilities"]["answer_policy"]="frozen_base"
    if damage == "eos": rows[0]["inner_m_step"]["steps"][0]["support"]["backward_eos_tokens"]=0
    if damage == "mode": rows[0]["objective"]["latent_mstep_objective"]="joint"
    if damage == "missing_step": rows[0]["inner_m_step"]["steps"].pop()
    with pytest.raises(ValueError):
        mechanism_rows(rows,"PIS","joint_token_mean")


def test_paired_contrasts_require_all_seeds():
    rows=[]
    for treatment,control in CONTROL_CELLS.items():
        for cell in (treatment,control):
            for seed in SEEDS:
                score=.81 if cell==treatment else .80
                rows.append(dict(cell=cell,seed=seed,final_extracted=score,final_strict=score,
                                 extracted_auc=score,strict_auc=score))
    frame=pd.DataFrame(rows)
    assert contrasts(frame).difference_pp.tolist()==pytest.approx([1.]*8)
    with pytest.raises(ValueError): contrasts(frame.iloc[:-1])


def test_frozen_design_is_checked(tmp_path):
    path=tmp_path/"study.yaml"
    payload=build_payload(); path.write_text(yaml.safe_dump(payload))
    assert len(validate_design(path)[1])==2
    payload["algos"]["AC-ALG1"][0]["responsibility_score"]="token_mean"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError): validate_design(path)
