from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from score_q5_prior_segments import SETTINGS, SCOPES, four_way, masks_for_row, reconstruct
from analyze_q5_segment_scoring import summarize, verify_record
from ac_alg1_prior_segments import split_reasoning_marker_mask


class CharacterTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        values = [ord(c)+2 for c in text]
        ids = torch.tensor([values]) if kwargs.get("return_tensors") == "pt" else values
        return SimpleNamespace(input_ids=ids)

    def decode(self, values, **kwargs):
        return "".join(chr(int(v)-2) for v in values if int(v) >= 2)


def fixture():
    tok = CharacterTokenizer()
    task = SimpleNamespace(prompts=["Question: one plus one?\nAnswer:"],gold_answer=[2])
    group = {"pid":0,"round":0,"candidates":[{
        "trace":{"trace_id":"r0:p0","source":"answer_derive","proposal_tokens":18},
        "sample":{"text":" One plus one is two.\n#### 3"}}]}
    return tok,task,group


def test_native_reconstruction_preserves_prompt_and_gold_eos():
    tok,task,group = fixture()
    row = reconstruct(tok,task,group)[0]
    assert tok.decode(row.ids) == task.prompts[0]+" One plus one is two.\n#### 2"
    assert row.ids[-1] == tok.eos_token_id and row.ans[-1]
    masks = masks_for_row(row)
    for head,tail,marker in masks.values():
        assert torch.equal((head|tail|marker)[0],row.span & ~row.ans)
        assert not ((head & tail)|(head & marker)|(tail & marker)).any()
    head,tail,marker = masks["reasoning_marker_fixed"]
    train_masks = split_reasoning_marker_mask((row.span & ~row.ans).unsqueeze(0), [row.reasoning_token_count])
    assert all(torch.equal(a,b) for a,b in zip(train_masks, (head,tail,marker)))
    assert tok.decode(row.ids[marker[0]]) == "####"
    assert not marker[0][row.ans].any()
    assert head.sum() == row.reasoning_token_count//2
    assert tail.sum() == row.reasoning_token_count-row.reasoning_token_count//2
    assert tok.decode(row.ids[(masks["historical_prior"][0]|masks["historical_prior"][1])[0]]).endswith("####")


def test_all_rules_share_joint_but_marker_scope_can_change_ranking():
    factors=[]
    for answer,h,t,marker in [(-1.,-2.,-3.,-4.),(-2.,-4.,-1.,-1.)]:
        factors.append({"joint":answer+h+t+marker,
            "historical_prior":{"answer":answer,"head":h,"tail":t+marker,"fixed_marker":0},
            "reasoning_marker_fixed":{"answer":answer,"head":h,"tail":t,"fixed_marker":marker}})
    result=four_way(factors)
    for scope in SCOPES:
        assert set(result[scope]) == set(SETTINGS)
        assert all(sum(r["weights"]) == pytest.approx(1) for r in result[scope].values())
    np.testing.assert_allclose(result[SCOPES[0]]["joint"]["logits"],result[SCOPES[1]]["joint"]["logits"])
    assert result[SCOPES[0]]["uniform075"]["logits"] != result[SCOPES[1]]["uniform075"]["logits"]
    factors[0]["joint"] += 1
    with pytest.raises(ValueError,match="partition"):
        four_way(factors)


def test_scoring_code_has_no_training_or_dataset_loader():
    root=Path(__file__).parents[1]
    code=(root/"analysis/score_q5_prior_segments.py").read_text()
    for forbidden in (".generate(",".backward(","torch.optim.","load_dataset("):
        assert forbidden not in code
    assert "local_files_only=True" in code
    assert "model.requires_grad_(False)" in code
    runner=(root/"lm_study/run_qwen3_17b_q5_segment_scoring_ucl.sh").read_text()
    assert "#$ -t 1-3" in runner and "#$ -tc" not in runner


def test_token_tape_roundtrip_and_corruption():
    tok,task,group=fixture()
    row=reconstruct(tok,task,group)[0]
    lp=torch.linspace(-.2,-2,len(row.ids),dtype=torch.float64)
    factors={"joint":float(lp[row.span].sum())}; positions={}
    for scope,(head,tail,marker) in masks_for_row(row).items():
        factors[scope]={"answer":float(lp[row.ans].sum()),"head":float(lp[head[0]].sum()),
                        "tail":float(lp[tail[0]].sum()),"fixed_marker":float(lp[marker[0]].sum())}
        positions[scope]={k:v[0].nonzero().flatten().tolist() for k,v in (("head",head),("tail",tail),("fixed_marker",marker))}
    trace={"token_ids":row.ids.tolist(),"score_positions":row.span.nonzero().flatten().tolist(),
           "answer_positions":row.ans.nonzero().flatten().tolist(),"token_logprobs":lp[row.span].tolist(),
           "segment_positions":positions,"factors":factors,"original_objective_tokens":int(row.span.sum()),
           "reconstructed_objective_tokens":int(row.span.sum())}
    record={"traces":[trace],"comparisons":four_way([factors])}
    verify_record(record)
    summary=summarize([record])
    json.dumps(summary,allow_nan=False)
    for scope in SCOPES:
        for cell in summary[scope]["cells"].values():
            assert cell["defined_length_correlations"] == 0
            assert cell["mean_weight_length_spearman"] is None
            assert cell["mean_weighted_reasoning_tokens"] == row.reasoning_token_count
    record["traces"][0]["factors"]["historical_prior"]["head"]+=1
    with pytest.raises(ValueError,match="factors"):
        verify_record(record)
