"""Mass conservation, RNG isolation, native lineage and default-path parity."""

from copy import deepcopy
from dataclasses import replace
import ast
import math
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ac_alg1 as alg
from ac_alg1_component_kernel import component_kernel_support
from generate_qwen3_17b_q5_component_kernel import CONTROL_COMMIT, build_payload, runtime_configs


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def decode(self, ids, **kwargs):
        return "".join(chr(int(i)) for i in ids if i)

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [ord(c) for c in text]
        return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)


class Generator(torch.nn.Module):
    def __init__(self, suffix="xyz#### 9"):
        super().__init__()
        self.suffix = suffix
        self.calls = []

    def generate(self, **kw):
        self.calls.append(kw)
        torch.rand(3)
        suffix = torch.tensor([[ord(c) for c in self.suffix] + [0]] * len(kw["input_ids"]))
        return torch.cat([kw["input_ids"], suffix[:, :kw["max_new_tokens"]]], dim=1)


def fixtures():
    tok = Tokenizer()
    task = SimpleNamespace(prompts=["Question: q\nAnswer:"], gold_answer=[7], max_new=32)
    rows = [alg._trace_row_from_h_ids(tok, task, 0, [ord(c) for c in h], 0, "buffer",
        trace_id=f"parent{i}", reasoning_token_count=len(h)-4,
        answer_event_mode="strict_terminal_marker", answer_target_termination="eos")
        for i, h in enumerate(("abcdef####", "ghijkl####"))]
    w = torch.tensor([.8, .2])
    for row, weight in zip(rows, w):
        row.responsibility_logit = float(weight.log())
        row.trace_logprob = float(weight.log()) - .3
        row.answer_logprob = .3
    def make(pid, completion, trace_id):
        ids = torch.tensor(completion)
        return alg._sampled_trace_row(tok, task, pid, ids, torch.ones_like(ids, dtype=torch.bool),
            tok.decode(completion), 0, "component_kernel", trace_id=trace_id,
            answer_event_mode="strict_terminal_marker", answer_target_termination="eos")
    return dict(tok=tok, task=task, buffers={0:rows}, weights={0:w}, pids=[0],
        epsilon=.25, seed=1201, outer_round=0, build_prompt=lambda _: "proposal", make_row=make, device="cpu")


def test_inherited_mass_native_prefix_and_no_buffer_mutation():
    args, model = fixtures(), Generator()
    originals = [r.ids.clone() for r in args["buffers"][0]]
    support, mass, metadata = component_kernel_support(model=model, **args)
    assert mass[0].tolist() == pytest.approx([.6,.2,.15,.05])
    assert len(args["buffers"][0]) == 2
    assert all(torch.equal(row.ids, old) for row,old in zip(args["buffers"][0], originals))
    assert metadata["weights_rescored"] is False and metadata["children_persisted"] is False
    assert metadata["generated_draws"] == 2
    for i, record in enumerate(metadata["parents"]):
        assert record["prefix_ids"] == originals[i][len("Question: q\nAnswer:"):][:3].tolist()
        assert record["child_h_ids"][:3] == record["prefix_ids"]
        assert record["same_as_parent"] is False
        # The wrong sampled answer 9 is neither a filter nor the training target.
        assert args["tok"].decode(support[0][2*i+1].ids[support[0][2*i+1].ans]) == " 7"
        assert support[0][2*i+1].ids[-1] == 0
        assert support[0][2*i+1].joint_logprob == -math.inf
    assert metadata["kernel_mass"] == pytest.approx(.25)
    assert metadata["changed_mass"] == pytest.approx(.25)


@pytest.mark.parametrize("suffix", ["missing marker", ""])
def test_bad_boundary_returns_mass_to_parent_without_resampling(suffix):
    args, model = fixtures(), Generator(suffix)
    support, mass, meta = component_kernel_support(model=model, **args)
    assert len(model.calls) == 1
    assert meta["fallback_mass"] == pytest.approx(.25)
    assert meta["changed_mass"] == 0
    assert mass[0].sum() == pytest.approx(1.)
    assert support[0][0] is support[0][1] and support[0][2] is support[0][3]


def test_empty_prefix_and_zero_mass_do_not_sample():
    args, model = fixtures(), Generator()
    args["weights"][0] = torch.tensor([1.,0.])
    args["buffers"][0][0].reasoning_token_count = 1
    support, mass, meta = component_kernel_support(model=model, **args)
    assert len(support[0]) == 2 and mass[0].tolist() == [.75,.25]
    assert not model.calls and meta["generated_draws"] == 0
    assert meta["fallback_mass"] == .25


def test_rng_and_model_mode_preserved_on_success_and_failure():
    args = fixtures()
    for fail in (False,True):
        model = Generator().train()
        if fail:
            def broken(**kwargs):
                torch.rand(5)
                raise RuntimeError("failure")
            model.generate = broken
        state = torch.random.get_rng_state().clone()
        py, npstate = random.getstate(), np.random.get_state()
        if fail:
            with pytest.raises(RuntimeError):
                component_kernel_support(model=model, **args)
        else:
            component_kernel_support(model=model, **args)
        assert model.training
        assert torch.equal(state, torch.random.get_rng_state())
        assert py == random.getstate()
        assert np.array_equal(npstate[1], np.random.get_state()[1])


def test_disabled_returns_originals_without_model_access():
    args = fixtures(); args["epsilon"] = 0.
    state = torch.random.get_rng_state().clone()
    b,w,m = component_kernel_support(model=None, **args)
    assert b is args["buffers"] and w is args["weights"] and m is None
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("epsilon", [-1.,1.1,float("nan"),float("inf")])
def test_invalid_epsilon(epsilon):
    args = fixtures(); args["epsilon"] = epsilon
    with pytest.raises(ValueError):
        component_kernel_support(model=Generator(), **args)


def test_monte_carlo_objective_has_exact_inherited_gradient(monkeypatch):
    args = fixtures()
    support, masses, _ = component_kernel_support(model=Generator(), **args)
    monkeypatch.setattr(alg, "DEV", "cpu")
    model = torch.nn.Linear(1,1,bias=False)
    # A distinguishable score for every target, to catch count-normalisation.
    def logps(model, ids, mask, **kwargs):
        return model.weight.sum() * (ids.float()*mask).sum(1)
    monkeypatch.setattr(alg, "seq_logprobs", logps)
    value, took = alg._backward_B_unsup_for_questions(model, args["tok"], support, [0], masses)
    expected = sum(float(w)*float(row.ids[row.span].sum()) for row,w in zip(support[0], masses[0]))
    assert took and model.weight.grad.item() == pytest.approx(-expected)
    assert value == pytest.approx(model.weight.item()*expected)


def test_tiny_real_qwen_kernel_generation_and_weighted_backward(monkeypatch):
    from transformers import Qwen3Config, Qwen3ForCausalLM, LogitsProcessorList
    args=fixtures()
    torch.manual_seed(29)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=128,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=256,eos_token_id=0,pad_token_id=0,attention_dropout=0.))
    original=model.generate
    suffix=[ord(c) for c in "xy#### 9"]+[0]
    def generate(**kwargs):
        width=kwargs["input_ids"].shape[1]
        def force(ids,scores):
            result=torch.full_like(scores,-torch.inf)
            result[:,suffix[min(ids.shape[1]-width,len(suffix)-1)]]=0.
            return result
        return original(**kwargs,logits_processor=LogitsProcessorList([force]))
    monkeypatch.setattr(model,"generate",generate)
    monkeypatch.setattr(alg,"DEV","cpu")
    before=torch.random.get_rng_state().clone()
    support,masses,metadata=component_kernel_support(model=model,**args)
    assert torch.equal(before,torch.random.get_rng_state())
    assert metadata["changed_mass"] == pytest.approx(.25)
    value,took=alg._backward_B_unsup_for_questions(model,args["tok"],support,[0],masses)
    assert took and math.isfinite(value)
    actual=[p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    model.zero_grad()
    reference=alg._B_unsup_for_questions(model,args["tok"],support,[0],masses)
    (-reference).backward()
    assert value == pytest.approx(float(reference.detach()),abs=1e-4)
    for grad,param in zip(actual,model.parameters()):
        if grad is not None:
            assert torch.isfinite(grad).all()
            assert torch.allclose(grad,param.grad,atol=1e-5,rtol=1e-4)


def test_inner_loop_refreshes_only_original_buffer(monkeypatch):
    args = fixtures(); model = Generator()
    model.weight = torch.nn.Parameter(torch.tensor([.2]))
    monkeypatch.setattr(alg, "DEV", "cpu")
    seen = []
    def backward(model, tok, buffers, pids, weights, **kwargs):
        seen.append((buffers,weights))
        (-model.weight.sum()).backward()
        return float(model.weight.detach()), True
    def refresh(model,tok,buffers,*other,**kwargs):
        assert buffers is args["buffers"]
        assert all(row.source != "component_kernel" for row in buffers[0])
        return {},args["weights"]
    monkeypatch.setattr(alg,"_backward_B_unsup_for_questions",backward)
    monkeypatch.setattr(alg,"_refresh_minibatch_weights",refresh)
    _,returned,stats = alg._inner_weighted_em_steps(model,args["tok"],torch.optim.Adam(model.parameters(),lr=1e-5),
        args["task"],args["buffers"],[],[0],{},args["weights"],1,supervised_weight=0.,labelled_em_weight=0.,
        answer_target_termination="eos",component_kernel_epsilon=.25)
    assert len(seen)==1 and len(seen[0][0][0])==4
    assert returned is args["weights"] and stats["component_kernel"]["epsilon"]==.25


def test_disabled_inner_path_matches_deployed_source_bit_exact(monkeypatch):
    source = subprocess.check_output(["git","show",f"{CONTROL_COMMIT}:lm_study/ac_alg1.py"],text=True)
    tree = ast.parse(source)
    old_node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_inner_weighted_em_steps")
    scope = dict(vars(alg))
    exec(compile(ast.Module(body=[old_node],type_ignores=[]),"deployed_ac_alg1.py","exec"),scope)
    old = scope["_inner_weighted_em_steps"]
    monkeypatch.setattr(alg,"DEV","cpu"); scope["DEV"]="cpu"
    def backward(model,tok,buffers,pids,weights,**kwargs):
        noise = torch.rand(())
        value = -(model.weight - .7).square().sum() * (1+noise)
        (-value).backward()
        return float(value.detach()),True
    monkeypatch.setattr(alg,"_backward_B_unsup_for_questions",backward)
    scope["_backward_B_unsup_for_questions"]=backward
    monkeypatch.setattr(alg,"component_kernel_support",lambda **k: pytest.fail("disabled path sampled"))
    results=[]
    for fn in (old,alg._inner_weighted_em_steps):
        torch.manual_seed(1201)
        model=torch.nn.Linear(1,1,bias=False)
        opt=torch.optim.Adam(model.parameters(),lr=1e-5)
        _,_,stats=fn(model,None,opt,None,{0:[]},[],[0],{},{0:torch.ones(1)},1,
            supervised_weight=0.,labelled_em_weight=0.,responsibility_refresh="outer_round")
        results.append((model.weight.detach().clone(),deepcopy(opt.state_dict()),torch.random.get_rng_state().clone(),stats))
    assert torch.equal(results[0][0],results[1][0]) and torch.equal(results[0][2],results[1][2])
    for key in ("exp_avg","exp_avg_sq","step"):
        assert torch.equal(results[0][1]["state"][0][key],results[1][1]["state"][0][key])
    assert results[0][3]==results[1][3]


def test_frozen_config_and_profile_fail_closed():
    from generate_qwen3_17b_q5_prefix_continuations import build_payload as control
    new=build_payload()["algos"]["AC-ALG1"][0]
    old=control()["algos"]["AC-ALG1"][0]
    omit={"cell_id","algorithm_profile","component_kernel_epsilon"}
    assert {k:v for k,v in new.items() if k not in omit} == {k:v for k,v in old.items() if k not in omit}
    for config in runtime_configs():
        alg._validate_ac_alg1_run_config(config,diagnostics_fn=lambda _:None,diagnostics_probe_fn=None)
        for changes in ({"inner_steps":2},{"component_kernel_epsilon":.5},
                        {"proposal_continuation_mode":"prefix_half"},{"responsibility_answer_policy":"frozen_base"},
                        {"algorithm_profile":"legacy"},{"mstep_sample_size":4}):
            with pytest.raises(ValueError):
                alg._validate_ac_alg1_run_config(replace(config,**changes),diagnostics_fn=lambda _:None,diagnostics_probe_fn=None)


def test_complete_round_cost_and_buffer_pipeline(monkeypatch):
    from dataclasses import asdict
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from test_q5_prefix_continuations import pack
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=128,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=256,eos_token_id=0,pad_token_id=0,attention_dropout=0.))
    fake=Generator()
    monkeypatch.setattr(model,"generate",fake.generate)
    monkeypatch.setattr(alg,"DEV","cpu")
    monkeypatch.setattr(alg,"sample_multi",lambda m,t,prompts,**kw:pack(["abcdef#### 7"]*len(prompts)))
    task=SimpleNamespace(prompts=[f"Question: {i}\nAnswer:" for i in range(4)],
        gold_answer=[7]*4,gold_solution=[None]*4,max_new=32,
        reward=lambda texts,pids:[1.]*len(texts),answer_event_mode="strict_terminal_marker")
    observed=[]; records=[]
    original=alg._run_ac_alg1_round
    class RoundDone(Exception):
        pass
    def one_round(**kwargs):
        original(**kwargs)
        observed.append(kwargs["state"])
        raise RoundDone()
    monkeypatch.setattr(alg,"_run_ac_alg1_round",one_round)
    params=asdict(next(runtime_configs()))
    with pytest.raises(RoundDone):
        alg.run_ac_alg1(task,model_tok=(model,Tokenizer()),diagnostics_fn=records.append,log=lambda x:None,**params)
    state=observed[0]; diagnostic=records[0]
    k=diagnostic["generation"]["component_kernel"]
    assert k["generated_draws"]==4
    assert state.total_generated==68 and state.records[0]["llm_gen"]==68
    assert diagnostic["generation"]["this_round"]==68
    assert all(len(rows)==1 for rows in state.buffers.values())
    assert all(row.source!="component_kernel" for rows in state.buffers.values() for row in rows)
    assert diagnostic["compute"]["tokens"]["generated"] == 64*len("abcdef#### 7\0")+k["generated_tokens"]
    assert diagnostic["inner_m_step"]["steps"][0]["support"]["backward_eos_tokens"]==8
