"""Real tiny-Qwen adapter isolation, learned sampling and main-gradient tests."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

import ac_alg1 as alg
from ac_alg1_learned_posterior import LearnedPosterior, adapter_context
from generate_qwen3_17b_q5_learned_posterior import build_payload, runtime_configs
from test_q5_component_kernel import Tokenizer, fixtures


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(123)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=512,
        eos_token_id=0, pad_token_id=0, attention_dropout=0.))
    return get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))


def fit_args():
    x = fixtures()
    return {k:x[k] for k in ("tok", "task", "buffers", "weights", "pids")}, dict(
        seed=1201, outer_round=0, draws=16, build_prompt=lambda _: "guided",
        make_row=lambda pid,ids,tid: alg._learned_posterior_row(x["tok"],x["task"],pid,ids,0,tid),
        score=alg.seq_logprobs, device="cpu")


def force_generation(model, monkeypatch, suffix="xy#### 9"):
    from transformers import LogitsProcessorList
    generate = model.generate
    calls = []
    tokens = [ord(c) for c in suffix] + [0]
    def wrapped(**kwargs):
        calls.append(model.active_adapter)
        width = kwargs["input_ids"].shape[1]
        def force(ids, scores):
            result = torch.full_like(scores, -torch.inf)
            result[:,tokens[min(ids.shape[1]-width,len(tokens)-1)]] = 0.
            return result
        return generate(**kwargs, logits_processor=LogitsProcessorList([force]))
    monkeypatch.setattr(model, "generate", wrapped)
    return calls


def test_two_adapters_init_fit_generation_and_main_gradient(monkeypatch, tmp_path):
    from peft import get_peft_model_state_dict
    from result_contract import adapter_artifacts
    monkeypatch.setattr(alg, "DEV", "cpu")
    model = tiny_model()
    theta = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in theta]
    rng = torch.random.get_rng_state().clone()
    posterior = LearnedPosterior(model, lr=1e-3, seed=1201)
    assert torch.equal(rng, torch.random.get_rng_state())
    main_state = get_peft_model_state_dict(model, adapter_name="default")
    phi_state = get_peft_model_state_dict(model, adapter_name="learned_posterior")
    assert main_state.keys() == phi_state.keys()
    assert all(torch.equal(main_state[k],phi_state[k]) for k in main_state)
    calls = force_generation(model, monkeypatch)
    args, kwargs = fit_args()
    b,w,m = posterior.fit_and_sample(model, **args, **kwargs)
    assert calls == ["learned_posterior"]*2
    assert model.active_adapter == "default" and posterior.steps == 1
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(a,p) for a,p in zip(before,theta))
    assert {id(p) for p in theta}.isdisjoint(id(p) for p in posterior.parameters)
    assert all(p.grad is None and not p.requires_grad for p in posterior.parameters)
    assert len(b[0])==16 and w[0].tolist()==[1/16]*16
    assert len(args["buffers"][0])==2
    assert m["weights_rescored"] is False and m["children_persisted"] is False
    assert m["generated_draws"]==16 and m["repaired_draws"]==0
    assert m["posterior_backward_eos_tokens"]==2
    phi_before = [p.clone() for p in posterior.parameters]
    opt = torch.optim.Adam(theta, lr=1e-3)
    value,took = alg._backward_B_unsup_for_questions(model,args["tok"],b,[0],w)
    assert took
    actual=[None if p.grad is None else p.grad.clone() for p in theta]
    model.zero_grad()
    reference=alg._B_unsup_for_questions(model,args["tok"],b,[0],w)
    (-reference).backward()
    assert value==pytest.approx(float(reference),abs=1e-4)
    for expected,p in zip(actual,theta):
        if expected is not None:
            assert torch.allclose(expected,p.grad,atol=1e-5,rtol=1e-4)
    opt.step()
    assert any(not torch.equal(a,p) for a,p in zip(before,theta))
    assert all(torch.equal(a,p) for a,p in zip(phi_before,posterior.parameters))
    model.save_pretrained(tmp_path)
    artifacts=adapter_artifacts(tmp_path)
    assert len(artifacts)==4 and sum("learned_posterior" in str(p) for p in artifacts)==2


@pytest.mark.parametrize("text", ["", "no marker", "#### 5", "xy#### 9 trailing #### 3"])
def test_total_map_keeps_bad_outputs_and_uses_gold_eos(text):
    x=fixtures()
    row,mapping=alg._learned_posterior_row(x["tok"],x["task"],0,[ord(c) for c in text]+[0],0,"child")
    assert row is not None and row.ids[-1]==0
    assert x["tok"].decode(row.ids[row.ans]).strip()=="7"
    assert mapping in {"native_first_marker","canonical_boundary_repair"}


def test_adapter_context_restores_on_failure():
    model=tiny_model(); LearnedPosterior(model,lr=1e-3,seed=1)
    flags=[p.requires_grad for p in model.parameters()]
    with pytest.raises(RuntimeError):
        with adapter_context(model,"learned_posterior"):
            model.eval()
            raise RuntimeError("failure")
    assert model.active_adapter=="default" and model.training
    assert flags==[p.requires_grad for p in model.parameters()]


def test_frozen_config_rejects_other_interventions():
    from generate_qwen3_17b_q5_prefix_continuations import build_payload as old
    new=build_payload()["algos"]["AC-ALG1"][0]
    control=old()["algos"]["AC-ALG1"][0]
    assert {k:v for k,v in new.items() if k not in {"cell_id","algorithm_profile"}}=={
        k:v for k,v in control.items() if k not in {"cell_id","algorithm_profile"}}
    for config in runtime_configs():
        alg._validate_ac_alg1_run_config(config,diagnostics_fn=lambda _:None,diagnostics_probe_fn=None)
        for change in ({"inner_steps":2},{"component_kernel_epsilon":.25},
                       {"responsibility_prior_exponent":.5},{"responsibility_answer_policy":"frozen_base"},
                       {"mstep_sample_size":4},{"proposal_continuation_mode":"prefix_half"}):
            with pytest.raises(ValueError):
                alg._validate_ac_alg1_run_config(replace(config,**change),diagnostics_fn=lambda _:None,diagnostics_probe_fn=None)


def test_complete_round_separate_cost_and_buffer_pipeline(monkeypatch):
    from test_q5_prefix_continuations import pack
    model=tiny_model(); force_generation(model,monkeypatch)
    monkeypatch.setattr(alg,"DEV","cpu")
    monkeypatch.setattr(alg,"sample_multi",lambda m,t,prompts,**kw:pack(["abcdef#### 7"]*len(prompts)))
    task=SimpleNamespace(prompts=[f"Question: {i}\nAnswer:" for i in range(4)],
        gold_answer=[7]*4,gold_solution=[None]*4,max_new=32,
        reward=lambda texts,pids:[1.]*len(texts),answer_event_mode="strict_terminal_marker")
    observed=[]; records=[]; original=alg._run_ac_alg1_round
    class RoundDone(Exception): pass
    def one_round(**kwargs):
        original(**kwargs); observed.append(kwargs["state"]); raise RoundDone()
    monkeypatch.setattr(alg,"_run_ac_alg1_round",one_round)
    with pytest.raises(RoundDone):
        alg.run_ac_alg1(task,model_tok=(model,Tokenizer()),diagnostics_fn=records.append,
            log=lambda _:None,**asdict(next(runtime_configs())))
    state=observed[0]; d=records[0]; p=d["generation"]["learned_posterior"]
    assert state.total_generated==128 and state.total_steps==1 and state.learned_posterior.steps==1
    assert state.model.active_adapter=="default"
    assert state.records[0]["llm_gen"]==128 and d["generation"]["this_round"]==128
    assert p["generated_draws"]==64 and p["posterior_backward_eos_tokens"]==4
    assert all(len(rows)==1 for rows in state.buffers.values())
    assert all(not r.source.startswith("learned_posterior") for rows in state.buffers.values() for r in rows)
    tokens=d["compute"]["tokens"]
    assert tokens["generated"]==64*len("abcdef#### 7\0")+p["generated_tokens"]
    assert tokens["backward"]==tokens["main_backward"]+p["posterior_backward_tokens"]
    assert tokens["backward_eos"]==68 and tokens["main_backward_eos"]==64
    assert d["inner_m_step"]["steps"][0]["learned_posterior"]==p
