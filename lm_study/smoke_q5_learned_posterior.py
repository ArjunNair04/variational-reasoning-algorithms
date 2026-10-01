"""Offline CPU runtime check for the two-adapter path; no datasets/downloads."""

from types import SimpleNamespace
import json

import torch


def smoke():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM, LogitsProcessorList
    import ac_alg1 as alg
    from ac_alg1_learned_posterior import LearnedPosterior

    class Tokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def decode(self, ids, **kwargs):
            return "".join(chr(int(i)) for i in ids if i)

        def __call__(self, text, return_tensors=None, **kwargs):
            ids = [ord(c) for c in text]
            return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)

    model = get_peft_model(Qwen3ForCausalLM(Qwen3Config(vocab_size=128,
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=256, eos_token_id=0, pad_token_id=0)),
        LoraConfig(r=2, lora_alpha=4, lora_dropout=0., task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]))
    tok = Tokenizer()
    task = SimpleNamespace(prompts=["Question: 2+2?\nAnswer:"],gold_answer=[4],max_new=16)
    row = alg._trace_row_from_h_ids(tok,task,0,[ord(c) for c in "Two plus two is four.####"],
        0,"teacher",trace_id="teacher",answer_event_mode="strict_terminal_marker",answer_target_termination="eos")
    row.responsibility_logit=row.trace_logprob=row.answer_logprob=0.
    main=[p for p in model.parameters() if p.requires_grad]
    before=[p.clone() for p in main]
    rng=torch.random.get_rng_state().clone()
    posterior=LearnedPosterior(model,lr=1e-3,seed=1201)
    generate=model.generate; suffix=[ord(c) for c in "4#### 4"]+[0]
    def sampled(**kwargs):
        width=kwargs["input_ids"].shape[1]
        assert model.active_adapter=="learned_posterior"
        def force(ids,scores):
            out=torch.full_like(scores,-torch.inf)
            out[:,suffix[min(ids.shape[1]-width,len(suffix)-1)]]=0.
            return out
        return generate(**kwargs,logits_processor=LogitsProcessorList([force]))
    model.generate=sampled
    buffers,weights,meta=posterior.fit_and_sample(model,tok,task,{0:[row]},{0:torch.ones(1)},[0],
        seed=1201,outer_round=0,draws=16,build_prompt=lambda _:"Known answer 4. Derive:",
        make_row=lambda p,c,i:alg._learned_posterior_row(tok,task,p,c,0,i),
        score=alg.seq_logprobs,device="cpu")
    assert model.active_adapter=="default" and posterior.steps==1
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(a,p) for a,p in zip(before,main))
    assert len(buffers[0])==16 and weights[0].tolist()==[1/16]*16
    old_device=alg.DEV
    try:
        alg.DEV="cpu"
        _,took=alg._backward_B_unsup_for_questions(model,tok,buffers,[0],weights)
    finally:
        alg.DEV=old_device
    assert took and any(p.grad is not None for p in main)
    assert all(p.grad is None for p in posterior.parameters)
    return dict(status="ok",main_adapter=model.active_adapter,posterior_steps=posterior.steps,
        posterior_draws=meta["generated_draws"],rng_preserved=True,adapters_isolated=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(smoke(),sort_keys=True))
