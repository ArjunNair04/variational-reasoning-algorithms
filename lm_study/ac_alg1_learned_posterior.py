"""A separate conditional adapter fitted to Q5, then sampled for the M-step.

The learned law is a distribution over completions. A deterministic, total
completion-to-rationale map induces the variational distribution over h.
This is posterior distillation, not optimisation of the variational entropy.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math
import time

import torch

from ac_alg1_component_kernel import token_hash
from answer_events import parse_gsm8k_answer_event


def stream_seed(seed, outer_round, purpose):
    return int.from_bytes(hashlib.sha256(
        f"q5-learned-posterior:{seed}:{outer_round}:{purpose}".encode()
    ).digest()[:8], "big") % (2**63 - 1)


@contextmanager
def adapter_context(model, name):
    previous = model.active_adapter
    was_training = model.training
    flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        model.set_adapter(name)
        yield
    finally:
        model.set_adapter(previous)
        for p, enabled in flags:
            p.requires_grad_(enabled)
        model.train(was_training)


class LearnedPosterior:
    """Persistent independent Adam/LoRA state, sharing only the frozen backbone."""

    adapter_name = "learned_posterior"

    def __init__(self, model, *, lr, seed):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        if not math.isfinite(lr) or lr <= 0 or model.active_adapter != "default":
            raise ValueError("posterior pilot requires positive LR and default main adapter")
        if self.adapter_name in model.peft_config:
            raise ValueError("posterior adapter already exists")
        original = get_peft_model_state_dict(model, adapter_name="default")
        with torch.random.fork_rng():
            torch.manual_seed(stream_seed(seed, -1, "initialise"))
            model.add_adapter(self.adapter_name, deepcopy(model.peft_config["default"]))
            set_peft_model_state_dict(model, original, adapter_name=self.adapter_name)
        with adapter_context(model, self.adapter_name):
            self.parameters = [p for p in model.parameters() if p.requires_grad]
        model.set_adapter("default")
        main = {id(p) for p in model.parameters() if p.requires_grad}
        if not self.parameters or main.intersection(id(p) for p in self.parameters):
            raise ValueError("main and posterior trainable parameters overlap")
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)
        self.steps = 0

    def fit_and_sample(self, model, tok, task, buffers, weights, pids, *,
                       seed, outer_round, draws, build_prompt, make_row,
                       score, device, generation_batch_size=8):
        if draws != 16 or generation_batch_size < 1 or self.steps != outer_round:
            raise ValueError("posterior pilot requires 16 draws and one fit per round")
        pids = [int(p) for p in pids]
        if not pids or len(set(pids)) != len(pids):
            raise ValueError("posterior fit requires distinct active questions")
        teachers, examples = [], []
        posterior_backward_tokens = 0
        for pid in pids:
            w = weights[pid].detach().cpu()
            if (len(w) != len(buffers[pid]) or not torch.isfinite(w).all()
                    or (w < 0).any() or not math.isclose(float(w.sum()), 1., abs_tol=1e-5)):
                raise ValueError("invalid posterior teacher weights")
            prompt = tok(build_prompt(pid), return_tensors="pt").input_ids[0].tolist()
            for parent, weight in zip(buffers[pid], w.tolist()):
                if weight == 0:
                    continue
                target = parent.ids[parent.span].cpu().tolist()
                if not target or target[-1] != tok.eos_token_id:
                    raise ValueError("posterior teacher requires answer+EOS")
                ids = torch.tensor([prompt + target], dtype=torch.long, device=device)
                mask = torch.zeros_like(ids, dtype=torch.bool)
                mask[:, len(prompt):] = True
                examples.append((ids, mask, weight / len(pids)))
                posterior_backward_tokens += len(target)
                teachers.append(dict(pid=pid, trace_id=parent.trace_id, weight=weight,
                    logit=float(parent.responsibility_logit),
                    trace_logprob=float(parent.trace_logprob),
                    answer_logprob=float(parent.answer_logprob),
                    h_ids=parent.ids[parent.span & ~parent.ans].cpu().tolist(),
                    target_ids=target, target_sha256=token_hash(target)))

        fit_started = time.perf_counter()
        phi_before = [p.detach().clone() for p in self.parameters]
        fit_seed = stream_seed(seed, outer_round, "fit")
        with adapter_context(model, self.adapter_name), torch.random.fork_rng():
            torch.manual_seed(fit_seed)
            model.train()
            self.optimizer.zero_grad(set_to_none=True)
            objective = 0.
            for ids, mask, mass in examples:
                value = score(model, ids, mask, micro=1, grad=True)[0] * mass
                if not torch.isfinite(value):
                    raise ValueError("nonfinite posterior fit objective")
                (-value).backward()
                objective += float(value.detach())
            grad_norm = math.sqrt(sum(float(p.grad.detach().float().square().sum())
                for p in self.parameters if p.grad is not None))
            if not math.isfinite(grad_norm) or grad_norm <= 0:
                raise ValueError("posterior adapter has no finite learning signal")
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.steps += 1
        fit_seconds = time.perf_counter() - fit_started
        parameter_delta = math.sqrt(sum(float((p.detach()-old).float().square().sum())
            for p,old in zip(self.parameters,phi_before)))
        del phi_before
        if not math.isfinite(parameter_delta) or parameter_delta <= 0:
            raise ValueError("posterior update did not change its adapter")

        support, masses, samples = dict(buffers), dict(weights), []
        sample_started = time.perf_counter()
        sample_seed = stream_seed(seed, outer_round, "sample")
        with adapter_context(model, self.adapter_name), torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(sample_seed)
            model.eval()
            for pid in pids:
                prompt = tok(build_prompt(pid), return_tensors="pt").input_ids.to(device)
                support[pid] = []
                for start in range(0, draws, generation_batch_size):
                    count = min(generation_batch_size, draws-start)
                    ids = prompt.repeat(count, 1)
                    output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        do_sample=True, max_new_tokens=int(task.max_new),
                        pad_token_id=int(tok.eos_token_id), temperature=1., top_k=0, top_p=1.)
                    if output.shape[0] != count or not torch.equal(output[:, :ids.shape[1]], ids):
                        raise ValueError("posterior generator changed its prompt")
                    for j in range(count):
                        completion = output[j, ids.shape[1]:].cpu().tolist()
                        if tok.eos_token_id in completion:
                            completion = completion[:completion.index(tok.eos_token_id)+1]
                        if not completion or len(completion)>int(task.max_new):
                            raise ValueError("invalid posterior generation length")
                        trace_id = f"posterior:{outer_round}:{pid}:{start+j}"
                        row, mapping = make_row(pid, completion, trace_id)
                        if row is None:
                            raise ValueError("posterior map must retain every draw")
                        support[pid].append(row)
                        text = tok.decode(completion, skip_special_tokens=True)
                        raw = parse_gsm8k_answer_event(text, mode="legacy")
                        strict = parse_gsm8k_answer_event(text, mode="strict_terminal_marker")
                        samples.append(dict(pid=pid, trace_id=trace_id, weight=1/draws,
                            completion_ids=completion, generated_tokens=len(completion),
                            h_ids=row.ids[row.span & ~row.ans].cpu().tolist(),
                            mapping=mapping, target_ids=row.ids[row.span].cpu().tolist(),
                            raw_extracted_correct=raw.answer == task.gold_answer[pid],
                            raw_strict_correct=strict.strict_valid and strict.answer == task.gold_answer[pid],
                            raw_strict_format=strict.strict_valid))
                masses[pid] = torch.full((draws,), 1/draws,
                    dtype=weights[pid].dtype, device=weights[pid].device)
        if model.active_adapter != "default":
            raise ValueError("posterior sampling failed to restore main adapter")
        metadata = dict(mode="distilled_completion_pushforward", posterior_updates=1,
            cumulative_posterior_updates=self.steps, posterior_lr=self.optimizer.param_groups[0]["lr"],
            draws_per_question=draws, fit_rng_seed=fit_seed, sample_rng_seed=sample_seed,
            main_adapter=model.active_adapter, posterior_adapter=self.adapter_name,
            weights_rescored=False, children_persisted=False, teacher_objective=objective,
            posterior_gradient_norm=grad_norm, posterior_backward_tokens=posterior_backward_tokens,
            posterior_parameter_delta_norm=parameter_delta,
            posterior_backward_eos_tokens=len(examples), posterior_fit_seconds=fit_seconds,
            elapsed_seconds=time.perf_counter()-sample_started, generated_draws=len(samples),
            generated_tokens=sum(s["generated_tokens"] for s in samples),
            repaired_draws=sum(s["mapping"] != "native_first_marker" for s in samples),
            max_new_tokens=int(task.max_new), teachers=teachers, samples=samples)
        return support, masses, metadata
