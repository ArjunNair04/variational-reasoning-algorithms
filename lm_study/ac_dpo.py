"""AC-DPO -- ANSWER-CONDITIONED preference optimisation (M-step repair: the contrastive form).

Replaces AC-EM's weighted-MLE M-step with the DPO sigmoid loss while keeping its E-step signal:
each question's G fresh samples are RANKED by the dense answer-likelihood log p(a*| q ++ h) (gold
answer teacher-forced after the reason-cut h -- exactly AC-EM's E-step score), and the per-question
(best, worst) pair trains the standard reference-anchored DPO objective (preference.PreferenceTrainer).
This imports at once the two structural properties the knob screen showed eq-9 lacks: NEGATIVE
gradients (the rejected trace is pushed down; MLE can only push up) and a BOUNDED, base-referenced
update (the trust region behind GRPO/DPO's kl~0 vs AC arms' 0.3-1.7 drift). VERIFIER-FREE like AC-EM:
pairs come from p(a*|h), never from a reward oracle (oracle stays 0 -- the cost-axis selling point).

Deviation noted for the writeup: the ranking score re-tokenises [q ++ h-text ++ gold] (decode ->
retokenise), unlike AC-EM's exact-id slicing -- acceptable because the score only RANKS within a
question; the TRAINED pair uses the exact sampled ids untouched.
"""
from __future__ import annotations
import numpy as np
import torch

from ac_em import _reason_cut
from common import (DEV, MODEL_NAME, seq_logprobs, sample_round, pairs_by_reward, comp_len,
                    frac_correct, format_rate, mean_token_lp)
from preference import PreferenceTrainer


class _ACDPOTrainer(PreferenceTrainer):
    def gather_round(self, t, B, G):
        task, model, tok = self.task, self.model, self.tok
        pids, pid_row, ids, mask, rew, texts = sample_round(model, tok, task, B, G, self.rng)
        self.gen += B                                      # oracle stays 0: no verifier, answer-likelihood only
        pad = torch.nn.utils.rnn.pad_sequence
        s_ids, s_ans = [], []                              # score batch: [q ++ h ++ '\n#### gold_answer'], answer span
        for b in range(len(pid_row)):
            pid = int(pid_row[b]); gold_answer = task.gold_answer[pid]
            h_txt = texts[b][:_reason_cut(texts[b])]       # reasoning only (model's own answer cut away)
            p = tok(task.prompts[pid] + h_txt, return_tensors="pt").input_ids[0].tolist()
            a = tok(f"\n#### {gold_answer}", add_special_tokens=False).input_ids
            f = torch.tensor(p + a, dtype=torch.long)
            sp = torch.zeros(len(f), dtype=torch.bool); sp[len(p):] = True
            s_ids.append(f); s_ans.append(sp)
        with torch.no_grad():                              # the E-step signal at theta_old (ranking only)
            lp = seq_logprobs(model, pad(s_ids, batch_first=True, padding_value=tok.eos_token_id).to(DEV),
                              pad(s_ans, batch_first=True, padding_value=False).to(DEV), micro=16)
        sig = lp.float().cpu().numpy()
        info = dict(mean_reward=float(rew.mean()), frac_correct=frac_correct(rew), gen_len=comp_len(mask),
                    fmt=format_rate(texts), samp_lp=mean_token_lp(model, ids, mask))
        ci, li = pairs_by_reward(pids, pid_row, sig)       # chosen=argmax, rejected=argmin p(a*|h) per question
        if len(ci) == 0:
            return None, info
        cw_ids, cw_mask = ids[ci.copy()], mask[ci.copy()]
        cl_ids, cl_mask = ids[li.copy()], mask[li.copy()]
        return (cw_ids, cw_mask, cl_ids, cl_mask,
                self.ref_logprobs(cw_ids, cw_mask), self.ref_logprobs(cl_ids, cl_mask)), info


def run_ac_dpo(task, rounds=40, B=64, G=4, seed=0, lr=1e-6, beta=0.1, epochs=1,
               model_name=MODEL_NAME, model_tok=None, micro=2, eval_every=0, eval_fn=None, log=print):
    assert hasattr(task, "gold_answer"), "AC-DPO needs a task with KNOWN gold answers (GSM8K-style)"
    tr = _ACDPOTrainer(task, model_tok, seed=seed, lr=lr, beta=beta, epochs=epochs, micro=micro,
                       model_name=model_name, eval_every=eval_every, eval_fn=eval_fn, log=log)
    return tr.run(rounds, B, G, tag="AC-DPO")
