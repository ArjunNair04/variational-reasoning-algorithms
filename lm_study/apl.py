"""APL -- active preference learning / active DPO (Muldrew et al. 2024, ICML; Barber group, no repo).

Generate a pool (compute), ACQUIRE the most informative M prompts with model-internal scores only (no
oracle), label ONLY those with the reward oracle, then take a DPO step. Implicit reward r̂=β·log(π_θ/π_ref)
(eq 5); acquisition = predictive entropy -mean log π_θ (eq 8) or preference certainty |r̂_max-r̂_min| (eq 9).
Cost (oracle calls) = M·G/round though generation touches the whole pool. The DPO step lives in the base;
this gathers pairs via active acquisition.
"""
from __future__ import annotations
import numpy as np
import torch

from common import (MAX_NEW, MODEL_NAME, seq_logprobs, sample_multi, pairs_by_reward, comp_len,
                    frac_correct, format_rate)
from preference import PreferenceTrainer


class _APLTrainer(PreferenceTrainer):
    def __init__(self, *args, acq="certainty", pool_mult=4, **kw):
        super().__init__(*args, **kw)
        self.acq, self.pool_mult = acq, pool_mult

    def gather_round(self, t, B, G):
        task, model, tok = self.task, self.model, self.tok
        n, M = len(task.prompts), max(1, B // G)
        S = min(M * self.pool_mult, n)
        pids = self.rng.choice(n, size=S, replace=False)
        pid_row = np.repeat(pids, G)
        prompts = [task.prompts[i] for i in pid_row]
        ids, mask, texts = sample_multi(model, tok, prompts, max_new=getattr(task, "max_new", MAX_NEW))
        with torch.no_grad():                                       # acquisition: model-internal only
            lp = seq_logprobs(model, ids, mask, length_norm=True).float().cpu().numpy()
            with model.disable_adapter():
                lp0 = seq_logprobs(model, ids, mask, length_norm=True).float().cpu().numpy()
        rhat = self.beta * (lp - lp0)                               # implicit reward r̂ (eq 5)
        score = {}
        for p in pids:
            idx = np.where(pid_row == p)[0]
            score[p] = (-lp[idx].mean() if self.acq == "entropy"    # predictive entropy (eq 8)
                        else abs(rhat[idx].max() - rhat[idx].min()))  # preference certainty (eq 9)
        top = sorted(pids, key=lambda p: score[p], reverse=True)[:M]
        rows = np.concatenate([np.where(pid_row == p)[0] for p in top])
        sel_texts = [texts[i] for i in rows]; sel_pid = pid_row[rows]
        sel_rew = np.asarray(task.reward(sel_texts, sel_pid), dtype=np.float64)   # oracle ONLY on selected
        self.gen += len(pid_row); self.oracle += len(rows)         # generate the POOL, label only the selected
        info = dict(mean_reward=float(sel_rew.mean()), frac_correct=frac_correct(sel_rew), gen_len=comp_len(mask),
                    fmt=format_rate(texts),                        # parseable-'####' rate over the whole pool
                    samp_lp=float(lp.mean()))                      # policy-entropy proxy (free: reuses the pool lp)
        ci, li = pairs_by_reward(np.array(top), sel_pid, sel_rew)
        if len(ci) == 0:
            return None, info
        cw_ids, cw_mask = ids[rows][ci.copy()], mask[rows][ci.copy()]
        cl_ids, cl_mask = ids[rows][li.copy()], mask[rows][li.copy()]
        ref_c, ref_l = self.ref_logprobs(cw_ids, cw_mask), self.ref_logprobs(cl_ids, cl_mask)
        return (cw_ids, cw_mask, cl_ids, cl_mask, ref_c, ref_l), info


def run_apl(task, rounds=40, B=64, G=4, seed=0, lr=1e-6, beta=0.1, epochs=1, acq="certainty",
            pool_mult=4, model_name=MODEL_NAME, model_tok=None, micro=2,
            eval_every=0, eval_fn=None, log=print):    # micro=2 like DPO
    tr = _APLTrainer(task, model_tok, seed=seed, lr=lr, beta=beta, epochs=epochs, micro=micro,
                     model_name=model_name, eval_every=eval_every, eval_fn=eval_fn, log=log,
                     acq=acq, pool_mult=pool_mult)
    return tr.run(rounds, B, G, tag="APL")
