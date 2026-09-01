"""Shared base for preference-optimisation trainers (DPO, APL).

Both optimise the same DPO sigmoid loss over (chosen, rejected) pairs (Rafailov et al. 2023; ref TRL
DPOTrainer):  -log σ(β·[(logπ_θ-logπ_ref)_chosen - (logπ_θ-logπ_ref)_rejected]),  summed seq logprobs.
They differ ONLY in how each round produces its pairs (DPO regenerates / pairs by reward; APL adds an
active-acquisition front-end), which subclasses supply via gather_round(). The loss, frozen-base
reference logprobs, micro-batched accumulation, round loop, and recording live here.
"""
from __future__ import annotations
import numpy as np
import torch

from common import (MODEL_NAME, load_model, seq_logprobs, kl_from_base, _rounds, build_gold_batch,
                    gold_lp, maybe_eval)


class PreferenceTrainer:
    def __init__(self, task, model_tok=None, *, seed=0, lr=1e-6, beta=0.1, epochs=1,
                 micro=2, model_name=MODEL_NAME, eval_every=0, eval_fn=None, log=print):
        self.task = task
        self.model, self.tok = (model_tok if model_tok is not None
                                else load_model(seed=seed, model=model_name))
        self.opt = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.rng = np.random.default_rng(seed)
        self.beta, self.epochs, self.micro, self.log = beta, epochs, micro, log
        self.eval_every, self.eval_fn = eval_every, eval_fn
        self.recs, self.oracle, self.gen, self.gsteps = [], 0, 0, 0

    def ref_logprobs(self, ids, mask):
        """Frozen-base summed seq logprob -- the DPO reference term log π_ref."""
        with torch.no_grad(), self.model.disable_adapter():
            return seq_logprobs(self.model, ids, mask, length_norm=False)

    def dpo_step(self, cw_ids, cw_mask, cl_ids, cl_mask, ref_c, ref_l):
        """epochs × micro-batched DPO update on one pair set (Rafailov eq 7). micro = memory only.
        Returns (mean round loss, mean implicit-reward margin, pair accuracy); increments self.gsteps
        by one per epoch. margin = β·Δ (the DPO logit); pair_acc = fraction of pairs the policy already
        ranks correctly (margin>0) -- the two standard DPO health diagnostics."""
        self.model.train()
        n = cw_ids.shape[0]
        rloss = 0.0
        msum = macc = mcnt = 0.0                          # margin / pair-accuracy accumulators (diagnostic)
        for _ in range(self.epochs):
            self.opt.zero_grad()
            for i in range(0, n, self.micro):
                sl = slice(i, i + self.micro)
                lp_c = seq_logprobs(self.model, cw_ids[sl], cw_mask[sl], grad=True)
                lp_l = seq_logprobs(self.model, cl_ids[sl], cl_mask[sl], grad=True)
                logits = self.beta * ((lp_c - ref_c[sl]) - (lp_l - ref_l[sl]))            # Rafailov eq 7
                loss = -torch.nn.functional.logsigmoid(logits).sum() / n                  # -log σ(β·Δ)
                with torch.no_grad():                     # diagnostics only
                    msum += float(logits.sum()); macc += float((logits > 0).sum()); mcnt += len(logits)
                loss.backward(); rloss += float(loss)
            self.opt.step(); self.gsteps += 1
        return (rloss / max(self.epochs, 1),
                msum / max(mcnt, 1), macc / max(mcnt, 1))

    def gather_round(self, t, B, G):
        """Subclass hook -> (pairs, info). pairs = (cw_ids,cw_mask,cl_ids,cl_mask,ref_c,ref_l) to train on,
        or None to skip (no usable pairs). info = dict(mean_reward, frac_correct, gen_len, ...). Implementations
        own sampling, any acquisition, the oracle call, and cost accounting (self.oracle/self.gen += ...)."""
        raise NotImplementedError

    def run(self, rounds, B, G, tag="PREF"):
        gbatch = build_gold_batch(self.tok, self.task)     # gold-CoT batch for the gold_lp diagnostic
        for t in _rounds(rounds):
            pairs, info = self.gather_round(t, B, G)
            base = dict(round=t, oracle=self.oracle, gen=self.gen, gsteps=self.gsteps, **info)
            if pairs is None:                              # no usable pair this round (ties / empty)
                self.recs.append(dict(base, loss=float("nan"), kl=float("nan"),
                                      margin=float("nan"), pair_acc=float("nan"),
                                      gold_lp=gold_lp(self.model, gbatch), test_acc=float("nan")))
                continue
            cw_ids, cw_mask, cl_ids, cl_mask, ref_c, ref_l = pairs
            loss, margin, pair_acc = self.dpo_step(cw_ids, cw_mask, cl_ids, cl_mask, ref_c, ref_l)
            kl = kl_from_base(self.model, cw_ids, cw_mask)
            glp = gold_lp(self.model, gbatch)              # gold-CoT likelihood after this round's update
            tacc = maybe_eval(self.model, t, rounds, self.eval_every, self.eval_fn)   # opt-in held-out point
            self.recs.append(dict(base, loss=loss, kl=kl, margin=margin, pair_acc=pair_acc,
                                  gold_lp=glp, test_acc=tacc))
            self.log(f"  [{tag} r{t:>3}] gen={self.gen:>6} oracle={self.oracle:>6} R={info['mean_reward']:.3f}"
                     f" len={info['gen_len']:.0f} kl={kl:+.3f}")
        return self.recs
