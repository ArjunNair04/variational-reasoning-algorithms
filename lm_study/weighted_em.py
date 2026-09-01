"""LEGACY verifier-reward weighted-EM family (Barber VRO; kept reproducible, NOT run going forward).

No-RL weighted MLE: detached E-step weight + supervised M-step over the FULL reward-reuse history
(eq 17-18; "all previous reward values are used"). A=(r-baseline)/std is the group advantage; the
"reward" r is an EXTERNAL verifier (vs the answer-conditioned AC-EM in ac_em.py). `weight` picks:
    RW   w ∝ p_θ·r          (Reward-Weighted = Barber VRO)
    AAW  w ∝ p_θ·exp(β·A)   (Anchored Advantage-Weighted = discrete Hybrid)
    AW   w ∝ exp(β·A)       (Advantage-Weighted; policy factor deleted)
anchor: w *= (p_base/p_θ)^anchor (variational prior). window=None: full history (an int caps it).
length_norm: per-token-mean logprob (LM length-bias guard; the discrete doc has no length).
"""
from __future__ import annotations
import numpy as np
import torch

from common import (DEV, MODEL_NAME, load_model, seq_logprobs, kl_from_base, _rounds, sample_round,
                    comp_len, frac_correct, ess_frac)


def run_weighted_em(task, rounds=40, B=64, G=4, seed=0, beta=2.0, window=None, epochs=8, lr=1e-4,
              weight="AW", anchor=0.0, model_name=MODEL_NAME, model_tok=None, micro=4,
              length_norm=True, log=print):
    assert weight in ("AW", "AAW", "RW"), f"unknown weight family {weight!r}"
    model, tok = (model_tok if model_tok is not None else load_model(seed=seed, model=model_name))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed)
    n_prompts = len(task.prompts)
    tag = f"{weight}-EM" + ("+a" if anchor > 0 else "")
    base_sum = np.zeros(n_prompts); base_cnt = np.zeros(n_prompts)   # per-prompt running advantage baseline
    H_ids, H_mask, H_rew, H_pid = [], [], [], []                     # the FULL reward history (eq 17), on CPU
    recs, oracle, gen, gsteps = [], 0, 0, 0                          # verifier-reward family: oracle == gen
    for t in _rounds(rounds):
        pids, pid_row, ids, mask, rew, _ = sample_round(model, tok, task, B, G, rng)
        oracle += B; gen += B                              # every sampled completion is scored by the verifier
        for b in range(len(pid_row)):                     # append new (state, FIXED reward) to the history
            H_ids.append(ids[b].cpu()); H_mask.append(mask[b].cpu())
            H_rew.append(float(rew[b])); H_pid.append(int(pid_row[b]))
            base_sum[pid_row[b]] += rew[b]; base_cnt[pid_row[b]] += 1
        if window is not None and len(H_rew) > window * B:
            k = len(H_rew) - window * B
            del H_ids[:k]; del H_mask[:k]; del H_rew[:k]; del H_pid[:k]
        rew_h = np.asarray(H_rew)
        if np.ptp(rew_h) < 1e-9:                           # no contrast anywhere -> nothing to learn
            kl = kl_from_base(model, ids, mask)
            recs.append(dict(round=t, oracle=oracle, gen=gen, gsteps=gsteps, mean_reward=float(rew.mean()),
                             frac_correct=frac_correct(rew), gen_len=comp_len(mask), ess=float("nan"),
                             loss=float("nan"), kl=kl, hist=len(H_rew)))
            log(f"  [{tag} G{G} r{t:>3}] gen={gen:>6} R={rew.mean():.3f} kl={kl:+.3f} skip"); continue
        all_ids = torch.nn.utils.rnn.pad_sequence(H_ids, batch_first=True,
                                                  padding_value=tok.eos_token_id).to(DEV)
        all_mask = torch.nn.utils.rnn.pad_sequence(H_mask, batch_first=True, padding_value=False).to(DEV)
        # ---- E-step: posterior weights over the FULL history (eq 17) ----
        lp_cur = None
        if weight in ("RW", "AAW") or anchor > 0:          # these need p_θ over the history (O(H) no-grad fwd)
            with torch.no_grad():
                lp_cur = seq_logprobs(model, all_ids, all_mask, micro=16, length_norm=length_norm)
        if weight == "RW":                                 # w ∝ p_θ·r
            rew_t = torch.tensor(rew_h, device=DEV, dtype=torch.float32)
            logits = torch.log(rew_t.clamp_min(1e-9)) + lp_cur
        else:                                              # advantage forms: A=(r-b(x))/std
            mu = base_sum / np.maximum(base_cnt, 1)
            A = (rew_h - mu[np.asarray(H_pid)]); A = A / (A.std() + 1e-8)
            logits = torch.tensor(beta * A, device=DEV, dtype=torch.float32)
            if weight == "AAW":                            # w ∝ p_θ·e^{βA}
                logits = logits + lp_cur
        if anchor > 0:                                     # variational prior: w *= (p_base/p_θ)^anchor
            with torch.no_grad(), model.disable_adapter():
                lp_base = seq_logprobs(model, all_ids, all_mask, micro=16, length_norm=length_norm)
            logits = logits + anchor * (lp_base - lp_cur).clamp(-10, 10)
        w = torch.softmax(logits, 0)
        # ---- M-step: `epochs` updates over the history, minibatches drawn ∝ q (eq 18) ----
        wn = w.detach().double().cpu().numpy(); wn = wn / wn.sum()
        H = all_ids.shape[0]; m = min(micro, H)
        model.train()
        rloss = 0.0
        for _ in range(epochs):
            sel = torch.as_tensor(rng.choice(H, size=m, replace=True, p=wn), dtype=torch.long, device=DEV)
            lp = seq_logprobs(model, all_ids[sel], all_mask[sel], grad=True, length_norm=length_norm)
            loss = -lp.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            rloss += float(loss); gsteps += 1
        kl = kl_from_base(model, ids, mask)
        recs.append(dict(round=t, oracle=oracle, gen=gen, gsteps=gsteps, mean_reward=float(rew.mean()),
                         frac_correct=frac_correct(rew), gen_len=comp_len(mask), ess=ess_frac(wn),
                         loss=rloss / max(epochs, 1), kl=kl, hist=H))
        log(f"  [{tag} G{G} r{t:>3}] gen={gen:>6} R={rew.mean():.3f} ess={ess_frac(wn):.2f} kl={kl:+.3f} H={H}")
    return recs
