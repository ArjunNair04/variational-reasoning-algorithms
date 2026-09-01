"""RLOO -- REINFORCE with a leave-one-out group baseline (Ahmadian et al. 2024; ref TRL RLOOTrainer).

KL-shaped return R_i = r_i - β·KL[π_θ‖π_ref]; LOO baseline A_i = R_i - mean_{j≠i} R_j;
`epochs` policy-gradient steps per sampled batch on SUMMED trajectory logprobs. The paper uses two
gradient steps per batch; the historical local default remains one. TRL adds PPO clipping for multi-epoch.
length_norm=False + lr 1e-6: summed logps are ~(len)× the token-mean, so the PG scale matches GRPO's.
"""
from __future__ import annotations
import numpy as np
import torch

from common import (DEV, MODEL_NAME, QuestionSampler,
                    resolve_question_schedule_rng,
                    load_model, seq_logprobs,
                    kl_from_base, _rounds, sample_round, comp_len, frac_correct,
                    format_rate, task_format_rate, build_gold_batch, gold_lp, maybe_eval,
                    natural_eos_mask)


def run_rloo(task, rounds=40, B=64, G=4, seed=0, lr=1e-6, kl_coef=0.02, epochs=1,
                model_name=MODEL_NAME, model_tok=None, micro=4, eval_every=0, eval_fn=None,
                eval_rounds=None, diagnostics_fn=None, checkpoint_every=0,
                checkpoint_fn=None, question_sampling="random",
                question_schedule_rng="dedicated",
                proposal_prompt="question", reward_requires_eos=False,
                log=print):
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if checkpoint_every < 0:
        raise ValueError(f"checkpoint_every must be nonnegative, got {checkpoint_every}")
    model, tok = (model_tok if model_tok is not None else load_model(seed=seed, model=model_name))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed); recs, oracle, gen, gsteps = [], 0, 0, 0   # RL: oracle == gen
    question_sampler = None
    if question_sampling != "random":
        question_sampler = QuestionSampler(
            range(len(task.prompts)),
            resolve_question_schedule_rng(
                rng,
                seed,
                question_sampling,
                question_schedule_rng,
            ),
            mode=question_sampling,
        )
    generated_tokens = backward_tokens = question_exposures = 0
    unique_questions = set()
    gbatch = build_gold_batch(tok, task)                  # gold-CoT batch for the gold_lp diagnostic
    for t in _rounds(rounds):
        pids, pid_row, ids, mask, rew, texts = sample_round(
            model, tok, task, B, G, rng, question_sampler=question_sampler,
            proposal_prompt=proposal_prompt,
            reward_requires_eos=reward_requires_eos,
        )
        oracle += B; gen += B
        generated_tokens += int(mask.sum())
        question_exposures += len(pids)
        unique_questions.update(int(pid) for pid in pids)
        fmt = task_format_rate(task, texts)               # task-valid terminal-answer rate
        natural_eos = natural_eos_mask(ids, mask, tok.eos_token_id)
        with torch.no_grad():                             # per-sequence KL penalty, fixed at sampling time
            lp_cur = seq_logprobs(model, ids, mask, length_norm=False)
            with model.disable_adapter():
                lp_ref = seq_logprobs(model, ids, mask, length_norm=False)
        slp = float((lp_cur / mask[:, 1:].sum(1).clamp(min=1)).mean())   # policy-entropy proxy at θ_old on its
                                                                          # OWN samples (free: reuses lp_cur)
        rew_kl = rew - kl_coef * (lp_cur - lp_ref).cpu().numpy()   # R_i = r_i - β·KL  (KL-shaped return)
        adv = np.empty(len(rew_kl)); dead = 0
        for p in pids:                                    # leave-one-out baseline within each group
            m = pid_row == p; g = rew_kl[m]
            if rew[m].std() == 0:                         # RAW rewards flat -> only KL noise left in the LOO
                dead += 1                                 # advantage (RL starvation on weak base models)
            adv[m] = g - (g.sum() - g) / max(len(g) - 1, 1)        # A_i = R_i - mean_{j≠i} R_j
        adv_t = torch.tensor(adv, device=DEV, dtype=torch.float32)
        model.train()
        N = len(rew)                                      # micro controls accumulation, not update count
        rloss = 0.0
        for _ in range(epochs):
            opt.zero_grad()
            for i in range(0, N, micro):
                sel = slice(i, i + micro)
                lp = seq_logprobs(model, ids[sel], mask[sel], grad=True, length_norm=False)
                loss = -(adv_t[sel] * lp).sum() / N       # ∇L = -E[A_i·∇log π_θ(o_i|q)]
                loss.backward()
                backward_tokens += int(mask[sel].sum())
                rloss += float(loss.detach())
            opt.step(); gsteps += 1
        kl = kl_from_base(model, ids, mask)
        glp = gold_lp(model, gbatch)                      # gold-CoT likelihood after this round's update
        if eval_rounds is None:
            tacc = maybe_eval(model, t, rounds, eval_every, eval_fn)
        else:
            tacc = maybe_eval(
                model, t, rounds, eval_every, eval_fn, eval_rounds=eval_rounds
            )
        recs.append(dict(round=t, oracle=oracle, verifier_calls=oracle, gen=gen, llm_gen=gen,
                         generated_tokens=generated_tokens, backward_tokens=backward_tokens,
                         question_exposures=question_exposures,
                         unique_questions_seen=len(unique_questions),
                         question_ids_this_round=[int(pid) for pid in pids],
                         questions_this_round=len(pids),
                         gsteps=gsteps, mean_reward=float(rew.mean()),
                         proposal_prompt=proposal_prompt,
                         question_schedule_rng=question_schedule_rng,
                         reward_requires_eos=reward_requires_eos,
                         natural_eos_fraction=float(natural_eos.mean()),
                         frac_correct=frac_correct(rew), gen_len=comp_len(mask),
                         adv_std=float(adv.std()), dead_frac=dead / max(len(pids), 1),
                         fmt=fmt, samp_lp=slp, gold_lp=glp, test_acc=tacc,
                         loss=rloss / epochs, kl=kl))
        if diagnostics_fn is not None:
            diagnostics_fn({
                "schema_version": 1,
                "method_family": "rloo",
                "round": t,
                "completed_rounds": t + 1,
                "generation": {
                    "generations_this_round": B,
                    "generations_cumulative": gen,
                    "generated_tokens_cumulative": generated_tokens,
                    "questions_this_round": len(pids),
                    "question_ids_this_round": [int(pid) for pid in pids],
                    "question_exposures_cumulative": question_exposures,
                    "unique_questions_seen": len(unique_questions),
                    "question_schedule_rng": question_schedule_rng,
                    "correct_fraction": frac_correct(rew),
                    "format_fraction": fmt,
                    "natural_eos_fraction": float(natural_eos.mean()),
                },
                "optimizer": {
                    "gradient_steps_this_round": epochs,
                    "gradient_steps_cumulative": gsteps,
                    "backward_tokens_cumulative": backward_tokens,
                    "loss": rloss / epochs,
                    "kl": kl,
                },
                "signal": {
                    "advantage_std": float(adv.std()),
                    "dead_group_fraction": dead / max(len(pids), 1),
                },
                "reward": {
                    "requires_natural_eos": bool(reward_requires_eos),
                },
                "test_acc": None if not np.isfinite(tacc) else tacc,
            })
        if (checkpoint_fn is not None and checkpoint_every > 0
                and (t + 1) % checkpoint_every == 0 and (t + 1) < rounds):
            checkpoint_fn(model, t + 1)
        log(f"  [RLOO r{t:>3}] gen={gen:>6} R={rew.mean():.3f} len={comp_len(mask):.0f} kl={kl:+.3f}")
    return recs
