"""GRPO -- clipped policy gradient + KL (Shao et al. 2024, DeepSeekMath; ref TRL GRPOTrainer).

Â=(r-mean)/std per-prompt group; ρ=π_θ/π_θ_old; loss = -(min(ρÂ, clip(ρ,1±ε)Â) - β·KL_k3),
token-mean normalised, μ=`epochs` inner updates. Deviations vs DeepSeekMath: token-level (not
per-seq 1/|o|) loss norm (DAPO / current-TRL default); KL kept (β=`kl_coef`=0.02) where TRL now defaults 0.
"""
from __future__ import annotations
import numpy as np
import torch

from common import (DEV, MODEL_NAME, QuestionSampler,
                    resolve_question_schedule_rng,
                    load_model, token_logps, kl_from_base,
                    _rounds, sample_round, comp_len, frac_correct, format_rate,
                    task_format_rate,
                    build_gold_batch, gold_lp, maybe_eval, natural_eos_mask)

OPTIMIZER_STEP_SCOPES = {"microbatch", "batch"}


def run_grpo(task, rounds=40, B=64, G=4, seed=0, lr=5e-5, clip=0.2, epochs=4, kl_coef=0.02,
                model_name=MODEL_NAME, model_tok=None, micro=4, eval_every=0, eval_fn=None,
                eval_rounds=None, diagnostics_fn=None,
                checkpoint_every=0, checkpoint_fn=None, question_sampling="random",
                question_schedule_rng="dedicated",
                optimizer_step_scope="microbatch", proposal_prompt="question",
                reward_requires_eos=False,
                log=print):
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if checkpoint_every < 0:
        raise ValueError(f"checkpoint_every must be nonnegative, got {checkpoint_every}")
    if optimizer_step_scope not in OPTIMIZER_STEP_SCOPES:
        raise ValueError(
            "optimizer_step_scope must be 'microbatch' or 'batch', got "
            f"{optimizer_step_scope!r}"
        )
    model, tok = (model_tok if model_tok is not None else load_model(seed=seed, model=model_name))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed)
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
    gbatch = build_gold_batch(tok, task)                  # gold-CoT batch for the gold_lp diagnostic
    recs, oracle, gen, gsteps = [], 0, 0, 0               # RL: every sampled completion is scored (oracle == gen)
    generated_tokens = backward_tokens = question_exposures = 0
    unique_questions = set()
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
        A_seq = np.empty(len(pid_row)); dead = 0
        for p in pids:                                    # group-standardise within (prompt, round)
            m = pid_row == p; g = rew[m]
            if g.std() == 0:                              # all-same rewards -> A=0 -> NO gradient from this group
                dead += 1                                 # (RL starvation: the bootstrap failure on weak base models)
            A_seq[m] = (g - g.mean()) / (g.std() + 1e-8)  # Â=(r-mean)/std  [DeepSeekMath]
        A = torch.tensor(A_seq, device=DEV, dtype=torch.float32)[:, None]
        m_tok = mask[:, 1:].float()
        with torch.no_grad():                             # π_θ_old and frozen-base π_ref logps, fixed per round
            old_lp = torch.cat([token_logps(model, ids[i:i + micro]) for i in range(0, len(pid_row), micro)])
            with model.disable_adapter():
                ref_lp = torch.cat([token_logps(model, ids[i:i + micro]) for i in range(0, len(pid_row), micro)])
        slp = float((old_lp * m_tok).sum() / m_tok.sum().clamp(min=1))   # policy-entropy proxy at θ_old
                                                                          # on its OWN samples (free: reuses old_lp)
        model.train()
        rloss = 0.0; nstep = 0
        cnum = cden = 0.0                                 # PPO clip fraction (standard "updates too big" gauge)
        for _ in range(epochs):
            perm = torch.randperm(len(pid_row))
            if optimizer_step_scope == "batch":
                opt.zero_grad()
                epoch_tokens = m_tok[perm].sum().clamp(min=1)
            for i in range(0, len(perm), micro):
                sel = perm[i:i + micro]
                lp = token_logps(model, ids[sel], grad=True)
                ratio = torch.exp(lp - old_lp[sel])       # ρ
                surr = torch.min(ratio * A[sel], ratio.clamp(1 - clip, 1 + clip) * A[sel])
                logr = (ref_lp[sel] - lp).clamp(-5, 5)
                kl_tok = torch.exp(logr) - logr - 1       # KL k3 estimator (Schulman 2020)
                per_tok = -(surr - kl_coef * kl_tok)
                denominator = (
                    epoch_tokens
                    if optimizer_step_scope == "batch"
                    else m_tok[sel].sum().clamp(min=1)
                )
                loss = (per_tok * m_tok[sel]).sum() / denominator
                with torch.no_grad():                     # diagnostic only: fraction of tokens hitting the clip
                    cnum += float((((ratio - 1).abs() > clip).float() * m_tok[sel]).sum())
                    cden += float(m_tok[sel].sum())
                if optimizer_step_scope == "microbatch":
                    opt.zero_grad()
                loss.backward()
                backward_tokens += int(m_tok[sel].sum())
                rloss += float(loss.detach())
                if optimizer_step_scope == "microbatch":
                    opt.step()
                    nstep += 1
            if optimizer_step_scope == "batch":
                opt.step()
                nstep += 1
        gsteps += nstep
        kl = kl_from_base(model, ids, mask)
        glp = gold_lp(model, gbatch)                      # gold-CoT likelihood after this round's updates
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
                         frac_correct=frac_correct(rew), gen_len=comp_len(mask),
                         adv_std=float(A_seq.std()), dead_frac=dead / max(len(pids), 1),
                         clip_frac=cnum / max(cden, 1.0), fmt=fmt, samp_lp=slp, gold_lp=glp,
                         optimizer_step_scope=optimizer_step_scope,
                         question_schedule_rng=question_schedule_rng,
                         proposal_prompt=proposal_prompt,
                         reward_requires_eos=reward_requires_eos,
                         natural_eos_fraction=float(natural_eos.mean()),
                         test_acc=tacc, loss=rloss / max(nstep, 1), kl=kl))
        if diagnostics_fn is not None:
            diagnostics_fn({
                "schema_version": 1,
                "method_family": "grpo",
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
                    "step_scope": optimizer_step_scope,
                    "gradient_steps_this_round": nstep,
                    "gradient_steps_cumulative": gsteps,
                    "backward_tokens_cumulative": backward_tokens,
                    "loss": rloss / max(nstep, 1),
                    "clip_fraction": cnum / max(cden, 1.0),
                    "kl": kl,
                },
                "signal": {
                    "advantage_std": float(A_seq.std()),
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
        log(f"  [GRPO G{G} r{t:>3}] gen={gen:>6} R={rew.mean():.3f} len={comp_len(mask):.0f} kl={kl:+.3f}")
    return recs
