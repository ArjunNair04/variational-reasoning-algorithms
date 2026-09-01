"""Online rejection SFT (historical method label: RAFT).

Sample grouped completions from the current policy, retain only answer-correct
traces, cap accepted traces per question, and apply next-token SFT. This is the
minimal answer-only SFT comparator for verifiable reasoning.
"""
from __future__ import annotations
import numpy as np
import torch

from common import (MODEL_NAME, QuestionSampler, matched_question_rng,
                    load_model, seq_logprobs,
                    kl_from_base, _rounds, sample_round, comp_len, frac_correct,
                    format_rate, task_format_rate, mean_token_lp, build_gold_batch, gold_lp,
                    maybe_eval, natural_eos_mask)


def select_answer_correct_indices(
    pids,
    pid_row,
    rewards,
    *,
    per_question_limit,
    reward_floor,
    reward_best,
):
    """Return a balanced, per-question-capped answer-correct SFT subset."""

    threshold = reward_floor + 0.5 * (reward_best - reward_floor)
    selected = []
    for pid in pids:
        local = np.where(pid_row == pid)[0]
        correct = local[rewards[local] > threshold]
        selected.extend(correct[:per_question_limit])
    return np.asarray(selected, dtype=int)


def run_raft(task, rounds=40, B=64, G=4, seed=0, lr=1e-4, top_frac=0.25, epochs=1,
                model_name=MODEL_NAME, model_tok=None, micro=4, eval_every=0,
                eval_fn=None, eval_rounds=None, diagnostics_fn=None,
                checkpoint_every=0, checkpoint_fn=None,
                question_sampling="random", proposal_prompt="question",
                reward_requires_eos=False, log=print):
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if not 0 < top_frac <= 1:
        raise ValueError(f"top_frac must be in (0,1], got {top_frac}")
    if checkpoint_every < 0:
        raise ValueError(f"checkpoint_every must be nonnegative, got {checkpoint_every}")
    model, tok = (model_tok if model_tok is not None else load_model(seed=seed, model=model_name))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed); recs, oracle, gen, gsteps = [], 0, 0, 0   # oracle == gen (all scored)
    question_sampler = QuestionSampler(
        range(len(task.prompts)),
        matched_question_rng(rng, seed, question_sampling),
        mode=question_sampling,
    )
    generated_tokens = backward_tokens = question_exposures = 0
    unique_questions = set()
    gbatch = build_gold_batch(tok, task)                  # gold-CoT batch for the gold_lp diagnostic
    kp = max(1, int(top_frac * G))                        # per-prompt accepts (best-of-n)
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
        slp = mean_token_lp(model, ids, mask)             # policy-entropy proxy at θ_old on its OWN samples
        reward_floor = float(getattr(task, "floor", 0.0))
        reward_best = float(getattr(task, "best", 1.0))
        sel = select_answer_correct_indices(
            pids,
            pid_row,
            rew,
            per_question_limit=kp,
            reward_floor=reward_floor,
            reward_best=reward_best,
        )
        diag = dict(
            fmt=fmt,
            samp_lp=slp,
            proposal_prompt=proposal_prompt,
            reward_requires_eos=reward_requires_eos,
            natural_eos_fraction=float(natural_eos.mean()),
        )                                                # shared diagnostics for both branches below
        if sel.size == 0:                                 # nothing cleared the bar -> skip (no garbage SFT)
            tacc = maybe_eval(
                model, t, rounds, eval_every, eval_fn, eval_rounds=eval_rounds
            )
            recs.append(dict(round=t, oracle=oracle, verifier_calls=oracle,
                             gen=gen, llm_gen=gen,
                             generated_tokens=generated_tokens,
                             backward_tokens=backward_tokens,
                             question_exposures=question_exposures,
                             unique_questions_seen=len(unique_questions),
                             questions_this_round=len(pids),
                             gsteps=gsteps, mean_reward=float(rew.mean()),
                             frac_correct=frac_correct(rew), gen_len=comp_len(mask), n_accept=0,
                             gold_lp=gold_lp(model, gbatch), test_acc=tacc, **diag,
                             loss=float("nan"), kl=kl_from_base(model, ids, mask)))
            if diagnostics_fn is not None:
                diagnostics_fn({
                    "schema_version": 1,
                    "method_family": "online_rejection_sft",
                    "round": t,
                    "completed_rounds": t + 1,
                    "generation": {
                        "generations_this_round": B,
                        "generations_cumulative": gen,
                        "generated_tokens_cumulative": generated_tokens,
                        "questions_this_round": len(pids),
                        "question_exposures_cumulative": question_exposures,
                        "unique_questions_seen": len(unique_questions),
                        "correct_fraction": frac_correct(rew),
                        "format_fraction": fmt,
                        "natural_eos_fraction": float(natural_eos.mean()),
                        "accepted_traces": 0,
                    },
                    "optimizer": {
                        "gradient_steps_this_round": 0,
                        "gradient_steps_cumulative": gsteps,
                        "backward_tokens_cumulative": backward_tokens,
                        "loss": None,
                        "kl": recs[-1]["kl"],
                    },
                    "reward": {
                        "requires_natural_eos": bool(reward_requires_eos),
                    },
                    "test_acc": None if not np.isfinite(tacc) else tacc,
                })
            log(f"  [RAFT r{t:>3}] gen={gen:>6} R={rew.mean():.3f} accept=0 skip"); continue
        s_ids, s_mask = ids[sel.copy()], mask[sel.copy()]
        k = len(sel)
        model.train()
        rloss = 0.0
        for _ in range(epochs):                           # micro = memory only: accumulate, ONE SFT step
            opt.zero_grad()
            for i in range(0, k, micro):
                lp = seq_logprobs(model, s_ids[i:i + micro], s_mask[i:i + micro], grad=True, length_norm=True)
                loss = -lp.sum() / k                       # max Σ log π_θ(o|q) over the accepted set
                loss.backward()
                backward_tokens += int(s_mask[i:i + micro].sum())
                rloss += float(loss)
            opt.step(); gsteps += 1
        kl = kl_from_base(model, ids, mask)
        glp = gold_lp(model, gbatch)                      # gold-CoT likelihood after this round's SFT step(s)
        tacc = maybe_eval(
            model, t, rounds, eval_every, eval_fn, eval_rounds=eval_rounds
        )
        recs.append(dict(round=t, oracle=oracle, verifier_calls=oracle,
                         gen=gen, llm_gen=gen,
                         generated_tokens=generated_tokens,
                         backward_tokens=backward_tokens,
                         question_exposures=question_exposures,
                         unique_questions_seen=len(unique_questions),
                         questions_this_round=len(pids),
                         gsteps=gsteps, mean_reward=float(rew.mean()),
                         frac_correct=frac_correct(rew), gen_len=comp_len(mask), n_accept=k,
                         gold_lp=glp, test_acc=tacc, **diag,
                         loss=rloss / max(epochs, 1), kl=kl))
        if diagnostics_fn is not None:
            diagnostics_fn({
                "schema_version": 1,
                "method_family": "online_rejection_sft",
                "round": t,
                "completed_rounds": t + 1,
                "generation": {
                    "generations_this_round": B,
                    "generations_cumulative": gen,
                    "generated_tokens_cumulative": generated_tokens,
                    "questions_this_round": len(pids),
                    "question_exposures_cumulative": question_exposures,
                    "unique_questions_seen": len(unique_questions),
                    "correct_fraction": frac_correct(rew),
                    "format_fraction": fmt,
                    "natural_eos_fraction": float(natural_eos.mean()),
                    "accepted_traces": k,
                },
                "optimizer": {
                    "gradient_steps_this_round": epochs,
                    "gradient_steps_cumulative": gsteps,
                    "backward_tokens_cumulative": backward_tokens,
                    "loss": rloss / max(epochs, 1),
                    "kl": kl,
                },
                "reward": {
                    "requires_natural_eos": bool(reward_requires_eos),
                },
                "test_acc": None if not np.isfinite(tacc) else tacc,
            })
        if (checkpoint_fn is not None and checkpoint_every > 0
                and (t + 1) % checkpoint_every == 0 and (t + 1) < rounds):
            checkpoint_fn(model, t + 1)
        log(f"  [RAFT r{t:>3}] gen={gen:>6} R={rew.mean():.3f} accept={k} len={comp_len(mask):.0f} kl={kl:+.3f}")
    return recs
