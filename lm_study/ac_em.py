"""AC-EM -- answer-conditioned EM (Barber, Learning to Reason; the ACTIVE EM method).

GSM8K is "not really an RL problem ... we have a dataset with KNOWN answers" (§1.1): never score
with a verifier, CONDITION on the gold answer a* and search over latent reasoning h. Dense signal
p(a*|h), no oracle. Maximise log p(a*|q) >= Σ_s w_s log[p(a*|h^s)p(h^s)] (eq 2-4), delta posterior
over the full per-question history (eq 5). M-step (eq 9) always = weighted MLE on the
[reasoning ++ teacher-forced GOLD answer] span, weights normalised PER QUESTION (eq 10). Only the
WEIGHT varies -- the answer-conditioned "ways to calculate the expectation":
    weight="RW"   w_s ∝ p(a*|h^s)·p(h^s)^pi_temp   faithful eq 8 at pi_temp=1 (reward-form)
    weight="AW"   w_s ∝ exp(β·A_s)                 A_s = per-question standardised log p(a*|h^s)
    weight="AAW"  w_s ∝ p(h^s)^pi_temp·exp(β·A_s)  advantage × (tempered) policy factor
pi_temp (extension): TEMPERATURE POWER on the policy factor p(h^s|q,θ_old) -- 1=faithful eq 8, <1 tempers
the length/genericness bias behind weight-collapse, 0=pure answer-likelihood w ∝ p(a*|h). M-step objective
is the CONVEX blend (1-sup)·B_unsup + sup·B_sup (eq 14), with `sup` the SINGLE supervision-fraction knob in
[0,1] (the EM weight is its complement 1-sup); sup=1 = pure SFT baseline, sup=0 = plain unsupervised AC-EM.
anchor (extension): w *= (p_base/p_θ)^anchor. Deviation from eq 10: per-ROUND (not per-update)
weight recompute (tractable; θ barely moves over `iters` small steps). h^s = EXACT sampled reasoning
tokens (sliced, no decode round-trip); a* teacher-forced. length_norm=False keeps RW the true joint prob.

sup (the June-2026 §1.2 update, eq 11-14, now ON BY DEFAULT for the whole AC-* family): a SINGLE-STAGE
objective B_unsup + sup·B_sup, mixing the unsupervised answer-conditioned EM above with a SUPERVISED term
B_sup = SFT on GOLD reasoning trajectories (GSM8K's worked solutions, task.gold_solution). Both terms share one
M-step gradient (eq 14), not the usual SFT-then-RL two stages. There is a SINGLE supervision-fraction knob
`sup` in [0,1]; the EM weight is its complement 1-sup, so the M-step is (1-sup)·B_unsup + sup·B_sup. Default
sup=0.5 is the equal mix; --sup 0 = unsupervised ablation, --sup 1 = pure SFT (AC-SFT). Tasks without
gold_solution (e.g. IMDB) auto-disable the term and fall back to plain unsupervised AC-EM.

causal>0 -- Causal Variational Trace Optimisation (separate note): multiply the E-step weight by a causal
ablation reward exp(causal*ΔC), ΔC_s = log p(a*|h^s) - log p(a*|A(h^s)) with A = token-level number masking
(length-preserving, so no decode->retokenise artefact; scorer = θ_old, the round's frozen weights). This
rewards traces that LOSE answer-likelihood when their numbers are damaged -- causally useful, not merely
answer-correlated. The reward enters only the DETACHED weight, so the M-step is unchanged (doc §14).
Full-trace only; step-level (prefix / leave-one-step-out) deferred. No-op + zero overhead when causal=0.

hindsight=True -- Answer-Conditioned Trace Proposal (Extension 1, ρ=0): draw the candidate traces from
p_θ(h|q,a) instead of p_θ(h|q), by generating with a gold-answer hint in the prompt. ONLY generation is
hindsight -- the hint is stripped and the build loop re-anchors every training sequence on the question-
only prompt, so the E-step weights and the M-step (learning p_θ(h|q)) are byte-identical to on-policy.
Attacks trace DISCOVERY when the model rarely samples answer-reaching traces (frac_correct≈0); ρ=0 keeps
the existing per-question weights (no importance-sampling variance). Caveat: the proposal can leak/parrot
the answer -- pair with causal>0 (AC-CW-hs) to down-weight traces that don't survive number ablation.
"""
from __future__ import annotations
import numpy as np
import torch

from common import (DEV, MODEL_NAME, load_model, seq_logprobs, kl_from_base, _rounds, sample_round,
                    sample_multi, comp_len, frac_correct, ess_frac,
                    format_rate, mean_token_lp, build_gold_batch, build_gold_lists, gold_lp, maybe_eval)


def _round_diversity(texts, pid_row):
    """Mean over questions of (distinct completions / count) among THIS round's fresh samples: a cheap
    output-diversity metric, complementing the weight ESS. -> 1/G under full mode collapse, 1 if all
    distinct. Diagnostic only (never enters training)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for t_, p_ in zip(texts, pid_row):
        groups[int(p_)].append(t_)
    fr = [len(set(g)) / len(g) for g in groups.values() if g]
    return float(np.mean(fr)) if fr else float("nan")


def _within_q_corr(a, b, pids):
    """Mean over questions of corr(a_q, b_q). Weights are softmaxed PER QUESTION (eq 10), so only
    within-question associations are meaningful -- a global corr would be confounded by question
    difficulty. Skips questions with <2 samples or zero variance; nan if none remain. Diagnostic only."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cs = []
    for q in np.unique(pids):
        gi = np.where(pids == q)[0]
        if len(gi) < 2:
            continue
        aa, bb = a[gi], b[gi]
        if aa.std() < 1e-12 or bb.std() < 1e-12:
            continue
        cs.append(float(np.corrcoef(aa, bb)[0, 1]))
    return float(np.mean(cs)) if cs else float("nan")


def _reason_cut(text):
    """Char index where the sampled reasoning ends (model's own answer marker / run-on into the next
    few-shot block); keep h^s up to here, teacher-force the gold answer in place of the rest."""
    ci = len(text)
    for marker in ("####", "\nQuestion:", "\n\nQuestion:", "\nQ:"):
        i = text.find(marker)
        if i != -1:
            ci = min(ci, i)
    return ci


def _mask_number_tokens(tok, ids, mask_id, cache):
    """Length-preserving causal ablation A(h): replace every token whose decode contains a digit with
    `mask_id`. Token-level (not text) so clean and ablated traces share tokenisation -- the causal lift
    isolates the numbers with no decode->retokenise artefact, and the clean baseline reuses lp_ans."""
    out = []
    for tid in ids:
        hit = cache.get(tid)
        if hit is None:
            hit = cache[tid] = any(c.isdigit() for c in tok.decode([tid]))
        out.append(mask_id if hit else tid)
    return out


def _greedy(model, tok, prompts, max_new=8, bs=16):
    """Greedy continuations (left-padded, chunked) -- the decode-consistency check's verifier-free probe:
    does the model's OWN decoding from [q ++ h] reach the gold answer the E-step teacher-forced?"""
    model.eval()
    tok.padding_side = "left"
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(DEV)
            g = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
            out += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return out


def _hindsight_prompt(prompt, gold_answer):
    """Insert a gold-answer hint into the final question so traces are drawn from p_θ(h|q,a) (Extension 1).
    Used ONLY for generation -- the build loop strips it and re-anchors training on the question-only prompt."""
    if gold_answer is None:
        return prompt
    hint, cut = f" (the final answer is {gold_answer})", "\nAnswer:"
    i = prompt.rfind(cut)
    return prompt + hint if i == -1 else prompt[:i] + hint + prompt[i:]


def _sample_round_hindsight(model, tok, task, B, G, rng):
    """common.sample_round with ANSWER-CONDITIONED proposals (Extension 1, ρ=0): same prompt draw and
    return signature, but generate with the gold-answer hint. Calls the shared sample_multi unmodified;
    downstream h-extraction / weights / M-step are untouched (they re-anchor on the question-only prompt)."""
    n = len(task.prompts)
    pids = rng.choice(n, size=B // G, replace=False)
    pid_row = np.repeat(pids, G)
    prompts = [_hindsight_prompt(task.prompts[i], task.gold_answer[i]) for i in pid_row]
    ids, mask, texts = sample_multi(model, tok, prompts, max_new=getattr(task, "max_new", 40))
    rew = np.asarray(task.reward(texts, pid_row), dtype=np.float64)
    return pids, pid_row, ids, mask, rew, texts


def run_ac_em(task, rounds=40, B=64, G=4, seed=0, beta=2.0, iters=8, lr=1e-4, weight="RW",
               anchor=0.0, sup=0.5, pi_temp=1.0, causal=0.0, hindsight=False,
               window=0, train_span="both", select="soft", centered=False, dconsist=False, kl_coef=0.0,
               gate=0.0, sup_warmup=0, reset_every=0, frozen_critic=False, gold_in_buffer=False,
               model_name=MODEL_NAME, model_tok=None, micro=4, length_norm=False,
               eval_every=0, eval_fn=None, log=print):
    # ---- M-STEP REPAIR KNOBS (all default to the faithful eq-9 M-step; each reinstates ONE property the
    # knob screen showed the weighted-MLE M-step lacks vs RL/RAFT/DPO -- see
    # docs/experiments/ac_em_primer/ac_em_experiment_primer.tex:
    #   window      N>0: train only on the last N rounds' samples (fresh-only at 1). Fixes the STALE-HISTORY
    #               diet (hist_age~7 of 15): eq-5's full history compounds off-policyness every round.
    #   train_span  "both" (eq 9: h ++ a*) | "h" (imitate reasoning only, no conditional sharpening)
    #               | "ans" (answer head only, no reinforcement of sampled h). SPLITS the two damage
    #               vectors: a*-span sharpens p(a*|bad h) (gaming), h-span reinforces short/generic h.
    #   select      "soft" (draw ∝ w, eq 9) | "top1" (hard/Viterbi EM: per-question argmax only -- no
    #               averaging over garbage below the mode).
    #   centered    True: PG-form B_unsup on FRESH samples with per-question MEAN-CENTRED weights ->
    #               below-average traces get NEGATIVE gradient. Weighted MLE can only push up; this is
    #               the one structural property every working method (GRPO/RLOO/DPO) has and eq-9 lacks.
    #   dconsist    True: admit a fresh h to the history only if the model's own GREEDY decode from
    #               [q ++ h] yields a* -- closes the teacher-forcing/decoding gap directly (the
    #               pgold-up-accuracy-flat divergence trains on conditionals decoding never reaches).
    #   kl_coef     >0: sequence-level k3 KL penalty toward the FROZEN BASE inside the M-step loss (GRPO's
    #               trust region, which keeps its kl~0 while AC arms drift 0.3-1.7). NOTE the `anchor`
    #               knob only reweights samples in the E-step; it never constrains the update -- this does.
    #   gate        >0: per-question EVIDENCE GATE (SNIS validity floor). A question receives M-step mass
    #               this round only if its BEST sample's per-token answer prob exp(max_s lp_ans/|a*|) >= gate.
    #               Eq-10's softmax has no abstain: when ALL of a question's samples are garbage it still
    #               normalises to 1 and eq-9's 1/Q forces imitation of the least-bad garbage (RAFT survives
    #               via exactly this rejection bar). Gated questions get w=0; mass renormalises over the
    #               survivors; a round with no survivors skips the unsup term (like RAFT's skip).
    #   sup_warmup  N>0: run the FIRST N rounds at sup=1 (pure gold SFT), THEN drop to `sup`. A staged
    #               curriculum (STaR/ReST shape): make the proposer competent before trusting its own
    #               samples, so the E-step is fed evidence-bearing traces instead of garbage. The
    #               CONSTANT-blend sup axis only diluted (knob screen); a SCHEDULE changes what the E-step
    #               is fed. Needs gold_solution even when the final `sup` is 0.
    #   reset_every N>0: re-initialise the LoRA adapter + optimiser to base every N rounds, JUST BEFORE the
    #               M-step (verifier-free ReST-EM: sample with the improved policy, keep traces via the
    #               dconsist/gate filter, then IMPROVE FROM BASE). Removes drift accumulation structurally
    #               rather than braking it -- kl_coef only slows the cumulative walk; this resets it. Pair
    #               with larger `iters` so each from-base improve phase is substantial.
    #   frozen_critic True: score the REWARD term p(a*|h) under the FROZEN BASE model, not the current theta.
    #               L2R sets r(h)=p(a*|h,theta) -- ENDOGENOUS, so raising the objective moves the thing
    #               scoring it (the bootstrap behind the pgold-up / accuracy-down gaming). VRO keeps r
    #               EXOGENOUS. Freezing the reader to the base makes the weight r_base(a*|h)·p_theta(h)
    #               (exactly VRO eq-17), ungameable by construction -- the one VRO assumption L2R dropped.
    #               Pair with train_span="h" to train only the trace generator, never the answer head.
    #   gold_in_buffer True: L2R v5 Algorithm 1, step 3 -- inject each question's GOLD reasoning trace into
    #               its buffer (once), so the E-step weight is ALWAYS formed over a buffer containing a
    #               known-correct trace. The gold trace carries the highest honest p(a*|h)p(h), so it
    #               takes weight and the M-step pulls p(h|q) toward gold-like reasoning rather than the
    #               model's own gamed self-samples -- attacks evidence starvation and gaming at the root.
    #               Barber's own prescription. Use with window=0 (persistent buffer) and sup>0 (the
    #               B_sup + B'_sup additive form); gold is excluded from the fresh-sample diagnostics.
    # M-step objective (eq 14) is a CONVEX blend: (1-sup)·B_unsup + sup·B_sup. `sup` is the SINGLE
    # supervision-fraction knob in [0,1]; the EM weight is its complement unsup = 1-sup. B_sup (eq 11-12)
    # is SFT on the GOLD reasoning trajectory; B_unsup (eq 9) is the answer-conditioned EM term. Default
    # sup=0.5 = equal mix (the supervised term is ON for the whole AC-* family). sup=1 = pure SFT (AC-SFT);
    # sup=0 = plain unsupervised AC-EM; sup=0.75 = 3:1 supervised:EM.
    # pi_temp is a TEMPERATURE POWER on the policy factor p(h^s|q,θ_old) in the E-step weight (eq 8/10):
    # w ∝ p(a*|h)·p(h)^pi_temp. pi_temp=1 = faithful eq 8; <1 tempers the length/genericness bias that
    # drives weight-collapse; 0 = pure answer-likelihood weight w ∝ p(a*|h). (No-op for weight="AW".)
    # causal>0 tilts the weight by an ABLATION LIFT (CVTO): w *= exp(causal·[log p(a*|h) - log p(a*|A(h))]),
    # A = token-level number masking. Detached scorer (θ_old) -> M-step unchanged. Zero overhead if causal=0.
    # hindsight=True draws traces from p_θ(h|q,a) (answer-hinted generation, Extension 1 ρ=0); generation
    # only -- the hint is stripped so the weights + M-step are identical to on-policy. Zero change if False.
    assert weight in ("RW", "AW", "AAW"), f"unknown AC-EM weight {weight!r}"
    assert train_span in ("both", "h", "ans"), f"unknown train_span {train_span!r}"
    assert select in ("soft", "top1"), f"unknown select {select!r}"
    assert hasattr(task, "gold_answer"), "AC-EM needs a task with KNOWN gold answers (GSM8K-style)"
    assert 0.0 <= sup <= 1.0, f"AC-EM sup is a supervision FRACTION in [0,1], got {sup}"
    if (sup > 0 or sup_warmup > 0) and not hasattr(task, "gold_solution"):   # degrade (don't crash) without gold_solution
        log("[AC-EM] sup/sup_warmup set but task has no gold reasoning (gold_solution) -> supervised terms OFF")
        sup = 0.0; sup_warmup = 0
    unsup = 1.0 - sup                                       # COMPLEMENTARY: the EM weight is the complement of sup

    model, tok = (model_tok if model_tok is not None else load_model(seed=seed, model=model_name))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    _init_sd = ({n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                if reset_every > 0 else None)              # ReST: base-adapter snapshot (LoRA B=0 at init,
    def _reset_to_base():                                  # so restoring init == the base policy exactly)
        nonlocal opt
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(_init_sd[n])
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)   # clear momentum
    rng = np.random.default_rng(seed)
    eos = tok.eos_token_id
    wtag = (("SFT" if unsup == 0 else f"AC-{weight}") + ("+a" if anchor > 0 else "")
            + ("+sup" if sup > 0 and unsup != 0 else "") + (f"+pt{pi_temp:g}" if pi_temp != 1.0 else "")
            + (f"+c{causal:g}" if causal > 0 else "") + ("+hs" if hindsight else "")
            + (f"+w{window}" if window > 0 else "") + (f"+{train_span}" if train_span != "both" else "")
            + ("+top1" if select == "top1" else "") + ("+pg" if centered else "")
            + ("+dc" if dconsist else "") + (f"+kl{kl_coef:g}" if kl_coef > 0 else "")
            + (f"+g{gate:g}" if gate > 0 else "") + (f"+wu{sup_warmup}" if sup_warmup > 0 else "")
            + (f"+rst{reset_every}" if reset_every > 0 else "") + ("+fc" if frozen_critic else "")
            + ("+gib" if gold_in_buffer else ""))
    pad = torch.nn.utils.rnn.pad_sequence
    G_ids, G_span = [], []                                 # GOLD reasoning trajectories [q++gold_solution] (eq 11)
    if sup > 0 or sup_warmup > 0:                          # curriculum needs gold_solution even when final sup is 0
        G_ids, G_span = build_gold_lists(tok, task)        # shared builder (also behind the gold_lp probe)
    H_ids, H_span, H_ans, H_pid = [], [], [], []          # [q++h++a*] ids; reasoning+answer mask; answer mask; pid
    H_round = []                                          # round-of-origin per item (hist_age diagnostic)
    H_abl_ids = []                                        # number-masked traces A(h) for the causal lift (causal>0)
    mask_id = (tok("#", add_special_tokens=False).input_ids or [eos])[0]
    dig_cache = {}                                        # token-id -> "decodes to something containing a digit"
    # ---- Algorithm-1 gold buffer entries (step 3): each question's [q++gold_solution++a*] in the same
    # structure as a sampled trace, built once, injected into the history when gold_in_buffer. ----
    Gbuf_ids, Gbuf_span, Gbuf_ans, Gbuf_abl, gold_injected = None, None, None, None, set()
    if gold_in_buffer and not hasattr(task, "gold_solution"):
        log("[AC-EM] gold_in_buffer needs gold_solution -> OFF"); gold_in_buffer = False
    if gold_in_buffer:
        Gbuf_ids, Gbuf_span, Gbuf_ans, Gbuf_abl = [], [], [], []
        for pid in range(len(task.prompts)):
            gold_answer = task.gold_answer[pid]
            if gold_answer is None:                        # unparseable answer -> no gold entry for this q
                Gbuf_ids.append(None); Gbuf_span.append(None); Gbuf_ans.append(None); Gbuf_abl.append(None)
                continue
            p_ids = tok(task.prompts[pid], return_tensors="pt").input_ids[0].tolist()
            gold_reasoning = task.gold_solution[pid].split("####")[0].rstrip()   # gold reasoning, marker stripped
            h_ids = tok(" " + gold_reasoning, add_special_tokens=False).input_ids
            a_ids = tok(f"\n#### {gold_answer}", add_special_tokens=False).input_ids    # a* teacher-forced (matches samples)
            full = torch.tensor(p_ids + h_ids + a_ids, dtype=torch.long)
            span = torch.zeros(len(full), dtype=torch.bool); span[len(p_ids):] = True               # h ++ a*
            ans = torch.zeros(len(full), dtype=torch.bool); ans[len(p_ids) + len(h_ids):] = True     # a* | h
            Gbuf_ids.append(full); Gbuf_span.append(span); Gbuf_ans.append(ans)
            Gbuf_abl.append(torch.tensor(p_ids + _mask_number_tokens(tok, h_ids, mask_id, dig_cache) + a_ids,
                                         dtype=torch.long) if causal > 0 else None)
    gbatch = build_gold_batch(tok, task)                  # gold_solution batch for the gold_lp diagnostic (None if no gold)
    recs, oracle, gen, gsteps = [], 0, 0, 0               # oracle stays 0: AC-EM uses the KNOWN answer, no verifier
    for t in _rounds(rounds):
        sampler = _sample_round_hindsight if hindsight else sample_round   # Extension 1: answer-conditioned proposal
        pids, pid_row, ids, comp_mask, rew, texts = sampler(model, tok, task, B, G, rng)
        gen += B                                           # completions generated (oracle calls remain 0)
        slp = mean_token_lp(model, ids, comp_mask)         # policy-entropy proxy at θ_old on its OWN fresh samples
        fmt = format_rate(texts)                           # fraction with a parseable '####' answer marker
        if window > 0 and H_ids:                           # windowed history: drop samples older than `window` rounds
            keep = [i for i, r0 in enumerate(H_round) if r0 > t - window]   # window=1 -> fresh-only M-step
            H_ids = [H_ids[i] for i in keep]; H_span = [H_span[i] for i in keep]
            H_ans = [H_ans[i] for i in keep]; H_pid = [H_pid[i] for i in keep]
            H_round = [H_round[i] for i in keep]
            if causal > 0:
                H_abl_ids = [H_abl_ids[i] for i in keep]
        if gold_in_buffer:                                 # Alg.1 step 3: seed each question's buffer with its
            for pid in map(int, np.unique(pid_row)):       # GOLD trace (once; persists with window=0). Injected
                if pid in gold_injected or Gbuf_ids[pid] is None:   # BEFORE fresh0 -> it is history, not a fresh
                    continue                                        # sample, so it never enters the fresh diagnostics
                gold_injected.add(pid)                              # or the dconsist filter (which act on >= fresh0)
                H_ids.append(Gbuf_ids[pid]); H_span.append(Gbuf_span[pid]); H_ans.append(Gbuf_ans[pid])
                H_pid.append(pid); H_round.append(t)
                if causal > 0:
                    H_abl_ids.append(Gbuf_abl[pid])
        F_dc = []                                          # dconsist probes: (decode prompt [q++h++'\n####'], pid)
        F_rew = []                                         # verifier reward per APPENDED fresh row (1:1 with the
        fresh0 = len(H_ids)                                # fresh history slice, surviving any dconsist filter)
        for b in range(len(pid_row)):                      # build [q ++ reasoning ++ GOLD answer]
            pid = int(pid_row[b]); gold_answer = task.gold_answer[pid]
            if gold_answer is None:
                continue
            comp = ids[b][comp_mask[b]].tolist()           # EXACT sampled completion ids (reasoning + answer)
            if comp and comp[-1] == eos:
                comp = comp[:-1]
            ci = _reason_cut(texts[b])
            if ci >= len(texts[b]):
                h_ids = comp                               # no answer stated -> all of it is reasoning
            else:                                          # largest token-prefix whose decode stays before ci
                lo, hi = 0, len(comp)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if len(tok.decode(comp[:mid])) <= ci:
                        lo = mid
                    else:
                        hi = mid - 1
                h_ids = comp[:lo]
            p_ids = tok(task.prompts[pid], return_tensors="pt").input_ids[0].tolist()
            a_ids = tok(f"\n#### {gold_answer}", add_special_tokens=False).input_ids   # gold_answer (teacher-forced)
            full = torch.tensor(p_ids + h_ids + a_ids, dtype=torch.long)
            span = torch.zeros(len(full), dtype=torch.bool); span[len(p_ids):] = True              # h ++ a*
            ans = torch.zeros(len(full), dtype=torch.bool); ans[len(p_ids) + len(h_ids):] = True   # a* | h
            H_ids.append(full); H_span.append(span); H_ans.append(ans); H_pid.append(pid); H_round.append(t)
            F_rew.append(float(rew[b]))                     # diagnostic only (ans_acc_corr); never trains
            if causal > 0:                                  # A(h): same-length number-masked trace (reuses H_ans mask)
                mh = _mask_number_tokens(tok, h_ids, mask_id, dig_cache)
                H_abl_ids.append(torch.tensor(p_ids + mh + a_ids, dtype=torch.long))
            if dconsist:                                    # decode probe: greedy answer from [q ++ h] (marker forced)
                F_dc.append((task.prompts[pid] + texts[b][:ci] + "\n####", pid))
        if dconsist and len(H_ids) > fresh0:                # decode-consistency filter: keep a fresh h only if the
            conts = _greedy(model, tok, [p for p, _ in F_dc])   # model's OWN greedy decode from it reaches a*
            ok = [float(task.reward(["####" + c], [p])[0]) > 0.5 for c, (_, p) in zip(conts, F_dc)]
            keep = list(range(fresh0)) + [fresh0 + j for j, k in enumerate(ok) if k]
            H_ids = [H_ids[i] for i in keep]; H_span = [H_span[i] for i in keep]
            H_ans = [H_ans[i] for i in keep]; H_pid = [H_pid[i] for i in keep]
            H_round = [H_round[i] for i in keep]
            F_rew = [F_rew[j] for j, k in enumerate(ok) if k]   # keep the reward diagnostic aligned 1:1
            if causal > 0:
                H_abl_ids = [H_abl_ids[i] for i in keep]
        H = len(H_ids)
        if H == 0:
            continue
        all_ids = pad(H_ids, batch_first=True, padding_value=tok.eos_token_id).to(DEV)
        all_span = pad(H_span, batch_first=True, padding_value=False).to(DEV)
        pid_arr = np.asarray(H_pid)
        # ---- E-step: per-question weights (eq 10); only the WEIGHT FORM differs, M-step target is fixed ----
        with torch.no_grad():                              # everything here is at θ_old (this round's weights)
            lp_joint = seq_logprobs(model, all_ids, all_span, micro=16, length_norm=length_norm)  # log p(h,a*)
            all_ans = pad(H_ans, batch_first=True, padding_value=False).to(DEV)
            lp_ans = seq_logprobs(model, all_ids, all_ans, micro=16, length_norm=length_norm)     # log p(a*|h)
            lp_ans_abl = None
            if causal > 0:                                  # log p(a*|A(h)): answer-likelihood with numbers masked
                all_abl = pad(H_abl_ids, batch_first=True, padding_value=tok.eos_token_id).to(DEV)
                lp_ans_abl = seq_logprobs(model, all_abl, all_ans, micro=16, length_norm=length_norm)
        lp_pol = lp_joint - lp_ans                         # log p(h|q): the policy factor (from CURRENT theta)
        lp_ans_rew = lp_ans                                # the REWARD term p(a*|h). frozen_critic pins it to the
        if frozen_critic:                                  # BASE reader so theta cannot game its own reward: the
            with torch.no_grad(), model.disable_adapter():  # weight becomes r_base(a*|h)·p_theta(h) = VRO eq-17 with
                lp_ans_rew = seq_logprobs(model, all_ids, all_ans, micro=16, length_norm=length_norm)  # EXOGENOUS r
        if weight == "RW":                                 # eq 8 (pi_temp=1): w ∝ p(a*|h)·p(h)^pi_temp
            logits = lp_ans_rew + pi_temp * lp_pol
        else:                                              # advantage of log p(a*|h), standardised per question
            g = lp_ans_rew.detach().cpu().numpy()
            A = np.zeros_like(g)
            for q in np.unique(pid_arr):
                gi = np.where(pid_arr == q)[0]
                gg = g[gi]; A[gi] = (gg - gg.mean()) / (gg.std() + 1e-8)
            logits = torch.tensor(beta * A, device=DEV, dtype=torch.float32)   # AW
            if weight == "AAW":                            # × tempered policy factor p(h)^pi_temp
                logits = logits + pi_temp * lp_pol
        if anchor > 0:                                     # variational prior: w *= (p_base/p_θ)^anchor
            with torch.no_grad(), model.disable_adapter():
                lp_base = seq_logprobs(model, all_ids, all_span, micro=16, length_norm=length_norm)
            logits = logits + anchor * (lp_base - lp_joint).clamp(-10, 10)
        cscore = float("nan")
        if causal > 0:                                     # CVTO causal tilt: w *= exp(causal·ΔC), ΔC = ablation lift
            C = (lp_ans - lp_ans_abl).clamp(-10, 10)       # clip for stability (doc §8); trace-dependent via A(h)
            logits = logits + causal * C
            cscore = float(C.mean())                        # mean ablation lift (doc metric: avg causal score)
        lp_ans_n = None
        if gate > 0:                                       # per-token answer prob for the evidence gate (frozen
            lp_ans_n = lp_ans_rew if length_norm else lp_ans_rew / all_ans[:, 1:].sum(1).clamp(min=1)  # reward if fc
        w = torch.zeros_like(logits)
        alive = 0
        for q in np.unique(pid_arr):                       # eq 10: normalise over EACH question's S samples
            gi = np.where(pid_arr == q)[0]
            if gate > 0 and float(lp_ans_n[gi].max().exp()) < gate:
                continue                                   # evidence gate: no sample makes a* remotely likely
            w[gi] = torch.softmax(logits[gi], 0); alive += 1
        nq = len(np.unique(pid_arr))
        gate_frac = (alive / nq if gate > 0 else float("nan"))   # fraction of questions passing the gate
        Q = max(alive, 1)                                  # uniform over SURVIVING questions (the 1/M in eq 9)
        wn = (w.double() / Q).cpu().numpy()
        wn = wn / wn.sum() if wn.sum() > 0 else wn         # all-gated round: zeros (unsup term skips below)
        # ---- M-step target span + repair-mode precomputation (the E-step weights above are untouched) ----
        tspan = all_span if train_span == "both" else (all_span & ~all_ans if train_span == "h" else all_ans)
        lp_base_n = None
        if kl_coef > 0:                                    # frozen-base per-token lp on the SAME trained span
            with torch.no_grad(), model.disable_adapter():
                lp_base_n = seq_logprobs(model, all_ids, tspan, micro=16, length_norm=True)
        if select == "top1":                               # hard/Viterbi EM: each question's argmax-w sample only
            w_np = w.detach().cpu().numpy()
            top_idx = []
            for q in np.unique(pid_arr):
                gi = np.where(pid_arr == q)[0]
                if w_np[gi].sum() <= 0:                    # evidence-gated question: no mode to select
                    continue
                top_idx.append(int(gi[np.argmax(w_np[gi])]))
            top_idx = np.asarray(top_idx)
        cw = None
        if centered:                                       # PG-form: per-question MEAN-CENTRED weights over the
            cw = torch.zeros_like(w)                       # FRESH samples (on-policy) -> negative gradients exist
            for q in np.unique(pid_arr[fresh0:]):
                gi = np.where(pid_arr == q)[0]; gi = gi[gi >= fresh0]
                if len(gi):
                    cw[gi] = torch.softmax(logits[gi], 0) - 1.0 / len(gi)
        # ---- M-step: ascend unsup·B_unsup + sup·B_sup via `iters` minibatches (eq 9, 14 + repair modes) ----
        sup_t = 1.0 if t < sup_warmup else sup             # curriculum: pure gold-SFT for the first wu rounds
        unsup_t = 1.0 - sup_t
        if reset_every > 0 and t % reset_every == 0:       # ReST: this round's M-step IMPROVES FROM BASE
            _reset_to_base()                               # (weights wn came from the improved policy above)
        m = min(micro, H)
        model.train()
        rloss = sloss = 0.0
        for _ in range(iters):
            loss = torch.zeros((), device=DEV)
            if unsup_t != 0:                               # unsup·B_unsup: eq-9 weighted MLE, or a repair form
                if centered:                               # uniform draw over fresh; weight INSIDE the loss (signed)
                    nf = H - fresh0
                    sel_np = rng.choice(np.arange(fresh0, H), size=min(m, nf), replace=True) if nf else None
                elif select == "top1":
                    sel_np = (rng.choice(top_idx, size=min(m, len(top_idx)), replace=True)
                              if len(top_idx) else None)   # every question gated -> nothing to select
                else:                                      # faithful eq 9: draw ∝ w
                    sel_np = (rng.choice(H, size=m, replace=True, p=wn)
                              if wn.sum() > 0 else None)   # every question gated -> skip the unsup term
                if sel_np is not None:
                    sel = torch.as_tensor(sel_np, dtype=torch.long, device=DEV)
                    lp_g = seq_logprobs(model, all_ids[sel], tspan[sel], grad=True, length_norm=length_norm)
                    if centered:                           # below-average traces get pushed DOWN (MLE cannot)
                        loss = loss + unsup_t * (-(cw[sel] * lp_g).sum() / len(sel))
                    else:
                        loss = loss + unsup_t * (-lp_g.mean())
                    if kl_coef > 0:                        # seq-level k3 trust region toward the frozen base
                        lp_n = lp_g if length_norm else lp_g / tspan[sel].sum(1).clamp(min=1)
                        r = (lp_base_n[sel] - lp_n).clamp(-5, 5)
                        loss = loss + kl_coef * (r.exp() - r - 1).mean()
            if sup_t > 0:                                  # + sup·B_sup: SFT on gold reasoning (eq 11-12, 14)
                gsel = rng.choice(len(G_ids), size=min(micro, len(G_ids)), replace=False)
                g_ids = pad([G_ids[i] for i in gsel], batch_first=True, padding_value=eos).to(DEV)
                g_msk = pad([G_span[i] for i in gsel], batch_first=True, padding_value=False).to(DEV)
                ls = -seq_logprobs(model, g_ids, g_msk, grad=True, length_norm=length_norm).mean()
                loss = loss + sup_t * ls; sloss += float(sup_t * ls)
            if loss.requires_grad:                         # (a filtered round can leave the centered arm empty)
                opt.zero_grad(); loss.backward(); opt.step(); gsteps += 1
            rloss += float(loss)
        # ---- diagnostics: dense signal p(a*|h) on fresh samples + acc + weight-ESS + gen length ----
        pgold = float("nan")
        if len(H_ids) > fresh0:
            with torch.no_grad():
                fa_ids = pad(H_ids[fresh0:], batch_first=True, padding_value=tok.eos_token_id).to(DEV)
                fa_ans = pad(H_ans[fresh0:], batch_first=True, padding_value=False).to(DEV)
                pgold = float(seq_logprobs(model, fa_ids, fa_ans, micro=16, length_norm=True).exp().mean())
        acc = float(rew.mean()) - getattr(task, "floor", 0.0)   # sampled-answer accuracy (NOT used in training)
        ess = ess_frac(wn)                                 # weight concentration over the history (collapse->1/H)
        kl = kl_from_base(model, ids, comp_mask)
        div = _round_diversity(texts, pid_row)             # output diversity of this round's samples (collapse->1/G)
        # E-step evidence (both within-question; see _within_q_corr): does the weight prefer SHORT h
        # (w_len_corr<0 = the no-length-norm collapse critique), and is the dense signal p(a*|h) actually
        # informative of sampled correctness (ans_acc_corr = calibration of the E-step signal)?
        wlc = _within_q_corr(w.detach().cpu().numpy(), all_span.sum(1).float().cpu().numpy(), pid_arr)
        aac = _within_q_corr(lp_ans.detach().cpu().numpy()[fresh0:], np.asarray(F_rew), pid_arr[fresh0:])
        # F_rew tracks the appended fresh rows 1:1 through gold_answer-None skips AND the dconsist filter -- the
        # old guard (len(pid_arr)-fresh0 == len(rew)) NaN'd this calibration diagnostic on exactly the
        # dc arms, where whether the E-step still ranks correct traces higher is the question.
        glp = gold_lp(model, gbatch)                       # gold_solution likelihood AFTER this round's M-step:
                                                           # toward-gold vs sharpen-own-reasoning, per round
        hage = (float(t - np.dot(wn, np.asarray(H_round, dtype=np.float64)))   # weighted mean AGE (rounds) of the
                if wn.sum() > 0 else float("nan"))         # M-step's history mass; nan on an all-gated round
        tacc = maybe_eval(model, t, rounds, eval_every, eval_fn)   # opt-in held-out trajectory (--eval-every)
        recs.append(dict(round=t, oracle=oracle, gen=gen, gsteps=gsteps, mean_reward=acc,
                         frac_correct=frac_correct(rew), gen_len=comp_len(comp_mask), ess=ess, div=div,
                         w_len_corr=wlc, ans_acc_corr=aac, fmt=fmt, samp_lp=slp, gold_lp=glp, hist_age=hage,
                         gate_frac=gate_frac,              # fraction of questions with admissible evidence (gate>0)
                         pgold=pgold, cscore=cscore, loss=rloss / max(iters, 1), test_acc=tacc,
                         sup_t=sup_t,                       # effective supervision fraction THIS round (curriculum)
                         sup_loss=(sloss / max(iters, 1) if sup_t > 0 else float("nan")), kl=kl, hist=H))
        log(f"  [{wtag} G{G} r{t:>3}] gen={gen:>6} p(a*|h)={pgold:.3f} acc={acc:.3f} ess={ess:.2f}"
            f"{f' C={cscore:+.2f}' if causal > 0 else ''} len={comp_len(comp_mask):.0f} kl={kl:+.3f} H={H}")
    return recs
