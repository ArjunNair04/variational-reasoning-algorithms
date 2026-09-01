"""Conditional Variational Reward Optimisation for verifier-guided LM fine-tuning.

This is the finite-support, per-question LLM form of VRO equations 58--62
(the conditional counterpart of generic equations 17--18):

    q_i(h | question) proportional to reward_i * p_theta(h | question)
    theta_new = argmax_theta sum_i q_i log p_theta(h_i | question)

The faithful arm uses full-sequence log probabilities, all unique traces seen so far,
and no KL term. Named controls in ``run_sweep_lm.py`` alter one mechanism at a time.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch

from common import (
    DEV,
    MODEL_NAME,
    _rounds,
    format_rate,
    kl_from_base,
    load_model,
    maybe_eval,
    sample_round,
    seq_logprobs,
)


@dataclass
class VROTrace:
    ids: torch.Tensor
    comp_mask: torch.Tensor
    completion_key: tuple[int, ...]
    text: str
    reward: float
    pid: int
    round_added: int

    @property
    def length(self) -> int:
        return int(self.comp_mask.sum())


def conditional_responsibilities(
    rewards: torch.Tensor,
    joint_logps: torch.Tensor,
    mean_logps: torch.Tensor,
    pids: torch.Tensor,
    *,
    policy_factor: bool = True,
    responsibility_score: str = "joint",
) -> torch.Tensor:
    """Return independently normalised responsibilities for every question."""
    if responsibility_score not in {"joint", "token_mean"}:
        raise ValueError(
            "responsibility_score must be 'joint' or 'token_mean', "
            f"got {responsibility_score!r}"
        )
    if rewards.ndim != 1 or joint_logps.shape != rewards.shape:
        raise ValueError("rewards and log-probability vectors must be one-dimensional and aligned")
    if mean_logps.shape != rewards.shape or pids.shape != rewards.shape:
        raise ValueError("mean_logps and pids must align with rewards")
    if not bool(torch.isfinite(rewards).all()) or bool((rewards <= 0).any()):
        raise ValueError("VRO requires finite, strictly positive rewards")
    if not bool(torch.isfinite(joint_logps).all()) or not bool(torch.isfinite(mean_logps).all()):
        raise ValueError("VRO requires finite sequence log probabilities")

    logits = torch.log(rewards)
    if policy_factor:
        logits = logits + (joint_logps if responsibility_score == "joint" else mean_logps)

    weights = torch.empty_like(logits)
    for pid in torch.unique(pids, sorted=True):
        mask = pids == pid
        weights[mask] = torch.softmax(logits[mask], dim=0)
    return weights


def conditional_bound(
    weights: torch.Tensor,
    rewards: torch.Tensor,
    joint_logps: torch.Tensor,
    pids: torch.Tensor,
) -> torch.Tensor:
    """Average finite-support VRO bound, one equally weighted term per question."""
    terms = []
    for pid in torch.unique(pids, sorted=True):
        mask = pids == pid
        w = weights[mask]
        terms.append(
            -(w * torch.log(w.clamp_min(torch.finfo(w.dtype).tiny))).sum()
            + (w * (torch.log(rewards[mask]) + joint_logps[mask])).sum()
        )
    if not terms:
        raise ValueError("cannot evaluate the VRO bound on an empty support")
    return torch.stack(terms).mean()


def conditional_mstep_objective(
    weights: torch.Tensor,
    joint_logps: torch.Tensor,
    pids: torch.Tensor,
) -> torch.Tensor:
    """Average weighted sequence log-likelihood used by the VRO M-step."""
    terms = []
    for pid in torch.unique(pids, sorted=True):
        mask = pids == pid
        terms.append((weights[mask] * joint_logps[mask]).sum())
    if not terms:
        raise ValueError("cannot evaluate the VRO M-step on an empty support")
    return torch.stack(terms).mean()


def conditional_support_log_evidence(
    rewards: torch.Tensor,
    joint_logps: torch.Tensor,
    pids: torch.Tensor,
) -> torch.Tensor:
    """Average log reward mass on the observed support (VRO equation 24 per question)."""
    terms = []
    for pid in torch.unique(pids, sorted=True):
        mask = pids == pid
        terms.append(torch.logsumexp(torch.log(rewards[mask]) + joint_logps[mask], dim=0))
    if not terms:
        raise ValueError("cannot evaluate support evidence on an empty support")
    return torch.stack(terms).mean()


def draw_conditional_mstep_indices(
    weights: torch.Tensor,
    pids: torch.Tensor,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw equally many questions, then a completion from that question according to q."""
    if n_draws < 1:
        raise ValueError(f"n_draws must be positive, got {n_draws}")
    unique = np.asarray(sorted(int(pid) for pid in torch.unique(pids).tolist()), dtype=np.int64)
    if not len(unique):
        raise ValueError("cannot draw from an empty conditional support")

    if n_draws < len(unique):
        prompt_draws = rng.choice(unique, size=n_draws, replace=False)
    else:
        repeats, extra = divmod(n_draws, len(unique))
        prompt_draws = np.repeat(unique, repeats)
        if extra:
            prompt_draws = np.concatenate(
                [prompt_draws, rng.choice(unique, size=extra, replace=False)]
            )
    rng.shuffle(prompt_draws)

    weights_cpu = weights.detach().double().cpu().numpy()
    pids_cpu = pids.detach().cpu().numpy()
    selected = []
    for pid in prompt_draws:
        candidates = np.flatnonzero(pids_cpu == pid)
        probs = weights_cpu[candidates]
        probs = probs / probs.sum()
        selected.append(int(rng.choice(candidates, p=probs)))
    return np.asarray(selected, dtype=np.int64)


def _pad_traces(
    traces: Iterable[VROTrace],
    pad_token_id: int,
    *,
    device: torch.device | str = DEV,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = list(traces)
    if not rows:
        raise ValueError("cannot pad an empty trace list")
    ids = torch.nn.utils.rnn.pad_sequence(
        [row.ids for row in rows], batch_first=True, padding_value=pad_token_id
    )
    masks = torch.nn.utils.rnn.pad_sequence(
        [row.comp_mask for row in rows], batch_first=True, padding_value=False
    )
    return ids.to(device), masks.to(device)


def _make_trace(
    prompt_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    generated_mask: torch.Tensor,
    *,
    text: str,
    reward: float,
    pid: int,
    round_added: int,
) -> VROTrace:
    completion_ids = generated_ids[generated_mask].detach().cpu()
    ids = torch.cat([prompt_ids.detach().cpu(), completion_ids])
    mask = torch.zeros(len(ids), dtype=torch.bool)
    mask[len(prompt_ids):] = True
    return VROTrace(
        ids=ids,
        comp_mask=mask,
        completion_key=tuple(int(token) for token in completion_ids.tolist()),
        text=text,
        reward=float(reward),
        pid=int(pid),
        round_added=int(round_added),
    )


def _add_unique_trace(
    buffers: dict[int, list[VROTrace]],
    seen: dict[int, set[tuple[int, ...]]],
    trace: VROTrace,
) -> bool:
    keys = seen.setdefault(trace.pid, set())
    if trace.completion_key in keys:
        return False
    keys.add(trace.completion_key)
    buffers.setdefault(trace.pid, []).append(trace)
    return True


def _prune_history(
    buffers: dict[int, list[VROTrace]],
    seen: dict[int, set[tuple[int, ...]]],
    *,
    min_round: int,
) -> int:
    removed = 0
    for pid in list(buffers):
        kept = [row for row in buffers[pid] if row.round_added >= min_round]
        removed += len(buffers[pid]) - len(kept)
        if kept:
            buffers[pid] = kept
            seen[pid] = {row.completion_key for row in kept}
        else:
            del buffers[pid]
            seen.pop(pid, None)
    return removed


def _flatten_buffers(
    buffers: dict[int, list[VROTrace]],
    pids: Iterable[int] | None = None,
) -> list[VROTrace]:
    selected = sorted(buffers) if pids is None else sorted(int(pid) for pid in pids)
    return [row for pid in selected for row in buffers.get(pid, [])]


def _score_traces(
    model,
    traces: list[VROTrace],
    pad_token_id: int,
    *,
    micro: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    ids, masks = _pad_traces(traces, pad_token_id)
    joint = seq_logprobs(model, ids, masks, micro=micro, length_norm=False)
    lengths = masks.sum(1).float().clamp(min=1)
    return ids, masks, joint, joint / lengths


def _safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def _responsibility_diagnostics(
    traces: list[VROTrace],
    weights: torch.Tensor,
    pids: torch.Tensor,
    *,
    round_index: int,
    reward_floor: float,
    top_k: int = 3,
) -> tuple[dict, list[dict], list[dict]]:
    weights_np = weights.detach().double().cpu().numpy()
    pids_np = pids.detach().cpu().numpy()
    prompt_rows = []
    top_rows = []
    prompt_corr_len = []
    prompt_corr_reward = []

    for pid in sorted(set(int(pid) for pid in pids_np)):
        idx = np.flatnonzero(pids_np == pid)
        local_w = weights_np[idx]
        rewards = np.asarray([traces[i].reward for i in idx], dtype=np.float64)
        lengths = np.asarray([traces[i].length for i in idx], dtype=np.float64)
        ages = np.asarray([round_index - traces[i].round_added for i in idx], dtype=np.float64)
        correct = rewards > reward_floor + 0.5
        entropy = float(-(local_w * np.log(np.clip(local_w, 1e-300, None))).sum())
        ess_fraction = float(1.0 / (len(idx) * np.square(local_w).sum()))
        corr_len = _safe_corr(local_w, lengths)
        corr_reward = _safe_corr(local_w, rewards)
        if corr_len is not None:
            prompt_corr_len.append(corr_len)
        if corr_reward is not None:
            prompt_corr_reward.append(corr_reward)
        prompt_rows.append({
            "pid": pid,
            "support_size": int(len(idx)),
            "correct_fraction": float(correct.mean()),
            "correct_mass": float(local_w[correct].sum()),
            "ess_fraction": ess_fraction,
            "entropy": entropy,
            "max_weight": float(local_w.max()),
            "weighted_length": float(np.dot(local_w, lengths)),
            "weighted_age": float(np.dot(local_w, ages)),
            "previous_round_mass": float(local_w[ages > 0].sum()),
        })
        for local_index in np.argsort(-local_w)[:top_k]:
            trace_index = int(idx[int(local_index)])
            trace = traces[trace_index]
            top_rows.append({
                "pid": pid,
                "responsibility": float(weights_np[trace_index]),
                "reward": trace.reward,
                "correct": bool(trace.reward > reward_floor + 0.5),
                "length": trace.length,
                "age": int(round_index - trace.round_added),
                "round_added": trace.round_added,
                "text": trace.text,
            })

    summary = {
        "prompt_count": len(prompt_rows),
        "support_size": len(traces),
        "correct_fraction": float(np.mean(
            [trace.reward > reward_floor + 0.5 for trace in traces]
        )),
        "correct_mass": float(np.mean([row["correct_mass"] for row in prompt_rows])),
        "ess_fraction": float(np.mean([row["ess_fraction"] for row in prompt_rows])),
        "entropy": float(np.mean([row["entropy"] for row in prompt_rows])),
        "max_weight": float(np.mean([row["max_weight"] for row in prompt_rows])),
        "weighted_length": float(np.mean([row["weighted_length"] for row in prompt_rows])),
        "weighted_age": float(np.mean([row["weighted_age"] for row in prompt_rows])),
        "previous_round_mass": float(np.mean(
            [row["previous_round_mass"] for row in prompt_rows]
        )),
        "weight_length_corr": (
            float(np.mean(prompt_corr_len)) if prompt_corr_len else None
        ),
        "weight_reward_corr": (
            float(np.mean(prompt_corr_reward)) if prompt_corr_reward else None
        ),
    }
    return summary, prompt_rows, top_rows


def run_conditional_vro(
    task,
    rounds=40,
    B=64,
    G=4,
    seed=0,
    lr=3e-6,
    epochs=1,
    model_name=MODEL_NAME,
    model_tok=None,
    micro=4,
    window=None,
    responsibility_score="joint",
    policy_factor=True,
    eval_every=0,
    eval_fn=None,
    diagnostics_fn=None,
    checkpoint_every=0,
    checkpoint_fn=None,
    log=print,
):
    """Fine-tune a conditional LM with the VRO mixture-of-deltas E/M update."""
    if rounds < 1:
        raise ValueError(f"rounds must be positive, got {rounds}")
    if B < 1 or G < 1 or B % G:
        raise ValueError(f"B must be positive and divisible by G, got B={B}, G={G}")
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if micro < 1:
        raise ValueError(f"micro must be positive, got {micro}")
    if window is not None and window < 1:
        raise ValueError(f"window must be None or a positive number of rounds, got {window}")
    if checkpoint_every < 0:
        raise ValueError(f"checkpoint_every must be nonnegative, got {checkpoint_every}")
    if responsibility_score not in {"joint", "token_mean"}:
        raise ValueError(
            "responsibility_score must be 'joint' or 'token_mean', "
            f"got {responsibility_score!r}"
        )
    reward_floor = float(getattr(task, "floor", 0.0))
    if reward_floor <= 0:
        raise ValueError(
            "conditional VRO requires a strictly positive task reward floor; "
            f"task.floor={reward_floor}"
        )

    model, tok = model_tok if model_tok is not None else load_model(seed=seed, model=model_name)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rollout_rng = np.random.default_rng(seed)
    mstep_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    prompt_ids = [
        tok(prompt, return_tensors="pt").input_ids[0].detach().cpu()
        for prompt in task.prompts
    ]
    buffers: dict[int, list[VROTrace]] = {}
    seen: dict[int, set[tuple[int, ...]]] = {}
    recs = []
    oracle = gen = gsteps = accepted_gsteps = unique_scored = evictions = 0
    faithful = bool(policy_factor and responsibility_score == "joint" and window is None)
    if not policy_factor:
        tag = "VRO-RewardOnly"
    elif responsibility_score == "token_mean":
        tag = "VRO-TokenMean"
    elif window is None:
        tag = "VRO-Full"
    elif window == 1:
        tag = "VRO-Current"
    else:
        tag = f"VRO-Window{window}"

    for t in _rounds(rounds):
        pids_sampled, pid_row, generated_ids, generated_mask, rewards_np, texts = sample_round(
            model, tok, task, B, G, rollout_rng
        )
        if not np.isfinite(rewards_np).all() or np.any(rewards_np <= 0):
            raise ValueError("VRO received a non-finite or non-positive verifier reward")
        oracle += len(rewards_np)
        gen += len(rewards_np)

        if window is not None:
            evictions += _prune_history(
                buffers, seen, min_round=max(0, t - window + 1)
            )

        generated_traces = []
        generated_duplicate = []
        for row_index, (pid, reward, text) in enumerate(zip(pid_row, rewards_np, texts)):
            trace = _make_trace(
                prompt_ids[int(pid)],
                generated_ids[row_index],
                generated_mask[row_index],
                text=text,
                reward=float(reward),
                pid=int(pid),
                round_added=t,
            )
            generated_traces.append(trace)
            added = _add_unique_trace(buffers, seen, trace)
            generated_duplicate.append(not added)
            unique_scored += int(added)

        # Equation 62 is estimated over the uniformly sampled question minibatch, while each
        # selected question contributes its complete retained history. This matches GRPO's equal
        # per-question grouping without progressively overweighting questions sampled early.
        traces = _flatten_buffers(buffers, pids_sampled)
        total_buffer_size = sum(len(rows) for rows in buffers.values())
        all_ids, all_masks, joint_logps, mean_logps = _score_traces(
            model, traces, tok.eos_token_id, micro=micro
        )
        rewards = torch.tensor(
            [row.reward for row in traces], device=DEV, dtype=torch.float32
        )
        trace_pids = torch.tensor(
            [row.pid for row in traces], device=DEV, dtype=torch.long
        )
        weights = conditional_responsibilities(
            rewards,
            joint_logps,
            mean_logps,
            trace_pids,
            policy_factor=policy_factor,
            responsibility_score=responsibility_score,
        ).detach()
        bound_before = conditional_bound(weights, rewards, joint_logps, trace_pids)
        objective_before = conditional_mstep_objective(weights, joint_logps, trace_pids)
        support_before = conditional_support_log_evidence(rewards, joint_logps, trace_pids)

        weight_lookup = {
            (trace.pid, trace.completion_key): float(weights[i])
            for i, trace in enumerate(traces)
        }
        mean_lp_lookup = {
            (trace.pid, trace.completion_key): float(mean_logps[i])
            for i, trace in enumerate(traces)
        }
        sampled_mean_lps = [
            mean_lp_lookup[(trace.pid, trace.completion_key)]
            for trace in generated_traces
        ]

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        parameter_snapshot = [parameter.detach().clone() for parameter in trainable]
        optimizer_snapshot = copy.deepcopy(opt.state_dict())
        attempted_steps = 0
        total_loss = 0.0
        model.train()
        for _ in range(epochs):
            draw = draw_conditional_mstep_indices(weights, trace_pids, B, mstep_rng)
            mstep_rng.shuffle(draw)
            for start in range(0, len(draw), micro):
                selected = torch.as_tensor(
                    draw[start:start + micro], device=DEV, dtype=torch.long
                )
                selected_logps = seq_logprobs(
                    model,
                    all_ids[selected],
                    all_masks[selected],
                    micro=max(1, len(selected)),
                    grad=True,
                    length_norm=False,
                )
                loss = -selected_logps.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total_loss += float(loss.detach())
                attempted_steps += 1
        gsteps += attempted_steps

        model.eval()
        post_joint_logps = seq_logprobs(
            model, all_ids, all_masks, micro=micro, length_norm=False
        )
        if not bool(torch.isfinite(post_joint_logps).all()) or not math.isfinite(total_loss):
            with torch.no_grad():
                for parameter, snapshot in zip(trainable, parameter_snapshot):
                    parameter.copy_(snapshot)
            opt.load_state_dict(optimizer_snapshot)
            raise FloatingPointError(
                "conditional VRO produced a non-finite M-step loss or sequence score; "
                "the attempted update was rolled back"
            )
        bound_after_attempt = conditional_bound(
            weights, rewards, post_joint_logps, trace_pids
        )
        attempted_delta = float(bound_after_attempt - bound_before)
        tolerance = 1e-6 * max(1.0, abs(float(bound_before)))
        mstep_accepted = attempted_delta >= -tolerance
        if not mstep_accepted:
            with torch.no_grad():
                for parameter, snapshot in zip(trainable, parameter_snapshot):
                    parameter.copy_(snapshot)
            opt.load_state_dict(optimizer_snapshot)
            bound_after = bound_before
            support_after = support_before
        else:
            accepted_gsteps += attempted_steps
            bound_after = bound_after_attempt
            support_after = conditional_support_log_evidence(
                rewards, post_joint_logps, trace_pids
            )

        generated_batch_ids, generated_batch_masks = _pad_traces(
            generated_traces, tok.eos_token_id
        )
        kl = kl_from_base(model, generated_batch_ids, generated_batch_masks)
        test_acc = maybe_eval(model, t, rounds, eval_every, eval_fn)
        resp_summary, prompt_diagnostics, top_traces = _responsibility_diagnostics(
            traces,
            weights,
            trace_pids,
            round_index=t,
            reward_floor=reward_floor,
        )
        support_gap = float(support_before - bound_before)

        current_samples = []
        for trace, duplicate in zip(generated_traces, generated_duplicate):
            current_samples.append({
                "pid": trace.pid,
                "reward": trace.reward,
                "correct": bool(trace.reward > reward_floor + 0.5),
                "length": trace.length,
                "duplicate": duplicate,
                "responsibility": weight_lookup.get((trace.pid, trace.completion_key)),
                "text": trace.text,
            })

        if diagnostics_fn is not None:
            diagnostics_fn({
                "round": t,
                "completed_rounds": t + 1,
                "method_family": "conditional_vro",
                "faithful_vro": faithful,
                "policy_factor": bool(policy_factor),
                "responsibility_score": responsibility_score,
                "history_window": window,
                "e_step_exact": bool(policy_factor and responsibility_score == "joint"),
                "reward_floor": reward_floor,
                "bound": {
                    "before": float(bound_before),
                    "after": float(bound_after),
                    "attempted_after": float(bound_after_attempt),
                    "attempted_delta": attempted_delta,
                    "acceptance_tolerance": tolerance,
                    "support_log_evidence_before": float(support_before),
                    "support_log_evidence_after": float(support_after),
                    "posterior_kl_gap": support_gap,
                },
                "m_step": {
                    "accepted": mstep_accepted,
                    "attempted_steps": attempted_steps,
                    "accepted_steps": attempted_steps if mstep_accepted else 0,
                    "objective_before": float(objective_before),
                    "loss": total_loss / max(attempted_steps, 1),
                },
                "buffer": {
                    "active_support_size": len(traces),
                    "total_support_size": total_buffer_size,
                    "unique_scored": unique_scored,
                    "evictions": evictions,
                    "duplicates_this_round": int(sum(generated_duplicate)),
                },
                "responsibilities": resp_summary,
                "prompts": prompt_diagnostics,
                "top_traces": top_traces,
                "samples": current_samples,
            })

        mean_reward = float(np.mean(rewards_np))
        frac_correct = float(np.mean(rewards_np > reward_floor + 0.5))
        gen_len = float(np.mean([trace.length for trace in generated_traces]))
        recs.append({
            "round": t,
            "oracle": oracle,
            "verifier_calls": oracle,
            "diagnostic_verifier_calls": 0,
            "gen": gen,
            "llm_gen": gen,
            "gsteps": gsteps,
            "accepted_gsteps": accepted_gsteps,
            "mean_reward": mean_reward,
            "frac_correct": frac_correct,
            "gen_len": gen_len,
            "fmt": format_rate(texts),
            "samp_lp": (
                float(np.mean(sampled_mean_lps)) if sampled_mean_lps else float("nan")
            ),
            "gold_lp": float("nan"),
            "kl": kl,
            "test_acc": test_acc,
            "loss": total_loss / max(attempted_steps, 1),
            "ess": resp_summary["ess_fraction"],
            "posterior_correct_mass": resp_summary["correct_mass"],
            "responsibility_entropy": resp_summary["entropy"],
            "responsibility_max": resp_summary["max_weight"],
            "weight_length_corr": (
                resp_summary["weight_length_corr"]
                if resp_summary["weight_length_corr"] is not None
                else float("nan")
            ),
            "weighted_age": resp_summary["weighted_age"],
            "previous_round_mass": resp_summary["previous_round_mass"],
            "buffer_size": total_buffer_size,
            "active_support_size": len(traces),
            "buffer_evictions": evictions,
            "unique_scored": unique_scored,
            "duplicate_fraction": float(np.mean(generated_duplicate)),
            "vro_bound": float(bound_after),
            "vro_support_log_evidence": float(support_after),
            "vro_posterior_kl_gap": support_gap,
            "mstep_accepted": mstep_accepted,
            "mstep_attempted_delta": attempted_delta,
        })
        if (
            checkpoint_fn is not None
            and checkpoint_every > 0
            and (t + 1) % checkpoint_every == 0
            and (t + 1) < rounds
        ):
            checkpoint_fn(model, t + 1)
        log(
            f"  [{tag} G{G} r{t:>3}] gen={gen:>6} R={mean_reward:.3f} "
            f"q(correct)={resp_summary['correct_mass']:.3f} "
            f"ESS={resp_summary['ess_fraction']:.3f} "
            f"dB={attempted_delta:+.3e} accept={int(mstep_accepted)} "
            f"H={len(traces)}/{total_buffer_size}"
        )
    return recs
