"""Shared multi-prompt training primitives (one trainer per mp_*.py file imports from here).

Re-exports the low-level LM helpers from methods_lm so trainers import a single module.
"""
from __future__ import annotations
import re

import numpy as np
import torch

from methods_lm import (load_model, token_logps, kl_from_base, seq_logprobs,  # noqa: F401 (re-export)
                        MODEL_NAME, DEV)
from prompting import build_proposal_prompt

MAX_NEW = 40


def encode_task_prompt(tok, task, question_id: int, **kwargs):
    """Encode one task prompt without duplicating chat-template specials."""

    if bool(getattr(task, "rendered_chat_prompts", False)):
        kwargs.setdefault("add_special_tokens", False)
    return tok(task.prompts[int(question_id)], **kwargs)


def audit_chat_active_completions(tok, active_rows) -> None:
    """Record active chat tokens and fail immediately on pre-EOT EOD."""

    if not bool(getattr(tok, "_vrl_math_chat_rendered_prompts", False)):
        return
    audit = getattr(tok, "_vrl_math_chat_generation_audit", None)
    if not isinstance(audit, dict):
        raise ValueError("Qwen3 chat generation audit is not bound")
    rows = list(active_rows)
    eod = int(audit["eod_token_id"])
    eod_rows = sum(bool((row == eod).any()) for row in rows)
    audit["sequences"] = int(audit.get("sequences", 0)) + len(rows)
    audit["generated_tokens"] = int(audit.get("generated_tokens", 0)) + sum(
        int(row.numel()) for row in rows
    )
    audit["generated_eod_before_eot_count"] = int(
        audit.get("generated_eod_before_eot_count", 0)
    ) + eod_rows
    if eod_rows:
        raise ValueError("Qwen3 chat generated EOD before assistant EOT")


def task_pad_token_id(tok) -> int:
    """Use EOD padding only inside the bound chat runtime."""

    if bool(getattr(tok, "_vrl_math_chat_rendered_prompts", False)):
        if tok.pad_token_id is None:
            raise ValueError("Qwen3 chat tokenizer has no EOD padding token")
        return int(tok.pad_token_id)
    return int(tok.eos_token_id)


class QuestionSampler:
    """Sample prompt ids randomly or as reproducible shuffled epochs."""

    def __init__(self, pool, rng, mode="random"):
        self.pool = np.asarray(list(pool), dtype=int)
        self.rng = rng
        self.mode = str(mode)
        self._order = np.empty(0, dtype=int)
        self._cursor = 0
        if self.mode not in {"random", "epoch_shuffle"}:
            raise ValueError(f"unknown question sampling mode {self.mode!r}")

    def _reshuffle(self):
        self._order = self.rng.permutation(self.pool)
        self._cursor = 0

    def sample(self, size):
        size = int(size)
        if size <= 0 or len(self.pool) == 0:
            return np.empty(0, dtype=int)
        if self.mode == "random":
            return self.rng.choice(
                self.pool,
                size=size,
                replace=size > len(self.pool),
            )

        selected = []
        while len(selected) < size:
            if self._cursor >= len(self._order):
                self._reshuffle()
            take = min(size - len(selected), len(self._order) - self._cursor)
            selected.extend(self._order[self._cursor:self._cursor + take].tolist())
            self._cursor += take
        return np.asarray(selected, dtype=int)


def matched_question_rng(generation_rng, seed, mode):
    """Return the dedicated RNG used by matched shuffled-question studies."""

    if mode == "epoch_shuffle":
        return np.random.default_rng(int(seed) * 1013 + 23)
    return generation_rng


QUESTION_SCHEDULE_RNG_MODES = {"dedicated", "run_seed"}


def resolve_question_schedule_rng(generation_rng, seed, sampling_mode, rng_mode):
    """Resolve an explicit question-schedule RNG without changing old defaults.

    ``dedicated`` preserves the existing GRPO/RLOO schedule stream.  The
    ``run_seed`` mode is an audit control used by the low-data regime study so
    the policy-gradient baselines reproduce AC-ALG1's historical epoch stream
    exactly at a matched seed and question-pool size.
    """

    rng_mode = str(rng_mode)
    if rng_mode not in QUESTION_SCHEDULE_RNG_MODES:
        raise ValueError(
            "question_schedule_rng must be 'dedicated' or 'run_seed', got "
            f"{rng_mode!r}"
        )
    if rng_mode == "run_seed":
        return generation_rng
    return matched_question_rng(generation_rng, seed, sampling_mode)


def _rounds(rounds):
    """Round loop with a tqdm bar (auto-off when stderr isn't a TTY, e.g. under tee)."""
    try:
        from tqdm import tqdm
        return tqdm(range(rounds), desc="rounds", disable=None, leave=False, dynamic_ncols=True)
    except Exception:
        return range(rounds)


def _sample_multi_legacy(
    model,
    tok,
    prompts,
    *,
    temperature,
    max_new,
    gen_bs,
    return_token_logprobs,
    do_sample,
    top_k,
    top_p,
):
    """Exact historical sampler retained for every non-chat task."""

    model.eval()
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)
    p_len, eos = enc.input_ids.shape[1], tok.eos_token_id
    chunks = []
    logprob_chunks = []
    for i in range(0, enc.input_ids.shape[0], gen_bs):
        generation_kwargs = {
            "input_ids": enc.input_ids[i:i + gen_bs],
            "attention_mask": enc.attention_mask[i:i + gen_bs],
            "do_sample": bool(do_sample),
            "max_new_tokens": max_new,
            "pad_token_id": eos,
        }
        if do_sample:
            generation_kwargs.update(
                temperature=temperature,
                top_k=int(top_k),
                top_p=float(top_p),
            )
        if return_token_logprobs:
            generation_kwargs.update(
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated = model.generate(**generation_kwargs)
        if return_token_logprobs:
            chunks.append(generated.sequences)
            logprob_chunks.append(
                model.compute_transition_scores(
                    generated.sequences,
                    generated.scores,
                    normalize_logits=True,
                )
            )
        else:
            chunks.append(generated)
    width = max(c.shape[1] for c in chunks)
    out = torch.full(
        (enc.input_ids.shape[0], width),
        eos,
        dtype=chunks[0].dtype,
        device=DEV,
    )
    token_logprobs = (
        torch.full(
            (enc.input_ids.shape[0], width),
            float("nan"),
            dtype=torch.float32,
            device=DEV,
        )
        if return_token_logprobs
        else None
    )
    r = 0
    for chunk_index, chunk in enumerate(chunks):
        out[r:r + chunk.shape[0], :chunk.shape[1]] = chunk
        if token_logprobs is not None:
            values = logprob_chunks[chunk_index]
            token_logprobs[
                r:r + chunk.shape[0],
                p_len:p_len + values.shape[1],
            ] = values.to(dtype=torch.float32)
        r += chunk.shape[0]
    comp_mask = torch.zeros_like(out, dtype=torch.bool)
    comp_mask[:, p_len:] = True
    for row in range(out.shape[0]):
        idx = (out[row, p_len:] == eos).nonzero()
        if len(idx):
            comp_mask[row, p_len + int(idx[0]) + 1:] = False
    texts = tok.batch_decode(out[:, p_len:], skip_special_tokens=True)
    if token_logprobs is not None:
        return out, comp_mask, texts, token_logprobs
    return out, comp_mask, texts


@torch.no_grad()
def sample_multi(
    model,
    tok,
    prompts,
    temperature=1.0,
    max_new=MAX_NEW,
    gen_bs=32,
    return_token_logprobs=False,
    do_sample=True,
    top_k=0,
    top_p=1.0,
):
    """One sampled completion per prompt (left-padded, chunked at gen_bs to bound the KV-cache peak).
    Returns full token ids, a completion mask, decoded strings, and optionally
    normalized sample-time log probabilities aligned with the token ids."""
    if not bool(getattr(tok, "_vrl_math_chat_rendered_prompts", False)):
        return _sample_multi_legacy(
            model,
            tok,
            prompts,
            temperature=temperature,
            max_new=max_new,
            gen_bs=gen_bs,
            return_token_logprobs=return_token_logprobs,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
        )
    model.eval()
    tok.padding_side = "left"
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(DEV)
    padded_prompt_len = enc.input_ids.shape[1]
    eos = int(tok.eos_token_id)
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else eos)
    prompt_rows = [
        row_ids[row_attention.to(dtype=torch.bool)]
        for row_ids, row_attention in zip(
            enc.input_ids,
            enc.attention_mask,
            strict=True,
        )
    ]
    sequence_rows = []
    completion_rows = []
    transition_rows = []
    for i in range(0, enc.input_ids.shape[0], gen_bs):
        generation_kwargs = {
            "input_ids": enc.input_ids[i:i + gen_bs],
            "attention_mask": enc.attention_mask[i:i + gen_bs],
            "do_sample": bool(do_sample),
            "max_new_tokens": max_new,
            # Qwen3 chat declares EOD as padding but only assistant EOT is a
            # valid stop.  Passing both explicitly prevents its checkpoint's
            # dual-EOS generation default from treating generated EOD as EOS.
            "eos_token_id": eos,
            "pad_token_id": pad,
        }
        if do_sample:
            generation_kwargs.update(
                temperature=temperature,
                top_k=int(top_k),
                top_p=float(top_p),
            )
        if return_token_logprobs:
            generation_kwargs.update(
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated = model.generate(**generation_kwargs)
        if return_token_logprobs:
            sequences = generated.sequences
            transitions = model.compute_transition_scores(
                sequences,
                generated.scores,
                normalize_logits=True,
            )
        else:
            sequences = generated
            transitions = None
        continuation = sequences[:, padded_prompt_len:]
        for local_row, generated_row in enumerate(continuation):
            positions = (generated_row == eos).nonzero(as_tuple=False)
            active_count = (
                int(positions[0].item()) + 1
                if len(positions)
                else int(generated_row.numel())
            )
            active_completion = generated_row[:active_count]
            prompt_row = prompt_rows[i + local_row]
            sequence_rows.append(torch.cat((prompt_row, active_completion)))
            completion_rows.append(active_completion)
            if transitions is not None:
                transition_rows.append(transitions[local_row, :active_count])

    # Generation uses left padding, while teacher-forced sequence scorers do
    # not receive an attention mask.  Strip every leading prompt pad and
    # right-pad only after the active completion so rescoring conditions on
    # exactly the same per-row prompt tokens as generation.
    width = max(int(row.numel()) for row in sequence_rows)
    out = torch.full(
        (len(sequence_rows), width),
        pad,
        dtype=sequence_rows[0].dtype,
        device=DEV,
    )
    comp_mask = torch.zeros_like(out, dtype=torch.bool)
    token_logprobs = (
        torch.full(
            (len(sequence_rows), width),
            float("nan"),
            dtype=torch.float32,
            device=DEV,
        )
        if return_token_logprobs
        else None
    )
    for row_index, (sequence, prompt_row, completion) in enumerate(
        zip(sequence_rows, prompt_rows, completion_rows, strict=True)
    ):
        sequence_len = int(sequence.numel())
        prompt_len = int(prompt_row.numel())
        completion_len = int(completion.numel())
        out[row_index, :sequence_len] = sequence
        comp_mask[row_index, prompt_len:prompt_len + completion_len] = True
        if token_logprobs is not None:
            token_logprobs[
                row_index,
                prompt_len:prompt_len + completion_len,
            ] = transition_rows[row_index].to(dtype=torch.float32)

    audit_chat_active_completions(tok, completion_rows)
    texts = tok.batch_decode(completion_rows, skip_special_tokens=True)
    if token_logprobs is not None:
        return out, comp_mask, texts, token_logprobs
    return out, comp_mask, texts


def sample_round(
    model,
    tok,
    task,
    B,
    G,
    rng,
    question_sampler=None,
    proposal_prompt="question",
    reward_requires_eos=False,
):
    """One round: sample G completions for each of B//G random prompts and score them.
    Returns (pids, pid_row, ids, comp_mask, rew, texts)."""
    n = len(task.prompts)
    sampler = question_sampler or QuestionSampler(range(n), rng, mode="random")
    pids = sampler.sample(B // G)
    pid_row = np.repeat(pids, G)
    if proposal_prompt == "question":
        prompts = [task.prompts[i] for i in pid_row]
    else:
        if not hasattr(task, "gold_answer"):
            raise ValueError(
                "answer-conditioned proposal prompts require per-question "
                "gold answers"
            )
        task_builder = getattr(task, "build_proposal_prompt", None)
        prompts = [
            (
                task_builder(int(i), proposal_prompt)
                if task_builder is not None
                else build_proposal_prompt(
                    task.prompts[i],
                    task.gold_answer[i],
                    proposal_prompt,
                )
            )
            for i in pid_row
        ]
    ids, mask, texts = sample_multi(model, tok, prompts, max_new=getattr(task, "max_new", MAX_NEW))
    rew = np.asarray(task.reward(texts, pid_row), dtype=np.float64)
    if reward_requires_eos:
        eos = natural_eos_mask(ids, mask, tok.eos_token_id)
        reward_floor = float(getattr(task, "floor", 0.0))
        rew = np.where(eos, rew, reward_floor)
    return pids, pid_row, ids, mask, rew, texts


def natural_eos_mask(ids, completion_mask, eos_token_id):
    """Return whether each sampled completion emitted EOS before its limit.

    ``sample_multi`` keeps the first emitted EOS inside ``completion_mask`` and
    masks every later padding token.  Inspecting the final active token therefore
    distinguishes a real stop from a completion that exhausted ``max_new_tokens``.
    This check uses token ids rather than decoded text because special tokens are
    deliberately omitted by the decoder.
    """

    if ids.shape != completion_mask.shape:
        raise ValueError("token ids and completion mask must align")
    flags = []
    for row_ids, row_mask in zip(ids, completion_mask, strict=True):
        active = row_ids[row_mask]
        flags.append(bool(active.numel() and int(active[-1]) == int(eos_token_id)))
    return np.asarray(flags, dtype=bool)


def pairs_by_reward(pids, pid_row, rew):
    """One preference pair per prompt: chosen=best, rejected=worst (skip ties = no signal)."""
    ci, li = [], []
    for p in pids:
        idx = np.where(pid_row == p)[0]
        if rew[idx].max() - rew[idx].min() < 1e-9:
            continue
        ci.append(int(idx[rew[idx].argmax()])); li.append(int(idx[rew[idx].argmin()]))
    return np.array(ci, dtype=int), np.array(li, dtype=int)


# --------------------------------------------------------------------------- #
#  Per-round DIAGNOSTICS (cheap, read-only -- never affect training)
# --------------------------------------------------------------------------- #
def comp_len(mask):
    """Mean completion length in tokens over the batch (degeneration / truncation / length-collapse)."""
    return float(mask.sum(1).float().mean())


def frac_correct(rew, thresh=0.5):
    """Fraction of sampled completions that cleared the reward floor (base rate / signal strength).
    For the {0.05 floor, 1.05 correct} verifier rewards, thresh=0.5 separates correct from wrong."""
    return float((np.asarray(rew) > thresh).mean())


def ess_frac(w):
    """Effective-sample-size FRACTION of a weight vector, 1/(N·Σŵ²) in (1/N, 1]. ->1 uniform (healthy),
    ->1/N collapsed onto one sample (the EM weight-collapse failure)."""
    w = np.asarray(w, dtype=np.float64); s = w.sum()
    if s <= 0 or len(w) == 0:
        return float("nan")
    w = w / s
    return float(1.0 / (len(w) * np.sum(w ** 2)))


_FMT_RE = re.compile(r"####\s*\$?\s*-?[\d,]*\d")


def has_answer_marker(text):
    """Whether a completion contains the canonical numeric answer marker."""
    return bool(_FMT_RE.search(text))


def format_rate(texts):
    """Fraction of completions carrying a parseable ``#### number`` marker. When accuracy drops,
    this separates 'lost the format' from 'lost the math' AT TRAIN TIME (not just the final benchmark)."""
    if not len(texts):
        return float("nan")
    return float(np.mean([has_answer_marker(t) for t in texts]))


def task_format_rate(task, texts):
    """Format rate under the task's declared strict answer-event contract."""
    if not len(texts):
        return float("nan")
    parser = getattr(task, "parse_answer_event", None)
    if parser is None:
        return format_rate(texts)
    return float(
        np.mean(
            [
                parser(text, mode="strict_terminal_marker").strict_valid
                for text in texts
            ]
        )
    )


@torch.no_grad()
def mean_token_lp(model, ids, comp_mask, micro=8):
    """Mean per-token logprob of the model on the given (its own) samples -- the policy-entropy proxy
    (rising toward 0 = sharpening/entropy collapse, the RLHF collapse precursor). `div` is coarser:
    completions can be distinct yet near-deterministic."""
    return float(seq_logprobs(model, ids, comp_mask, micro=micro, length_norm=True).mean())


def build_gold_lists(tok, task):
    """Tokenised [prompt ++ gold_solution] per prompt with a loss mask over the solution span, as UNPADDED CPU
    lists (gi, gm) -- the ONE builder behind both the gold_lp diagnostic (padded once, below) and
    AC-EM's B_sup term (which pads per minibatch). None if the task has no gold reasoning."""
    if not hasattr(task, "gold_solution"):
        return None
    gi, gm = [], []
    for pid in range(len(task.prompts)):
        gp = encode_task_prompt(
            tok,
            task,
            pid,
            return_tensors="pt",
        ).input_ids[0].tolist()
        gc = tok(" " + task.gold_solution[pid], add_special_tokens=False).input_ids
        f = torch.tensor(gp + gc, dtype=torch.long)
        s = torch.zeros(len(f), dtype=torch.bool); s[len(gp):] = True     # score only the gold_solution span
        gi.append(f); gm.append(s)
    return gi, gm


def build_gold_batch(tok, task):
    """build_gold_lists padded to (ids, mask) CPU tensors, built ONCE for the gold_lp diagnostic;
    None if the task has no gold reasoning."""
    lists = build_gold_lists(tok, task)
    if lists is None:
        return None
    pad = torch.nn.utils.rnn.pad_sequence
    return (pad(lists[0], batch_first=True, padding_value=task_pad_token_id(tok)),
            pad(lists[1], batch_first=True, padding_value=False))


def maybe_eval(model, t, rounds, eval_every, eval_fn, eval_rounds=None):
    """Run a held-out checkpoint on a periodic or explicit completed-round schedule.

    ``eval_rounds`` supports sparse logarithmic long-horizon checks without paying
    for every multiple of the earliest checkpoint. The final round is always
    evaluated when either schedule is enabled.
    """
    completed = t + 1
    explicit = {int(value) for value in (eval_rounds or ())}
    enabled = eval_every > 0 or bool(explicit)
    scheduled = (
        (eval_every > 0 and completed % eval_every == 0)
        or completed in explicit
        or t == rounds - 1
    )
    if eval_fn is None or not enabled or not scheduled:
        return float("nan")
    model.eval()
    res = eval_fn(model)
    return float(res) if res is not None else float("nan")


@torch.no_grad()
def gold_lp(model, gold_batch, micro=8):
    """Mean per-token logprob of the GOLD reasoning under the current policy (teacher-forced, no
    generation -- cheap). Rising = training moves toward gold reasoning; falling while the model's
    own-sample signal rises = it is sharpening its own (possibly wrong) reasoning, not learning gold's.
    Per-round and irrecoverable post-hoc; the mechanism plot behind the sup/EM contrast."""
    if gold_batch is None:
        return float("nan")
    ids, msk = gold_batch
    return float(seq_logprobs(model, ids.to(DEV), msk.to(DEV), micro=micro, length_norm=True).mean())
