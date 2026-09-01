"""Common-protocol adaptations of supervised self-training baselines for GSM8K.

The historical ``RFT`` registry entry is an online rejection-SFT loop.  This
module deliberately uses new result identities for the paper algorithms:

* ``RFT-Source`` performs one Generate phase and one Improve phase;
* ``ReST-EM`` repeats Generate/Improve and starts every Improve phase from the
  original adapter parameters;
* ``STaR`` greedily generates one rationale per question, rationalizes failed
  questions with an answer hint, removes the hint from the training context,
  and starts every Improve phase from the original adapter parameters;
* ``Gold-CoT-SFT`` is a separately reported human-rationale supervision upper
  bound with an explicit EOS target.

Generate and Improve never interleave inside a phase.  This is the behavioural
contract that distinguishes these methods from the legacy online ``run_raft``
trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch

from common import (
    DEV,
    MODEL_NAME,
    QuestionSampler,
    _rounds,
    build_gold_batch,
    comp_len,
    encode_task_prompt,
    task_format_rate,
    gold_lp,
    kl_from_base,
    load_model,
    matched_question_rng,
    maybe_eval,
    natural_eos_mask,
    sample_multi,
    seq_logprobs,
    task_pad_token_id,
)
from prompting import build_proposal_prompt


SELF_TRAINING_MODES = frozenset({"rft_source", "rest_em", "star"})


@dataclass(frozen=True)
class TrainingExample:
    """An exact completion reanchored under the answer-blind question prompt."""

    question_id: int
    ids: torch.Tensor
    span: torch.Tensor
    source: str

    def __post_init__(self) -> None:
        if self.ids.ndim != 1 or self.span.ndim != 1:
            raise ValueError("training example ids and span must be one-dimensional")
        if self.ids.shape != self.span.shape:
            raise ValueError("training example ids and span must align")
        if self.span.dtype != torch.bool or not bool(self.span.any()):
            raise ValueError("training example must contain a nonempty boolean loss span")


def phase_question_schedule(
    n_questions: int,
    rounds: int,
    iterations: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Return a phase-balanced schedule that covers every question once/phase."""

    n_questions = int(n_questions)
    rounds = int(rounds)
    iterations = int(iterations)
    if n_questions <= 0 or rounds <= 0 or iterations <= 0:
        raise ValueError("questions, rounds and iterations must be positive")
    if rounds % iterations:
        raise ValueError("rounds must be divisible by self-training iterations")
    rounds_per_phase = rounds // iterations
    if n_questions % rounds_per_phase:
        raise ValueError(
            "question count must be divisible by rounds per self-training phase"
        )
    questions_per_round = n_questions // rounds_per_phase
    rng = np.random.default_rng(int(seed) * 104729 + 41)
    schedule: list[tuple[int, ...]] = []
    for _phase in range(iterations):
        order = [int(value) for value in rng.permutation(n_questions)]
        for start in range(0, n_questions, questions_per_round):
            schedule.append(tuple(order[start:start + questions_per_round]))
    if len(schedule) != rounds:
        raise AssertionError("self-training schedule has the wrong horizon")
    return tuple(schedule)


def _correct_mask(task, texts, pids, eos_flags, require_eos: bool) -> np.ndarray:
    rewards = np.asarray(task.reward(texts, pids=pids), dtype=np.float64)
    floor = float(getattr(task, "floor", 0.0))
    best = float(getattr(task, "best", 1.0))
    threshold = floor + 0.5 * (best - floor)
    correct = rewards > threshold
    if require_eos:
        correct &= np.asarray(eos_flags, dtype=bool)
    return correct


def select_correct_rows(
    pids: Sequence[int],
    correct: Sequence[bool],
    per_question_limit: int,
) -> tuple[int, ...]:
    """Select a balanced answer-correct multiset with a per-question cap."""

    if per_question_limit <= 0:
        raise ValueError("accepted_per_question must be positive")
    if len(pids) != len(correct):
        raise ValueError("question ids and correctness flags must align")
    counts: dict[int, int] = {}
    selected: list[int] = []
    for row, (pid, is_correct) in enumerate(zip(pids, correct, strict=True)):
        pid = int(pid)
        if not is_correct or counts.get(pid, 0) >= per_question_limit:
            continue
        selected.append(row)
        counts[pid] = counts.get(pid, 0) + 1
    return tuple(selected)


def _reanchor_completion(
    tok,
    task,
    question_id: int,
    sampled_ids: torch.Tensor,
    sampled_mask: torch.Tensor,
    source: str,
) -> TrainingExample:
    completion = sampled_ids[sampled_mask].detach().cpu()
    if completion.numel() == 0:
        raise ValueError("sampled completion has no scored token")
    prompt = encode_task_prompt(
        tok,
        task,
        int(question_id),
        return_tensors="pt",
    ).input_ids[0].detach().cpu()
    completion = completion.to(dtype=prompt.dtype)
    ids = torch.cat((prompt, completion))
    span = torch.zeros(ids.shape[0], dtype=torch.bool)
    span[prompt.shape[0]:] = True
    return TrainingExample(
        question_id=int(question_id),
        ids=ids,
        span=span,
        source=source,
    )


def _gold_example(tok, task, question_id: int) -> TrainingExample:
    if tok.eos_token_id is None:
        raise ValueError("Gold-CoT-SFT requires a tokenizer EOS token")
    prompt = encode_task_prompt(
        tok,
        task,
        int(question_id),
        return_tensors="pt",
    ).input_ids[0].detach().cpu()
    completion = torch.tensor(
        tok(
            " " + task.gold_solution[int(question_id)],
            add_special_tokens=False,
        ).input_ids,
        dtype=prompt.dtype,
    )
    if completion.numel() == 0:
        raise ValueError("gold rationale cannot be empty")
    eos = torch.tensor([tok.eos_token_id], dtype=prompt.dtype)
    ids = torch.cat((prompt, completion, eos))
    span = torch.zeros(ids.shape[0], dtype=torch.bool)
    span[prompt.shape[0]:] = True
    return TrainingExample(int(question_id), ids, span, "gold_rationale")


def _pad_examples(tok, examples: Sequence[TrainingExample]):
    pad = torch.nn.utils.rnn.pad_sequence
    ids = pad(
        [example.ids for example in examples],
        batch_first=True,
        padding_value=task_pad_token_id(tok),
    ).to(DEV)
    spans = pad(
        [example.span for example in examples],
        batch_first=True,
        padding_value=False,
    ).to(DEV)
    return ids, spans


def _snapshot_trainable(model) -> dict[str, torch.Tensor]:
    snapshot = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not snapshot:
        raise ValueError("self-training requires trainable model parameters")
    return snapshot


@torch.no_grad()
def _restore_trainable(model, snapshot: dict[str, torch.Tensor]) -> None:
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(snapshot):
        raise ValueError("trainable parameter set changed during self-training")
    for name, parameter in current.items():
        parameter.copy_(snapshot[name].to(parameter.device, dtype=parameter.dtype))


def _train_examples(
    model,
    tok,
    examples: Sequence[TrainingExample],
    optimizer,
    *,
    epochs: int,
    micro: int,
    rng: np.random.Generator,
) -> tuple[float, int, int, int]:
    """Apply token-mean SFT and return loss, steps, tokens, and EOS tokens."""

    if epochs < 1 or micro < 1:
        raise ValueError("epochs and microbatch size must be positive")
    if not examples:
        return float("nan"), 0, 0, 0
    losses: list[float] = []
    steps = 0
    backward_tokens = 0
    backward_eos_tokens = 0
    model.train()
    for _epoch in range(epochs):
        order = rng.permutation(len(examples))
        for start in range(0, len(order), micro):
            indices = order[start:start + micro]
            batch = [examples[int(index)] for index in indices]
            ids, spans = _pad_examples(tok, batch)
            optimizer.zero_grad()
            sequence_logprobs = seq_logprobs(
                model,
                ids,
                spans,
                grad=True,
                length_norm=False,
            )
            loss = -sequence_logprobs.sum() / spans.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite self-training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
            backward_tokens += int(spans.sum())
            backward_eos_tokens += int(((ids == tok.eos_token_id) & spans).sum())
    return float(np.mean(losses)), steps, backward_tokens, backward_eos_tokens


def _phase_boundary(round_index: int, rounds: int, iterations: int) -> bool:
    return (round_index + 1) % (rounds // iterations) == 0


def _records_generation_summary(task, texts, masks, rewards, eos_flags) -> dict:
    return {
        "mean_reward": float(np.mean(rewards)) if len(rewards) else float("nan"),
        "frac_correct": float(np.mean(np.asarray(rewards) > 0.5)) if len(rewards) else 0.0,
        "gen_len": comp_len(masks) if len(texts) else float("nan"),
        "fmt": task_format_rate(task, texts),
        "natural_eos_fraction": float(np.mean(eos_flags)) if len(eos_flags) else 0.0,
    }


def run_source_self_training(
    task,
    rounds=32,
    B=64,
    G=16,
    seed=0,
    lr=1e-5,
    epochs=1,
    model_name=MODEL_NAME,
    model_tok=None,
    micro=4,
    self_training_mode="rft_source",
    self_train_iterations=1,
    accepted_per_question=10,
    generation_temperature=0.7,
    generation_top_k=40,
    proposal_prompt="question",
    reward_requires_eos=True,
    answer_target_termination="eos",
    question_sampling="epoch_shuffle",
    eval_every=0,
    eval_rounds=None,
    eval_fn=None,
    diagnostics_fn=None,
    checkpoint_every=0,
    checkpoint_fn=None,
    log=print,
):
    """Run RFT, ReST-EM or STaR under the fixed common-protocol design."""

    if self_training_mode not in SELF_TRAINING_MODES:
        raise ValueError(f"unknown self-training mode {self_training_mode!r}")
    if rounds <= 0 or B <= 0 or G <= 0:
        raise ValueError("rounds, B and G must be positive")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    if proposal_prompt != "question":
        raise ValueError("source self-training requires answer-blind direct proposals")
    if answer_target_termination != "eos" or not reward_requires_eos:
        raise ValueError("common-protocol self-training requires natural terminal EOS")
    if question_sampling != "epoch_shuffle":
        raise ValueError("source self-training requires the exact phase-balanced schedule")
    expected_iterations = {"rft_source": {1}, "rest_em": {2, 4}, "star": {4}}
    if int(self_train_iterations) not in expected_iterations[self_training_mode]:
        raise ValueError(
            f"{self_training_mode} does not permit {self_train_iterations} iterations"
        )

    model, tok = (
        model_tok
        if model_tok is not None
        else load_model(seed=seed, model=model_name)
    )
    initial_trainable = _snapshot_trainable(model)
    schedule = phase_question_schedule(
        len(task.prompts), rounds, self_train_iterations, seed
    )
    questions_per_round = len(schedule[0])
    if self_training_mode == "star":
        if G != 1 or B != questions_per_round:
            raise ValueError("STaR requires one direct draw for every scheduled question")
    elif B != questions_per_round * G:
        raise ValueError(
            "RFT/ReST B must equal scheduled questions per round times G"
        )

    rng = np.random.default_rng(seed)
    training_rng = np.random.default_rng(int(seed) * 1009 + 7)
    gold_batch = build_gold_batch(tok, task)
    phase_examples: list[TrainingExample] = []
    records: list[dict] = []
    unique_questions: set[int] = set()
    generated = generated_tokens = question_exposures = 0
    backward_tokens = backward_eos_tokens = optimizer_steps = 0
    phase_index = 0

    for round_index in _rounds(rounds):
        pids = list(schedule[round_index])
        unique_questions.update(pids)
        question_exposures += len(pids)
        phase_index = round_index // (rounds // self_train_iterations)

        if self_training_mode == "star":
            direct_ids, direct_masks, direct_texts = sample_multi(
                model,
                tok,
                [task.prompts[pid] for pid in pids],
                max_new=getattr(task, "max_new", 40),
                do_sample=False,
            )
            direct_eos = natural_eos_mask(direct_ids, direct_masks, tok.eos_token_id)
            direct_rewards = np.asarray(task.reward(direct_texts, pids=pids))
            direct_correct = _correct_mask(
                task, direct_texts, pids, direct_eos, reward_requires_eos
            )
            for row in select_correct_rows(pids, direct_correct, 1):
                phase_examples.append(
                    _reanchor_completion(
                        tok, task, pids[row], direct_ids[row], direct_masks[row], "direct"
                    )
                )

            failed_rows = [row for row, ok in enumerate(direct_correct) if not ok]
            hint_texts: list[str] = []
            hint_rewards = np.empty(0, dtype=np.float64)
            hint_eos = np.empty(0, dtype=bool)
            hint_masks = torch.empty((0, 0), dtype=torch.bool)
            if failed_rows:
                failed_pids = [pids[row] for row in failed_rows]
                task_builder = getattr(task, "build_proposal_prompt", None)
                hinted_prompts = [
                    (
                        task_builder(int(pid), "answer_derive")
                        if task_builder is not None
                        else build_proposal_prompt(
                            task.prompts[pid],
                            task.gold_answer[pid],
                            "answer_derive",
                        )
                    )
                    for pid in failed_pids
                ]
                hint_ids, hint_masks, hint_texts = sample_multi(
                    model,
                    tok,
                    hinted_prompts,
                    max_new=getattr(task, "max_new", 40),
                    do_sample=False,
                )
                hint_eos = natural_eos_mask(
                    hint_ids, hint_masks, tok.eos_token_id
                )
                hint_rewards = np.asarray(task.reward(hint_texts, pids=failed_pids))
                hint_correct = _correct_mask(
                    task,
                    hint_texts,
                    failed_pids,
                    hint_eos,
                    reward_requires_eos,
                )
                for row in select_correct_rows(failed_pids, hint_correct, 1):
                    phase_examples.append(
                        _reanchor_completion(
                            tok,
                            task,
                            failed_pids[row],
                            hint_ids[row],
                            hint_masks[row],
                            "answer_rationalized",
                        )
                    )
            texts = list(direct_texts) + list(hint_texts)
            rewards = np.concatenate((direct_rewards, hint_rewards))
            eos_flags = np.concatenate((direct_eos, hint_eos))
            round_generated = len(texts)
            round_tokens = int(direct_masks.sum()) + int(hint_masks.sum())
            last_ids, last_masks = direct_ids, direct_masks
            summary = {
                "mean_reward": float(np.mean(rewards)),
                "frac_correct": float(np.mean(rewards > 0.5)),
                "gen_len": float(round_tokens / max(round_generated, 1)),
                "fmt": task_format_rate(task, texts),
                "natural_eos_fraction": float(np.mean(eos_flags)),
            }
        else:
            pid_rows = np.repeat(np.asarray(pids, dtype=int), G)
            sampled_ids, sampled_masks, texts = sample_multi(
                model,
                tok,
                [task.prompts[int(pid)] for pid in pid_rows],
                temperature=generation_temperature,
                top_k=generation_top_k,
                top_p=1.0,
                max_new=getattr(task, "max_new", 40),
            )
            eos_flags = natural_eos_mask(
                sampled_ids, sampled_masks, tok.eos_token_id
            )
            rewards = np.asarray(task.reward(texts, pids=pid_rows))
            correct = _correct_mask(
                task, texts, pid_rows, eos_flags, reward_requires_eos
            )
            selected = select_correct_rows(
                pid_rows, correct, accepted_per_question
            )
            for row in selected:
                phase_examples.append(
                    _reanchor_completion(
                        tok,
                        task,
                        int(pid_rows[row]),
                        sampled_ids[row],
                        sampled_masks[row],
                        "sampled_correct",
                    )
                )
            round_generated = len(texts)
            round_tokens = int(sampled_masks.sum())
            last_ids, last_masks = sampled_ids, sampled_masks
            summary = _records_generation_summary(
                task, texts, sampled_masks, rewards, eos_flags
            )

        generated += round_generated
        generated_tokens += round_tokens
        accepted_before_improve = len(phase_examples)
        phase_loss = float("nan")
        phase_steps = phase_backward = phase_backward_eos = 0
        improved = _phase_boundary(round_index, rounds, self_train_iterations)
        if improved:
            _restore_trainable(model, initial_trainable)
            optimizer = torch.optim.Adam(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=lr,
            )
            (
                phase_loss,
                phase_steps,
                phase_backward,
                phase_backward_eos,
            ) = _train_examples(
                model,
                tok,
                phase_examples,
                optimizer,
                epochs=epochs,
                micro=micro,
                rng=training_rng,
            )
            optimizer_steps += phase_steps
            backward_tokens += phase_backward
            backward_eos_tokens += phase_backward_eos

        evaluation = maybe_eval(
            model,
            round_index,
            rounds,
            eval_every,
            eval_fn,
            eval_rounds=eval_rounds,
        )
        kl = kl_from_base(model, last_ids, last_masks)
        record = {
            "round": round_index,
            "method": self_training_mode,
            "phase": phase_index,
            "phase_improved": improved,
            "oracle": generated,
            "verifier_calls": generated,
            "gen": generated,
            "llm_gen": generated,
            "generated_tokens": generated_tokens,
            "backward_tokens": backward_tokens,
            "backward_eos_tokens": backward_eos_tokens,
            "question_exposures": question_exposures,
            "unique_questions_seen": len(unique_questions),
            "questions_this_round": len(pids),
            "gsteps": optimizer_steps,
            "n_accept": accepted_before_improve,
            "phase_optimizer_steps": phase_steps,
            "phase_backward_tokens": phase_backward,
            "phase_backward_eos_tokens": phase_backward_eos,
            "reward_requires_eos": bool(reward_requires_eos),
            "loss": phase_loss,
            "gold_lp": gold_lp(model, gold_batch),
            "test_acc": evaluation,
            "kl": kl,
            **summary,
        }
        records.append(record)
        if diagnostics_fn is not None:
            diagnostics_fn(
                {
                    "schema_version": 1,
                    "method_family": "source_self_training",
                    "method": self_training_mode,
                    "answer_target_termination": answer_target_termination,
                    "round": round_index,
                    "completed_rounds": round_index + 1,
                    "phase": {
                        "index": phase_index,
                        "improve_after_round": improved,
                        "accepted_examples": accepted_before_improve,
                        "reset_to_original_adapter": improved,
                    },
                    "generation": {
                        "generations_this_round": round_generated,
                        "generations_cumulative": generated,
                        "generated_tokens_cumulative": generated_tokens,
                        "questions_this_round": len(pids),
                        "question_exposures_cumulative": question_exposures,
                        "unique_questions_seen": len(unique_questions),
                        "correct_fraction": summary["frac_correct"],
                        "format_fraction": summary["fmt"],
                        "natural_eos_fraction": summary["natural_eos_fraction"],
                    },
                    "reward": {
                        "requires_natural_eos": bool(reward_requires_eos),
                    },
                    "optimizer": {
                        "gradient_steps_this_round": phase_steps,
                        "gradient_steps_cumulative": optimizer_steps,
                        "backward_tokens_cumulative": backward_tokens,
                        "backward_eos_tokens_this_round": phase_backward_eos,
                        "backward_eos_tokens_cumulative": backward_eos_tokens,
                        "loss": None if not np.isfinite(phase_loss) else phase_loss,
                        "kl": kl,
                    },
                    "test_acc": None if not np.isfinite(evaluation) else evaluation,
                }
            )
        if improved:
            phase_examples = []
        if (
            checkpoint_fn is not None
            and checkpoint_every > 0
            and (round_index + 1) % checkpoint_every == 0
            and (round_index + 1) < rounds
        ):
            checkpoint_fn(model, round_index + 1)
        log(
            f"  [{self_training_mode} r{round_index:>3}] "
            f"gen={generated:>6} accepted={accepted_before_improve:>4} "
            f"improve={int(improved)} acc={summary['frac_correct']:.3f} "
            f"kl={kl:+.3f}"
        )
    return records


def run_gold_cot_sft(
    task,
    rounds=32,
    B=64,
    G=4,
    seed=0,
    lr=1e-5,
    epochs=1,
    model_name=MODEL_NAME,
    model_tok=None,
    micro=4,
    proposal_prompt="question",
    answer_target_termination="eos",
    question_sampling="epoch_shuffle",
    eval_every=0,
    eval_rounds=None,
    eval_fn=None,
    diagnostics_fn=None,
    checkpoint_every=0,
    checkpoint_fn=None,
    log=print,
):
    """Train on human GSM8K rationales with an explicit terminal EOS target."""

    if rounds <= 0 or B <= 0 or G <= 0 or B % G:
        raise ValueError("Gold-CoT-SFT requires positive rounds and B divisible by G")
    if proposal_prompt != "question" or answer_target_termination != "eos":
        raise ValueError("Gold-CoT-SFT requires question-only context and an EOS target")
    if question_sampling not in {"random", "epoch_shuffle"}:
        raise ValueError("unknown Gold-CoT-SFT question schedule")
    total_question_exposures = len(task.prompts) * int(epochs)
    if epochs < 1 or total_question_exposures % rounds:
        raise ValueError(
            "Gold-CoT-SFT epochs must give an integer question count per round"
        )
    questions_per_round = total_question_exposures // rounds
    if B // G != questions_per_round:
        raise ValueError(
            "Gold-CoT-SFT B/G must equal the full-dataset epoch schedule"
        )
    model, tok = (
        model_tok
        if model_tok is not None
        else load_model(seed=seed, model=model_name)
    )
    rng = np.random.default_rng(seed)
    sampler = QuestionSampler(
        range(len(task.prompts)),
        matched_question_rng(rng, seed, question_sampling),
        mode=question_sampling,
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )
    gold_batch = build_gold_batch(tok, task)
    records: list[dict] = []
    unique_questions: set[int] = set()
    question_exposures = backward_tokens = backward_eos_tokens = optimizer_steps = 0
    training_rng = np.random.default_rng(int(seed) * 1009 + 17)
    for round_index in _rounds(rounds):
        pids = [int(value) for value in sampler.sample(questions_per_round)]
        examples = [_gold_example(tok, task, pid) for pid in pids]
        unique_questions.update(pids)
        question_exposures += len(pids)
        loss, steps, round_backward, round_backward_eos = _train_examples(
            model,
            tok,
            examples,
            optimizer,
            epochs=1,
            micro=micro,
            rng=training_rng,
        )
        optimizer_steps += steps
        backward_tokens += round_backward
        backward_eos_tokens += round_backward_eos
        ids, spans = _pad_examples(tok, examples)
        evaluation = maybe_eval(
            model,
            round_index,
            rounds,
            eval_every,
            eval_fn,
            eval_rounds=eval_rounds,
        )
        kl = kl_from_base(model, ids, spans)
        record = {
            "round": round_index,
            "method": "gold_cot_sft",
            "oracle": 0,
            "verifier_calls": 0,
            "gen": 0,
            "llm_gen": 0,
            "generated_tokens": 0,
            "backward_tokens": backward_tokens,
            "backward_eos_tokens": backward_eos_tokens,
            "question_exposures": question_exposures,
            "unique_questions_seen": len(unique_questions),
            "questions_this_round": len(pids),
            "gsteps": optimizer_steps,
            "n_accept": len(examples),
            "loss": loss,
            "gold_lp": gold_lp(model, gold_batch),
            "test_acc": evaluation,
            "kl": kl,
        }
        records.append(record)
        if diagnostics_fn is not None:
            diagnostics_fn(
                {
                    "schema_version": 1,
                    "method_family": "gold_cot_sft",
                    "answer_target_termination": answer_target_termination,
                    "round": round_index,
                    "completed_rounds": round_index + 1,
                    "generation": {
                        "generations_this_round": 0,
                        "generations_cumulative": 0,
                        "questions_this_round": len(pids),
                        "question_exposures_cumulative": question_exposures,
                        "unique_questions_seen": len(unique_questions),
                    },
                    "optimizer": {
                        "gradient_steps_this_round": steps,
                        "gradient_steps_cumulative": optimizer_steps,
                        "backward_tokens_cumulative": backward_tokens,
                        "backward_eos_tokens_this_round": round_backward_eos,
                        "backward_eos_tokens_cumulative": backward_eos_tokens,
                        "loss": loss,
                        "kl": kl,
                    },
                    "test_acc": None if not np.isfinite(evaluation) else evaluation,
                }
            )
        if (
            checkpoint_fn is not None
            and checkpoint_every > 0
            and (round_index + 1) % checkpoint_every == 0
            and (round_index + 1) < rounds
        ):
            checkpoint_fn(model, round_index + 1)
        log(
            f"  [Gold-CoT-SFT r{round_index:>3}] "
            f"questions={question_exposures:>5} loss={loss:.4f} kl={kl:+.3f}"
        )
    return records


__all__ = [
    "SELF_TRAINING_MODES",
    "TrainingExample",
    "phase_question_schedule",
    "run_gold_cot_sft",
    "run_source_self_training",
    "select_correct_rows",
]
