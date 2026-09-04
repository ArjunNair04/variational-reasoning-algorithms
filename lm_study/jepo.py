"""Multi-sample JEPO for answer-labelled language-model reasoning.

The update follows Tang et al.'s multi-sample Jensen lower bound: sampled
reasoning traces receive a leave-one-out evidence advantage, while the known
answer receives the gradient of a log-average probability.  A separate format
advantage and frozen-base KL penalty use the same sampled generations.
"""

from __future__ import annotations

import numpy as np
import torch

from ac_alg1 import _pad_trace_rows, _sampled_trace_row
from common import (
    DEV,
    MODEL_NAME,
    QuestionSampler,
    _rounds,
    build_gold_batch,
    comp_len,
    frac_correct,
    gold_lp,
    kl_from_base,
    load_model,
    maybe_eval,
    natural_eos_mask,
    resolve_question_schedule_rng,
    sample_round,
    task_format_rate,
    token_logps,
)
from variational_reasoning.jepo import (
    fixed_masked_coefficients,
    jepo_multisample_terms,
    leave_one_out_advantages,
    standardize_and_clip,
)


def _strict_format_mask(task, texts, natural_eos):
    parser = getattr(task, "parse_answer_event", None)
    if parser is None:
        raise ValueError("JEPO requires the task's strict answer-event parser")
    strict = np.asarray(
        [
            bool(parser(text, mode="strict_terminal_marker").strict_valid)
            for text in texts
        ],
        dtype=bool,
    )
    return strict & np.asarray(natural_eos, dtype=bool)


def _group_logmeanexp(values, question_ids):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(question_ids)
    results = []
    for group in dict.fromkeys(groups.tolist()):
        local = values[groups == group]
        maximum = float(local.max())
        results.append(maximum + float(np.log(np.exp(local - maximum).mean())))
    return float(np.mean(results)) if results else float("nan")


def _token_logps_in_chunks(model, ids, micro, *, grad=False):
    return torch.cat(
        [
            token_logps(model, ids[start : start + micro], grad=grad)
            for start in range(0, ids.shape[0], micro)
        ]
    )


def run_jepo(
    task,
    rounds=32,
    B=64,
    G=4,
    seed=0,
    lr=1e-5,
    kl_coef=1e-3,
    jepo_supervised_coef=1e-2,
    jepo_format_penalty=10.0,
    jepo_advantage_clip=1.0,
    model_name=MODEL_NAME,
    model_tok=None,
    micro=4,
    eval_every=0,
    eval_fn=None,
    eval_rounds=None,
    diagnostics_fn=None,
    checkpoint_every=0,
    checkpoint_fn=None,
    question_sampling="random",
    question_schedule_rng="dedicated",
    proposal_prompt="question",
    proposal_temperature=1.0,
    reward_requires_eos=True,
    answer_event_mode="strict_terminal_marker",
    answer_target_termination="eos",
    log=print,
):
    """Train with one paper-derived multi-sample JEPO update per round."""

    if G < 2 or B < G or B % G:
        raise ValueError("JEPO requires B divisible by a group size G >= 2")
    if lr <= 0 or kl_coef < 0 or jepo_supervised_coef < 0:
        raise ValueError("JEPO learning rate and loss coefficients are invalid")
    if jepo_format_penalty <= 0 or jepo_advantage_clip <= 0:
        raise ValueError("JEPO format penalty and advantage clip must be positive")
    if proposal_prompt != "question":
        raise ValueError(
            "JEPO samples the ordinary question-conditioned rationale prior"
        )
    if proposal_temperature <= 0:
        raise ValueError("JEPO proposal temperature must be positive")
    if answer_event_mode != "strict_terminal_marker":
        raise ValueError("JEPO requires strict terminal answer-event segmentation")
    if answer_target_termination != "eos":
        raise ValueError(
            "JEPO requires the strict gold answer followed by tokenizer EOS"
        )
    if not reward_requires_eos:
        raise ValueError(
            "JEPO's common protocol requires natural EOS for valid generations"
        )
    if not hasattr(task, "gold_answer"):
        raise ValueError("JEPO requires a known answer for every training question")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")

    model, tok = (
        model_tok if model_tok is not None else load_model(seed=seed, model=model_name)
    )
    opt = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )
    rng = np.random.default_rng(seed)
    sampler = QuestionSampler(
        range(len(task.prompts)),
        resolve_question_schedule_rng(
            rng,
            seed,
            question_sampling,
            question_schedule_rng,
        ),
        mode=question_sampling,
    )
    gold_batch = build_gold_batch(tok, task)
    records = []
    generations = generated_tokens = backward_tokens = gradient_steps = 0
    question_exposures = 0
    unique_questions = set()

    for outer_round in _rounds(rounds):
        pids, pid_row, ids, mask, rewards, texts = sample_round(
            model,
            tok,
            task,
            B,
            G,
            rng,
            question_sampler=sampler,
            proposal_prompt=proposal_prompt,
            reward_requires_eos=reward_requires_eos,
            proposal_temperature=proposal_temperature,
        )
        generations += B
        generated_tokens += int(mask.sum())
        question_exposures += len(pids)
        unique_questions.update(int(pid) for pid in pids)

        natural_eos = natural_eos_mask(ids, mask, tok.eos_token_id)
        format_valid = _strict_format_mask(task, texts, natural_eos)
        format_rewards = np.where(format_valid, 0.0, -jepo_format_penalty)
        raw_format_advantage = leave_one_out_advantages(format_rewards, pid_row)
        format_advantage = standardize_and_clip(
            raw_format_advantage,
            clip=jepo_advantage_clip,
        )

        rows = []
        row_question_ids = []
        for index in np.flatnonzero(format_valid):
            row = _sampled_trace_row(
                tok,
                task,
                int(pid_row[index]),
                ids[index].detach().cpu(),
                mask[index].detach().cpu(),
                texts[index],
                outer_round,
                "jepo_question_prior",
                trace_id=f"r{outer_round}:g{index}",
                answer_event_mode=answer_event_mode,
                answer_target_termination=answer_target_termination,
            )
            if row is None:
                raise ValueError("strict-valid JEPO generation could not be segmented")
            rows.append(row)
            row_question_ids.append(int(pid_row[index]))

        answer_logp_old = np.empty(0, dtype=np.float64)
        answer_weights = np.empty(0, dtype=np.float64)
        trace_advantage = np.empty(0, dtype=np.float64)
        raw_trace_advantage = np.empty(0, dtype=np.float64)
        active_groups = 0
        row_ids = row_span = row_answer = None
        if rows:
            row_ids, row_span, row_answer = _pad_trace_rows(tok, rows)
            with torch.no_grad():
                answer_logp_old = (
                    _token_logps_in_chunks(model, row_ids, micro)
                    .mul(row_answer[:, 1:].float())
                    .sum(1)
                    .cpu()
                    .numpy()
                )
            terms = jepo_multisample_terms(
                answer_logp_old,
                row_question_ids,
                advantage_clip=jepo_advantage_clip,
            )
            answer_weights = terms.answer_weights
            trace_advantage = terms.trace_advantages
            raw_trace_advantage = terms.raw_trace_advantages
            active_groups = terms.active_groups

        completion_token_mask = mask[:, 1:].float()
        with torch.no_grad():
            old_token_logp = _token_logps_in_chunks(model, ids, micro)
            with model.disable_adapter():
                reference_token_logp = _token_logps_in_chunks(model, ids, micro)
        sample_logp = float(
            (
                (old_token_logp * completion_token_mask).sum(1)
                / completion_token_mask.sum(1).clamp(min=1)
            ).mean()
        )

        opt.zero_grad()
        lower_bound_loss = format_loss = kl_loss = 0.0
        if rows:
            trace_values, answer_values = fixed_masked_coefficients(
                terms,
                question_count=len(pids),
                samples_per_question=G,
                supervised_coefficient=jepo_supervised_coef,
            )
            trace_coefficients = torch.tensor(
                trace_values,
                dtype=torch.float32,
                device=DEV,
            )
            answer_coefficients = torch.tensor(
                answer_values,
                dtype=torch.float32,
                device=DEV,
            )
            trace_mask = row_span & ~row_answer
            for start in range(0, len(rows), micro):
                stop = start + micro
                token_lp = token_logps(model, row_ids[start:stop], grad=True)
                trace_lp = (token_lp * trace_mask[start:stop, 1:].float()).sum(1)
                answer_lp = (token_lp * row_answer[start:stop, 1:].float()).sum(1)
                loss = (
                    -(trace_coefficients[start:stop] * trace_lp).sum()
                    - (answer_coefficients[start:stop] * answer_lp).sum()
                )
                loss.backward()
                lower_bound_loss += float(loss.detach())
                backward_tokens += int(row_span[start:stop].sum())

        format_coefficients = torch.tensor(
            format_advantage / (len(pids) * G),
            dtype=torch.float32,
            device=DEV,
        )
        total_completion_tokens = completion_token_mask.sum().clamp(min=1)
        for start in range(0, B, micro):
            stop = start + micro
            current_lp = token_logps(model, ids[start:stop], grad=True)
            token_mask = completion_token_mask[start:stop]
            sequence_lp = (current_lp * token_mask).sum(1)
            format_component = -(format_coefficients[start:stop] * sequence_lp).sum()
            log_ratio = (reference_token_logp[start:stop] - current_lp).clamp(-5, 5)
            kl_k3 = torch.exp(log_ratio) - log_ratio - 1.0
            kl_component = (
                kl_coef * (kl_k3 * token_mask).sum() / total_completion_tokens
            )
            (format_component + kl_component).backward()
            format_loss += float(format_component.detach())
            kl_loss += float(kl_component.detach())
            backward_tokens += int(token_mask.sum())

        opt.step()
        gradient_steps += 1

        policy_kl = kl_from_base(model, ids, mask)
        gold_reasoning_logp = gold_lp(model, gold_batch)
        test_accuracy = maybe_eval(
            model,
            outer_round,
            rounds,
            eval_every,
            eval_fn,
            eval_rounds=eval_rounds,
        )
        format_fraction = task_format_rate(task, texts)
        raw_trace_std = (
            float(raw_trace_advantage.std()) if len(raw_trace_advantage) else 0.0
        )
        answer_ess = (
            float(
                np.mean(
                    [
                        1.0
                        / np.sum(
                            answer_weights[np.asarray(row_question_ids) == pid] ** 2
                        )
                        for pid in dict.fromkeys(row_question_ids)
                    ]
                )
            )
            if len(answer_weights)
            else 0.0
        )
        record = {
            "round": outer_round,
            "oracle": 0,
            "verifier_calls": 0,
            "gen": generations,
            "llm_gen": generations,
            "generated_tokens": generated_tokens,
            "backward_tokens": backward_tokens,
            "question_exposures": question_exposures,
            "unique_questions_seen": len(unique_questions),
            "question_ids_this_round": [int(pid) for pid in pids],
            "questions_this_round": len(pids),
            "gsteps": gradient_steps,
            "mean_reward": float(rewards.mean()),
            "frac_correct": frac_correct(rewards),
            "gen_len": comp_len(mask),
            "fmt": format_fraction,
            "natural_eos_fraction": float(natural_eos.mean()),
            "jepo_valid_fraction": float(format_valid.mean()),
            "jepo_valid_groups": active_groups,
            "jepo_raw_advantage_std": raw_trace_std,
            "jepo_advantage_clip_fraction": float(
                np.mean(np.abs(trace_advantage) >= jepo_advantage_clip)
            )
            if len(trace_advantage)
            else 0.0,
            "jepo_answer_ess": answer_ess,
            "jepo_logmean_answer_probability": _group_logmeanexp(
                answer_logp_old, row_question_ids
            )
            if rows
            else 0.0,
            "proposal_prompt": proposal_prompt,
            "proposal_temperature": proposal_temperature,
            "question_schedule_rng": question_schedule_rng,
            "reward_requires_eos": reward_requires_eos,
            "jepo_supervised_coef": jepo_supervised_coef,
            "jepo_format_penalty": jepo_format_penalty,
            "jepo_advantage_clip": jepo_advantage_clip,
            "samp_lp": sample_logp,
            "gold_lp": gold_reasoning_logp,
            "test_acc": test_accuracy,
            "loss": lower_bound_loss + format_loss + kl_loss,
            "jepo_lower_bound_loss": lower_bound_loss,
            "jepo_format_loss": format_loss,
            "jepo_kl_loss": kl_loss,
            "kl": policy_kl,
        }
        records.append(record)
        if diagnostics_fn is not None:
            diagnostics_fn(
                {
                    "schema_version": 1,
                    "method_family": "jepo_multisample",
                    "round": outer_round,
                    "completed_rounds": outer_round + 1,
                    "generation": {
                        "generations_this_round": B,
                        "generations_cumulative": generations,
                        "generated_tokens_cumulative": generated_tokens,
                        "questions_this_round": len(pids),
                        "question_ids_this_round": [int(pid) for pid in pids],
                        "question_exposures_cumulative": question_exposures,
                        "unique_questions_seen": len(unique_questions),
                        "correct_fraction": frac_correct(rewards),
                        "format_fraction": format_fraction,
                        "natural_eos_fraction": float(natural_eos.mean()),
                    },
                    "signal": {
                        "valid_generation_fraction": float(format_valid.mean()),
                        "valid_question_groups": active_groups,
                        "raw_trace_advantage_std": raw_trace_std,
                        "normalized_trace_advantage_std": float(trace_advantage.std())
                        if len(trace_advantage)
                        else 0.0,
                        "trace_advantage_clip_fraction": record[
                            "jepo_advantage_clip_fraction"
                        ],
                        "answer_weight_ess": answer_ess,
                        "logmean_gold_answer_probability": record[
                            "jepo_logmean_answer_probability"
                        ],
                    },
                    "optimizer": {
                        "gradient_steps_this_round": 1,
                        "gradient_steps_cumulative": gradient_steps,
                        "backward_tokens_cumulative": backward_tokens,
                        "loss": record["loss"],
                        "lower_bound_loss": lower_bound_loss,
                        "format_loss": format_loss,
                        "kl_loss": kl_loss,
                        "sampled_policy_kl": policy_kl,
                    },
                    "contract": {
                        "group_size": G,
                        "supervised_coefficient": jepo_supervised_coef,
                        "format_penalty": jepo_format_penalty,
                        "advantage_clip": jepo_advantage_clip,
                        "kl_coefficient": kl_coef,
                        "answer_target_termination": answer_target_termination,
                        "proposal_prompt": proposal_prompt,
                        "proposal_temperature": proposal_temperature,
                        "masked_objective_denominator": "fixed_sample_count",
                    },
                    "test_acc": None
                    if not np.isfinite(test_accuracy)
                    else test_accuracy,
                }
            )
        if (
            checkpoint_fn is not None
            and checkpoint_every > 0
            and (outer_round + 1) % checkpoint_every == 0
            and (outer_round + 1) < rounds
        ):
            checkpoint_fn(model, outer_round + 1)
        log(
            f"  [JEPO K{G} r{outer_round:>3}] gen={generations:>6} "
            f"valid={format_valid.mean():.3f} R={rewards.mean():.3f} "
            f"len={comp_len(mask):.0f} kl={policy_kl:+.3f}"
        )
    return records
