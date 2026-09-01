"""Held-out benchmarks (the 'benchmark it' step of the experiment loop).

Method-agnostic: given a fine-tuned (model, tok), score it on a HELD-OUT set, separate from the
prompts any method trained on. This is the number we report and compare across PO / RL / EM.

  benchmark_gsm8k  -> configured validation/test accuracy (few-shot CoT, greedy decode)
  benchmark_imdb   -> win-rate vs the frozen base model, judged by a sentiment classifier

Greedy decode for determinism; uses the device-agnostic DEV from methods_lm.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import urllib.request

import numpy as np
import torch

from answer_events import parse_gsm8k_answer_event
from common import audit_chat_active_completions, has_answer_marker
from evaluate_qwen3_8b_math_base_calibration import build_prompts
from math_answer_events import math_answers_equivalent, parse_math_answer_event
from methods_lm import DEV
from prompting import build_proposal_prompt


@torch.no_grad()
def _generate(
    model,
    tok,
    prompts,
    max_new=256,
    batch=16,
    temperature=0.0,
    return_metadata=False,
):
    """Continuation for a list of prompts (left-padded). Greedy by default (deterministic scoring);
    temperature>0 samples instead (used only by pass@k). Returns completion strings."""
    model.eval()
    tok.padding_side = "left"
    sample_kw = ({"do_sample": True, "temperature": temperature, "top_k": 0, "top_p": 1.0}
                 if temperature and temperature > 0 else {"do_sample": False})
    outs = []
    metadata = []
    chat_runtime = bool(getattr(tok, "_vrl_math_chat_rendered_prompts", False))
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        if chat_runtime:
            enc = tok(
                chunk,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(DEV)
            eos = int(tok.eos_token_id)
            pad = int(tok.pad_token_id)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new,
                eos_token_id=eos,
                pad_token_id=pad,
                **sample_kw,
            )
        else:
            # Preserve the historical evaluator byte-for-byte in semantics:
            # tokenizer defaults and the model's own EOS declaration remain
            # authoritative outside the dedicated chat runtime.
            enc = tok(chunk, return_tensors="pt", padding=True).to(DEV)
            eos = tok.eos_token_id
            gen = model.generate(
                **enc,
                max_new_tokens=max_new,
                pad_token_id=tok.eos_token_id,
                **sample_kw,
            )
        continuation = gen[:, enc.input_ids.shape[1]:]
        outs += tok.batch_decode(continuation, skip_special_tokens=True)
        for row in continuation:
            eos_positions = (row == eos).nonzero(as_tuple=False)
            generated_eos = bool(len(eos_positions))
            generated_tokens = (
                int(eos_positions[0].item()) + 1
                if generated_eos else int(row.numel())
            )
            row_metadata = {
                "generated_eos": generated_eos,
                "generated_tokens_until_eos": generated_tokens,
                "hit_max_new_tokens": (
                    not generated_eos and generated_tokens >= int(max_new)
                ),
            }
            if chat_runtime:
                active = row[:generated_tokens]
                audit = getattr(tok, "_vrl_math_chat_generation_audit", None)
                if not isinstance(audit, dict):
                    raise ValueError("Qwen3 chat generation audit is not bound")
                eod = int(audit["eod_token_id"])
                generated_eod_before_eot = bool((active == eod).any())
                row_metadata.update(
                    generated_stop_token_id=(eos if generated_eos else None),
                    generated_eod_before_eot=generated_eod_before_eot,
                )
                audit_chat_active_completions(tok, [active])
            metadata.append(row_metadata)
    return (outs, metadata) if return_metadata else outs


def _rep4(text):
    """Fraction of REPEATED 4-grams (whitespace words) in a completion -- the degeneration/looping
    metric, computable on every compact eval_* record (the full text is only kept for the N dumps)."""
    ws = text.split()
    if len(ws) < 8:
        return 0.0
    grams = [tuple(ws[i:i + 4]) for i in range(len(ws) - 3)]
    return round(1.0 - len(set(grams)) / len(grams), 4)


# --------------------------------------------------------------------------- #
#  GSM8K: held-out test-split accuracy
# --------------------------------------------------------------------------- #
# ONE answer parser for train reward and eval scoring. benchmark.py used to carry its own copy and
# the two drifted: tasks._final_int tolerates degenerate markers ('#### ,' -> None) where the old
# copy raised ValueError, killing the whole benchmark pass after the cell had already trained.
from tasks import (  # noqa: E402
    GSM8K_DATASET_REVISION,
    _final_int,
    _gsm8k_gold,
    _gsm8k_train_validation_pools,
)


_GSM8K_PARTITION_SEED = 20260709
_GSM8K_TUNE_MAX = 400

SVAMP_DATASET_REVISION = "689d7ccac74b9983a2ac7cc3b264f441b99e7c53"
SVAMP_DATASET_URL = (
    "https://raw.githubusercontent.com/arkilpatel/SVAMP/"
    f"{SVAMP_DATASET_REVISION}/SVAMP.json"
)
SVAMP_DATASET_SHA256 = (
    "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e"
)
SVAMP_DATASET_ROWS = 1000


def _offline_mode_enabled():
    """Whether cluster policy forbids the pinned-URL fallback."""

    truthy = {"1", "true", "yes", "on"}
    return any(
        str(os.environ.get(name, "")).strip().lower() in truthy
        for name in (
            "HF_HUB_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        )
    )


def _read_svamp_payload():
    """Read the immutable SVAMP payload locally, or fetch its pinned URL online."""

    configured_path = os.environ.get("SVAMP_DATA_PATH")
    if configured_path:
        path = Path(configured_path).expanduser()
        try:
            return path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"cannot read SVAMP_DATA_PATH={str(path)!r}"
            ) from exc
    if _offline_mode_enabled():
        raise RuntimeError(
            "SVAMP_DATA_PATH is required when offline mode is enabled"
        )
    try:
        with urllib.request.urlopen(SVAMP_DATASET_URL, timeout=60) as response:
            return response.read()
    except OSError as exc:
        raise RuntimeError(
            f"failed to fetch pinned SVAMP payload from {SVAMP_DATASET_URL}"
        ) from exc


def _validate_svamp_payload(raw_payload):
    """Verify the immutable bytes and normalize all 1,000 official examples."""

    if not isinstance(raw_payload, (bytes, bytearray)):
        raise TypeError("SVAMP payload must be bytes")
    digest = hashlib.sha256(raw_payload).hexdigest()
    if digest != SVAMP_DATASET_SHA256:
        raise ValueError(
            "SVAMP payload SHA-256 mismatch: "
            f"expected {SVAMP_DATASET_SHA256}, got {digest}"
        )
    try:
        payload = json.loads(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SVAMP payload is not valid UTF-8 JSON") from exc
    if not isinstance(payload, list) or len(payload) != SVAMP_DATASET_ROWS:
        observed = len(payload) if isinstance(payload, list) else type(payload).__name__
        raise ValueError(
            f"SVAMP payload must contain exactly {SVAMP_DATASET_ROWS} rows, got {observed}"
        )

    rows = []
    seen_ids = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"SVAMP row {index} is not an object")
        dataset_id = item.get("ID")
        if isinstance(dataset_id, bool) or not isinstance(dataset_id, (str, int)):
            raise ValueError(f"SVAMP row {index} has an invalid ID")
        dataset_id = str(dataset_id).strip()
        if not dataset_id or dataset_id in seen_ids:
            raise ValueError(
                f"SVAMP row {index} has an empty or duplicate ID {dataset_id!r}"
            )
        seen_ids.add(dataset_id)

        body = item.get("Body")
        question = item.get("Question")
        if not isinstance(body, str) or not body.strip():
            raise ValueError(f"SVAMP row {index} has an invalid Body")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"SVAMP row {index} has an invalid Question")

        answer = item.get("Answer")
        if (
            isinstance(answer, bool)
            or not isinstance(answer, (int, float))
            or not math.isfinite(float(answer))
            or not float(answer).is_integer()
        ):
            raise ValueError(
                f"SVAMP row {index} gold answer is not an integer: {answer!r}"
            )
        rows.append(
            {
                "dataset_index": index,
                "dataset_id": dataset_id,
                "question": f"{body.strip()} {question.strip()}",
                "gold": int(answer),
            }
        )
    if len(seen_ids) != SVAMP_DATASET_ROWS:
        raise ValueError("SVAMP payload does not contain 1,000 unique IDs")
    return rows


def load_svamp_transfer_records():
    """Public preflight loader: checksum, cardinality, ID, and gold validation."""

    return _validate_svamp_payload(_read_svamp_payload())


def _load_svamp_dataset():
    """Backward-compatible internal name used by the benchmark path."""

    return load_svamp_transfer_records()


def _gsm8k_eval_pool(n_items, eval_partition="all"):
    """Fixed disjoint pools for hyperparameter tuning and final reporting."""
    if eval_partition == "all":
        return np.arange(n_items, dtype=int)
    if eval_partition not in ("tune", "final"):
        raise ValueError(f"unknown GSM8K eval partition {eval_partition!r}")
    order = np.random.default_rng(_GSM8K_PARTITION_SEED).permutation(n_items)
    cut = min(_GSM8K_TUNE_MAX, max(1, n_items // 3))
    return order[:cut] if eval_partition == "tune" else order[cut:]


def _gsm8k_demonstrations(train, n_shots, seed, shot_bank_size, pool):
    """Build the shared seeded GSM8K training-only demonstration preamble."""

    n_shots = int(n_shots)
    if n_shots < 0:
        raise ValueError("n_shots must be nonnegative")
    if shot_bank_size is None:
        shot_bank_size = n_shots
    shot_bank_size = int(shot_bank_size)
    if shot_bank_size < n_shots:
        raise ValueError(
            f"shot_bank_size must be at least n_shots, got {shot_bank_size} < {n_shots}"
        )
    pool = np.asarray(pool, dtype=int)
    if shot_bank_size > len(pool):
        raise ValueError(
            f"shot_bank_size={shot_bank_size} exceeds GSM8K demonstration pool size {len(pool)}"
        )
    rng = np.random.default_rng(seed)
    shot_bank = rng.permutation(pool)[:shot_bank_size]
    shot_idx = shot_bank[:n_shots]
    shots = "".join(
        f"Question: {train[int(i)]['question']}\nAnswer: {train[int(i)]['answer']}\n\n"
        for i in shot_idx
    )
    return shots, shot_idx, shot_bank_size, rng


def _gsm8k_heldout(
    n_test,
    n_shots,
    seed,
    eval_partition="all",
    shot_bank_size=None,
    return_access_metadata=False,
):
    """Build aligned greedy/pass@k data under either legacy or train/validation/test protocols."""
    from datasets import load_dataset
    valid_partitions = {"all", "tune", "final", "validation", "test"}
    if eval_partition not in valid_partitions:
        raise ValueError(f"unknown GSM8K eval partition {eval_partition!r}")
    train = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=GSM8K_DATASET_REVISION,
    )
    test = None
    loaded_splits = ["train"]
    if eval_partition != "validation":
        test = load_dataset(
            "openai/gsm8k",
            "main",
            split="test",
            revision=GSM8K_DATASET_REVISION,
        )
        loaded_splits.append("test")
    if eval_partition in ("validation", "test"):
        train_pool, validation_pool = _gsm8k_train_validation_pools(len(train))
        demonstration_pool = train_pool
        if eval_partition == "validation":
            eval_split = train
            pool = validation_pool
        else:
            assert test is not None
            eval_split = test
            pool = np.arange(len(test), dtype=int)
    else:
        assert test is not None
        demonstration_pool = np.arange(len(train), dtype=int)
        eval_split = test
        pool = _gsm8k_eval_pool(len(test), eval_partition)
    shots, _shot_idx, _shot_bank_size, rng = _gsm8k_demonstrations(
        train,
        n_shots,
        seed,
        shot_bank_size,
        demonstration_pool,
    )
    if n_test > len(pool):
        raise ValueError(f"n_test={n_test} exceeds GSM8K {eval_partition} pool size {len(pool)}")
    idx = rng.permutation(pool)[:n_test]
    prompts = [shots + f"Question: {eval_split[int(i)]['question']}\nAnswer:" for i in idx]
    gold_answer = [_gsm8k_gold(eval_split[int(i)]["answer"]) for i in idx]
    result = (idx, prompts, gold_answer, eval_split)
    if not return_access_metadata:
        return result
    return (
        *result,
        {
            "loaded_splits": loaded_splits,
            "official_test_accessed": "test" in loaded_splits,
            "eval_source_split": (
                "train" if eval_partition == "validation" else "test"
            ),
        },
    )


def _score_math_completions(
    *,
    indices,
    dataset_ids,
    questions,
    completions,
    gold_answers,
    generation_metadata,
    tok,
    answer_event_mode,
    evaluation_prompt,
    access,
    dataset_metadata=None,
):
    """Apply the shared GSM8K answer-event and EOS contract to math outputs."""

    legacy_events = [
        parse_gsm8k_answer_event(completion, mode="legacy")
        for completion in completions
    ]
    strict_events = [
        parse_gsm8k_answer_event(completion, mode="strict_terminal_marker")
        for completion in completions
    ]
    events = (
        strict_events
        if answer_event_mode == "strict_terminal_marker"
        else legacy_events
    )
    preds = [event.answer for event in events]
    correct = [pred == gold for pred, gold in zip(preds, gold_answers)]
    records = []
    common = dict(dataset_metadata or {})
    for (
        index,
        dataset_id,
        question,
        completion,
        gold,
        pred,
        ok,
        event,
        legacy_event,
        strict_event,
        generation,
    ) in zip(
        indices,
        dataset_ids,
        questions,
        completions,
        gold_answers,
        preds,
        correct,
        events,
        legacy_events,
        strict_events,
        generation_metadata,
    ):
        marked = has_answer_marker(completion)
        strict_failure = not event.strict_valid
        strict_correct = bool(strict_event.answer == gold)
        generated_eos = generation.get("generated_eos")
        has_nonempty_reasoning = bool(strict_event.reasoning.strip())
        record = dict(
            idx=int(index),
            question=question,
            completion=completion,
            gold=gold,
            pred=pred,
            correct=bool(ok),
            legacy_pred=legacy_event.answer,
            legacy_correct=bool(legacy_event.answer == gold),
            strict_pred=strict_event.answer,
            strict_correct=strict_correct,
            strict_correct_and_eos=bool(
                strict_correct and generated_eos is True
            ),
            has_nonempty_reasoning=has_nonempty_reasoning,
            strict_correct_with_reasoning=bool(
                strict_correct and has_nonempty_reasoning
            ),
            direct_answer_only=bool(
                strict_event.strict_valid and not has_nonempty_reasoning
            ),
            has_answer_marker=marked,
            format_failure=(
                strict_failure
                if answer_event_mode == "strict_terminal_marker"
                else not marked
            ),
            strict_format_failure=strict_failure,
            answer_parse_mode=event.parse_mode,
            answer_event_mode=answer_event_mode,
            evaluation_prompt=evaluation_prompt,
            answer_marker_count=event.marker_count,
            answer_marker_terminal=event.terminal_marker,
            official_test_accessed=access["official_test_accessed"],
            eval_source_split=access["eval_source_split"],
            dataset_splits_loaded=list(access["loaded_splits"]),
            generated_eos=generated_eos,
            generated_tokens_until_eos=generation.get(
                "generated_tokens_until_eos"
            ),
            hit_max_new_tokens=generation.get("hit_max_new_tokens"),
            len=len(tok(completion, add_special_tokens=False).input_ids),
            rep4=_rep4(completion),
            **common,
        )
        if dataset_id is not None:
            record["dataset_id"] = str(dataset_id)
        records.append(record)
    return float(np.mean(correct)), records


def benchmark_gsm8k(
    model,
    tok,
    n_test=200,
    n_shots=4,
    seed=0,
    max_new=256,
    batch=16,
    return_records=False,
    eval_partition="all",
    shot_bank_size=None,
    answer_event_mode="legacy",
    evaluation_prompt="question",
):
    """Accuracy on held-out GSM8K validation or test questions.

    return_records=True also returns per-example dicts (question/completion/gold/pred/correct) from the
    SAME greedy pass that produced the score -- used by --dump-completions to eyeball whether a low
    number is genuine degradation (garbage/short reasoning) or just a parse miss."""
    idx, prompts, gold_answer, eval_split, access = _gsm8k_heldout(
        n_test,
        n_shots,
        seed,
        eval_partition=eval_partition,
        shot_bank_size=shot_bank_size,
        return_access_metadata=True,
    )
    prompts = [
        build_proposal_prompt(prompt, answer, evaluation_prompt)
        for prompt, answer in zip(prompts, gold_answer)
    ]
    generated = _generate(
        model,
        tok,
        prompts,
        max_new=max_new,
        batch=batch,
        return_metadata=True,
    )
    if (
        isinstance(generated, tuple)
        and len(generated) == 2
    ):
        comps, generation_metadata = generated
    else:
        comps = generated
        generation_metadata = [{} for _ in comps]
    acc, recs = _score_math_completions(
        indices=idx,
        dataset_ids=[None] * len(idx),
        questions=[eval_split[int(i)]["question"] for i in idx],
        completions=comps,
        gold_answers=gold_answer,
        generation_metadata=generation_metadata,
        tok=tok,
        answer_event_mode=answer_event_mode,
        evaluation_prompt=evaluation_prompt,
        access=access,
    )
    if not return_records:
        return acc
    return acc, recs


def benchmark_hendrycks_math(
    model,
    tok,
    n_test=400,
    n_shots=4,
    max_new=512,
    batch=4,
    return_records=False,
    answer_event_mode="strict_terminal_marker",
    evaluation_prompt="question",
    _chat_runtime=False,
):
    """Greedy evaluation on the fixed train-derived MATH validation set."""

    from tasks import _hendrycks_math_chat_partition, _hendrycks_math_partition
    from math_prompting import (
        bind_math_chat_generation_runtime,
        build_math_chat_messages,
        render_math_chat_prompts,
        validate_math_model_eos,
    )

    partition_loader = (
        _hendrycks_math_chat_partition
        if _chat_runtime
        else _hendrycks_math_partition
    )
    protocol, _optimization, validation, demonstrations, sources = partition_loader()
    if evaluation_prompt != "question":
        raise ValueError("MATH validation requires the answer-blind question prompt")
    if not 1 <= int(n_test) <= len(validation):
        raise ValueError("MATH n_test exceeds the fixed validation partition")
    if not 0 <= int(n_shots) <= len(demonstrations):
        raise ValueError("MATH n_shots exceeds the fixed demonstration bank")
    rows = validation[:int(n_test)]
    prompt_version = str(protocol["prompt"]["version"])
    if _chat_runtime:
        bind_math_chat_generation_runtime(
            model,
            tok,
            version=prompt_version,
        )
        prompts = render_math_chat_prompts(
            tok,
            build_math_chat_messages(
                rows,
                demonstrations[:int(n_shots)],
                version=prompt_version,
            ),
            version=prompt_version,
        )
        max_new = int(protocol["generation"]["max_new_tokens"])
    else:
        validate_math_model_eos(
            model,
            tok,
            version=str(protocol["prompt"]["version"]),
        )
        prompts = build_prompts(
            rows,
            demonstrations[:int(n_shots)],
            version=str(protocol["prompt"]["version"]),
        )
    completions, generation_metadata = _generate(
        model,
        tok,
        prompts,
        max_new=max_new,
        batch=batch,
        return_metadata=True,
    )
    records = []
    for index, (row, completion, generation) in enumerate(
        zip(rows, completions, generation_metadata, strict=True)
    ):
        legacy = parse_math_answer_event(completion, mode="legacy")
        strict_event = parse_math_answer_event(
            completion,
            mode="strict_terminal_marker",
            disallowed_exact_answers=(("answer",) if _chat_runtime else ()),
        )
        extracted_correct = math_answers_equivalent(legacy.answer, row["gold"])
        strict_correct = bool(
            strict_event.strict_valid
            and math_answers_equivalent(strict_event.answer, row["gold"])
        )
        generated_eos = bool(generation.get("generated_eos"))
        has_reasoning = bool(strict_event.reasoning.strip())
        records.append(
            {
                "idx": index,
                "dataset_id": row["dataset_id"],
                "question": row["problem"],
                "gold": row["gold"],
                "pred": legacy.answer,
                "correct": bool(extracted_correct),
                "legacy_pred": legacy.answer,
                "legacy_correct": bool(extracted_correct),
                "strict_pred": strict_event.answer,
                "strict_correct": strict_correct,
                "strict_correct_and_eos": bool(strict_correct and generated_eos),
                "has_nonempty_reasoning": has_reasoning,
                "strict_correct_with_reasoning": bool(
                    strict_correct and has_reasoning
                ),
                "direct_answer_only": bool(strict_event.strict_valid and not has_reasoning),
                "has_answer_marker": strict_event.marker_count > 0,
                "format_failure": not strict_event.strict_valid,
                "strict_format_failure": not strict_event.strict_valid,
                "answer_parse_mode": (
                    strict_event.parse_mode
                    if answer_event_mode == "strict_terminal_marker"
                    else legacy.parse_mode
                ),
                "answer_event_mode": answer_event_mode,
                "evaluation_prompt": evaluation_prompt,
                "answer_marker_count": strict_event.marker_count,
                "answer_marker_terminal": strict_event.terminal_marker,
                "generated_eos": generated_eos,
                "generated_tokens_until_eos": generation.get(
                    "generated_tokens_until_eos"
                ),
                "hit_max_new_tokens": bool(
                    generation.get("hit_max_new_tokens")
                ),
                "official_test_accessed": False,
                "eval_source_split": "train",
                "dataset_splits_loaded": ["train"],
                "dataset": protocol["dataset"]["id"],
                "dataset_revision": protocol["dataset"]["revision"],
                "dataset_source_files": [source["filename"] for source in sources],
                "len": len(tok(completion, add_special_tokens=False).input_ids),
                "rep4": _rep4(completion),
                "completion": completion,
                **(
                    {
                        "generated_stop_token_id": generation.get(
                            "generated_stop_token_id"
                        ),
                        "generated_eod_before_eot": bool(
                            generation.get("generated_eod_before_eot")
                        ),
                    }
                    if _chat_runtime
                    else {}
                ),
            }
        )
    accuracy = float(np.mean([record["correct"] for record in records]))
    return (accuracy, records) if return_records else accuracy


def benchmark_hendrycks_math_chat(
    model,
    tok,
    n_test=400,
    n_shots=4,
    max_new=1024,
    batch=4,
    return_records=False,
    answer_event_mode="strict_terminal_marker",
    evaluation_prompt="question",
):
    """Greedy fixed-cap evaluation under the qualified Qwen3 chat runtime."""

    if int(max_new) != 1024:
        raise ValueError("Qwen3 MATH chat evaluation cap must remain 1,024")
    return benchmark_hendrycks_math(
        model,
        tok,
        n_test=n_test,
        n_shots=n_shots,
        max_new=max_new,
        batch=batch,
        return_records=return_records,
        answer_event_mode=answer_event_mode,
        evaluation_prompt=evaluation_prompt,
        _chat_runtime=True,
    )


def benchmark_svamp_transfer(
    model,
    tok,
    n_test=1000,
    n_shots=4,
    seed=0,
    max_new=256,
    batch=16,
    shot_bank_size=None,
    answer_event_mode="legacy",
    train_partition="train",
):
    """Evaluate a GSM8K-trained model on immutable SVAMP without training on it."""

    n_test = int(n_test)
    if not 1 <= n_test <= SVAMP_DATASET_ROWS:
        raise ValueError(
            f"n_test must be in [1, {SVAMP_DATASET_ROWS}], got {n_test}"
        )
    rows = _load_svamp_dataset()

    from datasets import load_dataset

    gsm8k_train = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=GSM8K_DATASET_REVISION,
    )
    if train_partition == "train":
        demonstration_pool, _validation_pool = _gsm8k_train_validation_pools(
            len(gsm8k_train)
        )
    elif train_partition == "all":
        demonstration_pool = np.arange(len(gsm8k_train), dtype=int)
    else:
        raise ValueError(
            f"unknown GSM8K train partition {train_partition!r}"
        )
    shots, shot_indices, actual_shot_bank_size, rng = _gsm8k_demonstrations(
        gsm8k_train,
        n_shots,
        seed,
        shot_bank_size,
        demonstration_pool,
    )

    selected_indices = rng.permutation(SVAMP_DATASET_ROWS)[:n_test]
    selected = [rows[int(index)] for index in selected_indices]
    gold_answers = [row["gold"] for row in selected]
    prompts = [
        build_proposal_prompt(
            shots + f"Question: {row['question']}\nAnswer:",
            row["gold"],
            "question",
        )
        for row in selected
    ]
    generated = _generate(
        model,
        tok,
        prompts,
        max_new=max_new,
        batch=batch,
        return_metadata=True,
    )
    if isinstance(generated, tuple) and len(generated) == 2:
        completions, generation_metadata = generated
    else:
        completions = generated
        generation_metadata = [{} for _ in completions]

    access = {
        "loaded_splits": ["svamp_full"],
        "official_test_accessed": False,
        "eval_source_split": "svamp_full",
    }
    accuracy, records = _score_math_completions(
        indices=[row["dataset_index"] for row in selected],
        dataset_ids=[row["dataset_id"] for row in selected],
        questions=[row["question"] for row in selected],
        completions=completions,
        gold_answers=gold_answers,
        generation_metadata=generation_metadata,
        tok=tok,
        answer_event_mode=answer_event_mode,
        evaluation_prompt="question",
        access=access,
        dataset_metadata={
            "dataset": "svamp",
            "dataset_revision": SVAMP_DATASET_REVISION,
            "dataset_sha256": SVAMP_DATASET_SHA256,
        },
    )
    metadata = {
        "dataset": "svamp",
        "dataset_revision": SVAMP_DATASET_REVISION,
        "dataset_url": SVAMP_DATASET_URL,
        "dataset_sha256": SVAMP_DATASET_SHA256,
        "dataset_rows": SVAMP_DATASET_ROWS,
        "evaluation_n": n_test,
        "evaluation_prompt": "question",
        "answer_event_mode": answer_event_mode,
        "eval_source_split": "svamp_full",
        "dataset_splits_loaded": ["svamp_full"],
        "official_test_accessed": False,
        "demonstration_source": "openai/gsm8k:train",
        "demonstration_dataset_revision": GSM8K_DATASET_REVISION,
        "demonstration_train_partition": train_partition,
        "demonstration_shots": int(n_shots),
        "demonstration_shot_indices": [int(index) for index in shot_indices],
        "demonstration_shot_bank_size": actual_shot_bank_size,
        "gsm8k_dataset_splits_loaded": ["train"],
        "gsm8k_official_test_accessed": False,
        "unique_ids_validated": True,
        "integer_golds_validated": True,
    }
    return accuracy, records, metadata


def passk_gsm8k(
    model,
    tok,
    n_test=100,
    k=8,
    n_shots=4,
    seed=0,
    temp=0.7,
    max_new=256,
    batch=16,
    eval_partition="all",
    shot_bank_size=None,
    answer_event_mode="legacy",
    evaluation_prompt="question",
):
    """pass@k on held-out GSM8K: k sampled completions (temperature `temp`) per question. Returns
    (summary, records); records = per-question {idx, gold, n_correct, k}. The RLVR sharpening check:
    pass@1 rising while pass@k falls = the method narrowed the distribution rather than adding ability.
    Question idx values are the SAME split/permutation as benchmark_gsm8k(seed) -> records align."""
    idx, prompts, gold_answer, _, access = _gsm8k_heldout(
        n_test,
        n_shots,
        seed,
        eval_partition=eval_partition,
        shot_bank_size=shot_bank_size,
        return_access_metadata=True,
    )
    prompts = [
        build_proposal_prompt(prompt, answer, evaluation_prompt)
        for prompt, answer in zip(prompts, gold_answer)
    ]
    n_correct = np.zeros(len(idx), dtype=int)
    for _ in range(k):                                     # k independent sampled passes
        comps = _generate(model, tok, prompts, max_new=max_new, batch=batch, temperature=temp)
        n_correct += np.array(
            [
                _final_int(c, answer_event_mode) == g
                for c, g in zip(comps, gold_answer)
            ],
            dtype=int,
        )
    summary = {
        "pass1": float((n_correct / k).mean()),     # unbiased pass@1 under sampling
        f"pass{k}": float((n_correct > 0).mean()),  # pass@k (any of the k correct)
        "passk_official_test_accessed": access["official_test_accessed"],
        "passk_eval_source_split": access["eval_source_split"],
        "passk_dataset_splits_loaded": list(access["loaded_splits"]),
        "passk_evaluation_prompt": evaluation_prompt,
    }
    records = [
        dict(
            idx=int(i),
            gold=g,
            n_correct=int(c),
            k=k,
            evaluation_prompt=evaluation_prompt,
            official_test_accessed=access["official_test_accessed"],
            eval_source_split=access["eval_source_split"],
            dataset_splits_loaded=list(access["loaded_splits"]),
        )
        for i, g, c in zip(idx, gold_answer, n_correct)
    ]
    return summary, records


# --------------------------------------------------------------------------- #
#  IMDB sentiment: win-rate vs the frozen base, classifier-judged
# --------------------------------------------------------------------------- #
_SENTIMENT = {}


def _sentiment_scores(texts, model_name="lvwerra/distilbert-imdb", batch=32):
    """P(positive) per text from a frozen sentiment classifier (cached)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if model_name not in _SENTIMENT:
        clf = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEV).eval()
        ctok = AutoTokenizer.from_pretrained(model_name)
        _SENTIMENT[model_name] = (clf, ctok)
    clf, ctok = _SENTIMENT[model_name]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = ctok(texts[i:i + batch], return_tensors="pt", padding=True,
                       truncation=True, max_length=256).to(DEV)
            p = torch.softmax(clf(**enc).logits, -1)[:, 1]          # label 1 = positive
            out += p.cpu().tolist()
    return np.array(out)


def benchmark_imdb(model, tok, prompts, max_new=64, batch=16):
    """Win-rate vs frozen base: fraction of held-out prompts where the fine-tuned policy's
    completion is judged more positive than the base model's (same greedy decode)."""
    fine = _generate(model, tok, prompts, max_new=max_new, batch=batch)
    with model.disable_adapter():
        base = _generate(model, tok, prompts, max_new=max_new, batch=batch)
    s_fine = _sentiment_scores([p + c for p, c in zip(prompts, fine)])
    s_base = _sentiment_scores([p + c for p, c in zip(prompts, base)])
    return float(np.mean(s_fine > s_base)), float(s_fine.mean())
