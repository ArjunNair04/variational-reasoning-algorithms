"""PO-vs-RL-vs-EM sweep driver (docs/experiments/po_vs_rl/po_vs_rl_vs_em_study_design.md) — the nested experiment loop:

    for seed: for base model: for fine-tuning method: train -> BENCHMARK on held-out.

Multi-prompt tasks (gsm8k, imdb). The driver OWNS the model (creates it, passes via model_tok),
so after training it can run the held-out benchmark on that exact model. Outputs go to --out
(point this at HOME on the cluster -- /scratch0 is wiped at session end). CSV is checkpointed after
every (seed, model, method) cell.

    python run_sweep_lm.py --task gsm8k --models qwen2.5-1.5b-instruct \
        --methods AC-EM GRPO --seeds 3 --rounds 60 --G 8 --prompts 64 \
        --out ~/po_vs_rl_results --tag run1

Conditional VRO + weighted-EM family (AW/AAW/RW-EM) + AC-EM (eq-8) + RL/PO comparators
(GRPO, RLOO, RAFT, DPO, APL).
"""
from __future__ import annotations
import argparse
import functools
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import secrets
import time
from typing import TextIO, TypedDict, cast

import numpy as np
import pandas as pd
import torch

from tasks import (
    GSM8KTask,
    HendrycksMathChatTask,
    HendrycksMathTask,
    IMDBSentimentTask,
)
from methods import (
    run_ac_alg1,
    run_ac_dpo,
    run_ac_em,
    run_apl,
    run_conditional_vro,
    run_dpo,
    run_grpo,
    run_l2r,
    run_raft,
    run_rloo,
    run_weighted_em,
)
from methods_lm import (
    load_model,
    resolve_lora_target_modules,
    MODELS,
    MODEL_NAME,
)
from hparams import KNOBS, CONFIG_KEYS, flag
from method_registry import load_method_registry
from experiment_config import (
    ACAlg1BatchAllocation,
    validate_artifact_identifier,
)
from compute_accounting import ModelForwardCounter
import benchmark as B
from result_contract import (
    ResultContractError,
    acquire_cell_lock,
    adapter_artifacts,
    atomic_write_json,
    cell_fingerprint,
    receipt_path,
    release_cell_lock,
    validate_completion_receipt,
    validate_receipt_identity,
    write_completion_receipt,
)
from prompt_contract import build_gsm8k_prompt_contract, write_prompt_contract
from ac_alg1_q5_multichain import run_q5_multichain
from ac_alg1_trice import run_ac_alg1_trice
from self_training import run_gold_cot_sft, run_source_self_training

try:                                                      # per-round logging coexists with the tqdm bar
    from tqdm import tqdm
    def _log(s): tqdm.write(str(s))
except Exception:
    def _log(s): print(s, flush=True)

# The exact algorithm parameters are recorded ONCE per CSV (a single `params` cell on row 0), not
# repeated on every result row -- algos have been retuned across commits, so two CSVs with the same
# `method` names can be different algorithms, and this makes each file self-describing.
# Both lists derive from the ONE knob registry (hparams.py); CLI flags are generated from it too.
_HPARAM_KEYS = tuple(k.key for k in KNOBS) + CONFIG_KEYS

# Sweep hyperparameters overridable from the CLI / YAML. Every explicit CLI knob must apply to all
# selected trainable methods; heterogeneous method-specific settings belong in separate direct
# invocations or YAML cells. Remaining categorical / boolean knobs (hparams.CONFIG_KEYS) are varied
# via named method variants in the registry instead; `micro` stays a memory-only knob.
_OVERRIDE_KEYS = tuple(k.key for k in KNOBS)
_RUNNER_CONSUMED_OVERRIDE_KEYS = frozenset(
    {"lora_r", "lora_alpha", "lora_seed", "lora_target_set"}
)


class _EvalUsage(TypedDict):
    calls: int
    generations: int


class _DiagnosticProbeUsage(_EvalUsage):
    question_ids: list[int]


def _method_hparams(fn):
    """Resolve a trainer's EFFECTIVE kwargs: signature defaults overlaid with functools.partial bindings."""
    bound = {}
    while isinstance(fn, functools.partial):
        bound.update(fn.keywords or {})
        fn = fn.func
    sig = inspect.signature(fn)
    hp = {n: p.default for n, p in sig.parameters.items() if p.default is not inspect.Parameter.empty}
    hp.update(bound)
    return {k: hp[k] for k in _HPARAM_KEYS if k in hp}


def _accepts(fn, name):
    """Does the trainer accept this kwarg? (--beta is valid for EM/DPO/APL but not GRPO/RLOO/RAFT.)"""
    while isinstance(fn, functools.partial):
        fn = fn.func
    return name in inspect.signature(fn).parameters


def _validate_cli_method_overrides(methods, overrides):
    """Reject explicit trainer knobs that any selected trainable method would ignore."""
    explicit = {
        key: value
        for key, value in overrides.items()
        if value is not None
    }
    if not explicit:
        return
    trainable_methods = [method for method in methods if method != "base"]
    if not trainable_methods:
        raise ValueError(
            "training hyperparameter overrides cannot be used with base-only evaluation: "
            f"{sorted(explicit)}"
        )
    requested = set(explicit) - _RUNNER_CONSUMED_OVERRIDE_KEYS
    unsupported = {
        method: sorted(
            key
            for key in requested
            if not _accepts(METHODS[method], key)
        )
        for method in trainable_methods
    }
    unsupported = {
        method: keys for method, keys in unsupported.items() if keys
    }
    if unsupported:
        details = "; ".join(
            f"{method}: {keys}" for method, keys in sorted(unsupported.items())
        )
        raise ValueError(
            "explicit CLI overrides must apply to every selected trainable method; "
            f"split heterogeneous sweeps or remove unsupported flags ({details})"
        )


def _env_meta():
    """Code + environment provenance, stamped into every CSV's params blob. Two mid-project refactors
    have already made "same method name, different algorithm" a real hazard: the commit hash pins the
    exact code that produced each number (papers need this in the repro statement anyway)."""
    import platform
    import subprocess
    env = {"python": platform.python_version(), "host": platform.node()}
    try:
        env["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        env["commit"] = "unknown"
    pinned_commit = os.environ.get("EXPECTED_COMMIT", "")
    if (
        env["commit"] == "unknown"
        and len(pinned_commit) == 40
        and all(character in "0123456789abcdef" for character in pinned_commit)
    ):
        # SGE wrappers verify this full SHA before invoking the sweep. Preserve
        # that stronger provenance when the nested best-effort Git probe is
        # transiently unavailable under high array concurrency.
        env["commit"] = pinned_commit[:7]
        env["commit_source"] = "expected_commit_fallback"
    else:
        env["commit_source"] = "git" if env["commit"] != "unknown" else "unavailable"
    for mod in ("torch", "transformers", "peft"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda_runtime"] = torch.version.cuda
            env["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    try:
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if driver.returncode == 0 and driver.stdout.strip():
            env["nvidia_driver"] = driver.stdout.splitlines()[0].strip()
    except Exception:
        pass
    lock_value = os.environ.get("VRL_ENV_LOCK")
    if lock_value:
        import hashlib

        lock_path = Path(lock_value).expanduser().resolve()
        env["environment_lock"] = str(lock_path)
        try:
            env["environment_lock_sha256"] = hashlib.sha256(
                lock_path.read_bytes()
            ).hexdigest()
        except OSError:
            env["environment_lock_sha256"] = "unreadable"
    return env


def _params_blob(args):
    """One-cell JSON snapshot of the whole run: sweep config + every method's resolved hyperparameters.
    Written to the `params` column of row 0 only; every other row leaves that cell blank."""
    methods = {m: _method_hparams(METHODS[m]) for m in args.methods if m in METHODS}
    for k in _OVERRIDE_KEYS:                              # reflect each CLI override wherever the method exposes it
        v = getattr(args, k)
        if v is not None:
            for hp in methods.values():
                if k in hp:
                    hp[k] = v
    target_modules = {
        model: list(
            resolve_lora_target_modules(
                model,
                lora_target_set=args.lora_target_set,
            )
        )
        for model in args.models
    }
    return json.dumps({"run_id": args.run_id,
                       "sweep": {"rounds": args.rounds, "B": args.batch, "G": args.G,
                                 "seeds": args.seeds, "seed0": args.seed0,
                                 "prompts": args.prompts, "shots": args.shots,
                                 "shot_bank_size": args.shot_bank_size,
                                 "task_seed_from_run_seed": args.task_seed_from_run_seed,
                                 "question_sampling": args.question_sampling,
                                 "n_test": args.n_test,
                                 "train_partition": args.train_partition,
                                 "eval_partition": args.eval_partition,
                                 "answer_event_mode": args.answer_event_mode,
                                 "answer_target_termination": (
                                     args.answer_target_termination
                                 ),
                                 "evaluation_prompt": args.evaluation_prompt,
                                 "transfer_eval_dataset": (
                                     args.transfer_eval_dataset
                                 ),
                                 "transfer_eval_n": args.transfer_eval_n,
                                 "eval_batch": args.eval_batch,
                                 "gradient_checkpointing": args.grad_checkpoint,
                                 "lora_target_set": args.lora_target_set,
                                 "eval_every": args.eval_every,
                                 "eval_rounds": args.eval_rounds,
                                 "dump_completions": args.dump_completions,
                                 "save_adapter": args.save_adapter,
                                 "save_training_diagnostics": args.save_training_diagnostics,
                                 "training_diagnostics_level": args.training_diagnostics_level,
                                 "training_diagnostics_trace_tape": args.training_diagnostics_trace_tape,
                                 "training_diagnostics_gradient_questions": (
                                     args.training_diagnostics_gradient_questions
                                 ),
                                 "training_diagnostics_probe_size": (
                                     args.training_diagnostics_probe_size
                                 ),
                                 "l2r_candidate_utility_questions": (
                                     args.l2r_candidate_utility_questions
                                 ),
                                 "l2r_candidate_utility_batch": (
                                     args.l2r_candidate_utility_batch
                                 ),
                                 "checkpoint_every": args.checkpoint_every,
                                 "l2r_exact_cache": args.l2r_exact_cache,
                                 "l2r_state_checkpoint_every": (
                                     args.l2r_state_checkpoint_every
                                 ),
                                 "passk": args.passk,
                                 "passk_n": args.passk_n,
                                 "lora_target_modules": target_modules},
                       "env": _env_meta(),
                       "methods": methods}, default=str)


def _cell_identity(params_blob, model, task, method, seed, tag):
    """Stable scientific identity; host/GPU metadata are observations, not inputs."""
    configuration = json.loads(params_blob)
    configuration.pop("env", None)
    return {
        "run_id": configuration.get("run_id"),
        "model": model,
        "task": task,
        "method": method,
        "seed": int(seed),
        "tag": tag,
        "configuration": configuration,
    }


def _atomic_write_csv(rows, path):
    """Replace the shared sweep checkpoint only after a complete CSV is durable."""
    path = Path(path)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            pd.DataFrame(rows).to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_round_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _trim_gzip_jsonl(path, completed_rounds):
    """Remove records written after the latest durable round checkpoint."""

    path = Path(path)
    if not path.exists():
        return
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"cannot resume from malformed {path} line {line_number}"
                ) from exc
            if int(record.get("completed_rounds", 0)) <= completed_rounds:
                records.append(record)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    )
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for record in records:
                json.dump(record, stream, separators=(",", ":"), allow_nan=False)
                stream.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _row_has_valid_completion(row, out):
    """Modern rows require receipts; legacy rows retain exact-row resumability."""
    fingerprint = row.get("cell_fingerprint")
    receipt = row.get("completion_receipt")
    if not isinstance(fingerprint, str) or not fingerprint:
        return True
    if not isinstance(receipt, str) or not receipt:
        return False
    try:
        payload = validate_completion_receipt(
            Path(out) / receipt,
            expected_fingerprint=fingerprint,
            result_root=out,
            verify_hashes=True,
        )
        validate_receipt_identity(
            payload,
            {
                "run_id": str(row.get("run_id")),
                "model": row.get("model"),
                "method": row.get("method"),
                "seed": int(row.get("seed")),
            },
        )
    except ResultContractError:
        return False
    return True

# Adapter for the faithful Barber Algorithm 1 implementation. Keep --batch as the same total LLM
# generation budget used by the other methods: B//G questions per round, split across L' and U'.
def _run_ac_alg1_sweep(task, rounds=40, B=64, G=4, seed=0, lr=1e-4, iters=8,
                       model_name=MODEL_NAME, model_tok=None, length_norm=False, buffer_limit=0,
                       labelled_frac=0.5, buffer_strategy="fifo", proposal_prompt="question",
                       algorithm_profile="legacy",
                       buffer_semantics="multiset_legacy",
                       buffer_lifecycle="persistent",
                       buffer_max_age=-1,
                       labelled_proposal_prompt=None, answer_only_proposal_prompt=None,
                       proposal_mixture="single",
                       proposal_filter="all", proposal_policy="current",
                       proposal_temperature=1.0,
                       proposal_allocation_mode="uniform",
                       proposal_initial_traces=0,
                       proposal_allocation_max_traces=0,
                       responsibility_score="joint",
                       responsibility_posterior="softmax_entropy",
                       responsibility_temperature=1.0,
                       responsibility_ess_floor=0.0,
                       responsibility_abstention="none",
                       responsibility_rejection_threshold=0.0,
                       responsibility_null_log_evidence=0.0,
                       responsibility_null_prior=0.5,
                       responsibility_policy="current",
                       responsibility_answer_policy="current",
                       responsibility_refresh="inner_step",
                       responsibility_verifier_rollouts=0,
                       responsibility_verifier_temperature=1.0,
                       responsibility_verifier_max_new_tokens=64,
                       responsibility_verifier_batch_size=16,
                       responsibility_verifier_smoothing_alpha=0.5,
                       verifier_calibration_path=None,
                       reuse_fresh_traces=0,
                       reuse_importance_min=0.5,
                       reuse_importance_max=2.0,
                       variational_estimator="delta_joint",
                       labelled_em_weight=1.0,
                       answer_only_em_weight=1.0, policy_kl_coef=None,
                       supervised_weight=1.0,
                       policy_anchor_mode="fixed", policy_anchor_target_ratio=None,
                       policy_anchor_beta_min=0.0, policy_anchor_beta_max=10.0,
                       policy_anchor_ema=0.9,
                       policy_anchor_token_scope="objective",
                       labelled_numeric_constraint="off", numeric_penalty=2.0,
                       labelled_supervision="gold", compact_gold_weight=0.5,
                       numeric_contradiction_penalty=0.0, numeric_missing_penalty=0.0,
                       digit_token_weight=1.0, trace_representation="reasoning",
                       latent_mstep_objective="joint",
                       mstep_sample_size=0,
                       mstep_sampling_strategy="posterior_categorical",
                       answer_event_mode="legacy",
                       answer_target_termination="none",
                       update_geometry="sum", step_acceptance="none",
                       rollback_tolerance=1e-6, rollback_max_backtracks=0,
                       rollback_shrink=0.5, optimizer_state_scope="persistent",
                       question_sampling="random",
                       eval_every=0, eval_rounds=None, eval_fn=None,
                       diagnostics_fn=None, diagnostics_level="standard",
                       diagnostics_trace_tape=False,
                       diagnostics_gradient_questions=0,
                       diagnostics_probe_fn=None, checkpoint_every=0,
                       checkpoint_fn=None, log=print):
    allocation = ACAlg1BatchAllocation.from_budget(
        batch=B,
        generations=G,
        labelled_fraction=labelled_frac,
    )
    recs = run_ac_alg1(task, algorithm_profile=algorithm_profile,
                       rounds=rounds, L_batch=allocation.labelled,
                       U_batch=allocation.answer_only, G_label=G, G_answer_only=G,
                       inner_steps=iters, seed=seed, lr=lr, model_name=model_name, model_tok=model_tok,
                       length_norm=length_norm, buffer_limit=buffer_limit, labelled_frac=labelled_frac,
                       buffer_strategy=buffer_strategy, proposal_prompt=proposal_prompt,
                       buffer_semantics=buffer_semantics,
                       buffer_lifecycle=buffer_lifecycle,
                       buffer_max_age=buffer_max_age,
                       labelled_proposal_prompt=labelled_proposal_prompt,
                       answer_only_proposal_prompt=answer_only_proposal_prompt,
                       proposal_mixture=proposal_mixture,
                       proposal_filter=proposal_filter, proposal_policy=proposal_policy,
                       proposal_temperature=proposal_temperature,
                       proposal_allocation_mode=proposal_allocation_mode,
                       proposal_initial_traces=proposal_initial_traces,
                       proposal_allocation_max_traces=proposal_allocation_max_traces,
                       responsibility_score=responsibility_score,
                       responsibility_posterior=responsibility_posterior,
                       responsibility_temperature=responsibility_temperature,
                       responsibility_ess_floor=responsibility_ess_floor,
                       responsibility_abstention=responsibility_abstention,
                       responsibility_rejection_threshold=(
                           responsibility_rejection_threshold
                       ),
                       responsibility_null_log_evidence=(
                           responsibility_null_log_evidence
                       ),
                       responsibility_null_prior=responsibility_null_prior,
                       responsibility_policy=responsibility_policy,
                       responsibility_answer_policy=responsibility_answer_policy,
                       responsibility_refresh=responsibility_refresh,
                       responsibility_verifier_rollouts=(
                           responsibility_verifier_rollouts
                       ),
                       responsibility_verifier_temperature=(
                           responsibility_verifier_temperature
                       ),
                       responsibility_verifier_max_new_tokens=(
                           responsibility_verifier_max_new_tokens
                       ),
                       responsibility_verifier_batch_size=(
                           responsibility_verifier_batch_size
                       ),
                       responsibility_verifier_smoothing_alpha=(
                           responsibility_verifier_smoothing_alpha
                       ),
                       verifier_calibration_path=verifier_calibration_path,
                       reuse_fresh_traces=reuse_fresh_traces,
                       reuse_importance_min=reuse_importance_min,
                       reuse_importance_max=reuse_importance_max,
                       variational_estimator=variational_estimator,
                       labelled_em_weight=labelled_em_weight,
                       answer_only_em_weight=answer_only_em_weight,
                       policy_kl_coef=policy_kl_coef,
                       supervised_weight=supervised_weight,
                       policy_anchor_mode=policy_anchor_mode,
                       policy_anchor_target_ratio=policy_anchor_target_ratio,
                       policy_anchor_beta_min=policy_anchor_beta_min,
                       policy_anchor_beta_max=policy_anchor_beta_max,
                       policy_anchor_ema=policy_anchor_ema,
                       policy_anchor_token_scope=policy_anchor_token_scope,
                       labelled_numeric_constraint=labelled_numeric_constraint,
                       numeric_penalty=numeric_penalty,
                       numeric_contradiction_penalty=numeric_contradiction_penalty,
                       numeric_missing_penalty=numeric_missing_penalty,
                       labelled_supervision=labelled_supervision,
                       compact_gold_weight=compact_gold_weight,
                       digit_token_weight=digit_token_weight,
                       trace_representation=trace_representation,
                       latent_mstep_objective=latent_mstep_objective,
                       mstep_sample_size=mstep_sample_size,
                       mstep_sampling_strategy=mstep_sampling_strategy,
                       answer_event_mode=answer_event_mode,
                       answer_target_termination=answer_target_termination,
                       update_geometry=update_geometry,
                       step_acceptance=step_acceptance,
                       rollback_tolerance=rollback_tolerance,
                       rollback_max_backtracks=rollback_max_backtracks,
                       rollback_shrink=rollback_shrink,
                       optimizer_state_scope=optimizer_state_scope,
                       question_sampling=question_sampling,
                       eval_every=eval_every, eval_rounds=eval_rounds,
                       eval_fn=eval_fn, diagnostics_fn=diagnostics_fn,
                       diagnostics_level=diagnostics_level,
                       diagnostics_trace_tape=diagnostics_trace_tape,
                       diagnostics_gradient_questions=diagnostics_gradient_questions,
                       diagnostics_probe_fn=diagnostics_probe_fn,
                       checkpoint_every=checkpoint_every, checkpoint_fn=checkpoint_fn, log=log)
    for r in recs:
        r.setdefault("oracle", 0)
        r.setdefault("llm_gen", r.get("gen"))
    return recs


def _run_trice_sweep(
    task,
    rounds=40,
    B=64,
    G=1,
    seed=0,
    lr=1e-4,
    model_name=MODEL_NAME,
    model_tok=None,
    trice_estimator="control_variate",
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
    """Map the sweep budget to common-protocol answer-only TRICE macrocycles."""

    if G != 1:
        raise ValueError("TRICE requires exactly one prior proposal per question")
    if proposal_prompt != "question":
        raise ValueError("TRICE prior proposals must use the question-only prompt")
    if answer_target_termination != "eos" or not reward_requires_eos:
        raise ValueError("common-protocol TRICE requires a naturally emitted EOS")
    records = run_ac_alg1_trice(
        task,
        rounds=rounds,
        L_batch=0,
        U_batch=B,
        seed=seed,
        lr=lr,
        model_name=model_name,
        model_tok=model_tok,
        labelled_frac=0.0,
        estimator=trice_estimator,
        initializer_prompt="answer_derive",
        supervised_weight=0.0,
        labelled_trice_weight=0.0,
        answer_only_trice_weight=1.0,
        reward_requires_eos=reward_requires_eos,
        question_sampling=question_sampling,
        eval_every=eval_every,
        eval_rounds=eval_rounds,
        eval_fn=eval_fn,
        diagnostics_fn=diagnostics_fn,
        checkpoint_every=checkpoint_every,
        checkpoint_fn=checkpoint_fn,
        log=log,
    )
    for record in records:
        record.setdefault("oracle", record.get("llm_gen", 0))
        record.setdefault("mean_reward", record.get("acceptance_fraction", 0.0))
    return records


def _run_q5_multichain_sweep(
    task,
    rounds=32,
    B=64,
    G=1,
    seed=0,
    lr=1e-4,
    model_name=MODEL_NAME,
    model_tok=None,
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
    """Map ``G`` to Q5 particles while holding ``B * G`` at 64 proposals."""

    if G not in (1, 2, 4):
        raise ValueError("Q5-MP requires G in {1, 2, 4}")
    if B * G != 64:
        raise ValueError("Q5-MP requires exactly 64 current proposals per round")
    if proposal_prompt != "question":
        raise ValueError("Q5-MP current proposals must use the question-only prompt")
    if answer_target_termination != "eos" or not reward_requires_eos:
        raise ValueError("Q5-MP requires strict marker-plus-EOS states")
    records = run_q5_multichain(
        task,
        rounds=rounds,
        questions_per_round=B,
        chains_per_question=G,
        seed=seed,
        lr=lr,
        model_name=model_name,
        model_tok=model_tok,
        initializer_prompt="answer_derive",
        reward_requires_eos=reward_requires_eos,
        question_sampling=question_sampling,
        eval_every=eval_every,
        eval_rounds=eval_rounds,
        eval_fn=eval_fn,
        diagnostics_fn=diagnostics_fn,
        checkpoint_every=checkpoint_every,
        checkpoint_fn=checkpoint_fn,
        log=log,
    )
    for record in records:
        record.setdefault("oracle", record.get("llm_gen", 0))
        record.setdefault("mean_reward", record.get("acceptance_fraction", 0.0))
    return records


# Stable result names and their bound scientific defaults are data, not Python
# control flow. Historical aliases remain unchanged in method_presets.yaml.
METHODS = load_method_registry(
    {
        "run_conditional_vro": run_conditional_vro,
        "run_ac_em": run_ac_em,
        "_run_ac_alg1_sweep": _run_ac_alg1_sweep,
        "run_l2r": run_l2r,
        "run_ac_dpo": run_ac_dpo,
        "run_weighted_em": run_weighted_em,
        "run_raft": run_raft,
        "run_rloo": run_rloo,
        "run_grpo": run_grpo,
        "run_dpo": run_dpo,
        "run_apl": run_apl,
        "run_gold_cot_sft": run_gold_cot_sft,
        "run_source_self_training": run_source_self_training,
        "_run_trice_sweep": _run_trice_sweep,
        "_run_q5_multichain_sweep": _run_q5_multichain_sweep,
    }
)


def build_task(name, prompts, shots, seed, train_partition="all",
               shot_bank_size=0, task_seed_from_run_seed=False,
               answer_event_mode="legacy"):
    if name == "gsm8k":
        task_seed = seed if task_seed_from_run_seed else 0
        return GSM8KTask(
            n_prompts=prompts,
            n_shots=shots,
            seed=task_seed,
            train_partition=train_partition,
            shot_bank_size=(shot_bank_size or None),
            answer_event_mode=answer_event_mode,
        )
    if name == "imdb":
        return IMDBSentimentTask(n_prompts=prompts, seed=0)
    if name == "hendrycks_math":
        task_seed = seed if task_seed_from_run_seed else 0
        return HendrycksMathTask(
            n_prompts=prompts,
            n_shots=shots,
            seed=task_seed,
            train_partition=train_partition,
            shot_bank_size=(shot_bank_size or None),
            answer_event_mode=answer_event_mode,
        )
    if name == "hendrycks_math_chat":
        task_seed = seed if task_seed_from_run_seed else 0
        return HendrycksMathChatTask(
            n_prompts=prompts,
            n_shots=shots,
            seed=task_seed,
            train_partition=train_partition,
            shot_bank_size=(shot_bank_size or None),
            answer_event_mode=answer_event_mode,
        )
    raise ValueError(name)


def _summarize_math_records(accuracy, records):
    """Summarize one exact-answer math pass under both answer contracts."""

    metrics = {"test_acc": float(accuracy)}
    if not records:
        return metrics
    format_failures = [bool(record.get("format_failure")) for record in records]
    fallback = [
        record.get("answer_parse_mode") == "fallback_last_integer"
        for record in records
    ]
    unparsed = [record.get("answer_parse_mode") == "unparsed" for record in records]
    correct_without_marker = [
        bool(record.get("correct")) and bool(record.get("format_failure"))
        for record in records
    ]
    legacy_correct = [
        bool(record.get("legacy_correct", record.get("correct")))
        for record in records
    ]
    strict_correct = [
        bool(record.get("strict_correct", record.get("correct")))
        for record in records
    ]
    strict_correct_and_eos = [
        bool(record.get("strict_correct_and_eos")) for record in records
    ]
    has_nonempty_reasoning = [
        bool(record.get("has_nonempty_reasoning")) for record in records
    ]
    strict_correct_with_reasoning = [
        bool(record.get("strict_correct_with_reasoning")) for record in records
    ]
    direct_answer_only = [
        bool(record.get("direct_answer_only")) for record in records
    ]
    strict_failures = [
        bool(record.get("strict_format_failure", record.get("format_failure")))
        for record in records
    ]
    multiple_markers = [
        int(record.get("answer_marker_count", 0)) > 1 for record in records
    ]
    nonterminal_markers = [
        record.get("answer_parse_mode") == "nonterminal_or_invalid_marker"
        for record in records
    ]
    natural_eos = [bool(record.get("generated_eos")) for record in records]
    hit_max_new = [bool(record.get("hit_max_new_tokens")) for record in records]
    official_test_access = [
        record.get("official_test_accessed") for record in records
    ]
    eval_source_splits = {
        record.get("eval_source_split")
        for record in records
        if isinstance(record.get("eval_source_split"), str)
    }
    dataset_splits_loaded = {
        tuple(record.get("dataset_splits_loaded"))
        for record in records
        if isinstance(record.get("dataset_splits_loaded"), list)
    }
    tokens_until_eos = [
        int(record["generated_tokens_until_eos"])
        for record in records
        if record.get("generated_tokens_until_eos") is not None
    ]
    metrics.update({
        "eval_format_failure_fraction": float(np.mean(format_failures)),
        "test_acc_legacy": float(np.mean(legacy_correct)),
        "test_acc_strict": float(np.mean(strict_correct)),
        "eval_strict_correct_and_eos_fraction": float(
            np.mean(strict_correct_and_eos)
        ),
        "eval_eos_given_strict_correct_fraction": (
            float(
                np.mean([
                    strict_correct_and_eos[index]
                    for index, correct in enumerate(strict_correct)
                    if correct
                ])
            )
            if any(strict_correct) else None
        ),
        "eval_nonempty_reasoning_fraction": float(
            np.mean(has_nonempty_reasoning)
        ),
        "eval_strict_correct_with_reasoning_fraction": float(
            np.mean(strict_correct_with_reasoning)
        ),
        "eval_direct_answer_only_fraction": float(np.mean(direct_answer_only)),
        "eval_strict_format_failure_fraction": float(np.mean(strict_failures)),
        "eval_fallback_parse_fraction": float(np.mean(fallback)),
        "eval_unparsed_fraction": float(np.mean(unparsed)),
        "eval_multiple_marker_fraction": float(np.mean(multiple_markers)),
        "eval_nonterminal_marker_fraction": float(np.mean(nonterminal_markers)),
        "eval_correct_without_marker_fraction": float(
            np.mean(correct_without_marker)
        ),
        "eval_natural_eos_fraction": float(np.mean(natural_eos)),
        "eval_hit_max_new_fraction": float(np.mean(hit_max_new)),
        "eval_official_test_accessed": (
            any(official_test_access)
            if all(isinstance(value, bool) for value in official_test_access)
            else None
        ),
        "eval_source_split": (
            next(iter(eval_source_splits))
            if len(eval_source_splits) == 1 else None
        ),
        "eval_dataset_splits_loaded": (
            list(next(iter(dataset_splits_loaded)))
            if len(dataset_splits_loaded) == 1 else None
        ),
        "eval_mean_tokens_until_eos": (
            float(np.mean(tokens_until_eos)) if tokens_until_eos else None
        ),
    })
    return metrics


def ordered_dataset_ids_sha256(values) -> str | None:
    if values is None:
        return None
    encoded = json.dumps(
        [str(value) for value in values],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark(
    task_name,
    model,
    tok,
    n_test,
    shots,
    seed,
    eval_partition="all",
    eval_batch=16,
    shot_bank_size=0,
    answer_event_mode="legacy",
    evaluation_prompt="question",
):
    """Returns (metrics_dict, per_question_records). The records are EVERY held-out example
    (idx/question/completion/gold/pred/correct) from the SAME pass that scored the model -- the caller
    writes a compact per-question file (stratify by difficulty) and, optionally, the first N in full."""
    if task_name == "gsm8k":
        acc, recs = B.benchmark_gsm8k(model, tok, n_test=n_test, n_shots=shots, seed=seed,
                                      batch=eval_batch, return_records=True,
                                      eval_partition=eval_partition,
                                      shot_bank_size=(shot_bank_size or None),
                                      answer_event_mode=answer_event_mode,
                                      evaluation_prompt=evaluation_prompt)
        return _summarize_math_records(acc, recs), recs
    if task_name == "imdb":
        from datasets import load_dataset
        ds = load_dataset("imdb", split="test")
        idx = np.random.default_rng(seed).permutation(len(ds))[:n_test]
        prompts = [" ".join(ds[int(i)]["text"].split()[:8]) for i in idx]
        win, pos = B.benchmark_imdb(model, tok, prompts, batch=eval_batch)
        return {"win_rate": win, "mean_pos": pos}, []
    if task_name in {"hendrycks_math", "hendrycks_math_chat"}:
        evaluator = (
            B.benchmark_hendrycks_math_chat
            if task_name == "hendrycks_math_chat"
            else B.benchmark_hendrycks_math
        )
        acc, recs = evaluator(
            model,
            tok,
            n_test=n_test,
            n_shots=shots,
            batch=eval_batch,
            return_records=True,
            answer_event_mode=answer_event_mode,
            evaluation_prompt=evaluation_prompt,
        )
        return _summarize_math_records(acc, recs), recs
    raise ValueError(task_name)


def benchmark_transfer(
    dataset,
    model,
    tok,
    n_test,
    shots,
    seed,
    eval_batch=16,
    shot_bank_size=0,
    answer_event_mode="legacy",
    train_partition="train",
):
    """Run an evaluation-only transfer benchmark and namespace every result column."""

    if dataset == "none":
        return {}, [], {}
    if dataset != "svamp":
        raise ValueError(f"unknown transfer evaluation dataset {dataset!r}")
    accuracy, records, metadata = B.benchmark_svamp_transfer(
        model,
        tok,
        n_test=n_test,
        n_shots=shots,
        seed=seed,
        batch=eval_batch,
        shot_bank_size=(shot_bank_size or None),
        answer_event_mode=answer_event_mode,
        train_partition=train_partition,
    )
    summary = _summarize_math_records(accuracy, records)
    metrics = {
        f"svamp_transfer_{key}": value for key, value in summary.items()
    }
    metrics.update({
        f"svamp_transfer_{key}": value for key, value in metadata.items()
    })
    return metrics, records, metadata


_COMPACT_MATH_RECORD_FIELDS = (
    "idx",
    "dataset_id",
    "correct",
    "pred",
    "gold",
    "legacy_pred",
    "legacy_correct",
    "strict_pred",
    "strict_correct",
    "strict_correct_and_eos",
    "has_nonempty_reasoning",
    "strict_correct_with_reasoning",
    "direct_answer_only",
    "len",
    "rep4",
    "has_answer_marker",
    "format_failure",
    "strict_format_failure",
    "answer_parse_mode",
    "answer_event_mode",
    "evaluation_prompt",
    "answer_marker_count",
    "answer_marker_terminal",
    "generated_eos",
    "generated_tokens_until_eos",
    "hit_max_new_tokens",
    "generated_stop_token_id",
    "generated_eod_before_eot",
    "official_test_accessed",
    "eval_source_split",
    "dataset_splits_loaded",
    "dataset",
    "dataset_revision",
    "dataset_sha256",
)


def _compact_math_records(records):
    """Drop completion text while preserving every auditable per-item outcome."""

    return [
        {
            key: record.get(key)
            for key in _COMPACT_MATH_RECORD_FIELDS
            if key in record
        }
        for record in records
    ]


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        default="gsm8k",
        choices=["gsm8k", "hendrycks_math", "hendrycks_math_chat", "imdb"],
    )
    p.add_argument("--models", nargs="+", default=["qwen2.5-1.5b-instruct"])
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS) + ["base"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval-batch", type=int, default=16,
                   help="generation batch size for held-out and pass@k evaluation")
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--prompts", type=int, default=64)
    p.add_argument("--shots", type=int, default=4)
    p.add_argument(
        "--shot-bank-size",
        type=int,
        default=0,
        help=(
            "reserve this many leading task examples for a nested few-shot bank; "
            "training questions then remain fixed when --shots changes (zero keeps "
            "the historical dynamic offset)"
        ),
    )
    p.add_argument(
        "--task-seed-from-run-seed",
        action="store_true",
        help=(
            "use each run seed for GSM8K demonstrations and training-question "
            "selection; default keeps the historical seed-0 task across runs"
        ),
    )
    p.add_argument(
        "--question-sampling",
        choices=["random", "epoch_shuffle"],
        default="random",
        help=(
            "sample training questions independently each round or consume "
            "reproducible shuffled epochs before repeating a question"
        ),
    )
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="gradient checkpointing: saves VRAM, needed for 1.5B backward on the 16 GB box")
    for k in KNOBS:                                       # one flag per sweepable knob (hparams.py, the registry)
        p.add_argument(flag(k.key), type=k.cast, default=None, help=k.help)
    p.add_argument("--n-test", type=int, default=200, help="held-out benchmark size")
    p.add_argument("--train-partition", choices=["all", "train"], default="all",
                   help="GSM8K training pool: full official train split, or the fixed subset left "
                        "after reserving a training-derived validation set")
    p.add_argument("--eval-partition", choices=["all", "tune", "final", "validation", "test"],
                   default="all", help="GSM8K evaluation pool; validation is held out from train, "
                                        "test is the untouched official test split")
    p.add_argument(
        "--answer-event-mode",
        choices=["legacy", "strict_terminal_marker"],
        default="legacy",
        help=(
            "GSM8K answer contract used by training rewards and evaluation; "
            "strict_terminal_marker requires exactly one terminal #### integer"
        ),
    )
    p.add_argument(
        "--answer-target-termination",
        choices=["none", "eos"],
        default="none",
        help=(
            "teacher-forced answer-factor terminator; eos includes the tokenizer "
            "EOS token in the supervised answer loss while none preserves "
            "historical targets"
        ),
    )
    p.add_argument(
        "--evaluation-prompt",
        choices=["question", "answer_derive", "answer_derive_first"],
        default="question",
        help=(
            "evaluation-only prompt intervention; answer-conditioned modes "
            "expose the gold answer and are diagnostic, not valid baselines"
        ),
    )
    p.add_argument(
        "--transfer-eval-dataset",
        choices=["none", "svamp"],
        default="none",
        help=(
            "evaluation-only transfer benchmark after GSM8K training; "
            "never supplies transfer examples to a trainer"
        ),
    )
    p.add_argument(
        "--transfer-eval-n",
        type=int,
        default=1000,
        help="number of checksum-verified SVAMP examples to evaluate (1-1000)",
    )
    p.add_argument("--dump-completions", type=int, default=0,
                   help="if >0, write the first N held-out completions per method to dump_*.json "
                        "(reuses the benchmark pass; eyeball real degradation vs a parse miss)")
    p.add_argument("--eval-every", type=int, default=0,
                   help="if >0, run the held-out benchmark every K rounds DURING training and log test_acc "
                        "into the trajectory (all trainers; extra generation cost, off by default)")
    p.add_argument("--eval-rounds", default="",
                   help="comma-separated completed rounds for sparse held-out checks; mutually exclusive "
                        "with --eval-every and always includes the final round")
    p.add_argument("--passk", type=int, default=0,
                   help="if >0, run pass@K on held-out GSM8K at the end of each cell (K sampled completions "
                        "at T=0.7 per question) -> passk_*.json + pass1/passK columns. The RLVR sharpening "
                        "check (pass@1 up while pass@k down = narrowed distribution); finalists only")
    p.add_argument("--passk-n", type=int, default=100,
                   help="held-out questions for the pass@K estimate (default 100; cost = K x this in generations)")
    p.add_argument("--save-adapter", action="store_true",
                   help="save each trained cell's LoRA adapter to adapter_<task>__<tag>__<method>_s<seed>/ "
                        "(~20-40MB each): any post-hoc analysis without re-training the cell")
    p.add_argument("--save-training-diagnostics", action="store_true",
                   help="write method-specific per-round samples, responsibilities, buffer composition, "
                        "and optimisation health to a gzip JSONL artifact "
                        "(standard level adds no model calls)")
    p.add_argument(
        "--training-diagnostics-level",
        choices=["standard", "deep"],
        default="standard",
        help=(
            "standard reuses values already produced by training; deep adds "
            "fixed-surrogate, gradient-attribution, and behavioural-probe "
            "evaluations for supported AC-ALG1 and L2R methods"
        ),
    )
    p.add_argument(
        "--training-diagnostics-trace-tape",
        action="store_true",
        help=(
            "include full buffered token ids and masks in the compressed "
            "AC-ALG1 diagnostics stream"
        ),
    )
    p.add_argument(
        "--training-diagnostics-gradient-questions",
        type=int,
        default=0,
        help=(
            "in deep AC-ALG1/L2R diagnostics, recompute this many "
            "highest-responsibility question gradients per inner step"
        ),
    )
    p.add_argument(
        "--training-diagnostics-probe-size",
        type=int,
        default=0,
        help=(
            "in deep AC-ALG1/L2R diagnostics, evaluate this many fixed held-out "
            "questions before training and after every accepted inner M-step"
        ),
    )
    p.add_argument(
        "--l2r-candidate-utility-questions",
        type=int,
        default=0,
        help=(
            "reserve this many additional training-derived questions from "
            "both optimisation and trust safety, then log diagnostic-only "
            "paired candidate-step accuracy, direct-answer NLL, and "
            "capability-gradient alignment"
        ),
    )
    p.add_argument(
        "--l2r-candidate-utility-batch",
        type=int,
        default=16,
        help=(
            "greedy generation batch size for the reserved L2R candidate "
            "utility audit"
        ),
    )
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="save intermediate adapters every N completed rounds where supported; zero disables it")
    p.add_argument(
        "--l2r-exact-cache",
        action="store_true",
        help=(
            "reuse immutable frozen-reader and base-policy scores plus exact "
            "safety NLL carry-forward in L2R"
        ),
    )
    p.add_argument(
        "--l2r-state-checkpoint-every",
        type=int,
        default=0,
        help=(
            "atomically save complete L2R state every N rounds and resume an "
            "interrupted cell; zero disables it"
        ),
    )
    p.add_argument("--out", default="results", help="point at HOME on the cluster (scratch is wiped)")
    p.add_argument("--tag", default="run")
    p.add_argument("--run-id", default=None, help="run id linking these CSVs to the YAML that launched them")
    return p


def _normalise_and_validate_args(p, args):
    try:
        validate_artifact_identifier(args.tag, field="--tag")
        if args.run_id is not None:
            validate_artifact_identifier(args.run_id, field="--run-id")
        _validate_cli_method_overrides(
            args.methods,
            {key: getattr(args, key) for key in _OVERRIDE_KEYS},
        )
    except ValueError as exc:
        p.error(str(exc))
    if args.lora_target_set is None:
        args.lora_target_set = "attention"
    try:
        args.eval_rounds = tuple(
            sorted({int(value) for value in args.eval_rounds.split(",") if value.strip()})
        )
    except ValueError as exc:
        p.error(f"--eval-rounds must be comma-separated positive integers: {exc}")
    if any(value < 1 for value in args.eval_rounds):
        p.error("--eval-rounds values must be positive")
    if any(value > args.rounds for value in args.eval_rounds):
        p.error("--eval-rounds values cannot exceed --rounds")
    if args.eval_every and args.eval_rounds:
        p.error("--eval-every and --eval-rounds are mutually exclusive")
    if args.training_diagnostics_gradient_questions < 0:
        p.error("--training-diagnostics-gradient-questions must be nonnegative")
    if args.training_diagnostics_probe_size < 0:
        p.error("--training-diagnostics-probe-size must be nonnegative")
    if args.l2r_candidate_utility_questions < 0:
        p.error("--l2r-candidate-utility-questions must be nonnegative")
    if args.l2r_candidate_utility_batch < 1:
        p.error("--l2r-candidate-utility-batch must be positive")
    if args.l2r_state_checkpoint_every < 0:
        p.error("--l2r-state-checkpoint-every must be nonnegative")
    if not 1 <= args.transfer_eval_n <= B.SVAMP_DATASET_ROWS:
        p.error(
            f"--transfer-eval-n must be in [1, {B.SVAMP_DATASET_ROWS}]"
        )
    if args.transfer_eval_dataset != "none":
        if args.task != "gsm8k":
            p.error("transfer evaluation is supported only for --task gsm8k")
        if args.evaluation_prompt != "question":
            p.error("SVAMP transfer evaluation requires --evaluation-prompt question")
    chat_task = args.task == "hendrycks_math_chat"
    chat_model = args.models == ["qwen3-8b-chat"]
    if chat_task != chat_model:
        p.error(
            "hendrycks_math_chat and the sole qwen3-8b-chat model must be used together"
        )
    if chat_task:
        required = {
            "train_partition": (args.train_partition, "train"),
            "eval_partition": (args.eval_partition, "validation"),
            "evaluation_prompt": (args.evaluation_prompt, "question"),
            "answer_event_mode": (
                args.answer_event_mode,
                "strict_terminal_marker",
            ),
            "shots": (args.shots, 4),
            "task_seed_from_run_seed": (args.task_seed_from_run_seed, True),
            "passk": (args.passk, 0),
        }
        changed = [
            field
            for field, (observed, expected) in required.items()
            if observed != expected
        ]
        if changed:
            p.error(f"Qwen3 chat protocol field(s) changed: {changed}")
        expected_termination = (
            "none" if args.methods == ["GRPO"] else "eos"
        )
        if args.answer_target_termination != expected_termination:
            p.error(
                "Qwen3 chat answer-target termination changed for the selected method"
            )
    if (
        (
            args.training_diagnostics_level != "standard"
            or args.training_diagnostics_trace_tape
            or args.training_diagnostics_gradient_questions
            or args.training_diagnostics_probe_size
            or args.l2r_candidate_utility_questions
        )
        and not args.save_training_diagnostics
    ):
        p.error(
            "deep diagnostic options require --save-training-diagnostics"
        )
    if (
        (
            args.training_diagnostics_gradient_questions
            or args.training_diagnostics_probe_size
            or args.l2r_candidate_utility_questions
        )
        and args.training_diagnostics_level != "deep"
    ):
        p.error(
            "question-gradient, fixed-probe, and candidate-utility diagnostics require "
            "--training-diagnostics-level deep"
        )
    return args


def main():
    p = _build_parser()
    args = _normalise_and_validate_args(p, p.parse_args())
    blob = _params_blob(args)                             # one-cell snapshot of all algo params (row 0)
    qi_stamped = False                                    # retain first-cell provenance in the legacy blob

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"sweep_{args.task}__{args.tag}.csv")
    rows, t0 = [], time.time()
    failures = []
    done = set()                                          # RESUMABILITY: skip cells already in the CSV
    if os.path.exists(path):                              # survives crashes + the 3-day session cutoff
        rows = pd.read_csv(path).to_dict("records")
        done = {
            (r["model"], r["method"], int(r["seed"]))
            for r in rows
            if _row_has_valid_completion(r, out)
        }
        incomplete = len(rows) - len(done)
        suffix = f"; {incomplete} incomplete receipt(s) will rerun" if incomplete else ""
        print(
            f"resuming: {len(done)} cells already complete in {path}{suffix}",
            flush=True,
        )
    for model_name in args.models:
        # per-model artifact tag: traj/eval/dump/passk/adapter filenames carry no model of their own, so a
        # direct multi-model invocation would overwrite model A's JSONs with model B's (CSV rows are safe --
        # they carry a model column). Single-model runs (the run_yaml path) keep their unchanged names.
        mtag = args.tag if len(args.models) == 1 else f"{args.tag}_{model_name.replace('/', '-')}"
        for method in args.methods:
            for seed in range(args.seed0, args.seed0 + args.seeds):
                tag = f"{model_name} | {args.task} | {method} | seed {seed}"
                if (model_name, method, seed) in done:
                    print(f"== SKIP (done): {tag} ==", flush=True); continue
                rows = [
                    row
                    for row in rows
                    if not (
                        row.get("model") == model_name
                        and row.get("method") == method
                        and int(row.get("seed", -1)) == seed
                    )
                ]
                print(f"== {tag} | {time.time()-t0:.0f}s ==", flush=True)
                cell_t0 = time.time()                               # wall-clock cost axis (per cell)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()            # per-cell peak VRAM (vram_gb column)
                model = tok = None
                chat_runtime_contract = None
                lora_runtime_contract = None
                forward_counter: ModelForwardCounter | None = None
                cell_lock_fh = None
                diagnostics_fh: TextIO | None = None
                checkpoint_eval_fh: TextIO | None = None
                diagnostics_path = None
                checkpoint_eval_path = None
                round_state_path = None
                resume_state = None
                cell_identity_value = None
                cell_fingerprint_value = None
                artifact_paths: list[str | Path] = []
                eval_usage: _EvalUsage = {"calls": 0, "generations": 0}
                diagnostic_probe_usage: _DiagnosticProbeUsage = {
                    "calls": 0,
                    "generations": 0,
                    "question_ids": [],
                }
                try:                                                # one bad cell must not kill the grid
                    cell_lock_fh = acquire_cell_lock(
                        out,
                        task=args.task,
                        tag=mtag,
                        method=method,
                        seed=seed,
                    )
                    task = build_task(
                        args.task,
                        args.prompts,
                        args.shots,
                        seed,
                        args.train_partition,
                        shot_bank_size=args.shot_bank_size,
                        task_seed_from_run_seed=args.task_seed_from_run_seed,
                        answer_event_mode=args.answer_event_mode,
                    )
                    if args.task == "gsm8k":
                        prompt_contract_path = Path(out) / (
                            f"prompt_contract_{args.task}__{mtag}__{method}_s{seed}.json.gz"
                        )
                        prompt_contract = build_gsm8k_prompt_contract(
                            task,
                            proposal_prompt=args.proposal_prompt,
                            method=method,
                            seed=seed,
                            tag=mtag,
                            answer_event_mode=args.answer_event_mode,
                            answer_target_termination=args.answer_target_termination,
                        )
                        write_prompt_contract(prompt_contract_path, prompt_contract)
                        artifact_paths.append(prompt_contract_path)
                    if (
                        not args.task_seed_from_run_seed
                        and not qi_stamped
                        and hasattr(task, "train_qi")
                    ):
                        blob = json.dumps({**json.loads(blob), "train_qi": task.train_qi,
                                           "shot_qi": task.shot_qi})  # contamination/leakage audit;
                        qi_stamped = True
                    cell_identity_value = _cell_identity(
                        blob,
                        model_name,
                        args.task,
                        method,
                        seed,
                        mtag,
                    )
                    execution_binding = None
                    if args.task == "hendrycks_math_chat":
                        execution_commit = os.environ.get("EXPECTED_COMMIT", "")
                        execution_config_sha256 = os.environ.get(
                            "EXPECTED_CONFIG_SHA256", ""
                        )
                        if not re.fullmatch(r"[0-9a-f]{40}", execution_commit):
                            raise ValueError(
                                "Qwen3 chat execution requires a full immutable commit"
                            )
                        if not re.fullmatch(
                            r"[0-9a-f]{64}", execution_config_sha256
                        ):
                            raise ValueError(
                                "Qwen3 chat execution requires the exact config SHA-256"
                            )
                        execution_binding = {
                            "execution_commit": execution_commit,
                            "configuration_sha256": execution_config_sha256,
                        }
                        cell_identity_value = {
                            **cell_identity_value,
                            **execution_binding,
                        }
                    train_dataset_ids = (
                        getattr(task, "train_dataset_ids", None)
                        if args.task == "hendrycks_math_chat"
                        else None
                    )
                    train_dataset_ids_sha256 = ordered_dataset_ids_sha256(
                        train_dataset_ids
                    )
                    if train_dataset_ids is not None:
                        cell_identity_value = {
                            **cell_identity_value,
                            "train_dataset_ids": [
                                str(value) for value in train_dataset_ids
                            ],
                            "train_dataset_ids_sha256": train_dataset_ids_sha256,
                        }
                    cell_fingerprint_value = cell_fingerprint(
                        cell_identity_value
                    )
                    if (
                        args.l2r_state_checkpoint_every > 0
                        and method != "base"
                        and _accepts(METHODS[method], "resume_state")
                    ):
                        round_state_path = Path(out) / (
                            f"l2r_round_state_{args.task}__{mtag}__"
                            f"{method}_s{seed}.pt"
                        )
                        if round_state_path.exists():
                            resume_state = _load_round_state(
                                round_state_path
                            )
                            if (
                                resume_state.get("fingerprint")
                                != cell_fingerprint_value
                            ):
                                raise ValueError(
                                    "existing L2R round state belongs to a "
                                    "different cell configuration"
                                )
                            runner_state = resume_state.get(
                                "runner_state",
                                {},
                            )
                            eval_usage.update(
                                runner_state.get("eval_usage", {})
                            )
                            diagnostic_probe_usage.update(
                                runner_state.get(
                                    "diagnostic_probe_usage",
                                    {},
                                )
                            )
                    model, tok = load_model(seed=seed, model=model_name,
                                            lora_r=(
                                                args.lora_r
                                                if args.lora_r is not None
                                                else 16
                                            ),
                                            lora_alpha=(
                                                args.lora_alpha
                                                if args.lora_alpha is not None
                                                else 32
                                            ),
                                            lora_seed=args.lora_seed,
                                            lora_target_set=(
                                                args.lora_target_set
                                                or "attention"
                                            ),
                                            gradient_checkpointing=args.grad_checkpoint)
                    if args.task == "hendrycks_math_chat":
                        chat_runtime_contract = task.bind_runtime(model, tok)
                        lora_runtime_contract = getattr(
                            model,
                            "_vrl_lora_runtime_contract",
                            None,
                        )
                        if not isinstance(lora_runtime_contract, dict):
                            raise ValueError(
                                "Qwen3 chat LoRA runtime contract is missing"
                            )
                    elif args.task == "hendrycks_math":
                        from math_prompting import validate_math_model_eos

                        validate_math_model_eos(model, tok)
                    forward_counter = ModelForwardCounter().attach(model)
                    over = {k: getattr(args, k) for k in _OVERRIDE_KEYS      # CLI hyperparameter overrides
                            if getattr(args, k) is not None}                # (filtered per-method by _accepts below)
                    if method == "base":                            # no-train baseline -> benchmark base model
                        recs = []
                    else:
                        kw = {k: v for k, v in over.items() if _accepts(METHODS[method], k)}
                        if _accepts(METHODS[method], "answer_event_mode"):
                            kw["answer_event_mode"] = args.answer_event_mode
                        if _accepts(
                            METHODS[method],
                            "answer_target_termination",
                        ):
                            kw["answer_target_termination"] = (
                                args.answer_target_termination
                            )
                        if (
                            args.l2r_candidate_utility_questions
                            and not _accepts(
                                METHODS[method],
                                "candidate_utility_questions",
                            )
                        ):
                            raise ValueError(
                                f"{method} does not support the L2R candidate "
                                "utility audit"
                            )
                        if _accepts(METHODS[method], "question_sampling"):
                            kw["question_sampling"] = args.question_sampling
                        eval_enabled = bool(args.eval_every or args.eval_rounds)
                        if eval_enabled and _accepts(METHODS[method], "eval_every"):
                            kw["eval_every"] = args.eval_every
                            if args.eval_rounds:
                                if not _accepts(METHODS[method], "eval_rounds"):
                                    raise ValueError(
                                        f"{method} does not support sparse --eval-rounds"
                                    )
                                kw["eval_rounds"] = args.eval_rounds
                            if args.save_training_diagnostics:
                                checkpoint_eval_path = os.path.join(
                                    out,
                                    f"checkpoint_eval_{args.task}__{mtag}__{method}_s{seed}.jsonl.gz",
                                )
                                checkpoint_mode = "wt"
                                if resume_state is not None:
                                    _trim_gzip_jsonl(
                                        checkpoint_eval_path,
                                        int(resume_state["completed_rounds"]),
                                    )
                                    checkpoint_mode = "at"
                                checkpoint_eval_fh = cast(
                                    TextIO,
                                    gzip.open(
                                        checkpoint_eval_path,
                                        checkpoint_mode,
                                        encoding="utf-8",
                                    ),
                                )

                            def eval_current(m, s=seed, tokenizer=tok):
                                metrics, eval_records = benchmark(
                                    args.task, m, tokenizer, args.n_test, args.shots, s,
                                    args.eval_partition, args.eval_batch,
                                    args.shot_bank_size, args.answer_event_mode,
                                    args.evaluation_prompt,
                                )
                                eval_usage["calls"] += 1
                                eval_usage["generations"] += len(eval_records) if eval_records else args.n_test
                                if checkpoint_eval_fh is not None:
                                    compact = [
                                        {
                                            "idx": record.get("idx"),
                                            "dataset_id": record.get("dataset_id"),
                                            "correct": record.get("correct"),
                                            "pred": record.get("pred"),
                                            "gold": record.get("gold"),
                                            "legacy_pred": record.get("legacy_pred"),
                                            "legacy_correct": record.get("legacy_correct"),
                                            "strict_pred": record.get("strict_pred"),
                                            "strict_correct": record.get("strict_correct"),
                                            "strict_correct_and_eos": record.get(
                                                "strict_correct_and_eos"
                                            ),
                                            "has_nonempty_reasoning": record.get(
                                                "has_nonempty_reasoning"
                                            ),
                                            "strict_correct_with_reasoning": record.get(
                                                "strict_correct_with_reasoning"
                                            ),
                                            "direct_answer_only": record.get(
                                                "direct_answer_only"
                                            ),
                                            "len": record.get("len"),
                                            "rep4": record.get("rep4"),
                                            "has_answer_marker": record.get(
                                                "has_answer_marker"
                                            ),
                                            "format_failure": record.get(
                                                "format_failure"
                                            ),
                                            "answer_parse_mode": record.get(
                                                "answer_parse_mode"
                                            ),
                                            "strict_format_failure": record.get(
                                                "strict_format_failure"
                                            ),
                                            "answer_event_mode": record.get(
                                                "answer_event_mode"
                                            ),
                                            "evaluation_prompt": record.get(
                                                "evaluation_prompt"
                                            ),
                                            "answer_marker_count": record.get(
                                                "answer_marker_count"
                                            ),
                                            "answer_marker_terminal": record.get(
                                                "answer_marker_terminal"
                                            ),
                                            "generated_eos": record.get(
                                                "generated_eos"
                                            ),
                                            "generated_tokens_until_eos": record.get(
                                                "generated_tokens_until_eos"
                                            ),
                                            "hit_max_new_tokens": record.get(
                                                "hit_max_new_tokens"
                                            ),
                                            "generated_stop_token_id": record.get(
                                                "generated_stop_token_id"
                                            ),
                                            "generated_eod_before_eot": record.get(
                                                "generated_eod_before_eot"
                                            ),
                                            "official_test_accessed": record.get(
                                                "official_test_accessed"
                                            ),
                                            "eval_source_split": record.get(
                                                "eval_source_split"
                                            ),
                                            "dataset_splits_loaded": record.get(
                                                "dataset_splits_loaded"
                                            ),
                                        }
                                        for record in (eval_records or [])
                                    ]
                                    if args.task != "hendrycks_math_chat":
                                        for record in compact:
                                            record.pop("dataset_id", None)
                                            record.pop("evaluation_prompt", None)
                                            record.pop("generated_stop_token_id", None)
                                            record.pop("generated_eod_before_eot", None)
                                    if args.eval_rounds:
                                        schedule = sorted({
                                            *(
                                                value
                                                for value in args.eval_rounds
                                                if value <= args.rounds
                                            ),
                                            args.rounds,
                                        })
                                        completed_rounds = schedule[eval_usage["calls"] - 1]
                                    else:
                                        completed_rounds = min(
                                            eval_usage["calls"] * args.eval_every,
                                            args.rounds,
                                        )
                                    json.dump({
                                        "run_id": args.run_id,
                                        "model": model_name,
                                        "task": args.task,
                                        "method": method,
                                        "seed": seed,
                                        "tag": mtag,
                                        "completed_rounds": completed_rounds,
                                        "metrics": metrics,
                                        "records": compact,
                                    }, checkpoint_eval_fh, separators=(",", ":"), default=str)
                                    checkpoint_eval_fh.write("\n")
                                    checkpoint_eval_fh.flush()
                                return metrics.get("test_acc", metrics.get("win_rate"))
                            kw["eval_fn"] = eval_current
                        if (args.save_training_diagnostics
                                and _accepts(METHODS[method], "diagnostics_fn")):
                            diagnostics_path = os.path.join(
                                out,
                                f"training_diagnostics_{args.task}__{mtag}__{method}_s{seed}.jsonl.gz",
                            )
                            diagnostics_mode = "wt"
                            if resume_state is not None:
                                _trim_gzip_jsonl(
                                    diagnostics_path,
                                    int(resume_state["completed_rounds"]),
                                )
                                diagnostics_mode = "at"
                            diagnostics_fh = cast(
                                TextIO,
                                gzip.open(
                                    diagnostics_path,
                                    diagnostics_mode,
                                    encoding="utf-8",
                                ),
                            )

                            def write_diagnostics(payload, fh=diagnostics_fh):
                                if args.training_diagnostics_probe_size:
                                    payload["behavioural_utility"].update({
                                        "fixed_probe_size": (
                                            args.training_diagnostics_probe_size
                                        ),
                                        "fixed_probe_question_ids": list(
                                            diagnostic_probe_usage["question_ids"]
                                        ),
                                    })
                                artifact_record = {
                                    "run_id": args.run_id,
                                    "model": model_name,
                                    "task": args.task,
                                    "method": method,
                                    "seed": seed,
                                    "tag": mtag,
                                    **payload,
                                }
                                json.dump(artifact_record, fh, separators=(",", ":"), allow_nan=False)
                                fh.write("\n")
                                fh.flush()

                            kw["diagnostics_fn"] = write_diagnostics
                            if _accepts(METHODS[method], "diagnostics_level"):
                                kw["diagnostics_level"] = (
                                    args.training_diagnostics_level
                                )
                            if _accepts(
                                METHODS[method],
                                "diagnostics_trace_tape",
                            ):
                                kw["diagnostics_trace_tape"] = (
                                    args.training_diagnostics_trace_tape
                                )
                            if _accepts(
                                METHODS[method],
                                "diagnostics_gradient_questions",
                            ):
                                kw["diagnostics_gradient_questions"] = (
                                    args.training_diagnostics_gradient_questions
                                )
                            if (
                                args.training_diagnostics_probe_size
                                and _accepts(
                                    METHODS[method],
                                    "diagnostics_probe_fn",
                                )
                            ):
                                def diagnostic_probe_current(
                                    current_model,
                                    s=seed,
                                    tokenizer=tok,
                                ):
                                    metrics, probe_records = benchmark(
                                        args.task,
                                        current_model,
                                        tokenizer,
                                        args.training_diagnostics_probe_size,
                                        args.shots,
                                        s,
                                        args.eval_partition,
                                        args.eval_batch,
                                        args.shot_bank_size,
                                        args.answer_event_mode,
                                        args.evaluation_prompt,
                                    )
                                    diagnostic_probe_usage["calls"] += 1
                                    diagnostic_probe_usage["generations"] += (
                                        len(probe_records)
                                        if probe_records
                                        else args.training_diagnostics_probe_size
                                    )
                                    if (
                                        probe_records
                                        and not diagnostic_probe_usage["question_ids"]
                                    ):
                                        diagnostic_probe_usage["question_ids"] = [
                                            int(record["idx"])
                                            for record in probe_records
                                            if record.get("idx") is not None
                                        ]
                                    return metrics.get(
                                        "test_acc",
                                        metrics.get("win_rate"),
                                    )

                                kw["diagnostics_probe_fn"] = (
                                    diagnostic_probe_current
                                )
                            if args.l2r_candidate_utility_questions:
                                kw["candidate_utility_questions"] = (
                                    args.l2r_candidate_utility_questions
                                )
                                kw["candidate_utility_batch"] = (
                                    args.l2r_candidate_utility_batch
                                )
                        if (
                            args.l2r_exact_cache
                            and _accepts(METHODS[method], "exact_cache")
                        ):
                            kw["exact_cache"] = True
                        if (
                            args.l2r_state_checkpoint_every > 0
                            and _accepts(
                                METHODS[method],
                                "state_checkpoint_every",
                            )
                        ):
                            kw["state_checkpoint_every"] = (
                                args.l2r_state_checkpoint_every
                            )
                            kw["resume_fingerprint"] = (
                                cell_fingerprint_value
                            )
                            kw["resume_state"] = resume_state

                            def save_round_state(
                                payload,
                                path=round_state_path,
                            ):
                                payload["runner_state"] = {
                                    "eval_usage": dict(eval_usage),
                                    "diagnostic_probe_usage": dict(
                                        diagnostic_probe_usage
                                    ),
                                }
                                _atomic_torch_save(payload, path)

                            kw["state_checkpoint_fn"] = save_round_state
                        if (args.checkpoint_every > 0
                                and _accepts(METHODS[method], "checkpoint_every")):
                            kw["checkpoint_every"] = args.checkpoint_every

                            def save_checkpoint(current_model, completed_rounds):
                                checkpoint_path = os.path.join(
                                    out,
                                    f"adapter_checkpoint_{args.task}__{mtag}__{method}_s{seed}"
                                    f"_r{completed_rounds:04d}",
                                )
                                current_model.save_pretrained(checkpoint_path)
                                artifact_paths.extend(
                                    adapter_artifacts(checkpoint_path)
                                )

                            kw["checkpoint_fn"] = save_checkpoint
                        recs = METHODS[method](task, rounds=args.rounds, B=args.batch, G=args.G, seed=seed,
                                               model_tok=(model, tok), log=_log, **kw)
                    for r in recs:
                        if "gen" in r:
                            r.setdefault("llm_gen", r["gen"])
                    bench, samples = benchmark(args.task, model, tok, args.n_test, args.shots, seed,
                                               args.eval_partition, args.eval_batch,
                                               args.shot_bank_size, args.answer_event_mode,
                                               args.evaluation_prompt)
                    final_eval_gen = len(samples) if samples else args.n_test
                    transfer_metrics, transfer_samples, transfer_metadata = (
                        benchmark_transfer(
                            args.transfer_eval_dataset,
                            model,
                            tok,
                            args.transfer_eval_n,
                            args.shots,
                            seed,
                            eval_batch=args.eval_batch,
                            shot_bank_size=args.shot_bank_size,
                            answer_event_mode=args.answer_event_mode,
                            train_partition=args.train_partition,
                        )
                    )
                    transfer_eval_gen = len(transfer_samples)
                    if transfer_samples:
                        bench = {**bench, **transfer_metrics}
                        transfer_path = os.path.join(
                            out,
                            f"eval_svamp_transfer__{mtag}__{method}_s{seed}.json",
                        )
                        with open(transfer_path, "w") as fh:
                            json.dump(
                                {
                                    "run_id": args.run_id,
                                    "model": model_name,
                                    "method": method,
                                    "seed": seed,
                                    **transfer_metadata,
                                    **transfer_metrics,
                                    "records": _compact_math_records(
                                        transfer_samples
                                    ),
                                },
                                fh,
                                default=str,
                            )
                        artifact_paths.append(transfer_path)
                    passk_gen = 0
                    if args.passk > 0 and args.task == "gsm8k":      # end-of-cell pass@K (incl. base = the reference
                        pk, pk_recs = B.passk_gsm8k(model, tok, n_test=args.passk_n,   # distribution): sharpening check
                                                    k=args.passk, n_shots=args.shots, seed=seed,
                                                    batch=args.eval_batch,
                                                    eval_partition=args.eval_partition,
                                                    shot_bank_size=(args.shot_bank_size or None),
                                                    answer_event_mode=args.answer_event_mode,
                                                    evaluation_prompt=args.evaluation_prompt)
                        bench = {**bench, **pk}                      # pass1 / passK -> CSV columns
                        passk_gen = len(pk_recs) * args.passk
                        passk_path = os.path.join(
                            out, f"passk_{args.task}__{mtag}__{method}_s{seed}.json"
                        )
                        with open(passk_path, "w") as fh:
                            json.dump(dict(run_id=args.run_id, model=model_name, method=method, seed=seed,
                                           **pk, records=pk_recs), fh, default=str)
                        artifact_paths.append(passk_path)
                    if args.save_adapter and method != "base":      # ~20-40MB LoRA adapter per cell: enables ANY
                        adapter_path = os.path.join(
                            out, f"adapter_{args.task}__{mtag}__{method}_s{seed}"
                        )
                        model.save_pretrained(adapter_path)
                        artifact_paths.extend(adapter_artifacts(adapter_path))
                    final = recs[-1] if recs else {}
                    train_llm_gen = int(final.get("llm_gen", final.get("gen", 0)) or 0)
                    eval_llm_gen = int(
                        eval_usage["generations"]
                        + diagnostic_probe_usage["generations"]
                        + final_eval_gen
                        + transfer_eval_gen
                        + passk_gen
                    )
                    total_llm_gen = train_llm_gen + eval_llm_gen
                    verifier_calls = int(final.get("verifier_calls", final.get("oracle", 0)) or 0)
                    oracle_gen = int(final.get("oracle_gen", 0) or 0)
                    if recs:                                        # per-round trajectory (ess/kl/div/...) for diagnosis
                        trajectory_path = os.path.join(
                            out, f"traj_{args.task}__{mtag}__{method}_s{seed}.json"
                        )
                        with open(trajectory_path, "w") as fh:
                            json.dump(recs, fh, default=float)
                        artifact_paths.append(trajectory_path)
                    if samples:                                     # per-QUESTION held-out results (EVERY item) ->
                        compact = [dict(idx=s.get("idx"), dataset_id=s.get("dataset_id"),
                                        correct=s.get("correct"),   # stratify by difficulty, align
                                        pred=s.get("pred"), gold=s.get("gold"),       # methods by (idx,seed)
                                        legacy_pred=s.get("legacy_pred"),
                                        legacy_correct=s.get("legacy_correct"),
                                        strict_pred=s.get("strict_pred"),
                                        strict_correct=s.get("strict_correct"),
                                        strict_correct_and_eos=s.get(
                                            "strict_correct_and_eos"
                                        ),
                                        has_nonempty_reasoning=s.get(
                                            "has_nonempty_reasoning"
                                        ),
                                        strict_correct_with_reasoning=s.get(
                                            "strict_correct_with_reasoning"
                                        ),
                                        direct_answer_only=s.get(
                                            "direct_answer_only"
                                        ),
                                        len=s.get("len"), rep4=s.get("rep4"),
                                        has_answer_marker=s.get("has_answer_marker"),
                                        format_failure=s.get("format_failure"),
                                        strict_format_failure=s.get("strict_format_failure"),
                                        answer_parse_mode=s.get("answer_parse_mode"),
                                        answer_event_mode=s.get("answer_event_mode"),
                                        evaluation_prompt=s.get("evaluation_prompt"),
                                        answer_marker_count=s.get("answer_marker_count"),
                                        answer_marker_terminal=s.get("answer_marker_terminal"),
                                        generated_eos=s.get("generated_eos"),
                                        generated_tokens_until_eos=s.get(
                                            "generated_tokens_until_eos"
                                        ),
                                        hit_max_new_tokens=s.get(
                                            "hit_max_new_tokens"
                                        ),
                                        generated_stop_token_id=s.get(
                                            "generated_stop_token_id"
                                        ),
                                        generated_eod_before_eot=s.get(
                                            "generated_eod_before_eot"
                                        ),
                                        official_test_accessed=s.get(
                                            "official_test_accessed"
                                        ),
                                        eval_source_split=s.get(
                                            "eval_source_split"
                                        ),
                                        dataset_splits_loaded=s.get(
                                            "dataset_splits_loaded"
                                        ))
                                   for s in samples]  # length/degeneration and parse mode
                        if args.task != "hendrycks_math_chat":
                            for record in compact:
                                record.pop("dataset_id", None)
                                record.pop("generated_stop_token_id", None)
                                record.pop("generated_eod_before_eot", None)
                                                                                      # at full n_test scale
                        eval_path = os.path.join(
                            out, f"eval_{args.task}__{mtag}__{method}_s{seed}.json"
                        )
                        eval_payload = dict(
                            run_id=args.run_id,
                            model=model_name,
                            method=method,
                            seed=seed,
                            train_qi=getattr(task, "train_qi", None),
                            shot_qi=getattr(task, "shot_qi", None),
                            **bench,
                            records=compact,
                        )
                        if train_dataset_ids is not None:
                            eval_payload.update(
                                train_dataset_ids=train_dataset_ids,
                                train_dataset_ids_sha256=(
                                    train_dataset_ids_sha256
                                ),
                            )
                        with open(eval_path, "w") as fh:
                            json.dump(eval_payload, fh, default=str)
                        artifact_paths.append(eval_path)
                    if samples and args.dump_completions > 0:        # first N held-out generations IN FULL to eyeball
                        dump_path = os.path.join(
                            out, f"dump_{args.task}__{mtag}__{method}_s{seed}.json"
                        )
                        with open(dump_path, "w") as fh:
                            json.dump(dict(run_id=args.run_id, method=method, seed=seed, **bench,
                                           train_qi=getattr(task, "train_qi", None),
                                           shot_qi=getattr(task, "shot_qi", None),
                                           samples=samples[:args.dump_completions]), fh, indent=2, default=str)
                        artifact_paths.append(dump_path)
                    if diagnostics_fh is not None:
                        diagnostics_fh.close()
                        diagnostics_fh = None
                        assert diagnostics_path is not None
                        artifact_paths.append(diagnostics_path)
                    if checkpoint_eval_fh is not None:
                        checkpoint_eval_fh.close()
                        checkpoint_eval_fh = None
                        assert checkpoint_eval_path is not None
                        artifact_paths.append(checkpoint_eval_path)
                    identity = cell_identity_value
                    fingerprint = cell_fingerprint_value
                    completion_path = receipt_path(
                        out, args.task, mtag, method, seed
                    )
                    completion_relative = completion_path.relative_to(Path(out)).as_posix()
                    cell_secs = round(time.time() - cell_t0, 1)
                    cuda_available = torch.cuda.is_available()
                    accelerator_count = (
                        torch.cuda.device_count() if cuda_available else 0
                    )
                    forward_receipt = (
                        forward_counter.snapshot()
                        if forward_counter is not None
                        else {
                            "model_forward_calls": None,
                            "model_forward_input_tokens": None,
                            "model_forward_keyword_inputs_observed": None,
                        }
                    )
                    chat_audit = getattr(
                        tok,
                        "_vrl_math_chat_generation_audit",
                        None,
                    )
                    if args.task == "hendrycks_math_chat":
                        if not isinstance(chat_audit, dict):
                            raise ValueError("Qwen3 chat generation audit is missing")
                        if int(chat_audit.get("sequences", 0)) <= 0:
                            raise ValueError("Qwen3 chat generation audit is empty")
                        if int(
                            chat_audit.get("generated_eod_before_eot_count", -1)
                        ) != 0:
                            raise ValueError(
                                "Qwen3 chat generated EOD before assistant EOT"
                            )
                    result_row = dict(run_id=args.run_id, model=model_name, task=args.task,
                                     train_partition=args.train_partition,
                                     eval_partition=args.eval_partition,
                                     evaluation_prompt=args.evaluation_prompt,
                                     task_seed=(
                                         seed if args.task_seed_from_run_seed else 0
                                     ),
                                     method=method, seed=seed,
                                     train_qi=getattr(task, "train_qi", None),
                                     shot_qi=getattr(task, "shot_qi", None),
                                     train_reward=final.get("mean_reward"),
                                     frac_correct=final.get("frac_correct"),  # base rate / signal at last round
                                     kl=final.get("kl"),              # final drift from base (degrade vs inert)
                                     oracle=final.get("oracle"),      # reward-oracle calls (AC-EM=0; the cost axis)
                                     verifier_calls=verifier_calls,
                                     diagnostic_verifier_calls=final.get("diagnostic_verifier_calls"),
                                     oracle_gen=oracle_gen,
                                     gen=final.get("gen"),            # completions generated (compute proxy)
                                     generated_tokens=final.get("generated_tokens"),
                                     teacher_forced_scoring_tokens=final.get("scored_tokens"),
                                     teacher_forced_scoring_tokens_status=(
                                         "trainer_reported_exact"
                                         if final.get("scored_tokens") is not None
                                         else "unavailable_not_yet_reported_by_trainer"
                                     ),
                                     backward_tokens=final.get("backward_tokens"),
                                     question_exposures=final.get("question_exposures"),
                                     unique_questions_seen=final.get("unique_questions_seen"),
                                     train_llm_gen=train_llm_gen,
                                     eval_llm_gen=eval_llm_gen,
                                     llm_gen=total_llm_gen,
                                     gsteps=final.get("gsteps"),      # gradient steps taken
                                     optimizer_steps=final.get("gsteps"),
                                     buffer_evictions=final.get("buffer_evictions"),
                                     cost=final.get("oracle"),        # back-compat alias = oracle calls
                                     gen_len=final.get("gen_len"),    # mean completion length (degeneration)
                                     ess=final.get("ess"),            # EM weight ESS (collapse); NaN for non-EM
                                     fmt=final.get("fmt"),            # last-round '####' format rate (train-time)
                                     samp_lp=final.get("samp_lp"),    # policy-entropy proxy at the last round
                                     gold_lp=final.get("gold_lp"),    # gold-CoT likelihood at the last round
                                     loss=final.get("loss"),
                                     secs=cell_secs,  # wall-clock compute (gen+train+bench)
                                     vram_gb=(round(torch.cuda.max_memory_allocated() / 2**30, 2)
                                              if cuda_available else None),  # peak VRAM this cell
                                     compute_accounting_version=1,
                                     **forward_receipt,
                                     accelerator_count=accelerator_count,
                                     accelerator_name=(
                                         torch.cuda.get_device_name(0)
                                         if cuda_available else None
                                     ),
                                     accelerator_hours=(
                                         cell_secs * accelerator_count / 3600.0
                                         if cuda_available else None
                                     ),
                                     peak_cuda_reserved_gb=(
                                         round(
                                             torch.cuda.max_memory_reserved()
                                             / 2**30,
                                             2,
                                         )
                                         if cuda_available else None
                                     ),
                                     gpu_utilization_mean_percent=None,
                                     gpu_utilization_status=(
                                         "unavailable_runtime_not_sampled"
                                         if cuda_available else "not_applicable_cpu"
                                     ),
                                     cell_fingerprint=fingerprint,
                                     completion_receipt=completion_relative,
                                     **bench)
                    if train_dataset_ids is not None:
                        result_row.update(
                            train_dataset_ids=train_dataset_ids,
                            train_dataset_ids_sha256=(
                                train_dataset_ids_sha256
                            ),
                        )
                    if args.task == "hendrycks_math_chat":
                        result_row.update(
                            **execution_binding,
                            source_job_id=os.environ.get("JOB_ID"),
                            source_task_id=os.environ.get("SGE_TASK_ID"),
                            chat_generation_audit=dict(chat_audit),
                            chat_runtime_contract=chat_runtime_contract,
                            lora_runtime_contract=lora_runtime_contract,
                        )
                    rows.append(result_row)
                    rows[0]["params"] = blob                        # all algo params live in ONE cell (row 0)
                    cell_result_path = Path(out) / (
                        f"cell_result_{args.task}__{mtag}__{method}_s{seed}.json"
                    )
                    atomic_write_json(
                        cell_result_path,
                        {"identity": identity, "result": result_row},
                    )
                    artifact_paths.append(cell_result_path)
                    _atomic_write_csv(rows, path)                    # checkpoint after every cell
                    write_completion_receipt(
                        completion_path,
                        identity=identity,
                        artifact_paths=artifact_paths,
                        result_root=out,
                    )
                    if round_state_path is not None:
                        round_state_path.unlink(missing_ok=True)
                    print(f"   -> {bench} | {tag}", flush=True)
                except Exception:                                   # log + skip; resume retries it later
                    import traceback
                    print(f"!! FAILED (skipping; will retry on resume): {tag}", flush=True)
                    traceback.print_exc()
                    failures.append(tag)
                finally:
                    if forward_counter is not None:
                        forward_counter.close()
                    if diagnostics_fh is not None:
                        diagnostics_fh.close()
                    if checkpoint_eval_fh is not None:
                        checkpoint_eval_fh.close()
                    if cell_lock_fh is not None:
                        release_cell_lock(cell_lock_fh)
                    del model, tok
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    print(f"done in {time.time()-t0:.0f}s -> {path}")
    if failures:
        print(f"FAILED: {len(failures)} cell(s): {', '.join(failures)}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
