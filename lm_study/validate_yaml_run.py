#!/usr/bin/env python3
"""Validate every artifact expected from one run_yaml experiment."""
from __future__ import annotations

import argparse
import ast
import gzip
import glob
import hashlib
import json
import math
from pathlib import Path
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yaml

from experiment_config import validate_run_yaml_config
from result_contract import (
    ResultContractError,
    atomic_write_json,
    validate_completion_receipt,
    validate_receipt_identity,
)
from run_yaml import _models, _prepare_cells


FAILURE_RE = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemoryError|No space left on device|"
    r"Disk quota exceeded|FAILED",
    re.IGNORECASE,
)

_EXTERNAL_REWARD_ONLY_METHOD_FAMILIES = {
    "GRPO": "grpo",
    "RLOO": "rloo",
}

_SOURCE_SELF_TRAINING_METHODS = {"RFT-Source", "ReST-EM", "STaR"}


def _glob_paths(pattern: str) -> list[Path]:
    """Return relative or absolute glob matches as paths."""

    expanded = str(Path(pattern).expanduser())
    return [Path(path) for path in sorted(glob.glob(expanded))]


def _read_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _as_mapping(value: Any, *, context: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _as_string_list(value: Any) -> list[str] | None:
    """Normalise list-valued provenance read directly or through a CSV row."""

    candidate = value
    if isinstance(candidate, str):
        try:
            candidate = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(candidate, list) or not all(
        isinstance(item, str) for item in candidate
    ):
        return None
    return candidate


def _expected_result_access_fields(*, passk_enabled: bool) -> dict[str, Any]:
    """Return only the provenance columns expected from enabled evaluations."""

    fields: dict[str, Any] = {
        "eval_official_test_accessed": False,
        "eval_source_split": "train",
        "eval_dataset_splits_loaded": ["train"],
    }
    if passk_enabled:
        fields.update(
            {
                "passk_official_test_accessed": False,
                "passk_eval_source_split": "train",
                "passk_dataset_splits_loaded": ["train"],
            }
        )
    return fields


def _read_jsonl_gz(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def _validate_eos_generation_records(
    problems: list[str],
    *,
    path: Path,
    records: list[dict],
    require_outcome_diagnostics: bool = False,
) -> None:
    """Require coherent natural-termination and optional outcome diagnostics."""

    for index, record in enumerate(records):
        generated_eos = record.get("generated_eos")
        generated_tokens = record.get("generated_tokens_until_eos")
        hit_limit = record.get("hit_max_new_tokens")
        if not isinstance(generated_eos, bool):
            problems.append(f"{path}: record {index} omits generated_eos")
            return
        if not isinstance(generated_tokens, int) or generated_tokens < 1:
            problems.append(
                f"{path}: record {index} has invalid generated_tokens_until_eos"
            )
            return
        if not isinstance(hit_limit, bool):
            problems.append(f"{path}: record {index} omits hit_max_new_tokens")
            return
        if generated_eos and hit_limit:
            problems.append(
                f"{path}: record {index} cannot emit EOS and hit the token limit"
            )
            return
        if not require_outcome_diagnostics:
            continue
        strict_correct = record.get("strict_correct")
        strict_correct_and_eos = record.get("strict_correct_and_eos")
        has_reasoning = record.get("has_nonempty_reasoning")
        strict_correct_with_reasoning = record.get(
            "strict_correct_with_reasoning"
        )
        direct_answer_only = record.get("direct_answer_only")
        strict_format_failure = record.get("strict_format_failure")
        values = {
            "strict_correct": strict_correct,
            "strict_correct_and_eos": strict_correct_and_eos,
            "has_nonempty_reasoning": has_reasoning,
            "strict_correct_with_reasoning": strict_correct_with_reasoning,
            "direct_answer_only": direct_answer_only,
            "strict_format_failure": strict_format_failure,
        }
        invalid = [
            field for field, value in values.items()
            if not isinstance(value, bool)
        ]
        if invalid:
            problems.append(
                f"{path}: record {index} omits boolean protocol diagnostics "
                f"{invalid}"
            )
            return
        if strict_correct_and_eos != (strict_correct and generated_eos):
            problems.append(
                f"{path}: record {index} has incoherent strict-correct/EOS joint"
            )
            return
        if strict_correct_with_reasoning != (
            strict_correct and has_reasoning
        ):
            problems.append(
                f"{path}: record {index} has incoherent reasoning-correct joint"
            )
            return
        if direct_answer_only != (
            not strict_format_failure and not has_reasoning
        ):
            problems.append(
                f"{path}: record {index} has incoherent direct-answer diagnostic"
            )
            return


def _validate_official_test_access_records(
    problems: list[str],
    *,
    path: Path,
    records: list[dict],
) -> None:
    """Require runtime evidence that validation loaded only GSM8K train."""

    for index, record in enumerate(records):
        if record.get("official_test_accessed") is not False:
            problems.append(
                f"{path}: record {index} does not prove official-test non-access"
            )
            return
        if record.get("eval_source_split") != "train":
            problems.append(
                f"{path}: record {index} eval_source_split is not 'train'"
            )
            return
        if record.get("dataset_splits_loaded") != ["train"]:
            problems.append(
                f"{path}: record {index} loaded dataset splits are not ['train']"
            )
            return


def _validate_eos_training_diagnostics(
    problems: list[str],
    *,
    path: Path,
    diagnostics: list[dict],
    answer_target_termination: str,
    latent_mstep_objective: str = "joint",
    method: str | None = None,
    protocol: dict[str, Any] | None = None,
) -> None:
    """Require target-scale and first-step gradient evidence for EOS screens."""

    eos_target = answer_target_termination == "eos"
    protocol = protocol or {}
    if method == "Gold-CoT-SFT" or method in _SOURCE_SELF_TRAINING_METHODS:
        expected_family = (
            "gold_cot_sft"
            if method == "Gold-CoT-SFT"
            else "source_self_training"
        )
        previous_eos = 0
        previous_steps = 0
        expected_modes = {
            "RFT-Source": "rft_source",
            "ReST-EM": "rest_em",
            "STaR": "star",
        }
        iterations = int(protocol.get("self_train_iterations", 1))
        rounds = len(diagnostics)
        prompts = int(protocol.get("prompts", 0))
        batch = int(protocol.get("batch", 0))
        generations = int(protocol.get("G", 0))
        epochs = int(protocol.get("epochs", 1))
        micro = int(protocol.get("micro", 4))
        if rounds <= 0 or prompts <= 0 or batch <= 0 or generations <= 0:
            problems.append(f"{path}: incomplete SFT protocol dimensions")
            return
        if rounds % iterations:
            problems.append(f"{path}: SFT phase schedule does not divide rounds")
            return
        rounds_per_phase = rounds // iterations
        for index, diagnostic in enumerate(diagnostics):
            if diagnostic.get("method_family") != expected_family:
                problems.append(
                    f"{path}: record {index} declares method_family="
                    f"{diagnostic.get('method_family')!r}, expected "
                    f"{expected_family!r}"
                )
                return
            if diagnostic.get("answer_target_termination") != "eos":
                problems.append(
                    f"{path}: record {index} omits the terminal EOS target"
                )
                return
            optimizer = diagnostic.get("optimizer") or {}
            steps = optimizer.get("gradient_steps_this_round")
            cumulative_steps = optimizer.get("gradient_steps_cumulative")
            eos_this_round = optimizer.get("backward_eos_tokens_this_round")
            eos_cumulative = optimizer.get("backward_eos_tokens_cumulative")
            if not all(
                isinstance(value, int) and value >= 0
                for value in (
                    steps,
                    cumulative_steps,
                    eos_this_round,
                    eos_cumulative,
                )
            ):
                problems.append(
                    f"{path}: record {index} has invalid SFT EOS/step counts"
                )
                return
            if eos_cumulative != previous_eos + eos_this_round:
                problems.append(
                    f"{path}: record {index} has incoherent cumulative EOS count"
                )
                return
            if (steps > 0) != (eos_this_round > 0):
                problems.append(
                    f"{path}: record {index} disagrees on active SFT steps and EOS support"
                )
                return
            if cumulative_steps != previous_steps + steps:
                problems.append(
                    f"{path}: record {index} has incoherent cumulative step count"
                )
                return
            previous_steps = cumulative_steps
            previous_eos = eos_cumulative
            if method in _SOURCE_SELF_TRAINING_METHODS:
                reward = diagnostic.get("reward") or {}
                generation = diagnostic.get("generation") or {}
                phase = diagnostic.get("phase") or {}
                expected_improve = (index + 1) % rounds_per_phase == 0
                expected_phase = index // rounds_per_phase
                if diagnostic.get("method") != expected_modes[method]:
                    problems.append(f"{path}: record {index} has the wrong source method")
                    return
                if (
                    phase.get("index") != expected_phase
                    or phase.get("improve_after_round") is not expected_improve
                    or phase.get("reset_to_original_adapter") is not expected_improve
                ):
                    problems.append(f"{path}: record {index} violates the phase/reset schedule")
                    return
                natural_eos = generation.get("natural_eos_fraction")
                if reward.get("requires_natural_eos") is not True:
                    problems.append(
                        f"{path}: record {index} does not EOS-gate accepted traces"
                    )
                    return
                if (
                    not isinstance(natural_eos, (int, float))
                    or not math.isfinite(float(natural_eos))
                    or not 0.0 <= float(natural_eos) <= 1.0
                ):
                    problems.append(
                        f"{path}: record {index} has invalid natural-EOS rate"
                    )
                    return
                accepted = phase.get("accepted_examples")
                accepted_limit = prompts * int(
                    protocol.get("accepted_per_question", 1)
                )
                if not isinstance(accepted, int) or not 0 <= accepted <= accepted_limit:
                    problems.append(f"{path}: record {index} violates the acceptance cap")
                    return
                if not expected_improve and steps != 0:
                    problems.append(f"{path}: record {index} updates outside Improve")
                    return
                if expected_improve and (steps > 0) != (accepted > 0):
                    problems.append(
                        f"{path}: record {index} disagrees on accepted data and Improve steps"
                    )
                    return
        final_generation = diagnostics[-1].get("generation") or {}
        if method == "Gold-CoT-SFT":
            expected_exposures = prompts * epochs
            expected_steps = math.ceil((batch // generations) / micro) * rounds
            if final_generation.get("generations_cumulative") != 0:
                problems.append(f"{path}: Gold-CoT-SFT unexpectedly generated traces")
                return
            if previous_steps != expected_steps:
                problems.append(f"{path}: Gold-CoT-SFT optimizer-step schedule changed")
                return
        else:
            expected_exposures = prompts * iterations
            final_generated = final_generation.get("generations_cumulative")
            if method in {"RFT-Source", "ReST-EM"}:
                expected_generated = rounds * batch
                if final_generated != expected_generated:
                    problems.append(f"{path}: source proposal budget changed")
                    return
            elif not prompts * iterations <= int(final_generated) <= 2 * prompts * iterations:
                problems.append(f"{path}: STaR rationalization budget is out of range")
                return
        if final_generation.get("question_exposures_cumulative") != expected_exposures:
            problems.append(f"{path}: SFT question-exposure schedule changed")
            return
        return

    if method == "TRICE":
        previous_steps = 0
        batch = int(protocol.get("batch", 0))
        prompts = int(protocol.get("prompts", 0))
        estimator = str(protocol.get("trice_estimator", ""))
        if batch <= 0 or prompts <= 0 or estimator not in {"basic", "control_variate"}:
            problems.append(f"{path}: incomplete TRICE protocol dimensions")
            return
        for index, diagnostic in enumerate(diagnostics):
            if diagnostic.get("method_family") != "trice":
                problems.append(
                    f"{path}: record {index} does not declare TRICE diagnostics"
                )
                return
            if diagnostic.get("answer_target_termination") != "eos":
                problems.append(
                    f"{path}: record {index} omits the terminal EOS contract"
                )
                return
            reward = diagnostic.get("reward") or {}
            generation = diagnostic.get("generation") or {}
            optimizer = diagnostic.get("optimizer") or {}
            trice = diagnostic.get("trice") or {}
            contract = trice.get("parameter_contract") or {}
            expected_contract = {
                "proposal_temperature": 1.0,
                "proposal_prompt": "question",
                "reward_requires_eos": True,
                "frozen_until_optimizer_step": True,
            }
            if any(contract.get(key) != value for key, value in expected_contract.items()):
                problems.append(f"{path}: record {index} violates the TRICE proposal contract")
                return
            updates = contract.get("optimizer_updates_per_macrocycle")
            cumulative_steps = optimizer.get("gradient_steps_cumulative")
            if updates not in {0, 1} or cumulative_steps != previous_steps + updates:
                problems.append(f"{path}: record {index} violates one-update TRICE")
                return
            previous_steps = cumulative_steps
            if trice.get("estimator") != estimator:
                problems.append(f"{path}: record {index} uses the wrong TRICE estimator")
                return
            if trice.get("proposals_this_macrocycle") != batch:
                problems.append(f"{path}: record {index} changes the TRICE proposal count")
                return
            if generation.get("prior_generations_cumulative") != (index + 1) * batch:
                problems.append(f"{path}: record {index} has the wrong prior budget")
                return
            if generation.get("guide_generations_cumulative") != prompts:
                problems.append(f"{path}: record {index} did not initialize every chain once")
                return
            if reward.get("requires_natural_eos") is not True:
                problems.append(
                    f"{path}: record {index} does not EOS-gate TRICE acceptance"
                )
                return
            natural_eos = generation.get("natural_eos_fraction")
            accepted_eos = generation.get("accepted_state_eos_fraction")
            if (
                not isinstance(natural_eos, (int, float))
                or not math.isfinite(float(natural_eos))
                or not 0.0 <= float(natural_eos) <= 1.0
            ):
                problems.append(
                    f"{path}: record {index} has invalid TRICE natural-EOS rate"
                )
                return
            if accepted_eos is not None and not math.isclose(
                float(accepted_eos), 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                problems.append(
                    f"{path}: record {index} has a non-EOS accepted TRICE state"
                )
                return
        return

    external_method_family = _EXTERNAL_REWARD_ONLY_METHOD_FAMILIES.get(method)
    support_fields = (
        "supervised_backward_tokens",
        "supervised_backward_eos_tokens",
        "buffer_backward_tokens",
        "buffer_backward_eos_tokens",
        "backward_tokens",
        "backward_eos_tokens",
    )
    component_fields = ("B_sup", "B_prime_unsup", "B_unsup", "total")
    for index, diagnostic in enumerate(diagnostics):
        observed = diagnostic.get("answer_target_termination")
        if external_method_family is not None and not eos_target:
            if observed not in {None, "none"}:
                problems.append(
                    f"{path}: record {index} declares answer_target_termination="
                    f"{observed!r}, expected no teacher-forced target"
                )
                return
            if diagnostic.get("method_family") != external_method_family:
                problems.append(
                    f"{path}: record {index} declares method_family="
                    f"{diagnostic.get('method_family')!r}, expected "
                    f"{external_method_family!r}"
                )
                return
            continue
        if observed != answer_target_termination:
            problems.append(
                f"{path}: record {index} declares answer_target_termination="
                f"{observed!r}, expected {answer_target_termination!r}"
            )
            return
        steps = (diagnostic.get("inner_m_step") or {}).get("steps") or []
        if not steps:
            problems.append(f"{path}: record {index} has no inner M-step records")
            return
        first_step = steps[0]
        support = first_step.get("support") or {}
        invalid_support = [
            field
            for field in support_fields
            if not isinstance(support.get(field), int) or support[field] < 0
        ]
        if invalid_support:
            problems.append(
                f"{path}: record {index} first step has invalid support fields "
                f"{invalid_support}"
            )
            return
        if support["backward_tokens"] != (
            support["supervised_backward_tokens"]
            + support["buffer_backward_tokens"]
        ):
            problems.append(
                f"{path}: record {index} first-step backward-token totals disagree"
            )
            return
        if support["backward_eos_tokens"] != (
            support["supervised_backward_eos_tokens"]
            + support["buffer_backward_eos_tokens"]
        ):
            problems.append(
                f"{path}: record {index} first-step EOS-token totals disagree"
            )
            return
        components = (
            (
                "supervised",
                "supervised_backward_tokens",
                "supervised_backward_eos_tokens",
                eos_target,
            ),
            (
                "buffer",
                "buffer_backward_tokens",
                "buffer_backward_eos_tokens",
                eos_target
                and latent_mstep_objective
                in {"joint", "joint_token_mean", "answer"},
            ),
        )
        for label, token_field, eos_field, component_requires_eos in components:
            active = support[token_field] > 0
            eos_count = support[eos_field]
            if active and component_requires_eos and eos_count < 1:
                problems.append(
                    f"{path}: record {index} active {label} component has no "
                    "backward EOS token"
                )
                return
            if (
                eos_target
                and active
                and not component_requires_eos
                and eos_count != 0
            ):
                problems.append(
                    f"{path}: record {index} rationale-only {label} component "
                    f"has {eos_count} backward EOS token(s)"
                )
                return
            if not eos_target and eos_count != 0:
                problems.append(
                    f"{path}: record {index} no-EOS target has {eos_count} "
                    f"backward EOS token(s) in the {label} component"
                )
                return
        eos_component_active = any(
            support[token_field] > 0 and component_requires_eos
            for _, token_field, _, component_requires_eos in components
        )
        if eos_component_active and support["backward_eos_tokens"] < 1:
            problems.append(
                f"{path}: record {index} EOS target has no backward EOS token"
            )
            return
        if (
            eos_target
            and not eos_component_active
            and support["backward_eos_tokens"] != 0
        ):
            problems.append(
                f"{path}: record {index} inactive answer objective has aggregate "
                "backward EOS support"
            )
            return
        if not eos_target and support["backward_eos_tokens"] != 0:
            problems.append(
                f"{path}: record {index} no-EOS target has aggregate backward "
                "EOS support"
            )
            return
        component_geometry = first_step.get("component_gradient_geometry")
        if not isinstance(component_geometry, dict):
            problems.append(
                f"{path}: record {index} omits first-step component gradients"
            )
            return
        norms = component_geometry.get("norms") or {}
        invalid_norms = [
            field
            for field in component_fields
            if (
                not isinstance(norms.get(field), (int, float))
                or not math.isfinite(float(norms[field]))
                or float(norms[field]) < 0
            )
        ]
        if invalid_norms:
            problems.append(
                f"{path}: record {index} has invalid first-step component "
                f"gradient norms {invalid_norms}"
            )
            return


def _validate_answer_conditioned_importance_diagnostics(
    problems: list[str],
    *,
    path: Path,
    diagnostics: list[dict],
) -> None:
    """Require finite proposal-density evidence and coherent ACIS logits."""

    for index, diagnostic in enumerate(diagnostics):
        traces = (diagnostic.get("responsibilities") or {}).get("traces") or []
        if not traces:
            problems.append(
                f"{path}: record {index} has no importance-weighted traces"
            )
            return
        for trace_index, trace in enumerate(traces):
            values = {
                field: trace.get(field)
                for field in (
                    "trace_logprob",
                    "answer_logprob",
                    "proposal_trace_logprob",
                    "log_importance_correction",
                    "responsibility_logit",
                )
            }
            invalid = [
                field
                for field, value in values.items()
                if not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ]
            if invalid:
                problems.append(
                    f"{path}: record {index} trace {trace_index} has "
                    f"nonfinite importance fields {invalid}"
                )
                return
            expected_correction = (
                float(values["trace_logprob"])
                - float(values["proposal_trace_logprob"])
            )
            expected_logit = (
                expected_correction + float(values["answer_logprob"])
            )
            if not math.isclose(
                float(values["log_importance_correction"]),
                expected_correction,
                rel_tol=1e-5,
                abs_tol=1e-5,
            ):
                problems.append(
                    f"{path}: record {index} trace {trace_index} has an "
                    "incoherent proposal correction"
                )
                return
            if not math.isclose(
                float(values["responsibility_logit"]),
                expected_logit,
                rel_tol=1e-5,
                abs_tol=1e-5,
            ):
                problems.append(
                    f"{path}: record {index} trace {trace_index} has an "
                    "incoherent importance logit"
                )
                return


def _seed_artifact_groups(defaults: dict) -> list[tuple[str, tuple[int, ...]]]:
    """Return ``(tag_suffix, artifact_seeds)`` for each execution group."""

    seeds = int(defaults.get("seeds", 1))
    seed_values = defaults.get("seed_values")
    if seed_values is None:
        return [("", tuple(range(seeds)))]
    if not isinstance(seed_values, (list, tuple)):
        raise ValueError("defaults.seed_values must be a list of integers")
    try:
        actual = tuple(int(value) for value in seed_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("defaults.seed_values must contain integers") from exc
    if len(actual) != seeds:
        raise ValueError("defaults.seeds must equal len(defaults.seed_values)")
    if any(value < 0 for value in actual):
        raise ValueError("defaults.seed_values must be nonnegative")
    if len(set(actual)) != len(actual):
        raise ValueError("defaults.seed_values must be unique")
    return [(f"_seed{seed}", (seed,)) for seed in actual]


def validate(config_path: Path, log_glob: str | None = None) -> list[str]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        validate_run_yaml_config(cfg)
        run_id = str(cfg["run_id"])
    except (KeyError, TypeError, ValueError) as exc:
        return [f"invalid experiment config: {exc}"]
    defaults = cfg.get("defaults") or {}
    diagnostic = cfg.get("diagnostic") or {}
    require_protocol_outcomes = bool(
        diagnostic.get("require_protocol_outcome_diagnostics")
    )
    partitions = diagnostic.get("partitions") or {}
    require_official_test_unaccessed = (
        partitions.get("official_test_used") is False
    )
    task = defaults.get("task", "gsm8k")
    out = Path(str(defaults.get("out", "~/po_results"))).expanduser()
    try:
        cells = _prepare_cells(
            cfg,
            only=None,
            run_id=run_id,
            defaults=defaults,
        )
    except ValueError as exc:
        return [f"invalid experiment cells: {exc}"]
    problems = []
    if (
        require_official_test_unaccessed
        and defaults.get("eval_partition") != "validation"
    ):
        problems.append(
            "official-test non-access requires defaults.eval_partition='validation'"
        )
    try:
        seed_groups = _seed_artifact_groups(defaults)
    except ValueError as exc:
        return [str(exc)]

    for model, method, axes, base_tag in cells:
        rounds = int(axes.get("rounds", defaults.get("rounds", 60)))
        answer_target_termination = str(
            axes.get(
                "answer_target_termination",
                defaults.get("answer_target_termination", "none"),
            )
        )
        for tag_suffix, artifact_seeds in seed_groups:
            tag = base_tag + tag_suffix
            csv_path = out / f"sweep_{task}__{tag}.csv"
            if not csv_path.is_file():
                problems.append(f"missing sweep: {csv_path}")
                continue
            try:
                frame = pd.read_csv(csv_path)
            except Exception as exc:
                problems.append(f"unreadable sweep {csv_path}: {exc}")
                continue
            required_columns = {"model", "method", "seed"}
            missing_columns = required_columns - set(frame.columns)
            if missing_columns:
                problems.append(
                    f"{csv_path}: missing required columns {sorted(missing_columns)}"
                )
                continue
            selected = frame[
                (frame["model"].astype(str) == str(model))
                & (frame["method"].astype(str) == method)
            ]
            if len(selected) != len(artifact_seeds):
                problems.append(
                    f"{csv_path}: expected {len(artifact_seeds)} "
                    f"{model}/{method} row(s), found {len(selected)}"
                )
            numeric_seeds = pd.to_numeric(selected["seed"], errors="coerce")
            if numeric_seeds.isna().any():
                problems.append(f"{csv_path}: seed column contains non-integers")
            observed_seeds = tuple(
                sorted(int(value) for value in numeric_seeds.dropna().tolist())
            )
            if observed_seeds != tuple(sorted(artifact_seeds)):
                problems.append(
                    f"{csv_path}: expected seeds {list(artifact_seeds)}, "
                    f"found {list(observed_seeds)}"
                )

            for seed in artifact_seeds:
                matching_rows = selected[
                    numeric_seeds == int(seed)
                ]
                result_row = (
                    matching_rows.iloc[0].to_dict()
                    if len(matching_rows) == 1
                    else None
                )
                _validate_seed_artifacts(
                    problems,
                    out=out,
                    defaults=defaults,
                    run_id=run_id,
                    task=task,
                    model=model,
                    method=method,
                    tag=tag,
                    seed=seed,
                    rounds=rounds,
                    result_row=result_row,
                    answer_target_termination=answer_target_termination,
                    latent_mstep_objective=str(
                        axes.get("latent_mstep_objective", "joint")
                    ),
                    reward_requires_eos=bool(
                        axes.get("reward_requires_eos", False)
                    ),
                    variational_estimator=str(
                        axes.get("variational_estimator", "delta_joint")
                    ),
                    protocol={**defaults, **axes},
                    require_protocol_outcomes=require_protocol_outcomes,
                    require_official_test_unaccessed=(
                        require_official_test_unaccessed
                    ),
                )

    if log_glob:
        logs = _glob_paths(log_glob)
        if not logs:
            problems.append(f"no logs matched {log_glob!r}")
        for path in logs:
            match = FAILURE_RE.search(path.read_text(encoding="utf-8", errors="replace"))
            if match:
                problems.append(f"{path}: failure signature {match.group(0)!r}")

    if not cells:
        problems.append("configuration expanded to zero cells")
    if len(_models(cfg)) != 1:
        problems.append("validator currently requires one model per config")
    return problems


def _validate_seed_artifacts(
    problems: list[str],
    *,
    out: Path,
    defaults: dict,
    run_id: str,
    task: str,
    model: str,
    method: str,
    tag: str,
    seed: int,
    rounds: int,
    result_row: dict | None,
    answer_target_termination: str,
    latent_mstep_objective: str,
    reward_requires_eos: bool,
    variational_estimator: str,
    protocol: dict[str, Any],
    require_protocol_outcomes: bool,
    require_official_test_unaccessed: bool,
) -> None:
    eos_target = answer_target_termination == "eos"
    if result_row is not None:
        if eos_target or require_protocol_outcomes or reward_requires_eos:
            try:
                params = _as_mapping(
                    json.loads(str(result_row.get("params") or "")),
                    context="result params",
                )
                sweep = _as_mapping(
                    params.get("sweep") or {},
                    context="result params sweep",
                )
                observed_termination = sweep.get(
                    "answer_target_termination"
                )
                if observed_termination != answer_target_termination:
                    problems.append(
                        f"{model}/{method}/seed={seed}: result params do not "
                        "declare answer_target_termination="
                        f"{answer_target_termination!r}"
                    )
                if reward_requires_eos:
                    methods = _as_mapping(
                        params.get("methods") or {},
                        context="result params methods",
                    )
                    method_params = _as_mapping(
                        methods.get(method) or {},
                        context=f"result params method {method}",
                    )
                    if method_params.get("reward_requires_eos") is not True:
                        problems.append(
                            f"{model}/{method}/seed={seed}: result params do "
                            "not declare reward_requires_eos=true"
                        )
            except (TypeError, ValueError) as exc:
                problems.append(
                    f"{model}/{method}/seed={seed}: invalid result params: {exc}"
                )
        fingerprint = result_row.get("cell_fingerprint")
        receipt = result_row.get("completion_receipt")
        if pd.notna(fingerprint) and str(fingerprint):
            if pd.isna(receipt) or not str(receipt):
                problems.append(
                    f"{model}/{method}/seed={seed}: missing completion receipt"
                )
            else:
                try:
                    payload = validate_completion_receipt(
                        out / str(receipt),
                        expected_fingerprint=str(fingerprint),
                        result_root=out,
                        verify_hashes=True,
                    )
                    validate_receipt_identity(
                        payload,
                        {
                            "run_id": run_id,
                            "model": model,
                            "method": method,
                            "seed": seed,
                        },
                    )
                except ResultContractError as exc:
                    problems.append(
                        f"{model}/{method}/seed={seed}: invalid completion "
                        f"receipt: {exc}"
                    )
        if require_official_test_unaccessed:
            access_fields = _expected_result_access_fields(
                passk_enabled=int(defaults.get("passk", 0)) > 0
            )
            for field, expected in access_fields.items():
                observed = result_row.get(field)
                if isinstance(expected, list):
                    observed = _as_string_list(observed)
                if observed != expected:
                    problems.append(
                        f"{model}/{method}/seed={seed}: {field}={observed!r}, "
                        f"expected {expected!r}"
                    )

    eval_path = out / f"eval_{task}__{tag}__{method}_s{seed}.json"
    if not eval_path.is_file():
        problems.append(f"missing final evaluation: {eval_path}")
    else:
        try:
            payload = _read_json(eval_path)
            records = payload.get("records") or []
            if len(records) != int(defaults.get("n_test", 200)):
                problems.append(f"{eval_path}: incomplete per-question records")
            elif eos_target or require_protocol_outcomes:
                _validate_eos_generation_records(
                    problems,
                    path=eval_path,
                    records=records,
                    require_outcome_diagnostics=require_protocol_outcomes,
                )
            if require_official_test_unaccessed and len(records) == int(
                defaults.get("n_test", 200)
            ):
                _validate_official_test_access_records(
                    problems,
                    path=eval_path,
                    records=records,
                )
        except Exception as exc:
            problems.append(f"invalid final evaluation {eval_path}: {exc}")

    if int(defaults.get("dump_completions", 0)) > 0:
        dump_path = out / f"dump_{task}__{tag}__{method}_s{seed}.json"
        if not dump_path.is_file():
            problems.append(f"missing completion dump: {dump_path}")
        else:
            try:
                payload = _read_json(dump_path)
                expected = min(
                    int(defaults["dump_completions"]),
                    int(defaults.get("n_test", 200)),
                )
                if len(payload.get("samples") or []) != expected:
                    problems.append(f"{dump_path}: incomplete completion samples")
            except Exception as exc:
                problems.append(f"invalid completion dump {dump_path}: {exc}")

    if method == "base":
        if int(defaults.get("passk", 0)) > 0:
            passk_path = out / f"passk_{task}__{tag}__{method}_s{seed}.json"
            if not passk_path.is_file():
                problems.append(f"missing pass@K evaluation: {passk_path}")
            else:
                try:
                    payload = _read_json(passk_path)
                    if len(payload.get("records") or []) != int(
                        defaults.get("passk_n", 100)
                    ):
                        problems.append(f"{passk_path}: incomplete pass@K records")
                    elif require_official_test_unaccessed:
                        if payload.get("passk_official_test_accessed") is not False:
                            problems.append(
                                f"{passk_path}: does not prove official-test non-access"
                            )
                        if payload.get("passk_eval_source_split") != "train":
                            problems.append(
                                f"{passk_path}: passk_eval_source_split is not 'train'"
                            )
                        if payload.get("passk_dataset_splits_loaded") != ["train"]:
                            problems.append(
                                f"{passk_path}: loaded dataset splits are not ['train']"
                            )
                        _validate_official_test_access_records(
                            problems,
                            path=passk_path,
                            records=payload.get("records") or [],
                        )
                except Exception as exc:
                    problems.append(f"invalid pass@K evaluation {passk_path}: {exc}")
        return

    trajectory_path = out / f"traj_{task}__{tag}__{method}_s{seed}.json"
    if not trajectory_path.is_file():
        problems.append(f"missing trajectory: {trajectory_path}")
    else:
        try:
            trajectory = _read_json(trajectory_path)
            if len(trajectory) != rounds:
                problems.append(
                    f"{trajectory_path}: expected {rounds} rounds, found {len(trajectory)}"
                )
            elif int(trajectory[-1].get("round", -1)) != rounds - 1:
                problems.append(f"{trajectory_path}: terminal round marker is wrong")
            elif reward_requires_eos:
                invalid_rounds = [
                    int(record.get("round", index))
                    for index, record in enumerate(trajectory)
                    if (
                        record.get("reward_requires_eos") is not True
                        or not isinstance(
                            record.get("natural_eos_fraction"),
                            (int, float),
                        )
                        or not math.isfinite(
                            float(record.get("natural_eos_fraction", float("nan")))
                        )
                    )
                ]
                if invalid_rounds:
                    problems.append(
                        f"{trajectory_path}: EOS-gated reward evidence is "
                        f"missing at rounds {invalid_rounds}"
                    )
        except Exception as exc:
            problems.append(f"invalid trajectory {trajectory_path}: {exc}")

    if defaults.get("save_training_diagnostics"):
        diagnostics_path = (
            out
            / f"training_diagnostics_{task}__{tag}__{method}_s{seed}.jsonl.gz"
        )
        if not diagnostics_path.is_file():
            problems.append(f"missing training diagnostics: {diagnostics_path}")
        else:
            try:
                diagnostics = _read_jsonl_gz(diagnostics_path)
                if len(diagnostics) != rounds:
                    problems.append(
                        f"{diagnostics_path}: expected {rounds} records, "
                        f"found {len(diagnostics)}"
                    )
                elif require_protocol_outcomes:
                    _validate_eos_training_diagnostics(
                        problems,
                        path=diagnostics_path,
                        diagnostics=diagnostics,
                        answer_target_termination=answer_target_termination,
                        latent_mstep_objective=latent_mstep_objective,
                        method=method,
                        protocol=protocol,
                    )
                if (
                    len(diagnostics) == rounds
                    and variational_estimator in {
                        "answer_conditioned_importance",
                        "persistent_answer_conditioned_importance",
                    }
                ):
                    _validate_answer_conditioned_importance_diagnostics(
                        problems,
                        path=diagnostics_path,
                        diagnostics=diagnostics,
                    )
                if reward_requires_eos and len(diagnostics) == rounds:
                    invalid_rounds = [
                        int(diagnostic.get("round", index))
                        for index, diagnostic in enumerate(diagnostics)
                        if (
                            (diagnostic.get("reward") or {}).get(
                                "requires_natural_eos"
                            )
                            is not True
                            or not isinstance(
                                (diagnostic.get("generation") or {}).get(
                                    "natural_eos_fraction"
                                ),
                                (int, float),
                            )
                            or not math.isfinite(
                                float(
                                    (diagnostic.get("generation") or {}).get(
                                        "natural_eos_fraction",
                                        float("nan"),
                                    )
                                )
                            )
                        )
                    ]
                    if invalid_rounds:
                        problems.append(
                            f"{diagnostics_path}: EOS-gated reward evidence is "
                            f"missing at rounds {invalid_rounds}"
                        )
                utility_questions = int(
                    defaults.get("l2r_candidate_utility_questions", 0)
                )
                if utility_questions:
                    evaluated_candidates = 0
                    for diagnostic in diagnostics:
                        configuration = diagnostic.get("configuration") or {}
                        question_ids = configuration.get(
                            "candidate_utility_question_ids"
                        ) or []
                        safety_ids = configuration.get(
                            "trust_safety_question_ids"
                        ) or []
                        if len(question_ids) != utility_questions:
                            problems.append(
                                f"{diagnostics_path}: candidate utility reserve "
                                f"has {len(question_ids)} questions, expected "
                                f"{utility_questions}"
                            )
                            break
                        if set(question_ids) & set(safety_ids):
                            problems.append(
                                f"{diagnostics_path}: candidate utility and "
                                "trust-safety reserves overlap"
                            )
                            break
                        partition = diagnostic.get("candidate_utility") or {}
                        if not all(
                            partition.get(field) is True
                            for field in (
                                "enabled",
                                "disjoint_from_optimization",
                                "disjoint_from_trust_safety",
                                "disjoint_from_validation",
                            )
                        ):
                            problems.append(
                                f"{diagnostics_path}: candidate utility "
                                "partition is not fully disjoint"
                            )
                            break
                        for step in (
                            diagnostic.get("inner_m_step") or {}
                        ).get("steps", []):
                            utility = step.get("candidate_utility") or {}
                            if utility.get("enabled") is not True:
                                problems.append(
                                    f"{diagnostics_path}: inner step omitted "
                                    "candidate utility diagnostics"
                                )
                                break
                            if not utility.get("evaluated"):
                                continue
                            evaluated_candidates += 1
                            decode = utility.get("free_decode") or {}
                            answer = utility.get("gold_answer") or {}
                            alignment = utility.get("alignment") or {}
                            if (
                                decode.get("question_count")
                                != utility_questions
                                or len(decode.get("paired_outcomes") or [])
                                != utility_questions
                            ):
                                problems.append(
                                    f"{diagnostics_path}: incomplete paired "
                                    "candidate utility outcomes"
                                )
                                break
                            finite_values = (
                                decode.get("accuracy_before"),
                                decode.get("accuracy_after"),
                                decode.get("accuracy_delta"),
                                answer.get("loss_before"),
                                answer.get("loss_after"),
                                answer.get("loss_delta"),
                                alignment.get(
                                    "candidate_parameter_delta_cosine"
                                ),
                            )
                            if not all(
                                isinstance(value, (int, float))
                                and math.isfinite(float(value))
                                for value in finite_values
                            ):
                                problems.append(
                                    f"{diagnostics_path}: non-finite candidate "
                                    "utility measurement"
                                )
                                break
                    if not evaluated_candidates:
                        problems.append(
                            f"{diagnostics_path}: no terminal candidate "
                            "utility measurements were evaluated"
                        )
            except Exception as exc:
                problems.append(f"invalid training diagnostics {diagnostics_path}: {exc}")

        eval_rounds = defaults.get("eval_rounds") or []
        eval_every = int(defaults.get("eval_every", 0))
        if eval_rounds or eval_every:
            if eval_rounds:
                expected_rounds = sorted({
                    *(int(value) for value in eval_rounds if int(value) <= rounds),
                    rounds,
                })
            else:
                expected_rounds = list(range(eval_every, rounds + 1, eval_every))
                if not expected_rounds or expected_rounds[-1] != rounds:
                    expected_rounds.append(rounds)
            checkpoint_eval_path = (
                out
                / f"checkpoint_eval_{task}__{tag}__{method}_s{seed}.jsonl.gz"
            )
            if not checkpoint_eval_path.is_file():
                problems.append(
                    f"missing checkpoint evaluations: {checkpoint_eval_path}"
                )
            else:
                try:
                    checkpoint_evals = _read_jsonl_gz(checkpoint_eval_path)
                    observed_rounds = [
                        int(row.get("completed_rounds", -1))
                        for row in checkpoint_evals
                    ]
                    if observed_rounds != expected_rounds:
                        problems.append(
                            f"{checkpoint_eval_path}: expected completed rounds "
                            f"{expected_rounds}, found {observed_rounds}"
                        )
                    for row in checkpoint_evals:
                        records = row.get("records") or []
                        if len(records) != int(
                            defaults.get("n_test", 200)
                        ):
                            problems.append(
                                f"{checkpoint_eval_path}: incomplete per-question records "
                                f"at completed round {row.get('completed_rounds')}"
                            )
                        elif eos_target or require_protocol_outcomes:
                            _validate_eos_generation_records(
                                problems,
                                path=checkpoint_eval_path,
                                records=records,
                                require_outcome_diagnostics=(
                                    require_protocol_outcomes
                                ),
                            )
                        if (
                            require_official_test_unaccessed
                            and len(records) == int(defaults.get("n_test", 200))
                        ):
                            _validate_official_test_access_records(
                                problems,
                                path=checkpoint_eval_path,
                                records=records,
                            )
                except Exception as exc:
                    problems.append(
                        f"invalid checkpoint evaluations {checkpoint_eval_path}: {exc}"
                    )

    if int(defaults.get("passk", 0)) > 0:
        passk_path = out / f"passk_{task}__{tag}__{method}_s{seed}.json"
        if not passk_path.is_file():
            problems.append(f"missing pass@K evaluation: {passk_path}")
        else:
            try:
                payload = _read_json(passk_path)
                if len(payload.get("records") or []) != int(
                    defaults.get("passk_n", 100)
                ):
                    problems.append(f"{passk_path}: incomplete pass@K records")
                elif require_official_test_unaccessed:
                    if payload.get("passk_official_test_accessed") is not False:
                        problems.append(
                            f"{passk_path}: does not prove official-test non-access"
                        )
                    if payload.get("passk_eval_source_split") != "train":
                        problems.append(
                            f"{passk_path}: passk_eval_source_split is not 'train'"
                        )
                    if payload.get("passk_dataset_splits_loaded") != ["train"]:
                        problems.append(
                            f"{passk_path}: loaded dataset splits are not ['train']"
                        )
                    _validate_official_test_access_records(
                        problems,
                        path=passk_path,
                        records=payload.get("records") or [],
                    )
            except Exception as exc:
                problems.append(f"invalid pass@K evaluation {passk_path}: {exc}")

    if defaults.get("save_adapter") and method != "base":
        adapter_path = out / f"adapter_{task}__{tag}__{method}_s{seed}"
        weights = (
            adapter_path / "adapter_model.safetensors",
            adapter_path / "adapter_model.bin",
        )
        if not any(path.is_file() for path in weights):
            problems.append(f"missing final adapter: {adapter_path}")


def _structured_marker_payload(
    *,
    config_path: Path,
    expected_run_id: str,
    expected_commit: str,
    expected_config_sha256: str,
    source_job_id: str,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    observed_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if observed_sha256 != expected_config_sha256:
        raise ValueError(
            "configuration hash mismatch: "
            f"expected {expected_config_sha256}, found {observed_sha256}"
        )
    config = yaml.safe_load(config_bytes)
    observed_run_id = str(config.get("run_id"))
    if observed_run_id != expected_run_id:
        raise ValueError(
            f"run ID mismatch: expected {expected_run_id}, found {observed_run_id}"
        )
    if not re.fullmatch(r"[0-9a-f]{7,40}", expected_commit):
        raise ValueError(f"invalid execution commit {expected_commit!r}")
    if not source_job_id.isdigit():
        raise ValueError(f"invalid source job ID {source_job_id!r}")
    return {
        "schema_version": 1,
        "status": "ok",
        "run_id": expected_run_id,
        "execution_commit": expected_commit,
        "configuration_sha256": observed_sha256,
        "source_job_id": source_job_id,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--log-glob")
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--marker-run-id")
    parser.add_argument("--marker-commit")
    parser.add_argument("--marker-config-sha256")
    parser.add_argument("--marker-source-job-id")
    args = parser.parse_args()

    problems = validate(args.config, args.log_glob)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        raise SystemExit(1)
    if args.marker:
        args.marker.expanduser().parent.mkdir(parents=True, exist_ok=True)
        marker_fields = (
            args.marker_run_id,
            args.marker_commit,
            args.marker_config_sha256,
            args.marker_source_job_id,
        )
        if any(marker_fields) and not all(marker_fields):
            parser.error(
                "structured marker requires run ID, commit, config hash, "
                "and source job ID"
            )
        if all(marker_fields):
            payload = _structured_marker_payload(
                config_path=args.config,
                expected_run_id=args.marker_run_id,
                expected_commit=args.marker_commit,
                expected_config_sha256=args.marker_config_sha256,
                source_job_id=args.marker_source_job_id,
            )
            atomic_write_json(args.marker.expanduser(), payload)
        else:
            args.marker.expanduser().write_text("ok\n", encoding="utf-8")
    print(f"validated {args.config}")


if __name__ == "__main__":
    main()
