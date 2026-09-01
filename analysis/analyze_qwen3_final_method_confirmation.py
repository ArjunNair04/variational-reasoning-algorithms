#!/usr/bin/env python3
"""Analyse the frozen ten-method, seven-seed Qwen3 confirmation.

The script is intentionally outcome-independent. It verifies provenance,
official-test non-access, prompt parity, paired question support, complete
checkpoint trajectories and receipt-bound adapters before reporting final
extracted Acc@1, strict final accuracy and trajectory AUC in that order.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LM_STUDY = REPOSITORY_ROOT / "lm_study"
if str(LM_STUDY) not in sys.path:
    sys.path.insert(0, str(LM_STUDY))

from generate_qwen3_17b_final_method_confirmation import (  # noqa: E402
    CELL_ORDER,
    CHECKPOINTS,
    RUN_ID,
    SEEDS,
    build_payload,
)
from result_contract import (  # noqa: E402
    ResultContractError,
    validate_completion_receipt,
    validate_receipt_identity,
)
from run_yaml import _prepare_cells  # noqa: E402


SOURCE_REGISTRY_ID = "qwen3_final_method_confirmation_20260818"
METRIC_ORDER = (
    "final_extracted_answer_accuracy",
    "final_strict_terminal_accuracy",
    "normalized_extracted_trajectory_auc",
)
BOOTSTRAP_METRICS = (
    "final_extracted",
    "final_strict",
    "extracted_auc",
    "strict_auc",
)
Q5_CELL = "Q5-LR1e-5-U1-K16"
ANSWER_DERIVED_CELLS = {Q5_CELL}
PIS_CELL = "PIS-S8-B8-U4"
BASE_CELL = "CTRL-base"
GOLD_CELL = "GOLD-LR3e-6-E2"
RFT_CELL = "RFT-LR1e-5-E2"
REST_CELL = "REST-LR1e-5-E1-I4"
STAR_CELL = "STAR-LR3e-6-E2"
TRICE_CELL = "TRICE-LR1e-4-CV"
GRPO_CELL = "GRPO-S16-B4-U4"
RLOO_CELL = "RLOO-S16-B8-U4"
ROLE_CANDIDATES = {
    "posterior": (Q5_CELL, PIS_CELL),
    "non_rl_self_training": (RFT_CELL, REST_CELL, STAR_CELL, TRICE_CELL),
    "rl": (GRPO_CELL, RLOO_CELL),
}
DIAGNOSTIC_METHOD_NAMES = {
    "AC-ALG1": "AC-ALG1",
    "Gold-CoT-SFT": "Gold-CoT-SFT",
    "RFT-Source": "rft_source",
    "ReST-EM": "rest_em",
    "STaR": "star",
    "TRICE": "TRICE",
    "GRPO": "GRPO",
    "RLOO": "RLOO",
}
RESOURCE_FIELDS = (
    "train_llm_gen",
    "eval_llm_gen",
    "llm_gen",
    "generated_tokens",
    "backward_tokens",
    "optimizer_steps",
    "model_forward_calls",
    "model_forward_input_tokens",
    "accelerator_hours",
    "peak_cuda_reserved_gb",
    "pass8",
)
OPTIONAL_RESOURCE_FIELDS = ("teacher_forced_scoring_tokens",)
EXPECTED_LORA_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    method: str
    axes: dict[str, Any]
    base_tag: str


@dataclass(frozen=True)
class Coordinate:
    cell: CellSpec
    seed: int

    @property
    def tag(self) -> str:
        return f"{self.cell.base_tag}_seed{self.seed}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _validate_lora_surface(
    *, cell_id: str, sweep: Mapping[str, Any], context: Path
) -> None:
    # The frozen control never constructs or trains an adapter, so its inert
    # parser default cannot define the adaptation surface of the trained cells.
    if cell_id == BASE_CELL:
        return
    if sweep.get("lora_target_set") != "attention_mlp":
        raise ValueError(f"{context}: LoRA surface changed")
    modules = (sweep.get("lora_target_modules") or {}).get("qwen3-1.7b-base")
    if modules != EXPECTED_LORA_MODULES:
        raise ValueError(f"{context}: LoRA module set changed")


def _read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a compressed JSON object")
    return payload


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _finite(value: Any, *, context: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{context}: expected a finite number")
    return numeric


def _optional_finite(value: Any, *, context: str) -> float:
    if value is None:
        return float("nan")
    return _finite(value, context=context)


def _resource_totals(
    method: str,
    result: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
) -> tuple[float, float, float]:
    """Resolve exact generated, backward and scored token totals."""

    if method == "base":
        return 0.0, 0.0, 0.0
    generated = result.get("generated_tokens")
    backward = result.get("backward_tokens")
    scored = result.get("teacher_forced_scoring_tokens")
    if generated is not None and backward is not None:
        return (
            _finite(generated, context=f"{method} generated tokens"),
            _finite(backward, context=f"{method} backward tokens"),
            _optional_finite(scored, context=f"{method} scored tokens"),
        )
    if method == "AC-ALG1":
        generated_total = 0.0
        backward_total = 0.0
        scored_total = 0.0
        for index, diagnostic in enumerate(diagnostics):
            tokens = dict((diagnostic.get("compute") or {}).get("tokens") or {})
            generated_total += _finite(
                tokens.get("generated"), context=f"{method} diagnostic {index}/generated"
            )
            backward_total += _finite(
                tokens.get("backward"), context=f"{method} diagnostic {index}/backward"
            )
            scored_total += _finite(
                tokens.get("scored"), context=f"{method} diagnostic {index}/scored"
            )
        return generated_total, backward_total, scored_total
    if not diagnostics:
        raise ValueError(f"{method}: missing resource diagnostics")
    final = diagnostics[-1]
    generation = dict(final.get("generation") or {})
    optimizer = dict(final.get("optimizer") or {})
    generated = (
        0.0
        if method == "Gold-CoT-SFT"
        else _finite(
            generation.get("generated_tokens_cumulative"),
            context=f"{method} generated tokens",
        )
    )
    backward = _finite(
        optimizer.get("backward_tokens_cumulative"),
        context=f"{method} backward tokens",
    )
    return generated, backward, _optional_finite(
        scored, context=f"{method} scored tokens"
    )


def _require_resource_contract(
    coordinate: Coordinate,
    result: Mapping[str, Any],
) -> None:
    """Reject runs whose realised proposal or update schedule changed."""

    method = coordinate.cell.method
    axes = coordinate.cell.axes
    train_generations = int(
        _finite(result.get("train_llm_gen"), context=f"{method} train generations")
    )
    if method == "base" or method == "Gold-CoT-SFT":
        expected_generations = 0
    elif method == "TRICE":
        expected_generations = 32 * int(axes["batch"]) + 128
    elif method == "STaR":
        expected_generations = None
        if not 512 <= train_generations <= 1024:
            raise ValueError("STaR direct-plus-rationalisation count is out of range")
    else:
        expected_generations = 32 * int(axes.get("batch", 64))
    if expected_generations is not None and train_generations != expected_generations:
        raise ValueError(
            f"{coordinate.cell.cell_id}/{coordinate.seed}: expected "
            f"{expected_generations} training generations, found {train_generations}"
        )
    if method == "base":
        if result.get("optimizer_steps") not in (None, 0):
            raise ValueError("frozen base unexpectedly reports optimiser steps")
    elif method == "AC-ALG1":
        expected_steps = 32 * int(axes["iters"])
        if int(result.get("optimizer_steps", -1)) != expected_steps:
            raise ValueError(
                f"{coordinate.cell.cell_id}/{coordinate.seed}: optimiser-step schedule changed"
            )
    elif method == "Gold-CoT-SFT":
        expected_steps = 32 * int(axes["epochs"])
        if int(result.get("optimizer_steps", -1)) != expected_steps:
            raise ValueError(
                f"{coordinate.cell.cell_id}/{coordinate.seed}: optimiser-step schedule changed"
            )


def _raw_axes_by_cell(config: Mapping[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    axes = {}
    for method, raw in (config.get("algos") or {}).items():
        variants = raw if isinstance(raw, list) else [raw]
        for variant in variants:
            cell = dict(variant)
            cell_id = str(cell.pop("cell_id"))
            if cell_id in axes:
                raise ValueError(f"duplicate configured cell {cell_id}")
            axes[cell_id] = (str(method), cell)
    return axes


def load_design(config_path: Path) -> tuple[dict[str, Any], list[CellSpec]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config != build_payload():
        raise ValueError("configuration differs from the frozen generated payload")
    defaults = config["defaults"]
    prepared = _prepare_cells(
        config,
        only=None,
        run_id=str(config["run_id"]),
        defaults=defaults,
    )
    raw = _raw_axes_by_cell(config)
    if len(prepared) != len(CELL_ORDER):
        raise ValueError(
            f"confirmation must contain exactly {len(CELL_ORDER)} cells"
        )
    cells = []
    for index, (model, method, _merged_axes, base_tag) in enumerate(prepared):
        cell_id = CELL_ORDER[index]
        raw_method, axes = raw[cell_id]
        if str(model) != "qwen3-1.7b-base" or str(method) != raw_method:
            raise ValueError(f"{cell_id}: prepared method/model mismatch")
        cells.append(CellSpec(cell_id, raw_method, axes, str(base_tag)))
    return config, cells


def verify_marker(
    marker_path: Path,
    config_path: Path,
    *,
    expected_commit: str,
    expected_source_job: str,
) -> dict[str, Any]:
    marker = _read_json(marker_path)
    expected = {
        "schema_version": 1,
        "status": "ok",
        "run_id": RUN_ID,
        "execution_commit": expected_commit,
        "configuration_sha256": _sha256(config_path),
        "source_job_id": expected_source_job,
        "task_count": len(CELL_ORDER) * len(SEEDS),
        "trained_adapter_count": (len(CELL_ORDER) - 1) * len(SEEDS),
        "official_test_used": False,
    }
    mismatches = {
        key: (marker.get(key), value)
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise ValueError(f"validator marker mismatch: {mismatches}")
    return marker


def _expected_coordinates(cells: Sequence[CellSpec]) -> list[Coordinate]:
    return [Coordinate(cell, seed) for cell in cells for seed in SEEDS]


def _artifact(
    artifact_dir: Path,
    receipt: Mapping[str, Any],
    *,
    prefix: str,
    suffix: str | None = None,
) -> Path:
    matches = []
    for record in receipt.get("artifacts") or []:
        relative = str(record.get("path") or "")
        name = Path(relative).name
        if name.startswith(prefix) and (suffix is None or name.endswith(suffix)):
            matches.append(artifact_dir / relative)
    if len(matches) != 1:
        raise ValueError(
            f"receipt expected one {prefix!r} artifact, found {len(matches)}"
        )
    return matches[0]


def _require_non_access(payload: Mapping[str, Any], *, context: str) -> None:
    expected = {
        "eval_official_test_accessed": False,
        "eval_source_split": "train",
        "eval_dataset_splits_loaded": ["train"],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{context}: {field}={payload.get(field)!r}")


def _require_record_non_access(record: Mapping[str, Any], *, context: str) -> None:
    expected = {
        "official_test_accessed": False,
        "eval_source_split": "train",
        "dataset_splits_loaded": ["train"],
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"{context}: {field}={record.get(field)!r}")


def _validate_prompt_contract(
    payload: Mapping[str, Any],
    coordinate: Coordinate,
) -> dict[int, str]:
    if payload.get("schema_version") != 1:
        raise ValueError(f"{coordinate.cell.cell_id}/{coordinate.seed}: prompt schema changed")
    if payload.get("tag") != coordinate.tag:
        raise ValueError(f"{coordinate.cell.cell_id}/{coordinate.seed}: prompt tag changed")
    if int(payload.get("training_seed", -1)) != coordinate.seed:
        raise ValueError(f"{coordinate.cell.cell_id}/{coordinate.seed}: prompt seed changed")
    expected_mode = (
        "answer_derive"
        if coordinate.cell.cell_id in ANSWER_DERIVED_CELLS
        else "question"
    )
    if payload.get("proposal_prompt_mode") != expected_mode:
        raise ValueError(f"{coordinate.cell.cell_id}/{coordinate.seed}: proposal mode changed")
    if payload.get("answer_event_mode") != "strict_terminal_marker":
        raise ValueError(f"{coordinate.cell.cell_id}/{coordinate.seed}: answer event changed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 128:
        raise ValueError(f"{coordinate.cell.cell_id}/{coordinate.seed}: expected 128 prompt rows")
    canonical = {}
    for row in rows:
        dataset_index = int(row.get("dataset_train_index", -1))
        prompt_hash = str(row.get("canonical_prompt_sha256") or "")
        if dataset_index < 0 or len(prompt_hash) != 64 or dataset_index in canonical:
            raise ValueError(f"{coordinate.cell.cell_id}/{coordinate.seed}: invalid prompt row")
        if row.get("canonical_contains_proposal_answer_hint") is not False:
            raise ValueError(f"{coordinate.cell.cell_id}/{coordinate.seed}: canonical answer leak")
        expected_hint = expected_mode == "answer_derive"
        if row.get("proposal_contains_gold_answer_hint") is not expected_hint:
            raise ValueError(f"{coordinate.cell.cell_id}/{coordinate.seed}: proposal hint mismatch")
        canonical[dataset_index] = prompt_hash
    return canonical


def _records_by_question(
    records: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> tuple[tuple[int, ...], dict[int, Mapping[str, Any]]]:
    if len(records) != 400:
        raise ValueError(f"{context}: expected 400 evaluation records")
    order = tuple(int(record.get("idx", -1)) for record in records)
    if len(set(order)) != 400 or any(index < 0 for index in order):
        raise ValueError(f"{context}: evaluation question identities changed")
    mapped = {}
    for position, record in enumerate(records):
        _require_record_non_access(record, context=f"{context}/record{position}")
        for field in (
            "legacy_correct",
            "strict_correct",
            "generated_eos",
            "strict_format_failure",
            "hit_max_new_tokens",
        ):
            if not isinstance(record.get(field), bool):
                raise ValueError(f"{context}/record{position}: {field} is not boolean")
        mapped[int(record["idx"])] = record
    return order, mapped


def _mean_bool(records: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([bool(record[field]) for record in records]))


def _normalized_question_auc(
    values: np.ndarray,
    rounds: Sequence[int],
) -> np.ndarray:
    rounds_array = np.asarray(rounds, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(rounds_array):
        raise ValueError("question AUC received an invalid trajectory matrix")
    if tuple(int(value) for value in rounds_array) != (0, *CHECKPOINTS):
        raise ValueError("question AUC checkpoint schedule changed")
    # NumPy 2 removed trapz, while Beaker's older NumPy predates trapezoid.
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return integrate(values, rounds_array, axis=1) / float(CHECKPOINTS[-1])


def _check_close(observed: Any, expected: float, *, context: str) -> None:
    if not math.isclose(_finite(observed, context=context), expected, abs_tol=1e-12):
        raise ValueError(f"{context}: aggregate does not match question records")


def load_results(
    artifact_dir: Path,
    config: Mapping[str, Any],
    cells: Sequence[CellSpec],
    *,
    expected_commit: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    coordinates = _expected_coordinates(cells)
    receipt_paths = sorted(artifact_dir.glob("complete_*.json"))
    if len(receipt_paths) != len(coordinates):
        raise ValueError(
            f"expected {len(coordinates)} completion receipts, found {len(receipt_paths)}"
        )
    question_rows = []
    seed_rows = []
    resource_rows = []
    receipt_details: dict[tuple[str, int], dict[str, Any]] = {}
    base_records: dict[int, dict[int, Mapping[str, Any]]] = {}
    base_order: dict[int, tuple[int, ...]] = {}
    canonical_prompts: dict[int, dict[int, str]] = {}
    train_questions: dict[int, tuple[int, ...]] = {}

    for coordinate in coordinates:
        method = coordinate.cell.method
        seed = coordinate.seed
        tag = coordinate.tag
        receipt_path = artifact_dir / f"complete_gsm8k__{tag}__{method}_s{seed}.json"
        try:
            receipt = validate_completion_receipt(
                receipt_path, result_root=artifact_dir, verify_hashes=True
            )
            validate_receipt_identity(
                receipt,
                {
                    "run_id": RUN_ID,
                    "task": "gsm8k",
                    "model": "qwen3-1.7b-base",
                    "method": method,
                    "seed": seed,
                    "tag": tag,
                },
            )
        except (ResultContractError, OSError) as exc:
            raise ValueError(f"{receipt_path}: invalid receipt: {exc}") from exc
        cell_result_path = _artifact(
            artifact_dir, receipt, prefix="cell_result_", suffix=".json"
        )
        cell_result = _read_json(cell_result_path)
        result = dict(cell_result.get("result") or {})
        identity = dict(cell_result.get("identity") or {})
        if identity.get("tag") != tag or str(result.get("run_id")) != RUN_ID:
            raise ValueError(f"{cell_result_path}: identity mismatch")
        _require_non_access(result, context=cell_result_path.name)
        params = json.loads(str(result.get("params") or ""))
        commit = str((params.get("env") or {}).get("commit") or "")
        if len(commit) < 7 or not expected_commit.startswith(commit):
            raise ValueError(f"{cell_result_path}: execution commit mismatch")
        sweep = dict(params.get("sweep") or {})
        _validate_lora_surface(
            cell_id=coordinate.cell.cell_id,
            sweep=sweep,
            context=cell_result_path,
        )
        if int(result.get("accelerator_count", -1)) != 1 or "H100" not in str(
            result.get("accelerator_name")
        ):
            raise ValueError(f"{cell_result_path}: confirmation did not run on one H100")

        eval_path = _artifact(artifact_dir, receipt, prefix="eval_", suffix=".json")
        evaluation = _read_json(eval_path)
        _require_non_access(evaluation, context=eval_path.name)
        order, final_by_id = _records_by_question(
            evaluation.get("records") or [], context=eval_path.name
        )
        if tuple(int(value) for value in evaluation.get("train_qi") or ()) != tuple(
            int(value) for value in result.get("train_qi") or ()
        ):
            raise ValueError(f"{eval_path}: train-question metadata mismatch")
        observed_train = tuple(int(value) for value in evaluation.get("train_qi") or ())
        if len(observed_train) != 128 or len(set(observed_train)) != 128:
            raise ValueError(f"{eval_path}: expected 128 unique optimization questions")
        if seed in train_questions and train_questions[seed] != observed_train:
            raise ValueError(f"{eval_path}: paired optimization schedule changed")
        train_questions.setdefault(seed, observed_train)

        prompt_path = _artifact(
            artifact_dir, receipt, prefix="prompt_contract_", suffix=".json.gz"
        )
        prompt_hashes = _validate_prompt_contract(
            _read_json_gz(prompt_path), coordinate
        )
        if seed in canonical_prompts and canonical_prompts[seed] != prompt_hashes:
            raise ValueError(f"{prompt_path}: canonical prompt parity changed")
        canonical_prompts.setdefault(seed, prompt_hashes)

        if coordinate.cell.cell_id == BASE_CELL:
            base_order[seed] = order
            base_records[seed] = final_by_id
            checkpoint_rows: list[dict[str, Any]] = []
            diagnostics: list[dict[str, Any]] = []
        else:
            checkpoint_path = _artifact(
                artifact_dir,
                receipt,
                prefix="checkpoint_eval_",
                suffix=".jsonl.gz",
            )
            checkpoint_rows = _read_jsonl_gz(checkpoint_path)
            observed_rounds = tuple(
                int(row.get("completed_rounds", -1)) for row in checkpoint_rows
            )
            if observed_rounds != CHECKPOINTS:
                raise ValueError(f"{checkpoint_path}: checkpoint schedule changed")
            for row_index, checkpoint in enumerate(checkpoint_rows):
                metrics = dict(checkpoint.get("metrics") or {})
                _require_non_access(metrics, context=f"{checkpoint_path.name}/{row_index}")
                checkpoint_order, checkpoint_records = _records_by_question(
                    checkpoint.get("records") or [],
                    context=f"{checkpoint_path.name}/{row_index}",
                )
                if checkpoint_order != order:
                    raise ValueError(f"{checkpoint_path}: checkpoint question order changed")
                checkpoint_values = list(checkpoint_records.values())
                _check_close(
                    metrics.get("test_acc_legacy"),
                    _mean_bool(checkpoint_values, "legacy_correct"),
                    context=f"{checkpoint_path.name}/test_acc_legacy",
                )
                _check_close(
                    metrics.get("test_acc_strict"),
                    _mean_bool(checkpoint_values, "strict_correct"),
                    context=f"{checkpoint_path.name}/test_acc_strict",
                )
            diagnostics_path = _artifact(
                artifact_dir,
                receipt,
                prefix="training_diagnostics_",
                suffix=".jsonl.gz",
            )
            diagnostics = _read_jsonl_gz(diagnostics_path)
            if len(diagnostics) != 32:
                raise ValueError(f"{diagnostics_path}: expected 32 diagnostic rows")
            expected_diagnostic_method = DIAGNOSTIC_METHOD_NAMES[method]
            for round_index, diagnostic in enumerate(diagnostics):
                expected_diagnostic = {
                    "run_id": RUN_ID,
                    "model": "qwen3-1.7b-base",
                    "task": "gsm8k",
                    "method": expected_diagnostic_method,
                    "seed": seed,
                    "tag": tag,
                    "round": round_index,
                }
                mismatches = {
                    field: (diagnostic.get(field), value)
                    for field, value in expected_diagnostic.items()
                    if diagnostic.get(field) != value
                }
                if mismatches:
                    raise ValueError(
                        f"{diagnostics_path}: diagnostic identity mismatch {mismatches}"
                    )
        _check_close(
            evaluation.get("test_acc_legacy"),
            _mean_bool(list(final_by_id.values()), "legacy_correct"),
            context=f"{eval_path.name}/test_acc_legacy",
        )
        _check_close(
            evaluation.get("test_acc_strict"),
            _mean_bool(list(final_by_id.values()), "strict_correct"),
            context=f"{eval_path.name}/test_acc_strict",
        )

        adapter_records = [
            str(record.get("path") or "")
            for record in receipt.get("artifacts") or []
            if str(record.get("path") or "").startswith("adapter_")
        ]
        if method == "base":
            if adapter_records:
                raise ValueError(f"{receipt_path}: frozen base has an adapter")
            adapter_dir = None
        else:
            adapter_dirs = {str(Path(path).parent) for path in adapter_records}
            if len(adapter_dirs) != 1:
                raise ValueError(f"{receipt_path}: expected one receipt-bound adapter")
            adapter_dir = next(iter(adapter_dirs))

        receipt_details[(coordinate.cell.cell_id, seed)] = {
            "receipt_name": receipt_path.name,
            "receipt_sha256": _sha256(receipt_path),
            "adapter_dir": adapter_dir,
        }
        _require_resource_contract(coordinate, result)
        generated_tokens, backward_tokens, scored_tokens = _resource_totals(
            method, result, diagnostics
        )
        resource = {
            "cell": coordinate.cell.cell_id,
            "seed": seed,
            "train_llm_gen": _finite(
                result.get("train_llm_gen"), context=f"{cell_result_path}/train_llm_gen"
            ),
            "eval_llm_gen": _finite(
                result.get("eval_llm_gen"), context=f"{cell_result_path}/eval_llm_gen"
            ),
            "llm_gen": _finite(
                result.get("llm_gen"), context=f"{cell_result_path}/llm_gen"
            ),
            "generated_tokens": generated_tokens,
            "backward_tokens": backward_tokens,
            "teacher_forced_scoring_tokens": scored_tokens,
            "optimizer_steps": (
                0.0
                if method == "base"
                else _finite(
                    result.get("optimizer_steps"),
                    context=f"{cell_result_path}/optimizer_steps",
                )
            ),
        }
        for field in (
            "model_forward_calls",
            "model_forward_input_tokens",
            "accelerator_hours",
            "peak_cuda_reserved_gb",
            "pass8",
        ):
            resource[field] = _finite(
                result.get(field), context=f"{cell_result_path}/{field}"
            )
        resource_rows.append(resource)

        # Question trajectories are completed after every base coordinate has
        # been read, because base is the registered round-zero reference.
        seed_rows.append(
            {
                "cell": coordinate.cell.cell_id,
                "seed": seed,
                "tag": tag,
                "final_records": final_by_id,
                "checkpoint_rows": checkpoint_rows,
                "question_order": order,
            }
        )

    if set(base_records) != set(SEEDS):
        raise ValueError("paired frozen-base records are incomplete")

    summarized_seed_rows = []
    for row in seed_rows:
        cell = str(row["cell"])
        seed = int(row["seed"])
        order = tuple(row["question_order"])
        if order != base_order[seed]:
            raise ValueError(f"{cell}/{seed}: paired validation questions changed")
        final_by_id = row["final_records"]
        base_by_id = base_records[seed]
        rounds = (0, *CHECKPOINTS)
        extracted = np.empty((400, len(rounds)), dtype=float)
        strict = np.empty((400, len(rounds)), dtype=float)
        extracted[:, 0] = [bool(base_by_id[index]["legacy_correct"]) for index in order]
        strict[:, 0] = [bool(base_by_id[index]["strict_correct"]) for index in order]
        if cell == BASE_CELL:
            extracted[:, 1:] = extracted[:, :1]
            strict[:, 1:] = strict[:, :1]
        else:
            for checkpoint_position, checkpoint in enumerate(row["checkpoint_rows"], start=1):
                checkpoint_map = {
                    int(record["idx"]): record for record in checkpoint.get("records") or []
                }
                extracted[:, checkpoint_position] = [
                    bool(checkpoint_map[index]["legacy_correct"]) for index in order
                ]
                strict[:, checkpoint_position] = [
                    bool(checkpoint_map[index]["strict_correct"]) for index in order
                ]
            final_extracted = np.asarray(
                [bool(final_by_id[index]["legacy_correct"]) for index in order],
                dtype=float,
            )
            final_strict = np.asarray(
                [bool(final_by_id[index]["strict_correct"]) for index in order],
                dtype=float,
            )
            if not np.array_equal(extracted[:, -1], final_extracted):
                raise ValueError(f"{cell}/{seed}: terminal extracted records disagree")
            if not np.array_equal(strict[:, -1], final_strict):
                raise ValueError(f"{cell}/{seed}: terminal strict records disagree")
        extracted_auc = _normalized_question_auc(extracted, rounds)
        strict_auc = _normalized_question_auc(strict, rounds)
        for position, question_id in enumerate(order):
            final_record = final_by_id[question_id]
            question_rows.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "question_id": question_id,
                    "final_extracted": float(extracted[position, -1]),
                    "final_strict": float(strict[position, -1]),
                    "extracted_auc": float(extracted_auc[position]),
                    "strict_auc": float(strict_auc[position]),
                    "natural_eos": float(bool(final_record["generated_eos"])),
                    "strict_format_failure": float(
                        bool(final_record["strict_format_failure"])
                    ),
                    "hit_token_limit": float(bool(final_record["hit_max_new_tokens"])),
                }
            )
        summarized_seed_rows.append(
            {
                "cell": cell,
                "seed": seed,
                "final_extracted": float(extracted[:, -1].mean()),
                "final_strict": float(strict[:, -1].mean()),
                "extracted_auc": float(extracted_auc.mean()),
                "strict_auc": float(strict_auc.mean()),
                "natural_eos": float(
                    np.mean([bool(final_by_id[index]["generated_eos"]) for index in order])
                ),
                "strict_format_failure": float(
                    np.mean(
                        [
                            bool(final_by_id[index]["strict_format_failure"])
                            for index in order
                        ]
                    )
                ),
                "hit_token_limit": float(
                    np.mean(
                        [bool(final_by_id[index]["hit_max_new_tokens"]) for index in order]
                    )
                ),
            }
        )

    questions = pd.DataFrame(question_rows)
    seeds = pd.DataFrame(summarized_seed_rows)
    resources = pd.DataFrame(resource_rows)
    expected = {(cell, seed) for cell in CELL_ORDER for seed in SEEDS}
    if set(zip(seeds["cell"], seeds["seed"], strict=True)) != expected:
        raise ValueError("seed-level coordinate coverage is incomplete")
    if len(questions) != len(coordinates) * 400:
        raise ValueError("question-level coordinate coverage is incomplete")
    return questions, seeds, resources, receipt_details


def question_metric_cube(
    question_metrics: pd.DataFrame,
) -> tuple[np.ndarray, tuple[int, ...]]:
    question_ids_by_seed = []
    cube = np.empty(
        (len(CELL_ORDER), len(SEEDS), 400, len(BOOTSTRAP_METRICS)), dtype=float
    )
    for seed_index, seed in enumerate(SEEDS):
        base = question_metrics[
            (question_metrics["cell"] == BASE_CELL)
            & (question_metrics["seed"] == seed)
        ].sort_values("question_id")
        ids = tuple(int(value) for value in base["question_id"])
        if len(ids) != 400:
            raise ValueError(f"seed {seed}: incomplete question support")
        question_ids_by_seed.append(ids)
        for cell_index, cell in enumerate(CELL_ORDER):
            frame = question_metrics[
                (question_metrics["cell"] == cell)
                & (question_metrics["seed"] == seed)
            ].sort_values("question_id")
            if tuple(int(value) for value in frame["question_id"]) != ids:
                raise ValueError(f"{cell}/{seed}: unpaired question support")
            cube[cell_index, seed_index] = frame[list(BOOTSTRAP_METRICS)].to_numpy(float)
    if len(set(question_ids_by_seed)) != 1:
        raise ValueError("validation question identities changed across seeds")
    return cube, question_ids_by_seed[0]


def hierarchical_method_bootstrap(
    cube: np.ndarray,
    *,
    draws: int,
    seed: int,
    chunk_size: int = 250,
) -> np.ndarray:
    """Nested paired bootstrap over training seeds and matched questions."""

    if cube.shape != (len(CELL_ORDER), len(SEEDS), 400, len(BOOTSTRAP_METRICS)):
        raise ValueError(f"unexpected metric cube shape {cube.shape}")
    if draws < 1 or chunk_size < 1:
        raise ValueError("bootstrap draws and chunk size must be positive")
    rng = np.random.default_rng(seed)
    output = np.empty((draws, len(CELL_ORDER), len(BOOTSTRAP_METRICS)), dtype=float)
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        count = stop - start
        sampled_seeds = rng.integers(0, len(SEEDS), size=(count, len(SEEDS)))
        sampled_questions = rng.integers(
            0, 400, size=(count, len(SEEDS), 400)
        )
        chunk = np.zeros((count, len(CELL_ORDER), len(BOOTSTRAP_METRICS)))
        for replicate_position in range(len(SEEDS)):
            selected = cube[:, sampled_seeds[:, replicate_position], :, :]
            question_indices = sampled_questions[:, replicate_position, :]
            gather = np.broadcast_to(
                question_indices[None, :, :, None],
                (len(CELL_ORDER), count, 400, len(BOOTSTRAP_METRICS)),
            )
            sampled = np.take_along_axis(selected, gather, axis=2)
            chunk += sampled.mean(axis=2).transpose(1, 0, 2)
        output[start:stop] = chunk / float(len(SEEDS))
    return output


def _exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    signs = np.array(
        [
            [1.0 if mask & (1 << bit) else -1.0 for bit in range(len(values))]
            for mask in range(1 << len(values))
        ]
    )
    null = np.abs((signs * values).mean(axis=1))
    return float(np.mean(null >= observed - 1e-15))


def contrast_table(
    seed_metrics: pd.DataFrame,
    cube: np.ndarray,
    bootstrap_means: np.ndarray,
) -> pd.DataFrame:
    pairs: list[tuple[str, str, str]] = []

    def add(family: str, treatment: str, control: str) -> None:
        item = (family, treatment, control)
        if item not in pairs:
            pairs.append(item)

    for cell in CELL_ORDER[1:]:
        add("all_methods_vs_base", cell, BASE_CELL)
    for posterior in ROLE_CANDIDATES["posterior"]:
        for comparator in (GRPO_CELL, RLOO_CELL):
            add("posterior_vs_online_rl", posterior, comparator)
        for comparator in (TRICE_CELL, STAR_CELL, REST_CELL):
            add("posterior_vs_non_rl", posterior, comparator)
    add("posterior_construction", Q5_CELL, PIS_CELL)
    generated = (Q5_CELL, PIS_CELL, REST_CELL, STAR_CELL, TRICE_CELL, GRPO_CELL, RLOO_CELL)
    for cell in generated:
        if cell != RFT_CELL:
            add("generated_vs_rft", cell, RFT_CELL)
        if cell != GOLD_CELL:
            add("generated_vs_gold", cell, GOLD_CELL)

    cell_index = {cell: index for index, cell in enumerate(CELL_ORDER)}
    metric_index = {metric: index for index, metric in enumerate(BOOTSTRAP_METRICS)}
    rows = []
    pivot = {
        metric: seed_metrics.pivot(index="seed", columns="cell", values=metric).reindex(SEEDS)
        for metric in BOOTSTRAP_METRICS
    }
    for family, treatment, control in pairs:
        for metric in BOOTSTRAP_METRICS:
            seed_differences = pivot[metric][treatment] - pivot[metric][control]
            draws = (
                bootstrap_means[:, cell_index[treatment], metric_index[metric]]
                - bootstrap_means[:, cell_index[control], metric_index[metric]]
            )
            point = float(cube[cell_index[treatment], :, :, metric_index[metric]].mean()) - float(
                cube[cell_index[control], :, :, metric_index[metric]].mean()
            )
            low, high = np.quantile(draws, [0.025, 0.975])
            if low > 0:
                status = "positive"
            elif high < 0:
                status = "negative"
            else:
                status = "unresolved"
            rows.append(
                {
                    "family": family,
                    "treatment": treatment,
                    "control": control,
                    "metric": metric,
                    "mean_difference": point,
                    "mean_difference_pp": 100.0 * point,
                    "hierarchical_ci95_low": float(low),
                    "hierarchical_ci95_high": float(high),
                    "hierarchical_ci95_low_pp": 100.0 * float(low),
                    "hierarchical_ci95_high_pp": 100.0 * float(high),
                    "interval_status": status,
                    "positive_seeds": int((seed_differences > 0).sum()),
                    "exact_sign_flip_p": _exact_sign_flip_p(seed_differences.to_numpy(float)),
                    "multiplicity_status": "descriptive_unadjusted_prespecified_contrast",
                }
            )
    return pd.DataFrame(rows)


def method_summary(
    seed_metrics: pd.DataFrame,
    resources: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for cell in CELL_ORDER:
        metrics = seed_metrics[seed_metrics["cell"] == cell]
        cost = resources[resources["cell"] == cell]
        row: dict[str, Any] = {
            "cell": cell,
            "seeds": len(metrics),
            "final_extracted": float(metrics["final_extracted"].mean()),
            "final_strict": float(metrics["final_strict"].mean()),
            "extracted_auc": float(metrics["extracted_auc"].mean()),
            "strict_auc": float(metrics["strict_auc"].mean()),
            "natural_eos": float(metrics["natural_eos"].mean()),
            "strict_format_failure": float(metrics["strict_format_failure"].mean()),
            "hit_token_limit": float(metrics["hit_token_limit"].mean()),
        }
        for field in (*RESOURCE_FIELDS, *OPTIONAL_RESOURCE_FIELDS):
            row[f"mean_{field}"] = float(cost[field].mean())
            row[f"sd_{field}"] = float(cost[field].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _rank(summary: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    frame = summary[summary["cell"].isin(candidates)].copy()
    frame["cell_order"] = frame["cell"].map({cell: index for index, cell in enumerate(CELL_ORDER)})
    frame = frame.sort_values(
        ["final_extracted", "final_strict", "extracted_auc", "cell_order"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    return [str(value) for value in frame["cell"]]


def _contrast_status(
    contrasts: pd.DataFrame,
    treatment: str,
    control: str,
    metric: str = "final_extracted",
) -> dict[str, Any]:
    rows = contrasts[
        (contrasts["treatment"] == treatment)
        & (contrasts["control"] == control)
        & (contrasts["metric"] == metric)
    ]
    if len(rows) != 1:
        # The reverse registered contrast carries the same inference with sign flipped.
        reverse = contrasts[
            (contrasts["treatment"] == control)
            & (contrasts["control"] == treatment)
            & (contrasts["metric"] == metric)
        ]
        if len(reverse) != 1:
            raise ValueError(f"missing contrast {treatment} minus {control}/{metric}")
        row = reverse.iloc[0]
        low = -float(row["hierarchical_ci95_high"])
        high = -float(row["hierarchical_ci95_low"])
        mean = -float(row["mean_difference"])
    else:
        row = rows.iloc[0]
        low = float(row["hierarchical_ci95_low"])
        high = float(row["hierarchical_ci95_high"])
        mean = float(row["mean_difference"])
    status = "positive" if low > 0 else "negative" if high < 0 else "unresolved"
    return {"mean_difference": mean, "ci95_low": low, "ci95_high": high, "status": status}


def build_generality_decision(
    config: Mapping[str, Any],
    config_path: Path,
    marker_path: Path,
    cells: Sequence[CellSpec],
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    receipt_details: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    rankings = {
        role: _rank(summary, candidates) for role, candidates in ROLE_CANDIDATES.items()
    }
    selected = {role: ranking[0] for role, ranking in rankings.items()}
    comparator_ranking = _rank(
        summary,
        (selected["non_rl_self_training"], selected["rl"]),
    )
    scale_comparator_role = (
        "non_rl_self_training"
        if comparator_ranking[0] == selected["non_rl_self_training"]
        else "rl"
    )
    raw_axes = _raw_axes_by_cell(config)
    cluster_root = str(config["defaults"]["out"]).rstrip("/")

    def finalist(role: str, cell_id: str) -> dict[str, Any]:
        method, axes = raw_axes[cell_id]
        if role == "frozen_base":
            return {
                "role": role,
                "algorithm": method,
                "cell_id": cell_id,
                "axes": {},
                "adapter_by_seed": None,
            }
        adapters = []
        for seed in SEEDS:
            details = receipt_details[(cell_id, seed)]
            adapter_dir = details.get("adapter_dir")
            if not adapter_dir:
                raise ValueError(f"{cell_id}/{seed}: selected cell has no adapter")
            adapters.append(
                {
                    "seed": seed,
                    "path": f"{cluster_root}/{adapter_dir}",
                    "receipt_path": f"{cluster_root}/{details['receipt_name']}",
                    "receipt_sha256": details["receipt_sha256"],
                }
            )
        return {
            "role": role,
            "algorithm": method,
            "cell_id": cell_id,
            "axes": axes,
            "adapter_by_seed": adapters,
        }

    defaults = config["defaults"]
    common_fields = (
        "task",
        "model",
        "rounds",
        "prompts",
        "shots",
        "shot_bank_size",
        "n_test",
        "train_partition",
        "eval_partition",
        "answer_event_mode",
        "answer_target_termination",
        "evaluation_prompt",
        "task_seed_from_run_seed",
        "question_sampling",
        "lora_r",
        "lora_alpha",
        "lora_target_set",
        "save_adapter",
    )
    final_rows = [
        finalist("frozen_base", BASE_CELL),
        finalist("posterior", selected["posterior"]),
        finalist("non_rl_self_training", selected["non_rl_self_training"]),
        finalist("rl", selected["rl"]),
    ]
    return {
        "schema_version": 1,
        "analysis_status": "final_validation_confirmation_complete",
        "source_registry_id": SOURCE_REGISTRY_ID,
        "source_run_id": RUN_ID,
        "source_config_sha256": _sha256(config_path),
        "validator_marker_sha256": _sha256(marker_path),
        "official_test_used": False,
        "metric_order": list(METRIC_ORDER),
        "checkpoint_rule": "final_round_32",
        "confirmation_seed_values": list(SEEDS),
        "common_defaults": {field: defaults[field] for field in common_fields},
        "finalists": final_rows,
        "scale_comparator_role": scale_comparator_role,
        "rankings": rankings,
        "posterior_vs_base": {
            cell: _contrast_status(contrasts, cell, BASE_CELL)
            for cell in ROLE_CANDIDATES["posterior"]
        },
        "posterior_head_to_head": _contrast_status(
            contrasts, Q5_CELL, PIS_CELL
        ),
        "selection_interpretation": (
            "Roles are selected deterministically by mean final extracted Acc@1, "
            "then strict final accuracy and extracted trajectory AUC. Interval "
            "status controls superiority language, not whether the frozen methods "
            "are carried into the prespecified robustness evaluation."
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-source-job", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2_026_081_801)
    args = parser.parse_args()

    config, cells = load_design(args.config)
    marker = verify_marker(
        args.marker,
        args.config,
        expected_commit=args.expected_commit,
        expected_source_job=args.expected_source_job,
    )
    questions, seeds, resources, receipts = load_results(
        args.artifact_dir,
        config,
        cells,
        expected_commit=args.expected_commit,
    )
    cube, question_ids = question_metric_cube(questions)
    bootstrap = hierarchical_method_bootstrap(
        cube,
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    contrasts = contrast_table(seeds, cube, bootstrap)
    summary = method_summary(seeds, resources)
    decision = build_generality_decision(
        config,
        args.config,
        args.marker,
        cells,
        summary,
        contrasts,
        receipts,
    )
    analysis = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "analysis_status": "complete",
        "official_test_used": False,
        "metric_order": list(METRIC_ORDER),
        "observational_unit": "paired training seed",
        "uncertainty": (
            "nested paired bootstrap over training seeds and matched evaluation questions"
        ),
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "checkpoint_rule": "final_round_32",
        "question_count": len(question_ids),
        "cell_count": len(CELL_ORDER),
        "seed_count": len(SEEDS),
        "validator_marker": marker,
        "generality_decision_file": "generality_decision.json",
        "limitations": [
            "The fixed 400-question validation partition has been reused for development.",
            "Seven fresh training seeds reduce seed uncertainty but do not create a fresh question holdout.",
            "Unadjusted prespecified intervals do not establish equivalence when they include zero.",
            "The official GSM8K test remains sealed and no post-confirmation tuning is permitted.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    questions.to_csv(args.output_dir / "question_metrics.csv.gz", index=False)
    seeds.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    resources.to_csv(args.output_dir / "seed_resources.csv", index=False)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "paired_contrasts.csv", index=False)
    tradeoffs = summary[
        [
            "cell",
            "final_extracted",
            "final_strict",
            "extracted_auc",
            *[
                f"mean_{field}"
                for field in (*RESOURCE_FIELDS, *OPTIONAL_RESOURCE_FIELDS)
            ],
        ]
    ]
    tradeoffs.to_csv(args.output_dir / "accuracy_resource_tradeoffs.csv", index=False)
    _write_json(args.output_dir / "generality_decision.json", decision)
    _write_json(args.output_dir / "analysis.json", analysis)
    print(json.dumps(decision, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
