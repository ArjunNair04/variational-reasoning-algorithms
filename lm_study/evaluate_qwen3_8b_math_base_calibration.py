"""Frozen Qwen3-8B calibration on a train-derived MATH validation partition."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import sys
import time
from typing import Any

import numpy as np
import yaml

from math_answer_events import (
    canonical_math_solution,
    gold_math_answer,
    math_answers_equivalent,
    parse_math_answer_event,
)
from math_prompting import (
    MATH_CHAT_BOUNDARY_PROMPT_VERSION,
    MATH_CHAT_EOD_TOKEN_ID,
    MATH_CHAT_EOT_TOKEN_ID,
    MATH_CHAT_PROMPT_VERSION,
    MATH_DEMONSTRATION_SEPARATOR_TOKEN_ID,
    MATH_INSTRUCTION_PROMPT_VERSION,
    MATH_PROMPT_VERSION,
    build_math_chat_messages,
    build_math_prompts,
    is_math_chat_prompt_version,
    math_prompt_contract,
    render_math_chat_prompts,
    validate_math_chat_tokenizer,
    validate_math_model_eos,
    validate_math_prompt_tokenizer,
)


TRAIN_FILENAME = "train-00000-of-00001.parquet"
EMPTY_BOX_GOLD_REPAIRS = {
    (
        "number_theory",
        661,
        "5e1b77771c858fd0c05e248fc23f6cc172fa28b273ea4bbe40c69fd9b816993d",
    ): "0",
    (
        "number_theory",
        663,
        "641bf04313760e5b7b3660e328a8bdc073aaf748752ae313ccae7c04a8d33f74",
    ): "0",
}
GENERATED_PROBLEM_HEADER = re.compile(r"(?m)^Problem:\s*$")
GENERATED_SOLUTION_HEADER = re.compile(r"(?m)^Solution:\s*$")


def _current_rss_bytes() -> int | None:
    """Return the current Linux resident set without adding a dependency."""

    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def emit_stage(stage: str, **details: Any) -> None:
    """Write a flushed breadcrumb with enough memory context for SGE failures."""

    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        max_rss *= 1024
    payload = {
        "event": "qwen3_8b_math_calibration_stage",
        "stage": stage,
        "pid": os.getpid(),
        "host_current_rss_bytes": _current_rss_bytes(),
        "host_max_rss_bytes": max_rss,
        **details,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_id_sha256(ids: list[str]) -> str:
    """Hash an ordered ID list using the frozen compact-JSON encoding."""

    encoded = json.dumps(
        [str(dataset_id) for dataset_id in ids],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_chat_qualification_identities(
    protocol: dict[str, Any],
    demonstrations: list[dict[str, Any]],
    qualification: list[dict[str, Any]],
    *,
    partition_role: str = "qualification",
) -> dict[str, Any]:
    """Bind the selected chat cohort and demonstrations before generation."""

    if partition_role not in {"qualification", "validation"}:
        raise ValueError("unknown chat identity partition role")

    frozen = protocol["ordered_id_provenance"]
    demonstration_ids = [str(row["dataset_id"]) for row in demonstrations]
    qualification_ids = [str(row["dataset_id"]) for row in qualification]
    demonstration_digest = ordered_id_sha256(demonstration_ids)
    qualification_digest = ordered_id_sha256(qualification_ids)
    if demonstration_ids != list(frozen["demonstration_ids"]):
        raise ValueError("ordered MATH demonstration IDs changed")
    if demonstration_digest != frozen["demonstration_ids_sha256"]:
        raise ValueError("ordered MATH demonstration ID digest changed")
    digest_key = f"{partition_role}_ids_sha256"
    if qualification_digest != frozen[digest_key]:
        raise ValueError(f"ordered MATH chat {partition_role} ID digest changed")
    result = {
        "serialization": frozen["serialization"],
        "digest": frozen["digest"],
        "demonstration_ids": demonstration_ids,
        "demonstration_ids_sha256": demonstration_digest,
        "partition_role": partition_role,
        f"{partition_role}_ids": qualification_ids,
        digest_key: qualification_digest,
    }
    # Exact historical qualification summaries predate the explicit role key.
    if partition_role == "qualification":
        result.pop("partition_role")
    return result


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = yaml.safe_load(raw)
    required = {"run_id", "model", "dataset", "partition", "generation", "decision"}
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"protocol missing fields: {missing}")
    if protocol["dataset"].get("source_split") != "train":
        raise ValueError("MATH calibration may load only the training split")
    if protocol["decision"].get("official_test_used") is not False:
        raise ValueError("official_test_used must be false")
    prompt = protocol.get("prompt")
    if prompt is not None:
        try:
            expected_prompt = math_prompt_contract(str(prompt.get("version")))
        except ValueError as error:
            raise ValueError("unknown MATH prompt contract") from error
        if prompt != expected_prompt:
            raise ValueError("unknown MATH prompt contract")
    if prompt is not None:
        partition = protocol["partition"]
        if prompt["version"] == MATH_INSTRUCTION_PROMPT_VERSION or (
            is_math_chat_prompt_version(prompt["version"])
        ):
            is_chat = is_math_chat_prompt_version(prompt["version"])
            is_boundary = prompt["version"] == MATH_CHAT_BOUNDARY_PROMPT_VERSION
            prior_qualification_size = 96 if is_boundary else 64 if is_chat else 32
            qualification_start = 404 + prior_qualification_size
            expected_partition = {
                "seed": 2026081501,
                "validation_size": 400,
                "shots": 4,
                "prior_qualification_size": prior_qualification_size,
                "qualification_size": 32,
                "validation_positions": [0, 400],
                "demonstration_positions": [400, 404],
                "prior_qualification_positions": [404, qualification_start],
                "qualification_positions": [
                    qualification_start,
                    qualification_start + 32,
                ],
                "optimization_positions": [qualification_start + 32, 7500],
            }
            protocol_role = protocol.get("protocol_role")
            is_fixed_cap_chat_calibration = (
                protocol_role == "fixed_cap_chat_calibration"
            )
            if protocol_role not in (None, "fixed_cap_chat_calibration"):
                raise ValueError("unknown MATH protocol role")
            if is_chat:
                expected_partition["eos_qualification_positions"] = [404, 436]
                expected_partition["instruction_qualification_positions"] = [436, 468]
                if is_boundary:
                    expected_partition["chat_qualification_positions"] = [468, 500]
                expected_id_provenance = {
                    "serialization": "compact_json_utf8",
                    "digest": "sha256",
                    "demonstration_ids": [
                        "prealgebra:965",
                        "intermediate_algebra:1081",
                        "algebra:333",
                        "counting_and_probability:13",
                    ],
                    "demonstration_ids_sha256": (
                        "b8593446f39fd81b0fdf0e5363509ebc41452fd514ed2b5588bcc479d2504d92"
                    ),
                    "qualification_ids_sha256": (
                        "7f1e27c22b219bcae772430badd885fa39fab2c248cd0c9b4ff5c78eb84f3289"
                        if is_boundary
                        else "e18d47ed562af88ae3399dee6df4b75e497c063c658099f2fe397e62e6f3a701"
                    ),
                }
                if is_fixed_cap_chat_calibration:
                    if not is_boundary:
                        raise ValueError(
                            "fixed-cap chat calibration requires prompt v4"
                        )
                    expected_id_provenance["validation_ids_sha256"] = (
                        "7994f3aa83f61214626422c65f90a89a92c3d334fb4e997c7c95b56e1f6abf0c"
                    )
                if protocol.get("ordered_id_provenance") != expected_id_provenance:
                    raise ValueError("unknown MATH chat ordered-ID provenance")
            expected_modes = (
                ["full"] if is_fixed_cap_chat_calibration else ["qualification"]
            )
            if protocol.get("allowed_modes") != expected_modes:
                if is_fixed_cap_chat_calibration:
                    raise ValueError("fixed-cap chat calibration must be full-only")
                raise ValueError("prompt successor must be qualification-only")
            manifest = protocol.get("source_manifest")
            categories = list(protocol["dataset"]["categories"])
            if not isinstance(manifest, list) or len(manifest) != len(categories):
                raise ValueError("prompt successor requires seven source files")
            if [source.get("category") for source in manifest] != categories:
                raise ValueError("source manifest category order mismatch")
            for category, source in zip(categories, manifest, strict=True):
                if set(source) != {"category", "filename", "rows", "sha256"}:
                    raise ValueError("source manifest fields changed")
                if source["filename"] != f"{category}/{TRAIN_FILENAME}":
                    raise ValueError("source manifest filename mismatch")
                if int(source["rows"]) <= 0:
                    raise ValueError("source manifest row count must be positive")
                if re.fullmatch(r"[0-9a-f]{64}", str(source["sha256"])) is None:
                    raise ValueError("source manifest SHA-256 is malformed")
            if sum(int(source["rows"]) for source in manifest) != 7_500:
                raise ValueError("source manifest must contain 7,500 rows")
        else:
            expected_partition = {
                "seed": 2026081501,
                "validation_size": 400,
                "shots": 4,
                "qualification_size": 32,
                "validation_positions": [0, 400],
                "demonstration_positions": [400, 404],
                "qualification_positions": [404, 436],
                "optimization_positions": [436, 7500],
            }
        if any(partition.get(key) != value for key, value in expected_partition.items()):
            raise ValueError("unknown MATH successor partition contract")
        generation = protocol["generation"]
        if prompt["version"] == MATH_CHAT_BOUNDARY_PROMPT_VERSION:
            expected_model = {
                "id": "Qwen/Qwen3-8B",
                "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                "max_position_embeddings": 40960,
            }
            if protocol["model"] != expected_model:
                raise ValueError("MATH boundary qualification model contract changed")
            expected_generation = (
                {
                    "max_new_tokens": 1024,
                    "batch_size": 4,
                    "do_sample": False,
                }
                if protocol.get("protocol_role") == "fixed_cap_chat_calibration"
                else {
                    "max_new_tokens": 8192,
                    "batch_size": 4,
                    "do_sample": False,
                    "qualification_cap_ladder": [1024, 2048, 4096, 8192],
                }
            )
            if generation != expected_generation:
                raise ValueError("MATH boundary qualification generation contract changed")
            if protocol.get("protocol_role") == "fixed_cap_chat_calibration":
                if protocol.get("run_id") != "c7a4d912":
                    raise ValueError("MATH fixed-cap calibration run ID changed")
                qualification = protocol.get("qualification")
                expected_qualification = {
                    "run_id": "8e4c21d7",
                    "execution_commit": (
                        "93f78ef82f249ad0fd97bc247842bca986e153f7"
                    ),
                    "configuration_sha256": (
                        "6435d53d6df963447afade809a6b5e074736eca2a275aa9694b34e163672b5ec"
                    ),
                    "marker_sha256": (
                        "af5cdb3da80f7afc434694f2696314c2cc11ccc167cdbceb5550ad30c16a9b1a"
                    ),
                    "classification": "ready_for_fixed_cap_calibration",
                    "selected_operational_cap": 1024,
                }
                if qualification != expected_qualification:
                    raise ValueError("MATH fixed-cap qualification binding changed")
                expected_decision = {
                    "minimum_extracted_accuracy": 0.20,
                    "maximum_extracted_accuracy": 0.95,
                    "minimum_strict_given_extracted": 0.80,
                    "maximum_token_limit_rate": 0.10,
                    "maximum_generated_header_count": 0,
                    "minimum_natural_eos_rate": 0.90,
                    "minimum_strict_valid_rate": 0.70,
                    "maximum_generated_eod_count": 0,
                    "official_test_used": False,
                }
                if protocol.get("decision") != expected_decision:
                    raise ValueError("MATH fixed-cap prospective gates changed")
                if protocol.get("reporting_order") != [
                    "final_extracted_answer_accuracy",
                    "final_strict_terminal_accuracy",
                    "trajectory_auc_not_applicable_to_frozen_calibration",
                ]:
                    raise ValueError("MATH fixed-cap reporting order changed")
        elif (
            generation.get("max_new_tokens") != 512
            or generation.get("do_sample") is not False
        ):
            raise ValueError("MATH successor generation contract changed")
    return protocol, hashlib.sha256(raw).hexdigest()


def source_math_gold(
    category: str,
    local_index: int,
    solution: str,
) -> tuple[str, str]:
    """Extract a gold answer, repairing two checksum-bound empty source boxes."""

    gold = gold_math_answer(solution)
    canonical = canonical_math_solution(solution)
    if gold and canonical:
        return gold, canonical
    solution_sha256 = hashlib.sha256(solution.encode("utf-8")).hexdigest()
    repair = EMPTY_BOX_GOLD_REPAIRS.get((category, local_index, solution_sha256))
    if repair is None or solution.count(r"\boxed{}") != 1:
        raise ValueError("missing boxed gold")
    repaired = solution.replace(r"\boxed{}", rf"\boxed{{{repair}}}", 1)
    gold = gold_math_answer(repaired)
    canonical = canonical_math_solution(repaired)
    if gold != repair or not canonical:
        raise AssertionError("MATH source-gold repair failed")
    return gold, canonical


def validate_math_source_manifest(
    protocol: dict[str, Any],
    computed_sources: list[dict[str, Any]],
) -> None:
    """Bind runtime source files to the manifest frozen in the protocol."""

    expected_sources = protocol.get("source_manifest")
    if expected_sources is not None and computed_sources != expected_sources:
        raise ValueError("computed MATH source manifest does not match protocol")


def load_math_train_records(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Download only pinned train parquet files and validate all gold answers."""

    import pandas as pd
    from huggingface_hub import hf_hub_download

    dataset = protocol["dataset"]
    records: list[dict[str, Any]] = []
    sources = []
    for category in dataset["categories"]:
        filename = f"{category}/{TRAIN_FILENAME}"
        if "/test" in filename or filename.startswith("test"):
            raise ValueError(f"prohibited MATH test path: {filename}")
        path = Path(
            hf_hub_download(
                repo_id=dataset["id"],
                repo_type="dataset",
                filename=filename,
                revision=dataset["revision"],
            )
        )
        frame = pd.read_parquet(path)
        if set(frame.columns) != {"problem", "level", "type", "solution"}:
            raise ValueError(f"unexpected MATH schema for {category}: {list(frame.columns)}")
        sources.append(
            {
                "category": category,
                "filename": filename,
                "rows": int(len(frame)),
                "sha256": file_sha256(path),
            }
        )
        for local_index, row in frame.iterrows():
            solution = str(row["solution"])
            try:
                gold, canonical = source_math_gold(
                    category, int(local_index), solution
                )
            except ValueError as error:
                raise ValueError(
                    f"missing boxed gold at {category}:{local_index}"
                ) from error
            records.append(
                {
                    "dataset_id": f"{category}:{int(local_index)}",
                    "category": category,
                    "level": str(row["level"]),
                    "problem": str(row["problem"]),
                    "solution": solution,
                    "gold": gold,
                    "canonical_solution": canonical,
                }
            )
    if len(records) != 7_500:
        raise ValueError(f"expected 7,500 MATH train rows, found {len(records)}")
    if len({record["dataset_id"] for record in records}) != len(records):
        raise ValueError("MATH dataset IDs are not unique")
    validate_math_source_manifest(protocol, sources)
    return records, sources


def partition_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    validation_size: int,
    shots: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_size + shots > len(records):
        raise ValueError("MATH partition request exceeds training rows")
    order = np.random.default_rng(seed).permutation(len(records))
    validation = [records[int(index)] for index in order[:validation_size]]
    demonstrations = [
        records[int(index)]
        for index in order[validation_size : validation_size + shots]
    ]
    if {row["dataset_id"] for row in validation} & {
        row["dataset_id"] for row in demonstrations
    }:
        raise AssertionError("MATH demonstrations overlap validation")
    return validation, demonstrations


def partition_successor_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    validation_size: int,
    shots: int,
    qualification_size: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Carve fixed validation, demos, qualification and optimization cohorts."""

    reserved = validation_size + shots + qualification_size
    if reserved > len(records):
        raise ValueError("MATH successor partition exceeds training rows")
    order = np.random.default_rng(seed).permutation(len(records))
    validation = [records[int(index)] for index in order[:validation_size]]
    demonstrations = [
        records[int(index)]
        for index in order[validation_size : validation_size + shots]
    ]
    qualification = [
        records[int(index)]
        for index in order[
            validation_size + shots : validation_size + shots + qualification_size
        ]
    ]
    optimization = [records[int(index)] for index in order[reserved:]]
    cohorts = (validation, demonstrations, qualification, optimization)
    ids = [{row["dataset_id"] for row in cohort} for cohort in cohorts]
    if any(ids[left] & ids[right] for left in range(4) for right in range(left + 1, 4)):
        raise AssertionError("MATH successor cohorts overlap")
    if sum(len(cohort) for cohort in cohorts) != len(records):
        raise AssertionError("MATH successor partition is not exhaustive")
    return cohorts


def partition_instruction_successor_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    validation_size: int,
    shots: int,
    prior_qualification_size: int,
    qualification_size: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Reserve both prompt qualifications and exclude both from optimization."""

    prior_start = validation_size + shots
    qualification_start = prior_start + prior_qualification_size
    reserved = qualification_start + qualification_size
    if reserved > len(records):
        raise ValueError("MATH instruction-successor partition exceeds training rows")
    order = np.random.default_rng(seed).permutation(len(records))

    def rows(start: int, stop: int) -> list[dict[str, Any]]:
        return [records[int(index)] for index in order[start:stop]]

    validation = rows(0, validation_size)
    demonstrations = rows(validation_size, prior_start)
    prior_qualification = rows(prior_start, qualification_start)
    qualification = rows(qualification_start, reserved)
    optimization = rows(reserved, len(records))
    cohorts = (
        validation,
        demonstrations,
        prior_qualification,
        qualification,
        optimization,
    )
    ids = [{row["dataset_id"] for row in cohort} for cohort in cohorts]
    if any(
        ids[left] & ids[right]
        for left in range(5)
        for right in range(left + 1, 5)
    ):
        raise AssertionError("MATH instruction-successor cohorts overlap")
    if sum(len(cohort) for cohort in cohorts) != len(records):
        raise AssertionError("MATH instruction-successor partition is not exhaustive")
    return cohorts


def build_prompts(
    validation: list[dict[str, Any]],
    demonstrations: list[dict[str, Any]],
    *,
    version: str = MATH_PROMPT_VERSION,
) -> list[str]:
    return build_math_prompts(validation, demonstrations, version=version)


def build_legacy_prompts(
    validation: list[dict[str, Any]],
    demonstrations: list[dict[str, Any]],
) -> list[str]:
    """Retain the historical c881ab4d rendering for exact source replay."""

    preamble = "".join(
        f"Problem:\n{row['problem']}\n\nSolution:\n{row['canonical_solution']}\n\n"
        for row in demonstrations
    )
    return [preamble + f"Problem:\n{row['problem']}\n\nSolution:\n" for row in validation]


def score_completions(
    rows: list[dict[str, Any]],
    completions: list[str],
    generation_metadata: list[dict[str, Any]],
    *,
    strict_disallowed_exact_answers: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if not (len(rows) == len(completions) == len(generation_metadata)):
        raise ValueError("MATH scoring inputs are not aligned")
    scored = []
    for row, completion, generation in zip(rows, completions, generation_metadata):
        extracted = parse_math_answer_event(completion, mode="legacy")
        strict = parse_math_answer_event(
            completion,
            mode="strict_terminal_marker",
            disallowed_exact_answers=strict_disallowed_exact_answers,
        )
        extracted_correct = math_answers_equivalent(extracted.answer, row["gold"])
        strict_correct = strict.strict_valid and math_answers_equivalent(
            strict.answer, row["gold"]
        )
        scored_row = {
            "dataset_id": row["dataset_id"],
            "category": row["category"],
            "level": row["level"],
            "problem": row["problem"],
            "gold": row["gold"],
            "completion": completion,
            "extracted_answer": extracted.answer,
            "extracted_parse_mode": extracted.parse_mode,
            "extracted_correct": bool(extracted_correct),
            "strict_answer": strict.answer,
            "strict_valid": bool(strict.strict_valid),
            "strict_correct": bool(strict_correct),
            "strict_correct_and_eos": bool(
                strict_correct and generation.get("generated_eos") is True
            ),
            "generated_eos": bool(generation.get("generated_eos")),
            "generated_tokens_until_eos": generation.get("generated_tokens_until_eos"),
            "hit_max_new_tokens": bool(generation.get("hit_max_new_tokens")),
            "generated_problem_header": bool(
                GENERATED_PROBLEM_HEADER.search(completion)
            ),
            "generated_solution_header": bool(
                GENERATED_SOLUTION_HEADER.search(completion)
            ),
        }
        if "generated_stop_token_id" in generation:
            scored_row["generated_stop_token_id"] = generation[
                "generated_stop_token_id"
            ]
            scored_row["generated_eod_before_eot"] = bool(
                generation["generated_eod_before_eot"]
            )
        for field in (
            "active_generated_token_ids",
            "active_generated_token_ids_sha256",
            "decoded_prefixes_by_cap",
        ):
            if field in generation:
                scored_row[field] = generation[field]
        scored.append(scored_row)
    return scored


def summarize(
    scored: list[dict[str, Any]],
    *,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    def mean(field: str) -> float:
        return float(np.mean([bool(row.get(field, False)) for row in scored]))

    extracted = sum(bool(row["extracted_correct"]) for row in scored)
    strict = sum(bool(row["strict_correct"]) for row in scored)
    by_category = {}
    for category in sorted({row["category"] for row in scored}):
        selected = [row for row in scored if row["category"] == category]
        by_category[category] = {
            "n": len(selected),
            "extracted_accuracy": float(
                np.mean([bool(row["extracted_correct"]) for row in selected])
            ),
            "strict_accuracy": float(
                np.mean([bool(row["strict_correct"]) for row in selected])
            ),
        }
    summary = {
        "n": len(scored),
        "final_extracted_answer_accuracy": extracted / len(scored),
        "final_strict_terminal_accuracy": strict / len(scored),
        "strict_given_extracted": strict / extracted if extracted else 0.0,
        "natural_eos_rate": mean("generated_eos"),
        "token_limit_rate": mean("hit_max_new_tokens"),
        "strict_correct_and_eos_rate": mean("strict_correct_and_eos"),
        "strict_valid_rate": mean("strict_valid"),
        "generated_problem_header_count": sum(
            bool(row.get("generated_problem_header", False)) for row in scored
        ),
        "generated_solution_header_count": sum(
            bool(row.get("generated_solution_header", False)) for row in scored
        ),
        "by_category": by_category,
        "trajectory_auc": None,
    }
    if prompt_version is not None and is_math_chat_prompt_version(prompt_version):
        summary["generated_stop_token_counts"] = {
            str(MATH_CHAT_EOT_TOKEN_ID): sum(
                row.get("generated_stop_token_id") == MATH_CHAT_EOT_TOKEN_ID
                for row in scored
            )
        }
        summary["generated_eod_before_eot_count"] = sum(
            bool(row.get("generated_eod_before_eot", False)) for row in scored
        )
    return summary


def generation_metadata_for_tokens(
    token_ids: list[int],
    *,
    max_new_tokens: int,
    prompt_version: str,
) -> dict[str, Any]:
    """Classify a generated row without treating Qwen3 padding as natural EOS."""

    ids = [int(token_id) for token_id in token_ids]
    if is_math_chat_prompt_version(prompt_version):
        eot_positions = [
            index for index, token_id in enumerate(ids) if token_id == MATH_CHAT_EOT_TOKEN_ID
        ]
        generated_eos = bool(eot_positions)
        tokens = eot_positions[0] + 1 if generated_eos else len(ids)
        active_ids = ids[:tokens]
        return {
            "generated_eos": generated_eos,
            "generated_stop_token_id": (
                MATH_CHAT_EOT_TOKEN_ID if generated_eos else None
            ),
            "generated_eod_before_eot": MATH_CHAT_EOD_TOKEN_ID in active_ids,
            "generated_tokens_until_eos": tokens,
            "hit_max_new_tokens": not generated_eos and tokens >= max_new_tokens,
        }
    eos_positions = [
        index
        for index, token_id in enumerate(ids)
        if token_id == MATH_DEMONSTRATION_SEPARATOR_TOKEN_ID
    ]
    generated_eos = bool(eos_positions)
    tokens = eos_positions[0] + 1 if generated_eos else len(ids)
    return {
        "generated_eos": generated_eos,
        "generated_tokens_until_eos": tokens,
        "hit_max_new_tokens": not generated_eos and tokens >= max_new_tokens,
    }


def boundary_prefix_provenance(
    tokenizer: Any,
    token_ids: list[int],
    *,
    active_token_count: int,
    cap_ladder: list[int],
) -> dict[str, Any]:
    """Freeze raw tokens and exact decoded prefixes for cap counterfactuals."""

    provenance = active_generated_token_provenance(
        token_ids,
        active_token_count=active_token_count,
    )
    active_ids = provenance["active_generated_token_ids"]
    return {
        **provenance,
        "decoded_prefixes_by_cap": {
            str(cap): tokenizer.decode(
                active_ids[:cap],
                skip_special_tokens=True,
            )
            for cap in cap_ladder
        },
    }


def active_generated_token_provenance(
    token_ids: list[int],
    *,
    active_token_count: int,
) -> dict[str, Any]:
    """Persist the exact active generation prefix for termination auditing."""

    active_ids = [int(token_id) for token_id in token_ids[:active_token_count]]
    encoded_ids = json.dumps(active_ids, separators=(",", ":")).encode("utf-8")
    return {
        "active_generated_token_ids": active_ids,
        "active_generated_token_ids_sha256": hashlib.sha256(encoded_ids).hexdigest(),
    }


def validate_boundary_context_window(
    protocol: dict[str, Any],
    model: Any,
    *,
    prompt_tokens: int,
) -> None:
    """Fail before generation if the frozen prefix plus ceiling cannot fit."""

    expected = int(protocol["model"]["max_position_embeddings"])
    actual = int(model.config.max_position_embeddings)
    if actual != expected:
        raise ValueError(
            "Qwen3 boundary max-position contract changed: "
            f"expected {expected}, found {actual}"
        )
    requested = int(prompt_tokens) + int(protocol["generation"]["max_new_tokens"])
    if requested > actual:
        raise ValueError(
            "Qwen3 boundary prompt plus generation ceiling exceeds context: "
            f"{prompt_tokens} + {protocol['generation']['max_new_tokens']} > {actual}"
        )


def generate(protocol: dict[str, Any], prompts: list[Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    emit_stage("generation_runtime_imported", cuda_available=torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-8B MATH calibration requires CUDA")
    model_spec = protocol["model"]
    emit_stage("tokenizer_load_started")
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["id"], revision=model_spec["revision"]
    )
    prompt_contract = None
    prompt_version = (
        str(protocol["prompt"]["version"])
        if protocol.get("prompt") is not None
        else MATH_PROMPT_VERSION
    )
    is_chat = is_math_chat_prompt_version(prompt_version)
    if is_chat:
        prompt_contract = validate_math_chat_tokenizer(tokenizer, version=prompt_version)
        prompts = render_math_chat_prompts(
            tokenizer,
            prompts,
            version=prompt_version,
        )
    elif protocol.get("prompt") is not None:
        prompt_contract = validate_math_prompt_tokenizer(
            tokenizer,
            version=prompt_version,
        )
    emit_stage("tokenizer_loaded")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    emit_stage("model_load_started")
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["id"],
        revision=model_spec["revision"],
        torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
        low_cpu_mem_usage=True,
    )
    emit_stage("model_loaded_on_host")
    model = model.to("cuda")
    validate_math_model_eos(model, tokenizer, version=prompt_version)
    emit_stage(
        "model_loaded_on_gpu",
        gpu_memory_allocated_bytes=int(torch.cuda.memory_allocated()),
        gpu_memory_reserved_bytes=int(torch.cuda.memory_reserved()),
    )
    model.eval()
    tokenizer.padding_side = "left"
    completions: list[str] = []
    metadata: list[dict[str, Any]] = []
    batch_size = int(protocol["generation"]["batch_size"])
    max_new = int(protocol["generation"]["max_new_tokens"])
    max_prompt_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    emit_stage(
        "generation_started",
        prompts=len(prompts),
        batch_size=batch_size,
        max_new_tokens=max_new,
    )
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            emit_stage(
                "generation_batch_started",
                batch_start=start,
                batch_size=len(chunk),
                gpu_memory_allocated_bytes=int(torch.cuda.memory_allocated()),
            )
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                add_special_tokens=not is_chat,
            ).to("cuda")
            max_prompt_tokens = max(max_prompt_tokens, int(encoded.input_ids.shape[1]))
            if prompt_version == MATH_CHAT_BOUNDARY_PROMPT_VERSION:
                validate_boundary_context_window(
                    protocol,
                    model,
                    prompt_tokens=int(encoded.input_ids.shape[1]),
                )
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new,
                pad_token_id=(
                    MATH_CHAT_EOD_TOKEN_ID if is_chat else tokenizer.eos_token_id
                ),
                eos_token_id=(
                    MATH_CHAT_EOT_TOKEN_ID
                    if is_chat
                    else MATH_DEMONSTRATION_SEPARATOR_TOKEN_ID
                ),
            )
            continuation = generated[:, encoded.input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            for token_row in continuation:
                generation = generation_metadata_for_tokens(
                    token_row.tolist(),
                    max_new_tokens=max_new,
                    prompt_version=prompt_version,
                )
                if protocol.get("protocol_role") == "fixed_cap_chat_calibration":
                    generation.update(
                        active_generated_token_provenance(
                            token_row.tolist(),
                            active_token_count=int(
                                generation["generated_tokens_until_eos"]
                            ),
                        )
                    )
                metadata.append(generation)
            if "qualification_cap_ladder" in protocol["generation"]:
                cap_ladder = [
                    int(cap)
                    for cap in protocol["generation"]["qualification_cap_ladder"]
                ]
                decoded = []
                for token_row, generation in zip(
                    continuation,
                    metadata[-len(chunk) :],
                    strict=True,
                ):
                    generation.update(
                        boundary_prefix_provenance(
                            tokenizer,
                            token_row.tolist(),
                            active_token_count=int(
                                generation["generated_tokens_until_eos"]
                            ),
                            cap_ladder=cap_ladder,
                        )
                    )
                    decoded.append(
                        generation["decoded_prefixes_by_cap"][str(max_new)]
                    )
            completions.extend(decoded)
            emit_stage(
                "generation_batch_completed",
                completed_prompts=min(start + len(chunk), len(prompts)),
                gpu_memory_allocated_bytes=int(torch.cuda.memory_allocated()),
                gpu_peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            )
    emit_stage("generation_completed", completions=len(completions))
    return completions, metadata, {
        "max_prompt_tokens": max_prompt_tokens,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_name": torch.cuda.get_device_name(0),
    }, prompt_contract


def write_outputs(
    output: Path,
    *,
    protocol: dict[str, Any],
    config_sha256: str,
    mode: str,
    expected_commit: str,
    sources: list[dict[str, Any]],
    demonstrations: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    runtime: dict[str, Any],
    elapsed_seconds: float,
    prompt_contract: dict[str, Any] | None = None,
    partition_role: str | None = None,
    partition_positions: list[int] | None = None,
    ordered_id_provenance: dict[str, Any] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    records_path = output / "records.jsonl.gz"
    with gzip.open(records_path, "wt", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": 1,
        "run_id": protocol["run_id"],
        "mode": mode,
        "execution_commit": expected_commit,
        "configuration_sha256": config_sha256,
        "model": protocol["model"],
        "dataset": protocol["dataset"],
        "dataset_splits_loaded": ["train"],
        "official_test_accessed": False,
        "source_files": sources,
        "demonstration_ids": [row["dataset_id"] for row in demonstrations],
        "partition_role": partition_role or mode,
        "partition_positions": partition_positions,
        "prompt_contract": prompt_contract,
        "metrics": summarize(
            scored,
            prompt_version=(
                str(protocol["prompt"]["version"])
                if protocol.get("prompt") is not None
                else None
            ),
        ),
        "runtime": {**runtime, "elapsed_seconds": elapsed_seconds},
    }
    if ordered_id_provenance is not None:
        summary["ordered_id_provenance"] = ordered_id_provenance
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "run_id": protocol["run_id"],
        "mode": mode,
        "execution_commit": expected_commit,
        "configuration_sha256": config_sha256,
        "source_job_id": str(__import__("os").environ.get("JOB_ID", "local")),
        "records": len(scored),
        "artifacts": {
            "records.jsonl.gz": file_sha256(records_path),
            "summary.json": file_sha256(summary_path),
        },
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("smoke", "qualification", "full"), required=True
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    args = parser.parse_args()
    emit_stage("process_started", mode=args.mode)
    protocol, config_sha = load_protocol(args.config)
    emit_stage("protocol_loaded", mode=args.mode)
    allowed_modes = protocol.get("allowed_modes")
    if allowed_modes is not None and args.mode not in allowed_modes:
        raise ValueError(f"mode {args.mode!r} is blocked by this protocol")
    if config_sha != args.expected_config_sha256:
        raise ValueError("MATH calibration configuration SHA-256 mismatch")
    emit_stage("dataset_load_started")
    records, sources = load_math_train_records(protocol)
    emit_stage("dataset_loaded", rows=len(records), source_files=len(sources))
    partition = protocol["partition"]
    ordered_id_provenance = None
    if protocol.get("prompt") is None:
        validation, demonstrations = partition_records(
            records,
            seed=int(partition["seed"]),
            validation_size=int(partition["validation_size"]),
            shots=int(partition["shots"]),
        )
        if args.mode == "qualification":
            raise ValueError("qualification requires the successor prompt contract")
        if args.mode == "smoke":
            validation = validation[:8]
            positions = [0, 8]
        else:
            positions = [0, int(partition["validation_size"])]
        prompts = build_legacy_prompts(validation, demonstrations)
    else:
        prompt_version = str(protocol["prompt"]["version"])
        if prompt_version == MATH_INSTRUCTION_PROMPT_VERSION or (
            is_math_chat_prompt_version(prompt_version)
        ):
            (
                validation,
                demonstrations,
                _prior_qualification,
                qualification,
                _optimization,
            ) = partition_instruction_successor_records(
                records,
                seed=int(partition["seed"]),
                validation_size=int(partition["validation_size"]),
                shots=int(partition["shots"]),
                prior_qualification_size=int(partition["prior_qualification_size"]),
                qualification_size=int(partition["qualification_size"]),
            )
        else:
            validation, demonstrations, qualification, _optimization = (
                partition_successor_records(
                    records,
                    seed=int(partition["seed"]),
                    validation_size=int(partition["validation_size"]),
                    shots=int(partition["shots"]),
                    qualification_size=int(partition["qualification_size"]),
                )
            )
        if args.mode == "qualification":
            validation = qualification
            positions = list(partition["qualification_positions"])
        elif args.mode == "full":
            positions = list(partition["validation_positions"])
        else:
            raise ValueError("successor calibration has no smoke mode")
        if is_math_chat_prompt_version(prompt_version):
            identity_role = "validation" if args.mode == "full" else "qualification"
            ordered_id_provenance = validate_chat_qualification_identities(
                protocol,
                demonstrations,
                validation,
                partition_role=identity_role,
            )
            cohort_digest_key = f"{identity_role}_ids_sha256"
            emit_stage(
                "chat_provenance_verified",
                demonstration_ids_sha256=ordered_id_provenance[
                    "demonstration_ids_sha256"
                ],
                partition_role=args.mode,
                cohort_ids_sha256=ordered_id_provenance[cohort_digest_key],
                chat_template_sha256=protocol["prompt"]["chat_template_sha256"],
            )
            prompts = build_math_chat_messages(
                validation,
                demonstrations,
                version=prompt_version,
            )
        else:
            prompts = build_prompts(
                validation,
                demonstrations,
                version=prompt_version,
            )
    emit_stage(
        "prompts_ready",
        validation_rows=len(validation),
        demonstrations=len(demonstrations),
    )
    start = time.monotonic()
    completions, metadata, runtime, prompt_contract = generate(protocol, prompts)
    scored = score_completions(
        validation,
        completions,
        metadata,
        strict_disallowed_exact_answers=tuple(
            protocol.get("prompt", {}).get(
                "strict_terminal_disallowed_exact_answers",
                [],
            )
        ),
    )
    emit_stage("scoring_completed", scored_rows=len(scored))
    write_outputs(
        args.output,
        protocol=protocol,
        config_sha256=config_sha,
        mode=args.mode,
        expected_commit=args.expected_commit,
        sources=sources,
        demonstrations=demonstrations,
        scored=scored,
        runtime=runtime,
        prompt_contract=prompt_contract,
        partition_role=args.mode,
        partition_positions=positions,
        ordered_id_provenance=ordered_id_provenance,
        elapsed_seconds=time.monotonic() - start,
    )
    emit_stage("outputs_written", output=str(args.output))


if __name__ == "__main__":
    main()
