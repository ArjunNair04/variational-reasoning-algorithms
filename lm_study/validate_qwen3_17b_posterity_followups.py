#!/usr/bin/env python3
"""Fail-closed validator for the 42-task posterior-update follow-up."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import re

import yaml

from generate_qwen3_17b_posterity_followups import CELL_ORDER, RUN_ID, SEEDS, build_payload
from result_contract import (
    ResultContractError,
    atomic_write_json,
    validate_completion_receipt,
    validate_receipt_identity,
)
from run_yaml import _prepare_cells
from validate_yaml_run import validate as validate_yaml_outputs


TASK_COUNT = len(CELL_ORDER) * len(SEEDS)


def _expected_coordinates(config: dict) -> list[dict]:
    cells = _prepare_cells(
        config,
        only=None,
        run_id=str(config["run_id"]),
        defaults=config["defaults"],
    )
    rows = []
    for cell_index, cell in enumerate(cells):
        for seed_index, seed in enumerate(SEEDS):
            rows.append(
                {
                    "task_id": cell_index * len(SEEDS) + seed_index + 1,
                    "cell_id": CELL_ORDER[cell_index],
                    "model": str(cell.model),
                    "method": str(cell.method),
                    "seed": seed,
                    "tag": f"{cell.tag}_seed{seed}",
                }
            )
    return rows


def _validate_logs(pattern: str) -> list[str]:
    problems = []
    paths = [Path(value) for value in sorted(glob.glob(str(Path(pattern).expanduser())))]
    if len(paths) != TASK_COUNT:
        return [f"expected {TASK_COUNT} payload logs, found {len(paths)}"]
    observed = []
    failure = re.compile(
        r"Traceback|CUDA out of memory|OutOfMemoryError|No space left|Disk quota exceeded|FAILED|ERROR:"
    )
    for path in paths:
        match = re.search(r"\.([0-9]+)\.log$", path.name)
        if not match:
            problems.append(f"cannot parse task id from {path}")
            continue
        observed.append(int(match.group(1)))
        text = path.read_text(encoding="utf-8", errors="replace")
        if "=== done in" not in text:
            problems.append(f"payload log lacks terminal marker: {path}")
        if failure.search(text):
            problems.append(f"payload log contains a failure signature: {path}")
    if sorted(observed) != list(range(1, TASK_COUNT + 1)):
        problems.append(f"payload task coverage changed: {sorted(observed)}")
    return problems


def _validate_receipts(config: dict, result_root: Path) -> list[str]:
    problems = []
    expected = _expected_coordinates(config)
    receipts = sorted(result_root.glob("complete_*.json"))
    if len(receipts) != TASK_COUNT:
        problems.append(f"expected {TASK_COUNT} completion receipts, found {len(receipts)}")
    fingerprints = set()
    for coordinate in expected:
        seed = coordinate["seed"]
        tag = coordinate["tag"]
        method = coordinate["method"]
        receipt_path = result_root / f"complete_gsm8k__{tag}__{method}_s{seed}.json"
        if not receipt_path.is_file():
            problems.append(f"missing completion receipt: {receipt_path}")
            continue
        try:
            receipt = validate_completion_receipt(receipt_path, result_root=result_root)
            validate_receipt_identity(
                receipt,
                {
                    "run_id": RUN_ID,
                    "task": "gsm8k",
                    "model": coordinate["model"],
                    "method": method,
                    "seed": seed,
                    "tag": tag,
                },
            )
        except (OSError, ResultContractError, ValueError) as exc:
            problems.append(f"invalid completion receipt {receipt_path}: {exc}")
            continue
        fingerprint = str(receipt.get("cell_fingerprint") or "")
        if not fingerprint or fingerprint in fingerprints:
            problems.append(f"missing or duplicate cell fingerprint: {receipt_path}")
        fingerprints.add(fingerprint)
        artifacts = {str(record.get("path") or "") for record in receipt.get("artifacts") or []}
        for prefix in (
            "cell_result_",
            "eval_",
            "prompt_contract_",
            "passk_",
            "checkpoint_eval_",
            "training_diagnostics_",
            "traj_",
        ):
            if not any(Path(path).name.startswith(prefix) for path in artifacts):
                problems.append(f"{receipt_path}: no artifact starts with {prefix}")
        adapter_prefix = f"adapter_gsm8k__{tag}__{method}_s{seed}/"
        if not any(
            path in artifacts
            for path in {
                f"{adapter_prefix}adapter_model.safetensors",
                f"{adapter_prefix}adapter_model.bin",
            }
        ):
            problems.append(f"{receipt_path}: adapter weights are missing")
        if f"{adapter_prefix}adapter_config.json" not in artifacts:
            problems.append(f"{receipt_path}: adapter config is missing")
    return problems


def validate_followups(
    config_path: Path,
    *,
    log_glob: str,
    expected_commit: str,
    expected_config_sha256: str,
    source_job_id: str,
) -> tuple[list[str], dict]:
    problems = []
    observed_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if observed_sha != expected_config_sha256:
        problems.append("configuration SHA mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config != build_payload():
        problems.append("configuration differs from the generated frozen payload")
    if not re.fullmatch(r"[0-9a-f]{7,40}", expected_commit):
        problems.append(f"invalid execution commit {expected_commit!r}")
    if not source_job_id.isdigit():
        problems.append(f"invalid source job id {source_job_id!r}")
    problems.extend(validate_yaml_outputs(config_path, log_glob))
    problems.extend(_validate_logs(log_glob))
    result_root = Path(str(config["defaults"]["out"])).expanduser()
    problems.extend(_validate_receipts(config, result_root))
    marker = {
        "schema_version": 1,
        "status": "ok",
        "run_id": RUN_ID,
        "execution_commit": expected_commit,
        "configuration_sha256": observed_sha,
        "source_job_id": source_job_id,
        "task_count": TASK_COUNT,
        "trained_adapter_count": TASK_COUNT,
        "official_test_used": False,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    return problems, marker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-glob", required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--source-job-id", required=True)
    args = parser.parse_args()
    problems, marker = validate_followups(
        args.config,
        log_glob=args.log_glob,
        expected_commit=args.expected_commit,
        expected_config_sha256=args.expected_config_sha256,
        source_job_id=args.source_job_id,
    )
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 1
    atomic_write_json(args.marker, marker)
    print(json.dumps(marker, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
