#!/usr/bin/env python3
"""Fail-closed validator for the 70-task final-method confirmation."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime, timezone

import yaml

from generate_qwen3_17b_final_method_confirmation import (
    CELL_ORDER,
    RUN_ID,
    SEEDS,
    build_payload,
)
from result_contract import (
    ResultContractError,
    atomic_write_json,
    validate_completion_receipt,
    validate_receipt_identity,
)
from run_yaml import _prepare_cells
from validate_yaml_run import validate as validate_yaml_outputs


TERMINAL_LOG_MARKER = "=== done in"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_coordinates(config: dict) -> list[dict]:
    defaults = config["defaults"]
    cells = _prepare_cells(
        config,
        only=None,
        run_id=str(config["run_id"]),
        defaults=defaults,
    )
    coordinates = []
    for cell_index, (model, method, _axes, base_tag) in enumerate(cells):
        cell_id = CELL_ORDER[cell_index]
        for seed_index, seed in enumerate(SEEDS):
            tag = f"{base_tag}_seed{seed}"
            coordinates.append(
                {
                    "task_id": cell_index * len(SEEDS) + seed_index + 1,
                    "cell_id": cell_id,
                    "model": str(model),
                    "method": str(method),
                    "seed": int(seed),
                    "tag": tag,
                }
            )
    return coordinates


def _validate_logs(pattern: str) -> list[str]:
    problems = []
    paths = [Path(value) for value in sorted(glob.glob(str(Path(pattern).expanduser())))]
    if len(paths) != 70:
        return [f"expected 70 payload logs, found {len(paths)}"]
    observed_tasks = []
    for path in paths:
        match = re.search(r"\.([0-9]+)\.log$", path.name)
        if not match:
            problems.append(f"cannot parse task id from log {path}")
            continue
        observed_tasks.append(int(match.group(1)))
        text = path.read_text(encoding="utf-8", errors="replace")
        if TERMINAL_LOG_MARKER not in text:
            problems.append(f"payload log lacks terminal marker: {path}")
    if sorted(observed_tasks) != list(range(1, 71)):
        problems.append(f"payload log task coverage changed: {sorted(observed_tasks)}")
    return problems


def _required_artifact_prefixes(method: str) -> tuple[str, ...]:
    common = ("cell_result_", "eval_", "prompt_contract_", "passk_")
    if method == "base":
        return common
    return (*common, "checkpoint_eval_", "training_diagnostics_", "traj_")


def _validate_receipts(config: dict, result_root: Path) -> list[str]:
    problems = []
    expected = _expected_coordinates(config)
    receipt_paths = sorted(result_root.glob("complete_*.json"))
    if len(receipt_paths) != 70:
        problems.append(f"expected 70 completion receipts, found {len(receipt_paths)}")
    fingerprints = set()
    trained_adapters = 0
    for coordinate in expected:
        method = coordinate["method"]
        seed = coordinate["seed"]
        tag = coordinate["tag"]
        receipt = result_root / f"complete_gsm8k__{tag}__{method}_s{seed}.json"
        if not receipt.is_file():
            problems.append(f"missing completion receipt: {receipt}")
            continue
        try:
            payload = validate_completion_receipt(receipt, result_root=result_root)
            validate_receipt_identity(
                payload,
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
            problems.append(f"invalid completion receipt {receipt}: {exc}")
            continue
        fingerprint = str(payload.get("cell_fingerprint") or "")
        if not fingerprint or fingerprint in fingerprints:
            problems.append(f"missing or duplicate cell fingerprint: {receipt}")
        fingerprints.add(fingerprint)
        artifacts = {
            str(record.get("path")): record for record in payload.get("artifacts") or []
        }
        adapter_prefix = f"adapter_gsm8k__{tag}__{method}_s{seed}/"
        adapter_records = [
            path for path in artifacts if path.startswith(adapter_prefix)
        ]
        if method == "base":
            if adapter_records:
                problems.append(f"frozen base unexpectedly saved an adapter: {receipt}")
        else:
            trained_adapters += 1
            weights = {
                f"{adapter_prefix}adapter_model.safetensors",
                f"{adapter_prefix}adapter_model.bin",
            }
            if not any(path in artifacts for path in weights):
                problems.append(f"trained cell lacks receipt-bound adapter weights: {receipt}")
            if f"{adapter_prefix}adapter_config.json" not in artifacts:
                problems.append(f"trained cell lacks receipt-bound adapter config: {receipt}")
        for prefix in _required_artifact_prefixes(method):
            if not any(Path(path).name.startswith(prefix) for path in artifacts):
                problems.append(f"{receipt}: no receipt artifact starts with {prefix}")
    if trained_adapters != 63:
        problems.append(f"expected 63 trained adapters, found {trained_adapters}")
    return problems


def validate_confirmation(
    config_path: Path,
    *,
    log_glob: str,
    expected_commit: str,
    expected_config_sha256: str,
    source_job_id: str,
) -> tuple[list[str], dict]:
    problems = []
    config_bytes = config_path.read_bytes()
    observed_sha = hashlib.sha256(config_bytes).hexdigest()
    if observed_sha != expected_config_sha256:
        problems.append(
            f"configuration SHA mismatch: expected {expected_config_sha256}, found {observed_sha}"
        )
    config = yaml.safe_load(config_bytes)
    if config != build_payload():
        problems.append("configuration differs from the generated frozen payload")
    if not re.fullmatch(r"[0-9a-f]{7,40}", expected_commit):
        problems.append(f"invalid execution commit {expected_commit!r}")
    if not source_job_id.isdigit():
        problems.append(f"invalid source job id {source_job_id!r}")
    problems.extend(validate_yaml_outputs(config_path, log_glob))
    problems.extend(_validate_logs(log_glob))
    result_root = Path(str((config.get("defaults") or {}).get("out"))).expanduser()
    problems.extend(_validate_receipts(config, result_root))
    marker = {
        "schema_version": 1,
        "status": "ok",
        "run_id": RUN_ID,
        "execution_commit": expected_commit,
        "configuration_sha256": observed_sha,
        "source_job_id": source_job_id,
        "task_count": 70,
        "trained_adapter_count": 63,
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
    problems, marker = validate_confirmation(
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
