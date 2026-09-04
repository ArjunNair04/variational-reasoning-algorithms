#!/usr/bin/env python3
"""Fail closed over every panel-evaluation task and write one status marker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import yaml


FAILURE = re.compile(r"Traceback|CUDA out of memory|Killed|No space left|ERROR:")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--log-glob", required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--source-job-id", required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    from generate_qwen3_17b_gsm8k_panel_robustness import validate_payload

    validate_payload(config)
    if hashlib.sha256(config_bytes).hexdigest() != args.expected_config_sha256:
        raise SystemExit("configuration hash mismatch")
    logs = (
        sorted(Path().glob(args.log_glob))
        if not Path(args.log_glob).is_absolute()
        else sorted(Path(args.log_glob).parent.glob(Path(args.log_glob).name))
    )
    if len(logs) != 42:
        raise SystemExit(f"expected 42 task logs, found {len(logs)}")
    seen = set()
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"PANEL_EVALUATION_COMPLETE task=(\d+)", text)
        if match is None or FAILURE.search(text):
            raise SystemExit(f"failed or incomplete log: {path}")
        seen.add(int(match.group(1)))
    if seen != set(range(1, 43)):
        raise SystemExit("task-log coverage mismatch")
    import sys

    analysis_root = Path(__file__).resolve().parents[1] / "analysis"
    sys.path.insert(0, str(analysis_root))
    from analyze_qwen3_gsm8k_panel_robustness import load_and_validate

    _config, _methods, _seeds, _panels, _records, receipts = load_and_validate(
        args.config, args.result_root
    )
    if any(row.get("execution_commit") != args.expected_commit for row in receipts):
        raise SystemExit("receipt execution commit mismatch")
    marker = {
        "status": "ok",
        "run_id": config["run_id"],
        "source_run_id": config["source_run_id"],
        "task_count": 42,
        "panel_count": 3,
        "record_count": 42 * 3 * 400,
        "official_test_used": False,
        "training_performed": False,
        "execution_commit": args.expected_commit,
        "configuration_sha256": args.expected_config_sha256,
        "source_job_id": args.source_job_id,
    }
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.marker.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    print(json.dumps(marker, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
