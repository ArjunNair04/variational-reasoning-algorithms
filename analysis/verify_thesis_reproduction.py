#!/usr/bin/env python3
"""Fail-closed thesis evidence gate for the posterity replays."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/reproducibility_manifest.json"
LM_STUDY = ROOT / "lm_study"
if str(LM_STUDY) not in sys.path:
    sys.path.insert(0, str(LM_STUDY))
FAILURE_RE = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemoryError|No space left on device|"
    r"Disk quota exceeded|FAILED",
    re.IGNORECASE,
)
REQUIRED_ENVIRONMENT_FIELDS = (
    "python",
    "commit",
    "torch",
    "transformers",
    "peft",
    "gpu",
    "cuda_runtime",
    "cudnn",
    "nvidia_driver",
)
METRIC_ORDER = (
    "Final Acc@1",
    "Strict final accuracy",
    "Normalized trajectory AUC",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str | Path, *, root: Path = ROOT) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported reproducibility manifest schema")
    if tuple(manifest.get("citation_policy", {}).get("metric_order", ())) != METRIC_ORDER:
        raise ValueError("thesis metric reporting order changed")
    studies = manifest.get("studies")
    if not isinstance(studies, dict) or set(studies) != {
        "selected_methods",
        "temperature_mixture",
    }:
        raise ValueError("posterity study set changed")
    return manifest


def validate_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    from methods_lm import MODELS, MODEL_REVISIONS
    from tasks import GSM8K_DATASET_REVISION

    inputs = dict(manifest["immutable_inputs"])
    model = dict(inputs["model"])
    dataset = dict(inputs["dataset"])
    preset = str(model["preset"])
    observed_model_id = MODELS[preset][0]
    observed_revision = MODEL_REVISIONS[preset]
    if observed_model_id != model["model_id"]:
        raise ValueError("model ID differs from the frozen manifest")
    if observed_revision != model["revision"]:
        raise ValueError("model revision differs from the frozen manifest")
    if GSM8K_DATASET_REVISION != dataset["revision"]:
        raise ValueError("GSM8K revision differs from the frozen manifest")
    if dataset.get("official_test_used") is not False:
        raise ValueError("manifest does not prohibit official-test use")
    runtime = dict(manifest["runtime_environment"])
    lock_path = _resolve(runtime["lock_file"])
    if _sha256(lock_path) != runtime["lock_sha256"]:
        raise ValueError("runtime environment lock checksum changed")

    checked = {}
    for study_name, raw in manifest["studies"].items():
        study = dict(raw)
        config_path = _resolve(study["config"])
        observed_hash = _sha256(config_path)
        if observed_hash != study["config_sha256"]:
            raise ValueError(f"{study_name}: configuration checksum changed")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if str(config.get("run_id")) != study["run_id"]:
            raise ValueError(f"{study_name}: run ID changed")
        defaults = dict(config.get("defaults") or {})
        if defaults.get("model") != preset:
            raise ValueError(f"{study_name}: model preset changed")
        if defaults.get("eval_partition") != "validation":
            raise ValueError(f"{study_name}: evaluation partition changed")
        diagnostic = dict(config.get("diagnostic") or {})
        partitions = dict(diagnostic.get("partitions") or {})
        fixed = dict(diagnostic.get("fixed_contract") or {})
        official_test = partitions.get(
            "official_test_used", fixed.get("official_test_used")
        )
        if official_test is not False:
            raise ValueError(f"{study_name}: official-test prohibition is missing")
        seeds = tuple(int(value) for value in defaults.get("seed_values", ()))
        if seeds != tuple(study["seed_values"]):
            raise ValueError(f"{study_name}: paired seeds changed")
        expected_tasks = int(study["cell_count"]) * len(seeds)
        if expected_tasks != int(study["expected_tasks"]):
            raise ValueError(f"{study_name}: task expansion changed")
        checked[study_name] = {
            "run_id": study["run_id"],
            "configuration_sha256": observed_hash,
            "tasks": expected_tasks,
            "seeds": list(seeds),
        }
    return {
        "model_id": observed_model_id,
        "model_revision": observed_revision,
        "dataset_revision": GSM8K_DATASET_REVISION,
        "environment_lock_sha256": runtime["lock_sha256"],
        "studies": checked,
    }


def _exact_logs(pattern: str, expected_tasks: int) -> list[Path]:
    paths = [Path(value) for value in sorted(glob.glob(pattern))]
    if len(paths) != expected_tasks:
        raise ValueError(
            f"expected {expected_tasks} task logs, found {len(paths)} for {pattern}"
        )
    observed = []
    for path in paths:
        match = re.search(r"\.([0-9]+)\.log$", path.name)
        if not match:
            raise ValueError(f"cannot parse task ID from {path}")
        observed.append(int(match.group(1)))
        text = path.read_text(encoding="utf-8", errors="replace")
        if FAILURE_RE.search(text):
            raise ValueError(f"failure signature found in {path}")
        if "=== done in" not in text:
            raise ValueError(f"terminal payload marker missing from {path}")
    if sorted(observed) != list(range(1, expected_tasks + 1)):
        raise ValueError(f"task-log coverage is incomplete: {sorted(observed)}")
    return paths


def _environment_from_csv(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        try:
            row = next(rows)
        except StopIteration as exc:
            raise ValueError(f"empty sweep CSV: {path}") from exc
    raw = row.get("params")
    if not raw:
        raise ValueError(f"sweep CSV lacks params provenance: {path}")
    params = json.loads(raw)
    environment = params.get("env")
    if not isinstance(environment, dict):
        raise ValueError(f"sweep CSV lacks environment provenance: {path}")
    missing = [key for key in REQUIRED_ENVIRONMENT_FIELDS if environment.get(key) in (None, "")]
    if missing:
        raise ValueError(f"{path}: incomplete runtime provenance: {missing}")
    return environment


def audit_runtime_provenance(
    artifact_dir: Path,
    *,
    expected_tasks: int,
    execution_revision: str,
    expected_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipts = sorted(artifact_dir.glob("complete_*.json"))
    if len(receipts) != expected_tasks:
        raise ValueError(
            f"expected {expected_tasks} completion receipts, found {len(receipts)}"
        )
    environments = []
    for receipt_path in receipts:
        receipt = _read_json(receipt_path)
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError(f"{receipt_path}: artifacts are missing")
        csv_paths = [
            artifact_dir / str(record.get("path"))
            for record in artifacts
            if Path(str(record.get("path", ""))).name.startswith("sweep_")
            and str(record.get("path", "")).endswith(".csv")
        ]
        if len(csv_paths) != 1:
            raise ValueError(f"{receipt_path}: expected one receipt-bound sweep CSV")
        environments.append(_environment_from_csv(csv_paths[0]))

    stable_fields = (
        "python",
        "commit",
        "torch",
        "transformers",
        "peft",
        "gpu",
        "cuda_runtime",
        "cudnn",
        "nvidia_driver",
    )
    unique = {
        key: sorted({str(environment[key]) for environment in environments})
        for key in stable_fields
    }
    changed = {key: values for key, values in unique.items() if len(values) != 1}
    if changed:
        raise ValueError(f"runtime environment changed across tasks: {changed}")
    if unique["commit"] != [execution_revision[:7]]:
        raise ValueError(
            f"runtime commit {unique['commit']} does not match {execution_revision[:7]}"
        )
    if not all("H100" in str(environment["gpu"]) for environment in environments):
        raise ValueError("one or more tasks did not record an H100 runtime")
    observed = {key: values[0] for key, values in unique.items()}
    if expected_environment is not None:
        for field in ("python", "torch", "transformers", "peft"):
            expected = str(expected_environment[field])
            if observed[field] != expected:
                raise ValueError(
                    f"runtime {field}={observed[field]!r}, expected {expected!r}"
                )
        accelerator = str(expected_environment["accelerator_family"])
        if accelerator not in observed["gpu"]:
            raise ValueError(
                f"runtime GPU {observed['gpu']!r} is not {accelerator!r}"
            )
    return observed


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _analysis_command(
    study_name: str,
    study: Mapping[str, Any],
    *,
    artifact_dir: Path,
    marker: Path,
    log_dir: Path,
    output_dir: Path,
    source_summary: Path | None,
) -> list[str]:
    config = _resolve(study["config"])
    analyzer = _resolve(study["analyzer"])
    if study_name == "selected_methods":
        command = [
            sys.executable,
            str(analyzer),
            "--config",
            str(config),
            "--artifact-dir",
            str(artifact_dir),
            "--marker",
            str(marker),
            "--expected-commit",
            str(study["execution_revision"]),
            "--expected-source-job",
            str(study["payload_job"]),
            "--output-dir",
            str(output_dir),
        ]
        if source_summary is not None:
            command.extend(["--source-summary", str(source_summary)])
        return command
    return [
        sys.executable,
        str(analyzer),
        "--config",
        str(config),
        "--artifact-dir",
        str(artifact_dir),
        "--marker",
        str(marker),
        "--log-dir",
        str(log_dir),
        "--out-dir",
        str(output_dir),
        "--run-id",
        str(study["run_id"]),
        "--expected-commit",
        str(study["execution_revision"]),
        "--source-job",
        str(study["payload_job"]),
        "--config-sha256",
        str(study["config_sha256"]),
        "--tag-prefix",
        "q3_l2r_temperature_mixture",
        "--log-stem",
        "qwen3_17b_pis_temperature_posterity",
    ]


def _finite_metrics(output_dir: Path, study_name: str) -> list[str]:
    if study_name == "selected_methods":
        path = output_dir / "method_summary.csv"
        fields = ("final_extracted", "final_strict", "extracted_auc")
    else:
        path = output_dir / "seed_summary.csv"
        fields = ("final_extracted", "final_strict", "extracted_auc")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"analysis summary is empty: {path}")
    for row in rows:
        for field in fields:
            value = float(row[field])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{path}: invalid {field}={value}")
    return list(fields)


def _output_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(output_dir).as_posix(): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--study",
        choices=("selected_methods", "temperature_mixture"),
        required=True,
    )
    parser.add_argument("--validate-contract-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-summary", type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    for study in manifest["studies"].values():
        study["execution_revision"] = manifest["repository"]["execution_revision"]
    contract = validate_contract(manifest)
    if args.validate_contract_only:
        print(json.dumps(contract, sort_keys=True))
        return 0

    study = manifest["studies"][args.study]
    required = {
        "artifact_dir": args.artifact_dir,
        "marker": args.marker,
        "log_dir": args.log_dir,
        "output_dir": args.output_dir,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        parser.error(f"full evidence gate requires: {', '.join(missing)}")
    if args.output_dir.exists():
        raise ValueError(f"evidence output already exists: {args.output_dir}")

    expected_tasks = int(study["expected_tasks"])
    pattern = str(args.log_dir / Path(str(study["log_glob"])).name)
    logs = _exact_logs(pattern, expected_tasks)
    environment = audit_runtime_provenance(
        args.artifact_dir,
        expected_tasks=expected_tasks,
        execution_revision=manifest["repository"]["execution_revision"],
        expected_environment=manifest["runtime_environment"],
    )
    _run(
        _analysis_command(
            args.study,
            study,
            artifact_dir=args.artifact_dir,
            marker=args.marker,
            log_dir=args.log_dir,
            output_dir=args.output_dir,
            source_summary=args.source_summary,
        )
    )
    metric_fields = _finite_metrics(args.output_dir, args.study)
    source_claim_ready = (
        args.study == "selected_methods" and args.source_summary is not None
    )
    evidence = {
        "schema_version": 1,
        "status": "citable_posterity_replay",
        "source_reproduction_claim_ready": source_claim_ready,
        "study": args.study,
        "registry_id": study["registry_id"],
        "run_id": study["run_id"],
        "execution_revision": manifest["repository"]["execution_revision"],
        "configuration_sha256": study["config_sha256"],
        "model": manifest["immutable_inputs"]["model"],
        "dataset": manifest["immutable_inputs"]["dataset"],
        "payload_job": study["payload_job"],
        "validator_job": study["validator_job"],
        "task_count": len(logs),
        "runtime_environment": environment,
        "environment_lock": {
            "path": manifest["runtime_environment"]["lock_file"],
            "sha256": manifest["runtime_environment"]["lock_sha256"],
        },
        "metric_order": list(METRIC_ORDER),
        "metric_fields": metric_fields,
        "validator_marker_sha256": _sha256(args.marker),
        "analysis_artifacts": _output_hashes(args.output_dir),
        "limitations": [
            "The fixed validation set and seeds intentionally match earlier development evidence.",
            "This receipt does not license an official-test or fresh-generalization claim.",
            (
                "Historical source-result reproduction is checksum-comparable."
                if source_claim_ready
                else "Historical source-result reproduction remains unclaimed because no source summary was supplied."
            ),
        ],
    }
    receipt = args.output_dir / "THESIS_EVIDENCE.json"
    receipt.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
