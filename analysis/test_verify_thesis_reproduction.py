from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from verify_thesis_reproduction import (
    _environment_from_csv,
    audit_runtime_provenance,
    load_manifest,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/reproducibility_manifest.json"


def _environment(commit: str = "4a9a9ca") -> dict:
    return {
        "python": "3.11.0",
        "commit": commit,
        "torch": "2.6.0+cu124",
        "transformers": "4.51.3",
        "peft": "0.13.2",
        "gpu": "NVIDIA H100 80GB HBM3",
        "cuda_runtime": "12.4",
        "cudnn": 90100,
        "nvidia_driver": "550.54.15",
        "host": "seymour3",
    }


def _write_coordinate(root: Path, index: int, environment: dict) -> None:
    csv_path = root / f"sweep_{index}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["params"])
        writer.writeheader()
        writer.writerow({"params": json.dumps({"env": environment})})
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "identity": {"task": index},
        "artifacts": [{"path": csv_path.name}],
    }
    (root / f"complete_{index}.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )


def test_frozen_manifest_matches_runtime_constants_and_configs() -> None:
    manifest = load_manifest(MANIFEST)
    observed = validate_contract(manifest)
    assert observed["model_revision"] == "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    assert observed["dataset_revision"] == "740312add88f781978c0658806c59bc2815b9866"
    assert observed["environment_lock_sha256"] == "c9c2d77a54a807f40e5f7120403563e20b51c4c9ab6ab581d2b87163cbbb3942"
    assert observed["studies"]["selected_methods"]["tasks"] == 91
    assert observed["studies"]["temperature_mixture"]["tasks"] == 14


def test_environment_reader_rejects_missing_critical_versions(tmp_path: Path) -> None:
    path = tmp_path / "sweep.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["params"])
        writer.writeheader()
        writer.writerow({"params": json.dumps({"env": {"python": "3.11"}})})
    with pytest.raises(ValueError, match="incomplete runtime provenance"):
        _environment_from_csv(path)


def test_runtime_audit_requires_consistent_execution_environment(tmp_path: Path) -> None:
    _write_coordinate(tmp_path, 1, _environment())
    _write_coordinate(tmp_path, 2, _environment())
    observed = audit_runtime_provenance(
        tmp_path,
        expected_tasks=2,
        execution_revision="4a9a9cafc6b17cc18e7c1da8a422e847daeb4746",
        expected_environment=load_manifest(MANIFEST)["runtime_environment"],
    )
    assert observed["gpu"] == "NVIDIA H100 80GB HBM3"


def test_runtime_audit_rejects_mixed_package_versions(tmp_path: Path) -> None:
    first = _environment()
    second = _environment()
    second["transformers"] = "4.52.0"
    _write_coordinate(tmp_path, 1, first)
    _write_coordinate(tmp_path, 2, second)
    with pytest.raises(ValueError, match="runtime environment changed"):
        audit_runtime_provenance(
            tmp_path,
            expected_tasks=2,
            execution_revision="4a9a9cafc6b17cc18e7c1da8a422e847daeb4746",
            expected_environment=load_manifest(MANIFEST)["runtime_environment"],
        )
