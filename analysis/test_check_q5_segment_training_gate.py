import json
from pathlib import Path
from shutil import copyfile

import pytest
import yaml

import check_q5_segment_training_gate as gate
from generate_qwen3_17b_q5_prior_segments import RUN_ID, build_payload


def prepare(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(build_payload()))
    expected = json.loads((gate.EVIDENCE_DIR / "scoring_summary_2026-09-14.json").read_text())

    def revalidate(source, results, manifest, commit, job, logs, output):
        assert commit == expected["execution_commit"] and job == expected["job"]
        output.mkdir()
        (output / "summary.json").write_text(json.dumps(expected))

    monkeypatch.setattr(gate, "analyze", revalidate)
    return config, expected


def test_verified_evidence_passes_without_flipping_old_boolean(tmp_path, monkeypatch):
    config, expected = prepare(tmp_path, monkeypatch)
    assert expected["training_may_start"] is False
    result = gate.require_training_gate(config, tmp_path, tmp_path, tmp_path)
    assert result["training_run_id"] == RUN_ID
    assert result["segment_scope"] == "reasoning_marker_fixed"
    assert result["supports"] == 384 and result["traces"] == 2987


def test_receipt_revalidation_failure_blocks_training(tmp_path, monkeypatch):
    config, _ = prepare(tmp_path, monkeypatch)
    def fail(*args):
        raise ValueError("output checksum mismatch")
    monkeypatch.setattr(gate, "analyze", fail)
    with pytest.raises(ValueError, match="checksum"):
        gate.require_training_gate(config, tmp_path, tmp_path, tmp_path)


def test_changed_evidence_and_changed_design_are_rejected(tmp_path, monkeypatch):
    config, expected = prepare(tmp_path, monkeypatch)
    expected["receipts"][0]["scheduler_job"] = "123"
    with pytest.raises(ValueError, match="receipts"):
        gate.require_training_gate(config, tmp_path, tmp_path, tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in ("scoring_manifest.json", "scoring_summary_2026-09-14.json"):
        copyfile(gate.EVIDENCE_DIR / name, evidence / name)
    (evidence / "scoring_summary_2026-09-14.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        gate.require_training_gate(config, tmp_path, tmp_path, tmp_path, evidence)
    changed = build_payload()
    changed["diagnostic"]["segment_scope"] = "historical_prior"
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="design"):
        gate.require_training_gate(config, tmp_path, tmp_path, tmp_path)


def test_submission_gate_runs_in_preflight_too():
    shell = (Path(__file__).parents[1] / "lm_study/submit_qwen3_17b_q5_prior_segments_ucl.sh").read_text()
    assert shell.index("--check-training-gate") < shell.index('if [ "${PREFLIGHT_ONLY')
    assert "check_q5_segment_training_gate.py" in shell
