"""Focused contracts for the answer-blind temperature-mixture experiment."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "lm_study" / "experiments_qwen3_17b_l2r_temperature_mixture.yaml"


def _payload():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_design_is_one_temperature_contrast_over_seven_fresh_seeds():
    payload = _payload()
    assert payload["run_id"] == "5c7a10e2"
    assert payload["diagnostic"]["design"] == {
        "cells": ["PIS-T1", "PIS-TMIX1.2"],
        "seed_values": [1481, 1483, 1487, 1489, 1499, 1511, 1523],
        "array_tasks": 14,
    }
    assert list(payload["algos"]) == ["L2R"]
    assert len(payload["algos"]["L2R"]) == 2


def test_only_support_proposal_changes():
    control, treatment = _payload()["algos"]["L2R"]
    changed = {
        key: (control.get(key), treatment.get(key))
        for key in set(control) | set(treatment)
        if control.get(key) != treatment.get(key)
    }
    assert changed == {
        "cell_id": ("PIS-T1", "PIS-TMIX1.2"),
        "proposal_mixture": ("single", "question_temperature"),
        "proposal_prior_fraction": (1.0, 0.5),
        "proposal_temperature": (1.0, 1.2),
        "responsibility_score": ("prior_corrected", "mixed_prior_corrected"),
    }


def test_primary_metric_and_official_test_boundary_are_fixed():
    diagnostic = _payload()["diagnostic"]
    assert diagnostic["reporting_order"][0] == "final_extracted_answer_accuracy"
    assert diagnostic["fixed_contract"]["official_test_used"] is False
    assert _payload()["defaults"]["eval_partition"] == "validation"
