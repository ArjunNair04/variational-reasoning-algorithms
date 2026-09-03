"""Outcome-independent tests for the Q5 support-follow-up analyzer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.analyze_qwen3_q5_support_followup import (
    CELL_ORDER,
    CONTROL_CELLS,
    METRICS,
    SEEDS,
    _inner_steps,
    paired_contrasts,
    validate_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "lm_study" / "experiments_qwen3_17b_q5_support_followup.yaml"


def test_design_only_validation_accepts_generated_yaml() -> None:
    config, cells = validate_design(CONFIG)
    assert len(cells) == len(CELL_ORDER)
    assert tuple(config["defaults"]["seed_values"]) == SEEDS


def test_inner_steps_uses_persisted_nested_schema() -> None:
    steps = [{"inner_step": 0, "mstep_sampling": {"mode": "full_posterior"}}]
    assert _inner_steps({"inner_m_step": {"steps": steps}}) is steps


def test_paired_contrasts_cover_all_registered_comparisons() -> None:
    controls = [
        {"cell": cell, "seed": seed, **{metric: 0.5 for metric in METRICS}}
        for cell in CONTROL_CELLS
        for seed in SEEDS
    ]
    followups = [
        {"cell": cell, "seed": seed, **{metric: 0.51 for metric in METRICS}}
        for cell in CELL_ORDER
        for seed in SEEDS
    ]
    frame = paired_contrasts(
        pd.DataFrame(followups), pd.DataFrame(controls), draws=100, seed=17
    )
    assert len(frame) == 3 * len(METRICS)
    assert set(frame["metric"]) == set(METRICS)
    assert set(frame["contrast"]) == {
        "s64_vs_s32_full",
        "topres15_vs_full",
        "topres15_vs_categorical16",
    }
