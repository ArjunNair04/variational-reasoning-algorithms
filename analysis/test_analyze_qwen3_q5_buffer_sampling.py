"""Outcome-independent tests for the Q5 buffer-sampling analyzer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.analyze_qwen3_q5_buffer_sampling import (
    CELL_ORDER,
    METRICS,
    SEEDS,
    _inner_steps,
    paired_contrasts,
    validate_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "lm_study" / "experiments_qwen3_17b_q5_buffer_sampling.yaml"


def test_design_only_validation_accepts_generated_yaml() -> None:
    config, cells = validate_design(CONFIG)
    assert len(cells) == len(CELL_ORDER)
    assert tuple(config["defaults"]["seed_values"]) == SEEDS


def test_inner_steps_uses_persisted_nested_schema() -> None:
    steps = [{"inner_step": 0, "mstep_sampling": {"mode": "full_posterior"}}]
    assert _inner_steps({"inner_m_step": {"steps": steps}}) is steps


def test_paired_contrasts_cover_all_registered_comparisons() -> None:
    controls = [
        {"cell": "Q5-MORE-S32-B16-U1", "seed": seed, **{metric: 0.5 for metric in METRICS}}
        for seed in SEEDS
    ]
    followups = [
        {"cell": cell, "seed": seed, **{metric: 0.51 for metric in METRICS}}
        for cell in CELL_ORDER
        for seed in SEEDS
    ]
    frame = paired_contrasts(
        pd.DataFrame(followups),
        pd.DataFrame(controls),
        draws=100,
        seed=17,
    )
    assert len(frame) == 3 * len(METRICS)
    assert set(frame["metric"]) == set(METRICS)
    assert set(frame["contrast"]) == {
        "b32_full_vs_b16_full",
        "b32_sample16_vs_b32_full",
        "b32_sample16_vs_b16_full",
    }
