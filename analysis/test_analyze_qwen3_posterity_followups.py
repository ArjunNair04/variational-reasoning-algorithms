"""Outcome-independent tests for the posterity follow-up analyzer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.analyze_qwen3_posterity_followups import (
    CELL_ORDER,
    METRICS,
    SEEDS,
    paired_contrasts,
    validate_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "lm_study" / "experiments_qwen3_17b_posterity_followups.yaml"


def test_design_only_validation_accepts_generated_yaml() -> None:
    config, cells = validate_design(CONFIG)
    assert len(cells) == len(CELL_ORDER)
    assert tuple(config["defaults"]["seed_values"]) == SEEDS


def test_paired_contrasts_cover_all_registered_families() -> None:
    controls = []
    for cell in ("CTRL-base", "Q5-AD-M-LR1e-5-U1-K16", "PIS-Q-S8-B8-U4"):
        for seed in SEEDS:
            controls.append({"cell": cell, "seed": seed, **{metric: 0.5 for metric in METRICS}})
    followups = []
    for cell in CELL_ORDER:
        for seed in SEEDS:
            followups.append({"cell": cell, "seed": seed, **{metric: 0.51 for metric in METRICS}})
    frame = paired_contrasts(
        pd.DataFrame(followups),
        pd.DataFrame(controls),
        draws=100,
        bootstrap_seed=17,
    )
    assert len(frame) == 7 * len(METRICS)
    assert set(frame["metric"]) == set(METRICS)
    assert "exact_signed_reuse_interaction" in set(frame["family"])
