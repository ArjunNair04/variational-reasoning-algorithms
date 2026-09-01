"""Outcome-independent tests for the posterity replay analyser."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.analyze_qwen3_selected_method_posterity import (
    BASE_CELL,
    CELL_ORDER,
    Q5_ANSWER_FROZEN,
    Q5_ANSWER_MOVING,
    SEEDS,
    compare_source_summary,
    validate_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "lm_study" / "experiments_qwen3_17b_selected_method_posterity.yaml"


def test_design_only_validation_accepts_frozen_yaml() -> None:
    config, cells = validate_design(CONFIG)
    assert tuple(cell.cell_id for cell in cells) == CELL_ORDER
    assert tuple(config["defaults"]["seed_values"]) == SEEDS
    assert Q5_ANSWER_MOVING in CELL_ORDER and Q5_ANSWER_FROZEN in CELL_ORDER


def test_source_summary_is_optional() -> None:
    replay = pd.DataFrame(
        [
            {
                "cell": cell,
                "final_extracted": 0.5,
                "final_strict": 0.4,
                "extracted_auc": 0.45,
                "strict_auc": 0.35,
            }
            for cell in CELL_ORDER
        ]
    )
    empty = compare_source_summary(replay, None)
    assert empty.empty
    assert "replay_minus_source" in empty.columns
    assert BASE_CELL in set(replay["cell"])
