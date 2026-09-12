"""Outcome-independent tests for the frozen JEPO analyzer."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from analysis.analyze_qwen3_jepo_comparator import (
    CELL_ID,
    SEEDS,
    _validate_evaluation,
    load_controls,
    summarize,
    validate_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "lm_study" / "experiments_qwen3_17b_jepo_comparator.yaml"


def _evaluation_payload(question_ids: list[int]) -> dict:
    return {
        "eval_official_test_accessed": False,
        "eval_source_split": "train",
        "eval_dataset_splits_loaded": ["train"],
        "test_acc_legacy": 0.0,
        "test_acc_strict": 0.0,
        "records": [
            {
                "idx": idx,
                "legacy_correct": False,
                "strict_correct": False,
                "official_test_accessed": False,
            }
            for idx in question_ids
        ],
    }


def test_validation_support_ignores_order_but_not_membership(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    ids = list(range(400))
    path.write_text(json.dumps(_evaluation_payload(ids)), encoding="utf-8")
    original = _validate_evaluation(path)
    path.write_text(json.dumps(_evaluation_payload(ids[::-1])), encoding="utf-8")
    assert _validate_evaluation(path) == original
    path.write_text(
        json.dumps(_evaluation_payload([*ids[:-1], 400])), encoding="utf-8"
    )
    assert _validate_evaluation(path)[2] != original[2]


def test_validation_support_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(_evaluation_payload([*range(399), 0])), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid validation question identities"):
        _validate_evaluation(path)


def test_design_only_validation_accepts_generated_yaml() -> None:
    config, cell = validate_design(CONFIG)
    assert config["run_id"] == "4124d5c8"
    assert cell.tag.endswith(f"_{CELL_ID}")


def test_control_loader_requires_exact_paired_seed_family(tmp_path: Path) -> None:
    path = tmp_path / "seed_metrics.csv"
    pd.DataFrame(
        [
            {
                "cell": "CTRL-base",
                "seed": seed,
                "final_extracted": 0.75,
                "final_strict": 0.60,
            }
            for seed in SEEDS
        ]
    ).to_csv(path, index=False)
    controls = load_controls(path)
    assert tuple(controls.index) == SEEDS


def test_summary_reports_paired_final_metrics_first() -> None:
    seeds = pd.DataFrame(
        [
            {
                "cell": CELL_ID,
                "seed": seed,
                "final_extracted": 0.80,
                "final_strict": 0.75,
                "extracted_auc": 0.78,
                "strict_auc": 0.71,
                "train_llm_gen": 2048,
                "optimizer_steps": 32,
                "generated_tokens": 100,
                "backward_tokens": 100,
                "accelerator_hours": 1.0,
            }
            for seed in SEEDS
        ]
    )
    controls = pd.DataFrame(
        {
            "final_extracted": [0.75] * len(SEEDS),
            "final_strict": [0.60] * len(SEEDS),
        },
        index=SEEDS,
    )
    summary, contrasts = summarize(seeds, controls)
    assert float(summary.iloc[0]["final_extracted"]) == pytest.approx(0.80)
    assert tuple(contrasts["metric"]) == ("final_extracted", "final_strict")
    assert tuple(round(value, 6) for value in contrasts["mean_difference_pp"]) == (
        5.0,
        15.0,
    )
