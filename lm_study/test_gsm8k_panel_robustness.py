"""Design-lock tests for the evaluation-only fresh-panel study."""

from __future__ import annotations

from generate_qwen3_17b_gsm8k_panel_robustness import (
    METHODS,
    PANEL_COUNT,
    PANEL_SIZE,
    SEEDS,
    build_payload,
    validate_payload,
)


def test_design_has_six_methods_seven_seeds_and_three_panels() -> None:
    payload = build_payload()
    validate_payload(payload)
    assert payload["design"]["array_tasks"] == len(METHODS) * len(SEEDS) == 42
    assert payload["design"]["training_performed"] is False
    assert payload["dataset"]["official_test_used"] is False
    assert len(payload["source_contract"]["panels"]) == PANEL_COUNT
    assert all(
        len(panel["dataset_train_indices"]) == PANEL_SIZE
        for panel in payload["source_contract"]["panels"]
    )


def test_panels_exclude_every_source_data_role() -> None:
    source = build_payload()["source_contract"]
    panels = {
        value for panel in source["panels"] for value in panel["dataset_train_indices"]
    }
    original = set(source["original_validation_indices"])
    shot_bank = {
        value for rows in source["shot_bank_indices_by_seed"].values() for value in rows
    }
    optimization = {
        value
        for rows in source["optimization_indices_by_seed"].values()
        for value in rows
    }
    assert len(panels) == PANEL_COUNT * PANEL_SIZE
    assert not panels & original
    assert not panels & shot_bank
    assert not panels & optimization


def test_only_declared_trained_methods_require_adapters() -> None:
    methods = build_payload()["methods"]
    assert methods[0] == {
        "method": "Frozen-base",
        "adapter_name_fragment": None,
        "trained": False,
    }
    assert all(row["trained"] for row in methods[1:])
    assert len(methods[1:]) * len(SEEDS) == 35
