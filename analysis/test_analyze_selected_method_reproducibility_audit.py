"""Tests for the same-seed posterity reproducibility audit."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.analyze_selected_method_reproducibility_audit import (
    METRICS,
    SEEDS,
    SOURCE_TO_REPLAY,
    _first_stage,
    build_audit_summary,
    compare_seed_metrics,
)


def _seed_frame(*, replay: bool) -> pd.DataFrame:
    rows = []
    cells = SOURCE_TO_REPLAY.values() if replay else SOURCE_TO_REPLAY
    for cell_index, cell in enumerate(cells):
        for seed_index, seed in enumerate(SEEDS):
            value = 0.70 + 0.001 * cell_index + 0.002 * seed_index
            if replay and cell == "PIS-Q-S8-B8-U4" and seed == 1213:
                value += 0.10
            rows.append(
                {
                    "cell": cell,
                    "seed": seed,
                    **{metric: value for metric in METRICS},
                }
            )
    return pd.DataFrame(rows)


def test_seed_comparison_selects_the_largest_divergence() -> None:
    seed_differences, method_differences = compare_seed_metrics(
        _seed_frame(replay=False),
        _seed_frame(replay=True),
    )
    training_paths = pd.DataFrame(
        [
            {
                "source_cell": cell,
                "seed": seed,
                "round": 0,
                "first_divergence_stage_this_round": "sampled_generation",
            }
            for cell in ("Q5-LR1e-5-U1-K16", "PIS-S8-B8-U4")
            for seed in SEEDS
        ]
    )
    summary = build_audit_summary(seed_differences, method_differences, training_paths)
    assert summary["sensitive_cell"] == "PIS-Q-S8-B8-U4"
    assert summary["sensitive_seed"] == 1213
    assert summary["sensitive_seed_final_difference_pp"] == pytest.approx(10.0)
    assert summary["repeatability_decision"] == {
        "new_gpu_repeat_required": False,
        "result_semantics": "statistical_not_bit_exact",
        "reason": (
            "The paired artifacts preserve validation support and isolate the first "
            "training-path divergence after matching question selection and sampled "
            "traces, so the preregistered no-compute gate is satisfied."
        ),
    }


def test_first_divergence_stage_is_ordered_by_pipeline() -> None:
    common = {
        "minibatch_equal": True,
        "sample_texts_equal": True,
        "responsibility_keys_equal": True,
        "maximum_responsibility_delta": 0.0,
        "parameter_drift_delta": 0.0,
    }
    assert _first_stage(**common) == "none_observed"
    assert _first_stage(**{**common, "parameter_drift_delta": 0.1}) == "optimizer_update"
    assert (
        _first_stage(**{**common, "maximum_responsibility_delta": 0.01})
        == "posterior_scoring"
    )
    assert (
        _first_stage(**{**common, "responsibility_keys_equal": False})
        == "posterior_support"
    )
    assert _first_stage(**{**common, "sample_texts_equal": False}) == "sampled_generation"
    assert _first_stage(**{**common, "minibatch_equal": False}) == "question_selection"
