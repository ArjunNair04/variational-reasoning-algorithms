from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.analyze_qwen3_final_method_confirmation import (
    BASE_CELL,
    BOOTSTRAP_METRICS,
    CELL_ORDER,
    CellSpec,
    Coordinate,
    Q5_CELL,
    SEEDS,
    _normalized_question_auc,
    _rank,
    _require_resource_contract,
    _resource_totals,
    _validate_lora_surface,
    _validate_prompt_contract,
    hierarchical_method_bootstrap,
    load_design,
    verify_marker,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "lm_study/experiments_qwen3_17b_final_method_confirmation.yaml"


def test_registered_design_has_ten_cells_and_seven_seeds() -> None:
    payload, cells = load_design(CONFIG)
    assert [cell.cell_id for cell in cells] == list(CELL_ORDER)
    assert tuple(payload["defaults"]["seed_values"]) == SEEDS
    assert payload["diagnostic"]["partitions"]["official_test_used"] is False


def test_question_auc_uses_legacy_numpy_compatible_trapezoid() -> None:
    rounds = (0, 1, 2, 4, 8, 16, 24, 32)
    values = np.asarray([[round_value / 32.0 for round_value in rounds]])
    observed = _normalized_question_auc(values, rounds)
    assert observed.shape == (1,)
    assert observed[0] == pytest.approx(0.5)


def test_question_auc_falls_back_for_beaker_numpy(monkeypatch) -> None:
    rounds = (0, 1, 2, 4, 8, 16, 24, 32)
    values = np.ones((1, len(rounds)))
    monkeypatch.delattr(np, "trapezoid")
    monkeypatch.setattr(
        np,
        "trapz",
        lambda observed, observed_rounds, axis: np.asarray([32.0]),
        raising=False,
    )
    assert _normalized_question_auc(values, rounds)[0] == 1.0


def _prompt_payload(mode: str, *, hint: bool) -> dict:
    return {
        "schema_version": 1,
        "tag": "tag_seed1201",
        "training_seed": 1201,
        "proposal_prompt_mode": mode,
        "answer_event_mode": "strict_terminal_marker",
        "rows": [
            {
                "dataset_train_index": index,
                "canonical_prompt_sha256": f"{index:064x}",
                "canonical_contains_proposal_answer_hint": False,
                "proposal_contains_gold_answer_hint": hint,
            }
            for index in range(128)
        ],
    }


def test_prompt_contract_allows_answer_hint_only_for_q5_proposals() -> None:
    q5 = Coordinate(
        CellSpec(Q5_CELL, "AC-ALG1", {}, "tag"),
        1201,
    )
    pis = Coordinate(
        CellSpec("PIS-S8-B8-U4", "AC-ALG1", {}, "tag"),
        1201,
    )
    assert len(_validate_prompt_contract(_prompt_payload("answer_derive", hint=True), q5)) == 128
    assert len(_validate_prompt_contract(_prompt_payload("question", hint=False), pis)) == 128
    with pytest.raises(ValueError, match="proposal hint mismatch"):
        _validate_prompt_contract(_prompt_payload("answer_derive", hint=False), q5)
    with pytest.raises(ValueError, match="proposal hint mismatch"):
        _validate_prompt_contract(_prompt_payload("question", hint=True), pis)


def test_lora_surface_check_exempts_only_frozen_base(tmp_path) -> None:
    context = tmp_path / "cell_result.json"
    attention_only = {
        "lora_target_set": "attention",
        "lora_target_modules": {
            "qwen3-1.7b-base": ["q_proj", "k_proj", "v_proj", "o_proj"]
        },
    }
    _validate_lora_surface(
        cell_id=BASE_CELL,
        sweep=attention_only,
        context=context,
    )
    with pytest.raises(ValueError, match="LoRA surface changed"):
        _validate_lora_surface(
            cell_id=Q5_CELL,
            sweep=attention_only,
            context=context,
        )


def test_resource_totals_reconstruct_posterior_tokens_from_diagnostics() -> None:
    diagnostics = [
        {"compute": {"tokens": {"generated": 10, "backward": 4, "scored": 7}}},
        {"compute": {"tokens": {"generated": 13, "backward": 5, "scored": 8}}},
    ]
    assert _resource_totals(
        "AC-ALG1",
        {
            "generated_tokens": None,
            "backward_tokens": None,
            "teacher_forced_scoring_tokens": None,
        },
        diagnostics,
    ) == (23.0, 9.0, 15.0)
    assert _resource_totals("base", {}, []) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("cell", "method", "axes", "result"),
    [
        (BASE_CELL, "base", {}, {"train_llm_gen": 0, "optimizer_steps": None}),
        (
            Q5_CELL,
            "AC-ALG1",
            {"batch": 64, "iters": 1},
            {"train_llm_gen": 2048, "optimizer_steps": 32},
        ),
        (
            "PIS-S8-B8-U4",
            "AC-ALG1",
            {"batch": 64, "iters": 4},
            {"train_llm_gen": 2048, "optimizer_steps": 128},
        ),
        (
            "GOLD-LR3e-6-E2",
            "Gold-CoT-SFT",
            {"epochs": 2},
            {"train_llm_gen": 0, "optimizer_steps": 64},
        ),
        (
            "TRICE-LR1e-4-CV",
            "TRICE",
            {"batch": 64},
            {"train_llm_gen": 2176, "optimizer_steps": 32},
        ),
        (
            "GRPO-S16-B4-U4",
            "GRPO",
            {"batch": 64},
            {"train_llm_gen": 2048, "optimizer_steps": 128},
        ),
        (
            "RLOO-S16-B8-U4",
            "RLOO",
            {"batch": 128},
            {"train_llm_gen": 4096, "optimizer_steps": 128},
        ),
    ],
)
def test_resource_contracts_fail_closed(cell, method, axes, result) -> None:
    coordinate = Coordinate(CellSpec(cell, method, axes, "tag"), 1201)
    _require_resource_contract(coordinate, result)
    broken = dict(result, train_llm_gen=int(result["train_llm_gen"]) + 1)
    with pytest.raises(ValueError):
        _require_resource_contract(coordinate, broken)


def test_nested_bootstrap_preserves_identical_method_equality() -> None:
    cube = np.ones(
        (len(CELL_ORDER), len(SEEDS), 400, len(BOOTSTRAP_METRICS)),
        dtype=float,
    )
    first = hierarchical_method_bootstrap(cube, draws=32, seed=7, chunk_size=8)
    second = hierarchical_method_bootstrap(cube, draws=32, seed=7, chunk_size=8)
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.ones_like(first))


def test_ranking_uses_registered_metric_order() -> None:
    summary = pd.DataFrame(
        [
            {"cell": CELL_ORDER[1], "final_extracted": 0.8, "final_strict": 0.7, "extracted_auc": 0.9},
            {"cell": CELL_ORDER[2], "final_extracted": 0.8, "final_strict": 0.71, "extracted_auc": 0.1},
        ]
    )
    assert _rank(summary, (CELL_ORDER[1], CELL_ORDER[2])) == [
        CELL_ORDER[2],
        CELL_ORDER[1],
    ]


def test_marker_is_bound_to_commit_config_job_and_official_test_non_access(tmp_path) -> None:
    import hashlib

    commit = "a" * 40
    marker = {
        "schema_version": 1,
        "status": "ok",
        "run_id": "978b99c8",
        "execution_commit": commit,
        "configuration_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "source_job_id": "12345",
        "task_count": 70,
        "trained_adapter_count": 63,
        "official_test_used": False,
    }
    path = tmp_path / "marker.json"
    path.write_text(json.dumps(marker), encoding="utf-8")
    assert verify_marker(
        path,
        CONFIG,
        expected_commit=commit,
        expected_source_job="12345",
    )["official_test_used"] is False
    marker["official_test_used"] = True
    path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="validator marker mismatch"):
        verify_marker(
            path,
            CONFIG,
            expected_commit=commit,
            expected_source_job="12345",
        )
