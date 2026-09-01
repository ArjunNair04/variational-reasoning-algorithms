from __future__ import annotations

import copy
import math

import pytest

from analysis.analyze_l2r_temperature_mixture import (
    CONTROL,
    ROOT,
    TREATMENT,
    _resource_counters,
    load_design,
    verify_logs,
    verify_mechanism,
)


def _diagnostic(cell: str, round_index: int) -> dict:
    treatment = cell == TREATMENT
    traces = []
    for pid in range(8):
        for replica in range(8):
            traces.append({
                "pid": pid,
                "replica": replica,
                "proposal_prompt": "question",
                "proposal_temperature": 1.2 if treatment and replica >= 4 else 1.0,
            })
    p1, p12 = -2.0, -2.4
    mixture = math.log(0.5 * math.exp(p1) + 0.5 * math.exp(p12))
    return {
        "schema_version": 2,
        "method_family": "l2r",
        "round": round_index,
        "configuration": {
            "proposal_mixture": "question_temperature" if treatment else "single",
            "proposal_temperature": 1.2 if treatment else 1.0,
        },
        "generation": {
            "generations": 64,
            "cumulative_generations": 64 * (round_index + 1),
            "resolved_initial": 7,
            "prior_generations": 32 if treatment else 64,
            "temperature_generations": 32 if treatment else 0,
        },
        "sampled_traces": traces,
        "posterior": {
            "question_count": 8,
            "ess_fraction": 0.5,
            "proposal_prior_posterior_mass": 0.55 if treatment else None,
        },
        "top_traces": ([{
            "policy_h_logp": p1,
            "answer_proposal_h_logp": p12,
            "proposal_mixture_logp": mixture,
            "proposal_log_importance_correction": p1 - mixture,
        }] if treatment else []),
    }


def test_frozen_design_loads() -> None:
    load_design(ROOT / "lm_study/experiments_qwen3_17b_l2r_temperature_mixture.yaml")


def _resources() -> dict[str, float]:
    return {
        "generated_tokens": 1000,
        "backward_tokens": 500,
        "optimizer_steps": 128,
        "accelerator_hours": 1.25,
        "gsteps": 128,
    }


def test_registered_resource_counters_are_finite_and_nonnegative() -> None:
    assert _resource_counters(_resources(), "cell") == {
        "generated_tokens": 1000.0,
        "backward_tokens": 500.0,
        "optimizer_steps": 128.0,
        "accelerator_hours": 1.25,
    }


@pytest.mark.parametrize(
    "field",
    (
        "generated_tokens",
        "backward_tokens",
        "optimizer_steps",
        "accelerator_hours",
    ),
)
@pytest.mark.parametrize("value", (None, math.nan, math.inf, -1))
def test_registered_resource_counter_rejects_invalid_value(
    field: str, value: float | None
) -> None:
    result = _resources()
    if value is None:
        result.pop(field)
    else:
        result[field] = value
    with pytest.raises(ValueError, match=field):
        _resource_counters(result, "cell")


@pytest.mark.parametrize("cell", [CONTROL, TREATMENT])
def test_mechanism_accepts_registered_allocation(cell: str) -> None:
    rows = [_diagnostic(cell, index) for index in range(32)]
    summary = verify_mechanism(cell, rows)
    assert summary["mean_ess_fraction"] == pytest.approx(0.5)


def test_mechanism_rejects_five_plus_three_temperature_split() -> None:
    rows = [_diagnostic(TREATMENT, index) for index in range(32)]
    broken = copy.deepcopy(rows)
    broken[0]["sampled_traces"][3]["proposal_temperature"] = 1.2
    with pytest.raises(ValueError, match="temperature allocation"):
        verify_mechanism(TREATMENT, broken)


def test_mechanism_rejects_wrong_mixture_density() -> None:
    rows = [_diagnostic(TREATMENT, index) for index in range(32)]
    rows[0]["top_traces"][0]["proposal_mixture_logp"] += 0.1
    with pytest.raises(ValueError, match="mixture log density"):
        verify_mechanism(TREATMENT, rows)


def _write_payload_logs(log_dir, epilog: str) -> None:
    for task in range(1, 15):
        (log_dir / f"qwen3_17b_l2r_temperature_mixture.7224195.{task}.log").write_text(
            f"completed\n{epilog}\n", encoding="utf-8"
        )


@pytest.mark.parametrize(
    "epilog",
    (
        "official_test_accessed=false",
        "'eval_official_test_accessed': False\n"
        "'passk_official_test_accessed': False",
    ),
)
def test_logs_accept_both_registered_nonaccess_encodings(tmp_path, epilog: str) -> None:
    _write_payload_logs(tmp_path, epilog)
    verify_logs(tmp_path)


@pytest.mark.parametrize(
    "epilog,match",
    (
        (
            "'eval_official_test_accessed': True\n"
            "'passk_official_test_accessed': False",
            "access found",
        ),
        ("'eval_official_test_accessed': False", "epilog missing"),
        ("completed", "epilog missing"),
    ),
)
def test_logs_fail_closed_on_access_or_missing_nonaccess(tmp_path, epilog: str, match: str) -> None:
    _write_payload_logs(
        tmp_path,
        "'eval_official_test_accessed': False\n"
        "'passk_official_test_accessed': False",
    )
    (tmp_path / "qwen3_17b_l2r_temperature_mixture.7224195.6.log").write_text(
        f"{epilog}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=match):
        verify_logs(tmp_path)
