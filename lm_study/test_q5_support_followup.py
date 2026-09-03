"""Design, estimator, profile and scheduler tests for the Q5 support follow-up."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import torch
import yaml

from ac_alg1 import (
    TraceRow,
    _posterior_sampled_mstep_support,
    _validate_ac_alg1_run_config,
    run_ac_alg1,
)
from experiment_config import ACAlg1BatchAllocation
from generate_qwen3_17b_q5_support_followup import (
    CELL_ORDER,
    RUN_ID,
    SEEDS,
    build_payload,
    validate_payload,
)
from run_yaml import _prepare_cells
from trainer_config import ACAlg1RunConfig
from validate_qwen3_17b_q5_support_followup import _expected_coordinates


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "experiments_qwen3_17b_q5_support_followup.yaml"


def _row(pid: int, value: int) -> TraceRow:
    ids = torch.tensor([value], dtype=torch.long)
    mask = torch.ones(1, dtype=torch.bool)
    return TraceRow(
        ids=ids,
        span=mask,
        ans=mask,
        pid=pid,
        round_added=0,
        source="test",
        trace_id=f"{pid}:{value}",
    )


def _runtime_base() -> ACAlg1RunConfig:
    runtime = {
        "task", "model_tok", "eval_fn", "diagnostics_fn",
        "diagnostics_probe_fn", "checkpoint_fn", "log",
    }
    values = {
        name: parameter.default
        for name, parameter in inspect.signature(run_ac_alg1).parameters.items()
        if name not in runtime
    }
    return ACAlg1RunConfig.from_call(values)


def _runtime_cell(cell, defaults) -> ACAlg1RunConfig:
    allocation = ACAlg1BatchAllocation.from_budget(
        batch=int(cell.axes["batch"]),
        generations=int(cell.axes["G"]),
        labelled_fraction=float(cell.axes["labelled_frac"]),
    )
    fields = set(ACAlg1RunConfig.__dataclass_fields__)
    updates = {
        "rounds": int(defaults["rounds"]),
        "L_batch": allocation.labelled,
        "U_batch": allocation.answer_only,
        "G_label": int(cell.axes["G"]),
        "G_answer_only": int(cell.axes["G"]),
        "inner_steps": int(cell.axes["iters"]),
        "answer_event_mode": defaults["answer_event_mode"],
        "diagnostics_level": defaults["training_diagnostics_level"],
        "diagnostics_trace_tape": defaults["training_diagnostics_trace_tape"],
        "checkpoint_every": defaults["checkpoint_every"],
        "question_sampling": defaults["question_sampling"],
    }
    updates.update({key: value for key, value in cell.axes.items() if key in fields})
    return replace(_runtime_base(), **updates)


def test_top_plus_residual_is_deterministic_and_preserves_mass() -> None:
    buffers = {7: [_row(7, value) for value in range(4)]}
    weights = {7: torch.tensor([0.04, 0.06, 0.1, 0.8])}
    kwargs = dict(
        sample_size=16,
        seed=47,
        outer_round=3,
        inner_step=0,
        sampling_strategy="top_plus_residual",
    )
    first = _posterior_sampled_mstep_support(buffers, weights, [7], **kwargs)
    second = _posterior_sampled_mstep_support(buffers, weights, [7], **kwargs)
    assert [row.trace_id for row in first[0][7]] == [
        row.trace_id for row in second[0][7]
    ]
    assert torch.equal(first[1][7], second[1][7])
    question = first[2]["questions"][0]
    selected = question["selected_indices"]
    top_position = selected.index(3)
    assert torch.isclose(first[1][7].sum(), torch.tensor(1.0))
    assert torch.isclose(first[1][7][top_position], torch.tensor(0.8))
    assert question["exact_indices"] == [3]
    assert question["exact_trace_count"] == 1
    assert question["residual_draw_count"] == 15
    assert sum(question["multiplicities"]) == 15
    assert first[2]["mode"] == "top_plus_residual"


def test_top_plus_residual_average_is_unbiased() -> None:
    buffers = {9: [_row(9, value) for value in range(4)]}
    weights = {9: torch.tensor([0.04, 0.06, 0.1, 0.8])}
    values = {row.trace_id: float(row.ids.item()) for row in buffers[9]}
    estimates = []
    for seed in range(1000):
        sampled_buffers, sampled_weights, _ = _posterior_sampled_mstep_support(
            buffers,
            weights,
            [9],
            sample_size=16,
            seed=seed,
            outer_round=0,
            inner_step=0,
            sampling_strategy="top_plus_residual",
        )
        estimates.append(sum(
            values[row.trace_id] * float(weight)
            for row, weight in zip(
                sampled_buffers[9], sampled_weights[9], strict=True
            )
        ))
    target = sum(float(row.ids.item()) * float(weight) for row, weight in zip(
        buffers[9], weights[9], strict=True
    ))
    assert abs(sum(estimates) / len(estimates) - target) < 0.01


def test_generated_design_is_exactly_two_by_seven() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_payload(payload)
    assert payload == build_payload()
    assert payload["run_id"] == RUN_ID
    assert tuple(payload["defaults"]["seed_values"]) == SEEDS
    prepared = _prepare_cells(payload, only=None, run_id=RUN_ID, defaults=payload["defaults"])
    assert len(prepared) == 2
    coordinates = _expected_coordinates(payload)
    assert len(coordinates) == 14
    assert tuple(row["task_id"] for row in coordinates) == tuple(range(1, 15))


def test_every_cell_passes_runtime_profile_validation() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    prepared = _prepare_cells(
        payload, only=None, run_id=RUN_ID, defaults=payload["defaults"]
    )
    for cell in prepared:
        _validate_ac_alg1_run_config(
            _runtime_cell(cell, payload["defaults"]),
            diagnostics_fn=lambda _row: None,
            diagnostics_probe_fn=None,
        )


def test_profile_rejects_crossed_estimators() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cell = _prepare_cells(
        payload, only=None, run_id=RUN_ID, defaults=payload["defaults"]
    )[0]
    invalid = replace(
        _runtime_cell(cell, payload["defaults"]),
        mstep_sample_size=16,
        mstep_sampling_strategy="top_plus_residual",
    )
    try:
        _validate_ac_alg1_run_config(
            invalid,
            diagnostics_fn=lambda _row: None,
            diagnostics_probe_fn=None,
        )
    except ValueError as exc:
        assert "undeclared support estimator coordinates" in str(exc)
    else:
        raise AssertionError("undeclared support estimator was accepted")


def test_scheduler_is_uncapped_and_submission_is_held() -> None:
    runner = (HERE / "run_qwen3_17b_q5_support_followup_ucl.sh").read_text(
        encoding="utf-8"
    )
    submitter = (HERE / "submit_qwen3_17b_q5_support_followup_ucl.sh").read_text(
        encoding="utf-8"
    )
    validator = (HERE / "validate_qwen3_17b_q5_support_followup_ucl.sh").read_text(
        encoding="utf-8"
    )
    assert "#$ -t 1-14" in runner
    assert "#$ -tc" not in runner
    assert "#$ -l gpu_type=h100" in runner
    assert "#$ -l tmem=24G" in runner
    assert "#$ -l h_vmem" not in runner
    assert "cell_count=2" in runner and "seed_count=7" in runner
    assert "qsub -h" in submitter
    assert 'qsub -hold_jid "$payload_job"' in submitter
    assert "qwen3_q5_support_followup.${SOURCE_JOB_ID}.*.log" in validator

