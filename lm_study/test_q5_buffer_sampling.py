"""Design, sampling, profile and scheduler tests for Q5 buffer sampling."""

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
from generate_qwen3_17b_q5_buffer_sampling import (
    CELL_ORDER,
    RUN_ID,
    SEEDS,
    build_payload,
    validate_payload,
)
from run_yaml import _prepare_cells
from trainer_config import ACAlg1RunConfig
from validate_qwen3_17b_q5_buffer_sampling import _expected_coordinates


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "experiments_qwen3_17b_q5_buffer_sampling.yaml"


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
        "task",
        "model_tok",
        "eval_fn",
        "diagnostics_fn",
        "diagnostics_probe_fn",
        "checkpoint_fn",
        "log",
    }
    values = {
        name: parameter.default
        for name, parameter in inspect.signature(run_ac_alg1).parameters.items()
        if name not in runtime
    }
    return ACAlg1RunConfig.from_call(values)


def test_full_support_mode_is_the_identity() -> None:
    buffers = {7: [_row(7, value) for value in range(3)]}
    weights = {7: torch.tensor([0.1, 0.2, 0.7])}
    sampled_buffers, sampled_weights, diagnostics = _posterior_sampled_mstep_support(
        buffers,
        weights,
        [7],
        sample_size=0,
        seed=31,
        outer_round=2,
        inner_step=0,
    )
    assert sampled_buffers is buffers
    assert sampled_weights is weights
    assert diagnostics == {
        "mode": "full_posterior",
        "configured_draws_per_question": 0,
        "questions": [],
    }


def test_posterior_sampling_is_deterministic_and_preserves_draw_mass() -> None:
    buffers = {7: [_row(7, value) for value in range(4)]}
    weights = {7: torch.tensor([0.05, 0.15, 0.3, 0.5])}
    first = _posterior_sampled_mstep_support(
        buffers,
        weights,
        [7],
        sample_size=16,
        seed=47,
        outer_round=3,
        inner_step=0,
    )
    second = _posterior_sampled_mstep_support(
        buffers,
        weights,
        [7],
        sample_size=16,
        seed=47,
        outer_round=3,
        inner_step=0,
    )
    assert [row.trace_id for row in first[0][7]] == [
        row.trace_id for row in second[0][7]
    ]
    assert torch.equal(first[1][7], second[1][7])
    question = first[2]["questions"][0]
    assert sum(question["multiplicities"]) == 16
    assert len(question["selected_indices"]) == len(first[0][7])
    assert torch.isclose(first[1][7].sum(), torch.tensor(1.0))
    assert first[2]["mode"] == "posterior_categorical"


def test_generated_design_is_exactly_two_by_seven() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_payload(payload)
    assert payload == build_payload()
    assert payload["run_id"] == RUN_ID
    assert tuple(payload["defaults"]["seed_values"]) == SEEDS
    prepared = _prepare_cells(
        payload,
        only=None,
        run_id=RUN_ID,
        defaults=payload["defaults"],
    )
    assert len(prepared) == 2
    coordinates = _expected_coordinates(payload)
    assert len(coordinates) == 14
    assert tuple(row["task_id"] for row in coordinates) == tuple(range(1, 15))


def test_every_cell_passes_runtime_profile_validation() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    defaults = payload["defaults"]
    prepared = _prepare_cells(
        payload,
        only=None,
        run_id=RUN_ID,
        defaults=defaults,
    )
    base = _runtime_base()
    fields = set(ACAlg1RunConfig.__dataclass_fields__)
    for cell in prepared:
        allocation = ACAlg1BatchAllocation.from_budget(
            batch=int(cell.axes["batch"]),
            generations=int(cell.axes["G"]),
            labelled_fraction=float(cell.axes["labelled_frac"]),
        )
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
        _validate_ac_alg1_run_config(
            replace(base, **updates),
            diagnostics_fn=lambda _row: None,
            diagnostics_probe_fn=None,
        )


def test_profile_rejects_undeclared_sample_size() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cell = _prepare_cells(
        payload,
        only=None,
        run_id=RUN_ID,
        defaults=payload["defaults"],
    )[0]
    allocation = ACAlg1BatchAllocation.from_budget(
        batch=int(cell.axes["batch"]),
        generations=int(cell.axes["G"]),
        labelled_fraction=float(cell.axes["labelled_frac"]),
    )
    fields = set(ACAlg1RunConfig.__dataclass_fields__)
    updates = {
        "rounds": int(payload["defaults"]["rounds"]),
        "L_batch": allocation.labelled,
        "U_batch": allocation.answer_only,
        "G_label": int(cell.axes["G"]),
        "G_answer_only": int(cell.axes["G"]),
        "inner_steps": int(cell.axes["iters"]),
        "answer_event_mode": payload["defaults"]["answer_event_mode"],
        "diagnostics_level": payload["defaults"]["training_diagnostics_level"],
        "diagnostics_trace_tape": payload["defaults"]["training_diagnostics_trace_tape"],
        "checkpoint_every": payload["defaults"]["checkpoint_every"],
        "question_sampling": payload["defaults"]["question_sampling"],
    }
    updates.update({key: value for key, value in cell.axes.items() if key in fields})
    updates["mstep_sample_size"] = 8
    invalid = replace(_runtime_base(), **updates)
    try:
        _validate_ac_alg1_run_config(
            invalid,
            diagnostics_fn=lambda _row: None,
            diagnostics_probe_fn=None,
        )
    except ValueError as exc:
        assert "permits only full support or 16 posterior draws" in str(exc)
    else:
        raise AssertionError("undeclared M-step sample size was accepted")


def test_scheduler_is_uncapped_and_submission_is_held() -> None:
    runner = (HERE / "run_qwen3_17b_q5_buffer_sampling_ucl.sh").read_text(
        encoding="utf-8"
    )
    submitter = (HERE / "submit_qwen3_17b_q5_buffer_sampling_ucl.sh").read_text(
        encoding="utf-8"
    )
    validator = (HERE / "validate_qwen3_17b_q5_buffer_sampling_ucl.sh").read_text(
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
    assert "qwen3_q5_buffer_sampling.${SOURCE_JOB_ID}.*.log" in validator
    assert 'source "$PROJ/lm_study/ucl_python_env.sh"' in validator
    assert validator.count('"$VENV/bin/python"') == 2
