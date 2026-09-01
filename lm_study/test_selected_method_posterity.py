"""Design and scheduler tests for the selected-method posterity replay."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import yaml

from ac_alg1 import _validate_ac_alg1_run_config, run_ac_alg1
from experiment_config import ACAlg1BatchAllocation
from generate_qwen3_17b_selected_method_posterity import (
    CELL_ORDER,
    METHOD_ORDER,
    RUN_ID,
    SEEDS,
    build_payload,
    validate_payload,
)
from run_yaml import _prepare_cells
from trainer_config import ACAlg1RunConfig
from validate_qwen3_17b_selected_method_posterity import _expected_coordinates


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "experiments_qwen3_17b_selected_method_posterity.yaml"


def _payload() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_generated_design_is_exactly_thirteen_by_seven() -> None:
    payload = _payload()
    validate_payload(payload)
    assert payload == build_payload()
    assert payload["run_id"] == RUN_ID
    prepared = _prepare_cells(
        payload,
        only=None,
        run_id=RUN_ID,
        defaults=payload["defaults"],
    )
    assert len(prepared) == 13
    assert tuple(cell.method for cell in prepared) == METHOD_ORDER
    assert tuple(payload["defaults"]["seed_values"]) == SEEDS
    coordinates = _expected_coordinates(payload)
    assert len(coordinates) == 91
    assert tuple(row["task_id"] for row in coordinates) == tuple(range(1, 92))


def test_q5_reader_pair_changes_only_answer_reader() -> None:
    payload = _payload()
    cells = {
        cell["cell_id"]: dict(cell)
        for raw in payload["algos"].values()
        for cell in (raw if isinstance(raw, list) else [raw])
    }
    question = cells["Q5-Q-LR1e-5-U1-K16"]
    moving = cells["Q5-AD-M-LR1e-5-U1-K16"]
    frozen = cells["Q5-AD-F-LR1e-5-U1-K16"]
    assert question["proposal_prompt"] == "question"
    assert moving["proposal_prompt"] == frozen["proposal_prompt"] == "answer_derive"
    assert moving["responsibility_answer_policy"] == "current"
    assert frozen["responsibility_answer_policy"] == "frozen_base"
    moving = {k: v for k, v in moving.items() if k != "cell_id"}
    frozen = {k: v for k, v in frozen.items() if k != "cell_id"}
    frozen["responsibility_answer_policy"] = "current"
    assert moving == frozen


def test_q5_stability_controls_are_isolated() -> None:
    payload = _payload()
    cells = {
        cell["cell_id"]: dict(cell)
        for raw in payload["algos"].values()
        for cell in (raw if isinstance(raw, list) else [raw])
    }
    control = cells["Q5-AD-M-LR1e-5-U1-K16"]
    expected = {
        "Q5-AD-M-ESS50-LR1e-5-U1-K16": {
            "responsibility_ess_floor": 0.5,
        },
        "Q5-AD-M-KL03R-LR1e-5-U1-K16": {
            "policy_anchor_mode": "grad_ratio",
            "policy_anchor_target_ratio": 0.03,
            "policy_anchor_token_scope": "reasoning",
        },
        "Q5-AD-M-T1p2-LR1e-5-U1-K16": {
            "proposal_temperature": 1.2,
        },
    }
    for cell_id, changes in expected.items():
        candidate = cells[cell_id]
        differing = {
            key
            for key in set(control) | set(candidate)
            if key != "cell_id" and control.get(key) != candidate.get(key)
        }
        assert differing == set(changes)
        assert all(candidate[key] == value for key, value in changes.items())


def test_pis_prompt_pair_repeats_the_registered_estimators() -> None:
    payload = _payload()
    cells = {
        cell["cell_id"]: dict(cell)
        for raw in payload["algos"].values()
        for cell in (raw if isinstance(raw, list) else [raw])
    }
    question = cells["PIS-Q-S8-B8-U4"]
    answer = cells["PIS-AD-S8-B8-U4"]
    assert question["proposal_prompt"] == "question"
    assert question["variational_estimator"] == "prior_importance"
    assert answer["proposal_prompt"] == "answer_derive"
    assert answer["variational_estimator"] == "answer_conditioned_importance"
    changed = {
        key
        for key in set(question) | set(answer)
        if question.get(key) != answer.get(key)
    }
    assert changed == {
        "cell_id",
        "algorithm_profile",
        "proposal_prompt",
        "variational_estimator",
    }


def test_selected_settings_and_closed_axes_are_explicit() -> None:
    payload = _payload()
    assert tuple(payload["diagnostic"]["design"]["cell_order"]) == CELL_ORDER
    assert payload["diagnostic"]["design"]["array_tasks"] == 91
    assert payload["defaults"]["lora_target_set"] == "attention_mlp"
    assert payload["defaults"]["lora_r"] == 16
    assert payload["defaults"]["eval_partition"] == "validation"
    assert payload["diagnostic"]["partitions"]["official_test_used"] is False
    controls = payload["diagnostic"]["provenance"]["controlled_interventions"]
    assert set(controls) == {"ess_floor", "proposal_temperature", "policy_anchor"}


def test_every_ac_alg1_cell_passes_runtime_profile_validation() -> None:
    payload = _payload()
    defaults = payload["defaults"]
    prepared = _prepare_cells(
        payload,
        only=None,
        run_id=RUN_ID,
        defaults=defaults,
    )
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
    base = ACAlg1RunConfig.from_call(values)
    fields = set(ACAlg1RunConfig.__dataclass_fields__)
    validated = 0
    for cell in prepared:
        if cell.method != "AC-ALG1":
            continue
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
            "answer_event_mode": defaults["answer_event_mode"],
            "diagnostics_level": defaults["training_diagnostics_level"],
            "diagnostics_trace_tape": defaults[
                "training_diagnostics_trace_tape"
            ],
            "checkpoint_every": defaults["checkpoint_every"],
            "question_sampling": defaults["question_sampling"],
        }
        updates.update(
            {key: value for key, value in cell.axes.items() if key in fields}
        )
        _validate_ac_alg1_run_config(
            replace(base, **updates),
            diagnostics_fn=lambda _row: None,
            diagnostics_probe_fn=None,
        )
        validated += 1
    assert validated == 8


def test_scheduler_is_uncapped_and_uses_exact_task_mapping() -> None:
    runner = (HERE / "run_qwen3_17b_selected_method_posterity_ucl.sh").read_text(
        encoding="utf-8"
    )
    submitter = (
        HERE / "submit_qwen3_17b_selected_method_posterity_ucl.sh"
    ).read_text(encoding="utf-8")
    validator = (
        HERE / "validate_qwen3_17b_selected_method_posterity_ucl.sh"
    ).read_text(encoding="utf-8")
    assert "#$ -t 1-91" in runner
    assert "#$ -tc" not in runner
    assert "#$ -l gpu_type=h100" in runner
    assert "#$ -l tmem=24G" in runner
    assert "#$ -l h_vmem" not in runner
    assert "cell_count=13" in runner
    assert "seed_count=7" in runner
    assert "qsub -h -v" in submitter
    assert "user_held_pending_release" in submitter
    assert "user_held_pending_herring_and_quota_gates" not in submitter
    assert "qwen3_selected_method_posterity.${SOURCE_JOB_ID}.*.log" in validator
    assert "qwen3_final_method_confirmation.${SOURCE_JOB_ID}" not in validator
