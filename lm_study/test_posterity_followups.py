"""Design, profile and scheduler tests for the posterity follow-ups."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import yaml

from ac_alg1 import _validate_ac_alg1_run_config, run_ac_alg1
from experiment_config import ACAlg1BatchAllocation
from generate_qwen3_17b_posterity_followups import (
    CELL_ORDER,
    RUN_ID,
    SEEDS,
    build_payload,
    validate_payload,
)
from run_yaml import _prepare_cells
from trainer_config import ACAlg1RunConfig
from validate_qwen3_17b_posterity_followups import _expected_coordinates


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "experiments_qwen3_17b_posterity_followups.yaml"


def _payload() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_generated_design_is_exactly_six_by_seven() -> None:
    payload = _payload()
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
    assert len(prepared) == 6
    coordinates = _expected_coordinates(payload)
    assert len(coordinates) == 42
    assert tuple(row["task_id"] for row in coordinates) == tuple(range(1, 43))


def test_registered_interventions_are_isolated() -> None:
    payload = _payload()
    cells = {cell["cell_id"]: cell for cell in payload["algos"]["AC-ALG1"]}
    assert tuple(cells) == CELL_ORDER
    assert cells["Q5-MORE-S32-B16-U1"]["batch"] == 128
    assert cells["Q5-MORE-S32-B16-U1"]["G"] == 32
    assert cells["Q5-TOKENMEAN-S16-B16-U1"]["responsibility_score"] == "token_mean"
    assert cells["PIS-Q-S8-B8-U1"]["iters"] == 1
    anchor = cells["PIS-Q-S8-B8-U4-KL03R"]
    assert anchor["policy_anchor_mode"] == "grad_ratio"
    assert anchor["policy_anchor_target_ratio"] == 0.03
    assert anchor["policy_anchor_token_scope"] == "reasoning"
    for cell_id, steps in (("EXACT-Q-S8-B8-U1", 1), ("EXACT-Q-S8-B8-U4", 4)):
        exact = cells[cell_id]
        assert exact["iters"] == steps
        assert exact["responsibility_refresh"] == "inner_step"
        assert exact["variational_estimator"] == "sampled_support_importance"
        assert exact["latent_mstep_objective"] == "exact_signed_trace_answer"


def test_every_cell_passes_runtime_profile_validation() -> None:
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


def test_scheduler_is_uncapped_and_dependency_gated() -> None:
    runner = (HERE / "run_qwen3_17b_posterity_followups_ucl.sh").read_text(encoding="utf-8")
    submitter = (HERE / "submit_qwen3_17b_posterity_followups_ucl.sh").read_text(encoding="utf-8")
    validator = (HERE / "validate_qwen3_17b_posterity_followups_ucl.sh").read_text(encoding="utf-8")
    assert "#$ -t 1-42" in runner
    assert "#$ -tc" not in runner
    assert "#$ -l gpu_type=h100" in runner
    assert "#$ -l tmem=24G" in runner
    assert "#$ -l h_vmem" not in runner
    assert "cell_count=6" in runner and "seed_count=7" in runner
    assert 'qsub -h -hold_jid "$CONTROL_VALIDATOR_JOB_ID"' in submitter
    assert "CONTROL_VALIDATOR_JOB_ID" in submitter
    assert "qwen3_posterity_followups.${SOURCE_JOB_ID}.*.log" in validator
