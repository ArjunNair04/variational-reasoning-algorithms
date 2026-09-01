"""Fail-closed design tests for the seven-seed final-method confirmation."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import inspect
from pathlib import Path

import yaml

from ac_alg1 import _validate_ac_alg1_run_config, run_ac_alg1
from experiment_config import ACAlg1BatchAllocation
from generate_qwen3_17b_final_method_confirmation import (
    CELL_ORDER,
    CHECKPOINTS,
    RUN_ID,
    SEEDS,
    build_payload,
    selected_cells,
    validate_payload,
)
from run_yaml import _prepare_cells
from trainer_config import ACAlg1RunConfig
from validate_qwen3_17b_final_method_confirmation import (
    _expected_coordinates,
    _required_artifact_prefixes,
)


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "experiments_qwen3_17b_final_method_confirmation.yaml"


def _payload() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_generated_yaml_matches_frozen_generator() -> None:
    committed = _payload()
    validate_payload(committed)
    assert committed == build_payload()
    assert committed["run_id"] == RUN_ID


def test_design_is_exactly_ten_methods_by_seven_fresh_seeds() -> None:
    payload = _payload()
    prepared = _prepare_cells(
        payload,
        only=None,
        run_id=str(payload["run_id"]),
        defaults=payload["defaults"],
    )
    assert len(prepared) == 10
    assert tuple(payload["defaults"]["seed_values"]) == SEEDS
    assert tuple(payload["defaults"]["eval_rounds"]) == CHECKPOINTS
    assert payload["diagnostic"]["design"]["array_tasks"] == 70
    assert tuple(
        coordinate["task_id"] for coordinate in _expected_coordinates(payload)
    ) == tuple(range(1, 71))
    assert Counter(coordinate["cell_id"] for coordinate in _expected_coordinates(payload)) == {
        cell_id: 7 for cell_id in CELL_ORDER
    }


def test_selected_cells_are_copied_without_axis_changes() -> None:
    payload = _payload()
    configured = {
        str(cell["cell_id"]): (method, cell)
        for method, raw in payload["algos"].items()
        for cell in (raw if isinstance(raw, list) else [raw])
    }
    for cell_id, source in selected_cells().items():
        assert configured[cell_id] == source


def test_common_protocol_and_information_paths_are_explicit() -> None:
    payload = _payload()
    defaults = payload["defaults"]
    assert defaults["model"] == "qwen3-1.7b-base"
    assert defaults["prompts"] == 128
    assert defaults["shots"] == 3
    assert defaults["n_test"] == 400
    assert defaults["eval_partition"] == "validation"
    assert defaults["evaluation_prompt"] == "question"
    assert defaults["answer_event_mode"] == "strict_terminal_marker"
    assert defaults["lora_target_set"] == "attention_mlp"
    assert defaults["save_adapter"] is True
    assert payload["diagnostic"]["partitions"]["official_test_used"] is False

    configured = {
        str(cell["cell_id"]): (method, cell)
        for method, raw in payload["algos"].items()
        for cell in (raw if isinstance(raw, list) else [raw])
    }
    assert configured["Q5-LR1e-5-U1-K16"][1]["proposal_prompt"] == "answer_derive"
    for cell_id in CELL_ORDER:
        method, cell = configured[cell_id]
        if cell_id not in {"CTRL-base", "Q5-LR1e-5-U1-K16"}:
            assert cell.get("proposal_prompt") == "question"
        termination = cell.get(
            "answer_target_termination", defaults["answer_target_termination"]
        )
        if method in {"base", "GRPO", "RLOO"}:
            assert termination == "none"
        else:
            assert termination == "eos"
        if method in {"GRPO", "RLOO"}:
            assert cell["reward_requires_eos"] is True


def test_both_posterior_cells_pass_runtime_profile_validation() -> None:
    payload = _payload()
    defaults = payload["defaults"]
    prepared = _prepare_cells(
        payload,
        only=None,
        run_id=str(payload["run_id"]),
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
    assert validated == 2


def test_base_and_trained_receipt_contracts_are_distinct() -> None:
    base = _required_artifact_prefixes("base")
    trained = _required_artifact_prefixes("AC-ALG1")
    assert "checkpoint_eval_" not in base
    assert "training_diagnostics_" not in base
    assert "traj_" not in base
    assert "checkpoint_eval_" in trained
    assert "training_diagnostics_" in trained
    assert "traj_" in trained
    assert "passk_" in base and "passk_" in trained


def test_scheduler_is_uncapped_h100_tmem_and_exactly_seventy_tasks() -> None:
    runner = (HERE / "run_qwen3_17b_final_method_confirmation_ucl.sh").read_text(
        encoding="utf-8"
    )
    submitter = (
        HERE / "submit_qwen3_17b_final_method_confirmation_ucl.sh"
    ).read_text(encoding="utf-8")
    assert "#$ -t 1-70" in runner
    assert "#$ -tc" not in runner
    assert "#$ -l gpu_type=h100" in runner
    assert "#$ -l tmem=24G" in runner
    assert "#$ -l h_vmem" not in runner
    assert "cell_count=10" in runner
    assert "seed_count=7" in runner
    assert "qsub -h -v" in submitter
    assert "state=user_held_pending_tracking" in submitter
