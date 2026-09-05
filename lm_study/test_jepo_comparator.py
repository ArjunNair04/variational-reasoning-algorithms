"""Estimator, design and scheduler tests for the JEPO comparator."""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from generate_qwen3_17b_jepo_comparator import (
    CELL_ID,
    RUN_ID,
    SEEDS,
    build_payload,
    validate_payload,
)
from jepo import _strict_format_mask, run_jepo
from run_sweep_lm import METHODS
from run_yaml import _prepare_cells


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "experiments_qwen3_17b_jepo_comparator.yaml"


def test_generated_design_is_one_cell_by_seven_seeds() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_payload(payload)
    assert payload == build_payload()
    assert payload["run_id"] == RUN_ID
    assert tuple(payload["defaults"]["seed_values"]) == SEEDS
    cells = _prepare_cells(
        payload, only=None, run_id=RUN_ID, defaults=payload["defaults"]
    )
    assert len(cells) == 1
    assert cells[0].method == "JEPO"
    assert cells[0].tag.endswith(f"_{CELL_ID}")


def test_jepo_runtime_signature_exposes_every_frozen_axis() -> None:
    signature = inspect.signature(run_jepo)
    cell = build_payload()["algos"]["JEPO"]
    # run_yaml maps the public `batch` axis to trainers' historical `B` name.
    unsupported = set(cell) - {"cell_id", "batch"} - set(signature.parameters)
    assert not unsupported
    assert METHODS["JEPO"] is run_jepo


def test_jepo_trainer_is_fail_closed_on_protocol_changes() -> None:
    for changed, message in (
        ({"G": 1}, "group size"),
        ({"proposal_prompt": "answer_derive"}, "question-conditioned"),
        ({"answer_event_mode": "legacy"}, "strict terminal"),
        ({"answer_target_termination": "none"}, "tokenizer EOS"),
        ({"reward_requires_eos": False}, "natural EOS"),
    ):
        try:
            run_jepo(object(), model_tok=(object(), object()), **changed)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"JEPO accepted invalid change {changed}")


def test_jepo_uses_shared_gsm8k_parser_when_task_has_no_parser_method() -> None:
    mask = _strict_format_mask(
        object(),
        ["Reasoning.\n#### 42", "Reasoning.\n#### 42 trailing"],
        [True, True],
    )
    assert mask.tolist() == [True, False]


def test_scheduler_is_uncapped_and_submission_is_held() -> None:
    runner = (HERE / "run_qwen3_17b_jepo_comparator_ucl.sh").read_text(encoding="utf-8")
    submitter = (HERE / "submit_qwen3_17b_jepo_comparator_ucl.sh").read_text(
        encoding="utf-8"
    )
    validator = (HERE / "validate_qwen3_17b_jepo_comparator_ucl.sh").read_text(
        encoding="utf-8"
    )
    assert "#$ -t 1-7" in runner
    assert "#$ -tc" not in runner
    assert "#$ -l gpu_type=h100" in runner
    assert "#$ -l tmem=24G" in runner
    assert "cell_count=1" in runner and "seed_count=7" in runner
    assert 'export PYTHONPATH="$PROJ/src:$PROJ/lm_study' in runner
    assert 'export PYTHONPATH="$PROJ/src:$SCRIPT_DIR' in submitter
    assert 'from jepo import _strict_format_mask' in submitter
    assert "qsub -h" in submitter
    assert 'qsub -hold_jid "$payload_job"' in submitter
    assert "qwen3_jepo_comparator.${SOURCE_JOB_ID}.*.log" in validator
    assert 'source "$PROJ/lm_study/ucl_python_env.sh"' in validator
    assert 'export PYTHONPATH="$PROJ/src:$PROJ/lm_study' in validator
    assert validator.count('"$VENV/bin/python"') == 2
