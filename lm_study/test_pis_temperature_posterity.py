"""Design and scheduler tests for the exact PIS temperature replay."""

from __future__ import annotations

from pathlib import Path

import yaml

from generate_qwen3_17b_pis_temperature_posterity import (
    CELLS,
    RUN_ID,
    SEEDS,
    build_payload,
    validate_payload,
)


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "experiments_qwen3_17b_pis_temperature_posterity.yaml"


def test_generated_temperature_replay_is_exact() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_payload(payload)
    assert payload == build_payload()
    assert payload["run_id"] == RUN_ID
    assert tuple(payload["diagnostic"]["design"]["cells"]) == CELLS
    assert tuple(payload["defaults"]["seed_values"]) == SEEDS
    assert payload["diagnostic"]["design"]["array_tasks"] == 14


def test_temperature_runner_is_uncapped_and_resource_safe() -> None:
    runner = (HERE / "run_qwen3_17b_pis_temperature_posterity_ucl.sh").read_text(
        encoding="utf-8"
    )
    assert "#$ -t 1-14" in runner
    assert "#$ -tc" not in runner
    assert "#$ -l gpu_type=h100" in runner
    assert "#$ -l tmem=24G" in runner
    assert "#$ -l h_vmem" not in runner
    assert "cell_count=2" in runner
    assert "seed_count=7" in runner
