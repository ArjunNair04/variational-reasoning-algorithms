"""Reader-only replication, negative profile gates and shared SGE delegates."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest
import yaml

import ac_alg1 as alg
import generate_qwen3_17b_q5_prior_segments as moving
import generate_qwen3_17b_q5_prior_segments_frozen as frozen
from generate_qwen3_17b_selected_method_posterity import build_payload as historical
from validate_qwen3_17b_q5_prior_segments import _expected_coordinates


def test_exact_reader_only_replication():
    before, after = moving.build_payload(), frozen.build_payload()
    assert frozen.SEEDS == (1201, 1213, 1217)
    assert len(frozen.CELL_ORDER) == 3
    assert {k:v for k,v in before["defaults"].items() if k != "out"} == {
        k:v for k,v in after["defaults"].items() if k != "out"}
    ignore = {"cell_id", "algorithm_profile", "responsibility_answer_policy"}
    for old, new in zip(before["algos"]["AC-ALG1"], after["algos"]["AC-ALG1"], strict=True):
        assert {k:v for k,v in old.items() if k not in ignore} == {k:v for k,v in new.items() if k not in ignore}
        assert new["responsibility_answer_policy"] == "frozen_base"
        assert new["latent_mstep_objective"] == "joint"
        assert new["responsibility_policy"] == new["proposal_policy"] == "current"
    control = next(c for c in historical()["algos"]["AC-ALG1"] if c["cell_id"] == frozen.CONTROL_CELL)
    ignore = {"cell_id", "algorithm_profile", "responsibility_prior_head_exponent", "responsibility_prior_tail_exponent"}
    assert {k:v for k,v in control.items() if k not in ignore} == {
        k:v for k,v in after["algos"]["AC-ALG1"][0].items() if k not in ignore}
    for config in frozen.runtime_configs():
        alg._validate_ac_alg1_run_config(config, diagnostics_fn=lambda _: None, diagnostics_probe_fn=None)
        for change in ({"responsibility_answer_policy": "current"}, {"responsibility_policy": "frozen_base"},
                       {"proposal_policy": "frozen_base"}, {"latent_mstep_objective": "generator_only"},
                       {"responsibility_ess_floor": .5}, {"proposal_temperature": 1.2},
                       {"policy_kl_coef": .02}, {"responsibility_prior_exponent": .5},
                       {"algorithm_profile": "q5_prior_segments"}, {"proposal_prompt": "question"}):
            with pytest.raises(ValueError):
                alg._validate_ac_alg1_run_config(replace(config, **change), diagnostics_fn=lambda _: None, diagnostics_probe_fn=None)


def test_frozen_coordinates_and_design_rejection():
    payload = frozen.build_payload()
    rows = _expected_coordinates(payload)
    assert [r["task_id"] for r in rows] == list(range(1, 10))
    assert [r["seed"] for r in rows] == list(frozen.SEEDS)*3
    assert [r["cell_id"] for r in rows] == [c for c in frozen.CELL_ORDER for _ in frozen.SEEDS]
    assert moving.study_for_payload(payload) is frozen
    assert moving.study_for_payload(moving.build_payload()) is moving
    bad = deepcopy(payload)
    bad["algos"]["AC-ALG1"][0]["responsibility_answer_policy"] = "current"
    with pytest.raises(ValueError, match="frozen design"):
        _expected_coordinates(bad)
    path = Path(__file__).parent/"experiments_qwen3_17b_q5_prior_segments_frozen.yaml"
    assert yaml.safe_load(path.read_text()) == payload


def test_frozen_scheduler_delegates_use_checkout_not_spool():
    root = Path(__file__).parent
    for kind in ("submit", "run", "validate"):
        path = root/f"{kind}_qwen3_17b_q5_prior_segments_frozen_ucl.sh"
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text()
        assert "-tc" not in text
        if kind != "submit":
            assert 'exec bash "$PROJ/lm_study/' in text
            assert 'dirname "$0"' not in text
        if kind == "run":
            assert "#$ -t 1-9" in text and "gpu_type=h100" in text
