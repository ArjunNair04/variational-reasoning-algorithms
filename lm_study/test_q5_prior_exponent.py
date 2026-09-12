"""Numerical E-step and complete runtime-design checks, without a model download."""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

import ac_alg1 as alg
from generate_qwen3_17b_q5_prior_exponent import CELL_ORDER, SEEDS, build_payload, runtime_configs


def test_exact_design_and_runtime():
    assert SEEDS == (1201, 1213, 1217)
    assert len(CELL_ORDER) * len(SEEDS) == 12
    for config in runtime_configs():
        alg._validate_ac_alg1_run_config(config, diagnostics_fn=lambda _: None, diagnostics_probe_fn=None)
        assert config.responsibility_prior_exponent in (0.0, 0.5)
        for change in ({"responsibility_ess_floor": 0.5}, {"proposal_temperature": 1.2},
                       {"algorithm_profile": "barber_q5_control"}, {"responsibility_score": "token_mean"}):
            with pytest.raises(ValueError):
                alg._validate_ac_alg1_run_config(replace(config, **change), diagnostics_fn=lambda _: None, diagnostics_probe_fn=None)


def test_only_declared_changes_from_historical_controls():
    from generate_qwen3_17b_selected_method_posterity import build_payload as old
    from generate_qwen3_17b_q5_prior_exponent import CONTROL_CELLS
    historical = old()
    current = build_payload()
    originals = {c["cell_id"]: c for c in historical["algos"]["AC-ALG1"]}
    for cell in current["algos"]["AC-ALG1"]:
        source = originals[CONTROL_CELLS[cell["cell_id"].split("-")[2]]]
        assert {k: v for k, v in cell.items() if k not in {"cell_id", "algorithm_profile", "responsibility_prior_exponent"}} == {
            k: v for k, v in source.items() if k not in {"cell_id", "algorithm_profile"}}
    skip = {"seed_values", "seeds", "out"}
    assert {k: v for k, v in current["defaults"].items() if k not in skip} == {
        k: v for k, v in historical["defaults"].items() if k not in skip}


@pytest.mark.parametrize("reader", ["current", "frozen_base"])
@pytest.mark.parametrize("tau", [0.0, 0.5, 1.0])
def test_actual_estep_factors_and_reader(monkeypatch, reader, tau):
    model = SimpleNamespace(policy="current")
    prior = torch.tensor([-5.0, -1.0])
    answers = {"current": torch.tensor([-0.1, -2.0]), "frozen_base": torch.tensor([-3.0, -0.2])}

    @contextmanager
    def policy_context(model, policy):
        previous = model.policy
        model.policy = policy
        try:
            yield
        finally:
            model.policy = previous

    def logprobs(model, ids, mask, **kwargs):
        return mask[0, 0] * prior + mask[0, 1] * answers[model.policy]

    monkeypatch.setattr(alg, "_adapter_policy_context", policy_context)
    monkeypatch.setattr(alg, "seq_logprobs", logprobs)
    monkeypatch.setattr(alg, "_pad_trace_rows", lambda *_: (
        torch.tensor([[1, 2], [3, 4]]), torch.tensor([[True, True]] * 2), torch.tensor([[False, True]] * 2)))
    rows = [alg.TraceRow(ids=torch.tensor([1, 2]), span=torch.tensor([True, True]),
        ans=torch.tensor([False, True]), pid=7, round_added=0, source="test") for _ in range(2)]
    result = alg._buffer_weights_for_questions(model, None, {7: rows}, [7],
        responsibility_answer_policy=reader, responsibility_prior_exponent=tau, record_joint_logprobs=True)
    expected = torch.softmax(answers[reader] + tau * prior, dim=0)
    torch.testing.assert_close(result[7], expected)
    assert not result[7].requires_grad
    assert model.policy == "current"
    if tau == 1:
        legacy = alg._buffer_weights_for_questions(model, None, {7: rows}, [7],
            responsibility_answer_policy=reader, record_joint_logprobs=True)
        assert torch.equal(result[7], legacy[7])


def test_invalid_exponent_rejected():
    for tau in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            alg._buffer_weights_for_questions(None, None, {}, [], responsibility_prior_exponent=tau)


def test_submitter_preserves_source_import_path():
    script = (Path(__file__).parent / "submit_qwen3_17b_q5_prior_exponent_ucl.sh").read_text()
    assert 'export PYTHONPATH="$PROJ/src:$SCRIPT_DIR:$PROJ/analysis:' in script
    assert 'PYTHONPATH="$SCRIPT_DIR:' not in script
