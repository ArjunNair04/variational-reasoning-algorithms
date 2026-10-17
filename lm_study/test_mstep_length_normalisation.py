"""Verify M-step token means without changing Q5/PIS posterior weights."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import ac_alg1 as alg
from test_q5_component_kernel import Tokenizer
from test_q5_learned_posterior import tiny_model


def support():
    tok = Tokenizer()
    task = SimpleNamespace(prompts=["Q: x\nA:", "Q: y\nA:"], gold_answer=[7, 12])
    buffers = {}
    for pid, texts in enumerate((("ab####", "a longer rationale####"), ("xyz####",))):
        buffers[pid] = [
            alg._trace_row_from_h_ids(
                tok, task, pid, [ord(c) for c in text], 0, "buffer",
                trace_id=f"{pid}:{i}", reasoning_token_count=len(text) - 4,
                answer_event_mode="strict_terminal_marker", answer_target_termination="eos",
            )
            for i, text in enumerate(texts)
        ]
    return tok, buffers


def direct_objective(model, buffers, weights, mean):
    questions = []
    for pid, rows in buffers.items():
        terms = []
        for row, weight in zip(rows, weights[pid]):
            ids = row.ids.unsqueeze(0)
            logps = model(ids).logits[:, :-1].float().log_softmax(-1)
            chosen = logps.gather(-1, ids[:, 1:, None]).squeeze(-1)[0]
            target = row.span[1:]
            value = chosen[target].sum()
            terms.append(weight * (value / target.sum() if mean else value))
        questions.append(torch.stack(terms).sum())
    return torch.stack(questions).mean()


def gradients(model):
    return [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]


@pytest.mark.parametrize("estimator", ["delta_joint", "prior_importance"])
@pytest.mark.parametrize("mode", ["joint", "joint_token_mean"])
@pytest.mark.parametrize("micro", [1, 4])
def test_real_qwen_objective_gradient_and_posterior_identity(monkeypatch, estimator, mode, micro):
    monkeypatch.setattr(alg, "DEV", "cpu")
    tok, buffers = support()
    model = tiny_model().eval()
    weights = alg._buffer_weights_for_questions(
        model, tok, buffers, [0, 1], variational_estimator=estimator,
        responsibility_score="joint", record_joint_logprobs=True,
    )
    saved = {pid: w.clone() for pid, w in weights.items()}
    ids_before = [[row.ids.clone() for row in rows] for rows in buffers.values()]
    assert all(not w.requires_grad for w in weights.values())
    for rows in buffers.values():
        for row in rows:
            assert row.ids[-1] == tok.eos_token_id
            assert row.span[-1] and row.ans[-1]
            assert not row.span[0]

    reference = deepcopy(model)
    expected = direct_objective(reference, buffers, weights, mode == "joint_token_mean")
    (-expected).backward()
    before_rng = torch.random.get_rng_state().clone()
    actual, took_grad = alg._backward_B_unsup_for_questions(
        model, tok, buffers, [0, 1], weights, micro=micro, latent_mstep_objective=mode,
    )
    assert took_grad
    assert actual == pytest.approx(float(expected.detach()), rel=2e-6, abs=2e-6)
    for got, wanted in zip(gradients(model), gradients(reference)):
        torch.testing.assert_close(got, wanted, rtol=2e-4, atol=2e-6)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    full = alg._B_unsup_for_questions(
        model, tok, buffers, [0, 1], weights, grad=False, latent_mstep_objective=mode,
    )
    assert float(full) == pytest.approx(actual, rel=2e-6, abs=2e-6)
    refreshed = alg._buffer_weights_for_questions(
        model, tok, buffers, [0, 1], variational_estimator=estimator,
        responsibility_score="joint", record_joint_logprobs=True,
    )
    for pid in weights:
        assert torch.equal(weights[pid], saved[pid])
        assert torch.equal(refreshed[pid], saved[pid])
    for rows, old_ids in zip(buffers.values(), ids_before):
        assert all(torch.equal(row.ids, old) for row, old in zip(rows, old_ids))


def test_per_trace_mean_is_not_global_token_mean_or_new_posterior(monkeypatch):
    monkeypatch.setattr(alg, "DEV", "cpu")
    tok, buffers = support()
    weights = {0: torch.tensor([0.8, 0.2]), 1: torch.tensor([1.0])}
    model = tiny_model().eval()
    mean = alg._B_unsup_for_questions(
        model, tok, buffers, [0, 1], weights, latent_mstep_objective="joint_token_mean",
    )
    total = alg._B_unsup_for_questions(model, tok, buffers, [0, 1], weights)
    weighted_length = sum(
        sum(float(w) * int(row.span[1:].sum()) for row, w in zip(rows, weights[pid]))
        for pid, rows in buffers.items()
    ) / 2
    assert not torch.isclose(mean, total / weighted_length, rtol=1e-6, atol=1e-6)
    for pid, rows in buffers.items():
        _, span, ans = alg._pad_trace_rows(tok, rows)
        components = alg._latent_mstep_components(span, ans, rows, weights[pid], "joint_token_mean")
        assert len(components) == 1
        assert torch.equal(components[0][0], span)
        assert components[0][1] is weights[pid]


def test_six_task_design_changes_only_the_mstep():
    from generate_qwen3_17b_mstep_length_normalisation import (
        CONTROL_CELLS, SEEDS, build_payload, runtime_configs,
    )
    from generate_qwen3_17b_selected_method_posterity import build_payload as controls
    original, current = controls(), build_payload()
    originals = {cell["cell_id"]: cell for cell in original["algos"]["AC-ALG1"]}
    assert SEEDS == (1201, 1213, 1217)
    assert len(current["algos"]["AC-ALG1"]) * len(SEEDS) == 6
    omitted = {"cell_id", "algorithm_profile", "latent_mstep_objective"}
    for cell in current["algos"]["AC-ALG1"]:
        source = originals[CONTROL_CELLS[cell["cell_id"]]]
        assert {k:v for k,v in cell.items() if k not in omitted} == {
            k:v for k,v in source.items() if k not in omitted}
    omitted = {"seed_values", "seeds", "out"}
    assert {k:v for k,v in current["defaults"].items() if k not in omitted} == {
        k:v for k,v in original["defaults"].items() if k not in omitted}
    for config in runtime_configs():
        validated, _ = alg._validate_ac_alg1_run_config(config, diagnostics_fn=lambda _:None, diagnostics_probe_fn=None)
        assert validated.latent_mstep_objective == "joint_token_mean"
        for changes in (
            {"responsibility_score":"token_mean"}, {"responsibility_temperature":2.},
            {"responsibility_answer_policy":"frozen_base"}, {"length_norm":True},
            {"proposal_temperature":1.2}, {"lr":1e-6}, {"latent_mstep_objective":"joint"},
            {"policy_kl_coef":.02}, {"inner_steps":8}, {"answer_target_termination":"none"},
        ):
            with pytest.raises(ValueError):
                alg._validate_ac_alg1_run_config(replace(config, **changes), diagnostics_fn=lambda _:None, diagnostics_probe_fn=None)
        old_profile = "barber_q5_control" if config.variational_estimator == "delta_joint" else "l2r_common_factorial"
        with pytest.raises(ValueError):
            alg._validate_ac_alg1_run_config(replace(config, algorithm_profile=old_profile), diagnostics_fn=lambda _:None, diagnostics_probe_fn=None)
