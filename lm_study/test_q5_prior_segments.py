"""Positional scoring, historical identity, RNG isolation and frozen protocol."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import ac_alg1 as alg
from ac_alg1_prior_segments import split_rationale_mask, split_reasoning_marker_mask, segment_prior_logits
from generate_qwen3_17b_q5_prior_segments import (
    CELL_ORDER, CONTROL_CELL, SEEDS, SETTINGS, build_payload, runtime_configs,
)


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 7, 12])
def test_mask_partition(length):
    mask = torch.zeros((2, length + 6), dtype=torch.bool)
    mask[0, 3:3 + length] = True
    mask[1, 1:1 + length] = True
    head, tail = split_rationale_mask(mask)
    assert torch.equal(head | tail, mask)
    assert not (head & tail).any()
    assert head.sum(1).tolist() == [length // 2] * 2
    assert tail.sum(1).tolist() == [length - length // 2] * 2
    for row in range(2):
        assert torch.equal(head[row].nonzero(), mask[row].nonzero()[:length // 2])


def test_bad_masks_and_factors():
    for mask in (torch.ones(3, dtype=torch.bool), torch.zeros((2, 3)), torch.ones((2, 3), dtype=torch.bool)):
        with pytest.raises(ValueError):
            split_rationale_mask(mask)
    a = torch.tensor([-1.0])
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            segment_prior_logits(a, a, a, value, 1)


@pytest.mark.parametrize("length", [0, 1, 2, 3, 7])
def test_reasoning_split_leaves_multitoken_marker_fixed(length):
    mask = torch.zeros((2, length + 8), dtype=torch.bool)
    mask[0, 2:2+length+2] = True
    mask[1, 4:4+length+2] = True
    head, tail, marker = split_reasoning_marker_mask(mask, [length, length])
    assert head.sum(1).tolist() == [length//2]*2
    assert tail.sum(1).tolist() == [length-length//2]*2
    assert marker.sum(1).tolist() == [2, 2]
    assert torch.equal(head | tail | marker, mask)
    assert not ((head & tail) | (head & marker) | (tail & marker)).any()
    for i in range(2):
        assert torch.equal(marker[i].nonzero(), mask[i].nonzero()[-2:])
    for counts in ([None, length], [True, length], [-1, length],
                   [length+2, length], [length], [1.5, length]):
        with pytest.raises(ValueError):
            split_reasoning_marker_mask(mask, counts)


@pytest.mark.parametrize("head,tail", [(1, 1), (0.5, 1), (1, 0.5), (0.75, 0.75)])
def test_actual_estep_and_rng(monkeypatch, head, tail):
    ids = torch.tensor([[9, 8, 7, 6, 5, 4, 3], [9, 8, 7, 6, 5, 4, 3]])
    span = torch.tensor([[False, False, True, True, True, True, True]] * 2)
    ans = torch.tensor([[False, False, False, False, False, True, True]] * 2)
    token_lp = torch.tensor([[0., 0., -1., -4., -2., -.2, -.1], [0., 0., -3., -1., -1., -.5, -.2]])
    calls = []

    def logprobs(model, actual_ids, mask, **kwargs):
        assert torch.equal(actual_ids, ids)  # All original conditioning tokens retained.
        calls.append(mask.clone())
        torch.rand(1)  # Extra scoring must not advance the main RNG stream.
        return (token_lp * mask).sum(1)

    monkeypatch.setattr(alg, "seq_logprobs", logprobs)
    monkeypatch.setattr(alg, "_pad_trace_rows", lambda *_: (ids, span, ans))
    rows = [alg.TraceRow(ids=ids[i], span=span[i], ans=ans[i], pid=7,
                         round_added=0, source="test", reasoning_token_count=2) for i in range(2)]
    model = SimpleNamespace()
    torch.manual_seed(7)
    legacy = alg._buffer_weights_for_questions(model, None, {7: rows}, [7], record_joint_logprobs=True)
    old_rng = torch.random.get_rng_state().clone()
    calls.clear()
    torch.manual_seed(7)
    actual = alg._buffer_weights_for_questions(model, None, {7: rows}, [7], record_joint_logprobs=True,
        responsibility_prior_head_exponent=head, responsibility_prior_tail_exponent=tail)
    assert torch.equal(torch.random.get_rng_state(), old_rng)
    early, late, marker = split_reasoning_marker_mask(span & ~ans, [2, 2])
    expected = (token_lp * (ans | marker)).sum(1) + head * (token_lp * early).sum(1) + tail * (token_lp * late).sum(1)
    torch.testing.assert_close(actual[7], torch.softmax(expected, dim=0))
    assert not actual[7].requires_grad
    assert len(calls) == (2 if head == tail == 1 else 4)
    if head == tail == 1:
        assert torch.equal(legacy[7], actual[7])
        assert all(row.prior_head_logprob is None for row in rows)
    else:
        assert all(row.prior_head_tokens == row.prior_tail_tokens == row.prior_marker_tokens == 1 for row in rows)
        assert torch.equal(calls[-2], early) and torch.equal(calls[-1], marker)
        for i, row in enumerate(rows):
            assert row.prior_marker_logprob == pytest.approx(float((token_lp * marker).sum(1)[i]))
            assert row.prior_head_logprob + row.prior_tail_logprob + row.prior_marker_logprob == pytest.approx(row.trace_logprob)


def test_uniform_matches_global_exponent_arithmetic():
    a, p, h = torch.tensor([-.1, -4.]), torch.tensor([-20., -5.]), torch.tensor([-7., -2.])
    assert torch.equal(segment_prior_logits(a, p, h, .75, .75), a + .75 * p)


def test_frozen_design_and_runtime():
    from generate_qwen3_17b_selected_method_posterity import build_payload as old
    assert SEEDS == (1201, 1213, 1217)
    assert len(CELL_ORDER) * len(SEEDS) == 9
    historical = old()
    current = build_payload()
    original = next(c for c in historical["algos"]["AC-ALG1"] if c["cell_id"] == CONTROL_CELL)
    exclude = {"cell_id", "algorithm_profile", "responsibility_prior_head_exponent", "responsibility_prior_tail_exponent"}
    for cell in current["algos"]["AC-ALG1"]:
        assert {k:v for k,v in cell.items() if k not in exclude} == {k:v for k,v in original.items() if k not in exclude}
    skip = {"seed_values", "seeds", "out"}
    assert {k:v for k,v in current["defaults"].items() if k not in skip} == {k:v for k,v in historical["defaults"].items() if k not in skip}
    for config in runtime_configs():
        alg._validate_ac_alg1_run_config(config, diagnostics_fn=lambda _: None, diagnostics_probe_fn=None)
        for change in ({"responsibility_answer_policy": "frozen_base"}, {"responsibility_ess_floor": .5},
                       {"proposal_temperature": 1.2}, {"responsibility_prior_exponent": .5},
                       {"responsibility_score": "token_mean"}, {"algorithm_profile": "barber_q5_control"},
                       {"policy_kl_coef": .02}, {"responsibility_prior_tail_exponent": float("nan")}):
            with pytest.raises(ValueError):
                alg._validate_ac_alg1_run_config(replace(config, **change), diagnostics_fn=lambda _: None, diagnostics_probe_fn=None)
