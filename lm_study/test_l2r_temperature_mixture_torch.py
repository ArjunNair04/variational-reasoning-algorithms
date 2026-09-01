"""Tensor identities for temperature-mixture proposal correction."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from l2r import _temperature_component_h_logps, l2r_responsibilities


class _FixedModel:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, ids):
        return SimpleNamespace(logits=self._logits[: ids.shape[0], : ids.shape[1]])


def test_temperature_density_matches_direct_autoregressive_calculation():
    ids = torch.tensor([[0, 1, 2], [0, 2, 1]])
    mask = torch.tensor([[False, True, True], [False, True, True]])
    logits = torch.tensor([
        [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]],
        [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]],
    ])
    observed = _temperature_component_h_logps(
        _FixedModel(logits), ids, mask, temperature=1.2, micro=1
    )
    scaled = logits[:, :-1] / 1.2
    token = scaled.gather(-1, ids[:, 1:, None]).squeeze(-1) - torch.logsumexp(
        scaled, dim=-1
    )
    assert torch.allclose(observed, token.sum(dim=1), atol=1e-7, rtol=1e-7)


def test_exact_mixture_responsibilities_use_the_mixture_denominator():
    p1 = torch.tensor([-2.0, -3.0, -4.0, -5.0])
    p12 = torch.tensor([-2.5, -2.7, -4.3, -4.8])
    answer = torch.tensor([-1.0, -0.2, -0.7, -0.4])
    pids = torch.tensor([0, 0, 1, 1])
    lengths = torch.ones(4)
    observed = l2r_responsibilities(
        p1,
        answer,
        lengths,
        lengths,
        pids,
        score="mixed_prior_corrected",
        answer_proposal_h_logps=p12,
        proposal_prior_fraction=0.5,
    )
    mixture = torch.logaddexp(p1 + torch.log(torch.tensor(0.5)), p12 + torch.log(torch.tensor(0.5)))
    expected = torch.zeros(4)
    expected[:2] = torch.softmax((p1 + answer - mixture)[:2], dim=0)
    expected[2:] = torch.softmax((p1 + answer - mixture)[2:], dim=0)
    assert torch.allclose(observed, expected, atol=1e-7, rtol=1e-7)
