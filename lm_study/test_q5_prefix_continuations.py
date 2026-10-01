"""Native-prefix construction, accounting, historical isolation and protocol."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import ac_alg1 as alg
from ac_alg1_prefix_continuations import reasoning_prefix, sample_prefix_continuations
from generate_qwen3_17b_q5_prefix_continuations import (
    CONTROL_CELL, CELL_ORDER, SEEDS, build_payload, runtime_configs,
)


class CharTokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids if i)

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]

    def __call__(self, text, return_tensors="pt"):
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) for c in text]]))


def pack(texts):
    rows = [[ord(c) for c in text] + [0] for text in texts]
    ids = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row)
        mask[i, :len(row)] = True
    return ids, mask, texts


class FakeModel:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        rows = kwargs["input_ids"]
        suffix = torch.tensor([[ord(c) for c in "z #### 7"] + [0]] * len(rows))
        return torch.cat([rows, suffix[:, :kwargs["max_new_tokens"]]], dim=1)


@pytest.mark.parametrize("text,prefix,reason", [
    ("abcdef\n#### 7", "abc", None),
    ("ab#### 7", "a", None),
    ("a#### 7", "", "empty_half_prefix"),
    ("abc", "", "missing_reasoning_boundary"),
    ("#### 7", "", "missing_reasoning_boundary"),
    ("abcd#### 7 next #### 8", "ab", None),
])
def test_prefix_boundary(text, prefix, reason):
    tok = CharTokenizer()
    state = torch.random.get_rng_state().clone()
    ids, actual_reason = reasoning_prefix(tok, [ord(c) for c in text] + [0])
    assert ids == [ord(c) for c in prefix]
    assert actual_reason == reason
    assert torch.equal(state, torch.random.get_rng_state())


def test_native_token_overlapping_marker_excluded():
    class Tokenizer(CharTokenizer):
        def decode(self, ids, **kwargs):
            pieces = {1: "ab", 2: "cd", 3: "ef###", 4: "# 9", 0: ""}
            return "".join(pieces[int(i)] for i in ids)
    assert reasoning_prefix(Tokenizer(), [1, 2, 3, 4, 0]) == ([1], None)


def test_partial_unicode_does_not_truncate_reasoning_boundary():
    class Tokenizer(CharTokenizer):
        def decode(self, ids, **kwargs):
            pieces = {1:b"abc", 2:b"\xc3", 3:b"\x97", 4:b"#### 9", 0:b""}
            return b"".join(pieces[int(i)] for i in ids).decode("utf-8", errors="replace")
    assert reasoning_prefix(Tokenizer(), [1, 2, 3, 4, 0]) == ([1], None)


def test_question_major_lineage_and_token_accounting():
    tok, model = CharTokenizer(), FakeModel()
    roots = ["abcdef#### 7", "missing boundary", "12345678#### 7", "ijkl#### 9"]
    def sample(_, __, prompts, **kwargs):
        assert prompts == ["Q1", "Q1", "Q2", "Q2"]
        assert kwargs == {"max_new": 32}
        return pack(roots)
    ids, mask, texts, meta = sample_prefix_continuations(model, tok,
        ["Q1"]*4 + ["Q2"]*4, traces_per_question=4, max_new=32,
        temperature=1., sample_fn=sample, device="cpu")
    assert [texts[i] for i in (0, 1, 4, 5)] == roots
    assert meta["parent_indices"] == [None, None, 0, 1, None, None, 4, 5]
    assert meta["prefix_tokens"] == [0, 0, 3, 0, 0, 0, 4, 2]
    assert meta["root_count"] == 4
    assert meta["fallback_count"] == 1
    assert meta["continuation_count"] == 3
    assert meta["fallback_reasons"][3] == "missing_reasoning_boundary"
    assert meta["completion_tokens"] == mask.sum(1).tolist()
    assert [g+p for g,p in zip(meta["generated_tokens"], meta["prefix_tokens"])] == meta["completion_tokens"]
    for child, parent in enumerate(meta["parent_indices"]):
        if parent is not None:
            n = meta["prefix_tokens"][child]
            assert torch.equal(ids[child, :n], ids[parent, :n])
            assert meta["prefix_sha256"][child]
    assert all(n <= 32 for n in meta["completion_tokens"])
    assert sorted(c["max_new_tokens"] for c in model.calls) == [28, 29, 30, 32]
    assert all(c["temperature"] == 1. and c["top_k"] == 0 and c["top_p"] == 1. for c in model.calls)


@pytest.mark.parametrize("g,prompts", [(1,["q"]), (3,["q"]*3), (4,["q"]*3), (2,["q","r"])])
def test_bad_allocation(g, prompts):
    with pytest.raises(ValueError):
        sample_prefix_continuations(FakeModel(), CharTokenizer(), prompts,
            traces_per_question=g, max_new=32, temperature=1., sample_fn=lambda *a, **k: None, device="cpu")


def test_generator_cannot_modify_prefix():
    class Broken(FakeModel):
        def generate(self, **kwargs):
            output = super().generate(**kwargs)
            output[0, 0] = 999
            return output
    with pytest.raises(ValueError, match="changed its input prefix"):
        sample_prefix_continuations(Broken(), CharTokenizer(), ["q"]*2,
            traces_per_question=2, max_new=32, temperature=1.,
            sample_fn=lambda *a, **k: pack(["abcd#### 9"]), device="cpu")


def test_real_qwen_generation_and_backward_without_downloads():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(43)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=128,
        eos_token_id=0, pad_token_id=0, attention_dropout=0.))
    ids, mask, _, meta = sample_prefix_continuations(model, CharTokenizer(), ["q"]*2,
        traces_per_question=2, max_new=16, temperature=1., device="cpu",
        sample_fn=lambda *a, **k: pack(["abcd#### 9"]))
    assert ids[1, :2].tolist() == [ord("a"), ord("b")]
    assert meta["completion_tokens"][1] <= 16
    labels = ids.clone()
    labels[~mask] = -100
    model.train()
    loss = model(input_ids=ids, attention_mask=mask.long(), labels=labels).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_legacy_sampler_and_rng_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(alg, "_sampled_trace_row", lambda *a, **k: None)
    def sample(*args, **kwargs):
        calls.append(kwargs)
        torch.rand(1)
        return pack(["abcd#### 7"]*2)
    monkeypatch.setattr(alg, "sample_multi", sample)
    monkeypatch.setattr(alg, "sample_prefix_continuations", lambda *a, **k: pytest.fail("legacy reached branching"))
    task = SimpleNamespace(prompts=["q"], gold_answer=[7], max_new=32,
        build_proposal_prompt=lambda pid, mode: "q")
    kwargs = dict(model=FakeModel(), tok=CharTokenizer(), task=task, buffers={0:[]},
        pids=[0], traces_per_question=2, round_added=0, source="answer_only_sample",
        buffer_limit=16, buffer_strategy="fifo", collect_token_counts=True)
    torch.manual_seed(123)
    default = alg._add_model_traces_to_buffer(**kwargs)
    state = torch.random.get_rng_state().clone()
    torch.manual_seed(123)
    explicit = alg._add_model_traces_to_buffer(**kwargs, proposal_continuation_mode="independent")
    assert default == explicit
    assert torch.equal(state, torch.random.get_rng_state())
    assert calls == [{"max_new":32}]*2


def test_buffer_uses_full_children_but_counts_only_new_tokens(monkeypatch):
    monkeypatch.setattr(alg, "DEV", "cpu")
    monkeypatch.setattr(alg, "sample_multi", lambda *a, **k: pack(["abcd#### 7"]))
    captured = []
    def construct(tok, task, pid, ids, mask, text, *args, **kwargs):
        captured.append((ids[mask].tolist(), kwargs))
        return None
    monkeypatch.setattr(alg, "_sampled_trace_row", construct)
    task = SimpleNamespace(prompts=["q"], gold_answer=[7], max_new=32,
        build_proposal_prompt=lambda pid, mode: "q", reward=lambda texts,pids: [1.]*len(texts))
    result = alg._add_model_traces_to_buffer(FakeModel(), CharTokenizer(), task, {0:[]},
        [0], 2, 0, "answer_only_sample", 16, "fifo",
        proposal_prompt="answer_derive", answer_event_mode="strict_terminal_marker",
        answer_target_termination="eos", proposal_continuation_mode="prefix_half",
        collect_token_counts=True, collect_proposal_outcomes=True)
    meta = result[-1]["continuations"]
    assert captured[1][0][:2] == captured[0][0][:2]
    assert len(captured[1][0]) == captured[1][1]["proposal_tokens"] + 2
    assert result[2] == meta["generated_tokens"]
    assert meta["parent_trace_ids"] == [None, meta["trace_ids"][0]]


def test_frozen_design_runtime_and_scope():
    from generate_qwen3_17b_selected_method_posterity import build_payload as old
    original = next(c for c in old()["algos"]["AC-ALG1"] if c["cell_id"] == CONTROL_CELL)
    assert SEEDS == (1201,1213,1217)
    assert len(CELL_ORDER)*len(SEEDS) == 6
    ignored = {"cell_id", "algorithm_profile", "proposal_continuation_mode"}
    for cell in build_payload()["algos"]["AC-ALG1"]:
        assert {k:v for k,v in cell.items() if k not in ignored} == {k:v for k,v in original.items() if k not in ignored}
    for config in runtime_configs():
        alg._validate_ac_alg1_run_config(config, diagnostics_fn=lambda _: None, diagnostics_probe_fn=None)
        for change in ({"proposal_continuation_mode":"bad"}, {"responsibility_answer_policy":"frozen_base"},
                       {"proposal_temperature":1.2}, {"responsibility_ess_floor":.5},
                       {"policy_kl_coef":.01}, {"G_answer_only":8}, {"inner_steps":2},
                       {"proposal_prompt":"question"}, {"responsibility_prior_exponent":.5}):
            with pytest.raises(ValueError):
                alg._validate_ac_alg1_run_config(replace(config, **change), diagnostics_fn=lambda _: None, diagnostics_probe_fn=None)
