from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from analyze_qwen3_q5_prefix_continuations import continuation_rows, paired_contrasts, validate_design
from generate_qwen3_17b_q5_prefix_continuations import CELL_ORDER, SEEDS, build_payload


def diagnostics(mode):
    output = []
    for k in range(32):
        parents = [None if i % 16 < 8 else i-8 for i in range(64)]
        prefix = [0 if p is None else 4 for p in parents]
        ids = [f"r{k}:t{i}" for i in range(64)]
        branch = dict(mode=mode, root_count=32, continuation_count=32, fallback_count=0,
            parent_indices=parents, prefix_tokens=prefix, generated_tokens=[10]*64,
            completion_tokens=[10+p for p in prefix], max_new_tokens=512,
            prefix_sha256=[None if p is None else "a"*64 for p in parents],
            fallback_reasons=[None]*64, trace_ids=ids,
            parent_trace_ids=[None if p is None else ids[p] for p in parents])
        output.append(dict(completed_rounds=k+1, buffer_set_duplicates_this_round=0,
            generation=dict(this_round=64, cumulative=64*(k+1), proposal_prompt="answer_derive",
                proposal_policy="current", proposal_temperature=1., sampling_intervention=branch if mode=="prefix_half" else None),
            responsibilities=dict(answer_policy="current", policy="current", score="joint", ess_floor_fraction=0.),
            compute=dict(tokens=dict(generated=640, backward=700, backward_eos=16),
                timings_seconds=dict(generation=10.,m_step=1.))))
    return output


@pytest.mark.parametrize("mode", ["independent", "prefix_half"])
def test_mechanism_accounting(mode):
    rows = continuation_rows(diagnostics(mode), mode)
    assert len(rows) == 32
    assert sum(r["generated_tokens"] for r in rows) == 20480
    assert sum(r["copied_prefix_tokens"] for r in rows) == (4096 if mode=="prefix_half" else 0)


@pytest.mark.parametrize("field,value", [("root_count", 31), ("fallback_count", 1),
    ("continuation_count", 0), ("max_new_tokens", 2)])
def test_tampered_totals_rejected(field,value):
    ds = diagnostics("prefix_half")
    ds[0]["generation"]["sampling_intervention"][field] = value
    with pytest.raises(ValueError):
        continuation_rows(ds, "prefix_half")


@pytest.mark.parametrize("field,value", [("parent_indices", 1), ("parent_trace_ids", "wrong"),
    ("prefix_tokens", 0), ("generated_tokens", 14), ("completion_tokens", 15),
    ("prefix_sha256", None), ("fallback_reasons", "missing")])
def test_tampered_child_rejected(field,value):
    ds = diagnostics("prefix_half")
    ds[0]["generation"]["sampling_intervention"][field][8] = value
    with pytest.raises(ValueError):
        continuation_rows(ds, "prefix_half")


def test_missing_round_and_contaminated_control_rejected():
    with pytest.raises(ValueError):
        continuation_rows(diagnostics("independent")[:-1], "independent")
    with pytest.raises(ValueError):
        continuation_rows(diagnostics("prefix_half"), "independent")


def test_paired_contrasts_and_duplicate_rejection():
    frame = pd.DataFrame([dict(cell=c,seed=s,final_extracted=.7+i*.01,
        final_strict=.6+i*.01,extracted_auc=.75+i*.01,strict_auc=.65+i*.01)
        for i,c in enumerate(CELL_ORDER) for s in SEEDS])
    contrasts = paired_contrasts(frame)
    assert contrasts.mean_difference_pp.to_numpy() == pytest.approx([1.]*4)
    assert contrasts.descriptive_low_pp.to_numpy() == pytest.approx([1.]*4)
    for bad in (frame.iloc[:-1], pd.concat([frame,frame.iloc[:1]])):
        with pytest.raises(ValueError):
            paired_contrasts(bad)


def test_exact_design_and_wrappers(tmp_path):
    path = tmp_path/"config.yaml"
    payload = build_payload()
    path.write_text(yaml.safe_dump(payload))
    assert len(validate_design(path)[1]) == 2
    payload["defaults"]["rounds"] = 64
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError):
        validate_design(path)
    root = Path(__file__).resolve().parents[1]
    runner = (root/"lm_study/run_qwen3_17b_q5_prefix_continuations_ucl.sh").read_text()
    assert "#$ -t 1-6" in runner and "#$ -tc" not in runner
    assert "EXPECTED_COMMIT" in runner and "EXPECTED_CONFIG_SHA256" in runner
    submitter = (root/"lm_study/submit_qwen3_17b_q5_prefix_continuations_ucl.sh").read_text()
    assert submitter.index("--runtime-check") < submitter.index("qsub -h")
    assert "--shard \"$cell\" --nshard 2" in submitter
