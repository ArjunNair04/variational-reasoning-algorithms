"""Validate every saved token/factor and summarize the pre-training ranking audit."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from audit_q5_prior_segments_pretraining import SEEDS, sha256, softmax
from score_q5_prior_segments import AUDIT_ID, MODEL_ID, MODEL_REVISION, SCOPES, SETTINGS, archived_supports, four_way, verify_inputs


CONTRASTS = (("early_half","joint"),("late_half","joint"),("uniform075","joint"),
             ("early_half","late_half"),("early_half","uniform075"),("late_half","uniform075"))


def verify_record(record):
    factors=[]
    for trace in record["traces"]:
        ids=trace["token_ids"]
        pos=trace["score_positions"]
        answer=set(trace["answer_positions"])
        lp=np.asarray(trace["token_logprobs"],dtype=float)
        if (len(lp)!=len(pos) or not pos or pos != list(range(pos[0],len(ids)))
                or pos[0]<1 or not np.isfinite(lp).all() or (lp>1e-6).any()
                or not answer or not answer <= set(pos) or len(ids)-1 not in answer):
            raise ValueError("invalid token tape or answer/EOS mask")
        lookup=dict(zip(pos,lp))
        expected={"joint":float(lp.sum())}
        for scope in SCOPES:
            masks=trace["segment_positions"][scope]
            head,tail,marker=(masks[k] for k in ("head","tail","fixed_marker"))
            all_positions=head+tail+marker+list(answer)
            if len(set(all_positions))!=len(all_positions) or set(all_positions)!=set(pos):
                raise ValueError("segment masks do not partition scored tokens")
            if head+tail != sorted(head+tail) or len(head)!=(len(head)+len(tail))//2:
                raise ValueError("not the declared ordered halfway split")
            if scope=="historical_prior" and marker:
                raise ValueError("historical prior cannot have fixed marker positions")
            if scope=="reasoning_marker_fixed" and (not marker or max(head+tail)>=min(marker) or max(marker)>=min(answer)):
                raise ValueError("reasoning, marker and answer boundary order changed")
            expected[scope]={"answer":sum(lookup[p] for p in answer),
                "head":sum(lookup[p] for p in head),"tail":sum(lookup[p] for p in tail),
                "fixed_marker":sum(lookup[p] for p in marker)}
            for name,value in expected[scope].items():
                if not np.isclose(value,trace["factors"][scope][name],atol=1e-8,rtol=0):
                    raise ValueError("token tape does not reconstruct factors")
        if not np.isclose(expected["joint"],trace["factors"]["joint"],atol=1e-8,rtol=0):
            raise ValueError("joint factor mismatch")
        factors.append(expected)
    recomputed=four_way(factors)
    for scope in SCOPES:
        for name in SETTINGS:
            for key in ("weights","logits"):
                if not np.allclose(record["comparisons"][scope][name][key],recomputed[scope][name][key],atol=1e-9,rtol=0):
                    raise ValueError("saved ranking arithmetic mismatch")
    return recomputed


def summarize(records):
    summary={}
    for scope in SCOPES:
        cells={}
        for name in SETTINGS:
            w=[np.array(r["comparisons"][scope][name]["weights"]) for r in records]
            cells[name]={"mean_ess":float(np.mean([1/(x@x) for x in w])),
                         "mean_largest_weight":float(np.mean([x.max() for x in w]))}
        contrasts={}
        for left,right in CONTRASTS:
            values=[(np.array(r["comparisons"][scope][left]["weights"]),
                     np.array(r["comparisons"][scope][right]["weights"])) for r in records]
            contrasts[f"{left}_vs_{right}"]={"top_changes":sum(bool(np.argmax(a)!=np.argmax(b)) for a,b in values),
                "supports":len(values),"mean_total_variation":float(np.mean([np.abs(a-b).sum()/2 for a,b in values]))}
        summary[scope]={"cells":cells,"contrasts":contrasts}
    summary["marker_scope_top_changes"]={name:sum(bool(
        np.argmax(r["comparisons"][SCOPES[0]][name]["weights"]) !=
        np.argmax(r["comparisons"][SCOPES[1]][name]["weights"])) for r in records) for name in SETTINGS}
    summary["token_length_mismatch_rows"]=sum(t["original_objective_tokens"]!=t["reconstructed_objective_tokens"] for r in records for t in r["traces"])
    return summary


def analyze(source, root, manifest_path, expected_commit, job, logs, output):
    if output.exists():
        raise ValueError("analysis output must be new")
    manifest=json.loads(manifest_path.read_text())
    all_records=[]; receipts=[]
    for index,seed in enumerate(SEEDS,1):
        contract=verify_inputs(source,manifest,seed,require_adapter=False)
        _,expected=archived_supports(source,contract,seed)
        directory=root/f"seed_{seed}"
        receipt=json.loads((directory/"complete.json").read_text())
        for key,value in {"status":"complete","audit_id":AUDIT_ID,"seed":seed,
                          "manifest_sha256":sha256(manifest_path),"execution_commit":expected_commit,
                          "scheduler_job":job,"scheduler_task":str(index),
                          "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
                          "adapter_files":contract["adapter"],
                          "optimizer_steps":0,"new_generations":0,"official_test_accessed":False}.items():
            if receipt.get(key)!=value:
                raise ValueError(f"invalid completion receipt: {seed} {key}")
        log=(logs/f"qwen3_q5_segment_scoring.{job}.{index}.log").read_text()
        if log.count("Q5_SEGMENT_SCORING_COMPLETE ")!=1 or re.search(r"Traceback|CUDA out of memory|Disk quota exceeded|ERROR:",log):
            raise ValueError("missing terminal log or first failure signature")
        data_path=directory/receipt["output"]["path"]
        if sha256(data_path)!=receipt["output"]["sha256"] or data_path.stat().st_size!=receipt["output"]["size"]:
            raise ValueError("output checksum mismatch")
        with gzip.open(data_path,"rt") as stream:
            records=[json.loads(line) for line in stream if line.strip()]
        if len(records)!=128 or receipt["supports"]!=128:
            raise ValueError("incomplete support coverage")
        if receipt["traces"]!=sum(len(r["traces"]) for r in records):
            raise ValueError("trace count mismatch")
        for record,group in zip(records,expected):
            if (record["seed"]!=seed or record["round"]!=group["round"] or record["pid"]!=group["pid"]
                    or record["prompt_sha256"]!=group["prompt"]["canonical_prompt_sha256"]
                    or [r["trace_id"] for r in record["traces"]]!=[r["trace"]["trace_id"] for r in group["candidates"]]):
                raise ValueError("candidate support or canonical prompt changed")
            for trace, candidate in zip(record["traces"], group["candidates"]):
                if trace["text_sha256"] != hashlib.sha256(candidate["sample"]["text"].encode()).hexdigest():
                    raise ValueError("candidate text provenance changed")
            verify_record(record)
        all_records.extend(records); receipts.append(receipt)
    summary={"audit_id":AUDIT_ID,"status":"validated_token_resolved_scoring",
        "supports":len(all_records),"traces":sum(len(r["traces"]) for r in all_records),
        "overall":summarize(all_records),"per_seed":{str(s):summarize([r for r in all_records if r["seed"]==s]) for s in SEEDS},
        "gpu_hours":sum(r["seconds"] for r in receipts)/3600,
        "training_may_start":False,"reason":"Ranking audit complete; resolve and document marker boundary before any training release.",
        "checkpoint_scope":manifest["checkpoint"],"tokenization":manifest["tokenization"],
        "execution_commit":expected_commit,"job":job,"manifest_sha256":sha256(manifest_path),
        "receipts":receipts}
    output.mkdir(parents=True)
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("per_seed","receipts")},indent=2))


def main():
    p=argparse.ArgumentParser()
    for name in ("source","results","manifest","logs","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--execution-commit",required=True); p.add_argument("--job",required=True)
    a=p.parse_args()
    analyze(a.source,a.results,a.manifest,a.execution_commit,a.job,a.logs,a.output)


if __name__=="__main__":
    main()
