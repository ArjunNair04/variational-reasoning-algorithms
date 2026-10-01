"""Frozen analysis of fixed-weight continuation kernels versus existing Q5."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lm_study"))
from ac_alg1_component_kernel import token_hash
from generate_qwen3_17b_q5_component_kernel import (
    RUN_ID, CELL, CONTROL_CELL, CONTROL_COMMIT, CONTROL_JOB, CONTROL_RUN,
    CONTROL_CONFIG_SHA256, SEEDS, build_payload,
)
from generate_qwen3_17b_q5_prefix_continuations import (
    build_payload as control_payload, BASELINES, SUPPORT_SHA256, CHECKPOINTS,
)
from analyze_qwen3_q5_prefix_continuations import continuation_rows
from analyze_qwen3_q5_prior_exponent import load_cell, mechanism_rows, METRICS
from analyze_qwen3_jepo_comparator import (
    _artifact, _read_jsonl_gz, _require_non_access, _finite, _normalized_auc,
)
from result_contract import atomic_write_json
from run_yaml import _prepare_cells
from validate_yaml_run import validate as validate_outputs


def validate_design(path):
    payload = yaml.safe_load(path.read_text())
    if payload != build_payload():
        raise ValueError("frozen component-kernel design changed")
    return payload, _prepare_cells(payload, only=None, run_id=RUN_ID, defaults=payload["defaults"])


def kernel_rows(diagnostics):
    if [r["completed_rounds"] for r in diagnostics] != list(range(1,33)):
        raise ValueError("expected 32 complete kernel diagnostic rounds")
    rows, cumulative = [], 0
    for row in diagnostics:
        gen, compute = row["generation"], row["compute"]
        k = gen["component_kernel"]
        if (k["mode"] != "fixed_weight_half_prefix" or k["epsilon"] != .25
                or k["draws_per_parent"] != 1 or k["weights_rescored"] is not False
                or k["children_persisted"] is not False):
            raise ValueError("kernel semantics changed")
        if (gen["proposal_prompt"] != "answer_derive" or gen["proposal_policy"] != "current"
                or gen["proposal_temperature"] != 1. or gen["sampling_intervention"] is not None
                or gen["buffer_proposals_this_round"] != 64):
            raise ValueError("original proposal protocol changed")
        r = row["responsibilities"]
        if (r["answer_policy"] != "current" or r["policy"] != "current" or r["score"] != "joint"
                or r["ess_floor_fraction"] != 0. or r["prior_exponent"] != 1.):
            raise ValueError("original Q5 E-step changed")
        retained = {t["trace_id"]: t for t in r["traces"]}
        if any(str(t.get("source", "")).startswith("component_kernel")
               or str(t["trace_id"]).startswith("kernel:") for t in r["traces"]):
            raise ValueError("kernel child entered persistent buffer")
        groups, seen = {}, set()
        for p in k["parents"]:
            key = (p["pid"],p["parent_index"])
            if key in seen or p["parent_trace_id"] not in retained:
                raise ValueError("duplicate or unknown kernel parent")
            seen.add(key); groups.setdefault(p["pid"],[]).append(p)
            parent = retained[p["parent_trace_id"]]
            if (parent["pid"] != p["pid"] or parent["buffer_index"] != p["parent_index"]
                    or p["prefix_tokens"] != parent["reasoning_token_count"] // 2):
                raise ValueError("parent identity or halfway boundary changed")
            joint = _finite(p["parent_trace_logprob"],context="parent prior") + _finite(p["parent_answer_logprob"],context="parent answer")
            if not np.isclose(p["parent_logit"],joint,atol=2e-3,rtol=2e-6):
                raise ValueError("parent weight was not the Q5 joint score")
            weight = _finite(p["parent_weight"],context="parent weight")
            if weight <= 0 or not np.isclose(p["parent_mass"],.75*weight,atol=1e-9,rtol=1e-6) or not np.isclose(p["kernel_mass"],.25*weight,atol=1e-9,rtol=1e-6):
                raise ValueError("inherited component mass changed")
            prefix = p["prefix_ids"]
            if (len(prefix) != p["prefix_tokens"] or prefix != p["parent_h_ids"][:len(prefix)]
                    or token_hash(prefix) != p["prefix_sha256"]
                    or token_hash(p["parent_h_ids"]) != p["parent_h_sha256"]):
                raise ValueError("native parent/prefix lineage mismatch")
            if p["generated_tokens"]:
                comp = p["completion_ids"]
                if comp[:len(prefix)] != prefix or len(comp) != len(prefix)+p["generated_tokens"] or len(comp)>k["max_new_tokens"]:
                    raise ValueError("generated/copied token accounting mismatch")
            elif p["completion_ids"] or not p["fallback"]:
                raise ValueError("missing kernel draw without deterministic fallback")
            if p["fallback"]:
                if p["child_h_ids"] != p["parent_h_ids"] or p["child_trace_id"] != p["parent_trace_id"]:
                    raise ValueError("fallback lost parent mass")
            elif p["child_h_ids"][:len(prefix)] != prefix or not p["child_trace_id"].startswith("kernel:"):
                raise ValueError("child was not built from the sampled prefix")
            if p["same_as_parent"] != (p["child_h_ids"] == p["parent_h_ids"]):
                raise ValueError("changed-mass flag mismatch")
        if not groups:
            raise ValueError("no active kernel parents")
        for group in groups.values():
            logits = np.array([_finite(p["parent_logit"],context="frozen parent score") for p in group])
            w = np.exp(logits-logits.max()); w /= w.sum()
            if not np.allclose(w,[p["parent_weight"] for p in group],atol=2e-6,rtol=2e-5):
                raise ValueError("inherited weights do not equal frozen Q5 weights")
        aggregates = {
            "generated_draws": sum(p["generated_tokens"]>0 for p in k["parents"]),
            "generated_tokens": sum(p["generated_tokens"] for p in k["parents"]),
            "copied_prefix_tokens": sum(p["prefix_tokens"] for p in k["parents"] if p["generated_tokens"]),
            "fallback_mass": sum(p["kernel_mass"] for p in k["parents"] if p["fallback"]),
            "changed_mass": sum(p["kernel_mass"] for p in k["parents"] if not p["same_as_parent"]),
            "kernel_mass": .25*len(groups),
        }
        if any(not np.isclose(k[name],value,atol=1e-6,rtol=1e-6) for name,value in aggregates.items()):
            raise ValueError("kernel aggregate accounting mismatch")
        cumulative += 64+k["generated_draws"]
        if gen["this_round"] != 64+k["generated_draws"] or gen["cumulative"] != cumulative:
            raise ValueError("total generation count omits kernel draws")
        steps = row["inner_m_step"]["steps"]
        if len(steps)!=1 or steps[0]["component_kernel"]!=k or steps[0]["status"]!="accepted":
            raise ValueError("kernel M-step receipt mismatch")
        tokens=compute["tokens"]
        for key in ("generated","backward","backward_eos"):
            if not isinstance(tokens[key],int) or tokens[key]<=0:
                raise ValueError("invalid token cost")
        if tokens["generated"] < k["generated_tokens"] or tokens["llm_generated_total"]!=tokens["generated"]:
            raise ValueError("kernel generation cost missing")
        rows.append(dict(round=row["completed_rounds"],train_draws=gen["this_round"],
            generated_tokens=tokens["generated"],backward_tokens=tokens["backward"],
            kernel_draws=k["generated_draws"],kernel_generated_tokens=k["generated_tokens"],
            copied_prefix_tokens=k["copied_prefix_tokens"],fallback_mass_per_question=k["fallback_mass"]/len(groups),
            changed_mass_per_question=k["changed_mass"]/len(groups),kernel_seconds=k["elapsed_seconds"]))
    return rows


def contrasts(frame):
    if len(frame)!=6 or set(zip(frame.cell,frame.seed))!={(c,s) for c in (CONTROL_CELL,CELL) for s in SEEDS}:
        raise ValueError("six unique paired coordinates required")
    data=frame.set_index(["cell","seed"])
    rows=[]
    for metric in METRICS:
        delta=np.array([data.loc[(CELL,s),metric]-data.loc[(CONTROL_CELL,s),metric] for s in SEEDS])
        if not np.isfinite(delta).all():
            raise ValueError("nonfinite paired metric")
        margin=4.302652729911275*delta.std(ddof=1)/np.sqrt(3)
        rows.append(dict(metric=metric,mean_difference_pp=100*delta.mean(),
            descriptive_low_pp=100*(delta.mean()-margin),descriptive_high_pp=100*(delta.mean()+margin),
            per_seed_difference_pp=json.dumps((100*delta).tolist())))
    return pd.DataFrame(rows)


def validate_log(path):
    text=path.read_text(errors="replace")
    if "=== done in" not in text or re.search(r"Traceback|CUDA out of memory|OutOfMemoryError|No space left|Disk quota exceeded|FAILED|ERROR:",text):
        raise ValueError(f"failed or incomplete payload log: {path}")


def validate_results(config_path, commit, config_sha, job, logs, control_logs, control_config):
    payload,cells=validate_design(config_path)
    if hashlib.sha256(config_path.read_bytes()).hexdigest()!=config_sha or not re.fullmatch(r"[0-9a-f]{40}",commit) or not job.isdigit():
        raise ValueError("immutable execution identity missing or changed")
    if hashlib.sha256(control_config.read_bytes()).hexdigest()!=CONTROL_CONFIG_SHA256:
        raise ValueError("reused control YAML hash changed")
    cp=control_payload()
    if yaml.safe_load(control_config.read_text())!=cp:
        raise ValueError("reused control design changed")
    control=_prepare_cells(cp,only=None,run_id=CONTROL_RUN,defaults=cp["defaults"])[0]
    for i in range(1,4):
        validate_log(logs/f"qwen3_q5_kernel.{job}.{i}.log")
        validate_log(control_logs/f"qwen3_q5_prefix.{CONTROL_JOB}.{i}.log")
    problems=validate_outputs(config_path,str(logs/f"qwen3_q5_kernel.{job}.*.log"))
    if problems:
        raise ValueError("; ".join(problems))
    root=Path(payload["defaults"]["out"]).expanduser()
    if len(list(root.glob("complete_*.json")))!=3:
        raise ValueError("expected three new completion receipts")
    metrics,mechanism,posterior=[],[],[]
    for seed in SEEDS:
        schedules=[]; environments=[]
        for name,cell,base,rid,rev in (
            (CONTROL_CELL,control,Path(cp["defaults"]["out"]).expanduser(),CONTROL_RUN,CONTROL_COMMIT),
            (CELL,cells[0],root,RUN_ID,commit),
        ):
            receipt,result,values,ids=load_cell(base,cell,seed,rid,rev)
            _require_non_access(result,context="cell result")
            if hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()!=SUPPORT_SHA256:
                raise ValueError("validation support changed")
            sweep=pd.read_csv(base/f"sweep_gsm8k__{cell.tag}_seed{seed}.csv")
            if len(sweep)!=1 or int(sweep.iloc[0]["seed"])!=seed or sweep.iloc[0]["method"]!=cell.method:
                raise ValueError("sweep row mismatch")
            params=json.loads(result["params"])
            environments.append({key:params["env"][key] for key in ("python","torch","transformers","peft","cuda_runtime","cudnn","gpu")})
            checkpoints=_read_jsonl_gz(_artifact(base,receipt,"checkpoint_eval_"))
            if tuple(c["completed_rounds"] for c in checkpoints)!=CHECKPOINTS:
                raise ValueError("checkpoint schedule changed")
            for c in checkpoints:
                _require_non_access(c["metrics"],context="checkpoint")
            for metric,key,endpoint in (("extracted_auc","test_acc_legacy","final_extracted"),("strict_auc","test_acc_strict","final_strict")):
                vals=[_finite(c["metrics"][key],context=key) for c in checkpoints]
                if any(not 0<=v<=1 for v in vals) or abs(vals[-1]-values[endpoint])>1e-12:
                    raise ValueError("invalid metric or final endpoint mismatch")
                values[metric]=_normalized_auc(BASELINES[seed][endpoint],vals)
            ds=_read_jsonl_gz(_artifact(base,receipt,"training_diagnostics_"))
            for r in ds:
                expected=dict(run_id=rid,model=cell.model,seed=seed,method=cell.method,tag=f"{cell.tag}_seed{seed}",
                    answer_event_mode="strict_terminal_marker",answer_target_termination="eos",
                    algorithm_profile="q5_component_kernel" if name==CELL else "q5_prefix_continuations")
                if any(r.get(k)!=v for k,v in expected.items()):
                    raise ValueError("diagnostic identity mismatch")
            schedules.append([r["minibatch"] for r in ds])
            mr=kernel_rows(ds) if name==CELL else continuation_rows(ds,"independent")
            pr=mechanism_rows(ds,1.)
            draws=sum(r["train_draws"] for r in mr) if name==CELL else 2048
            if int(result["optimizer_steps"])!=32 or int(result["train_llm_gen"])!=draws:
                raise ValueError("training budget/accounting mismatch")
            metrics.append(dict(cell=name,seed=seed,**values,train_llm_gen=draws,optimizer_steps=32,
                accelerator_hours=_finite(result["accelerator_hours"],context="compute"),
                generated_tokens=sum(r["generated_tokens"] for r in mr),backward_tokens=sum(r["backward_tokens"] for r in mr)))
            mechanism.extend(dict(cell=name,seed=seed,**r) for r in mr)
            posterior.extend(dict(cell=name,seed=seed,**r) for r in pr)
        if schedules[0]!=schedules[1] or environments[0]!=environments[1]:
            raise ValueError("paired question schedule or runtime differs")
    frame=pd.DataFrame(metrics)
    marker=dict(status="ok",run_id=RUN_ID,execution_commit=commit,configuration_sha256=config_sha,
        source_job_id=job,task_count=3,control_job_id=CONTROL_JOB,control_tasks=[1,2,3],
        official_test_used=False,validated_at=datetime.now(timezone.utc).isoformat())
    return marker,frame,pd.DataFrame(mechanism),pd.DataFrame(posterior),contrasts(frame)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--validate-design-only",action="store_true")
    for name in ("logs","control-logs","control-config","output-dir","marker"):
        parser.add_argument("--"+name,type=Path)
    for name in ("expected-commit","expected-config-sha256","source-job-id"):
        parser.add_argument("--"+name)
    args=parser.parse_args(); validate_design(args.config)
    if args.validate_design_only:
        print(f"{RUN_ID}: one new cell x three seeds; reuse 7483302.1-3")
        return
    if any(getattr(args,k) is None for k in ("logs","control_logs","control_config","output_dir","marker","expected_commit","expected_config_sha256","source_job_id")):
        parser.error("both log directories, control config and execution identity are required")
    marker,frame,mechanism,posterior,paired=validate_results(args.config,args.expected_commit,
        args.expected_config_sha256,args.source_job_id,args.logs,args.control_logs,args.control_config)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    frame.to_csv(args.output_dir/"seed_metrics.csv",index=False)
    means=frame.groupby("cell",sort=False).mean(numeric_only=True).drop(columns="seed")
    means.to_csv(args.output_dir/"method_summary.csv")
    mechanism.to_csv(args.output_dir/"kernel_diagnostics.csv",index=False)
    posterior.to_csv(args.output_dir/"posterior_diagnostics.csv.gz",index=False)
    paired.to_csv(args.output_dir/"paired_contrasts.csv",index=False)
    delta=means.loc[CELL]-means.loc[CONTROL_CELL]
    nominee=CELL if delta.final_extracted>0 and delta.final_strict>=0 else None
    atomic_write_json(args.output_dir/"analysis.json",dict(marker=marker,nominee=nominee,
        automatic_followup=False,metric_order=METRICS,evidence="Three-seed fixed-weight kernel screen; extra compute; reused concurrent controls."))
    atomic_write_json(args.marker,marker)
    print(json.dumps(marker,sort_keys=True))


if __name__=="__main__":
    main()
