"""Frozen analysis of learned-distribution M-steps versus receipt-bound Q5."""

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
from ac_alg1_learned_posterior import stream_seed
from generate_qwen3_17b_q5_learned_posterior import (
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
        raise ValueError("frozen learned-posterior design changed")
    return payload, _prepare_cells(payload, only=None, run_id=RUN_ID, defaults=payload["defaults"])


def posterior_rows(diagnostics):
    if [r["completed_rounds"] for r in diagnostics] != list(range(1,33)):
        raise ValueError("expected 32 posterior diagnostic rounds")
    rows=[]
    for round_index,row in enumerate(diagnostics):
        gen,compute=row["generation"],row["compute"]
        p=gen["learned_posterior"]
        required=dict(mode="distilled_completion_pushforward",posterior_updates=1,
            cumulative_posterior_updates=round_index+1,posterior_lr=1e-5,draws_per_question=16,
            main_adapter="default",posterior_adapter="learned_posterior",
            weights_rescored=False,children_persisted=False,generated_draws=64,max_new_tokens=256)
        if any(p.get(k)!=v for k,v in required.items()):
            raise ValueError("learned posterior contract changed")
        seed=int(row["seed"])
        for key,purpose in (("fit_rng_seed","fit"),("sample_rng_seed","sample")):
            if p[key]!=stream_seed(seed,round_index,purpose):
                raise ValueError("posterior RNG identity mismatch")
        if (gen["proposal_prompt"]!="answer_derive" or gen["proposal_policy"]!="current"
                or gen["proposal_temperature"]!=1. or gen["buffer_proposals_this_round"]!=64
                or gen["sampling_intervention"] is not None):
            raise ValueError("teacher proposal protocol changed")
        r=row["responsibilities"]
        if (r["score"]!="joint" or r["policy"]!="current" or r["answer_policy"]!="current"
                or r["ess_floor_fraction"]!=0 or r["prior_exponent"]!=1):
            raise ValueError("Q5 teacher changed")
        retained={t["trace_id"]:t for t in r["traces"]}
        if any(str(k).startswith("posterior:") for k in retained):
            raise ValueError("learned draws entered teacher buffer")
        teachers={}; seen=set()
        for t in p["teachers"]:
            if t["trace_id"] in seen or t["trace_id"] not in retained:
                raise ValueError("duplicate or unknown posterior teacher")
            seen.add(t["trace_id"])
            if retained[t["trace_id"]]["pid"]!=t["pid"]:
                raise ValueError("teacher question mismatch")
            if token_hash(t["target_ids"])!=t["target_sha256"] or not t["target_ids"]:
                raise ValueError("teacher native target mismatch")
            if not np.isclose(t["logit"],t["trace_logprob"]+t["answer_logprob"],atol=2e-3,rtol=2e-6):
                raise ValueError("teacher is not Q5 joint")
            teachers.setdefault(t["pid"],[]).append(t)
        pids=row["minibatch"]["answer_only_pids"]
        if len(pids)!=4 or set(teachers)!=set(pids):
            raise ValueError("teacher question coverage mismatch")
        for group in teachers.values():
            logits=np.array([_finite(t["logit"],context="teacher logit") for t in group])
            weights=np.exp(logits-logits.max()); weights/=weights.sum()
            if not np.allclose(weights,[t["weight"] for t in group],atol=2e-6,rtol=2e-5):
                raise ValueError("teacher fit changed Q5 weights")
        eos_ids={t["target_ids"][-1] for t in p["teachers"]}
        if len(eos_ids)!=1:
            raise ValueError("teacher EOS identity mismatch")
        samples={}; seen=set()
        for s in p["samples"]:
            if s["trace_id"] in seen or not s["trace_id"].startswith(f"posterior:{round_index}:{s['pid']}:"):
                raise ValueError("posterior sample identity mismatch")
            seen.add(s["trace_id"]); samples.setdefault(s["pid"],[]).append(s)
            if s["weight"]!=1/16 or s["mapping"] not in {"native_first_marker","canonical_boundary_repair"}:
                raise ValueError("posterior samples were reweighted or filtered")
            if not 0<len(s["completion_ids"])<=256 or len(s["completion_ids"])!=s["generated_tokens"]:
                raise ValueError("posterior generation cost mismatch")
            if s["target_ids"][-1] not in eos_ids or s["target_ids"][:len(s["h_ids"])]!=s["h_ids"]:
                raise ValueError("posterior target lost h or EOS")
        if set(samples)!=set(pids) or any(len(v)!=16 for v in samples.values()):
            raise ValueError("posterior draw coverage mismatch")
        aggregates=dict(generated_tokens=sum(s["generated_tokens"] for s in p["samples"]),
            repaired_draws=sum(s["mapping"]!="native_first_marker" for s in p["samples"]),
            posterior_backward_tokens=sum(len(t["target_ids"]) for t in p["teachers"]),
            posterior_backward_eos_tokens=len(p["teachers"]))
        if any(p[k]!=v for k,v in aggregates.items()):
            raise ValueError("posterior cost totals mismatch")
        if gen["this_round"]!=128 or gen["cumulative"]!=128*(round_index+1):
            raise ValueError("total generation accounting mismatch")
        steps=row["inner_m_step"]["steps"]
        if len(steps)!=1 or steps[0]["status"]!="accepted" or steps[0]["learned_posterior"]!=p:
            raise ValueError("posterior M-step receipt mismatch")
        tokens=compute["tokens"]
        if (tokens["backward"]!=tokens["main_backward"]+p["posterior_backward_tokens"]
                or tokens["backward_eos"]!=tokens["main_backward_eos"]+p["posterior_backward_eos_tokens"]
                or tokens["main_backward"]!=sum(len(s["target_ids"]) for s in p["samples"])
                or tokens["main_backward_eos"]!=64
                or tokens["generated"]<p["generated_tokens"]
                or tokens["llm_generated_total"]!=tokens["generated"]):
            raise ValueError("main/posterior token accounting mismatch")
        for key in ("posterior_fit_seconds","elapsed_seconds","posterior_gradient_norm","posterior_parameter_delta_norm"):
            if _finite(p[key],context=key)<=0:
                raise ValueError("missing posterior fit/sampling")
        _finite(p["teacher_objective"],context="teacher objective")
        unique=np.mean([len({tuple(s["h_ids"]) for s in group})/16 for group in samples.values()])
        matches=np.mean([tuple(s["h_ids"]) in {tuple(t["h_ids"]) for t in teachers[s["pid"]]}
            for s in p["samples"]])
        rows.append(dict(round=round_index+1,train_draws=128,
            generated_tokens=tokens["generated"],backward_tokens=tokens["backward"],
            main_backward_tokens=tokens["main_backward"],posterior_backward_tokens=p["posterior_backward_tokens"],
            posterior_fit_seconds=p["posterior_fit_seconds"],posterior_generation_seconds=p["elapsed_seconds"],
            posterior_unique_fraction=unique,boundary_repair_fraction=p["repaired_draws"]/64,
            posterior_teacher_match_fraction=matches,
            posterior_raw_extracted_accuracy=np.mean([s["raw_extracted_correct"] for s in p["samples"]]),
            posterior_raw_strict_accuracy=np.mean([s["raw_strict_correct"] for s in p["samples"]]),
            posterior_raw_strict_format=np.mean([s["raw_strict_format"] for s in p["samples"]]),
            posterior_parameter_delta_norm=p["posterior_parameter_delta_norm"],
            teacher_objective=p["teacher_objective"],posterior_gradient_norm=p["posterior_gradient_norm"]))
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
        validate_log(logs/f"qwen3_q5_posterior.{job}.{i}.log")
        validate_log(control_logs/f"qwen3_q5_prefix.{CONTROL_JOB}.{i}.log")
    problems=validate_outputs(config_path,str(logs/f"qwen3_q5_posterior.{job}.*.log"))
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
            if name==CELL:
                posterior_artifacts=[a for a in receipt["artifacts"] if "/learned_posterior/" in a["path"]]
                if len(posterior_artifacts)!=2:
                    raise ValueError("expected checksum-bound posterior adapter/config")

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
                    algorithm_profile="q5_learned_posterior" if name==CELL else "q5_prefix_continuations")
                if any(r.get(k)!=v for k,v in expected.items()):
                    raise ValueError("diagnostic identity mismatch")
            schedules.append([r["minibatch"] for r in ds])
            mr=posterior_rows(ds) if name==CELL else continuation_rows(ds,"independent")
            pr=mechanism_rows(ds,1.)
            draws=sum(r["train_draws"] for r in mr) if name==CELL else 2048
            if int(result["optimizer_steps"])!=32 or int(result["train_llm_gen"])!=draws:
                raise ValueError("training budget/accounting mismatch")
            metrics.append(dict(cell=name,seed=seed,**values,train_llm_gen=draws,optimizer_steps=32,
                posterior_optimizer_steps=32 if name==CELL else 0,total_optimizer_steps=64 if name==CELL else 32,
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
    mechanism.to_csv(args.output_dir/"learned_posterior_diagnostics.csv",index=False)
    posterior.to_csv(args.output_dir/"posterior_diagnostics.csv.gz",index=False)
    paired.to_csv(args.output_dir/"paired_contrasts.csv",index=False)
    delta=means.loc[CELL]-means.loc[CONTROL_CELL]
    nominee=CELL if delta.final_extracted>0 and delta.final_strict>=0 else None
    atomic_write_json(args.output_dir/"analysis.json",dict(marker=marker,nominee=nominee,
        automatic_followup=False,metric_order=METRICS,evidence="Three-seed learned-distribution replacement; extra compute; reused validated Q5 controls."))
    atomic_write_json(args.marker,marker)
    print(json.dumps(marker,sort_keys=True))


if __name__=="__main__":
    main()
