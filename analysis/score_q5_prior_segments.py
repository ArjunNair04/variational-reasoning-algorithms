"""Fixed-checkpoint scoring of archived Q5 candidates; no generation or optimizer."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from ac_alg1 import _sampled_trace_row
from ac_alg1_prior_segments import split_rationale_mask
from audit_q5_prior_segments_pretraining import CELL, RUN_ID, SEEDS, rank_correlation, sha256, softmax, source_file


AUDIT_ID = "q5_segment_scoring_20260914"
MODEL_ID = "Qwen/Qwen3-1.7B-Base"
MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SETTINGS = {"joint": (1., 1.), "early_half": (.5, 1.),
            "late_half": (1., .5), "uniform075": (.75, .75)}
SCOPES = ("historical_prior", "reasoning_marker_fixed")


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def prepare_manifest(source):
    seeds = {}
    for seed in SEEDS:
        receipt_path = source_file(source, "complete", seed, ".json")
        receipt = read_json(receipt_path)
        if (receipt["status"] != "complete" or receipt["identity"]["run_id"] != RUN_ID
                or receipt["identity"]["seed"] != seed):
            raise ValueError("historical receipt identity mismatch")
        files = {a["path"]: a for a in receipt["artifacts"]}
        selected = {}
        for role, prefix, suffix in (("diagnostics", "training_diagnostics", ".jsonl.gz"),
                                     ("prompts", "prompt_contract", ".json.gz")):
            path = source_file(source, prefix, seed, suffix)
            entry = files[path.name]
            if sha256(path) != entry["sha256"] or path.stat().st_size != entry["size"]:
                raise ValueError("local audit source does not match historical receipt")
            selected[role] = entry
        adapter = [a for a in files.values() if a["path"].endswith(("/adapter_config.json", "/adapter_model.safetensors"))]
        if len(adapter) != 2:
            raise ValueError("expected one final adapter pair")
        seeds[str(seed)] = {"inputs": selected, "adapter": adapter,
                            "receipt": {"path": receipt_path.name, "sha256": sha256(receipt_path),
                                        "size": receipt_path.stat().st_size}}
    return {"schema_version": 1, "audit_id": AUDIT_ID, "source_run": RUN_ID,
            "source_cell": CELL, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "seeds": seeds, "settings": SETTINGS, "scopes": SCOPES,
            "checkpoint": "each seed's receipt-bound FINAL adapter held fixed while rescoring every archived support",
            "tokenization": "retokenize saved generated text once, then use native strict-Q5 reconstruction; not bit-exact historical token replay",
            "scope": "384 archived training supports, all three seeds and all 32 rounds; no evaluation data loaded",
            "optimizer_steps": 0, "new_generations": 0,
            "decision": "No accuracy inference or automatic training release. Inspect marker-boundary sensitivity before approving a training mask."}


def verify_inputs(source, manifest, seed, *, require_adapter):
    if manifest["audit_id"] != AUDIT_ID or manifest["source_run"] != RUN_ID:
        raise ValueError("wrong scoring manifest")
    if manifest["model_id"] != MODEL_ID or manifest["model_revision"] != MODEL_REVISION:
        raise ValueError("model pin changed")
    if {k: tuple(v) for k, v in manifest["settings"].items()} != SETTINGS or tuple(manifest["scopes"]) != SCOPES:
        raise ValueError("four-way design changed")
    contract = manifest["seeds"][str(seed)]
    entries = [contract["receipt"], *contract["inputs"].values()]
    if require_adapter:
        entries += contract["adapter"]
    for item in entries:
        path = source / item["path"]
        if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError(f"audit source checksum mismatch: {path}")
    return contract


def archived_supports(source, contract, seed):
    prompts = read_json(source / contract["inputs"]["prompts"]["path"])
    if (prompts["training_seed"] != seed or prompts["answer_target_termination"] != "eos"
            or prompts["proposal_prompt_mode"] != "answer_derive"):
        raise ValueError("prompt contract changed")
    prompt_rows = {r["pid"]: r for r in prompts["rows"]}
    if set(prompt_rows) != set(range(128)):
        raise ValueError("expected 128 canonical training prompts")
    for row in prompt_rows.values():
        if hashlib.sha256(row["canonical_prompt"].encode()).hexdigest() != row["canonical_prompt_sha256"]:
            raise ValueError("canonical prompt hash mismatch")
        if row["canonical_contains_proposal_answer_hint"]:
            raise ValueError("answer hint leaked into canonical reconstruction")
    task = SimpleNamespace(prompts=[prompt_rows[i]["canonical_prompt"] for i in range(128)],
                           gold_answer=[int(prompt_rows[i]["gold_answer"]) for i in range(128)])
    with gzip.open(source / contract["inputs"]["diagnostics"]["path"], "rt") as stream:
        rounds = [json.loads(line) for line in stream if line.strip()]
    if [r["round"] for r in rounds] != list(range(32)):
        raise ValueError("expected all 32 historical rounds")
    groups = []
    for record in rounds:
        if record["run_id"] != RUN_ID or record["seed"] != seed:
            raise ValueError("diagnostic identity changed")
        samples = {r["trace_id"]: r for r in record["generation"]["samples"]["samples"]}
        by_pid = defaultdict(list)
        for trace in record["responsibilities"]["traces"]:
            sample = samples[trace["trace_id"]]
            if trace["age"] != 0 or trace["pid"] != sample["pid"] or not sample["retained_after_insertion"]:
                raise ValueError("cannot join historical active support to archived text")
            by_pid[trace["pid"]].append({"trace": trace, "sample": sample})
        for pid in record["minibatch"]["answer_only_pids"]:
            if not by_pid[pid]:
                raise ValueError("empty historical support")
            groups.append({"seed": seed, "round": record["round"], "pid": pid,
                           "prompt": prompt_rows[pid], "candidates": by_pid[pid]})
    if len(groups) != 128 or len({r["pid"] for r in groups}) != 128:
        raise ValueError("expected one epoch over 128 questions")
    return task, groups


def reconstruct(tok, task, group):
    rows = []
    for candidate in group["candidates"]:
        trace, sample = candidate["trace"], candidate["sample"]
        comp = torch.tensor(tok(sample["text"], add_special_tokens=False).input_ids, dtype=torch.long)
        row = _sampled_trace_row(tok, task, group["pid"], comp, torch.ones_like(comp, dtype=torch.bool),
                                 sample["text"], group["round"], trace["source"], trace_id=trace["trace_id"],
                                 proposal_tokens=trace["proposal_tokens"],
                                 answer_event_mode="strict_terminal_marker", answer_target_termination="eos")
        if row is None:
            raise ValueError(f"retokenized trace cannot preserve answer boundary: {trace['trace_id']}")
        if row.ids[-1].item() != tok.eos_token_id or not row.ans[-1]:
            raise ValueError("EOS answer mask missing")
        rows.append(row)
    return rows


def masks_for_row(row):
    prior = (row.span & ~row.ans).unsqueeze(0)
    positions = prior[0].nonzero().flatten()
    n_reason = row.reasoning_token_count
    if n_reason is None or not 0 < n_reason < len(positions):
        raise ValueError("missing nonempty reasoning or marker suffix")
    reasoning = torch.zeros_like(prior)
    reasoning[0, positions[:n_reason]] = True
    marker = prior & ~reasoning
    return {"historical_prior": (*split_rationale_mask(prior), torch.zeros_like(prior)),
            "reasoning_marker_fixed": (*split_rationale_mask(reasoning), marker)}


def four_way(factors):
    result = {}
    for scope in SCOPES:
        a, h, t, m = (np.array([r[scope][name] for r in factors]) for name in ("answer", "head", "tail", "fixed_marker"))
        result[scope] = {}
        for name, (alpha, beta) in SETTINGS.items():
            score = a + m + alpha*h + beta*t
            result[scope][name] = {"logits": score.tolist(), "weights": softmax(score).tolist()}
        if not np.allclose(result[scope]["joint"]["logits"],
                           [r["joint"] for r in factors], atol=1e-5, rtol=0):
            raise ValueError("factor partition fails to reconstruct joint score")
    return result


def score_seed(source, manifest_path, output, seed, *, preflight=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from methods_lm import cuda_dtype, token_logps

    manifest = read_json(manifest_path)
    contract = verify_inputs(source, manifest, seed, require_adapter=not preflight)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    task, groups = archived_supports(source, contract, seed)
    reconstructed = [(group, reconstruct(tok, task, group)) for group in groups]
    for _, rows in reconstructed:
        for row in rows:
            masks_for_row(row)
    if preflight:
        print(json.dumps({"preflight": "ok", "seed":seed, "supports":len(groups),
                          "traces":sum(len(rows) for _, rows in reconstructed)}), flush=True)
        return
    if output.exists() or not torch.cuda.is_available():
        raise ValueError("new output directory and a scheduler-allocated CUDA GPU required")
    started = time.monotonic()
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
            local_files_only=True, torch_dtype=cuda_dtype())
    adapter_dir = source / Path(contract["adapter"][0]["path"]).parent
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False).to("cuda").eval()
    model.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("scoring model must have no trainable parameters")
    output.mkdir(parents=True)
    record_path = output / "token_scores.jsonl.gz"
    with gzip.open(record_path, "wt") as stream, torch.inference_mode():
        for group, rows in reconstructed:
            factors, records = [], []
            for row, candidate in zip(rows, group["candidates"]):
                # Single unpadded forward per candidate: all four rules see exactly the same token scores.
                lp = token_logps(model, row.ids[None].to("cuda"))[0].cpu().double()
                lp = torch.cat((torch.zeros(1, dtype=lp.dtype), lp))
                if not torch.isfinite(lp).all() or (lp[1:] > 1e-6).any():
                    raise ValueError("invalid token log probabilities")
                item = {"joint": float(lp[row.span].sum())}
                positions = {}
                for scope, (head, tail, marker) in masks_for_row(row).items():
                    item[scope] = {"answer":float(lp[row.ans].sum()), "head":float(lp[head[0]].sum()),
                                   "tail":float(lp[tail[0]].sum()), "fixed_marker":float(lp[marker[0]].sum())}
                    positions[scope] = {name: mask[0].nonzero().flatten().tolist()
                                       for name, mask in (("head",head),("tail",tail),("fixed_marker",marker))}
                factors.append(item)
                records.append({"trace_id":row.trace_id, "token_ids":row.ids.tolist(),
                    "score_positions":row.span.nonzero().flatten().tolist(),
                    "answer_positions":row.ans.nonzero().flatten().tolist(), "segment_positions":positions,
                    "token_logprobs":lp[row.span].tolist(), "factors":item,
                    "original_objective_tokens":candidate["trace"]["objective_tokens"],
                    "reconstructed_objective_tokens":int(row.span.sum()),
                    "text_sha256":hashlib.sha256(candidate["sample"]["text"].encode()).hexdigest()})
            payload = {"seed":seed,"round":group["round"],"pid":group["pid"],
                       "prompt_sha256":group["prompt"]["canonical_prompt_sha256"],
                       "traces":records,"comparisons":four_way(factors)}
            stream.write(json.dumps(payload, allow_nan=False) + "\n")
            print(f"scored seed={seed} support={group['round']}:{group['pid']} traces={len(rows)}", flush=True)
    elapsed = time.monotonic() - started
    receipt = {"status":"complete", "audit_id":AUDIT_ID,"seed":seed,"supports":len(groups),
        "traces":sum(len(rows) for _, rows in reconstructed), "manifest_sha256":sha256(manifest_path),
        "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "scheduler_job": __import__("os").environ.get("JOB_ID"),
        "scheduler_task": __import__("os").environ.get("SGE_TASK_ID"),
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,"adapter_files":contract["adapter"],
        "scope":manifest["checkpoint"],"tokenization":manifest["tokenization"],
        "optimizer_steps":0,"new_generations":0,"official_test_accessed":False,
        "seconds":elapsed,"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
        "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),
        "output":{"path":record_path.name,"sha256":sha256(record_path),"size":record_path.stat().st_size}}
    (output / "complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Q5_SEGMENT_SCORING_COMPLETE " + json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.manifest.exists():
            raise ValueError("manifest already exists")
        args.manifest.write_text(json.dumps(prepare_manifest(args.source),indent=2)+"\n")
    else:
        if args.seed is None or (args.output is None and not args.preflight):
            parser.error("--seed and --output required for scoring")
        score_seed(args.source,args.manifest,args.output,args.seed,preflight=args.preflight)


if __name__ == "__main__":
    main()
