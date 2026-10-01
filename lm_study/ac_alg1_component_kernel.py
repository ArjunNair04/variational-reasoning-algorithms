"""Fixed-weight Q5 component kernels, separate from persistent buffer proposals."""

from collections import defaultdict
import hashlib
import json
import math
import time

import torch


def token_hash(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


@torch.no_grad()
def component_kernel_support(
    model, tok, task, buffers, weights, pids, *, epsilon, seed, outer_round,
    build_prompt, make_row, device, generation_batch_size=8,
):
    """One unbiased kernel draw per positive-mass parent, with fixed inherited mass.

    K maps invalid boundary draws back to the parent (without resampling).
    It is the pushforward of the frozen-within-round continuation distribution,
    not a softmax on the realised children. Returned rows never enter buffers.
    """
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("component kernel epsilon must be finite in [0,1]")
    if epsilon == 0:
        return buffers, weights, None
    if generation_batch_size < 1:
        raise ValueError("kernel generation batch size must be positive")
    if bool(getattr(task, "rendered_chat_prompts", False)):
        raise ValueError("component screen supports the registered GSM8K base prompt only")
    started = time.perf_counter()
    support, masses, records, pending = dict(buffers), dict(weights), [], defaultdict(list)
    for pid in dict.fromkeys(int(p) for p in pids):
        if pid not in weights:
            continue
        parent_rows = buffers[pid]
        w = weights[pid].detach().cpu()
        if w.ndim != 1 or len(w) != len(parent_rows):
            raise ValueError("kernel parent weight/support mismatch")
        if not torch.isfinite(w).all() or (w < 0).any() or not math.isclose(float(w.sum()), 1., abs_tol=1e-5):
            raise ValueError("kernel requires finite normalised parent weights")
        rows, allocated = [], []
        support[pid] = rows
        for index, (parent, weight) in enumerate(zip(parent_rows, w.tolist())):
            if weight == 0:
                continue
            h = parent.ids[parent.span & ~parent.ans].cpu().tolist()
            n = parent.reasoning_token_count
            if n is None or n < 0 or n > len(h):
                raise ValueError("kernel parent has no valid native reasoning boundary")
            prefix = h[:n // 2]
            reason = None
            if not prefix or not tok.decode(prefix, skip_special_tokens=True).strip():
                reason = "empty_half_prefix"
            elif tok.eos_token_id in prefix or len(prefix) >= int(task.max_new):
                reason = "invalid_prefix_budget"
            rows.extend([parent, parent])
            allocated.extend([(1 - epsilon) * weight, epsilon * weight])
            record = {
                "pid": pid, "parent_index": index, "parent_trace_id": parent.trace_id,
                "parent_weight": weight, "parent_mass": (1 - epsilon) * weight,
                "parent_logit": float(parent.responsibility_logit), "parent_h_ids": h,
                "parent_trace_logprob": float(parent.trace_logprob),
                "parent_answer_logprob": float(parent.answer_logprob),
                "kernel_mass": epsilon * weight, "prefix_tokens": len(prefix),
                "prefix_ids": prefix, "prefix_sha256": token_hash(prefix),
                "parent_h_sha256": token_hash(h), "fallback": reason,
                "generated_tokens": 0, "completion_ids": [], "child_h_ids": h,
                "child_trace_id": parent.trace_id, "same_as_parent": True,
            }
            records.append(record)
            if reason is None:
                prompt = build_prompt(pid)
                prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
                pending[int(task.max_new) - len(prefix)].append(
                    (pid, len(rows) - 1, prompt_ids + prefix, prefix, record, parent)
                )
        masses[pid] = torch.tensor(allocated, dtype=w.dtype, device=weights[pid].device)
        if not torch.allclose(masses[pid].sum(), weights[pid].sum(), atol=1e-6, rtol=1e-6):
            raise ValueError("kernel allocation changed question mass")

    rng_seed = int.from_bytes(hashlib.sha256(
        f"q5-component-kernel:{int(seed)}:{int(outer_round)}".encode()
    ).digest()[:8], "big") % (2**63 - 1)
    was_training = model.training
    try:
        model.eval()
        # fork_rng restores CPU and every visible CUDA stream even on failure.
        with torch.random.fork_rng():
            torch.manual_seed(rng_seed)
            for remaining, requests in sorted(pending.items()):
                for start in range(0, len(requests), generation_batch_size):
                    batch = requests[start:start + generation_batch_size]
                    width = max(len(item[2]) for item in batch)
                    ids = torch.full((len(batch), width), int(tok.eos_token_id), dtype=torch.long, device=device)
                    attention = torch.zeros_like(ids)
                    for i, item in enumerate(batch):
                        ids[i, -len(item[2]):] = torch.tensor(item[2], device=device)
                        attention[i, -len(item[2]):] = 1
                    output = model.generate(input_ids=ids, attention_mask=attention,
                        do_sample=True, max_new_tokens=remaining,
                        pad_token_id=int(tok.eos_token_id), temperature=1., top_k=0, top_p=1.)
                    if output.shape[0] != len(batch) or not torch.equal(output[:, :width], ids):
                        raise ValueError("kernel generator changed the native prefix")
                    for i, (pid, destination, _, prefix, record, parent) in enumerate(batch):
                        suffix = output[i, width:].cpu().tolist()
                        if tok.eos_token_id in suffix:
                            suffix = suffix[:suffix.index(tok.eos_token_id) + 1]
                        if not suffix or len(suffix) > remaining:
                            raise ValueError("invalid kernel suffix length")
                        completion = prefix + suffix
                        record.update(generated_tokens=len(suffix), completion_ids=completion)
                        child = make_row(pid, completion, f"kernel:{outer_round}:{pid}:{record['parent_index']}")
                        if child is None:
                            record["fallback"] = "unusable_sampled_boundary"
                            continue
                        child_h = child.ids[child.span & ~child.ans].cpu().tolist()
                        if child_h[:len(prefix)] != prefix:
                            raise ValueError("kernel row lost its native reasoning prefix")
                        support[pid][destination] = child
                        record.update(child_h_ids=child_h, child_trace_id=child.trace_id,
                            same_as_parent=child_h == parent.ids[parent.span & ~parent.ans].cpu().tolist())
    finally:
        model.train(was_training)
    metadata = {
        "mode": "fixed_weight_half_prefix", "epsilon": epsilon, "draws_per_parent": 1,
        "rng_seed": rng_seed, "weights_rescored": False, "children_persisted": False,
        "max_new_tokens": int(task.max_new), "generation_batch_size": generation_batch_size,
        "generated_draws": sum(r["generated_tokens"] > 0 for r in records),
        "generated_tokens": sum(r["generated_tokens"] for r in records),
        "copied_prefix_tokens": sum(r["prefix_tokens"] for r in records if r["generated_tokens"]),
        "fallback_mass": sum(r["kernel_mass"] for r in records if r["fallback"]),
        "changed_mass": sum(r["kernel_mass"] for r in records if not r["same_as_parent"]),
        "kernel_mass": sum(r["kernel_mass"] for r in records),
        "elapsed_seconds": time.perf_counter() - started, "parents": records,
    }
    return support, masses, metadata
