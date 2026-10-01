"""Native-token prefix branching for the isolated Q5 proposal screen."""

from collections import defaultdict
import hashlib
import json

import torch

from answer_events import parse_gsm8k_answer_event


def reasoning_prefix(tok, completion):
    """Return half the pre-marker tokens, without re-encoding sampled text."""
    tokens = [int(token) for token in completion]
    if tokens and tokens[-1] == tok.eos_token_id:
        tokens.pop()
    text = tok.decode(tokens, skip_special_tokens=True)
    event = parse_gsm8k_answer_event(text, mode="strict_terminal_marker")
    if event.marker_start is None or not event.reasoning.strip():
        return [], "missing_reasoning_boundary"
    # Tokens can contain both whitespace and part of the marker. Such tokens
    # must stay outside the retained reasoning prefix.
    count = 0
    for stop in range(1, len(tokens) + 1):
        decoded = tok.decode(tokens[:stop], skip_special_tokens=True)
        if len(decoded) > event.marker_start:
            break
        # Byte-level tokens can temporarily decode to a replacement character
        # before the next token completes the Unicode character.
        if text.startswith(decoded):
            count = stop
    prefix = tokens[:count // 2]
    if not prefix or not tok.decode(prefix, skip_special_tokens=True).strip():
        return [], "empty_half_prefix"
    if tok.eos_token_id in prefix:
        return [], "eos_in_prefix"
    return prefix, None


def _trim_eos(tokens, eos):
    values = tokens.tolist()
    if eos in values:
        values = values[:values.index(eos) + 1]
    return values


@torch.no_grad()
def sample_prefix_continuations(
    model, tok, prompts, *, traces_per_question, max_new, temperature,
    sample_fn, device, gen_bs=32,
):
    """Keep G/2 complete roots and draw one native-prefix child per root.

    The output is question-major, roots then children. Returned completion
    masks include reused prefixes for training; ``generated_tokens`` excludes
    them for compute accounting. No root is selected by answer correctness.
    """
    if traces_per_question < 2 or traces_per_question % 2:
        raise ValueError("prefix branching requires an even proposal count >= 2")
    if not prompts or len(prompts) % traces_per_question:
        raise ValueError("prefix branching requires complete question groups")
    if max_new < 2 or gen_bs < 1:
        raise ValueError("invalid prefix generation budget")
    if bool(getattr(tok, "_vrl_math_chat_rendered_prompts", False)):
        raise ValueError("prefix screen supports the registered base-model prompt only")
    half = traces_per_question // 2
    roots = []
    for start in range(0, len(prompts), traces_per_question):
        if len(set(prompts[start:start + traces_per_question])) != 1:
            raise ValueError("all proposals for one question must use the same prompt")
        roots.extend(range(start, start + half))
    root_prompts = [prompts[i] for i in roots]
    kwargs = {"max_new": max_new}
    if temperature != 1.0:
        kwargs["temperature"] = temperature
    ids, mask, _ = sample_fn(model, tok, root_prompts, **kwargs)
    if ids.shape != mask.shape or ids.shape[0] != len(roots):
        raise ValueError("root sampler shape mismatch")
    completions = [None] * len(prompts)
    counts = [0] * len(prompts)
    prefixes = [0] * len(prompts)
    parents = [None] * len(prompts)
    prefix_hashes = [None] * len(prompts)
    fallback = [None] * len(prompts)
    pending = defaultdict(list)
    for root_index, destination in enumerate(roots):
        completion = ids[root_index][mask[root_index]].cpu().tolist()
        if not completion or len(completion) > max_new:
            raise ValueError("root completion exceeds the registered token cap or is empty")
        completions[destination] = completion
        counts[destination] = len(completion)
        prefix, reason = reasoning_prefix(tok, completion)
        child = destination + half
        prefixes[child] = len(prefix)
        parents[child] = destination
        prefix_hashes[child] = hashlib.sha256(
            json.dumps(prefix, separators=(",", ":")).encode()
        ).hexdigest()
        fallback[child] = reason
        # Encode only the original proposal prompt. Sampled prefix IDs are
        # appended directly, so byte/token boundaries cannot be normalised away.
        prompt_ids = tok(prompts[child], return_tensors="pt").input_ids[0].tolist()
        pending[max_new - len(prefix)].append((child, prompt_ids + prefix, prefix))

    model.eval()
    eos = int(tok.eos_token_id)
    for remaining, rows in sorted(pending.items()):
        for start in range(0, len(rows), gen_bs):
            batch = rows[start:start + gen_bs]
            width = max(len(row[1]) for row in batch)
            inputs = torch.full((len(batch), width), eos, dtype=torch.long, device=device)
            attention = torch.zeros_like(inputs)
            for i, (_, context, _) in enumerate(batch):
                inputs[i, -len(context):] = torch.tensor(context, device=device)
                attention[i, -len(context):] = 1
            output = model.generate(
                input_ids=inputs, attention_mask=attention, do_sample=True,
                max_new_tokens=remaining, pad_token_id=eos,
                temperature=temperature, top_k=0, top_p=1.0,
            )
            if output.shape[0] != len(batch) or not torch.equal(output[:, :width], inputs):
                raise ValueError("continuation generator changed its input prefix")
            for i, (destination, _, prefix) in enumerate(batch):
                suffix = _trim_eos(output[i, width:].cpu(), eos)
                if not suffix or len(suffix) > remaining:
                    raise ValueError("continuation exceeds the token cap or is empty")
                completions[destination] = prefix + suffix
                counts[destination] = len(suffix)
    if any(value is None for value in completions):
        raise ValueError("incomplete prefix proposal allocation")
    width = max(len(value) for value in completions)
    output = torch.full((len(prompts), width), eos, dtype=torch.long, device=device)
    active = torch.zeros_like(output, dtype=torch.bool)
    for i, completion in enumerate(completions):
        output[i, :len(completion)] = torch.tensor(completion, device=device)
        active[i, :len(completion)] = True
    texts = tok.batch_decode(completions, skip_special_tokens=True)
    metadata = {
        "mode": "prefix_half", "root_count": len(roots),
        "continuation_count": sum(n > 0 for n in prefixes),
        "fallback_count": sum(reason is not None for reason in fallback),
        "parent_indices": parents, "prefix_tokens": prefixes,
        "prefix_sha256": prefix_hashes, "fallback_reasons": fallback,
        "generated_tokens": counts,
        "completion_tokens": [len(value) for value in completions],
        "max_new_tokens": max_new,
    }
    return output, active, texts, metadata
