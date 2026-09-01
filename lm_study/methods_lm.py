"""Shared LM plumbing for the study: the model registry + LoRA loader and the log-prob primitives.

Everything here is method-agnostic and used by every trainer (via common.py's re-export):
    MODELS / load_model   preset -> (HF id, LoRA target modules); LoRA'd causal LM on DEV
    seq_logprobs          summed (or per-token mean) completion log-probs, micro-batched
    token_logps           per-token log p(token | prefix), (B, L-1)
    kl_from_base          mean per-token drift from the frozen base (k1 estimator)

The phase-1/2 trainers that used to live here (run_em_lm / run_grpo_lm / run_raft_lm / VARIANTS --
the MATHS.md M1-M14 study) are preserved verbatim in legacy/legacy_lm.py.
"""
from __future__ import annotations
import torch

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def cuda_dtype():
    """Best CUDA dtype for the visible GPU; older cards may not support bf16."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Model registry: preset -> (HF id, LoRA target modules). target_modules are architecture-specific
# (NeoX/Pythia: query_key_value; GPT-2: c_attn; Llama/Qwen: q/k/v_proj), so switching model families
# needs the right names here. A raw HF id not in the registry falls back to NeoX-style modules
# (override with target_modules=).
MODELS = {
    "pythia-70m":   ("EleutherAI/pythia-70m",  ["query_key_value"]),
    "pythia-160m":  ("EleutherAI/pythia-160m", ["query_key_value"]),
    "pythia-410m":  ("EleutherAI/pythia-410m", ["query_key_value"]),
    "pythia-1b":    ("EleutherAI/pythia-1b",   ["query_key_value"]),
    "pythia-2.8b-deduped": ("EleutherAI/pythia-2.8b-deduped", ["query_key_value"]),
    "gpt2":         ("gpt2",                   ["c_attn"]),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B",      ["q_proj", "k_proj", "v_proj"]),
    # Qwen instruct models used by the earlier GSM8K study. q/k/v/o_proj LoRA.
    "qwen2.5-0.5b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    # Selected pretrained base for the next GSM8K study. Register the full HF id so it does not take
    # the raw-id fallback to NeoX's nonexistent query_key_value module.
    "google/gemma-2-2b":      ("google/gemma-2-2b",                ["q_proj", "k_proj", "v_proj", "o_proj"]),
    # Cross-model transfer check for the frozen Gemma protocol. Qwen3 requires transformers>=4.51.
    "qwen3-1.7b-base":        ("Qwen/Qwen3-1.7B-Base",             ["q_proj", "k_proj", "v_proj", "o_proj"]),
    # Same-family scale replication. The exact revision is shared with the train-only MATH
    # calibration and broad LoRA is enabled explicitly below.
    "qwen3-8b-base":          ("Qwen/Qwen3-8B-Base",               ["q_proj", "k_proj", "v_proj", "o_proj"]),
    # Post-trained checkpoint used only by the dedicated train-only MATH chat
    # path.  Keeping a separate preset prevents historical Base-model replays
    # from silently changing checkpoint or interface.
    "qwen3-8b-chat":          ("Qwen/Qwen3-8B",                    ["q_proj", "k_proj", "v_proj", "o_proj"]),
    # ---- secondary robustness-check models: DIFFERENT families, ~1-2.6B, moderate GSM8K (headroom to
    # measure fine-tuning lift), NOT GSM8K-saturated. All Llama-style attn -> q/k/v/o_proj LoRA. ----
    "llama-3.2-1b-instruct":  ("meta-llama/Llama-3.2-1B-Instruct", ["q_proj", "k_proj", "v_proj", "o_proj"]),   # gated on HF
    "gemma-2-2b-it":          ("google/gemma-2-2b-it",             ["q_proj", "k_proj", "v_proj", "o_proj"]),   # gated on HF
    "smollm2-1.7b-instruct":  ("HuggingFaceTB/SmolLM2-1.7B-Instruct", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    "olmo-2-1b-sft":          ("allenai/OLMo-2-0425-1B-SFT",       ["q_proj", "k_proj", "v_proj", "o_proj"]),   # SFT, NOT -Instruct (its RLVR uses GSM8K)
    # ---- scale-check tier: 8B BASE (zero post-training -> clean fine-tuning lift, ~55% GSM8K @ 8-shot).
    # A base model needs MORE shots (>=8) to lock the '#### answer' format (instruct runs use 2). ----
    "llama-3.1-8b":           ("meta-llama/Llama-3.1-8B",          ["q_proj", "k_proj", "v_proj", "o_proj"]),   # gated on HF
}
MODEL_REVISIONS = {
    # Pin the exact checkpoint used by the confirmatory Qwen/GSM8K study.
    "qwen3-1.7b-base": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    "qwen3-8b-base": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
    "qwen3-8b-chat": "b968826d9c46dd6066d109eabc6255188de91218",
}
LORA_TARGET_SETS = {
    "qwen3-1.7b-base": {
        "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "attention_mlp": (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    },
    "qwen3-8b-base": {
        "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "attention_mlp": (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    },
    "qwen3-8b-chat": {
        "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "attention_mlp": (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    },
}
MODEL_NAME = "pythia-160m"                                # default preset (back-compat)
QWEN3_8B_CHAT_LORA_TRAINABLE_PARAMETERS = 43_646_976


def validate_qwen3_8b_chat_lora_runtime(
    net,
    *,
    model: str,
    lora_r: int,
    lora_alpha: int,
    lora_target_set: str,
    target_modules,
):
    """Bind the exact broad adapter selected for the chat MATH pilot."""

    if model != "qwen3-8b-chat":
        return None
    expected_modules = LORA_TARGET_SETS["qwen3-8b-chat"]["attention_mlp"]
    if (
        int(lora_r) != 16
        or int(lora_alpha) != 32
        or lora_target_set != "attention_mlp"
        or tuple(target_modules) != expected_modules
    ):
        raise ValueError(
            "Qwen3-8B chat methods require attention_mlp LoRA r=16 alpha=32"
        )
    trainable_parameters = sum(
        int(parameter.numel())
        for parameter in net.parameters()
        if parameter.requires_grad
    )
    if trainable_parameters != QWEN3_8B_CHAT_LORA_TRAINABLE_PARAMETERS:
        raise ValueError(
            "Qwen3-8B chat LoRA trainable count changed: "
            f"{trainable_parameters} != "
            f"{QWEN3_8B_CHAT_LORA_TRAINABLE_PARAMETERS}"
        )
    return {
        "model_id": MODELS["qwen3-8b-chat"][0],
        "model_revision": MODEL_REVISIONS["qwen3-8b-chat"],
        "target_set": "attention_mlp",
        "target_modules": list(expected_modules),
        "rank": 16,
        "alpha": 32,
        "trainable_parameters": trainable_parameters,
    }


# --------------------------------------------------------------------------- #
#  model / generation plumbing
# --------------------------------------------------------------------------- #
def resolve_lora_target_modules(
    model: str,
    *,
    lora_target_set: str = "attention",
    target_modules=None,
) -> tuple[str, ...]:
    """Resolve a named, architecture-checked LoRA coverage set.

    ``target_modules`` remains available for legacy direct callers. YAML
    experiments should use ``lora_target_set`` so coverage is fingerprinted.
    """
    if target_modules is not None:
        if lora_target_set != "attention":
            raise ValueError(
                "target_modules cannot be combined with a non-default "
                "lora_target_set"
            )
        return tuple(target_modules)

    _hf_id, default_modules = MODELS.get(
        model,
        (model, ["query_key_value"]),
    )
    if lora_target_set == "attention":
        return tuple(default_modules)
    try:
        return LORA_TARGET_SETS[model][lora_target_set]
    except KeyError as exc:
        supported = sorted(LORA_TARGET_SETS.get(model, {"attention": ()}))
        raise ValueError(
            f"LoRA target set {lora_target_set!r} is not supported for "
            f"{model!r}; supported sets: {supported}"
        ) from exc


def load_model(
    seed=0,
    lora_r=16,
    lora_alpha=32,
    lora_seed=None,
    model=MODEL_NAME,
    target_modules=None,
    lora_target_set="attention",
    gradient_checkpointing=False,
):
    """Load a LoRA'd causal LM on DEV. ``model`` is a MODELS preset key or a raw HF id.
    ``gradient_checkpointing`` trades ~30-40% peak activation memory for compute; needed for the
    1.5B model's backward pass on the 16 GB box (EM/DPO were OOMing there)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    hf_id, _default_tmods = MODELS.get(model, (model, ["query_key_value"]))
    tmods = resolve_lora_target_modules(
        model,
        lora_target_set=lora_target_set,
        target_modules=target_modules,
    )
    revision = MODEL_REVISIONS.get(model)
    hf_kwargs = {"revision": revision} if revision is not None else {}
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(hf_id, **hf_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = cuda_dtype()
    # Transformers 4.x accepts torch_dtype; its generic loader can forward dtype to model __init__.
    base = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        **hf_kwargs,
    )
    torch.manual_seed(seed if lora_seed is None else int(lora_seed))
    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
                     target_modules=list(tmods), task_type="CAUSAL_LM")
    net = get_peft_model(base, cfg)
    lora_runtime_contract = validate_qwen3_8b_chat_lora_runtime(
        net,
        model=model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target_set=lora_target_set,
        target_modules=tmods,
    )
    if lora_runtime_contract is not None:
        net._vrl_lora_runtime_contract = lora_runtime_contract
    net = net.to(DEV)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if gradient_checkpointing:                            # less peak VRAM (use_cache off -> slower generate)
        net.enable_input_require_grads()
        net.gradient_checkpointing_enable()
    return net, tok


def seq_logprobs(model, ids, comp_mask, micro=4, grad=False, length_norm=False):
    """Sum (or mean) log p(token) over completion positions. (B,) tensor.

    Memory-efficient: per-token logprob = logit_target - logsumexp(logits), which avoids
    materialising a full-vocabulary float32 log_softmax tensor (Qwen vocab ~152k would OOM 16GB on
    long prompts). Small micro keeps the per-step logits tensor bounded."""
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, ids.shape[0], micro):
            mb_ids, mb_mask = ids[i:i + micro], comp_mask[i:i + micro]
            logits = model(mb_ids).logits[:, :-1]                       # (b, L-1, V), model dtype
            tok_lp = (logits.gather(-1, mb_ids[:, 1:, None]).squeeze(-1)
                      - torch.logsumexp(logits, -1)).float()           # (b, L-1), no full-vocab copy
            m = mb_mask[:, 1:].float()
            s = (tok_lp * m).sum(1)
            if length_norm:
                s = s / m.sum(1).clamp(min=1)
            outs.append(s)
    return torch.cat(outs)


@torch.no_grad()
def kl_from_base(model, ids, comp_mask):
    """Mean per-token (log p_theta - log p_base) on the given samples (k1 estimator)."""
    lp = seq_logprobs(model, ids, comp_mask, length_norm=True)
    with model.disable_adapter():
        lp0 = seq_logprobs(model, ids, comp_mask, length_norm=True)
    return float((lp - lp0).mean())


def token_logps(model, ids, grad=False):
    """Per-token log p(token | prefix), shape (B, L-1). logsumexp form (no full-vocab float copy)."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(ids).logits[:, :-1]
        return (logits.gather(-1, ids[:, 1:, None]).squeeze(-1)
                - torch.logsumexp(logits, -1)).float()
