"""Shared prompt boundary for train-only Hendrycks MATH experiments."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence


MATH_PROMPT_VERSION = "hendrycks_math_four_shot_eos_v1"
MATH_INSTRUCTION_PROMPT_VERSION = (
    "hendrycks_math_four_shot_eos_target_instruction_v2"
)
MATH_CHAT_PROMPT_VERSION = "hendrycks_math_four_shot_qwen3_chat_non_thinking_v3"
MATH_CHAT_BOUNDARY_PROMPT_VERSION = (
    "hendrycks_math_four_shot_qwen3_chat_non_thinking_boundary_v4"
)
MATH_CHAT_PROMPT_VERSIONS = {
    MATH_CHAT_PROMPT_VERSION,
    MATH_CHAT_BOUNDARY_PROMPT_VERSION,
}
MATH_DEMONSTRATION_SEPARATOR = "<|endoftext|>"
MATH_DEMONSTRATION_SEPARATOR_TOKEN_ID = 151643
MATH_CHAT_EOT_TOKEN = "<|im_end|>"
MATH_CHAT_EOT_TOKEN_ID = 151645
MATH_CHAT_EOD_TOKEN_ID = 151643
MATH_CHAT_TEMPLATE_CHARACTERS = 4168
MATH_CHAT_TEMPLATE_SHA256 = (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
)
MATH_TARGET_INSTRUCTION = (
    "Answer only the next problem. End with exactly one final line of the form "
    "#### answer, then stop. Do not begin another Problem: or Solution: section."
)
MATH_CHAT_BOUNDARY_TARGET_INSTRUCTION = (
    "Answer only the next problem. End with exactly one final line starting #### "
    "followed by a space and the actual answer (for example, #### 42; replace 42 "
    "with the actual answer), then stop. Do not begin another Problem: or "
    "Solution: section."
)


def is_math_chat_prompt_version(version: str) -> bool:
    """Return whether ``version`` uses the pinned Qwen3 chat contract."""

    return version in MATH_CHAT_PROMPT_VERSIONS


def math_prompt_contract(version: str) -> dict[str, Any]:
    """Return the exact checked-in contract for one supported prompt version."""

    if is_math_chat_prompt_version(version):
        target_instruction = (
            MATH_CHAT_BOUNDARY_TARGET_INSTRUCTION
            if version == MATH_CHAT_BOUNDARY_PROMPT_VERSION
            else MATH_TARGET_INSTRUCTION
        )
        contract = {
            "version": str(version),
            "renderer": "tokenizer.apply_chat_template",
            "demonstration_user_assistant_turns": 4,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "target_instruction": target_instruction,
            "tokenizer_eos_token": MATH_CHAT_EOT_TOKEN,
            "tokenizer_eos_token_id": MATH_CHAT_EOT_TOKEN_ID,
            "model_config_eos_token_id": MATH_CHAT_EOT_TOKEN_ID,
            "generation_config_eos_token_ids": [
                MATH_CHAT_EOT_TOKEN_ID,
                MATH_CHAT_EOD_TOKEN_ID,
            ],
            "generation_config_pad_token_id": MATH_CHAT_EOD_TOKEN_ID,
            "evaluation_eos_token_ids": [MATH_CHAT_EOT_TOKEN_ID],
            "evaluation_pad_token_id": MATH_CHAT_EOD_TOKEN_ID,
            "prohibited_generated_token_ids": [MATH_CHAT_EOD_TOKEN_ID],
            "chat_template_characters": MATH_CHAT_TEMPLATE_CHARACTERS,
            "chat_template_sha256": MATH_CHAT_TEMPLATE_SHA256,
        }
        if version == MATH_CHAT_BOUNDARY_PROMPT_VERSION:
            contract["strict_terminal_disallowed_exact_answers"] = ["answer"]
        return contract
    contract = {
        "version": str(version),
        "demonstration_separator": MATH_DEMONSTRATION_SEPARATOR,
        "demonstration_separator_token_id": MATH_DEMONSTRATION_SEPARATOR_TOKEN_ID,
    }
    if version == MATH_INSTRUCTION_PROMPT_VERSION:
        contract["target_instruction"] = MATH_TARGET_INSTRUCTION
    elif version != MATH_PROMPT_VERSION:
        raise ValueError(f"unknown MATH prompt version: {version}")
    return contract


def validate_math_prompt_tokenizer(
    tokenizer: Any,
    *,
    version: str = MATH_PROMPT_VERSION,
) -> dict[str, Any]:
    """Prove that the checked-in boundary is exactly the model's one EOS token."""

    encoded = tokenizer.encode(
        MATH_DEMONSTRATION_SEPARATOR,
        add_special_tokens=False,
    )
    token_ids = [int(token_id) for token_id in encoded]
    eos_token_id = int(tokenizer.eos_token_id)
    if tokenizer.eos_token != MATH_DEMONSTRATION_SEPARATOR:
        raise ValueError(
            "Qwen EOS token literal does not match the frozen MATH prompt contract"
        )
    if token_ids != [MATH_DEMONSTRATION_SEPARATOR_TOKEN_ID]:
        raise ValueError(
            "MATH demonstration separator is not exactly token 151643: "
            f"{token_ids}"
        )
    if eos_token_id != MATH_DEMONSTRATION_SEPARATOR_TOKEN_ID:
        raise ValueError(
            "Qwen EOS token ID does not match the frozen MATH prompt contract: "
            f"{eos_token_id}"
        )
    return {**math_prompt_contract(version), "separator_token_count": 1}


def validate_math_chat_tokenizer(
    tokenizer: Any,
    *,
    version: str = MATH_CHAT_PROMPT_VERSION,
) -> dict[str, Any]:
    """Bind the official Qwen3 chat tokenizer and non-thinking template."""

    if not is_math_chat_prompt_version(version):
        raise ValueError(f"not a Qwen3 MATH chat prompt version: {version}")

    if tokenizer.eos_token != MATH_CHAT_EOT_TOKEN:
        raise ValueError("Qwen3 chat tokenizer EOS literal is not <|im_end|>")
    if int(tokenizer.eos_token_id) != MATH_CHAT_EOT_TOKEN_ID:
        raise ValueError("Qwen3 chat tokenizer EOS ID is not 151645")
    if int(tokenizer.pad_token_id) != MATH_CHAT_EOD_TOKEN_ID:
        raise ValueError("Qwen3 chat tokenizer pad ID is not 151643")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Qwen3 chat tokenizer has no official chat template")
    template = tokenizer.chat_template
    if not isinstance(template, str):
        raise ValueError("Qwen3 chat template is not a decoded string")
    template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if (
        len(template) != MATH_CHAT_TEMPLATE_CHARACTERS
        or template_sha256 != MATH_CHAT_TEMPLATE_SHA256
    ):
        raise ValueError(
            "Qwen3 chat template identity changed: "
            f"characters={len(template)}, sha256={template_sha256}"
        )
    return {**math_prompt_contract(version), "messages_per_prompt": 9}


def validate_math_model_eos(
    model: Any,
    tokenizer: Any,
    *,
    version: str = MATH_PROMPT_VERSION,
) -> None:
    """Bind both model EOS declarations to the already-validated tokenizer."""

    if is_math_chat_prompt_version(version):
        validate_math_chat_tokenizer(tokenizer, version=version)
        model_eos = int(model.config.eos_token_id)
        generation_eos_raw = model.generation_config.eos_token_id
        if not isinstance(generation_eos_raw, (list, tuple)):
            raise ValueError("Qwen3 chat generation EOS declaration is not a list")
        generation_eos = [int(token_id) for token_id in generation_eos_raw]
        generation_pad = int(model.generation_config.pad_token_id)
        if (
            model_eos != MATH_CHAT_EOT_TOKEN_ID
            or generation_eos != [MATH_CHAT_EOT_TOKEN_ID, MATH_CHAT_EOD_TOKEN_ID]
            or generation_pad != MATH_CHAT_EOD_TOKEN_ID
        ):
            raise ValueError(
                "Qwen3 chat model EOS declarations changed: "
                f"config={model_eos}, generation={generation_eos}, pad={generation_pad}"
            )
        return
    validate_math_prompt_tokenizer(tokenizer, version=version)
    model_eos = int(model.config.eos_token_id)
    generation_eos = int(model.generation_config.eos_token_id)
    if (
        model_eos != MATH_DEMONSTRATION_SEPARATOR_TOKEN_ID
        or generation_eos != MATH_DEMONSTRATION_SEPARATOR_TOKEN_ID
    ):
        raise ValueError(
            "Qwen model EOS declarations do not match token 151643: "
            f"config={model_eos}, generation={generation_eos}"
        )


def bind_math_chat_generation_runtime(
    model: Any,
    tokenizer: Any,
    *,
    version: str = MATH_CHAT_BOUNDARY_PROMPT_VERSION,
    reset_audit: bool = False,
) -> dict[str, Any]:
    """Bind EOT-only generation while retaining EOD solely as right padding.

    The official checkpoint declares both EOT and EOD as generation stops.
    The qualified experimental contract counts only assistant EOT as natural
    termination.  This function changes stopping semantics, not logits: EOD is
    not suppressed, and any generated EOD remains active, is counted by the
    runtime audit, and invalidates the downstream result.
    """

    contract = validate_math_chat_tokenizer(tokenizer, version=version)
    suppression_fields = (
        getattr(model.generation_config, "suppress_tokens", None) or [],
        getattr(model.generation_config, "begin_suppress_tokens", None) or [],
    )
    if any(
        MATH_CHAT_EOD_TOKEN_ID in [int(token_id) for token_id in values]
        for values in suppression_fields
    ):
        raise ValueError("Qwen3 chat checkpoint suppresses EOD token 151643")
    bad_words = getattr(model.generation_config, "bad_words_ids", None) or []
    if any(
        MATH_CHAT_EOD_TOKEN_ID in [int(token_id) for token_id in sequence]
        for sequence in bad_words
    ):
        raise ValueError("Qwen3 chat checkpoint logit-suppresses EOD via bad_words_ids")
    already_bound = bool(getattr(model, "_vrl_math_chat_runtime_bound", False))
    if not already_bound:
        validate_math_model_eos(model, tokenizer, version=version)
        model.generation_config.eos_token_id = MATH_CHAT_EOT_TOKEN_ID
        model.generation_config.pad_token_id = MATH_CHAT_EOD_TOKEN_ID
        model._vrl_math_chat_runtime_bound = True
    if (
        int(model.config.eos_token_id) != MATH_CHAT_EOT_TOKEN_ID
        or int(model.generation_config.eos_token_id) != MATH_CHAT_EOT_TOKEN_ID
        or int(model.generation_config.pad_token_id) != MATH_CHAT_EOD_TOKEN_ID
    ):
        raise ValueError("Qwen3 chat runtime EOT/pad binding changed")
    if reset_audit or not isinstance(
        getattr(tokenizer, "_vrl_math_chat_generation_audit", None),
        dict,
    ):
        tokenizer._vrl_math_chat_generation_audit = {
            "eot_token_id": MATH_CHAT_EOT_TOKEN_ID,
            "eod_token_id": MATH_CHAT_EOD_TOKEN_ID,
            "sequences": 0,
            "generated_tokens": 0,
            "generated_eod_before_eot_count": 0,
        }
    tokenizer._vrl_math_chat_rendered_prompts = True
    return {
        **contract,
        "runtime_eos_token_ids": [MATH_CHAT_EOT_TOKEN_ID],
        "runtime_pad_token_id": MATH_CHAT_EOD_TOKEN_ID,
        "generated_eod_suppressed": False,
    }


def build_math_preamble(
    demonstrations: Sequence[dict[str, Any]],
) -> str:
    """Render complete demonstrations with a native EOS document boundary."""

    return "".join(
        f"Problem:\n{row['problem']}\n\nSolution:\n"
        f"{row['canonical_solution']}{MATH_DEMONSTRATION_SEPARATOR}\n\n"
        for row in demonstrations
    )


def build_math_prompts(
    rows: Sequence[dict[str, Any]],
    demonstrations: Sequence[dict[str, Any]],
    *,
    version: str = MATH_PROMPT_VERSION,
) -> list[str]:
    preamble = build_math_preamble(demonstrations)
    math_prompt_contract(version)
    target_prefix = (
        f"{MATH_TARGET_INSTRUCTION}\n\n"
        if version == MATH_INSTRUCTION_PROMPT_VERSION
        else ""
    )
    prompts = [
        preamble + target_prefix + f"Problem:\n{row['problem']}\n\nSolution:\n"
        for row in rows
    ]
    expected_separators = len(demonstrations)
    if any(
        prompt.count(MATH_DEMONSTRATION_SEPARATOR) != expected_separators
        for prompt in prompts
    ):
        raise AssertionError("MATH prompt demonstration-boundary count mismatch")
    return prompts


def build_math_chat_messages(
    rows: Sequence[dict[str, Any]],
    demonstrations: Sequence[dict[str, Any]],
    *,
    version: str = MATH_CHAT_PROMPT_VERSION,
) -> list[list[dict[str, str]]]:
    """Build four worked chat turns and one target user turn."""

    contract = math_prompt_contract(version)
    if not is_math_chat_prompt_version(version):
        raise ValueError(f"not a Qwen3 MATH chat prompt version: {version}")
    if len(demonstrations) != 4:
        raise ValueError("Qwen3 MATH chat qualification requires four demonstrations")
    message_sets: list[list[dict[str, str]]] = []
    for row in rows:
        messages: list[dict[str, str]] = []
        for demonstration in demonstrations:
            messages.extend(
                [
                    {"role": "user", "content": f"Problem:\n{demonstration['problem']}"},
                    {
                        "role": "assistant",
                        "content": str(demonstration["canonical_solution"]),
                    },
                ]
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{contract['target_instruction']}\n\nProblem:\n{row['problem']}"
                ),
            }
        )
        message_sets.append(messages)
    return message_sets


def render_math_chat_prompts(
    tokenizer: Any,
    message_sets: Sequence[Sequence[dict[str, str]]],
    *,
    version: str = MATH_CHAT_PROMPT_VERSION,
) -> list[str]:
    """Render only through the checkpoint's standard non-thinking chat template."""

    validate_math_chat_tokenizer(tokenizer, version=version)
    return [
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in message_sets
    ]
