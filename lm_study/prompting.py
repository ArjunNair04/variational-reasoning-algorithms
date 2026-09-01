"""Shared training and evaluation prompt interventions."""

from __future__ import annotations


PROPOSAL_PROMPTS = {
    "question",
    "answer_derive",
    "answer_derive_concise",
    "answer_derive_first",
}


def build_proposal_prompt(base_prompt: str, answer, mode: str) -> str:
    """Add a known-answer rationale instruction without changing demonstrations."""

    if mode not in PROPOSAL_PROMPTS:
        raise ValueError(
            f"unknown proposal prompt {mode!r}; expected one of "
            f"{sorted(PROPOSAL_PROMPTS)}"
        )
    if mode == "question":
        return base_prompt
    marker = "\nAnswer:"
    split = base_prompt.rfind(marker)
    if split < 0 or base_prompt[split + len(marker):].strip():
        raise ValueError(
            "answer-conditioned proposals require a prompt ending in "
            "'\\nAnswer:'"
        )
    question_prefix = base_prompt[:split]
    target = str(answer)
    if mode == "answer_derive":
        instruction = (
            f"(Note: the correct final answer is {target}. Please create a "
            "complete step-by-step rationale for the question that leads to "
            f"this answer. End with #### {target}.)"
        )
    elif mode == "answer_derive_concise":
        instruction = (
            f"The correct final answer is {target}. Give the shortest complete "
            "derivation that justifies this answer. Include each necessary "
            "calculation exactly once and do not restate intermediate results. "
            f"Finish with #### {target}."
        )
        return f"{question_prefix}\n{instruction}{marker}"
    else:
        instruction = (
            f"(For verification only, the correct final answer is {target}. "
            "Solve the problem forward from the stated quantities. Do not "
            "reveal or cite the supplied answer until the final line. Show "
            f"every calculation, then end with #### {target}.)"
        )
    return f"{question_prefix}\n{instruction}{marker}"
