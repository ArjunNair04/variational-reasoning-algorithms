"""MATH answer extraction and benchmark-compatible string equivalence.

The normalization helpers follow the public EleutherAI Hendrycks-MATH task
implementation. The strict event is project-specific: one terminal
``#### answer`` marker, with no substantive text after the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


MATH_ANSWER_EVENT_MODES = {"legacy", "strict_terminal_marker"}


@dataclass(frozen=True)
class MathAnswerEvent:
    answer: str | None
    normalized_answer: str | None
    parse_mode: str
    marker_count: int
    terminal_marker: bool
    strict_valid: bool
    reasoning: str
    marker_start: int | None
    marker_end: int | None

    @property
    def format_failure(self) -> bool:
        return not self.strict_valid


def last_boxed_only_string(text: str) -> str | None:
    """Return the final balanced ``\boxed`` or ``\fbox`` expression."""

    index = text.rfind(r"\boxed")
    if r"\boxed " in text:
        return r"\boxed " + text.split(r"\boxed ")[-1].split("$")[0]
    if index < 0:
        index = text.rfind(r"\fbox")
        if index < 0:
            return None
    opened = 0
    saw_open = False
    for position in range(index, len(text)):
        if text[position] == "{":
            opened += 1
            saw_open = True
        elif text[position] == "}":
            opened -= 1
            if saw_open and opened == 0:
                return text[index : position + 1]
    return None


def remove_boxed(value: str) -> str:
    if value.startswith(r"\boxed "):
        return value[len(r"\boxed ") :]
    for prefix in (r"\boxed{", r"\fbox{"):
        if value.startswith(prefix) and value.endswith("}"):
            return value[len(prefix) : -1]
    raise ValueError(f"not a boxed expression: {value!r}")


def _fix_fracs(value: str) -> str:
    parts = value.split(r"\frac")
    rebuilt = parts[0]
    for part in parts[1:]:
        rebuilt += r"\frac"
        if not part or part[0] == "{":
            rebuilt += part
            continue
        if len(part) < 2:
            return value
        numerator, denominator = part[0], part[1]
        if denominator == "{":
            rebuilt += "{" + numerator + "}" + part[1:]
        else:
            rebuilt += "{" + numerator + "}{" + denominator + "}" + part[2:]
    return rebuilt


def _fix_simple_slash(value: str) -> str:
    parts = value.split("/")
    if len(parts) != 2:
        return value
    try:
        numerator, denominator = (int(part) for part in parts)
    except ValueError:
        return value
    if value != f"{numerator}/{denominator}":
        return value
    return rf"\frac{{{numerator}}}{{{denominator}}}"


def _fix_sqrt(value: str) -> str:
    parts = value.split(r"\sqrt")
    rebuilt = parts[0]
    for part in parts[1:]:
        if not part or part[0] == "{":
            rebuilt += r"\sqrt" + part
        else:
            rebuilt += r"\sqrt{" + part[0] + "}" + part[1:]
    return rebuilt


def _remove_right_units(value: str) -> str:
    if r"\text{ " not in value:
        return value
    return value.split(r"\text{ ", 1)[0]


def normalize_math_answer(value: str | None) -> str | None:
    """Normalize as in the standard Hendrycks-MATH exact-match harness."""

    if value is None:
        return None
    value = str(value).replace("\n", "").replace(r"\!", "")
    value = value.replace(r"\\", "\\")
    value = value.replace("tfrac", "frac").replace("dfrac", "frac")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"^{\circ}", "").replace(r"^\circ", "")
    value = value.replace(r"\$", "")
    value = _remove_right_units(value)
    value = value.replace(r"\%", "").replace("%", "")
    value = value.replace(" .", " 0.").replace("{.", "{0.")
    if not value:
        return value
    if value.startswith("."):
        value = "0" + value
    if len(value.split("=")) == 2 and len(value.split("=", 1)[0]) <= 2:
        value = value.split("=", 1)[1]
    value = _fix_sqrt(value)
    value = value.replace(" ", "")
    value = _fix_fracs(value)
    if value == "0.5":
        value = r"\frac{1}{2}"
    return _fix_simple_slash(value)


def math_answers_equivalent(left: str | None, right: str | None) -> bool:
    """Benchmark-compatible normalized string equality, not model grading."""

    if left is None or right is None:
        return left is None and right is None
    try:
        return normalize_math_answer(left) == normalize_math_answer(right)
    except (AssertionError, IndexError, TypeError, ValueError):
        return left == right


def gold_math_answer(solution: str) -> str | None:
    boxed = last_boxed_only_string(solution)
    return remove_boxed(boxed) if boxed is not None else None


def canonical_math_solution(solution: str) -> str | None:
    """Convert an official boxed solution to the project's terminal marker."""

    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return None
    answer = remove_boxed(boxed)
    reasoning = solution[: solution.rfind(boxed)].rstrip()
    return f"{reasoning}\n#### {answer}" if reasoning else f"#### {answer}"


_STRICT_RE = re.compile(
    r"(?s)^(?P<reason>.*?)(?P<marker>####)[ \t]*"
    r"(?P<answer>\S(?:[^\r\n]*?\S)?)[ \t]*$"
)
_FINAL_ANSWER_RE = re.compile(r"(?is)final\s+answer\s*:\s*(.+?)\s*$")


def parse_math_answer_event(
    text: str,
    *,
    mode: str = "legacy",
    disallowed_exact_answers: Sequence[str] = (),
) -> MathAnswerEvent:
    if mode not in MATH_ANSWER_EVENT_MODES:
        raise ValueError(
            f"unknown MATH answer-event mode {mode!r}; expected one of "
            f"{sorted(MATH_ANSWER_EVENT_MODES)}"
        )
    marker_count = text.count("####")
    strict_match = _STRICT_RE.fullmatch(text) if marker_count == 1 else None
    strict_answer = strict_match.group("answer") if strict_match else None
    strict_normalized = normalize_math_answer(strict_answer)
    disallowed = {str(value).strip().casefold() for value in disallowed_exact_answers}
    strict_valid = bool(strict_normalized) and (
        strict_answer is not None
        and strict_answer.strip().casefold() not in disallowed
    )
    reasoning = strict_match.group("reason").rstrip() if strict_match else ""
    if mode == "strict_terminal_marker":
        return MathAnswerEvent(
            answer=strict_answer if strict_valid else None,
            normalized_answer=strict_normalized if strict_valid else None,
            parse_mode=(
                "strict_terminal_marker"
                if strict_valid
                else (
                    "missing_marker"
                    if marker_count == 0
                    else "multiple_markers"
                    if marker_count > 1
                    else "nonterminal_or_invalid_marker"
                )
            ),
            marker_count=marker_count,
            terminal_marker=strict_valid,
            strict_valid=strict_valid,
            reasoning=reasoning,
            marker_start=(strict_match.start("marker") if strict_match else None),
            marker_end=(strict_match.end("marker") if strict_match else None),
        )

    if strict_valid:
        answer = strict_answer
        parse_mode = "marker"
    else:
        boxed = last_boxed_only_string(text)
        if boxed is not None:
            answer = remove_boxed(boxed)
            parse_mode = "boxed"
        else:
            final_match = _FINAL_ANSWER_RE.search(text)
            answer = final_match.group(1).strip() if final_match else None
            parse_mode = "final_answer" if answer else "unparsed"
    normalized = normalize_math_answer(answer)
    return MathAnswerEvent(
        answer=answer if normalized else None,
        normalized_answer=normalized if normalized else None,
        parse_mode=parse_mode,
        marker_count=marker_count,
        terminal_marker=strict_valid,
        strict_valid=strict_valid,
        reasoning=reasoning,
        marker_start=(strict_match.start("marker") if strict_match else None),
        marker_end=(strict_match.end("marker") if strict_match else None),
    )
