"""Shared GSM8K answer-event parsing for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import re


ANSWER_EVENT_MODES = {"legacy", "strict_terminal_marker"}

_MARKER_RE = re.compile(r"####\s*(-?\d[\d,]*)")
_INTEGER_RE = re.compile(r"-?\d[\d,]*")
_STRICT_TERMINAL_RE = re.compile(
    r"(?s)^(?P<reason>.*?)(?P<marker>####)\s*"
    r"(?P<answer>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+))\s*$"
)


@dataclass(frozen=True)
class GSM8KAnswerEvent:
    """Parsed answer plus structural facts about the canonical marker event."""

    answer: int | None
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


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.replace(",", ""))
    except ValueError:
        return None


def _strict_event(text: str) -> GSM8KAnswerEvent:
    marker_count = text.count("####")
    first_marker_start = text.find("####")
    first_marker_end = (
        first_marker_start + len("####")
        if first_marker_start >= 0 else None
    )
    match = _STRICT_TERMINAL_RE.fullmatch(text) if marker_count == 1 else None
    if match is not None:
        answer = _as_int(match.group("answer"))
        return GSM8KAnswerEvent(
            answer=answer,
            parse_mode="strict_terminal_marker",
            marker_count=marker_count,
            terminal_marker=True,
            strict_valid=answer is not None,
            reasoning=match.group("reason").rstrip(),
            marker_start=match.start("marker"),
            marker_end=match.end("marker"),
        )
    if marker_count == 0:
        parse_mode = "missing_marker"
    elif marker_count > 1:
        parse_mode = "multiple_markers"
    else:
        parse_mode = "nonterminal_or_invalid_marker"
    return GSM8KAnswerEvent(
        answer=None,
        parse_mode=parse_mode,
        marker_count=marker_count,
        terminal_marker=False,
        strict_valid=False,
        reasoning=(
            text[:first_marker_start].rstrip()
            if first_marker_start >= 0 else ""
        ),
        marker_start=(
            first_marker_start if first_marker_start >= 0 else None
        ),
        marker_end=first_marker_end,
    )


def parse_gsm8k_answer_event(
    text: str,
    *,
    mode: str = "legacy",
) -> GSM8KAnswerEvent:
    """Parse one completion under replay-compatible or strict event semantics."""

    if mode not in ANSWER_EVENT_MODES:
        raise ValueError(
            f"unknown GSM8K answer-event mode {mode!r}; expected one of "
            f"{sorted(ANSWER_EVENT_MODES)}"
        )
    strict = _strict_event(text)
    if mode == "strict_terminal_marker":
        return strict

    marker = _MARKER_RE.search(text)
    if marker is not None:
        answer = _as_int(marker.group(1))
        parse_mode = "marker" if answer is not None else "unparsed"
    else:
        numbers = _INTEGER_RE.findall(text)
        answer = _as_int(numbers[-1]) if numbers else None
        parse_mode = "fallback_last_integer" if answer is not None else "unparsed"
    return GSM8KAnswerEvent(
        answer=answer,
        parse_mode=parse_mode,
        marker_count=strict.marker_count,
        terminal_marker=strict.terminal_marker,
        strict_valid=strict.strict_valid,
        reasoning=strict.reasoning,
        marker_start=strict.marker_start,
        marker_end=strict.marker_end,
    )
