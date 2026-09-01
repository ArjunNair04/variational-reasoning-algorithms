"""Low-overhead, behaviour-neutral compute accounting for sweep cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _forward_target(model: torch.nn.Module) -> torch.nn.Module:
    """Return the causal-LM module invoked by both PEFT training and generation."""
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        target = get_base_model()
        if isinstance(target, torch.nn.Module):
            return target
    return model


@dataclass
class ModelForwardCounter:
    """Count top-level causal-LM forwards and input token positions.

    This deliberately does not estimate FLOPs. Gradient-checkpoint recomputation
    below the causal-LM boundary is not counted as another top-level forward.
    """

    calls: int = 0
    input_tokens: int = 0
    keyword_inputs_observed: bool = True
    _handle: Any = None

    def attach(self, model: torch.nn.Module) -> "ModelForwardCounter":
        target = _forward_target(model)

        def hook(_module, args, kwargs):
            self.calls += 1
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if isinstance(input_ids, torch.Tensor):
                self.input_tokens += int(input_ids.numel())

        try:
            self._handle = target.register_forward_pre_hook(
                hook,
                with_kwargs=True,
            )
        except TypeError:
            self.keyword_inputs_observed = False

            def positional_hook(_module, args):
                hook(_module, args, {})

            self._handle = target.register_forward_pre_hook(positional_hook)
        return self

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def snapshot(self) -> dict[str, int | bool]:
        return {
            "model_forward_calls": self.calls,
            "model_forward_input_tokens": self.input_tokens,
            "model_forward_keyword_inputs_observed": (
                self.keyword_inputs_observed
            ),
        }
