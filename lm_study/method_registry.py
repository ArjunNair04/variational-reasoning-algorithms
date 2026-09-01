"""Load method aliases and their scientific defaults from checked-in YAML."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

METHOD_PRESETS_PATH = Path(__file__).with_name("method_presets.yaml")


class MethodRegistryError(ValueError):
    """The YAML method registry is malformed or incompatible with its trainers."""


def load_method_registry(
    implementations: Mapping[str, Callable[..., Any]],
    *,
    path: Path = METHOD_PRESETS_PATH,
) -> dict[str, Callable[..., Any]]:
    """Build the runtime method registry from schema-shaped YAML presets."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise MethodRegistryError(f"{path}: expected schema_version: 1")
    presets = payload.get("methods")
    if not isinstance(presets, dict) or not presets:
        raise MethodRegistryError(f"{path}: methods must be a non-empty mapping")

    registry: dict[str, Callable[..., Any]] = {}
    for method_name, preset in presets.items():
        if not isinstance(method_name, str) or not method_name:
            raise MethodRegistryError(f"{path}: method names must be non-empty strings")
        if not isinstance(preset, dict):
            raise MethodRegistryError(f"{path}: {method_name} must be a mapping")
        unknown_fields = set(preset) - {"implementation", "parameters", "description"}
        if unknown_fields:
            raise MethodRegistryError(
                f"{path}: {method_name} has unknown fields {sorted(unknown_fields)}"
            )
        implementation_name = preset.get("implementation")
        if implementation_name not in implementations:
            raise MethodRegistryError(
                f"{path}: {method_name} names unknown implementation "
                f"{implementation_name!r}"
            )
        parameters = preset.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise MethodRegistryError(
                f"{path}: {method_name}.parameters must be a mapping"
            )
        implementation = implementations[implementation_name]
        signature = inspect.signature(implementation)
        unsupported = set(parameters) - set(signature.parameters)
        if unsupported:
            raise MethodRegistryError(
                f"{path}: {method_name} sets unsupported parameters "
                f"{sorted(unsupported)}"
            )
        registry[method_name] = (
            functools.partial(implementation, **parameters)
            if parameters
            else implementation
        )
    return registry
