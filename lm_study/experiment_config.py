"""Typed compatibility boundary for experiment expansion and core run defaults."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, NamedTuple


RUN_YAML_TOP_LEVEL_FIELDS = frozenset(
    {
        "run_id",
        "tag_prefix",
        "defaults",
        "algos",
        "per_model",
        "diagnostic",
        "selection",
    }
)

RUN_DEFAULT_FIELDS = frozenset(
    {
        "task",
        "model",
        "models",
        "rounds",
        "seeds",
        "seed_values",
        "n_test",
        "train_partition",
        "eval_partition",
        "answer_event_mode",
        "answer_target_termination",
        "evaluation_prompt",
        "transfer_eval_dataset",
        "transfer_eval_n",
        "batch",
        "eval_batch",
        "G",
        "prompts",
        "shots",
        "shot_bank_size",
        "task_seed_from_run_seed",
        "question_sampling",
        "grad_checkpoint",
        "out",
        "lora_r",
        "lora_alpha",
        "lora_seed",
        "lora_target_set",
        "dump_completions",
        "eval_every",
        "eval_rounds",
        "save_adapter",
        "save_training_diagnostics",
        "training_diagnostics_level",
        "training_diagnostics_trace_tape",
        "training_diagnostics_gradient_questions",
        "training_diagnostics_probe_size",
        "l2r_candidate_utility_questions",
        "l2r_candidate_utility_batch",
        "checkpoint_every",
        "l2r_exact_cache",
        "l2r_state_checkpoint_every",
        "passk",
        "passk_n",
    }
)

_ARTIFACT_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

_POLICY_KL_FLAT_FIELDS = frozenset(
    {
        "kl_coef",
        "policy_kl_coef",
        "policy_anchor_mode",
        "policy_anchor_target_ratio",
        "policy_anchor_beta_min",
        "policy_anchor_beta_max",
        "policy_anchor_ema",
        "trust_kl_budget",
        "trust_safety_questions",
        "trust_safety_tolerance",
        "trust_boundary_failure_ceiling",
        "trust_max_backtracks",
        "trust_backtrack_shrink",
    }
)


class ExperimentCell(NamedTuple):
    """One expanded model/method/configuration point.

    NamedTuple preserves historical tuple unpacking while giving orchestration
    code and new callers explicit field names.
    """

    model: str
    method: str
    axes: dict[str, Any]
    tag: str


@dataclass(frozen=True)
class RunDefaults:
    task: str = "gsm8k"
    out: str = "~/po_results"
    rounds: int = 60
    batch: int = 32
    eval_batch: int = 16
    generations: int = 8
    prompts: int = 64
    shots: int = 2
    n_test: int = 200
    train_partition: str = "all"
    eval_partition: str = "all"
    answer_event_mode: str = "legacy"
    answer_target_termination: str = "none"
    evaluation_prompt: str = "question"
    transfer_eval_dataset: str = "none"
    transfer_eval_n: int = 1000

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RunDefaults":
        unknown = set(values) - RUN_DEFAULT_FIELDS
        if unknown:
            raise ValueError(f"unknown defaults field(s): {sorted(unknown)}")
        for field in (
            "rounds",
            "batch",
            "eval_batch",
            "G",
            "prompts",
            "n_test",
            "transfer_eval_n",
        ):
            if field in values:
                _nonnegative_int(values[field], field=f"defaults.{field}", positive=True)
        if "shots" in values:
            _nonnegative_int(values["shots"], field="defaults.shots")
        defaults = cls(
            task=str(values.get("task", cls.task)),
            out=str(values.get("out", cls.out)),
            rounds=int(values.get("rounds", cls.rounds)),
            batch=int(values.get("batch", cls.batch)),
            eval_batch=int(values.get("eval_batch", cls.eval_batch)),
            generations=int(values.get("G", cls.generations)),
            prompts=int(values.get("prompts", cls.prompts)),
            shots=int(values.get("shots", cls.shots)),
            n_test=int(values.get("n_test", cls.n_test)),
            train_partition=str(
                values.get("train_partition", cls.train_partition)
            ),
            eval_partition=str(values.get("eval_partition", cls.eval_partition)),
            answer_event_mode=str(
                values.get("answer_event_mode", cls.answer_event_mode)
            ),
            answer_target_termination=str(
                values.get(
                    "answer_target_termination",
                    cls.answer_target_termination,
                )
            ),
            evaluation_prompt=str(
                values.get("evaluation_prompt", cls.evaluation_prompt)
            ),
            transfer_eval_dataset=str(
                values.get(
                    "transfer_eval_dataset",
                    cls.transfer_eval_dataset,
                )
            ),
            transfer_eval_n=int(
                values.get("transfer_eval_n", cls.transfer_eval_n)
            ),
        )
        positive = {
            "rounds": defaults.rounds,
            "batch": defaults.batch,
            "eval_batch": defaults.eval_batch,
            "G": defaults.generations,
            "prompts": defaults.prompts,
            "n_test": defaults.n_test,
            "transfer_eval_n": defaults.transfer_eval_n,
        }
        invalid = {name: value for name, value in positive.items() if value < 1}
        if invalid:
            raise ValueError(f"run defaults must be positive: {invalid}")
        if defaults.shots < 0:
            raise ValueError("defaults.shots must be nonnegative")
        return defaults

    def cli_args(self) -> list[str]:
        return [
            "--task",
            self.task,
            "--rounds",
            str(self.rounds),
            "--batch",
            str(self.batch),
            "--eval-batch",
            str(self.eval_batch),
            "--G",
            str(self.generations),
            "--prompts",
            str(self.prompts),
            "--shots",
            str(self.shots),
            "--n-test",
            str(self.n_test),
            "--train-partition",
            self.train_partition,
            "--eval-partition",
            self.eval_partition,
            "--answer-event-mode",
            self.answer_event_mode,
            "--answer-target-termination",
            self.answer_target_termination,
            "--evaluation-prompt",
            self.evaluation_prompt,
            "--transfer-eval-dataset",
            self.transfer_eval_dataset,
            "--transfer-eval-n",
            str(self.transfer_eval_n),
        ]


@dataclass(frozen=True)
class ACAlg1BatchAllocation:
    labelled: int
    answer_only: int

    @classmethod
    def from_budget(
        cls, *, batch: int, generations: int, labelled_fraction: float
    ) -> "ACAlg1BatchAllocation":
        if batch < 1 or generations < 1:
            raise ValueError("AC-ALG1 batch and generations must be positive")
        if not 0.0 <= labelled_fraction <= 1.0:
            raise ValueError(
                "AC-ALG1 labelled fraction must be in [0, 1], "
                f"got {labelled_fraction}"
            )
        questions = max(batch // generations, 1)
        labelled = min(
            questions,
            max(0, int(questions * labelled_fraction + 0.5)),
        )
        if 0.0 < labelled_fraction < 1.0 and questions > 1:
            labelled = min(questions - 1, max(1, labelled))
        return cls(labelled=labelled, answer_only=questions - labelled)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def validate_artifact_identifier(value: Any, *, field: str) -> str:
    """Return a filesystem-safe identifier suitable for result artifact names."""
    identifier = str(value)
    if not _ARTIFACT_IDENTIFIER.fullmatch(identifier):
        raise ValueError(
            f"{field} must be 1-128 filesystem-safe characters starting with "
            "a letter or digit"
        )
    return identifier


def _known_fields(
    values: Mapping[str, Any],
    *,
    field: str,
    allowed: set[str],
) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {field} field(s): {sorted(unknown)}")


def _required_fields(
    values: Mapping[str, Any],
    *,
    field: str,
    required: set[str],
) -> None:
    missing = required - set(values)
    if missing:
        raise ValueError(f"{field} is missing required field(s): {sorted(missing)}")


def _each(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def _finite_nonnegative(value: Any, *, field: str) -> None:
    for item in _each(value):
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric, got {item!r}") from exc
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{field} must be finite and nonnegative, got {item!r}")


def _fraction(
    value: Any,
    *,
    field: str,
    include_one: bool = True,
) -> None:
    for item in _each(value):
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric, got {item!r}") from exc
        upper_ok = number <= 1 if include_one else number < 1
        if not math.isfinite(number) or number < 0 or not upper_ok:
            interval = "[0, 1]" if include_one else "[0, 1)"
            raise ValueError(f"{field} must be in {interval}, got {item!r}")


def _nonnegative_int(value: Any, *, field: str, positive: bool = False) -> None:
    for item in _each(value):
        if isinstance(item, bool):
            raise ValueError(f"{field} must be an integer, got {item!r}")
        try:
            number = int(item)
            exact = float(item) == number
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be an integer, got {item!r}") from exc
        if not exact or number < int(positive):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{field} must be a {qualifier} integer, got {item!r}")


def _positive_shrink(value: Any, *, field: str) -> None:
    for item in _each(value):
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric, got {item!r}") from exc
        if not math.isfinite(number) or not 0 < number < 1:
            raise ValueError(f"{field} must be in (0, 1), got {item!r}")


def _method_family(method: str) -> str | None:
    if method == "AC-ALG1":
        return "ac_alg1"
    if method.startswith("L2R-"):
        return "l2r"
    return None


def _optional_mapping(
    values: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = values.get(key)
    return _mapping({} if value is None else value, field=key)


def _validate_algorithms(config: Mapping[str, Any]) -> Mapping[str, Any]:
    algos = _optional_mapping(config, "algos")
    if not algos:
        raise ValueError("algos must contain at least one method")
    for method, method_spec in algos.items():
        if not isinstance(method, str) or not method:
            raise ValueError(f"algorithm names must be nonempty strings, got {method!r}")
        variants = method_spec if isinstance(method_spec, list) else [method_spec or {}]
        if not variants:
            raise ValueError(f"algorithm {method!r} has an empty variant list")
        for index, variant in enumerate(variants):
            if not isinstance(variant, Mapping):
                raise ValueError(
                    f"algos.{method}[{index}] must be a mapping"
                )
    return algos


def _configured_models(defaults: Mapping[str, Any]) -> tuple[str, ...]:
    if "model" in defaults and "models" in defaults:
        raise ValueError("defaults.model and defaults.models are mutually exclusive")
    raw_models = defaults.get(
        "models",
        [defaults.get("model", "qwen2.5-1.5b-instruct")],
    )
    if not isinstance(raw_models, (list, tuple)) or not raw_models:
        raise ValueError("defaults.models must be a nonempty list")
    models = tuple(str(model) for model in raw_models)
    if any(not model for model in models) or len(set(models)) != len(models):
        raise ValueError("configured models must be nonempty and unique")
    return models


def _validate_per_model(
    config: Mapping[str, Any],
    *,
    models: tuple[str, ...],
    algorithms: Mapping[str, Any],
) -> None:
    per_model = _optional_mapping(config, "per_model")
    unknown_models = set(per_model) - set(models)
    if unknown_models:
        raise ValueError(
            f"per_model contains unconfigured model(s): {sorted(unknown_models)}"
        )
    for model, overrides in per_model.items():
        if not isinstance(model, str) or not model:
            raise ValueError(f"per_model keys must be nonempty strings, got {model!r}")
        method_overrides = _mapping(
            overrides,
            field=f"per_model.{model}",
        )
        unknown_methods = set(method_overrides) - set(algorithms)
        if unknown_methods:
            raise ValueError(
                f"per_model.{model} contains unconfigured method(s): "
                f"{sorted(unknown_methods)}"
            )
        for method, override in method_overrides.items():
            _mapping(
                {} if override is None else override,
                field=f"per_model.{model}.{method}",
            )


def _validate_run_default_values(defaults: Mapping[str, Any]) -> None:
    for field in (
        "seeds",
        "shot_bank_size",
        "lora_seed",
        "dump_completions",
        "eval_every",
        "checkpoint_every",
        "training_diagnostics_gradient_questions",
        "training_diagnostics_probe_size",
        "l2r_candidate_utility_questions",
        "l2r_state_checkpoint_every",
        "passk",
    ):
        if field in defaults:
            _nonnegative_int(defaults[field], field=f"defaults.{field}")
    for field in ("lora_r", "lora_alpha", "l2r_candidate_utility_batch", "passk_n"):
        if field in defaults:
            _nonnegative_int(
                defaults[field],
                field=f"defaults.{field}",
                positive=True,
            )
    if "seeds" in defaults and int(defaults["seeds"]) < 1:
        raise ValueError("defaults.seeds must be positive")

    seed_values = defaults.get("seed_values")
    if seed_values is not None:
        if not isinstance(seed_values, (list, tuple)) or not seed_values:
            raise ValueError("defaults.seed_values must be a nonempty list")
        for seed in seed_values:
            _nonnegative_int(seed, field="defaults.seed_values")
        if len(set(int(seed) for seed in seed_values)) != len(seed_values):
            raise ValueError("defaults.seed_values must be unique")
        if "seeds" in defaults and int(defaults["seeds"]) != len(seed_values):
            raise ValueError(
                "defaults.seeds must equal len(defaults.seed_values)"
            )

    eval_rounds = defaults.get("eval_rounds")
    if eval_rounds is not None:
        if not isinstance(eval_rounds, (list, tuple)) or not eval_rounds:
            raise ValueError("defaults.eval_rounds must be a nonempty list")
        for round_number in eval_rounds:
            _nonnegative_int(
                round_number,
                field="defaults.eval_rounds",
                positive=True,
            )
        if len(set(int(value) for value in eval_rounds)) != len(eval_rounds):
            raise ValueError("defaults.eval_rounds must be unique")

    choices = {
        "train_partition": {"all", "train"},
        "eval_partition": {"all", "tune", "final", "validation", "test"},
        "question_sampling": {"random", "epoch_shuffle"},
        "training_diagnostics_level": {"standard", "deep"},
        "answer_event_mode": {"legacy", "strict_terminal_marker"},
        "answer_target_termination": {"none", "eos"},
        "evaluation_prompt": {"question", "answer_derive", "answer_derive_first"},
        "transfer_eval_dataset": {"none", "svamp"},
    }
    for field, allowed in choices.items():
        if field in defaults and defaults[field] not in allowed:
            raise ValueError(
                f"defaults.{field} must be one of {sorted(allowed)}, "
                f"got {defaults[field]!r}"
            )

    transfer_dataset = str(defaults.get("transfer_eval_dataset", "none"))
    transfer_n = int(defaults.get("transfer_eval_n", RunDefaults.transfer_eval_n))
    if not 1 <= transfer_n <= 1000:
        raise ValueError("defaults.transfer_eval_n must be in [1, 1000]")
    if transfer_dataset != "none":
        if str(defaults.get("task", RunDefaults.task)) != "gsm8k":
            raise ValueError(
                "defaults.transfer_eval_dataset is supported only for GSM8K runs"
            )
        if str(defaults.get("evaluation_prompt", "question")) != "question":
            raise ValueError(
                "SVAMP transfer evaluation requires evaluation_prompt='question'"
            )

    for field in (
        "task_seed_from_run_seed",
        "grad_checkpoint",
        "save_adapter",
        "save_training_diagnostics",
        "training_diagnostics_trace_tape",
        "l2r_exact_cache",
    ):
        if field in defaults and not isinstance(defaults[field], bool):
            raise ValueError(f"defaults.{field} must be boolean")


def _validate_run_metadata(config: Mapping[str, Any]) -> None:
    for metadata_field in ("diagnostic", "selection"):
        if metadata_field in config:
            _mapping(config[metadata_field], field=metadata_field)

    for identity_field in ("run_id", "tag_prefix"):
        if identity_field not in config:
            continue
        validate_artifact_identifier(config[identity_field], field=identity_field)


def validate_run_yaml_config(value: Any) -> Mapping[str, Any]:
    """Validate the non-scientific YAML envelope before expanding any cells.

    Method-specific axes are validated after expansion because structured
    policy controls first resolve into flat trainer arguments.
    """

    config = _mapping(value, field="experiment config")
    _known_fields(
        config,
        field="top-level experiment",
        allowed=set(RUN_YAML_TOP_LEVEL_FIELDS),
    )
    defaults = _optional_mapping(config, "defaults")
    RunDefaults.from_mapping(defaults)
    _validate_run_default_values(defaults)
    algorithms = _validate_algorithms(config)
    models = _configured_models(defaults)
    _validate_per_model(
        config,
        models=models,
        algorithms=algorithms,
    )
    _validate_run_metadata(config)
    return config


def expand_policy_kl_control(
    values: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    """Resolve the user-facing policy-KL block into stable trainer arguments.

    Legacy flat fields remain accepted for exact replay. New configs may use
    ``policy_kl_control`` but cannot mix the two interfaces in one cell.
    """

    resolved = dict(values)
    if "policy_kl_control" not in resolved:
        return resolved

    family = _method_family(method)
    if family is None:
        raise ValueError(
            f"policy_kl_control is not supported for method {method!r}"
        )
    mixed = set(resolved) & _POLICY_KL_FLAT_FIELDS
    if mixed:
        raise ValueError(
            "policy_kl_control cannot be combined with legacy flat field(s): "
            f"{sorted(mixed)}"
        )

    control = _mapping(
        resolved.pop("policy_kl_control"),
        field="policy_kl_control",
    )
    _known_fields(
        control,
        field="policy_kl_control",
        allowed={"mode", "coefficient", "anchor", "trust_region", "backtracking"},
    )
    mode = control.get("mode")
    if mode not in {"none", "fixed_penalty", "adaptive_gradient_ratio"}:
        raise ValueError(
            "policy_kl_control.mode must be one of "
            "'none', 'fixed_penalty', or 'adaptive_gradient_ratio'"
        )

    has_trust = "trust_region" in control
    has_backtracking = "backtracking" in control
    if family != "l2r" and (has_trust or has_backtracking):
        raise ValueError(
            "policy_kl_control trust_region/backtracking is currently supported "
            "only by L2R methods"
        )
    if has_backtracking and not has_trust:
        raise ValueError(
            "policy_kl_control.backtracking requires policy_kl_control.trust_region"
        )

    if mode == "none":
        extras = set(control) - {"mode"}
        if extras:
            raise ValueError(
                "policy_kl_control mode 'none' accepts no other field(s): "
                f"{sorted(extras)}"
            )
        resolved["policy_anchor_mode"] = "fixed"
        resolved["kl_coef" if family == "l2r" else "policy_kl_coef"] = (
            0.0 if family == "l2r" else None
        )
        return resolved

    if mode == "fixed_penalty":
        _required_fields(
            control,
            field="policy_kl_control",
            required={"coefficient"},
        )
        if "anchor" in control:
            raise ValueError(
                "policy_kl_control.anchor requires mode 'adaptive_gradient_ratio'"
            )
        coefficient = control["coefficient"]
        _finite_nonnegative(
            coefficient,
            field="policy_kl_control.coefficient",
        )
        resolved["policy_anchor_mode"] = "fixed"
        resolved["kl_coef" if family == "l2r" else "policy_kl_coef"] = coefficient
    else:
        if "coefficient" in control:
            raise ValueError(
                "policy_kl_control.coefficient requires mode 'fixed_penalty'"
            )
        _required_fields(
            control,
            field="policy_kl_control",
            required={"anchor"},
        )
        anchor = _mapping(
            control["anchor"],
            field="policy_kl_control.anchor",
        )
        anchor_fields = {
            "target_gradient_ratio",
            "beta_min",
            "beta_max",
            "gradient_norm_ema",
        }
        _known_fields(
            anchor,
            field="policy_kl_control.anchor",
            allowed=anchor_fields,
        )
        _required_fields(
            anchor,
            field="policy_kl_control.anchor",
            required=anchor_fields,
        )
        _finite_nonnegative(
            anchor["target_gradient_ratio"],
            field="policy_kl_control.anchor.target_gradient_ratio",
        )
        _finite_nonnegative(
            anchor["beta_min"],
            field="policy_kl_control.anchor.beta_min",
        )
        _finite_nonnegative(
            anchor["beta_max"],
            field="policy_kl_control.anchor.beta_max",
        )
        _fraction(
            anchor["gradient_norm_ema"],
            field="policy_kl_control.anchor.gradient_norm_ema",
            include_one=False,
        )
        beta_min = _each(anchor["beta_min"])
        beta_max = _each(anchor["beta_max"])
        if len(beta_min) == len(beta_max) == 1 and float(beta_min[0]) > float(
            beta_max[0]
        ):
            raise ValueError(
                "policy_kl_control.anchor.beta_min cannot exceed beta_max"
            )
        resolved.update(
            {
                "policy_anchor_mode": "grad_ratio",
                "policy_anchor_target_ratio": anchor["target_gradient_ratio"],
                "policy_anchor_beta_min": anchor["beta_min"],
                "policy_anchor_beta_max": anchor["beta_max"],
                "policy_anchor_ema": anchor["gradient_norm_ema"],
                "kl_coef" if family == "l2r" else "policy_kl_coef": 0.0
                if family == "l2r"
                else None,
            }
        )

    if has_trust:
        trust = _mapping(
            control["trust_region"],
            field="policy_kl_control.trust_region",
        )
        trust_fields = {
            "max_realized_token_kl",
            "safety_questions",
            "max_safety_nll_increase",
            "max_boundary_failure_fraction",
        }
        _known_fields(
            trust,
            field="policy_kl_control.trust_region",
            allowed=trust_fields,
        )
        _required_fields(
            trust,
            field="policy_kl_control.trust_region",
            required=trust_fields,
        )
        _finite_nonnegative(
            trust["max_realized_token_kl"],
            field="policy_kl_control.trust_region.max_realized_token_kl",
        )
        _nonnegative_int(
            trust["safety_questions"],
            field="policy_kl_control.trust_region.safety_questions",
            positive=True,
        )
        _finite_nonnegative(
            trust["max_safety_nll_increase"],
            field="policy_kl_control.trust_region.max_safety_nll_increase",
        )
        _fraction(
            trust["max_boundary_failure_fraction"],
            field=(
                "policy_kl_control.trust_region."
                "max_boundary_failure_fraction"
            ),
        )
        resolved.update(
            {
                "trust_kl_budget": trust["max_realized_token_kl"],
                "trust_safety_questions": trust["safety_questions"],
                "trust_safety_tolerance": trust["max_safety_nll_increase"],
                "trust_boundary_failure_ceiling": trust[
                    "max_boundary_failure_fraction"
                ],
            }
        )

    if has_backtracking:
        backtracking = _mapping(
            control["backtracking"],
            field="policy_kl_control.backtracking",
        )
        backtracking_fields = {"max_backtracks", "shrink_factor"}
        _known_fields(
            backtracking,
            field="policy_kl_control.backtracking",
            allowed=backtracking_fields,
        )
        _required_fields(
            backtracking,
            field="policy_kl_control.backtracking",
            required=backtracking_fields,
        )
        _nonnegative_int(
            backtracking["max_backtracks"],
            field="policy_kl_control.backtracking.max_backtracks",
        )
        _positive_shrink(
            backtracking["shrink_factor"],
            field="policy_kl_control.backtracking.shrink_factor",
        )
        resolved.update(
            {
                "trust_max_backtracks": backtracking["max_backtracks"],
                "trust_backtrack_shrink": backtracking["shrink_factor"],
            }
        )

    return resolved
