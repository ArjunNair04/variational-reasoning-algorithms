"""Run a sweep / finalist from a YAML config instead of long CLI arg lists.

Only the algos listed under `algos:` in the YAML are run. Runtime allocation fields
(`rounds`, `batch`, `G`, `prompts`, and `shots`) and any registered hyperparameter may be a SCALAR
(one cell) or a LIST (sweep that axis -> Cartesian product), and is applied only where the trainer
accepts it: lr (all); epochs (GRPO/RLOO/DPO/RFT update passes); iters (AC-EM M-step grad iterations);
beta (AC-AW/AAW, DPO, APL); anchor / causal / pi_temp / sup (AC-EM); clip / kl_coef
(GRPO, RLOO); top_frac (RAFT); pool_mult (APL); window (conditional VRO / legacy weighted-EM);
buffer_strategy, buffer_semantics, proposal_prompt, proposal_mixture, proposal_filter,
proposal_policy, responsibility_score, responsibility_posterior
(conditional VRO / AC-ALG1), responsibility_temperature,
responsibility_answer_policy,
responsibility_ess_floor, responsibility_policy, responsibility_refresh,
labelled_em_weight, answer_only_em_weight, labelled_proposal_prompt, answer_only_proposal_prompt,
labelled_numeric_constraint (including graph_hard), numeric_penalty, labelled_supervision,
compact_gold_weight, policy_kl_coef, supervised_weight, policy_anchor_mode,
policy_anchor_target_ratio, policy_anchor_beta_min, policy_anchor_beta_max, policy_anchor_ema,
L2R proposal_prior_fraction, proposal_temperature, l2r_buffer_semantics, mstep_objective,
policy_anchor_scope, reader_mode, reader_decode_filter, gold_in_buffer,
length_norm, reward_requires_eos, and
latent_mstep_objective, update_geometry, step_acceptance, rollback_tolerance, rollback_max_backtracks,
rollback_shrink, and optimizer_state_scope
(AC-ALG1).
New AC-ALG1/L2R configs should group the policy anchor and L2R trust-region
settings under `policy_kl_control`; run_yaml resolves that block to the flat
trainer arguments above. Historical flat configs remain valid for exact replay.
All trainer parameters used by historical named methods are YAML-overridable.
`method_presets.yaml` defines the stable historical aliases; use the plain
`L2R`, `AC-EM`, `AC-ALG1`, `Conditional-VRO`, or comparator method with
explicit fields when defining a new method rather than adding Python aliases.
Every cell becomes ONE invocation of run_sweep_lm.py with a unique --tag, so you keep
one-CSV-per-cell, resumability, the row-0 params blob, and the per-round trajectory JSONs.

    python run_yaml.py experiments.yaml
    python run_yaml.py experiments.yaml --gpus 2                       # split cells across 2 GPUs
    python run_yaml.py experiments.yaml --only AW-EM GRPO base   # subset of listed algos
    python run_yaml.py experiments.yaml --dry-run                      # list cells, run nothing

Multi-GPU: --gpus N fans out one pinned worker per card (CUDA_VISIBLE_DEVICES), each running a disjoint
slice of the cells (~Nx throughput, no DDP). --workers-per-gpu W runs W cells concurrently per card
(N*W workers total, round-robin onto the cards) to fill big-VRAM cards. A pre-set CUDA_VISIBLE_DEVICES
list is honoured. Per-shard logs go to <out>/run_shard<k>.log.

YAML schema (see experiments.yaml):
    tag_prefix: final60
    defaults: { task, model|models, rounds, seeds, seed_values, n_test, train_partition, eval_partition,
                batch, eval_batch, G, prompts, shots, shot_bank_size,
                answer_event_mode, answer_target_termination,
                transfer_eval_dataset, transfer_eval_n,
                task_seed_from_run_seed, question_sampling, grad_checkpoint, out,
                lora_target_set,
                dump_completions, eval_every|eval_rounds, save_adapter,
                save_training_diagnostics, training_diagnostics_level,
                training_diagnostics_trace_tape,
                training_diagnostics_gradient_questions,
                training_diagnostics_probe_size,
                l2r_candidate_utility_questions,
                l2r_candidate_utility_batch, checkpoint_every,
                l2r_exact_cache, l2r_state_checkpoint_every,
                passk, passk_n }   # seed_values is optional; non-contiguous lists use --seed INDEX
    algos:
      base: {}
      AW-EM: { lr: 2.0e-4, epochs: 16 }
      GRPO:  { lr: [2.0e-5, 5.0e-5, 1.0e-4] }   # list -> 3 cells
      AC-CW: { lr: 1.0e-4, epochs: 8, causal: [0.5, 1, 2] }   # sweep the causal strength gamma
      L2R:
        reader_mode: moving
        gold_in_buffer: true
        responsibility_score: joint
        mstep_objective: joint
      L2R-Frozen-NoGold-LN:
        lr: 1.0e-5
        policy_kl_control:
          mode: adaptive_gradient_ratio
          anchor: { target_gradient_ratio: 0.03, beta_min: 0, beta_max: 10,
                    gradient_norm_ema: 0.9 }
          trust_region: { max_realized_token_kl: 0.03, safety_questions: 64,
                          max_safety_nll_increase: 0.01,
                          max_boundary_failure_fraction: 0.5 }
          backtracking: { max_backtracks: 3, shrink_factor: 0.5 }
      # A list variant may set cell_id: concise_name to keep artifact names short.
    per_model: { gemma-2-2b-it: { GRPO: { lr: 1.0e-5 } } }   # optional per-(model,method) axis override

`models:` (a list) sweeps the base model as an extra axis -- each cell runs one model and the model goes
in its tag/CSV name (see experiments_secondary.yaml). `per_model:` overrides algos for a given model.

Idempotent: a cell whose CSV already exists is skipped, so you can scp + re-run / resume freely.
Prints sweep-level progress + ETA after every cell (same accounting as sweep.sh).
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

import yaml

from experiment_config import (
    ExperimentCell,
    RunDefaults,
    expand_policy_kl_control,
    validate_run_yaml_config,
)
from hparams import KNOBS, flag
from methods_lm import resolve_lora_target_modules
from result_contract import (
    ResultContractError,
    validate_completion_receipt,
    validate_receipt_identity,
)


def _as_list(v):
    """A scalar config value means one cell; a list means sweep that axis."""
    if v is None:
        return [None]
    return v if isinstance(v, (list, tuple)) else [v]


def _fmt(s):
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m"


def _done_rows(
    csv,
    *,
    run_id=None,
    model=None,
    method=None,
    expected_seeds=None,
):
    """Count identity-matched, uniquely seeded, receipt-valid rows in a cell CSV.

    With no expected identity this retains the historical row-count API used by
    older callers. Modern rows carry a completion receipt; legacy rows are
    accepted only when their explicit model/method/seed identity matches.

    run_sweep_lm checkpoints the CSV after EVERY seed, so an interrupted multi-seed cell leaves a
    partial file -- existence alone must not mean done (the finalist runs 5 seeds/cell), else the
    remaining seeds are silently never run and the seed-mean quietly averages fewer seeds."""
    if not os.path.exists(csv):
        return 0
    try:
        import pandas as pd
        frame = pd.read_csv(csv)
    except Exception:
        return 0                                            # unreadable/empty -> rerun (resume no-ops per seed)
    if all(value is None for value in (run_id, model, method, expected_seeds)):
        return len(frame)
    required = {"run_id", "model", "method", "seed"}
    if not required <= set(frame.columns):
        return 0
    expected = set(int(seed) for seed in expected_seeds or [])
    valid: set[int] = set()
    duplicate: set[int] = set()
    root = Path(csv).parent
    for row in frame.to_dict("records"):
        if run_id is not None and str(row.get("run_id")) != str(run_id):
            continue
        if model is not None and row.get("model") != model:
            continue
        if method is not None and row.get("method") != method:
            continue
        try:
            seed = int(row["seed"])
        except (TypeError, ValueError):
            continue
        if expected and seed not in expected:
            continue
        fingerprint = row.get("cell_fingerprint")
        receipt = row.get("completion_receipt")
        if pd.notna(fingerprint) and str(fingerprint):
            if pd.isna(receipt) or not str(receipt):
                continue
            try:
                payload = validate_completion_receipt(
                    root / str(receipt),
                    expected_fingerprint=str(fingerprint),
                    result_root=root,
                    verify_hashes=True,
                )
                validate_receipt_identity(
                    payload,
                    {
                        "run_id": str(run_id),
                        "model": model,
                        "method": method,
                        "seed": seed,
                    },
                )
            except ResultContractError:
                continue
        if seed in valid:
            duplicate.add(seed)
        valid.add(seed)
    return len(valid - duplicate)


def _seed_selection(configured_seeds, requested_seed=None, seed_values=None):
    """Return ``(actual_seed0, count, tag_suffix)`` for a seed task.

    ``requested_seed`` remains an index into the configured seed family.  With
    no explicit ``seed_values`` this preserves the historical 0..N-1
    behaviour.  An explicit non-contiguous seed family is intentionally
    accepted only one array task at a time, because ``run_sweep_lm`` represents
    multi-seed execution as a contiguous ``seed0`` range.
    """
    configured_seeds = int(configured_seeds)
    if configured_seeds < 1:
        raise ValueError(f"configured seeds must be positive, got {configured_seeds}")
    if seed_values is not None:
        if not isinstance(seed_values, (list, tuple)):
            raise ValueError("defaults.seed_values must be a list of integers")
        try:
            seed_values = tuple(int(value) for value in seed_values)
        except (TypeError, ValueError) as exc:
            raise ValueError("defaults.seed_values must contain integers") from exc
        if len(seed_values) != configured_seeds:
            raise ValueError(
                "defaults.seeds must equal len(defaults.seed_values)"
            )
        if any(value < 0 for value in seed_values):
            raise ValueError("defaults.seed_values must be nonnegative")
        if len(set(seed_values)) != len(seed_values):
            raise ValueError("defaults.seed_values must be unique")
        if requested_seed is None:
            raise ValueError(
                "defaults.seed_values require --seed INDEX (one array task per seed)"
            )
    if requested_seed is None:
        return 0, configured_seeds, ""
    requested_seed = int(requested_seed)
    if not 0 <= requested_seed < configured_seeds:
        raise ValueError(
            f"requested seed {requested_seed} is outside configured range "
            f"0..{configured_seeds - 1}"
        )
    actual_seed = (
        seed_values[requested_seed]
        if seed_values is not None
        else requested_seed
    )
    return actual_seed, 1, f"_seed{actual_seed}"


def get_run_id(path, new_run=False, *, persist=True):
    """Read the YAML's run_id, or generate one and WRITE it back (comment-preserving text edit, NOT a
    yaml re-dump). The id is reused on re-runs so the resumable sweep finds its CSVs; `new_run` forces
    a fresh one. Dry-run callers set ``persist=False`` and receive a deterministic
    preview id without touching the config."""

    path = Path(path)

    def resolve(text):
        match = re.search(r"(?m)^run_id:\s*(\S+)\s*$", text)
        if match and not new_run:
            return match.group(1), text
        if not persist:
            salt = b"\0new-run" if new_run else b"\0missing-run-id"
            preview = hashlib.sha256(text.encode("utf-8") + salt).hexdigest()[:8]
            return f"dry{preview}", text
        run_id = secrets.token_hex(4)
        if match:
            updated = re.sub(
                r"(?m)^run_id:.*$",
                f"run_id: {run_id}",
                text,
            )
        else:
            lines = text.splitlines(keepends=True)
            index = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if line.strip()
                    and not line.startswith(("#", " "))
                    and ":" in line
                ),
                0,
            )
            lines.insert(index, f"run_id: {run_id}\n")
            updated = "".join(lines)
        return run_id, updated

    if not persist:
        return resolve(path.read_text(encoding="utf-8"))[0]

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Linux/macOS execution contract
        raise RuntimeError("run-id persistence requires fcntl") from exc

    lock_path = path.parent / ".run_yaml.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        text = path.read_text(encoding="utf-8")
        run_id, updated = resolve(text)
        if updated == text:
            return run_id
        if path.is_symlink():
            raise ValueError(f"refusing to replace symlinked config: {path}")
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
        )
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(updated)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, path.stat().st_mode & 0o7777)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return run_id


# Per-algo sweepable axes: YAML key -> (run_sweep_lm flag, tag abbrev, caster), derived from the ONE
# knob registry (hparams.py -- registry order = tag order, so never reorder there). Scalar = 1 cell,
# list = sweep that axis; the Cartesian product over all set axes becomes one cell each (own CSV).
# Each axis is forwarded only to methods whose trainer accepts it (run_sweep_lm._accepts).
_AXES = [(k.key, flag(k.key), k.abbr, k.cast) for k in KNOBS]
_RUN_AXES = (
    ("rounds", "--rounds", "r", int),
    ("batch", "--batch", "B", int),
    ("G", "--G", "G", int),
    ("prompts", "--prompts", "P", int),
    ("shots", "--shots", "S", int),
)


def _answer_target_termination_axis(value):
    """Validate the per-cell answer-target termination protocol."""

    termination = str(value)
    if termination not in {"none", "eos"}:
        raise ValueError(
            "answer_target_termination must be one of ['eos', 'none'], "
            f"got {value!r}"
        )
    return termination


def _evaluation_prompt_axis(value):
    """Validate the per-cell evaluation prompt intervention."""

    prompt = str(value)
    allowed = {"question", "answer_derive", "answer_derive_first"}
    if prompt not in allowed:
        raise ValueError(
            f"evaluation_prompt must be one of {sorted(allowed)}, got {value!r}"
        )
    return prompt


_PROTOCOL_AXES = (
    (
        "answer_target_termination",
        "--answer-target-termination",
        "aterm",
        _answer_target_termination_axis,
    ),
    (
        "evaluation_prompt",
        "--evaluation-prompt",
        "eprompt",
        _evaluation_prompt_axis,
    ),
)
_CELL_AXES = _RUN_AXES + tuple(_AXES) + _PROTOCOL_AXES


def _axis_cli_args(axes):
    """Build argv tokens for configured axes without adding shell quotes to categorical values."""
    args = []
    for key, cli_flag, _abbr, _cast in _CELL_AXES:
        if key not in axes:
            continue
        value = axes[key]
        args += [cli_flag, value if isinstance(value, str) else repr(value)]
    return args


def _slug(s):
    """Filesystem-safe model token for the tag (raw HF ids contain '/')."""
    return str(s).replace("/", "-")


def _axis_tag_value(v):
    """Compact, filesystem-safe tag fragment for numeric or categorical axes."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:g}"
    return _slug(v)


def _models(cfg):
    """The model axis: `models:` (a list) sweeps the base model as an extra dimension; a scalar
    `model:` (back-compat) is the single-model case. Defaults to the Qwen flagship."""
    d = cfg.get("defaults") or {}
    ms = d.get("models") or [d.get("model", "qwen2.5-1.5b-instruct")]
    return list(ms) if isinstance(ms, (list, tuple)) else [ms]


def _cells(cfg, only, run_id):
    """Expand the YAML `algos` map (x the model axis) into a flat list of cells (model, method, axes, tag).
    ANY numeric hyperparameter may be a scalar (1 cell) or a list (sweep). `per_model: {model: {method:
    {axis: val}}}` overrides algos for that (model, method) -- e.g. a secondary model whose screened
    optimum differs. The model appears in the tag/CSV name only when >1 is swept (single-model tags unchanged)."""
    prefix = f"{cfg.get('tag_prefix', 'exp')}_{run_id}"    # run_id in the tag -> in every CSV/traj name
    algos = cfg.get("algos") or {}
    per_model = cfg.get("per_model") or {}
    models = _models(cfg)
    multi = len(models) > 1
    cells = []
    for model in models:
        ov = per_model.get(model) or {}
        for method, method_spec in algos.items():
            if only and method not in only:
                continue
            variants = method_spec if isinstance(method_spec, list) else [method_spec or {}]
            override = ov.get(method) or {}
            if not isinstance(override, dict):
                raise ValueError(f"per_model override for {model}/{method} must be a mapping")
            for variant in variants:
                if not isinstance(variant, dict):
                    raise ValueError(f"each grid variant for {method} must be a mapping")
                spec = {**variant, **override}               # per-model override wins over each shared variant
                cell_id = spec.pop("cell_id", None)
                if cell_id is not None:
                    cell_id = str(cell_id)
                    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", cell_id):
                        raise ValueError(
                            "cell_id must be 1-64 filesystem-safe characters "
                            "starting with a letter or digit"
                        )
                spec = expand_policy_kl_control(spec, method=method)
                unknown = set(spec) - {key for key, *_ in _CELL_AXES}
                if unknown:
                    raise ValueError(
                        f"unknown cell field(s) for {method}: {sorted(unknown)}"
                    )
                grids = [_as_list(spec.get(key)) for key, *_ in _CELL_AXES]
                for combo in itertools.product(*grids):
                    axes = {}
                    tag = prefix + (f"_{_slug(model)}" if multi else "") + f"_{method}"
                    if cell_id is not None:
                        tag += f"_{cell_id}"
                    for (key, _flag, abbr, cast), v in zip(_CELL_AXES, combo):
                        if v is None:
                            continue
                        v = cast(v)                          # PyYAML may hand back "2e-4" as a str
                        axes[key] = v
                        if cell_id is None:
                            tag += f"_{abbr}{_axis_tag_value(v)}"
                    cells.append(ExperimentCell(model, method, axes, tag))
    tags = [tag for *_rest, tag in cells]
    if len(tags) != len(set(tags)):
        raise ValueError("grid expansion produced duplicate cell tags")
    return cells


def _validate_cell_method_contracts(cells, defaults):
    """Reject unknown methods and axes the selected trainer cannot consume."""

    from run_sweep_lm import METHODS, _accepts

    runner_axes = {
        key for key, *_rest in (*_RUN_AXES, *_PROTOCOL_AXES)
    }
    configured_trainers = {
        cell.method: METHODS.get(cell.method)
        for cell in cells
        if cell.method != "base"
    }
    diagnostic_requirements = {
        "diagnostics_fn": bool(defaults.get("save_training_diagnostics")),
        "diagnostics_level": (
            defaults.get("training_diagnostics_level")
            not in (None, "standard")
        ),
        "diagnostics_trace_tape": bool(
            defaults.get("training_diagnostics_trace_tape")
        ),
        "diagnostics_gradient_questions": bool(
            defaults.get("training_diagnostics_gradient_questions")
        ),
        "diagnostics_probe_fn": bool(
            defaults.get("training_diagnostics_probe_size")
        ),
        "candidate_utility_questions": bool(
            defaults.get("l2r_candidate_utility_questions")
        ),
        "candidate_utility_batch": bool(
            defaults.get("l2r_candidate_utility_questions")
        ),
    }
    for parameter, enabled in diagnostic_requirements.items():
        if enabled and configured_trainers and not any(
            trainer is not None and _accepts(trainer, parameter)
            for trainer in configured_trainers.values()
        ):
            methods = ", ".join(sorted(configured_trainers))
            raise ValueError(
                f"{methods} does not support requested run default(s): "
                f"['{parameter}']"
            )
    for cell in cells:
        if cell.method == "base":
            unsupported = set(cell.axes) - runner_axes
        else:
            trainer = METHODS.get(cell.method)
            if trainer is None:
                raise ValueError(f"unknown method in algos: {cell.method!r}")
            unsupported = {
                key
                for key in cell.axes
                if key not in runner_axes and not _accepts(trainer, key)
            }
            effective_answer_termination = cell.axes.get(
                "answer_target_termination",
                defaults.get("answer_target_termination", "none"),
            )
            if (
                effective_answer_termination != "none"
                and not _accepts(trainer, "answer_target_termination")
            ):
                unsupported.add("answer_target_termination")
        if unsupported:
            raise ValueError(
                f"{cell.method} does not accept configured cell field(s): "
                f"{sorted(unsupported)}"
            )
        if cell.method == "base":
            continue
        requirements = {
            "eval_every": bool(
                defaults.get("eval_every") or defaults.get("eval_rounds")
            ),
            "eval_fn": bool(
                defaults.get("eval_every") or defaults.get("eval_rounds")
            ),
            "eval_rounds": bool(defaults.get("eval_rounds")),
            # Diagnostics are study-level instrumentation and are deliberately
            # selective in mixed-method YAMLs. The runner emits them only for
            # trainers exposing each capability. The study must contain at
            # least one compatible trainer for every requested diagnostic.
            **{
                parameter: enabled and _accepts(trainer, parameter)
                for parameter, enabled in diagnostic_requirements.items()
            },
            "checkpoint_every": bool(defaults.get("checkpoint_every")),
            "checkpoint_fn": bool(defaults.get("checkpoint_every")),
            "exact_cache": bool(defaults.get("l2r_exact_cache")),
            "state_checkpoint_every": bool(
                defaults.get("l2r_state_checkpoint_every")
            ),
            "state_checkpoint_fn": bool(
                defaults.get("l2r_state_checkpoint_every")
            ),
            "resume_fingerprint": bool(
                defaults.get("l2r_state_checkpoint_every")
            ),
            "resume_state": bool(
                defaults.get("l2r_state_checkpoint_every")
            ),
        }
        unsupported_defaults = {
            parameter
            for parameter, enabled in requirements.items()
            if enabled and not _accepts(trainer, parameter)
        }
        if unsupported_defaults:
            raise ValueError(
                f"{cell.method} does not support requested run default(s): "
                f"{sorted(unsupported_defaults)}"
            )


def _prepare_cells(cfg, *, only, run_id, defaults):
    cells = _cells(cfg, only, run_id)
    _validate_cell_method_contracts(cells, defaults)
    eval_rounds = tuple(int(value) for value in defaults.get("eval_rounds") or ())
    default_rounds = int(defaults.get("rounds", RunDefaults.rounds))
    longest_cell = max(
        int(cell.axes.get("rounds", default_rounds)) for cell in cells
    )
    beyond_horizon = sorted(
        value for value in eval_rounds if value > longest_cell
    )
    if beyond_horizon:
        raise ValueError(
            "defaults.eval_rounds cannot exceed the longest cell's "
            f"{longest_cell} training rounds: {beyond_horizon}"
        )
    for model, _method, axes, _tag in cells:
        resolve_lora_target_modules(
            model,
            lora_target_set=str(
                axes.get(
                    "lora_target_set",
                    defaults.get("lora_target_set", "attention"),
                )
            ),
        )
    return cells


def _default_cli_args(defaults, axes, *, method=None):
    """Translate optional run defaults into run_sweep_lm CLI arguments."""

    args = []
    if defaults.get("shot_bank_size") is not None:
        args += ["--shot-bank-size", str(int(defaults["shot_bank_size"]))]
    if defaults.get("task_seed_from_run_seed"):
        args.append("--task-seed-from-run-seed")
    if defaults.get("question_sampling") is not None:
        args += ["--question-sampling", str(defaults["question_sampling"])]
    if defaults.get("grad_checkpoint", True):
        args.append("--grad-checkpoint")
    # Frozen-base evaluation never constructs a trainable adapter.  A study may
    # still place LoRA defaults at the study level so every trained cell shares
    # one surface; do not forward those training-only flags to the base cell.
    if method != "base":
        for key, flag_name, caster in (
            ("lora_r", "--lora-r", int),
            ("lora_seed", "--lora-seed", int),
            ("lora_alpha", "--lora-alpha", int),
            ("lora_target_set", "--lora-target-set", str),
        ):
            if key in defaults and key not in axes:
                args += [flag_name, str(caster(defaults[key]))]
    if defaults.get("dump_completions"):
        args += ["--dump-completions", str(defaults["dump_completions"])]
    if defaults.get("eval_every"):
        args += ["--eval-every", str(defaults["eval_every"])]
    if defaults.get("eval_rounds"):
        values = defaults["eval_rounds"]
        if not isinstance(values, (list, tuple)):
            raise ValueError("defaults.eval_rounds must be a list of completed rounds")
        cell_rounds = int(
            axes.get(
                "rounds",
                defaults.get("rounds", RunDefaults.rounds),
            )
        )
        cell_eval_rounds = sorted(
            {
                *(int(value) for value in values if int(value) <= cell_rounds),
                cell_rounds,
            }
        )
        args += [
            "--eval-rounds",
            ",".join(str(value) for value in cell_eval_rounds),
        ]
    if defaults.get("save_adapter"):
        args.append("--save-adapter")
    if defaults.get("save_training_diagnostics"):
        args.append("--save-training-diagnostics")
        diagnostics_level = defaults.get(
            "training_diagnostics_level",
            "standard",
        )
        if (
            (
                defaults.get("training_diagnostics_gradient_questions")
                or defaults.get("training_diagnostics_probe_size")
                or defaults.get("l2r_candidate_utility_questions")
            )
            and diagnostics_level != "deep"
        ):
            raise ValueError(
                "question-gradient, fixed-probe, and candidate-utility "
                "diagnostics require training_diagnostics_level=deep"
            )
        if defaults.get("training_diagnostics_level") is not None:
            args += [
                "--training-diagnostics-level",
                str(defaults["training_diagnostics_level"]),
            ]
        if defaults.get("training_diagnostics_trace_tape"):
            args.append("--training-diagnostics-trace-tape")
        if defaults.get("training_diagnostics_gradient_questions"):
            args += [
                "--training-diagnostics-gradient-questions",
                str(int(defaults["training_diagnostics_gradient_questions"])),
            ]
        if defaults.get("training_diagnostics_probe_size"):
            args += [
                "--training-diagnostics-probe-size",
                str(int(defaults["training_diagnostics_probe_size"])),
            ]
        if defaults.get("l2r_candidate_utility_questions"):
            args += [
                "--l2r-candidate-utility-questions",
                str(int(defaults["l2r_candidate_utility_questions"])),
                "--l2r-candidate-utility-batch",
                str(int(defaults.get("l2r_candidate_utility_batch", 16))),
            ]
    elif (
        defaults.get("training_diagnostics_level") not in (None, "standard")
        or defaults.get("training_diagnostics_trace_tape")
        or defaults.get("training_diagnostics_gradient_questions")
        or defaults.get("training_diagnostics_probe_size")
        or defaults.get("l2r_candidate_utility_questions")
    ):
        raise ValueError(
            "training diagnostic options require "
            "defaults.save_training_diagnostics=true"
        )
    if defaults.get("checkpoint_every"):
        args += ["--checkpoint-every", str(defaults["checkpoint_every"])]
    if defaults.get("l2r_exact_cache"):
        args.append("--l2r-exact-cache")
    if defaults.get("l2r_state_checkpoint_every"):
        args += [
            "--l2r-state-checkpoint-every",
            str(int(defaults["l2r_state_checkpoint_every"])),
        ]
    if defaults.get("passk"):
        args += ["--passk", str(defaults["passk"])]
        if defaults.get("passk_n"):
            args += ["--passk-n", str(defaults["passk_n"])]
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to the YAML experiment config")
    ap.add_argument("--only", nargs="+", default=None, help="run only these listed algos")
    ap.add_argument("--dry-run", action="store_true", help="print the cells and exit")
    ap.add_argument("--new-run", action="store_true",
                    help="force a fresh run_id (else reuse the YAML's, so the sweep resumes)")
    ap.add_argument("--gpus", type=int, default=1,
                    help="split cells across N GPUs (one sharded worker per card)")
    ap.add_argument("--workers-per-gpu", type=int, default=1,
                    help="run W cells concurrently per GPU (oversubscribe big-VRAM cards for more throughput)")
    ap.add_argument("--seed", type=int, default=None,
                    help="run one configured seed with a collision-free tag (for SGE seed arrays)")
    ap.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)    # internal: this worker's index
    ap.add_argument("--nshard", type=int, default=1, help=argparse.SUPPRESS)   # internal: total worker count
    ap.add_argument(
        "--expect-cells",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()

    if args.nshard < 1:
        ap.error("--nshard must be positive")
    if args.shard < 0 or args.shard >= args.nshard:
        ap.error("--shard must be in [0, --nshard)")
    if args.expect_cells is not None and args.expect_cells < 0:
        ap.error("--expect-cells must be nonnegative")

    path = os.path.expanduser(args.config)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    validate_run_yaml_config(cfg)
    d = cfg.get("defaults") or {}
    defaults = RunDefaults.from_mapping(d)
    utility_questions = int(d.get("l2r_candidate_utility_questions", 0))
    utility_batch = int(d.get("l2r_candidate_utility_batch", 16))
    if utility_questions < 0:
        raise ValueError(
            "defaults.l2r_candidate_utility_questions must be nonnegative"
        )
    if utility_batch < 1:
        raise ValueError(
            "defaults.l2r_candidate_utility_batch must be positive"
        )
    if utility_questions and (
        not d.get("save_training_diagnostics")
        or d.get("training_diagnostics_level") != "deep"
    ):
        raise ValueError(
            "the L2R candidate utility audit requires saved deep diagnostics"
        )
    task = defaults.task
    out = os.path.expanduser(defaults.out)
    seed_values = d.get("seed_values")
    configured_seeds = d.get(
        "seeds",
        len(seed_values) if isinstance(seed_values, (list, tuple)) else 1,
    )
    dry_run_seed_family = (
        args.dry_run
        and args.seed is None
        and isinstance(seed_values, (list, tuple))
    )
    if dry_run_seed_family:
        seed_selections = [
            _seed_selection(
                configured_seeds,
                index,
                seed_values=seed_values,
            )
            for index in range(int(configured_seeds))
        ]
        seed0, n_seeds, seed_suffix = seed_selections[0]
    else:
        seed_selections = []
        seed0, n_seeds, seed_suffix = _seed_selection(
            configured_seeds,
            args.seed,
            seed_values=seed_values,
        )

    # Validate every expanded cell before assigning a persistent identity. A
    # dry run is strictly read-only, including when the YAML has no run_id.
    preview_run_id = get_run_id(
        path,
        new_run=args.new_run,
        persist=False,
    )
    cells = _prepare_cells(
        cfg,
        only=set(args.only) if args.only else None,
        run_id=preview_run_id,
        defaults=d,
    )
    if args.dry_run:
        run_id = preview_run_id
    else:
        run_id = get_run_id(path, new_run=args.new_run, persist=True)
        if run_id != preview_run_id:
            cells = _prepare_cells(
                cfg,
                only=set(args.only) if args.only else None,
                run_id=run_id,
                defaults=d,
            )
    if dry_run_seed_family:
        cells = [
            ExperimentCell(model, method, axes, tag + suffix)
            for _seed, _count, suffix in seed_selections
            for model, method, axes, tag in cells
        ]
    elif seed_suffix:
        cells = [
            ExperimentCell(model, method, axes, tag + seed_suffix)
            for model, method, axes, tag in cells
        ]
    if args.nshard > 1:
        cells = [
            cell
            for index, cell in enumerate(cells)
            if index % args.nshard == args.shard
        ]
    if args.expect_cells is not None and len(cells) != args.expect_cells:
        raise ValueError(
            "cell-selection contract failed: "
            f"expected {args.expect_cells} cell(s), selected {len(cells)} "
            f"for shard {args.shard}/{args.nshard}"
        )
    if not cells:
        print("no cells to run (check `algos:` / --only)")
        return

    if not args.dry_run:
        os.makedirs(out, exist_ok=True)

    # ---- multi-GPU fan-out (parent only): WORKERS_PER_GPU workers per card, disjoint slice of cells ----
    if args.nshard == 1 and not args.dry_run:              # nshard>1 => already a spawned worker
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_list = cvd.split(",") if cvd else [str(i) for i in range(args.gpus)]
        n = len(gpu_list) * args.workers_per_gpu           # total workers (round-robin onto the cards)
        if n > 1:
            print(f"=== {n}-worker run ({args.workers_per_gpu}/GPU x {len(gpu_list)} GPUs; "
                  f"progress: tail -f {out}/run_shard*.log) ===", flush=True)
            procs = []
            for k in range(n):
                dev = gpu_list[k % len(gpu_list)]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": dev}
                cmd = [sys.executable, os.path.abspath(__file__), args.config,
                       "--gpus", "1", "--workers-per-gpu", "1", "--shard", str(k), "--nshard", str(n)]
                if args.only:
                    cmd += ["--only", *args.only]
                if args.seed is not None:
                    cmd += ["--seed", str(args.seed)]
                fh = open(os.path.join(out, f"run_shard{k}.log"), "w")
                procs.append(subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT))
                print(f"  shard {k}/{n} -> physical GPU {dev} (pid {procs[-1].pid})", flush=True)
            rc = 0
            for p in procs:
                rc |= p.wait()
            print(f"=== all {n} shards finished (rc={rc}) ===")
            sys.exit(rc)

    if dry_run_seed_family:
        seed_label = "seed_values=" + ",".join(
            str(seed) for seed, _count, _suffix in seed_selections
        )
    else:
        seed_label = f"seed={seed0}" if args.seed is not None else f"seeds={n_seeds}"
    print(f"=== run_id={run_id} | {len(cells)} cells | {len(_models(cfg))} model(s) | task={task} "
          f"rounds={d.get('rounds')} {seed_label} ===")
    for model, method, axes, tag in cells:
        print(f"  {tag:48s}  model={model} method={method} " + " ".join(f"{k}={v}" for k, v in axes.items()))
    if args.dry_run:
        return

    sweep0 = time.time()
    run = runt = 0
    failures = []
    for i, (model, method, axes, tag) in enumerate(cells, 1):
        csv = os.path.join(out, f"sweep_{task}__{tag}.csv")
        have = _done_rows(
            csv,
            run_id=run_id,
            model=model,
            method=method,
            expected_seeds=range(seed0, seed0 + n_seeds),
        )                                                   # one cell = one model x method x n_seeds rows
        if have >= n_seeds:
            print(f"[{i}/{len(cells)}] SKIP (done, {have}/{n_seeds} seeds): {tag}", flush=True); continue
        if have:
            print(f"[{i}/{len(cells)}] RESUME ({have}/{n_seeds} seeds): {tag}", flush=True)
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "run_sweep_lm.py"),
               *defaults.cli_args(), "--models", str(model),
               "--methods", method, "--seeds", str(n_seeds), "--seed0", str(seed0),
               "--out", out, "--tag", tag, "--run-id", run_id]
        cmd += _default_cli_args(d, axes, method=method)
        cmd += _axis_cli_args(axes)                          # forward each set axis to run_sweep_lm.py
        print(f"[{i}/{len(cells)}] {time.strftime('%H:%M:%S')}  RUN {tag}", flush=True)
        c0 = time.time()
        proc = subprocess.run(cmd, check=False)             # finish this shard, then return failure to SGE
        dt = int(time.time() - c0); run += 1; runt += dt
        if proc.returncode:
            failures.append((tag, proc.returncode))
        avg = runt // run; eta = avg * (len(cells) - i)
        status = "FAILED" if proc.returncode else "done"
        print(f"[{i}/{len(cells)}] {status} {tag} in {dt}s | avg {avg}s/cell | "
              f"elapsed {_fmt(int(time.time() - sweep0))} | ETA ~{_fmt(eta)}", flush=True)
    print(f"=== done in {_fmt(int(time.time() - sweep0))} -- CSVs: {out}/sweep_{task}__{cfg.get('tag_prefix','exp')}_*.csv ===")
    if failures:
        print(f"ERROR: {len(failures)} failed cell(s) in this shard:", file=sys.stderr)
        for tag, rc in failures:
            print(f"  {tag}: exit {rc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
