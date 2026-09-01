"""Structural controls for answer-conditioned Learning-to-Reason.

The helpers in this module make three interventions explicit and testable:

* low-rank, LoRA-B-only updates whose effective weight changes can be averaged;
* projection of an M-step gradient into either an empirical or random subspace;
* deterministic uniform, cyclic, or failure-prioritised question scheduling.

They are deliberately independent of GSM8K and of the L2R objective itself so
their invariants can be unit-tested without loading a language model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


LORA_TRAINABLE_MODES = {"all", "b_only"}
GRADIENT_PROJECTION_MODES = {"none", "basis", "random"}
QUESTION_SCHEDULES = {"uniform", "cyclic", "priority"}


@dataclass(frozen=True)
class ParameterSpec:
    """Stable flattened-parameter layout used by saved gradient bases."""

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    numels: tuple[int, ...]

    @property
    def total(self) -> int:
        return sum(self.numels)

    @property
    def fingerprint(self) -> str:
        payload = "\n".join(
            f"{name}:{','.join(map(str, shape))}"
            for name, shape in zip(self.names, self.shapes)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configure_lora_trainable(model, mode: str) -> list[tuple[str, torch.nn.Parameter]]:
    """Return the trainable layout after applying a LoRA update restriction."""

    if mode not in LORA_TRAINABLE_MODES:
        raise ValueError(f"unknown LoRA trainable mode {mode!r}")
    if mode == "b_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_("lora_B" in name)
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not selected:
        raise ValueError(f"LoRA trainable mode {mode!r} selected no parameters")
    if mode == "b_only" and any("lora_B" not in name for name, _ in selected):
        raise ValueError("b_only mode left a non-LoRA-B parameter trainable")
    return selected


def parameter_spec(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> ParameterSpec:
    named_parameters = list(named_parameters)
    return ParameterSpec(
        names=tuple(name for name, _ in named_parameters),
        shapes=tuple(tuple(parameter.shape) for _, parameter in named_parameters),
        numels=tuple(parameter.numel() for _, parameter in named_parameters),
    )


def flatten_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    gradients: bool = False,
) -> torch.Tensor:
    """Flatten parameters or gradients in a stable, float32 layout."""

    chunks = []
    for _name, parameter in named_parameters:
        value = parameter.grad if gradients else parameter.data
        if value is None:
            value = torch.zeros_like(parameter)
        chunks.append(value.detach().reshape(-1).float())
    if not chunks:
        raise ValueError("cannot flatten an empty parameter layout")
    return torch.cat(chunks)


def assign_flat_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    flat: torch.Tensor,
    *,
    gradients: bool = False,
) -> None:
    """Assign a flattened vector to parameters or their gradients."""

    named_parameters = list(named_parameters)
    expected = sum(parameter.numel() for _, parameter in named_parameters)
    if flat.numel() != expected:
        raise ValueError(f"flat vector has {flat.numel()} values; expected {expected}")
    offset = 0
    for _name, parameter in named_parameters:
        stop = offset + parameter.numel()
        value = flat[offset:stop].reshape(parameter.shape).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        if gradients:
            if parameter.grad is None:
                parameter.grad = value.clone()
            else:
                parameter.grad.copy_(value)
        else:
            parameter.data.copy_(value)
        offset = stop


def snapshot_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> torch.Tensor:
    return flatten_parameters(named_parameters).detach().cpu()


def restore_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    snapshot: torch.Tensor,
) -> None:
    assign_flat_parameters(named_parameters, snapshot, gradients=False)


def _orthonormal_rows(matrix: torch.Tensor, *, eps: float = 1e-10) -> torch.Tensor:
    """Modified Gram-Schmidt for a small number of very long row vectors."""

    rows = []
    for candidate in matrix.float():
        vector = candidate.clone()
        for row in rows:
            vector -= torch.dot(vector, row) * row
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(norm) or norm <= eps:
            continue
        rows.append(vector / norm)
    if not rows:
        raise ValueError("basis contains no finite non-zero direction")
    return torch.stack(rows)


def save_gradient_basis(
    path: str | Path,
    basis: torch.Tensor,
    spec: ParameterSpec,
    *,
    metadata: dict | None = None,
) -> None:
    """Persist an orthonormal basis with a fail-closed parameter layout."""

    basis = _orthonormal_rows(basis.detach().cpu())
    if basis.shape[1] != spec.total:
        raise ValueError(
            f"basis width {basis.shape[1]} does not match parameter width {spec.total}"
        )
    payload = {
        "schema_version": 1,
        "parameter_names": list(spec.names),
        "parameter_shapes": [list(shape) for shape in spec.shapes],
        "parameter_fingerprint": spec.fingerprint,
        "basis": basis.to(torch.float16),
        "metadata": metadata or {},
    }
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_gradient_basis(
    path: str | Path,
    spec: ParameterSpec,
    *,
    rank: int,
) -> tuple[torch.Tensor, dict]:
    """Load and validate an empirical basis against the current model."""

    if rank < 1:
        raise ValueError("gradient projection rank must be positive")
    path = Path(path).expanduser()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported gradient basis schema in {path}")
    if tuple(payload.get("parameter_names", ())) != spec.names:
        raise ValueError("gradient basis parameter names do not match the model")
    shapes = tuple(tuple(shape) for shape in payload.get("parameter_shapes", ()))
    if shapes != spec.shapes:
        raise ValueError("gradient basis parameter shapes do not match the model")
    if payload.get("parameter_fingerprint") != spec.fingerprint:
        raise ValueError("gradient basis parameter fingerprint does not match the model")
    basis = _orthonormal_rows(payload["basis"].float())
    if basis.shape[1] != spec.total:
        raise ValueError("gradient basis width does not match the model")
    if rank > len(basis):
        raise ValueError(f"requested rank {rank}, but basis contains only {len(basis)} rows")
    return basis[:rank].contiguous(), dict(payload.get("metadata") or {})


class GradientProjector:
    """Constrain optimizer updates to a fixed low-dimensional subspace.

    Projecting a gradient before an adaptive optimizer is insufficient: Adam's
    coordinate-wise preconditioner can rotate the resulting parameter update
    out of the requested subspace. ``step`` therefore lets the optimizer form
    its proposed update, projects that actual parameter delta, and writes back
    only the constrained delta.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        mode: str = "none",
        rank: int = 0,
        basis_path: str | Path | None = None,
        seed: int = 0,
        preserve_norm: bool = True,
    ):
        if mode not in GRADIENT_PROJECTION_MODES:
            raise ValueError(f"unknown gradient projection mode {mode!r}")
        self.named_parameters = list(named_parameters)
        self.spec = parameter_spec(self.named_parameters)
        self.mode = mode
        self.rank = int(rank)
        self.preserve_norm = bool(preserve_norm)
        self.metadata: dict = {}
        self._device_basis: dict[str, torch.Tensor] = {}
        if mode == "none":
            if self.rank not in (0,):
                raise ValueError("projection rank must be zero when projection is disabled")
            self.basis = None
            return
        if self.rank < 1 or self.rank > self.spec.total:
            raise ValueError(
                f"projection rank must be in [1, {self.spec.total}], got {self.rank}"
            )
        if mode == "basis":
            if not basis_path:
                raise ValueError("basis projection requires a basis path")
            self.basis, self.metadata = load_gradient_basis(
                basis_path,
                self.spec,
                rank=self.rank,
            )
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            random = torch.randn(
                self.rank,
                self.spec.total,
                generator=generator,
                dtype=torch.float32,
            )
            self.basis = _orthonormal_rows(random)
            if len(self.basis) != self.rank:
                raise ValueError("failed to construct the requested random projection rank")
            self.metadata = {"random_seed": int(seed)}

    def _basis_on(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._device_basis:
            self._device_basis[key] = self.basis.to(device=device, dtype=torch.float32)
        return self._device_basis[key]

    def _project_vector(self, vector: torch.Tensor) -> tuple[torch.Tensor, dict]:
        raw_norm = float(torch.linalg.vector_norm(vector))
        if self.mode == "none":
            return vector, {
                "optimizer_update_norm_raw": raw_norm,
                "optimizer_update_norm_projected": raw_norm,
                "optimizer_update_norm_applied": raw_norm,
                "projection_retained_fraction": 1.0,
            }
        if raw_norm == 0.0:
            return vector, {
                "optimizer_update_norm_raw": 0.0,
                "optimizer_update_norm_projected": 0.0,
                "optimizer_update_norm_applied": 0.0,
                "projection_retained_fraction": 0.0,
            }
        basis = self._basis_on(vector.device)
        projected = basis.T @ (basis @ vector)
        projected_norm_tensor = torch.linalg.vector_norm(projected)
        projected_norm = float(projected_norm_tensor)
        retained = projected_norm / raw_norm
        if self.preserve_norm and projected_norm > 0:
            projected = projected * (raw_norm / projected_norm)
        applied_norm = float(torch.linalg.vector_norm(projected))
        return projected, {
            "optimizer_update_norm_raw": raw_norm,
            "optimizer_update_norm_projected": projected_norm,
            "optimizer_update_norm_applied": applied_norm,
            "projection_retained_fraction": retained,
        }

    def project(self) -> dict:
        """Project current gradients.

        This helper is retained for isolated diagnostics. Training should use
        ``step`` so adaptive-optimizer updates cannot escape the subspace.
        """
        gradient = flatten_parameters(self.named_parameters, gradients=True)
        raw_norm = float(torch.linalg.vector_norm(gradient))
        if self.mode == "none":
            return {
                "gradient_norm_raw": raw_norm,
                "gradient_norm_projected": raw_norm,
                "gradient_norm_applied": raw_norm,
                "projection_retained_fraction": 1.0,
            }
        if raw_norm == 0.0:
            return {
                "gradient_norm_raw": 0.0,
                "gradient_norm_projected": 0.0,
                "gradient_norm_applied": 0.0,
                "projection_retained_fraction": 0.0,
            }
        basis = self._basis_on(gradient.device)
        projected = basis.T @ (basis @ gradient)
        projected_norm_tensor = torch.linalg.vector_norm(projected)
        projected_norm = float(projected_norm_tensor)
        retained = projected_norm / raw_norm
        if self.preserve_norm and projected_norm > 0:
            projected = projected * (raw_norm / projected_norm)
        applied_norm = float(torch.linalg.vector_norm(projected))
        assign_flat_parameters(self.named_parameters, projected, gradients=True)
        return {
            "gradient_norm_raw": raw_norm,
            "gradient_norm_projected": projected_norm,
            "gradient_norm_applied": applied_norm,
            "projection_retained_fraction": retained,
        }

    def step(self, optimizer: torch.optim.Optimizer) -> dict:
        """Apply one optimizer step, constraining the actual parameter delta."""

        gradient = flatten_parameters(self.named_parameters, gradients=True)
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        before = flatten_parameters(self.named_parameters)
        optimizer.step()
        proposed = flatten_parameters(self.named_parameters) - before
        applied, diagnostics = self._project_vector(proposed)
        if self.mode != "none":
            assign_flat_parameters(
                self.named_parameters,
                before + applied,
                gradients=False,
            )
        return {
            "gradient_norm_raw": gradient_norm,
            **diagnostics,
        }


def replicated_responsibilities(
    logits: torch.Tensor,
    pids: torch.Tensor,
    replicas: torch.Tensor,
    is_gold: torch.Tensor,
    active: torch.Tensor,
    *,
    replicate_count: int,
    temperature: float,
) -> torch.Tensor:
    """Average independently normalised per-replica posteriors.

    Sampled traces belong to exactly one replica. A gold trace, when present,
    participates in every replica. Each replica contributes equal total mass,
    so the resulting weights still sum to one per question.
    """

    if replicate_count < 1:
        raise ValueError("replicate_count must be positive")
    shape = logits.shape
    if any(value.shape != shape for value in (pids, replicas, is_gold, active)):
        raise ValueError("replicated responsibility tensors must share one shape")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if bool(((replicas < 0) & ~is_gold).any()):
        raise ValueError("only gold traces may use the shared replica id -1")
    weights = torch.zeros_like(logits)
    for pid in torch.unique(pids, sorted=True):
        question = pids == pid
        for replica in range(replicate_count):
            local = question & active & (is_gold | (replicas == replica))
            if not bool(local.any()):
                raise ValueError(
                    f"question {int(pid)} has no active support in replica {replica}"
                )
            weights[local] += (
                torch.softmax(logits[local] / temperature, dim=0)
                / replicate_count
            )
    return weights


@dataclass
class _QuestionState:
    recent_correct: float = 0.0
    best_correct: float = 0.0
    uncertainty: float = 1.0
    last_seen: int = -1
    exposures: int = 0


class QuestionScheduler:
    """Deterministic uniform, cyclic, or failure-prioritised sampler."""

    def __init__(
        self,
        n_questions: int,
        *,
        seed: int,
        mode: str = "uniform",
        exploration: float = 0.1,
    ):
        if n_questions < 1:
            raise ValueError("n_questions must be positive")
        if mode not in QUESTION_SCHEDULES:
            raise ValueError(f"unknown question schedule {mode!r}")
        if not math.isfinite(exploration) or not 0 <= exploration <= 1:
            raise ValueError("schedule exploration must be in [0, 1]")
        self.n_questions = int(n_questions)
        self.mode = mode
        self.exploration = float(exploration)
        self.rng = np.random.default_rng(seed)
        self.states = [_QuestionState() for _ in range(self.n_questions)]
        self._cycle: list[int] = []
        self.last_priority_count = 0

    def _cyclic(self, count: int) -> list[int]:
        selected = []
        while len(selected) < count:
            if not self._cycle:
                self._cycle = [
                    int(value) for value in self.rng.permutation(self.n_questions)
                ]
            take = min(count - len(selected), len(self._cycle))
            selected.extend(self._cycle[:take])
            del self._cycle[:take]
        return selected

    def _priority_score(self, pid: int, round_index: int) -> float:
        state = self.states[pid]
        if state.exposures == 0:
            return 2.0
        unresolved = 1.0 - state.recent_correct
        forgetting = max(state.best_correct - state.recent_correct, 0.0)
        age = min(max(round_index - state.last_seen, 0) / 10.0, 1.0)
        return (
            0.45 * unresolved
            + 0.25 * state.uncertainty
            + 0.20 * forgetting
            + 0.10 * age
        )

    def select(self, count: int, round_index: int) -> list[int]:
        if count < 1 or count > self.n_questions:
            raise ValueError(
                f"question count must be in [1, {self.n_questions}], got {count}"
            )
        if self.mode == "uniform":
            self.last_priority_count = 0
            return [
                int(value)
                for value in self.rng.choice(
                    self.n_questions,
                    size=count,
                    replace=False,
                )
            ]
        if self.mode == "cyclic":
            self.last_priority_count = 0
            return self._cyclic(count)

        scores = np.asarray(
            [self._priority_score(pid, round_index) for pid in range(self.n_questions)],
            dtype=np.float64,
        )
        expected_priority = (1.0 - self.exploration) * count
        priority_count = math.floor(expected_priority)
        if self.rng.random() < expected_priority - priority_count:
            priority_count += 1
        priority_count = min(count, max(0, priority_count))
        self.last_priority_count = priority_count
        jitter = self.rng.uniform(0.0, 1e-9, size=self.n_questions)
        order = np.argsort(-(scores + jitter))
        selected = [int(pid) for pid in order[:priority_count]]
        if len(selected) < count:
            remaining = np.asarray(
                [pid for pid in range(self.n_questions) if pid not in set(selected)],
                dtype=int,
            )
            extra = self.rng.choice(
                remaining,
                size=count - len(selected),
                replace=False,
            )
            selected.extend(int(pid) for pid in extra)
        return selected

    def observe(
        self,
        pid: int,
        *,
        correct_rate: float,
        uncertainty: float,
        round_index: int,
    ) -> None:
        if pid < 0 or pid >= self.n_questions:
            raise ValueError(f"question id out of range: {pid}")
        if not math.isfinite(correct_rate) or not 0 <= correct_rate <= 1:
            raise ValueError("correct_rate must be finite and in [0, 1]")
        if not math.isfinite(uncertainty) or not 0 <= uncertainty <= 1:
            raise ValueError("uncertainty must be finite and in [0, 1]")
        state = self.states[pid]
        state.recent_correct = float(correct_rate)
        state.best_correct = max(state.best_correct, float(correct_rate))
        state.uncertainty = float(uncertainty)
        state.last_seen = int(round_index)
        state.exposures += 1

    def diagnostics(self, selected: Iterable[int], round_index: int) -> dict:
        selected = list(selected)
        scores = [self._priority_score(pid, round_index) for pid in selected]
        return {
            "schedule": self.mode,
            "selected_priority_count": self.last_priority_count,
            "selected_exploration_count": len(selected) - self.last_priority_count,
            "selected_priority_mean": float(np.mean(scores)) if scores else None,
            "selected_unseen": sum(self.states[pid].exposures == 0 for pid in selected),
            "pool_unseen": sum(state.exposures == 0 for state in self.states),
            "max_exposures": max((state.exposures for state in self.states), default=0),
            "min_exposures": min((state.exposures for state in self.states), default=0),
        }

    def state_dict(self) -> dict:
        """Return the complete deterministic scheduling state."""

        return {
            "schema_version": 1,
            "n_questions": self.n_questions,
            "mode": self.mode,
            "exploration": self.exploration,
            "rng_state": self.rng.bit_generator.state,
            "states": [
                {
                    "recent_correct": state.recent_correct,
                    "best_correct": state.best_correct,
                    "uncertainty": state.uncertainty,
                    "last_seen": state.last_seen,
                    "exposures": state.exposures,
                }
                for state in self.states
            ],
            "cycle": list(self._cycle),
            "last_priority_count": self.last_priority_count,
        }

    def load_state_dict(self, payload: dict) -> None:
        """Restore a state only when its immutable scheduler contract matches."""

        if payload.get("schema_version") != 1:
            raise ValueError("unsupported question-scheduler state schema")
        expected = {
            "n_questions": self.n_questions,
            "mode": self.mode,
            "exploration": self.exploration,
        }
        observed = {key: payload.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                f"question-scheduler state mismatch: expected {expected}, got {observed}"
            )
        states = payload.get("states")
        if not isinstance(states, list) or len(states) != self.n_questions:
            raise ValueError("question-scheduler state has the wrong question count")
        restored = []
        for state in states:
            restored.append(
                _QuestionState(
                    recent_correct=float(state["recent_correct"]),
                    best_correct=float(state["best_correct"]),
                    uncertainty=float(state["uncertainty"]),
                    last_seen=int(state["last_seen"]),
                    exposures=int(state["exposures"]),
                )
            )
        cycle = [int(pid) for pid in payload.get("cycle", [])]
        if any(pid < 0 or pid >= self.n_questions for pid in cycle):
            raise ValueError("question-scheduler cycle contains an invalid question id")
        self.rng.bit_generator.state = payload["rng_state"]
        self.states = restored
        self._cycle = cycle
        self.last_priority_count = int(payload.get("last_priority_count", 0))
