"""Immutable scientific configuration objects for the two largest trainers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Self, cast


class _ConfigFromCall:
    @classmethod
    def from_call(cls, values: Mapping[str, Any]) -> Self:
        """Capture named public-function arguments without duplicating defaults."""
        config_type = cast(Any, cls)
        return cast(
            Self,
            config_type(**{field.name: values[field.name] for field in fields(config_type)}),
        )


@dataclass(frozen=True)
class L2RRunConfig(_ConfigFromCall):
    rounds: int
    B: int
    G: int
    seed: int
    lr: float
    iters: int
    model_name: str
    micro: int
    reader_mode: str
    gold_in_buffer: bool
    l2r_buffer_semantics: str
    proposal_prompt: str
    proposal_mixture: str
    proposal_prior_fraction: float
    proposal_temperature: float
    trace_segmentation: str
    answer_event_mode: str
    answer_target_termination: str
    responsibility_score: str
    responsibility_temperature: float
    responsibility_projection: str
    responsibility_ess_floor: float
    responsibility_max_weight: float
    length_norm: bool
    mstep_objective: str
    archive_limit: int
    replay_limit: int
    adaptive_max_g: int
    adaptive_batch_g: int
    adaptive_min_correct: int
    reader_decode_filter: bool
    kl_coef: float
    policy_anchor_mode: str
    policy_anchor_target_ratio: float | None
    policy_anchor_beta_min: float
    policy_anchor_beta_max: float
    policy_anchor_ema: float
    policy_anchor_scope: str
    trust_kl_budget: float | None
    trust_safety_questions: int
    trust_safety_tolerance: float
    trust_boundary_failure_ceiling: float
    trust_max_backtracks: int
    trust_backtrack_shrink: float
    historical_replay_fraction: float
    lora_r: int
    lora_alpha: int
    lora_seed: int | None
    lora_trainable: str
    gradient_projection: str
    gradient_projection_rank: int
    gradient_basis_path: str | None
    gradient_projection_preserve_norm: bool
    buffer_replicates: int
    question_schedule: str
    schedule_exploration: float
    eval_every: int
    eval_rounds: tuple[int, ...] | list[int] | None
    diagnostics_level: str
    diagnostics_gradient_questions: int
    candidate_utility_questions: int
    candidate_utility_batch: int
    checkpoint_every: int
    exact_cache: bool
    state_checkpoint_every: int
    resume_fingerprint: str | None


@dataclass(frozen=True)
class ACAlg1RunConfig(_ConfigFromCall):
    algorithm_profile: str
    rounds: int
    L_batch: int
    U_batch: int
    G_label: int
    G_answer_only: int
    inner_steps: int
    seed: int
    lr: float
    model_name: str
    length_norm: bool
    buffer_limit: int
    labelled_frac: float
    buffer_strategy: str
    buffer_semantics: str
    buffer_lifecycle: str
    buffer_max_age: int
    proposal_prompt: str
    labelled_proposal_prompt: str | None
    answer_only_proposal_prompt: str | None
    proposal_mixture: str
    proposal_filter: str
    proposal_policy: str
    proposal_temperature: float
    proposal_allocation_mode: str
    proposal_initial_traces: int
    proposal_allocation_max_traces: int
    responsibility_score: str
    responsibility_posterior: str
    responsibility_temperature: float
    responsibility_ess_floor: float
    responsibility_abstention: str
    responsibility_rejection_threshold: float
    responsibility_null_log_evidence: float
    responsibility_null_prior: float
    responsibility_policy: str
    responsibility_answer_policy: str
    responsibility_refresh: str
    responsibility_verifier_rollouts: int
    responsibility_verifier_temperature: float
    responsibility_verifier_max_new_tokens: int
    responsibility_verifier_batch_size: int
    responsibility_verifier_smoothing_alpha: float
    verifier_calibration_path: str | None
    reuse_fresh_traces: int
    reuse_importance_min: float
    reuse_importance_max: float
    variational_estimator: str
    labelled_em_weight: float
    answer_only_em_weight: float
    policy_kl_coef: float | None
    supervised_weight: float
    policy_anchor_mode: str
    policy_anchor_target_ratio: float | None
    policy_anchor_beta_min: float
    policy_anchor_beta_max: float
    policy_anchor_ema: float
    policy_anchor_token_scope: str
    labelled_numeric_constraint: str
    numeric_penalty: float
    numeric_contradiction_penalty: float
    numeric_missing_penalty: float
    labelled_supervision: str
    compact_gold_weight: float
    digit_token_weight: float
    trace_representation: str
    latent_mstep_objective: str
    answer_event_mode: str
    answer_target_termination: str
    update_geometry: str
    step_acceptance: str
    rollback_tolerance: float
    rollback_max_backtracks: int
    rollback_shrink: float
    optimizer_state_scope: str
    question_sampling: str
    eval_every: int
    eval_rounds: tuple[int, ...] | list[int] | None
    diagnostics_level: str
    diagnostics_trace_tape: bool
    diagnostics_gradient_questions: int
    checkpoint_every: int
