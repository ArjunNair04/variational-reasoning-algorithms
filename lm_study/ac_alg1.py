"""Faithful Barber Learning-to-Reason Algorithm 1 implementation.

This file implements the 26 July 2026 version of Learning to Reason. The
default ``proposal_prompt="question"`` follows Algorithm 1; answer-conditioned
proposal modes are isolated experimental extensions.

The objective is

    F = B_sup + B'_unsup + B_unsup

where:
    B_sup       is supervised learning on labelled triples (a, h, q),
    B'_unsup    is the EM/buffer contribution on labelled questions,
    B_unsup     is the EM/buffer contribution on answer-only questions.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import math
import random
import re
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from typing import Any

import numpy as np
import torch

from answer_events import ANSWER_EVENT_MODES, parse_gsm8k_answer_event
from answer_targets import ANSWER_TARGET_TERMINATIONS, terminated_answer_ids
from ac_alg1_diagnostics import (
    binary_score_calibration,
    cuda_memory_diagnostics,
    optimizer_moment_diagnostics,
    parameter_delta_norm,
    posterior_churn,
    responsibility_gini,
    responsibility_margin,
    run_diagnostic_probe,
    spearman_correlation,
    tensor_list_cosine,
    tensor_list_norm,
    validate_diagnostic_level,
)
from ac_alg1_null_latent import (
    null_latent_responsibilities,
    threshold_rejection_responsibilities,
)
from ac_alg1_multi_verifier import (
    MULTI_VERIFIER_POSTERIORS,
    joint_validity_probability,
    multi_verifier_responsibilities,
)
from ac_alg1_bayesian_fusion import (
    fuse_validity,
    load_fusion_calibration,
)
from ac_alg1_age_one_reuse import (
    MAX_LOG_IMPORTANCE_RATIO,
    MIN_LOG_IMPORTANCE_RATIO,
    select_age_one_reuse,
)
from ac_alg1_uncertainty_allocation import allocate_uncertainty_budget
from ac_alg1_two_witness import two_witness_responsibilities
from ac_alg1_update_geometry import (
    STEP_ACCEPTANCE_MODES,
    UPDATE_GEOMETRIES,
    assign_trainable_gradients,
    combine_component_gradients,
    component_gradients_from_cumulative,
    fixed_surrogate_acceptance,
)
from ac_alg1_verifier import (
    score_trace_continuations,
    verifier_log_values,
    verifier_posterior_logits,
)
from common import (
    DEV,
    MODEL_NAME,
    QuestionSampler,
    encode_task_prompt,
    load_model,
    maybe_eval,
    sample_multi,
    seq_logprobs,
    task_pad_token_id,
    token_logps,
)
from trainer_config import ACAlg1RunConfig


PROPOSAL_PROMPTS = (
    "question",
    "derive_only",
    "answer_hint",
    "answer_derive",
    "answer_derive_concise",
    "answer_graph_derive",
    "tagged_zero_shot",
    "tagged_gold_rationale",
)
PROPOSAL_MIXTURES = (
    "single",
    "question_answer",
    "question_answer_graph",
)
PROPOSAL_FILTERS = ("all", "answer_correct", "answer_correct_numeric")
BUFFER_SEMANTICS = ("multiset_legacy", "unique_set")
BUFFER_LIFECYCLES = ("persistent", "fresh_round", "fixed_bank")
PROPOSAL_ALLOCATION_MODES = (
    "uniform",
    "posterior_uncertainty",
    "posterior_uncertainty_shifted",
)
RESPONSIBILITY_SCORES = ("joint", "token_mean", "rollout_value")
RESPONSIBILITY_POSTERIORS = (
    "softmax_entropy",
    "hard_delta_no_entropy",
    "two_witness",
    *MULTI_VERIFIER_POSTERIORS,
    "verifier_bayesian",
)
RESPONSIBILITY_ABSTENTION_MODES = ("none", "hard_threshold", "null_latent")
VARIATIONAL_ESTIMATORS = (
    "delta_joint",
    "uniform_mc",
    "prior_importance",
    "frozen_prior_importance",
    "answer_conditioned_importance",
    "persistent_answer_conditioned_importance",
    "persistent_prior_importance",
    "sampled_support_importance",
)
LATENT_MSTEP_OBJECTIVES = (
    "joint",
    "joint_token_mean",
    "answer",
    "rationale",
    "centered_trace_answer",
    "segment_responsibility_flow",
    "exact_signed_trace_answer",
)
ALGORITHM_PROFILES = (
    "legacy",
    "barber_source",
    "barber_fixed_kl_ablation",
    "barber_q5_control",
    "barber_q5_token_mean_followup",
    "l2r_common_factorial",
    "l2r_pis_rationale_kl_followup",
    "l2r_answer_conditioned_importance",
    "barber_stability_ablation",
    "barber_reader_ablation",
    "barber_refresh_ablation",
    "barber_importance_ablation",
    "barber_persistent_bridge",
    "barber_verifier",
    "q5_support_reallocation",
    "q5_revisit_concise",
    "l2r_reader_ess_closure",
    "l2r_credit_pilot",
    "l2r_abstention_pilot",
    "l2r_two_witness_pilot",
    "l2r_multi_verifier_pilot",
    "l2r_bayesian_fusion_pilot",
    "l2r_uncertainty_allocation_pilot",
    "l2r_age_one_reuse_pilot",
    "l2r_curated_buffer_pilot",
    "l2r_small_group_replay_pilot",
    "l2r_exact_signed_factorial",
)
ADAPTER_POLICY_MODES = ("current", "frozen_base")
RESPONSIBILITY_REFRESH_MODES = ("inner_step", "outer_round")
OPTIMIZER_STATE_SCOPES = ("persistent", "outer_round")
POLICY_ANCHOR_MODES = ("fixed", "grad_ratio")
POLICY_ANCHOR_TOKEN_SCOPES = ("objective", "reasoning")
LABELLED_NUMERIC_CONSTRAINTS = ("off", "hard", "soft", "graph_hard")
LABELLED_SUPERVISION_MODES = (
    "gold",
    "gold_answer",
    "gold_compact_mix",
    "gold_compact_set",
    "gold_graph_factorized",
)
TRACE_REPRESENTATIONS = ("reasoning", "calculation_graph")


@dataclass
class _ACAlg1RuntimeState:
    model: Any
    tok: Any
    opt: Any
    labelled_pool: list[int]
    answer_only_pool: list[int]
    labelled_sampler: QuestionSampler
    answer_only_sampler: QuestionSampler
    buffers: dict[int, list["TraceRow"]]
    records: list[dict]
    total_generated: int
    total_steps: int
    total_buffer_evictions: int
    total_set_duplicates: int
    total_filter_verifier_calls: int
    total_diagnostic_verifier_calls: int
    total_responsibility_verifier_calls: int
    total_responsibility_verifier_tokens: int
    policy_anchor_state: dict[str, float]
    training_diagnostic_state: dict[str, Any]
    initial_trainable_parameters: list[torch.Tensor] | None


@dataclass
class _ACAlg1RoundOutcome:
    round_started: float
    labelled_pids: list[int]
    answer_only_pids: list[int]
    labelled_sample_pid_row: list[int]
    labelled_sample_texts: list[str]
    labelled_sample_tokens: list[int]
    labelled_filter_stats: dict[str, Any]
    answer_only_sample_pid_row: list[int]
    answer_only_sample_texts: list[str]
    answer_only_sample_tokens: list[int]
    answer_only_filter_stats: dict[str, Any]
    labelled_weights: Any
    answer_only_weights: Any
    generation_elapsed: float
    e_step_elapsed: float
    m_step_elapsed: float
    diagnostic_probe_elapsed: float
    training_m_step_elapsed: float
    evaluation_elapsed: float
    gradient_geometry: Any
    update_geometry_diagnostics: dict[str, Any]
    inner_step_diagnostics: list[dict]
    posterior_refresh_diagnostics: list[dict]
    stats: dict[str, Any]
    round_rows_added: int
    round_evictions: int
    round_set_duplicates: int
    round_filter_attempted: int
    round_filter_accepted: int
    round_filter_rejected: int
    round_filter_verifier_calls: int
    responsibility_verifier_stats: dict[str, Any]
    test_acc: float
    sampling_intervention: dict[str, Any] | None = None
    sample_diagnostics: dict[str, Any] | None = None
    record: dict[str, Any] | None = None


@dataclass(frozen=True)
class NumericAudit:
    """Deterministic arithmetic audit for one labelled reasoning trace."""

    equation_mentions: int = 0
    parsed_equations: int = 0
    invalid_equations: int = 0
    gold_matches: int = 0
    gold_contradictions: int = 0
    gold_graph_available: bool = False
    gold_graph_nodes: int = 0
    gold_graph_edges: int = 0
    candidate_graph_nodes: int = 0
    candidate_graph_edges: int = 0
    graph_node_matches: int = 0
    graph_edge_matches: int = 0
    graph_node_coverage: float | None = None
    graph_edge_coverage: float | None = None
    graph_fully_covered: bool | None = None
    equations: tuple[str, ...] = ()
    invalid: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    missing_graph_nodes: tuple[str, ...] = ()
    missing_graph_edges: tuple[str, ...] = ()


@dataclass(frozen=True)
class LatentVerifierAudit:
    """Exact local-arithmetic and answer-closure observations for one trace."""

    raw_rationale: str = ""
    raw_question: str = ""
    target_answer: str = ""
    equation_mentions: int = 0
    parsed_equations: int = 0
    invalid_equations: int = 0
    candidate_graph_nodes: int = 0
    candidate_graph_edges: int = 0
    target_nodes: int = 0
    target_ancestor_nodes: int = 0
    binary_operations: int = 0
    question_numbers: int = 0
    input_leaf_numbers: int = 0
    grounded_input_leaf_numbers: int = 0
    arithmetic_observation: str = "missing"
    graph_observation: str = "missing"


@dataclass(frozen=True)
class StructuredTraceAudit:
    """Format audit for the explicit calculation-graph trace representation."""

    has_calculation_block: bool = False
    has_reasoning_block: bool = False
    calculation_precedes_reasoning: bool = False
    well_formed: bool = False
    calculation_characters: int = 0
    reasoning_characters: int = 0


@dataclass
class TraceRow:
    """One candidate reasoning trace stored in a per-question buffer.

    Attributes:
        ids: Token ids for the full sequence q ++ h_s ++ a.
        span: Boolean mask for h_s ++ a, used to score log p(h_s, a | q).
        ans: Boolean mask for a only, used when we need log p(a | h_s, q).
        pid: Integer question id in task.prompts.
        round_added: Outer training round when this trace entered the buffer.
        source: Human-readable source label, e.g. "gold" or "labelled_sample".
        is_gold: True only for the official gold reasoning trace.
        elite_score: Most recently computed detached responsibility, also used by hybrid pruning.
        joint_logprob: Most recently computed log p_theta(h_s, a* | q).
        trace_logprob: Most recently computed trace-prior factor log p(h_s | q).
        answer_logprob: Most recently computed reader factor log p(a* | h_s, q).
        responsibility_logit: Logit used by the latest E-step before temperature scaling.
        proposal_trace_logprob: Log density under the proposal distribution
            when an importance ratio is required.
        log_importance_correction: Current trace-prior log density minus the
            proposal trace-prior log density.
        reuse_admission_trace_logprob: Current trace-prior log density scored
            once when an age-one cached row passes its bounded reuse gate.  It
            is the authoritative numerator density for that row's subsequent
            persistent-prior E-step, so the accepted ratio is not changed by a
            second, differently shaped BF16 scoring batch.  None for fresh and
            non-reuse rows.
        trace_id: Stable identifier linking generation, buffer, and responsibility records.
        proposal_correct: Whether the sampled proposal decoded to the known answer.
        proposal_tokens: Number of tokens generated for the sampled proposal.
        numeric_audit: Arithmetic audit against labelled gold calculations.
            Normally None for answer-only rows. The registered
            ``answer_correct_numeric`` admission arm is the sole exception: it
            uses the reference calculation only to decide buffer admission and
            never exposes it to proposal generation or the M-step target.
        numeric_log_potential: Fixed log compatibility potential most recently
            added to this row's labelled E-step logit.
        responsibility_before_potential: Responsibility under the same score and
            temperature before applying the fixed numerical potential.
        responsibility_temperature_used: Actual one-sided adaptive temperature
            used for the latest E-step. It equals the configured temperature
            unless an ESS floor required additional smoothing.
        verifier_successes: Correct free-decoding continuations in the latest
            rollout-value E-step.
        verifier_trials: Number of continuations sampled in that E-step.
        verifier_raw_rate: Unsmoothed Monte Carlo estimate of p(a* | q,h).
        verifier_value: Symmetric-Beta-smoothed value used in log space.
        verifier_policy: ``current`` or ``frozen_base`` continuation policy.
        verifier_generated_tokens: Generated answer-continuation token count.
        verifier_outputs: Exact decoded continuations for failure analysis.
        verifier_correct: Per-continuation correctness indicators.
        trace_representation: ``reasoning`` or the explicit ``calculation_graph``
            serialization of z followed by h.
        structured_audit: Format audit for explicit calculation-graph traces.
        reasoning_token_count: Number of retained sampled-trace tokens strictly
            before the terminal answer marker.  The marker itself is excluded.
        segment_responsibility_deltas: Fixed round-level changes in posterior
            responsibility after each rationale segment for the experimental
            responsibility-flow M-step.
        responsibility_real_coverage: Total M-step mass assigned to real
            traces for the question after optional abstention.
        responsibility_null_mass: Complementary mass assigned to rejection or
            the explicit null latent state.
        responsibility_real_log_mean_evidence: Aggregate evidence statistic
            shared by the hard and smooth abstention rules.
        responsibility_insufficient_witness: Whether a registered two-witness
            update had fewer than two finite retained traces and therefore
            assigned zero question-level update mass.
        latent_verifier_audit: Deterministic local-arithmetic and answer-closure
            observations computed from the sampled rationale only.
        latent_verifier_mode: Registered posterior mode used for this trace.
        latent_verifier_applied_arithmetic: Arithmetic observation actually
            applied to this trace after an optional within-question shuffle.
        latent_verifier_applied_graph: Graph observation actually applied to
            this trace after an optional within-question shuffle.
        latent_verifier_validity_probability: Posterior probability that all
            active verifier properties hold for this trace.
        latent_verifier_source_index: Buffer index from which the applied
            observation pair came; equals this row's index in aligned arms.
        latent_verifier_global_null_mass: Question posterior mass on the
            existing answer-evidence null state.
        latent_verifier_invalid_mass: Question posterior mass on trace-specific
            verifier-invalid routes.
        sampled_support_prior_mass: Normalised retained-support prior mass
            r_s proportional to p_theta(h_s|x) / p_behaviour(h_s|x).  It is
            cached alongside the posterior q_s for the exact signed update.
        sampled_support_log_marginal: Self-normalised retained-support log
            answer marginal, logsumexp(d + log p(a|h)) - logsumexp(d).
        sampled_support_outer_initial: True only for the first E-step after
            sampling, where current and behaviour policies are identical by
            construction and d is set to exactly zero without a BF16 rescore.
        calculation_path_signature: Deterministic arithmetic-path cluster used
            only by the registered diversity-buffer intervention and diagnostics.
    """

    ids: torch.Tensor
    span: torch.Tensor
    ans: torch.Tensor
    pid: int
    round_added: int
    source: str
    is_gold: bool = False
    elite_score: float = float("-inf")
    joint_logprob: float = float("-inf")
    trace_logprob: float = float("-inf")
    answer_logprob: float = float("-inf")
    responsibility_logit: float = float("-inf")
    proposal_trace_logprob: float = float("-inf")
    log_importance_correction: float = 0.0
    trace_id: str | None = None
    proposal_correct: bool | None = None
    proposal_tokens: int | None = None
    numeric_audit: NumericAudit | None = None
    numeric_log_potential: float = 0.0
    responsibility_before_potential: float | None = None
    responsibility_temperature_used: float | None = None
    verifier_successes: int | None = None
    verifier_trials: int | None = None
    verifier_raw_rate: float | None = None
    verifier_value: float | None = None
    verifier_policy: str | None = None
    verifier_generated_tokens: int = 0
    verifier_outputs: tuple[str, ...] = ()
    verifier_correct: tuple[bool, ...] = ()
    trace_representation: str = "reasoning"
    structured_audit: StructuredTraceAudit | None = None
    reasoning_token_count: int | None = None
    segment_responsibility_deltas: tuple[float, ...] = ()
    responsibility_real_coverage: float = 1.0
    responsibility_null_mass: float = 0.0
    responsibility_real_log_mean_evidence: float | None = None
    responsibility_insufficient_witness: bool = False
    latent_verifier_audit: LatentVerifierAudit | None = None
    latent_verifier_mode: str | None = None
    latent_verifier_applied_arithmetic: str | None = None
    latent_verifier_applied_graph: str | None = None
    latent_verifier_validity_probability: float | None = None
    latent_verifier_source_index: int | None = None
    latent_verifier_global_null_mass: float | None = None
    latent_verifier_invalid_mass: float | None = None
    reuse_admission_trace_logprob: float | None = None
    sampled_support_prior_mass: float | None = None
    sampled_support_log_marginal: float | None = None
    sampled_support_outer_initial: bool | None = None
    calculation_path_signature: str = ""


def _adapter_policy_context(model, mode: str):
    """Select the current adapter policy or the frozen pretrained base."""

    if mode not in ADAPTER_POLICY_MODES:
        raise ValueError(f"unknown AC-ALG1 adapter policy mode {mode!r}")
    if mode == "current":
        return nullcontext()
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        raise ValueError(
            "frozen_base policy mode requires a model with disable_adapter()"
        )
    return disable_adapter()


def _responsibility_effective_sample_size(weights: torch.Tensor) -> float:
    """Return inverse-squared-mass ESS for a detached weight vector."""

    return 1.0 / float(torch.sum(weights.detach().float().square()).item())


def _posterior_weights(
    logits: torch.Tensor,
    posterior: str,
    *,
    temperature: float,
    ess_floor_fraction: float,
) -> tuple[torch.Tensor, float | None]:
    """Return the variational posterior over one finite trace support."""

    if posterior not in RESPONSIBILITY_POSTERIORS:
        raise ValueError(f"unknown AC-ALG1 responsibility_posterior {posterior!r}")
    finite = torch.isfinite(logits)
    valid = finite | torch.isneginf(logits)
    if not bool(valid.all()):
        raise ValueError(
            "responsibility logits must be finite or -inf hard exclusions"
        )
    if not bool(finite.any()):
        raise ValueError("responsibility posterior has no finite admissible logits")
    if posterior == "hard_delta_no_entropy":
        if ess_floor_fraction != 0.0:
            raise ValueError(
                "hard_delta_no_entropy requires responsibility_ess_floor=0"
            )
        weights = torch.zeros_like(logits)
        weights[int(torch.argmax(logits).item())] = 1.0
        return weights.detach(), None
    if posterior == "two_witness":
        if temperature != 1.0:
            raise ValueError("two_witness requires responsibility_temperature=1")
        if ess_floor_fraction != 0.0:
            raise ValueError("two_witness requires responsibility_ess_floor=0")
        return two_witness_responsibilities(logits).detach(), 1.0
    if posterior in MULTI_VERIFIER_POSTERIORS:
        raise ValueError(
            "multi-verifier posteriors require trace observations and must be "
            "computed through _buffer_weights_for_questions"
        )

    effective_temperature = _one_sided_ess_temperature(
        logits,
        base_temperature=temperature,
        ess_floor_fraction=ess_floor_fraction,
    )
    return (
        torch.softmax(logits / effective_temperature, dim=0).detach(),
        effective_temperature,
    )


def _one_sided_ess_temperature(
    logits: torch.Tensor,
    base_temperature: float,
    ess_floor_fraction: float,
) -> float:
    """Raise temperature only when needed to prevent posterior concentration.

    Unlike a two-sided ESS target, this rule leaves flat or already diffuse
    evidence unchanged. The floor is measured relative to the number of
    finite, admissible logits, so hard constraints are never softened.
    """

    if not math.isfinite(base_temperature) or base_temperature <= 0:
        raise ValueError("base temperature must be finite and positive")
    if (
        not math.isfinite(ess_floor_fraction)
        or not 0.0 <= ess_floor_fraction <= 1.0
    ):
        raise ValueError("ESS floor fraction must be finite and in [0, 1]")

    # ``-inf`` is the deliberate representation of a hard-excluded trace, but
    # NaN and ``+inf`` do not define a categorical posterior.  Validate before
    # the disabled-floor fast path: even with no adaptive smoothing, an
    # all-excluded support would otherwise become a vector of NaNs in softmax.
    if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()):
        raise FloatingPointError(
            "responsibility logits must be finite or -inf hard exclusions"
        )
    finite_count = int(torch.isfinite(logits).sum().item())
    if finite_count == 0:
        raise ValueError(
            "responsibility posterior has no finite admissible logits"
        )
    if ess_floor_fraction == 0.0 or finite_count == 1:
        return float(base_temperature)
    target = max(1.0, ess_floor_fraction * finite_count)

    def ess_at(temperature: float) -> float:
        weights = torch.softmax(logits / temperature, dim=0)
        return _responsibility_effective_sample_size(weights)

    if ess_at(base_temperature) >= target:
        return float(base_temperature)

    low = float(base_temperature)
    high = low
    for _ in range(32):
        high *= 2.0
        if ess_at(high) >= target:
            break
    else:
        return high

    for _ in range(40):
        middle = 0.5 * (low + high)
        if ess_at(middle) >= target:
            high = middle
        else:
            low = middle
    return high


def _answer_known_pids(task) -> list[int]:
    """Return question ids with a parsed gold answer.

    Args:
        task: GSM8K-style task with prompts and gold_answer.

    Returns:
        List of integer prompt ids that can supply answer-known pairs (a, q).
    """

    if not hasattr(task, "gold_answer"):
        return []

    return [
        i
        for i in range(len(task.prompts))
        if task.gold_answer[i] is not None
    ]


def _labelled_answer_only_pools(task, labelled_frac: float = 0.5) -> tuple[list[int], list[int]]:
    """Partition the task into Algorithm-1 L and U pools.

    GSM8K ships gold solutions for every prompt. To represent Barber's distinct
    labelled L={(a,h,q)} and answer-only U={(a,q)} sets in this single dataset,
    prompts with gold_solution are split deterministically according to
    labelled_frac. The rest enter U with their gold_solution deliberately ignored.
    """

    if not 0.0 <= labelled_frac <= 1.0:
        raise ValueError(f"labelled_frac must be in [0,1], got {labelled_frac}")

    answer_known = _answer_known_pids(task)
    if not hasattr(task, "gold_solution"):
        return [], answer_known

    labelled_candidates = [pid for pid in answer_known if task.gold_solution[pid]]
    n_labelled = min(len(labelled_candidates), max(0, int(len(labelled_candidates) * labelled_frac + 0.5)))
    if n_labelled == 0:
        labelled_set = set()
    elif n_labelled == len(labelled_candidates):
        labelled_set = set(labelled_candidates)
    else:
        step = len(labelled_candidates) / n_labelled
        labelled_set = {labelled_candidates[int(i * step)] for i in range(n_labelled)}

    labelled = [pid for pid in answer_known if pid in labelled_set]
    answer_only = [pid for pid in answer_known if pid not in labelled_set]
    return labelled, answer_only


def _sample_minibatch(pool: list[int], rng, batch_size: int) -> list[int]:
    """Uniformly sample prompt ids from a fixed pool."""

    if not pool or batch_size <= 0:
        return []

    replace = batch_size > len(pool)
    return [int(pid) for pid in rng.choice(pool, size=batch_size, replace=replace)]


def _labelled_pids(task, labelled_frac: float = 0.5) -> list[int]:
    """Return question ids in the labelled L={(a,h,q)} split."""

    labelled, _answer_only = _labelled_answer_only_pools(task, labelled_frac=labelled_frac)
    return labelled


def _sample_labelled_minibatch(task, rng, L_batch: int) -> list[int]:
    """Uniformly sample labelled question ids.

    Args:
        task: GSM8K-style task.
        rng: NumPy random generator.
        L_batch: Number of labelled questions to sample.

    Returns:
        List of sampled integer prompt ids. Empty if no labelled data is available.
    """

    return _sample_minibatch(_labelled_pids(task), rng, L_batch)


def _answer_only_pids(task, labelled_frac: float = 0.5) -> list[int]:
    """Return question ids in the answer-only U={(a,q)} split."""

    _labelled, answer_only = _labelled_answer_only_pools(task, labelled_frac=labelled_frac)
    return answer_only


def _sample_answer_only_minibatch(task, rng, U_batch: int) -> list[int]:
    """Uniformly sample question ids for the answer-only objective.

    Args:
        task: GSM8K-style task.
        rng: NumPy random generator.
        U_batch: Number of answer-only questions to sample.

    Returns:
        List of sampled integer prompt ids. Empty if no answer-only data is available.
    """

    return _sample_minibatch(_answer_only_pids(task), rng, U_batch)


def _build_supervised_batch(
    tok,
    task,
    pids: list[int],
    solutions: list[str] | None = None,
    answer_target_termination: str = "none",
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Tokenise labelled triples for the supervised objective B_sup.

    Args:
        tok: Tokenizer.
        task: GSM8K-style task with prompts and gold_solution.
        pids: Question ids for the labelled minibatch.
        solutions: Optional target solution aligned one-to-one with pids.
        answer_target_termination: Optional tokenizer-EOS target termination.

    Returns:
        None if pids is empty. Otherwise:
            ids: Padded token ids for q ++ gold_solution.
            span: Boolean mask selecting only gold_solution tokens.
    """

    if solutions is not None and len(solutions) != len(pids):
        raise ValueError("supervised solutions must align one-to-one with pids")

    rows, spans = [], []

    for row_index, pid in enumerate(pids):
        prompt_ids = encode_task_prompt(
            tok,
            task,
            int(pid),
            return_tensors="pt",
        ).input_ids[0].tolist()

        # gold_solution is the full GSM8K target string: gold_reasoning followed by
        # the "#### gold_answer" marker. For B_sup we score the whole continuation:
        # log p_theta(gold_reasoning, gold_answer | q).
        solution = (
            solutions[row_index]
            if solutions is not None
            else task.gold_solution[int(pid)]
        )
        solution_ids = terminated_answer_ids(
            tok,
            tok(" " + solution, add_special_tokens=False).input_ids,
            termination=answer_target_termination,
        )

        ids = torch.tensor(prompt_ids + solution_ids, dtype=torch.long)
        span = torch.zeros(len(ids), dtype=torch.bool)
        span[len(prompt_ids):] = True

        rows.append(ids)
        spans.append(span)

    if not rows:
        return None

    pad = torch.nn.utils.rnn.pad_sequence
    ids = pad(rows, batch_first=True, padding_value=task_pad_token_id(tok)).to(DEV)
    span = pad(spans, batch_first=True, padding_value=False).to(DEV)
    return ids, span


def _digit_weighted_span(
    tok,
    ids: torch.Tensor,
    span: torch.Tensor,
    digit_token_weight: float,
) -> torch.Tensor:
    """Upweight digit-bearing target tokens without changing total row weight."""

    if not math.isfinite(digit_token_weight) or digit_token_weight < 1.0:
        raise ValueError(
            "digit_token_weight must be finite and at least 1, "
            f"got {digit_token_weight}"
        )
    if digit_token_weight == 1.0:
        return span

    weighted = span.to(dtype=torch.float32).clone()
    ids_cpu = ids.detach().cpu()
    span_cpu = span.detach().cpu()
    for row_index in range(ids_cpu.shape[0]):
        for token_index in span_cpu[row_index].nonzero(as_tuple=False).flatten().tolist():
            piece = tok.decode([int(ids_cpu[row_index, token_index])])
            if any(character.isdigit() for character in piece):
                weighted[row_index, token_index] = digit_token_weight

    target_counts = span.sum(dim=1).to(weighted.dtype)
    weighted_mass = weighted.sum(dim=1).clamp_min(1.0)
    weighted *= (target_counts / weighted_mass).unsqueeze(1)
    return weighted


def _B_sup(
    model,
    tok,
    task,
    labelled_pids: list[int],
    labelled_supervision: str = "gold",
    compact_gold_weight: float = 0.5,
    digit_token_weight: float = 1.0,
    answer_target_termination: str = "none",
    grad: bool = True,
) -> torch.Tensor:
    """Supervised log-likelihood of gold solutions given questions.

    Args:
        model: Trainable language model.
        tok: Tokenizer.
        task: GSM8K-style task.
        labelled_pids: Labelled question ids.
        labelled_supervision: ``gold`` uses only the official solution;
            ``gold_answer`` teacher-forces only the canonical final answer;
            ``gold_compact_mix`` uses a fixed mixture with a compact target;
            ``gold_compact_set`` maximises set-valued evidence; and
            ``gold_graph_factorized`` supervises an explicit z, h, a sequence.
        compact_gold_weight: Mixture weight on the compact target.
        digit_token_weight: Relative weight on digit-bearing target tokens.
            Per-row weights are renormalised to preserve total objective scale.
        answer_target_termination: Optional tokenizer-EOS target termination.
        grad: Whether the returned score must retain an autograd graph.
    Returns:
        Scalar tensor containing B_sup. Returns zero if labelled_pids is empty.
    """

    if labelled_supervision not in LABELLED_SUPERVISION_MODES:
        raise ValueError(f"unknown AC-ALG1 labelled_supervision {labelled_supervision!r}")
    if not math.isfinite(compact_gold_weight) or not 0.0 <= compact_gold_weight <= 1.0:
        raise ValueError(
            "compact_gold_weight must be finite and in [0,1], "
            f"got {compact_gold_weight}"
        )
    if not math.isfinite(digit_token_weight) or digit_token_weight < 1.0:
        raise ValueError(
            "digit_token_weight must be finite and at least 1, "
            f"got {digit_token_weight}"
        )

    gold_solutions = [task.gold_solution[int(pid)] for pid in labelled_pids]
    if labelled_supervision == "gold_answer":
        gold_solutions = [
            f"#### {task.gold_answer[int(pid)]}"
            for pid in labelled_pids
        ]
    elif labelled_supervision == "gold_graph_factorized":
        gold_solutions = [
            _structured_gold_solution(solution, task.gold_answer[int(pid)])
            for pid, solution in zip(labelled_pids, gold_solutions)
        ]

    gold_batch = _build_supervised_batch(
        tok,
        task,
        labelled_pids,
        solutions=(
            gold_solutions
            if labelled_supervision in ("gold_answer", "gold_graph_factorized")
            else None
        ),
        answer_target_termination=answer_target_termination,
    )
    if gold_batch is None:
        return torch.zeros((), device=DEV)

    gold_ids, gold_span = gold_batch
    gold_score_span = _digit_weighted_span(
        tok,
        gold_ids,
        gold_span,
        digit_token_weight,
    )
    gold_logp = seq_logprobs(
        model, gold_ids, gold_score_span, grad=grad, length_norm=False
    )
    if labelled_supervision in ("gold", "gold_answer", "gold_graph_factorized"):
        return gold_logp.mean()

    compact_solutions = [
        _compact_gold_solution(task.gold_solution[int(pid)], task.gold_answer[int(pid)])
        for pid in labelled_pids
    ]
    compact_ids, compact_span = _build_supervised_batch(
        tok,
        task,
        labelled_pids,
        solutions=compact_solutions,
        answer_target_termination=answer_target_termination,
    )
    compact_score_span = _digit_weighted_span(
        tok,
        compact_ids,
        compact_span,
        digit_token_weight,
    )
    compact_logp = seq_logprobs(
        model, compact_ids, compact_score_span, grad=grad, length_norm=False
    )
    if labelled_supervision == "gold_compact_set":
        set_logp = torch.logaddexp(gold_logp, compact_logp)
        duplicate_target = torch.tensor(
            [gold == compact for gold, compact in zip(gold_solutions, compact_solutions)],
            dtype=torch.bool,
            device=set_logp.device,
        )
        return torch.where(duplicate_target, gold_logp, set_logp).mean()

    if compact_gold_weight == 0.0:
        return gold_logp.mean()
    return (
        (1.0 - compact_gold_weight) * gold_logp
        + compact_gold_weight * compact_logp
    ).mean()


def _gold_reasoning_from_solution(gold_solution: str) -> str:
    """Extract gold reasoning h from a full GSM8K gold solution.

    Args:
        gold_solution: Full GSM8K answer string, usually "reasoning ... #### answer".

    Returns:
        The text before the first "####" marker, stripped on the right.
    """

    return gold_solution.split("####", 1)[0].rstrip()


_GOLD_CALCULATION_RE = re.compile(r"<<([^<>]+)>>")
_UNSIGNED_NUMBER_RE = r"(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
_NUMBER_RE = rf"[-+]?\s*\$?\s*{_UNSIGNED_NUMBER_RE}"
_ARITHMETIC_TERM_RE = (
    rf"(?:\s*(?:\+|-|\*|/|×|÷)\s*{_NUMBER_RE}"
    rf"|\s*[xX]\s*\$?\s*{_UNSIGNED_NUMBER_RE})"
)
_ARITHMETIC_EXPRESSION_RE = rf"{_NUMBER_RE}(?:{_ARITHMETIC_TERM_RE})*"
_SINGLE_EQUALS_RE = r"(?<![=])=(?!=)"
_EXPLICIT_EQUATION_CHAIN_RE = re.compile(
    rf"(?P<chain>{_ARITHMETIC_EXPRESSION_RE}"
    rf"(?:\s*{_SINGLE_EQUALS_RE}\s*{_ARITHMETIC_EXPRESSION_RE})+)"
)
_SINGLE_EQUALS_SPLIT_RE = re.compile(_SINGLE_EQUALS_RE)


@dataclass(frozen=True)
class _ParsedEquation:
    text: str
    lhs_key: str
    lhs_value: Fraction
    rhs_value: Fraction
    lhs_node: ast.AST


@dataclass(frozen=True)
class _CalculationNode:
    """One canonical equation node in an ordered calculation DAG."""

    index: int
    text: str
    result: Fraction
    signature: tuple
    dependencies: tuple[int, ...]


@dataclass(frozen=True)
class _CalculationGraph:
    """Canonical calculation nodes plus dependency edges."""

    nodes: tuple[_CalculationNode, ...] = ()

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (parent, node.index)
            for node in self.nodes
            for parent in node.dependencies
        )


def _normalise_arithmetic(expression: str) -> str:
    """Normalise display variants before parsing a numeric expression."""

    expression = expression.replace(",", "").replace("$", "")
    expression = expression.replace("×", "*").replace("÷", "/")
    expression = expression.replace("−", "-").replace("–", "-")
    expression = re.sub(r"(?<=\d)\s*[xX]\s*(?=[-+]?\s*\d)", "*", expression)
    return expression.strip()


def _eval_arithmetic_node(node) -> Fraction:
    """Evaluate a whitelisted arithmetic AST exactly as a Fraction."""

    if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
        if isinstance(node.value, int):
            return Fraction(node.value)
        if isinstance(node.value, float):
            return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_arithmetic_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_arithmetic_node(node.left)
        right = _eval_arithmetic_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            return left / right
    raise ValueError("unsupported arithmetic expression")


def _parse_arithmetic_tree(expression: str) -> tuple[Fraction, str, ast.AST] | None:
    """Parse a basic arithmetic expression without executing arbitrary code."""

    normalised = _normalise_arithmetic(expression)
    try:
        tree = ast.parse(normalised, mode="eval")
        value = _eval_arithmetic_node(tree.body)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
        return None
    return (
        value,
        ast.dump(tree.body, annotate_fields=False, include_attributes=False),
        tree.body,
    )


def _parse_arithmetic(expression: str) -> tuple[Fraction, str] | None:
    """Return the exact value and stable AST key for a basic expression."""

    parsed = _parse_arithmetic_tree(expression)
    if parsed is None:
        return None
    value, key, _node = parsed
    return value, key


def _parse_equation(lhs: str, rhs: str, text: str) -> _ParsedEquation | None:
    lhs_parsed = _parse_arithmetic_tree(lhs)
    rhs_parsed = _parse_arithmetic(rhs)
    if lhs_parsed is None or rhs_parsed is None:
        return None
    lhs_value, lhs_key, lhs_node = lhs_parsed
    rhs_value, _rhs_key = rhs_parsed
    return _ParsedEquation(text.strip(), lhs_key, lhs_value, rhs_value, lhs_node)


def _gold_calculations(gold_solution: str) -> list[_ParsedEquation]:
    """Extract ordered GSM8K ``<<expression=result>>`` calculations."""

    calculations = []
    for annotation in _GOLD_CALCULATION_RE.findall(gold_solution or ""):
        if "=" not in annotation:
            continue
        lhs, rhs = annotation.rsplit("=", 1)
        parsed = _parse_equation(lhs, rhs, annotation)
        if parsed is not None:
            calculations.append(parsed)
    return calculations


def _explicit_calculations(reasoning: str) -> tuple[int, list[_ParsedEquation]]:
    """Extract explicit arithmetic equalities, including equality chains.

    Each adjacent link in ``a=b=c`` is audited independently. A match beginning
    with a sign is ignored when it is the trailing numeric fragment of a
    symbolic equation such as ``5x + 1 = 31``; standalone ``x``/``X`` remains a
    multiplication marker only when followed directly by a numeric literal.
    """

    reasoning = reasoning or ""
    mentions = 0
    parsed = []
    for match in _EXPLICIT_EQUATION_CHAIN_RE.finditer(reasoning):
        chain = match.group("chain")
        prefix = reasoning[:match.start()].rstrip()
        starts_with_sign = chain.lstrip().startswith(("+", "-"))
        follows_symbol = bool(
            re.search(r"(?:^|[^A-Za-z])[A-Za-z]$", prefix)
        )
        if starts_with_sign and follows_symbol:
            continue

        expressions = [
            expression.strip()
            for expression in _SINGLE_EQUALS_SPLIT_RE.split(chain)
        ]
        mentions += len(expressions) - 1
        for index, (lhs, rhs) in enumerate(zip(expressions, expressions[1:])):
            text = (
                chain.strip()
                if len(expressions) == 2
                else f"{lhs}={rhs}"
            )
            equation = _parse_equation(lhs, rhs, text)
            if equation is not None:
                parsed.append(equation)
    return mentions, parsed


def _fraction_signature(value: Fraction) -> tuple[str, int, int]:
    """Return a stable, hashable representation of an exact number."""

    return ("number", value.numerator, value.denominator)


def _literal_fraction(node: ast.AST) -> Fraction | None:
    """Evaluate a numeric literal, including unary signs, but not an operation."""

    if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
        if isinstance(node.value, int):
            return Fraction(node.value)
        if isinstance(node.value, float):
            return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal_fraction(node.operand)
        if value is not None:
            return value if isinstance(node.op, ast.UAdd) else -value
    return None


def _arithmetic_leaf_values(node: ast.AST) -> tuple[Fraction, ...]:
    """Return numeric leaves without double-counting signed literals."""

    literal = _literal_fraction(node)
    if literal is not None:
        return (literal,)
    if isinstance(node, ast.BinOp):
        return (
            *_arithmetic_leaf_values(node.left),
            *_arithmetic_leaf_values(node.right),
        )
    return ()


def _graph_expression_signature(
    node: ast.AST,
    prior_by_value: dict[Fraction, _CalculationNode],
    dependencies: set[int],
) -> tuple:
    """Canonicalise an expression while resolving earlier results as DAG edges."""

    literal = _literal_fraction(node)
    if literal is not None:
        parent = prior_by_value.get(literal)
        if parent is not None:
            dependencies.add(parent.index)
        return _fraction_signature(literal)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _graph_expression_signature(node.operand, prior_by_value, dependencies)
        return operand if isinstance(node.op, ast.UAdd) else ("negate", operand)

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Mult)):
            operator = "add" if isinstance(node.op, ast.Add) else "multiply"
            operator_type = type(node.op)
            operands = []

            def collect(current: ast.AST) -> None:
                if isinstance(current, ast.BinOp) and isinstance(current.op, operator_type):
                    collect(current.left)
                    collect(current.right)
                else:
                    operands.append(
                        _graph_expression_signature(
                            current,
                            prior_by_value,
                            dependencies,
                        )
                    )

            collect(node)
            return (operator, tuple(sorted(operands, key=repr)))

        operator = "subtract" if isinstance(node.op, ast.Sub) else "divide"
        return (
            operator,
            _graph_expression_signature(node.left, prior_by_value, dependencies),
            _graph_expression_signature(node.right, prior_by_value, dependencies),
        )

    raise ValueError("unsupported arithmetic graph node")


def _calculation_graph(equations: list[_ParsedEquation]) -> _CalculationGraph:
    """Build a canonical DAG from ordered, explicit arithmetic equations.

    A numeric literal is treated as a dependency when it equals the most recent
    earlier equation result. Independent equations may therefore be reordered,
    while dependent equations must expose the same calculation path.
    """

    nodes = []
    prior_by_value: dict[Fraction, _CalculationNode] = {}
    for index, equation in enumerate(equations):
        dependencies: set[int] = set()
        expression = _graph_expression_signature(
            equation.lhs_node,
            prior_by_value,
            dependencies,
        )
        signature = (
            "equation",
            expression,
            _fraction_signature(equation.rhs_value),
        )
        node = _CalculationNode(
            index=index,
            text=equation.text,
            result=equation.rhs_value,
            signature=signature,
            dependencies=tuple(sorted(dependencies)),
        )
        nodes.append(node)
        prior_by_value[equation.rhs_value] = node
    return _CalculationGraph(tuple(nodes))


_ARITHMETIC_SKELETON_OPERATION_RE = re.compile(
    r"[+*/\u00d7\u00f7-]|\b(?:x|add(?:ed|ing|s)?|plus|subtract(?:ed|ing|s)?|minus|"
    r"multiply|multiplied|times|divide|divided|half|double|twice|triple)\b",
    re.IGNORECASE,
)


def _calculation_path_signature(reasoning: str) -> str:
    """Return a bounded deterministic signature of a trace's arithmetic path.

    Explicit equations use the canonical calculation DAG already employed by
    the graph diagnostics. Traces without parseable equations fall back to an
    ordered number-and-operation skeleton; this avoids treating every
    unparseable rationale as a distinct path while keeping the intervention
    independent of the gold rationale and answer.
    """

    _mentions, equations = _explicit_calculations(reasoning or "")
    if equations:
        graph = _calculation_graph(equations)
        node_signatures = tuple(
            sorted((node.signature for node in graph.nodes), key=repr)
        )
        edge_signatures = tuple(
            sorted(
                [
                    (
                        graph.nodes[parent].signature,
                        graph.nodes[child].signature,
                    )
                    for parent, child in graph.edges
                ],
                key=repr,
            )
        )
        kind = "graph"
        payload = (node_signatures, edge_signatures)
    else:
        numbers = []
        for match in re.finditer(_NUMBER_RE, reasoning or ""):
            parsed = _parse_arithmetic(match.group(0))
            if parsed is not None:
                numbers.append(_fraction_signature(parsed[0]))
        operations = tuple(
            match.group(0).lower().replace(" ", "")
            for match in _ARITHMETIC_SKELETON_OPERATION_RE.finditer(reasoning or "")
        )
        kind = "skeleton"
        payload = (tuple(numbers), operations)
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    return f"{kind}:{digest}"


def _graph_overlap(
    gold: _CalculationGraph,
    candidate: _CalculationGraph,
) -> tuple[int, int, tuple[str, ...], tuple[str, ...]]:
    """Count covered gold nodes and edges as multisets of canonical signatures."""

    candidate_nodes = Counter(node.signature for node in candidate.nodes)
    node_matches = 0
    missing_nodes = []
    for node in gold.nodes:
        if candidate_nodes[node.signature] > 0:
            candidate_nodes[node.signature] -= 1
            node_matches += 1
        else:
            missing_nodes.append(node.text)

    def edge_signatures(graph: _CalculationGraph):
        return [
            (
                graph.nodes[parent].signature,
                graph.nodes[child].signature,
            )
            for parent, child in graph.edges
        ]

    candidate_edges = Counter(edge_signatures(candidate))
    edge_matches = 0
    missing_edges = []
    for parent, child in gold.edges:
        signature = (
            gold.nodes[parent].signature,
            gold.nodes[child].signature,
        )
        if candidate_edges[signature] > 0:
            candidate_edges[signature] -= 1
            edge_matches += 1
        else:
            missing_edges.append(
                f"{gold.nodes[parent].text} -> {gold.nodes[child].text}"
            )

    return node_matches, edge_matches, tuple(missing_nodes), tuple(missing_edges)


def _numeric_audit(reasoning: str, gold_solution: str) -> NumericAudit:
    """Audit explicit candidate equations against arithmetic and gold checkpoints.

    The local audit permits alternative valid calculations. The graph audit is
    stricter: it asks whether the candidate contains the complete canonical DAG
    encoded by GSM8K's ordered ``<<expression=result>>`` annotations. Independent
    steps may be reordered and commutative operands may swap, but every gold node
    and dependency edge must be present. A graph is unavailable when any annotated
    gold calculation cannot be parsed safely.
    """

    gold_annotations = [
        annotation
        for annotation in _GOLD_CALCULATION_RE.findall(gold_solution or "")
        if "=" in annotation
    ]
    gold_equations = _gold_calculations(gold_solution)
    gold_by_lhs = {equation.lhs_key: equation.rhs_value for equation in gold_equations}
    mentions, equations = _explicit_calculations(reasoning)
    invalid = [equation.text for equation in equations
               if equation.lhs_value != equation.rhs_value]
    matches = []
    contradictions = []
    for equation in equations:
        expected = gold_by_lhs.get(equation.lhs_key)
        if expected is None:
            continue
        if equation.rhs_value == expected:
            matches.append(equation.text)
        else:
            contradictions.append(equation.text)

    gold_graph = _calculation_graph(gold_equations)
    candidate_graph = _calculation_graph(equations)
    graph_available = bool(gold_annotations) and (
        len(gold_equations) == len(gold_annotations)
        and all(equation.lhs_value == equation.rhs_value for equation in gold_equations)
    )
    if graph_available:
        node_matches, edge_matches, missing_nodes, missing_edges = _graph_overlap(
            gold_graph,
            candidate_graph,
        )
        node_coverage = node_matches / len(gold_graph.nodes)
        edge_coverage = (
            edge_matches / len(gold_graph.edges)
            if gold_graph.edges
            else 1.0
        )
        graph_fully_covered = (
            node_matches == len(gold_graph.nodes)
            and edge_matches == len(gold_graph.edges)
        )
    else:
        node_matches = edge_matches = 0
        node_coverage = edge_coverage = graph_fully_covered = None
        missing_nodes = missing_edges = ()

    return NumericAudit(
        equation_mentions=mentions,
        parsed_equations=len(equations),
        invalid_equations=len(invalid),
        gold_matches=len(matches),
        gold_contradictions=len(contradictions),
        gold_graph_available=graph_available,
        gold_graph_nodes=len(gold_graph.nodes),
        gold_graph_edges=len(gold_graph.edges),
        candidate_graph_nodes=len(candidate_graph.nodes),
        candidate_graph_edges=len(candidate_graph.edges),
        graph_node_matches=node_matches,
        graph_edge_matches=edge_matches,
        graph_node_coverage=node_coverage,
        graph_edge_coverage=edge_coverage,
        graph_fully_covered=graph_fully_covered,
        equations=tuple(equation.text for equation in equations),
        invalid=tuple(invalid),
        contradictions=tuple(contradictions),
        missing_graph_nodes=missing_nodes,
        missing_graph_edges=missing_edges,
    )


def _latent_verifier_audit(
    reasoning: str,
    target_answer,
    question: str | None = None,
) -> LatentVerifierAudit:
    """Return two complementary deterministic observations for a rationale.

    The arithmetic source checks every safely parsed equality locally and does
    not inspect the target answer.  The graph source checks a different
    property: all parsed calculation nodes must lie on a dependency path to a
    node whose stated result is the known answer, at least one actual arithmetic
    operation must be present, and at least one non-derived numeric input must
    occur in the raw question.  It deliberately does not re-check local
    equality, so the two observations can disagree.  Unsupported or absent
    equations produce ``missing`` rather than a fabricated failure.
    """

    mentions, equations = _explicit_calculations(reasoning)
    raw_question = "" if question is None else str(question)
    target_answer_text = str(target_answer)
    if not equations:
        return LatentVerifierAudit(
            raw_rationale=str(reasoning),
            raw_question=raw_question,
            target_answer=target_answer_text,
            equation_mentions=mentions,
        )

    invalid_count = sum(
        equation.lhs_value != equation.rhs_value for equation in equations
    )
    arithmetic_observation = "pass" if invalid_count == 0 else "fail"
    binary_operations = sum(
        isinstance(node, ast.BinOp)
        for equation in equations
        for node in ast.walk(equation.lhs_node)
    )
    question_available = question is not None
    question_values = {
        parsed[0]
        for match in re.finditer(_NUMBER_RE, question or "")
        if (parsed := _parse_arithmetic(match.group(0))) is not None
    }
    prior_results: set[Fraction] = set()
    input_leaf_values: list[Fraction] = []
    for equation in equations:
        for value in _arithmetic_leaf_values(equation.lhs_node):
            if value not in prior_results:
                input_leaf_values.append(value)
        prior_results.add(equation.rhs_value)
    grounded_leaf_count = sum(
        value in question_values for value in input_leaf_values
    )

    try:
        target = Fraction(str(target_answer).replace(",", ""))
    except (TypeError, ValueError, ZeroDivisionError):
        graph_observation = "missing"
        graph = _calculation_graph(equations)
        target_nodes: list[int] = []
        ancestors: set[int] = set()
    else:
        graph = _calculation_graph(equations)
        target_nodes = [
            node.index for node in graph.nodes if node.result == target
        ]
        ancestors = set(target_nodes)
        frontier = list(target_nodes)
        while frontier:
            index = frontier.pop()
            for parent in graph.nodes[index].dependencies:
                if parent not in ancestors:
                    ancestors.add(parent)
                    frontier.append(parent)
        if not question_available:
            graph_observation = "missing"
        else:
            graph_observation = "pass" if (
                target_nodes
                and len(ancestors) == len(graph.nodes)
                and binary_operations > 0
                and grounded_leaf_count > 0
            ) else "fail"

    return LatentVerifierAudit(
        raw_rationale=str(reasoning),
        raw_question=raw_question,
        target_answer=target_answer_text,
        equation_mentions=mentions,
        parsed_equations=len(equations),
        invalid_equations=invalid_count,
        candidate_graph_nodes=len(graph.nodes),
        candidate_graph_edges=len(graph.edges),
        target_nodes=len(target_nodes),
        target_ancestor_nodes=len(ancestors),
        binary_operations=binary_operations,
        question_numbers=len(question_values),
        input_leaf_numbers=len(input_leaf_values),
        grounded_input_leaf_numbers=grounded_leaf_count,
        arithmetic_observation=arithmetic_observation,
        graph_observation=graph_observation,
    )


def _latent_verifier_source_indices(
    count: int,
    *,
    shuffle: bool,
    seed: int,
    pid: int,
) -> list[int]:
    """Return an aligned or nontrivially cyclically shifted trace mapping.

    The shuffled control moves the *paired* arithmetic/graph observation tuple,
    rather than independently permuting the two sources.  It therefore keeps
    their empirical joint distribution and the full multiset of validity
    probabilities fixed while breaking which sampled rationale receives each
    value.  The posterior update mass may still change because validity is then
    paired with different answer-evidence logits; that interaction is exactly
    what the control is designed to test.
    """

    if count < 0:
        raise ValueError("trace count cannot be negative")
    indices = list(range(count))
    if not shuffle or count <= 1:
        return indices
    generator = random.Random(
        int(seed) + int(pid) * 1_000_003 + count * 9_973
    )
    shift = generator.randrange(1, count)
    return [(index + shift) % count for index in indices]


def _compact_gold_solution(gold_solution: str, gold_answer) -> str:
    """Render GSM8K's annotated calculations as a concise supervised target."""

    calculations = _gold_calculations(gold_solution)
    if not calculations:
        return gold_solution
    steps = [f"{equation.text.replace('=', ' = ', 1)}." for equation in calculations]
    return "\n".join(steps + [f"#### {gold_answer}"])


_CALCULATION_BLOCK_RE = re.compile(
    r"<calculations>\s*(.*?)\s*</calculations>",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_BLOCK_RE = re.compile(
    r"<reasoning>\s*(.*?)\s*</reasoning>",
    re.IGNORECASE | re.DOTALL,
)


def _structured_trace_sections(text: str) -> tuple[str | None, str | None]:
    """Extract the explicit z and h fields from a structured completion."""

    calculations = _CALCULATION_BLOCK_RE.search(text or "")
    reasoning = _REASONING_BLOCK_RE.search(text or "")
    return (
        calculations.group(1).strip() if calculations else None,
        reasoning.group(1).strip() if reasoning else None,
    )


def _structured_trace_audit(text: str) -> StructuredTraceAudit:
    """Check whether a generated trace cleanly separates z before h."""

    calculation_match = _CALCULATION_BLOCK_RE.search(text or "")
    reasoning_match = _REASONING_BLOCK_RE.search(text or "")
    precedes = bool(
        calculation_match
        and reasoning_match
        and calculation_match.end() <= reasoning_match.start()
    )
    calculations = calculation_match.group(1).strip() if calculation_match else ""
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    return StructuredTraceAudit(
        has_calculation_block=calculation_match is not None,
        has_reasoning_block=reasoning_match is not None,
        calculation_precedes_reasoning=precedes,
        well_formed=bool(calculation_match and reasoning_match and precedes),
        calculation_characters=len(calculations),
        reasoning_characters=len(reasoning),
    )


def _structured_gold_reasoning(gold_solution: str) -> str:
    """Serialize observed z* before a verbal h* without duplicating annotations."""

    calculations = _gold_calculations(gold_solution)
    calculation_lines = [
        f"{index}. {equation.text.replace('=', ' = ', 1)}"
        for index, equation in enumerate(calculations, start=1)
    ]
    reasoning = _GOLD_CALCULATION_RE.sub("", _gold_reasoning_from_solution(gold_solution))
    reasoning = re.sub(r"[ \t]+", " ", reasoning).strip()
    return (
        "<calculations>\n"
        + "\n".join(calculation_lines)
        + "\n</calculations>\n<reasoning>\n"
        + reasoning
        + "\n</reasoning>"
    )


def _structured_gold_solution(gold_solution: str, gold_answer) -> str:
    """Return the supervised z* ++ h* ++ a* target for Option III."""

    return f"{_structured_gold_reasoning(gold_solution)}\n#### {gold_answer}"


def _trace_row_from_h_ids(
    tok,
    task,
    pid: int,
    h_ids: list[int],
    round_added: int,
    source: str,
    *,
    is_gold: bool = False,
    trace_id: str | None = None,
    proposal_correct: bool | None = None,
    proposal_tokens: int | None = None,
    numeric_reference_solution: str | None = None,
    trace_representation: str = "reasoning",
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
    reasoning_token_count: int | None = None,
) -> TraceRow | None:
    """Build a q ++ h_s ++ gold_answer row and its training/scoring masks."""

    gold_answer = task.gold_answer[pid]
    if gold_answer is None:
        return None
    if answer_event_mode not in ANSWER_EVENT_MODES:
        raise ValueError(f"unknown answer event mode {answer_event_mode!r}")

    prompt_ids = encode_task_prompt(
        tok,
        task,
        pid,
        return_tensors="pt",
    ).input_ids[0].tolist()
    if answer_event_mode == "strict_terminal_marker":
        decoded_h = tok.decode(list(h_ids))
        separator = "" if decoded_h.endswith((" ", "\n", "\t")) else " "
        answer_ids = tok(
            f"{separator}{gold_answer}",
            add_special_tokens=False,
        ).input_ids
        reconstructed = tok.decode(list(h_ids) + list(answer_ids))
        parser = getattr(task, "parse_answer_event", parse_gsm8k_answer_event)
        comparator = getattr(task, "answers_equivalent", None)
        event = parser(reconstructed, mode="strict_terminal_marker")
        answer_matches = (
            bool(comparator(event.answer, gold_answer))
            if comparator is not None
            else event.answer == gold_answer
        )
        if not event.strict_valid or not answer_matches:
            raise ValueError(
                "strict AC-ALG1 trace cannot reconstruct one terminal #### "
                "answer event"
            )
    else:
        answer_ids = tok(f"\n#### {gold_answer}", add_special_tokens=False).input_ids
    answer_ids = terminated_answer_ids(
        tok,
        answer_ids,
        termination=answer_target_termination,
    )
    full = torch.tensor(prompt_ids + list(h_ids) + answer_ids, dtype=torch.long)

    span = torch.zeros(len(full), dtype=torch.bool)
    span[len(prompt_ids):] = True

    ans = torch.zeros(len(full), dtype=torch.bool)
    ans[len(prompt_ids) + len(h_ids):] = True
    if trace_representation not in TRACE_REPRESENTATIONS:
        raise ValueError(f"unknown AC-ALG1 trace_representation {trace_representation!r}")
    reasoning_text = tok.decode(list(h_ids))
    structured_audit = None
    audit_text = reasoning_text
    if trace_representation == "calculation_graph":
        structured_audit = _structured_trace_audit(reasoning_text)
        calculations, _reasoning = _structured_trace_sections(reasoning_text)
        audit_text = calculations or ""
    numeric_audit = (
        _numeric_audit(audit_text, numeric_reference_solution)
        if numeric_reference_solution is not None
        else None
    )

    return TraceRow(
        ids=full,
        span=span,
        ans=ans,
        pid=pid,
        round_added=round_added,
        source=source,
        is_gold=is_gold,
        trace_id=trace_id,
        proposal_correct=proposal_correct,
        proposal_tokens=proposal_tokens,
        numeric_audit=numeric_audit,
        trace_representation=trace_representation,
        structured_audit=structured_audit,
        reasoning_token_count=(
            len(h_ids) if reasoning_token_count is None else reasoning_token_count
        ),
        calculation_path_signature=_calculation_path_signature(audit_text),
    )


def _gold_trace_row(
    tok,
    task,
    pid: int,
    round_added: int,
    trace_representation: str = "reasoning",
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
) -> TraceRow | None:
    """Build the official gold trace row for a question buffer.

    Args:
        tok: Tokenizer.
        task: GSM8K-style task.
        pid: Question id.
        round_added: Current outer training round.

    Returns:
        TraceRow for q ++ gold_reasoning ++ gold_answer, or None if the answer
        or solution is unavailable.
    """

    gold_solution = task.gold_solution[pid]
    if task.gold_answer[pid] is None or not gold_solution:
        return None

    if trace_representation not in TRACE_REPRESENTATIONS:
        raise ValueError(f"unknown AC-ALG1 trace_representation {trace_representation!r}")
    gold_reasoning = (
        _structured_gold_reasoning(gold_solution)
        if trace_representation == "calculation_graph"
        else _gold_reasoning_from_solution(gold_solution)
    )
    h_text = (
        " " + gold_reasoning + "\n####"
        if answer_event_mode == "strict_terminal_marker"
        else " " + gold_reasoning
    )
    h_ids = tok(h_text, add_special_tokens=False).input_ids
    return _trace_row_from_h_ids(
        tok,
        task,
        pid,
        h_ids,
        round_added,
        "gold",
        is_gold=True,
        trace_id=f"gold:{pid}",
        proposal_correct=True,
        numeric_reference_solution=gold_solution,
        trace_representation=trace_representation,
        answer_event_mode=answer_event_mode,
        answer_target_termination=answer_target_termination,
    )


def _enforce_buffer_limit(rows: list[TraceRow], buffer_limit: int, buffer_strategy: str = "fifo") -> int:
    """Apply a per-question buffer cap while preserving the official gold trace.

    Args:
        rows: Mutable per-question buffer.
        buffer_limit: Maximum rows to keep. Values <= 0 mean unbounded.
        buffer_strategy: "fifo" keeps the newest rows. "hybrid" keeps gold,
            recent sampled rows, and high-weight sampled rows.
            "calculation_diverse" first keeps the newest representative of as
            many arithmetic-path clusters as fit, then fills spare capacity by
            recency.

    Returns:
        Number of rows evicted. Mutates rows in place.
    """

    if buffer_limit <= 0 or len(rows) <= buffer_limit:
        return 0
    if buffer_strategy not in ("fifo", "hybrid", "calculation_diverse"):
        raise ValueError(f"unknown AC-ALG1 buffer_strategy {buffer_strategy!r}")

    gold_rows = [row for row in rows if row.is_gold]
    sampled_rows = [row for row in rows if not row.is_gold]
    sampled_keep = max(buffer_limit - len(gold_rows), 0)

    if buffer_strategy == "fifo":
        keep_sampled_ids = {id(row) for row in sampled_rows[-sampled_keep:]} if sampled_keep else set()
    elif buffer_strategy == "hybrid":
        recent_keep = (sampled_keep + 1) // 2
        elite_keep = sampled_keep - recent_keep
        recent_rows = sampled_rows[-recent_keep:] if recent_keep else []
        recent_ids = {id(row) for row in recent_rows}
        elite_pool = [row for row in sampled_rows if id(row) not in recent_ids]
        elite_rows = sorted(elite_pool, key=lambda row: (row.elite_score, row.round_added), reverse=True)[:elite_keep]
        keep_sampled_ids = recent_ids | {id(row) for row in elite_rows}
    else:
        representatives = []
        represented = set()
        for row in reversed(sampled_rows):
            signature = row.calculation_path_signature
            if signature not in represented:
                representatives.append(row)
                represented.add(signature)
        selected = representatives[:sampled_keep]
        selected_ids = {id(row) for row in selected}
        if len(selected) < sampled_keep:
            selected.extend(
                row
                for row in reversed(sampled_rows)
                if id(row) not in selected_ids
            )
        keep_sampled_ids = {id(row) for row in selected[:sampled_keep]}

    before = len(rows)
    rows[:] = [row for row in rows if row.is_gold or id(row) in keep_sampled_ids]
    return before - len(rows)


def _prepare_active_buffers(
    buffers: dict[int, list[TraceRow]],
    pids: list[int],
    buffer_lifecycle: str,
) -> tuple[list[int], int]:
    """Apply support-lifetime semantics and return question ids to sample."""

    if buffer_lifecycle not in BUFFER_LIFECYCLES:
        raise ValueError(f"unknown AC-ALG1 buffer_lifecycle {buffer_lifecycle!r}")
    active = [int(pid) for pid in pids]
    if buffer_lifecycle == "persistent":
        return active, 0
    if buffer_lifecycle == "fresh_round":
        cleared = sum(len(buffers[pid]) for pid in active)
        for pid in active:
            buffers[pid].clear()
        return active, cleared
    return (
        [
            pid
            for pid in active
            if not any(not row.is_gold for row in buffers[pid])
        ],
        0,
    )


def _add_gold_traces_to_buffer(
    tok,
    task,
    buffers: dict[int, list[TraceRow]],
    labelled_pids: list[int],
    round_added: int,
    buffer_limit: int,
    buffer_strategy: str,
    trace_representation: str = "reasoning",
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
) -> tuple[int, int]:
    """Insert gold reasoning traces into per-question buffers.

    Args:
        tok: Tokenizer.
        task: GSM8K-style task.
        buffers: Dictionary mapping question id to list of TraceRow objects.
        labelled_pids: Labelled question ids.
        round_added: Current outer training round.
        buffer_limit: Per-question buffer cap. Values <= 0 mean unbounded.
        buffer_strategy: Buffer pruning strategy.

    Returns:
        Pair ``(rows_added, rows_evicted)``. Mutates buffers in place.
    """

    rows_added = 0
    rows_evicted = 0
    for pid in labelled_pids:
        # Keep one copy of the official trace per question. Repeated copies would
        # artificially increase its buffer mass just because the question recurred.
        if any(row.is_gold for row in buffers[pid]):
            continue

        row = _gold_trace_row(
            tok,
            task,
            int(pid),
            round_added,
            trace_representation=trace_representation,
            answer_event_mode=answer_event_mode,
            answer_target_termination=answer_target_termination,
        )
        if row is not None:
            buffers[int(pid)].append(row)
            rows_added += 1
            rows_evicted += _enforce_buffer_limit(buffers[int(pid)], buffer_limit, buffer_strategy)
    return rows_added, rows_evicted


def _reason_cut(text: str) -> int:
    """Find where sampled reasoning should stop before appending the gold answer.

    Args:
        text: Decoded model completion.

    Returns:
        Character index where h_s ends.
    """

    cut = len(text)
    for marker in (
        "####",
        "</think>",
        "<answer>",
        "\nQuestion:",
        "\n\nQuestion:",
        "\nQ:",
        "\nUser:",
    ):
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return cut


def _prefix_ids_through_marker(
    tok,
    completion_ids: list[int],
    text: str,
    marker_end: int,
) -> tuple[list[int], bool]:
    """Keep sampled tokens through ``####`` only at an exact token boundary."""

    target = text[:marker_end]
    for stop in range(1, len(completion_ids) + 1):
        decoded = tok.decode(completion_ids[:stop])
        if len(decoded) < marker_end:
            continue
        if decoded[:marker_end] != target:
            return [], False
        if decoded[marker_end:].strip():
            return [], False
        return completion_ids[:stop], True
    return [], False


def _validate_strict_answer_event_tokenization(tok) -> None:
    """Fail before training when the tokenizer cannot preserve the marker event."""

    text = "Reasoning\n#### 42"
    completion_ids = tok(text, add_special_tokens=False).input_ids
    event = parse_gsm8k_answer_event(text, mode="strict_terminal_marker")
    if event.marker_end is None:
        raise AssertionError("strict answer-event fixture is malformed")
    h_ids, aligned = _prefix_ids_through_marker(
        tok,
        list(completion_ids),
        text,
        int(event.marker_end),
    )
    if not aligned:
        raise ValueError(
            "tokenizer cannot represent the strict AC-ALG1 #### boundary exactly"
        )
    decoded_h = tok.decode(h_ids)
    separator = "" if decoded_h.endswith((" ", "\n", "\t")) else " "
    answer_ids = tok(f"{separator}42", add_special_tokens=False).input_ids
    reconstructed = tok.decode(h_ids + list(answer_ids))
    reconstructed_event = parse_gsm8k_answer_event(
        reconstructed,
        mode="strict_terminal_marker",
    )
    if not reconstructed_event.strict_valid or reconstructed_event.answer != 42:
        raise ValueError(
            "tokenizer cannot reconstruct the strict AC-ALG1 #### answer event"
        )


def _tagged_reasoning_prompt(
    question: str,
    gold_solution: str | None = None,
) -> str:
    """Build the zero-shot User/Assistant proposal format."""

    instruction = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The Assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the "
        "answer. The reasoning process and answer are enclosed within "
        "<think>...</think> and <answer>...</answer> tags, respectively, i.e., "
        "<think> reasoning process here </think> "
        "<answer> answer here </answer>."
    )
    user_prompt = question.strip()
    if gold_solution:
        rationale = _gold_reasoning_from_solution(gold_solution).strip()
        user_prompt += (
            "\nReference rationale for this labelled question: "
            f"<rationale>{rationale}</rationale>"
        )
    return f"{instruction}\nUser: {user_prompt}\nAssistant: <think>"


def _build_proposal_prompt(
    prompt: str,
    gold_answer,
    proposal_prompt: str = "question",
    *,
    question: str | None = None,
    gold_solution: str | None = None,
    trace_representation: str = "reasoning",
) -> str:
    """Build the prompt used only to propose a candidate reasoning trace.

    The returned prompt may expose the known answer, but sampled reasoning is
    subsequently reanchored under the original question-only prompt before it
    enters the buffer. Thus this changes candidate discovery, not the sequence
    scored by the E-step or trained by the M-step.
    """

    if proposal_prompt not in PROPOSAL_PROMPTS:
        raise ValueError(f"unknown AC-ALG1 proposal_prompt {proposal_prompt!r}")
    if trace_representation not in TRACE_REPRESENTATIONS:
        raise ValueError(f"unknown AC-ALG1 trace_representation {trace_representation!r}")
    if trace_representation == "calculation_graph" and proposal_prompt.startswith("tagged_"):
        raise ValueError(
            "calculation_graph traces cannot be combined with tagged proposal prompts"
        )

    if proposal_prompt in ("tagged_zero_shot", "tagged_gold_rationale"):
        if question is None:
            raise ValueError(
                f"proposal_prompt={proposal_prompt!r} requires the raw question text"
            )
        if proposal_prompt == "tagged_gold_rationale" and not gold_solution:
            raise ValueError(
                "proposal_prompt='tagged_gold_rationale' requires a labelled gold solution"
            )
        return _tagged_reasoning_prompt(
            question,
            gold_solution if proposal_prompt == "tagged_gold_rationale" else None,
        )

    if proposal_prompt == "question":
        insertion = ""
    elif proposal_prompt == "derive_only":
        insertion = (
            "\nDerive a step-by-step solution to the question and finish with "
            "#### followed by the final numeric answer."
        )
    elif gold_answer is None:
        return prompt
    elif proposal_prompt == "answer_hint":
        insertion = f" (the final answer is {gold_answer})"
    elif proposal_prompt == "answer_derive":
        insertion = (
            f"\nThe correct final answer is {gold_answer}. "
            "Derive a step-by-step solution and finish with "
            f"#### {gold_answer}."
        )
    elif proposal_prompt == "answer_derive_concise":
        insertion = (
            f"\nThe correct final answer is {gold_answer}. "
            "Give the shortest complete derivation that justifies this answer. "
            "Include each necessary calculation exactly once and do not restate "
            "intermediate results. Finish with "
            f"#### {gold_answer}."
        )
    else:
        if not gold_solution:
            raise ValueError(
                "proposal_prompt='answer_graph_derive' requires a labelled gold solution"
            )
        annotations = [
            annotation.strip()
            for annotation in _GOLD_CALCULATION_RE.findall(gold_solution)
            if "=" in annotation
        ]
        calculations = _gold_calculations(gold_solution)
        if calculations and len(calculations) == len(annotations):
            graph = _calculation_graph(calculations)
            steps = "\n".join(
                f"  step {node.index + 1}: {node.text}"
                for node in graph.nodes
            )
            dependencies = (
                "\n".join(
                    f"  step {parent + 1} -> step {child + 1}"
                    for parent, child in graph.edges
                )
                if graph.edges
                else "  none"
            )
            scaffold = (
                "Verified calculation nodes:\n"
                f"{steps}\n"
                "Verified dependencies:\n"
                f"{dependencies}"
            )
        elif annotations:
            scaffold = "Verified calculations:\n" + "\n".join(
                f"  step {index}: {equation}"
                for index, equation in enumerate(annotations, start=1)
            )
        else:
            scaffold = "No explicit calculation nodes are available for this example."
        insertion = (
            f"\nThe correct final answer is {gold_answer}.\n"
            f"{scaffold}\n"
            "Write a concise step-by-step solution that follows the verified "
            "calculations where available and finish with "
            f"#### {gold_answer}."
        )

    if trace_representation == "calculation_graph":
        insertion = (
            "\nRepresent the calculation graph and its verbal explanation separately. "
            "Use exactly this output structure:\n"
            "<calculations>\n"
            "one explicit arithmetic equality per line\n"
            "</calculations>\n"
            "<reasoning>\n"
            "a concise explanation that uses those calculations\n"
            "</reasoning>\n"
            "#### final numeric answer"
            + insertion
        )

    if not insertion:
        return prompt

    answer_marker = "\nAnswer:"
    marker_index = prompt.rfind(answer_marker)
    if marker_index == -1:
        return prompt + insertion
    return prompt[:marker_index] + insertion + prompt[marker_index:]


def _proposal_components(
    proposal_prompt: str,
    proposal_mixture: str,
    source: str,
) -> tuple[tuple[str, float], ...]:
    """Return the fixed proposal recipe for one AC-ALG1 partition.

    Proposal mixtures change only the finite support presented to the E-step.
    The posterior on that support remains the normalized joint sequence score.
    For the graph arm, U' receives the same 50/50 question/answer mixture as
    the answer-guided arm because U' has no gold calculation graph.
    """

    if proposal_mixture not in PROPOSAL_MIXTURES:
        raise ValueError(f"unknown AC-ALG1 proposal_mixture {proposal_mixture!r}")
    if proposal_mixture == "single":
        return ((proposal_prompt, 1.0),)
    if proposal_prompt != "question":
        raise ValueError(
            "proposal mixtures require proposal_prompt='question'; "
            "the guided components are defined by proposal_mixture"
        )
    if proposal_mixture == "question_answer":
        return (("question", 0.5), ("answer_derive", 0.5))
    if source == "labelled_sample":
        return (
            ("question", 0.5),
            ("answer_derive", 0.25),
            ("answer_graph_derive", 0.25),
        )
    return (("question", 0.5), ("answer_derive", 0.5))


def _allocate_proposal_modes(
    traces_per_question: int,
    components: tuple[tuple[str, float], ...],
) -> list[str]:
    """Allocate an exact integer trace budget by deterministic largest remainder."""

    if traces_per_question < 0:
        raise ValueError("traces_per_question must be nonnegative")
    if not components:
        raise ValueError("proposal components cannot be empty")
    total = sum(weight for _mode, weight in components)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"proposal component weights must sum to one, got {total}")
    if any(weight < 0 or not math.isfinite(weight) for _mode, weight in components):
        raise ValueError("proposal component weights must be finite and nonnegative")

    raw = [traces_per_question * weight for _mode, weight in components]
    counts = [math.floor(value) for value in raw]
    remaining = traces_per_question - sum(counts)
    order = sorted(
        range(len(components)),
        key=lambda index: (raw[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return [
        mode
        for (mode, _weight), count in zip(components, counts)
        for _ in range(count)
    ]


def _proposal_source(
    source: str,
    proposal_prompt: str,
    proposal_filter: str,
    trace_representation: str = "reasoning",
) -> str:
    """Build a trace-source label that preserves proposal provenance."""

    labels = [source]
    if proposal_prompt != "question":
        labels.append(proposal_prompt)
    if proposal_filter != "all":
        labels.append(proposal_filter)
    if trace_representation != "reasoning":
        labels.append(trace_representation)
    return ":".join(labels)


def _sampled_trace_row(
    tok,
    task,
    pid: int,
    ids: torch.Tensor,
    comp_mask: torch.Tensor,
    text: str,
    round_added: int,
    source: str,
    trace_id: str | None = None,
    proposal_correct: bool | None = None,
    proposal_tokens: int | None = None,
    proposal_token_logprobs: torch.Tensor | None = None,
    numeric_reference_solution: str | None = None,
    trace_representation: str = "reasoning",
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
) -> TraceRow | None:
    """Convert one sampled completion into a trace-buffer row.

    Args:
        tok: Tokenizer.
        task: GSM8K-style task.
        pid: Question id for this completion.
        ids: Full generated token ids returned by sample_multi for this row.
        comp_mask: Boolean mask selecting completion tokens inside ids.
        text: Decoded completion text.
        round_added: Current outer training round.
        source: Source label for diagnostics.

    Returns:
        TraceRow for q ++ sampled_h ++ gold_answer, or None if no gold answer exists.
    """

    comp = ids[comp_mask].tolist()
    if comp and comp[-1] == tok.eos_token_id:
        comp = comp[:-1]

    reasoning_token_count = None
    if answer_event_mode == "strict_terminal_marker":
        parser = getattr(task, "parse_answer_event", parse_gsm8k_answer_event)
        event = parser(text, mode="strict_terminal_marker")
        if event.marker_end is None or not event.reasoning.strip():
            return None
        h_ids, aligned = _prefix_ids_through_marker(
            tok,
            comp,
            text,
            int(event.marker_end),
        )
        if not aligned:
            return None
        # The tokenizer may merge whitespace with the marker.  Preserve the
        # longest original-token prefix that ends before the marker rather
        # than re-tokenising text and silently changing the sampled trace.
        lo, hi = 0, len(h_ids)
        marker_start = int(event.marker_start)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(tok.decode(h_ids[:mid])) <= marker_start:
                lo = mid
            else:
                hi = mid - 1
        reasoning_token_count = lo
    else:
        cut = _reason_cut(text)
        if cut >= len(text):
            h_ids = comp
        else:
            # Cut in character space, then preserve the longest original-token
            # prefix whose decoded text lies entirely before that boundary.
            lo, hi = 0, len(comp)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(tok.decode(comp[:mid])) <= cut:
                    lo = mid
                else:
                    hi = mid - 1
            h_ids = comp[:lo]

    row = _trace_row_from_h_ids(
        tok,
        task,
        pid,
        h_ids,
        round_added,
        source,
        trace_id=trace_id,
        proposal_correct=proposal_correct,
        proposal_tokens=proposal_tokens,
        numeric_reference_solution=numeric_reference_solution,
        trace_representation=trace_representation,
        answer_event_mode=answer_event_mode,
        answer_target_termination=answer_target_termination,
        reasoning_token_count=reasoning_token_count,
    )
    if row is not None and proposal_token_logprobs is not None:
        if proposal_token_logprobs.numel() < len(h_ids):
            raise ValueError(
                "proposal token log probabilities do not cover the retained trace"
            )
        retained = proposal_token_logprobs[:len(h_ids)]
        if not torch.isfinite(retained).all():
            raise ValueError("proposal trace contains nonfinite token log probabilities")
        row.proposal_trace_logprob = float(retained.sum().item())
    return row


def _numeric_filter_accepts(row: TraceRow) -> bool:
    """Require one locally valid equation and no gold-calculation contradiction."""

    audit = row.numeric_audit
    return bool(
        audit is not None
        and audit.parsed_equations > 0
        and audit.invalid_equations == 0
        and audit.gold_contradictions == 0
    )


def _add_model_traces_to_buffer(
    model,
    tok,
    task,
    buffers: dict[int, list[TraceRow]],
    pids: list[int],
    traces_per_question: int,
    round_added: int,
    source: str,
    buffer_limit: int,
    buffer_strategy: str,
    buffer_semantics: str = "multiset_legacy",
    proposal_prompt: str = "question",
    proposal_mixture: str = "single",
    proposal_filter: str = "all",
    proposal_policy: str = "current",
    proposal_temperature: float = 1.0,
    trace_representation: str = "reasoning",
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
    collect_token_counts: bool = False,
    collect_proposal_outcomes: bool = False,
    record_proposal_density: bool = False,
    trace_index_offset: int = 0,
) -> tuple[list[int], list[str], list[int], int, int, dict]:
    """Sample model traces and append them to per-question buffers.

    Args:
        model: Language model used to sample h_s.
        tok: Tokenizer.
        task: GSM8K-style task.
        buffers: Dictionary mapping question id to list of TraceRow objects.
        pids: Question ids to sample from.
        traces_per_question: Number of sampled traces per question.
        round_added: Current outer training round.
        source: Source label, e.g. "labelled_sample" or "answer_only_sample".
        buffer_limit: Per-question buffer cap. Values <= 0 mean unbounded.
        buffer_strategy: Buffer pruning strategy.
        buffer_semantics: ``unique_set`` admits each token-identical trace at
            most once per question; ``multiset_legacy`` preserves historical
            replay behaviour.
        proposal_prompt: Candidate-generation prompt mode. Proposal-only
            instructions are stripped before the candidate enters the trace buffer.
        proposal_mixture: ``single`` uses proposal_prompt for every draw;
            ``question_answer`` uses a 50/50 question/answer-guided mixture;
            ``question_answer_graph`` additionally assigns one quarter of
            labelled draws to gold-calculation-guided proposals while retaining
            the same U' mixture as ``question_answer``.
        proposal_filter: Candidate-admission rule. ``answer_correct`` retains
            only completions whose decoded final answer matches the known answer.
        proposal_policy: ``current`` samples with the trained adapter;
            ``frozen_base`` temporarily disables it, yielding a fixed sampler.
        proposal_temperature: Positive sampling temperature applied to each
            rationale proposal.
        answer_event_mode: Replay-compatible or strict terminal-marker trace
            boundary semantics.
        collect_token_counts: Return exact generated-token counts for diagnostics.
        collect_proposal_outcomes: Attach verifier outcomes and stable trace ids
            for observational diagnostics.

    Returns:
        Tuple containing question ids, decoded completions, generated token
        counts, rows added, rows evicted, and proposal-filter counts.
    """

    if proposal_filter not in PROPOSAL_FILTERS:
        raise ValueError(f"unknown AC-ALG1 proposal_filter {proposal_filter!r}")
    if buffer_semantics not in BUFFER_SEMANTICS:
        raise ValueError(f"unknown AC-ALG1 buffer_semantics {buffer_semantics!r}")
    if proposal_policy not in ADAPTER_POLICY_MODES:
        raise ValueError(f"unknown AC-ALG1 proposal_policy {proposal_policy!r}")
    if not math.isfinite(proposal_temperature) or proposal_temperature <= 0:
        raise ValueError(
            "proposal_temperature must be finite and positive, got "
            f"{proposal_temperature}"
        )
    if trace_representation not in TRACE_REPRESENTATIONS:
        raise ValueError(f"unknown AC-ALG1 trace_representation {trace_representation!r}")
    if answer_event_mode not in ANSWER_EVENT_MODES:
        raise ValueError(f"unknown answer event mode {answer_event_mode!r}")
    components = _proposal_components(proposal_prompt, proposal_mixture, source)
    component_modes = {mode for mode, _weight in components}
    if (
        component_modes & {"tagged_gold_rationale", "answer_graph_derive"}
        and source != "labelled_sample"
    ):
        raise ValueError(
            "gold-rationale and graph-guided proposals are restricted to labelled L' samples"
        )
    if not pids or traces_per_question <= 0:
        return [], [], [], 0, 0, {
            "mode": proposal_filter,
            "policy": proposal_policy,
            "mixture": proposal_mixture,
            "attempted": 0,
            "accepted": 0,
            "rejected": 0,
            "verifier_calls": 0,
            "diagnostic_verifier_calls": 0,
            "trace_ids": [],
            "proposal_modes": [],
            "sources": [],
            "component_attempted": {},
            "component_accepted": {},
            "rewards": [],
            "correct": [],
            "admitted": [],
            "retained_after_insertion": [],
            "set_duplicate": [],
            "set_duplicate_count": 0,
            "boundary_valid": [],
            "boundary_rejected_count": 0,
        }

    modes_per_question = _allocate_proposal_modes(
        traces_per_question,
        components,
    )
    pid_row = []
    proposal_modes = []
    for pid in pids:
        pid_row.extend([int(pid)] * len(modes_per_question))
        proposal_modes.extend(modes_per_question)
    prompts = []
    row_sources = []
    task_builder = getattr(task, "build_proposal_prompt", None)
    for pid, mode in zip(pid_row, proposal_modes):
        if task_builder is not None and mode in {
            "question",
            "answer_derive",
            "answer_derive_first",
        }:
            prompts.append(task_builder(int(pid), mode))
        else:
            prompts.append(
                _build_proposal_prompt(
                    task.prompts[pid],
                    task.gold_answer[pid],
                    mode,
                    question=(task.questions[pid] if hasattr(task, "questions") else None),
                    gold_solution=(
                        task.gold_solution[pid]
                        if mode in ("tagged_gold_rationale", "answer_graph_derive")
                        and hasattr(task, "gold_solution")
                        else None
                    ),
                    trace_representation=trace_representation,
                )
            )
        row_source = _proposal_source(
            source,
            mode,
            proposal_filter,
            trace_representation,
        )
        if proposal_policy != "current":
            row_source = f"{row_source}:{proposal_policy}"
        row_sources.append(row_source)

    with _adapter_policy_context(model, proposal_policy):
        sample_kwargs = {"max_new": getattr(task, "max_new", 40)}
        if proposal_temperature != 1.0:
            sample_kwargs["temperature"] = proposal_temperature
        if record_proposal_density:
            sample_kwargs["return_token_logprobs"] = True
        sampled = sample_multi(model, tok, prompts, **sample_kwargs)
    if record_proposal_density:
        ids, comp_mask, texts, proposal_token_logprobs = sampled
    else:
        ids, comp_mask, texts = sampled
        proposal_token_logprobs = None

    measured_token_counts = (
        [int(value) for value in comp_mask.sum(dim=1).detach().cpu().tolist()]
        if collect_token_counts or collect_proposal_outcomes
        else []
    )
    token_counts = measured_token_counts if collect_token_counts else []
    if int(trace_index_offset) < 0:
        raise ValueError("trace_index_offset must be nonnegative")
    trace_ids = [
        f"r{round_added}:{source}:{row_idx + int(trace_index_offset)}:p{pid}"
        for row_idx, pid in enumerate(pid_row)
    ]
    rewards = None
    if (
        proposal_filter in {"answer_correct", "answer_correct_numeric"}
        or collect_proposal_outcomes
    ):
        rewards = [float(value) for value in task.reward(texts, pids=pid_row)]
    correct = (
        [bool(reward > 0.5) for reward in rewards]
        if rewards is not None
        else [None] * len(texts)
    )
    accepted = [True] * len(texts)
    filter_verifier_calls = 0
    if proposal_filter in {"answer_correct", "answer_correct_numeric"}:
        accepted = list(correct)
        filter_verifier_calls = len(texts)

    filter_stats = {
        "mode": proposal_filter,
        "policy": proposal_policy,
        "mixture": proposal_mixture,
        "attempted": len(texts),
        "accepted": int(sum(accepted)),
        "rejected": int(len(accepted) - sum(accepted)),
        "verifier_calls": filter_verifier_calls,
        "diagnostic_verifier_calls": (
            len(texts) if collect_proposal_outcomes and proposal_filter == "all" else 0
        ),
        "trace_ids": trace_ids,
        "proposal_modes": proposal_modes,
        "sources": row_sources,
        "component_attempted": dict(Counter(proposal_modes)),
        "component_accepted": dict(
            Counter(
                mode
                for mode, admitted in zip(proposal_modes, accepted)
                if admitted
            )
        ),
        "rewards": rewards if rewards is not None else [None] * len(texts),
        "correct": correct,
        "admitted": accepted,
    }
    rows_added = 0
    rows_evicted = 0
    set_duplicates = []
    boundary_valid = []
    for row_idx, pid in enumerate(pid_row):
        if not accepted[row_idx]:
            set_duplicates.append(False)
            boundary_valid.append(None)
            continue
        row = _sampled_trace_row(
            tok,
            task,
            pid,
            ids[row_idx].cpu(),
            comp_mask[row_idx].cpu(),
            texts[row_idx],
            round_added,
            row_sources[row_idx],
            trace_id=trace_ids[row_idx],
            proposal_correct=correct[row_idx],
            proposal_tokens=(
                measured_token_counts[row_idx] if measured_token_counts else None
            ),
            proposal_token_logprobs=(
                proposal_token_logprobs[row_idx][comp_mask[row_idx]].cpu()
                if proposal_token_logprobs is not None
                else None
            ),
            numeric_reference_solution=(
                task.gold_solution[pid]
                if (
                    source == "labelled_sample"
                    or proposal_filter == "answer_correct_numeric"
                )
                and hasattr(task, "gold_solution")
                else None
            ),
            trace_representation=trace_representation,
            answer_event_mode=answer_event_mode,
            answer_target_termination=answer_target_termination,
        )
        if row is not None:
            if proposal_filter == "answer_correct_numeric":
                if not _numeric_filter_accepts(row):
                    accepted[row_idx] = False
                    set_duplicates.append(False)
                    boundary_valid.append(True)
                    continue
            boundary_valid.append(True)
            row_key = tuple(int(token) for token in row.ids.tolist())
            duplicate = (
                buffer_semantics == "unique_set"
                and any(
                    tuple(int(token) for token in existing.ids.tolist()) == row_key
                    for existing in buffers[pid]
                )
            )
            set_duplicates.append(duplicate)
            if duplicate:
                continue
            buffers[pid].append(row)
            rows_added += 1
            if buffer_strategy != "calculation_diverse":
                rows_evicted += _enforce_buffer_limit(
                    buffers[pid], buffer_limit, buffer_strategy
                )
        else:
            if proposal_filter == "answer_correct_numeric":
                accepted[row_idx] = False
            set_duplicates.append(False)
            boundary_valid.append(False)

    if buffer_strategy == "calculation_diverse":
        for pid in set(pid_row):
            rows_evicted += _enforce_buffer_limit(
                buffers[pid], buffer_limit, buffer_strategy
            )

    filter_stats["accepted"] = int(sum(accepted))
    filter_stats["rejected"] = int(len(accepted) - sum(accepted))
    filter_stats["admitted"] = list(accepted)
    filter_stats["component_accepted"] = dict(
        Counter(
            mode
            for mode, admitted in zip(proposal_modes, accepted)
            if admitted
        )
    )

    retained_ids = {
        row.trace_id
        for pid in set(pid_row)
        for row in buffers[pid]
        if row.trace_id is not None
    }
    filter_stats["retained_after_insertion"] = [
        bool(admitted and trace_id in retained_ids)
        for admitted, trace_id in zip(accepted, trace_ids)
    ]
    filter_stats["set_duplicate"] = set_duplicates
    filter_stats["set_duplicate_count"] = int(sum(set_duplicates))
    filter_stats["boundary_valid"] = boundary_valid
    filter_stats["boundary_rejected_count"] = sum(
        valid is False for valid in boundary_valid
    )
    return pid_row, texts, token_counts, rows_added, rows_evicted, filter_stats


def _merge_trace_filter_stats(left: dict, right: dict) -> dict:
    """Merge same-round sampler summaries without dropping row provenance."""

    for key in ("mode", "policy", "mixture"):
        if left[key] != right[key]:
            raise ValueError(f"cannot merge sampler summaries with different {key}")
    merged = {key: left[key] for key in ("mode", "policy", "mixture")}
    for key in (
        "attempted",
        "accepted",
        "rejected",
        "verifier_calls",
        "diagnostic_verifier_calls",
        "set_duplicate_count",
        "boundary_rejected_count",
    ):
        merged[key] = int(left.get(key, 0)) + int(right.get(key, 0))
    for key in (
        "trace_ids",
        "proposal_modes",
        "sources",
        "rewards",
        "correct",
        "admitted",
        "retained_after_insertion",
        "set_duplicate",
        "boundary_valid",
    ):
        merged[key] = list(left.get(key, [])) + list(right.get(key, []))
    for key in ("component_attempted", "component_accepted"):
        merged[key] = dict(
            Counter(left.get(key, {})) + Counter(right.get(key, {}))
        )
    return merged


def _pad_trace_rows(tok, rows: list[TraceRow]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad trace-buffer rows into a model batch.

    Args:
        tok: Tokenizer.
        rows: TraceRow objects from one question buffer.

    Returns:
        ids: Padded q ++ h_s ++ a token ids.
        span: Padded mask for h_s ++ a.
        ans: Padded mask for a only.
    """

    pad = torch.nn.utils.rnn.pad_sequence
    padding_token_id = getattr(tok, "eos_token_id", None)
    if padding_token_id is None:
        padding_token_id = tok.pad_token_id
    ids = pad(
        [row.ids for row in rows],
        batch_first=True,
        padding_value=padding_token_id,
    ).to(DEV)
    span = pad([row.span for row in rows], batch_first=True, padding_value=False).to(DEV)
    ans = pad([row.ans for row in rows], batch_first=True, padding_value=False).to(DEV)
    return ids, span, ans


def _numeric_log_potential(
    row: TraceRow,
    labelled_numeric_constraint: str,
    numeric_penalty: float,
    numeric_contradiction_penalty: float = 0.0,
    numeric_missing_penalty: float = 0.0,
) -> float:
    """Return the fixed log compatibility potential for one labelled row."""

    if labelled_numeric_constraint == "off" or row.is_gold or row.numeric_audit is None:
        return 0.0
    errors = row.numeric_audit.invalid_equations
    if labelled_numeric_constraint == "graph_hard":
        if errors > 0 or row.numeric_audit.gold_contradictions > 0:
            return float("-inf")
        if (row.numeric_audit.gold_graph_available
                and row.numeric_audit.graph_fully_covered is not True):
            return float("-inf")
        return 0.0
    if errors == 0:
        if labelled_numeric_constraint == "hard":
            return 0.0
    if labelled_numeric_constraint == "hard":
        return float("-inf")
    missing_graph_items = (
        len(row.numeric_audit.missing_graph_nodes)
        + len(row.numeric_audit.missing_graph_edges)
        if row.numeric_audit.gold_graph_available
        else 0
    )
    return -(
        numeric_penalty * errors
        + numeric_contradiction_penalty * row.numeric_audit.gold_contradictions
        + numeric_missing_penalty * missing_graph_items
    )


def _barber_variational_logits(
    variational_estimator: str,
    trace_logprobs: torch.Tensor,
    answer_logprobs: torch.Tensor,
    *,
    proposal_trace_logprobs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return finite-sample logits for the declared Barber approximation.

    ``uniform_mc`` estimates an expectation under the proposal distribution,
    so every draw retains its empirical multiplicity. ``prior_importance``
    samples from the current prior and cancels that prior from the posterior
    importance ratio. ``frozen_prior_importance`` samples from a frozen prior
    and therefore retains the current-to-frozen trace-density correction.
    ``answer_conditioned_importance`` uses the exact sample-time density of
    the answer-conditioned proposal on a fresh empirical multiset.
    ``persistent_answer_conditioned_importance`` applies the same per-draw
    correction after the draw has entered a persistent FIFO multiset; each
    row retains the proposal density measured when that row was sampled.
    ``persistent_prior_importance`` is the corresponding registered one-round
    replay estimator for question-only prior draws.
    ``sampled_support_importance`` retains the immutable sample-time
    question-only behaviour density.  It is used only by the exact signed
    finite-support objective, which normalises both the answer-weighted
    numerator and the current-prior denominator on the same retained draws.
    """

    if variational_estimator not in VARIATIONAL_ESTIMATORS:
        raise ValueError(
            f"unknown AC-ALG1 variational_estimator {variational_estimator!r}"
        )
    if trace_logprobs.shape != answer_logprobs.shape:
        raise ValueError("trace and answer log probabilities must align")
    if variational_estimator == "uniform_mc":
        return torch.zeros_like(answer_logprobs)
    if variational_estimator == "prior_importance":
        return answer_logprobs
    if variational_estimator in {
        "frozen_prior_importance",
        "answer_conditioned_importance",
        "persistent_answer_conditioned_importance",
        "persistent_prior_importance",
        "sampled_support_importance",
    }:
        if proposal_trace_logprobs is None:
            raise ValueError(
                f"{variational_estimator} requires proposal trace log probabilities"
            )
        if proposal_trace_logprobs.shape != trace_logprobs.shape:
            raise ValueError("proposal trace log probabilities must align")
        return trace_logprobs + answer_logprobs - proposal_trace_logprobs
    return trace_logprobs + answer_logprobs


def sampled_support_importance_factors(
    current_trace_logprobs: torch.Tensor,
    behaviour_trace_logprobs: torch.Tensor,
    answer_logprobs: torch.Tensor,
    *,
    behaviour_matches_current: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact finite-support ``q``, ``r``, objective and displacement.

    For immutable behaviour draws ``h_s ~ pi_behaviour(.|x)``, define
    ``d_s = log pi_current(h_s|x) - log pi_behaviour(h_s|x)``.  The
    self-normalised answer posterior and retained-support prior are

    ``q = softmax(d + log p_current(a|x,h))`` and ``r = softmax(d)``.

    The corresponding sampled-support log marginal is
    ``logsumexp(d + log p(a|h)) - logsumexp(d)``.  Its exact gradient with
    respect to current rationale log density is ``q-r`` and with respect to
    the answer factor is ``q``.  Immediately after sampling, current and
    behaviour policies are the same mathematical object; callers set
    ``behaviour_matches_current`` so ``d`` is exactly zero rather than the
    difference of two separately-shaped BF16 scoring passes.
    """

    if (
        current_trace_logprobs.shape != behaviour_trace_logprobs.shape
        or current_trace_logprobs.shape != answer_logprobs.shape
    ):
        raise ValueError("sampled-support trace, behaviour and answer scores must align")
    if current_trace_logprobs.ndim != 1 or current_trace_logprobs.numel() < 1:
        raise ValueError("sampled-support factors require a nonempty rank-one support")
    for name, values in (
        ("current trace", current_trace_logprobs),
        ("behaviour trace", behaviour_trace_logprobs),
        ("answer", answer_logprobs),
    ):
        if not torch.isfinite(values).all():
            raise ValueError(f"sampled-support {name} scores must be finite")
    if bool((answer_logprobs > 1e-6).any()):
        raise ValueError("sampled-support answer log probabilities cannot be positive")

    displacement = (
        torch.zeros_like(current_trace_logprobs)
        if behaviour_matches_current
        else current_trace_logprobs - behaviour_trace_logprobs
    )
    numerator_logits = displacement + answer_logprobs
    q = torch.softmax(numerator_logits, dim=0)
    r = (
        torch.full_like(displacement, 1.0 / displacement.numel())
        if behaviour_matches_current
        else torch.softmax(displacement, dim=0)
    )
    log_marginal = (
        torch.logsumexp(numerator_logits, dim=0)
        - torch.logsumexp(displacement, dim=0)
    )
    if not torch.isfinite(log_marginal):
        raise ValueError("sampled-support log marginal must be finite")
    if float(log_marginal.detach()) > 1e-6:
        raise ValueError("sampled-support log marginal exceeded its probability bound")
    return q, r, log_marginal, displacement


def _sampled_support_e_step_logits(
    current_trace_logprobs: torch.Tensor,
    behaviour_trace_logprobs: torch.Tensor,
    answer_logprobs: torch.Tensor,
    *,
    outer_initial: bool,
) -> torch.Tensor:
    """Match the E-step logits to the exact d=0 outer convention."""

    if outer_initial:
        return answer_logprobs
    return current_trace_logprobs + answer_logprobs - behaviour_trace_logprobs


def _persistent_prior_trace_logprobs(
    model,
    ids: torch.Tensor,
    trace_span: torch.Tensor,
    rows: list[TraceRow],
    *,
    policy: str,
) -> torch.Tensor:
    """Preserve accepted age-one densities over the unchanged live-score batch.

    An age-one row is admitted using one current-policy trace score. Repeating
    that score inside the responsibility batch can move the ratio that the gate
    just accepted. Preserve the original all-row scoring batch exactly, then
    replace only selected cached rows with their admission-time values. Fresh
    rows therefore keep their original BF16 and padding behaviour.
    """

    if len(rows) != ids.shape[0] or ids.shape != trace_span.shape:
        raise ValueError("persistent-prior rows and padded tensors must align")
    authoritative = [row.reuse_admission_trace_logprob for row in rows]
    with _adapter_policy_context(model, policy):
        live_values = seq_logprobs(
            model,
            ids,
            trace_span,
            micro=16,
            length_norm=False,
        )
    if not any(value is not None for value in authoritative):
        return live_values

    for row, value in zip(rows, authoritative):
        if value is None:
            continue
        proposal = float(row.proposal_trace_logprob)
        current = float(value)
        if not (math.isfinite(proposal) and math.isfinite(current)):
            raise ValueError(
                "age-one authoritative proposal and current densities must be finite"
            )
        ratio = current - proposal
        if not MIN_LOG_IMPORTANCE_RATIO <= ratio <= MAX_LOG_IMPORTANCE_RATIO:
            raise ValueError(
                "age-one authoritative importance ratio escaped its admission bounds"
            )

    result = live_values.clone()
    for index, value in enumerate(authoritative):
        if value is not None:
            result[index] = float(value)
    return result


def _buffer_weights_for_questions(
    model,
    tok,
    buffers: dict[int, list[TraceRow]],
    pids: list[int],
    responsibility_score: str = "joint",
    responsibility_posterior: str = "softmax_entropy",
    responsibility_temperature: float = 1.0,
    responsibility_ess_floor: float = 0.0,
    responsibility_abstention: str = "none",
    responsibility_rejection_threshold: float = 0.0,
    responsibility_null_log_evidence: float = 0.0,
    responsibility_null_prior: float = 0.5,
    responsibility_policy: str = "current",
    responsibility_answer_policy: str = "current",
    labelled_numeric_constraint: str = "off",
    numeric_penalty: float = 2.0,
    numeric_contradiction_penalty: float = 0.0,
    numeric_missing_penalty: float = 0.0,
    variational_estimator: str = "delta_joint",
    record_joint_logprobs: bool = False,
    task=None,
    responsibility_verifier_rollouts: int = 0,
    responsibility_verifier_temperature: float = 1.0,
    responsibility_verifier_max_new_tokens: int = 64,
    responsibility_verifier_batch_size: int = 16,
    responsibility_verifier_smoothing_alpha: float = 0.5,
    responsibility_verifier_seed: int = 0,
    responsibility_verifier_diagnostics: dict[str, Any] | None = None,
    verifier_calibration_path: str | None = None,
    sampled_support_outer_initial: bool = False,
) -> dict[int, torch.Tensor]:
    """Compute detached posterior responsibilities inside each question buffer.

    Args:
        model: Language model used for the E-step scoring pass.
        tok: Tokenizer.
        buffers: Dictionary mapping question id to list of TraceRow objects.
        pids: Question ids whose buffers should be weighted.
        responsibility_score: ``joint`` uses log p(h_s, a* | q),
            ``token_mean`` divides that value by the scored token count, and
            ``rollout_value`` replaces the teacher-forced answer factor with
            a free-decoding estimate of p(a* | q,h_s).
        responsibility_posterior: ``softmax_entropy`` uses the ordinary
            one-witness posterior; ``hard_delta_no_entropy`` selects one trace;
            ``two_witness`` uses the exact coefficients of the registered
            two-independent-witness marginal.
        responsibility_temperature: Positive softmax temperature for E-step logits.
        responsibility_ess_floor: Minimum ESS as a fraction of the finite
            support. Zero disables the one-sided adaptive temperature.
        responsibility_abstention: Full update, hard question rejection, or a
            smooth posterior over real traces plus a frozen null state.
        responsibility_policy: ``current`` scores with the trained adapter;
            ``frozen_base`` disables it for a stationary E-step scorer.
        responsibility_answer_policy: Policy used only for the answer-reader
            factor p(a* | h_s, q). ``current`` preserves the joint-policy
            E-step; ``frozen_base`` combines the configured trace prior with a
            frozen pretrained answer reader.
        labelled_numeric_constraint: ``off``, ``hard``, ``soft``, or
            ``graph_hard`` arithmetic compatibility potential. ``graph_hard``
            requires full gold calculation-DAG coverage when the gold graph is
            safely parseable. Callers must leave this off for U'.
        numeric_penalty: Per-invalid-equation log penalty in ``soft`` mode.
        numeric_contradiction_penalty: Per-gold-contradiction log penalty.
        numeric_missing_penalty: Per-missing gold graph node or edge log penalty.
        variational_estimator: Delta-set, Monte Carlo, or importance estimator.
        record_joint_logprobs: Copy logits to TraceRow objects for diagnostics.
        task: Answer-aware task required by rollout-value responsibilities.
        responsibility_verifier_rollouts: Number of independent answer
            continuations sampled for each trace.
        responsibility_verifier_temperature: Sampling temperature for verifier
            continuations.
        responsibility_verifier_max_new_tokens: Maximum answer-continuation
            tokens per rollout.
        responsibility_verifier_batch_size: Rollout generation microbatch.
        responsibility_verifier_smoothing_alpha: Symmetric Beta prior used to
            keep the finite-sample log value defined when no rollout succeeds.
        responsibility_verifier_seed: Deterministic E-step rollout seed.
        responsibility_verifier_diagnostics: Optional mutable aggregate
            populated with calls, successes, generated tokens, and traces.
        sampled_support_outer_initial: Use the exact d=0 identity for the
            sampled-support estimator immediately after current-policy draws.

    Returns:
        Dictionary mapping each question id to a detached weight vector over B(q).
    """

    if responsibility_score not in RESPONSIBILITY_SCORES:
        raise ValueError(f"unknown AC-ALG1 responsibility_score {responsibility_score!r}")
    if variational_estimator not in VARIATIONAL_ESTIMATORS:
        raise ValueError(
            f"unknown AC-ALG1 variational_estimator {variational_estimator!r}"
        )
    if responsibility_posterior not in RESPONSIBILITY_POSTERIORS:
        raise ValueError(
            "unknown AC-ALG1 responsibility_posterior "
            f"{responsibility_posterior!r}"
        )
    if responsibility_abstention not in RESPONSIBILITY_ABSTENTION_MODES:
        raise ValueError(
            "unknown AC-ALG1 responsibility_abstention "
            f"{responsibility_abstention!r}"
        )
    if sampled_support_outer_initial and variational_estimator != "sampled_support_importance":
        raise ValueError(
            "sampled_support_outer_initial is isolated to sampled_support_importance"
        )
    if variational_estimator == "sampled_support_importance" and (
        responsibility_score != "joint"
        or responsibility_posterior != "softmax_entropy"
        or responsibility_temperature != 1.0
        or responsibility_ess_floor != 0.0
        or responsibility_abstention != "none"
        or responsibility_policy != "current"
        or responsibility_answer_policy != "current"
        or labelled_numeric_constraint != "off"
    ):
        raise ValueError(
            "sampled_support_importance requires joint current-policy scoring, "
            "ordinary unit-temperature softmax, no ESS projection, no abstention "
            "and no labelled potential"
        )
    multi_verifier_posterior = responsibility_posterior in {
        *MULTI_VERIFIER_POSTERIORS,
        "verifier_bayesian",
    }
    if responsibility_abstention != "none" and (
        (
            responsibility_posterior != "softmax_entropy"
            and not multi_verifier_posterior
        )
        or responsibility_ess_floor != 0.0
    ):
        raise ValueError(
            "responsibility abstention requires softmax_entropy or a registered "
            "multi-verifier posterior and responsibility_ess_floor=0"
        )
    if responsibility_posterior == "two_witness" and (
        variational_estimator != "prior_importance"
        or responsibility_score != "joint"
    ):
        raise ValueError(
            "two_witness requires joint-scored prior_importance evidence"
        )
    if multi_verifier_posterior and (
        variational_estimator != "prior_importance"
        or responsibility_score != "joint"
        or responsibility_abstention != "null_latent"
        or task is None
        or not hasattr(task, "questions")
        or not hasattr(task, "gold_answer")
    ):
        raise ValueError(
            "multi-verifier posteriors require task-aware, joint-scored "
            "prior_importance evidence with raw questions, gold answers and "
            "null_latent abstention"
        )
    if labelled_numeric_constraint not in LABELLED_NUMERIC_CONSTRAINTS:
        raise ValueError(
            "unknown AC-ALG1 labelled_numeric_constraint "
            f"{labelled_numeric_constraint!r}"
        )
    if not math.isfinite(responsibility_temperature) or responsibility_temperature <= 0:
        raise ValueError(
            "responsibility_temperature must be finite and positive, "
            f"got {responsibility_temperature}"
        )
    if (
        not math.isfinite(responsibility_ess_floor)
        or not 0.0 <= responsibility_ess_floor <= 1.0
    ):
        raise ValueError(
            "responsibility_ess_floor must be finite and in [0, 1], "
            f"got {responsibility_ess_floor}"
        )
    if responsibility_policy not in ADAPTER_POLICY_MODES:
        raise ValueError(
            f"unknown AC-ALG1 responsibility_policy {responsibility_policy!r}"
        )
    if responsibility_answer_policy not in ADAPTER_POLICY_MODES:
        raise ValueError(
            "unknown AC-ALG1 responsibility_answer_policy "
            f"{responsibility_answer_policy!r}"
        )
    if responsibility_score == "rollout_value":
        if task is None:
            raise ValueError("rollout_value responsibilities require task")
        if variational_estimator not in {"delta_joint", "prior_importance"}:
            raise ValueError(
                "rollout_value supports only delta_joint or prior_importance"
            )
        for name, value in (
            ("responsibility_verifier_rollouts", responsibility_verifier_rollouts),
            (
                "responsibility_verifier_max_new_tokens",
                responsibility_verifier_max_new_tokens,
            ),
            (
                "responsibility_verifier_batch_size",
                responsibility_verifier_batch_size,
            ),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, value in (
            (
                "responsibility_verifier_temperature",
                responsibility_verifier_temperature,
            ),
            (
                "responsibility_verifier_smoothing_alpha",
                responsibility_verifier_smoothing_alpha,
            ),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
    elif responsibility_verifier_rollouts != 0:
        raise ValueError(
            "responsibility_verifier_rollouts is active only when "
            "responsibility_score='rollout_value'"
        )
    if not math.isfinite(numeric_penalty) or numeric_penalty < 0:
        raise ValueError(f"numeric_penalty must be finite and nonnegative, got {numeric_penalty}")
    for name, value in (
        ("numeric_contradiction_penalty", numeric_contradiction_penalty),
        ("numeric_missing_penalty", numeric_missing_penalty),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative, got {value}")

    weights = {}

    with torch.no_grad():
        for pid in pids:
            rows = list(buffers[int(pid)])
            if not rows:
                continue

            ids, span, ans = _pad_trace_rows(tok, rows)
            trace_span = span & ~ans
            proposal_trace_logprobs = None
            sampled_support_q = None
            sampled_support_r = None
            sampled_support_log_marginal = None
            sampled_support_displacement = None
            if responsibility_score == "rollout_value":
                need_trace = (
                    record_joint_logprobs
                    or variational_estimator == "delta_joint"
                )
                if need_trace:
                    with _adapter_policy_context(model, responsibility_policy):
                        trace_logprobs = seq_logprobs(
                            model,
                            ids,
                            trace_span,
                            micro=16,
                            length_norm=False,
                        )
                else:
                    trace_logprobs = torch.zeros(
                        len(rows),
                        dtype=torch.float32,
                        device=ids.device,
                    )
                if record_joint_logprobs:
                    with _adapter_policy_context(
                        model,
                        responsibility_answer_policy,
                    ):
                        answer_logprobs = seq_logprobs(
                            model,
                            ids,
                            ans,
                            micro=16,
                            length_norm=False,
                        )
                else:
                    answer_logprobs = torch.zeros_like(trace_logprobs)
                joint_logprobs = trace_logprobs + answer_logprobs
                verifier_scores, verifier_stats = score_trace_continuations(
                    model,
                    tok,
                    task,
                    rows,
                    policy=responsibility_answer_policy,
                    repeats=responsibility_verifier_rollouts,
                    temperature=responsibility_verifier_temperature,
                    max_new_tokens=responsibility_verifier_max_new_tokens,
                    batch_size=responsibility_verifier_batch_size,
                    smoothing_alpha=responsibility_verifier_smoothing_alpha,
                    generation_seed=(
                        int(responsibility_verifier_seed) + int(pid) * 1009
                    ),
                )
                verifier_logs = verifier_log_values(
                    [score.successes for score in verifier_scores],
                    trials=responsibility_verifier_rollouts,
                    alpha=responsibility_verifier_smoothing_alpha,
                    device=trace_logprobs.device,
                    dtype=trace_logprobs.dtype,
                )
                base_logits = verifier_posterior_logits(
                    variational_estimator,
                    trace_logprobs,
                    verifier_logs,
                )
                for row, score in zip(rows, verifier_scores):
                    row.verifier_successes = score.successes
                    row.verifier_trials = score.trials
                    row.verifier_raw_rate = score.raw_rate
                    row.verifier_value = score.smoothed_value
                    row.verifier_policy = responsibility_answer_policy
                    row.verifier_generated_tokens = score.generated_tokens
                    row.verifier_outputs = score.outputs
                    row.verifier_correct = score.correct
                if responsibility_verifier_diagnostics is not None:
                    for key in (
                        "calls",
                        "generated_tokens",
                        "successes",
                        "traces",
                    ):
                        responsibility_verifier_diagnostics[key] = (
                            int(responsibility_verifier_diagnostics.get(key, 0))
                            + int(verifier_stats[key])
                        )
                    responsibility_verifier_diagnostics["policy"] = (
                        responsibility_answer_policy
                    )
            elif (
                variational_estimator == "delta_joint"
                and responsibility_answer_policy == responsibility_policy
            ):
                with _adapter_policy_context(model, responsibility_policy):
                    joint_logprobs = seq_logprobs(
                        model,
                        ids,
                        span,
                        micro=16,
                        length_norm=False,
                    )
                    if record_joint_logprobs:
                        answer_logprobs = seq_logprobs(
                            model,
                            ids,
                            ans,
                            micro=16,
                            length_norm=False,
                        )
                        trace_logprobs = joint_logprobs - answer_logprobs
                    else:
                        trace_logprobs = joint_logprobs
                        answer_logprobs = torch.zeros_like(joint_logprobs)
                base_logits = joint_logprobs
            else:
                need_trace = (
                    record_joint_logprobs
                    or variational_estimator
                    in {
                        "delta_joint",
                        "frozen_prior_importance",
                        "answer_conditioned_importance",
                        "persistent_answer_conditioned_importance",
                        "persistent_prior_importance",
                        "sampled_support_importance",
                    }
                )
                need_answer = (
                    record_joint_logprobs
                    or variational_estimator
                    in {
                        "delta_joint",
                        "prior_importance",
                        "frozen_prior_importance",
                        "answer_conditioned_importance",
                        "persistent_answer_conditioned_importance",
                        "persistent_prior_importance",
                        "sampled_support_importance",
                    }
                )
                if need_trace:
                    if (
                        variational_estimator == "sampled_support_importance"
                        and sampled_support_outer_initial
                    ):
                        proposal_values = [
                            row.proposal_trace_logprob for row in rows
                        ]
                        if not all(math.isfinite(value) for value in proposal_values):
                            raise ValueError(
                                "sampled_support_importance requires finite immutable "
                                "sample-time behaviour densities for every trace"
                            )
                        trace_logprobs = torch.tensor(
                            proposal_values,
                            dtype=torch.float32,
                            device=ids.device,
                        )
                    elif variational_estimator == "persistent_prior_importance":
                        trace_logprobs = _persistent_prior_trace_logprobs(
                            model,
                            ids,
                            trace_span,
                            rows,
                            policy=responsibility_policy,
                        )
                    else:
                        with _adapter_policy_context(model, responsibility_policy):
                            trace_logprobs = seq_logprobs(
                                model,
                                ids,
                                trace_span,
                                micro=16,
                                length_norm=False,
                            )
                else:
                    trace_logprobs = torch.zeros(
                        len(rows),
                        dtype=torch.float32,
                        device=ids.device,
                    )
                if need_answer:
                    with _adapter_policy_context(
                        model,
                        responsibility_answer_policy,
                    ):
                        answer_logprobs = seq_logprobs(
                            model,
                            ids,
                            ans,
                            micro=16,
                            length_norm=False,
                        )
                else:
                    answer_logprobs = torch.zeros_like(trace_logprobs)
                joint_logprobs = trace_logprobs + answer_logprobs
                if variational_estimator == "frozen_prior_importance":
                    with _adapter_policy_context(model, "frozen_base"):
                        proposal_trace_logprobs = seq_logprobs(
                            model,
                            ids,
                            trace_span,
                            micro=16,
                            length_norm=False,
                        )
                elif variational_estimator in {
                    "answer_conditioned_importance",
                    "persistent_answer_conditioned_importance",
                    "persistent_prior_importance",
                    "sampled_support_importance",
                }:
                    proposal_values = [
                        row.proposal_trace_logprob for row in rows
                    ]
                    if not all(math.isfinite(value) for value in proposal_values):
                        raise ValueError(
                            f"{variational_estimator} requires finite "
                            "sample-time proposal densities for every trace"
                        )
                    proposal_trace_logprobs = torch.tensor(
                        proposal_values,
                        dtype=trace_logprobs.dtype,
                        device=trace_logprobs.device,
                    )
                base_logits = _barber_variational_logits(
                    variational_estimator,
                    trace_logprobs,
                    answer_logprobs,
                    proposal_trace_logprobs=proposal_trace_logprobs,
                )
                if variational_estimator == "sampled_support_importance":
                    base_logits = _sampled_support_e_step_logits(
                        trace_logprobs,
                        proposal_trace_logprobs,
                        answer_logprobs,
                        outer_initial=sampled_support_outer_initial,
                    )
                    (
                        sampled_support_q,
                        sampled_support_r,
                        sampled_support_log_marginal,
                        sampled_support_displacement,
                    ) = sampled_support_importance_factors(
                        trace_logprobs,
                        proposal_trace_logprobs,
                        answer_logprobs,
                        behaviour_matches_current=sampled_support_outer_initial,
                    )
                    if not torch.allclose(
                        torch.softmax(base_logits, dim=0),
                        sampled_support_q,
                        atol=1e-7,
                        rtol=1e-7,
                    ):
                        raise RuntimeError(
                            "sampled-support posterior disagrees with its E-step logits"
                        )
            if responsibility_score == "token_mean":
                if variational_estimator != "delta_joint":
                    raise ValueError(
                        "token_mean is defined only for delta_joint responsibilities"
                    )
                scored_tokens = span.sum(dim=1).clamp_min(1).to(joint_logprobs.dtype)
                base_logits = joint_logprobs / scored_tokens

            log_potentials = torch.tensor(
                [
                    _numeric_log_potential(
                        row,
                        labelled_numeric_constraint,
                        numeric_penalty,
                        numeric_contradiction_penalty,
                        numeric_missing_penalty,
                    )
                    for row in rows
                ],
                dtype=base_logits.dtype,
                device=base_logits.device,
            )
            logits = base_logits + log_potentials

            latent_verifier_audits: list[LatentVerifierAudit] | None = None
            latent_verifier_source_indices: list[int] | None = None
            applied_arithmetic: list[str] | None = None
            applied_graph: list[str] | None = None
            validity_probabilities: list[float] | None = None
            verifier_global_null_mass = None
            verifier_invalid_mass = None
            insufficient_witness = False
            if multi_verifier_posterior:
                if labelled_numeric_constraint != "off":
                    raise ValueError(
                        "multi-verifier posteriors cannot be combined with the "
                        "legacy labelled numeric potential"
                    )
                target_answer = task.gold_answer[int(pid)]
                question = task.questions[int(pid)]
                latent_verifier_audits = []
                for row in rows:
                    rationale_ids = row.ids[row.span & ~row.ans]
                    rationale = tok.decode(
                        [int(token) for token in rationale_ids.tolist()]
                    )
                    latent_verifier_audits.append(
                        _latent_verifier_audit(
                            rationale,
                            target_answer,
                            question,
                        )
                    )
                latent_verifier_source_indices = _latent_verifier_source_indices(
                    len(rows),
                    shuffle=(
                        responsibility_posterior
                        == "verifier_joint_shuffled"
                    ),
                    seed=responsibility_verifier_seed,
                    pid=int(pid),
                )
                applied_arithmetic = [
                    latent_verifier_audits[index].arithmetic_observation
                    for index in latent_verifier_source_indices
                ]
                applied_graph = [
                    latent_verifier_audits[index].graph_observation
                    for index in latent_verifier_source_indices
                ]
                if responsibility_posterior == "verifier_bayesian":
                    if not verifier_calibration_path:
                        raise ValueError(
                            "verifier_bayesian requires a calibration artifact"
                        )
                    calibration = load_fusion_calibration(
                        verifier_calibration_path
                    )
                    validity_probabilities = [
                        fuse_validity(
                            0.5,
                            {"arithmetic": arithmetic, "graph": graph},
                            calibration,
                        ).validity_probability
                        for arithmetic, graph in zip(
                            applied_arithmetic,
                            applied_graph,
                        )
                    ]
                else:
                    validity_probabilities = [
                        joint_validity_probability(
                            arithmetic,
                            graph,
                            posterior=responsibility_posterior,
                        )
                        for arithmetic, graph in zip(
                            applied_arithmetic,
                            applied_graph,
                        )
                    ]
                verifier_posterior = multi_verifier_responsibilities(
                    logits,
                    validity_probabilities,
                    null_log_evidence=responsibility_null_log_evidence,
                    null_prior=responsibility_null_prior,
                )
                w = verifier_posterior.m_step_coefficients.to(
                    dtype=logits.dtype,
                    device=logits.device,
                )
                before = w.detach().clone()
                effective_temperature = 1.0
                real_coverage = verifier_posterior.real_coverage
                null_mass = verifier_posterior.null_mass
                real_log_mean_evidence = (
                    verifier_posterior.valid_log_mean_evidence
                )
                verifier_global_null_mass = (
                    verifier_posterior.global_null_mass
                )
                verifier_invalid_mass = (
                    verifier_posterior.verifier_invalid_mass
                )
            else:
                insufficient_witness = bool(
                    responsibility_posterior == "two_witness"
                    and int(torch.isfinite(logits).sum().item()) < 2
                )
                if insufficient_witness:
                    w = torch.zeros_like(logits).detach()
                    effective_temperature = 1.0
                else:
                    w, effective_temperature = _posterior_weights(
                        logits,
                        responsibility_posterior,
                        temperature=responsibility_temperature,
                        ess_floor_fraction=responsibility_ess_floor,
                    )
                real_coverage = 0.0 if insufficient_witness else 1.0
                null_mass = 1.0 if insufficient_witness else 0.0
                real_log_mean_evidence = None
                if responsibility_abstention == "hard_threshold":
                    abstention = threshold_rejection_responsibilities(
                        logits,
                        threshold=responsibility_rejection_threshold,
                        temperature=responsibility_temperature,
                    )
                    w = abstention.m_step_coefficients.to(
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                    real_coverage = abstention.real_coverage
                    null_mass = abstention.rejection_mass
                    real_log_mean_evidence = (
                        abstention.real_log_mean_evidence
                    )
                elif responsibility_abstention == "null_latent":
                    abstention = null_latent_responsibilities(
                        logits,
                        null_log_evidence=responsibility_null_log_evidence,
                        null_prior=responsibility_null_prior,
                        temperature=responsibility_temperature,
                    )
                    w = abstention.m_step_coefficients.to(
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                    real_coverage = abstention.real_coverage
                    null_mass = abstention.null_mass
                    real_log_mean_evidence = (
                        abstention.real_log_mean_evidence
                    )
                insufficient_before = bool(
                    responsibility_posterior == "two_witness"
                    and int(torch.isfinite(base_logits).sum().item()) < 2
                )
                if insufficient_before:
                    before = torch.zeros_like(base_logits).detach()
                else:
                    before, _ = _posterior_weights(
                        base_logits,
                        responsibility_posterior,
                        temperature=responsibility_temperature,
                        ess_floor_fraction=responsibility_ess_floor,
                    )
            if sampled_support_q is not None and not torch.allclose(
                w,
                sampled_support_q,
                atol=1e-7,
                rtol=1e-7,
            ):
                raise RuntimeError(
                    "sampled-support q changed after the registered posterior step"
                )
            joint_values = joint_logprobs.cpu().tolist() if record_joint_logprobs else None
            trace_values = (
                trace_logprobs.cpu().tolist()
                if record_joint_logprobs and trace_logprobs is not None
                else None
            )
            answer_values = (
                answer_logprobs.cpu().tolist()
                if record_joint_logprobs
                else None
            )
            proposal_trace_values = (
                proposal_trace_logprobs.cpu().tolist()
                if record_joint_logprobs and proposal_trace_logprobs is not None
                else None
            )
            logit_values = logits.cpu().tolist() if record_joint_logprobs else None
            for index, (row, score) in enumerate(zip(rows, w.cpu().tolist())):
                row.numeric_log_potential = float(log_potentials[index].item())
                row.responsibility_before_potential = float(before[index].item())
                row.responsibility_temperature_used = effective_temperature
                row.responsibility_real_coverage = float(real_coverage)
                row.responsibility_null_mass = float(null_mass)
                row.responsibility_real_log_mean_evidence = (
                    None
                    if real_log_mean_evidence is None
                    else float(real_log_mean_evidence)
                )
                row.responsibility_insufficient_witness = insufficient_witness
                row.latent_verifier_audit = (
                    None
                    if latent_verifier_audits is None
                    else latent_verifier_audits[index]
                )
                row.latent_verifier_mode = (
                    responsibility_posterior
                    if latent_verifier_audits is not None
                    else None
                )
                row.latent_verifier_applied_arithmetic = (
                    None
                    if applied_arithmetic is None
                    else applied_arithmetic[index]
                )
                row.latent_verifier_applied_graph = (
                    None if applied_graph is None else applied_graph[index]
                )
                row.latent_verifier_validity_probability = (
                    None
                    if validity_probabilities is None
                    else float(validity_probabilities[index])
                )
                row.latent_verifier_source_index = (
                    None
                    if latent_verifier_source_indices is None
                    else int(latent_verifier_source_indices[index])
                )
                row.latent_verifier_global_null_mass = (
                    None
                    if verifier_global_null_mass is None
                    else float(verifier_global_null_mass)
                )
                row.latent_verifier_invalid_mass = (
                    None
                    if verifier_invalid_mass is None
                    else float(verifier_invalid_mass)
                )
                row.sampled_support_prior_mass = (
                    None
                    if sampled_support_r is None
                    else float(sampled_support_r[index].item())
                )
                row.sampled_support_log_marginal = (
                    None
                    if sampled_support_log_marginal is None
                    else float(sampled_support_log_marginal.item())
                )
                row.sampled_support_outer_initial = (
                    None
                    if sampled_support_r is None
                    else bool(sampled_support_outer_initial)
                )
                if joint_values is not None:
                    row.joint_logprob = float(joint_values[index])
                    row.trace_logprob = float(trace_values[index])
                    row.answer_logprob = float(answer_values[index])
                    row.responsibility_logit = float(logit_values[index])
                    if proposal_trace_values is not None:
                        observed_proposal = float(proposal_trace_values[index])
                        if variational_estimator == "sampled_support_importance":
                            if not math.isclose(
                                row.proposal_trace_logprob,
                                observed_proposal,
                                rel_tol=1e-7,
                                abs_tol=1e-5,
                            ):
                                raise RuntimeError(
                                    "immutable sampled-support behaviour density changed"
                                )
                        else:
                            row.proposal_trace_logprob = observed_proposal
                        row.log_importance_correction = (
                            row.trace_logprob - row.proposal_trace_logprob
                        )
                row.elite_score = float(score)
            weights[int(pid)] = w

    return weights


def _latent_mstep_mask(
    span: torch.Tensor,
    ans: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Select the trainable factor of a latent complete-data row."""

    if mode not in LATENT_MSTEP_OBJECTIVES:
        raise ValueError(f"unknown latent_mstep_objective {mode!r}")
    if mode == "answer":
        return ans
    if mode == "rationale":
        return span & ~ans
    return span


def _latent_mstep_uses_token_mean(mode: str) -> bool:
    """Return whether each trace contributes a token-mean log likelihood."""

    if mode not in LATENT_MSTEP_OBJECTIVES:
        raise ValueError(f"unknown latent_mstep_objective {mode!r}")
    return mode == "joint_token_mean"


def _segment_masks(
    span: torch.Tensor,
    ans: torch.Tensor,
    rows: list[TraceRow],
    segment_count: int,
) -> list[torch.Tensor]:
    """Partition each retained rationale into contiguous token segments."""

    masks = [torch.zeros_like(span) for _ in range(segment_count)]
    for row_index, row in enumerate(rows):
        active = torch.nonzero(span[row_index], as_tuple=False).flatten()
        answer = torch.nonzero(ans[row_index], as_tuple=False).flatten()
        if active.numel() == 0 or answer.numel() == 0:
            raise ValueError("segment flow requires nonempty trace and answer spans")
        trace_start = int(active[0].item())
        answer_start = int(answer[0].item())
        trace_tokens = answer_start - trace_start
        reasoning_tokens = row.reasoning_token_count
        if reasoning_tokens is None:
            raise ValueError("segment flow requires a recorded marker boundary")
        if not 0 < reasoning_tokens <= trace_tokens:
            raise ValueError(
                "invalid segment-flow rationale width: "
                f"reasoning={reasoning_tokens}, trace={trace_tokens}"
            )
        for segment_index, mask in enumerate(masks):
            left = reasoning_tokens * segment_index // segment_count
            right = reasoning_tokens * (segment_index + 1) // segment_count
            if right > left:
                mask[row_index, trace_start + left:trace_start + right] = True
    return masks


def _latent_mstep_components(
    span: torch.Tensor,
    ans: torch.Tensor,
    rows: list[TraceRow],
    weights: torch.Tensor,
    mode: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return exact masks and detached row weights for one latent objective."""

    if mode not in LATENT_MSTEP_OBJECTIVES:
        raise ValueError(f"unknown latent_mstep_objective {mode!r}")
    if mode == "centered_trace_answer":
        uniform = torch.full_like(weights, 1.0 / len(rows))
        return [
            (span & ~ans, weights - uniform),
            (ans, weights),
        ]
    if mode == "exact_signed_trace_answer":
        prior_values = [row.sampled_support_prior_mass for row in rows]
        if not all(value is not None and math.isfinite(value) for value in prior_values):
            raise ValueError(
                "exact signed trace credit requires cached sampled-support prior mass"
            )
        prior = torch.tensor(
            [float(value) for value in prior_values],
            dtype=weights.dtype,
            device=weights.device,
        )
        if not torch.allclose(
            weights.sum(),
            torch.ones((), dtype=weights.dtype, device=weights.device),
            atol=1e-6,
            rtol=1e-6,
        ) or not torch.allclose(
            prior.sum(),
            torch.ones((), dtype=prior.dtype, device=prior.device),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("exact signed q and r must each sum to one")
        rationale_coefficients = weights - prior
        if abs(float(rationale_coefficients.sum().detach())) > 1e-6:
            raise ValueError("exact signed rationale coefficients must sum to zero")
        return [
            (span & ~ans, rationale_coefficients),
            (ans, weights),
        ]
    if mode == "segment_responsibility_flow":
        delta_widths = {len(row.segment_responsibility_deltas) for row in rows}
        if len(delta_widths) != 1 or not delta_widths or next(iter(delta_widths)) < 1:
            raise ValueError("segment-flow responsibilities were not cached")
        segment_count = next(iter(delta_widths))
        segment_masks = _segment_masks(span, ans, rows, segment_count)
        components: list[tuple[torch.Tensor, torch.Tensor]] = []
        for segment_index, mask in enumerate(segment_masks):
            deltas = torch.tensor(
                [row.segment_responsibility_deltas[segment_index] for row in rows],
                dtype=weights.dtype,
                device=weights.device,
            )
            components.append((mask, deltas / segment_count))
        components.append((ans, weights))
        return components
    return [(_latent_mstep_mask(span, ans, mode), weights)]


def _cache_segment_responsibility_flow(
    model,
    tok,
    buffers: dict[int, list[TraceRow]],
    pids: list[int],
    weights: dict[int, torch.Tensor],
    *,
    segment_count: int = 4,
    answer_policy: str = "current",
) -> None:
    """Cache fixed posterior movement as progressively more rationale is read.

    For each question, q_0 is uniform.  q_k is the answer-reader posterior
    after the first k/K contiguous rationale segments, followed by the exact
    retained marker suffix and gold answer.  The final q_K must reproduce the
    already frozen prior-importance E-step, otherwise execution fails closed.
    """

    if segment_count < 1:
        raise ValueError("segment_count must be positive")
    with torch.no_grad(), _adapter_policy_context(model, answer_policy):
        for pid in pids:
            rows = list(buffers[int(pid)])
            final_weights = weights.get(int(pid))
            if not rows or final_weights is None:
                continue
            deltas_by_row = [[] for _ in rows]
            previous = torch.full(
                (len(rows),),
                1.0 / len(rows),
                dtype=torch.float32,
                device=DEV,
            )
            for segment_index in range(1, segment_count + 1):
                prefix_ids: list[torch.Tensor] = []
                prefix_answer_masks: list[torch.Tensor] = []
                for row in rows:
                    trace_positions = torch.nonzero(
                        row.span & ~row.ans, as_tuple=False
                    ).flatten()
                    answer_positions = torch.nonzero(row.ans, as_tuple=False).flatten()
                    if trace_positions.numel() == 0 or answer_positions.numel() == 0:
                        raise ValueError("segment flow requires trace and answer tokens")
                    trace_start = int(trace_positions[0].item())
                    answer_start = int(answer_positions[0].item())
                    reasoning_tokens = row.reasoning_token_count
                    if reasoning_tokens is None or reasoning_tokens < 1:
                        raise ValueError(
                            "segment flow requires a nonempty recorded rationale"
                        )
                    trace_ids = row.ids[trace_start:answer_start]
                    if reasoning_tokens > len(trace_ids):
                        raise ValueError("recorded rationale exceeds retained trace")
                    boundary = reasoning_tokens * segment_index // segment_count
                    marker_suffix = trace_ids[reasoning_tokens:]
                    answer_ids = row.ids[answer_start:]
                    sequence = torch.cat(
                        (
                            row.ids[:trace_start],
                            trace_ids[:boundary],
                            marker_suffix,
                            answer_ids,
                        )
                    )
                    answer_mask = torch.zeros(len(sequence), dtype=torch.bool)
                    answer_mask[len(sequence) - len(answer_ids):] = True
                    prefix_ids.append(sequence)
                    prefix_answer_masks.append(answer_mask)
                ids = torch.nn.utils.rnn.pad_sequence(
                    prefix_ids,
                    batch_first=True,
                    padding_value=tok.pad_token_id,
                ).to(DEV)
                answer_masks = torch.nn.utils.rnn.pad_sequence(
                    prefix_answer_masks,
                    batch_first=True,
                    padding_value=False,
                ).to(DEV)
                answer_logprobs = seq_logprobs(
                    model,
                    ids,
                    answer_masks,
                    micro=16,
                    length_norm=False,
                )
                current = torch.softmax(answer_logprobs, dim=0).detach()
                delta = current - previous
                for row_deltas, value in zip(deltas_by_row, delta.cpu().tolist()):
                    row_deltas.append(float(value))
                previous = current
            expected = final_weights.to(previous.device, dtype=previous.dtype)
            if not torch.allclose(previous, expected, atol=1e-5, rtol=1e-5):
                raise RuntimeError(
                    "segment-flow final posterior does not match the frozen "
                    "prior-importance E-step"
                )
            for row, row_deltas in zip(rows, deltas_by_row):
                row.segment_responsibility_deltas = tuple(row_deltas)


def _B_unsup_for_questions(
    model,
    tok,
    buffers: dict[int, list[TraceRow]],
    pids: list[int],
    weights: dict[int, torch.Tensor],
    grad: bool = True,
    latent_mstep_objective: str = "joint",
) -> torch.Tensor:
    """Weighted latent-trace objective over selected per-question buffers.

    Args:
        model: Trainable language model.
        tok: Tokenizer.
        buffers: Dictionary mapping question id to list of TraceRow objects.
        pids: Question ids selected for this objective term.
        weights: Detached posterior responsibilities for each selected B(q).
        grad: Whether the returned score must retain an autograd graph.

    Returns:
        Scalar tensor containing the weighted EM objective over the selected buffers.
    """

    terms = []

    for pid in pids:
        rows = list(buffers[int(pid)])
        w = weights.get(int(pid))

        if not rows or w is None:
            continue

        ids, span, ans = _pad_trace_rows(tok, rows)
        row_terms = []
        for target, component_weights in _latent_mstep_components(
            span,
            ans,
            rows,
            w.to(DEV),
            latent_mstep_objective,
        ):
            if not bool(target.any()):
                continue
            logp = seq_logprobs(
                model,
                ids,
                target,
                grad=grad,
                length_norm=_latent_mstep_uses_token_mean(
                    latent_mstep_objective
                ),
            )
            row_terms.append((component_weights * logp).sum())
        if row_terms:
            terms.append(torch.stack(row_terms).sum())

    if not terms:
        return torch.zeros((), device=DEV)

    return torch.stack(terms).mean()


def _reference_policy_row_kl(
    model,
    ids: torch.Tensor,
    span: torch.Tensor,
    micro: int = 4,
    grad: bool = True,
) -> torch.Tensor:
    """Return a per-row token-mean KL estimate to the frozen base policy.

    The non-negative ``k3`` estimator matches the GRPO implementation:
    ``exp(log p_ref - log p_theta) - (log p_ref - log p_theta) - 1``.
    It is evaluated only on the supplied target span, so policy anchoring does
    not introduce a new prompt or trace distribution.
    """

    if ids.shape[0] == 0:
        return torch.empty(0, device=DEV)

    rows = []
    for start in range(0, ids.shape[0], micro):
        chunk_ids = ids[start:start + micro]
        chunk_span = span[start:start + micro]
        with torch.no_grad(), model.disable_adapter():
            reference_logps = token_logps(model, chunk_ids, grad=False)
        current_logps = token_logps(model, chunk_ids, grad=grad)
        log_reference_ratio = (reference_logps - current_logps).clamp(-5, 5)
        token_kl = torch.exp(log_reference_ratio) - log_reference_ratio - 1
        target_mask = chunk_span[:, 1:].to(dtype=token_kl.dtype)
        rows.append(
            (token_kl * target_mask).sum(dim=1)
            / target_mask.sum(dim=1).clamp_min(1.0)
        )
    return torch.cat(rows)


def _supervised_reference_policy_kl(
    model,
    tok,
    task,
    labelled_pids: list[int],
    labelled_supervision: str = "gold",
    answer_target_termination: str = "none",
    micro: int = 4,
    grad: bool = True,
) -> torch.Tensor:
    """Measure the frozen-policy KL on the supervised M-step targets."""

    solutions = None
    if labelled_supervision == "gold_answer":
        solutions = [
            f"#### {task.gold_answer[int(pid)]}"
            for pid in labelled_pids
        ]
    elif labelled_supervision == "gold_graph_factorized":
        solutions = [
            _structured_gold_solution(
                task.gold_solution[int(pid)], task.gold_answer[int(pid)]
            )
            for pid in labelled_pids
        ]
    batch = _build_supervised_batch(
        tok,
        task,
        labelled_pids,
        solutions=solutions,
        answer_target_termination=answer_target_termination,
    )
    if batch is None:
        return torch.zeros((), device=DEV)
    ids, span = batch
    return _reference_policy_row_kl(
        model, ids, span, micro=micro, grad=grad
    ).mean()


def _backward_reference_policy_kl_for_questions(
    model,
    tok,
    buffers: dict[int, list[TraceRow]],
    pids: list[int],
    weights: dict[int, torch.Tensor],
    micro: int = 4,
    backward_scale: float = 0.0,
    latent_mstep_objective: str = "joint",
    token_scope: str = "objective",
) -> tuple[float, bool]:
    """Measure a responsibility-weighted buffer KL and optionally backpropagate it."""

    if token_scope not in POLICY_ANCHOR_TOKEN_SCOPES:
        raise ValueError(f"unknown policy anchor token scope {token_scope!r}")

    valid = [
        int(pid)
        for pid in pids
        if buffers[int(pid)] and weights.get(int(pid)) is not None
    ]
    if not valid:
        return 0.0, False

    raw_kl = 0.0
    took_grad = False
    denominator = len(valid)
    for pid in valid:
        rows = list(buffers[pid])
        row_weights = weights[pid].to(DEV)
        for start in range(0, len(rows), micro):
            chunk = rows[start:start + micro]
            ids, span, ans = _pad_trace_rows(tok, chunk)
            if token_scope == "reasoning":
                target = torch.zeros_like(span)
                for row_index, row in enumerate(chunk):
                    reasoning_tokens = row.reasoning_token_count
                    if reasoning_tokens is None:
                        raise ValueError(
                            "reasoning-only policy anchoring requires an exact "
                            "reasoning_token_count for every trace"
                        )
                    span_positions = torch.nonzero(
                        span[row_index], as_tuple=False
                    ).flatten()
                    if reasoning_tokens < 0 or reasoning_tokens > len(span_positions):
                        raise ValueError(
                            "invalid reasoning_token_count for policy anchoring: "
                            f"{reasoning_tokens} of {len(span_positions)}"
                        )
                    target[row_index, span_positions[:reasoning_tokens]] = True
            else:
                target = _latent_mstep_mask(
                    span,
                    ans,
                    latent_mstep_objective,
                )
            row_kl = _reference_policy_row_kl(
                model,
                ids,
                target,
                micro=micro,
                grad=backward_scale > 0,
            )
            contribution = (
                row_weights[start:start + len(chunk)] * row_kl
            ).sum() / denominator
            raw_kl += float(contribution.detach())
            if backward_scale > 0 and contribution.requires_grad:
                (backward_scale * contribution).backward()
                took_grad = True
    return raw_kl, took_grad


def _backward_B_unsup_for_questions(
    model,
    tok,
    buffers: dict[int, list[TraceRow]],
    pids: list[int],
    weights: dict[int, torch.Tensor],
    micro: int = 4,
    coefficient: float = 1.0,
    latent_mstep_objective: str = "joint",
) -> tuple[float, bool]:
    """Accumulate ascent gradients without retaining every buffer row's graph."""
    valid = [
        int(pid)
        for pid in pids
        if buffers[int(pid)] and weights.get(int(pid)) is not None
    ]
    if not valid:
        return 0.0, False

    objective = 0.0
    denominator = len(valid)
    took_grad = False
    for pid in valid:
        rows = list(buffers[pid])
        w = weights[pid].to(DEV)
        if not bool(torch.any(w != 0)):
            # A question-level rejection or an ineligible two-witness support
            # is a literal zero term. Avoid constructing a zero autograd path:
            # otherwise Adam can still apply stored momentum when an entire
            # minibatch has no nonzero latent objective.
            continue
        ids, span, ans = _pad_trace_rows(tok, rows)
        components = _latent_mstep_components(
            span,
            ans,
            rows,
            w,
            latent_mstep_objective,
        )
        for target, component_weights in components:
            for start in range(0, len(rows), micro):
                chunk_target = target[start:start + micro]
                if not bool(chunk_target.any()):
                    continue
                chunk_ids = ids[start:start + micro]
                logp = seq_logprobs(
                    model,
                    chunk_ids,
                    chunk_target,
                    grad=True,
                    micro=micro,
                    length_norm=_latent_mstep_uses_token_mean(
                        latent_mstep_objective
                    ),
                )
                contribution = (
                    coefficient
                    * (
                        component_weights[start:start + len(chunk_ids)]
                        * logp
                    ).sum()
                    / denominator
                )
                objective += float(contribution.detach())
                if contribution.requires_grad:
                    (-contribution).backward()
                    took_grad = True
    return objective, took_grad


def _total_objective(
    B_sup: torch.Tensor,
    B_prime_unsup: torch.Tensor,
    B_unsup: torch.Tensor,
) -> torch.Tensor:
    """Add the supervised and latent-trace objective contributions.

    Args:
        B_sup: Supervised gold-solution objective.
        B_prime_unsup: Weighted buffer objective on labelled questions, B'_unsup.
        B_unsup: Weighted buffer objective on answer-only questions.

    Returns:
        Scalar tensor F = B_sup + B'_unsup + B_unsup.
    """

    return B_sup + B_prime_unsup + B_unsup


def _gradient_ascent_step(opt, F: torch.Tensor) -> bool:
    """Take one optimizer step that ascends the objective F.

    Args:
        opt: PyTorch optimizer for the trainable model parameters.
        F: Scalar objective tensor to maximise.

    Returns:
        True if a gradient step was taken, False if F has no gradient path.
    """

    if not F.requires_grad:
        return False

    opt.zero_grad()
    (-F).backward()
    opt.step()
    return True


def _snapshot_trainable_gradients(model) -> list[torch.Tensor]:
    """Clone current trainable gradients for diagnostic decomposition."""

    snapshots = []
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                snapshots.append(torch.zeros_like(parameter, dtype=torch.float32))
            else:
                snapshots.append(parameter.grad.detach().float().clone())
    return snapshots


def _snapshot_native_trainable_gradients(
    model,
) -> list[torch.Tensor | None]:
    """Clone trainable gradients without changing their optimizer dtype."""

    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def _gradient_norm(gradients: list[torch.Tensor | None]) -> float:
    """Return a stable float32 L2 norm for a trainable-gradient snapshot."""

    squared = sum(
        float(torch.sum(gradient.detach().float().square()))
        for gradient in gradients
        if gradient is not None
    )
    if not math.isfinite(squared):
        raise FloatingPointError("non-finite policy-anchor gradient norm")
    return math.sqrt(max(squared, 0.0))


def _apply_gradient_ratio_policy_anchor(
    model,
    objective_gradients: list[torch.Tensor | None],
    state: dict[str, float],
    target_ratio: float,
    beta_min: float,
    beta_max: float,
    ema: float,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Scale the unit KL gradient to a target fraction of the objective gradient.

    ``parameter.grad`` must contain ``g_F + g_KL`` on entry, where ``g_F`` is
    the minimization gradient of ``-F`` captured in ``objective_gradients`` and
    ``g_KL`` is the unit-coefficient reference-policy gradient. The function
    replaces it with ``g_F + beta * g_KL`` using a detached, EMA-smoothed beta.
    """

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(parameters) != len(objective_gradients):
        raise ValueError("adaptive policy-anchor snapshots must contain the same parameters")

    raw_anchor_gradients: list[torch.Tensor | None] = []
    for parameter, objective in zip(parameters, objective_gradients):
        combined = parameter.grad
        if combined is None:
            raw_anchor_gradients.append(None)
        elif objective is None:
            raw_anchor_gradients.append(combined.detach().clone())
        else:
            raw_anchor_gradients.append(combined.detach() - objective)

    objective_norm = _gradient_norm(objective_gradients)
    raw_anchor_norm = _gradient_norm(raw_anchor_gradients)

    previous_objective = state.get("ema_objective_grad_norm")
    previous_anchor = state.get("ema_raw_anchor_grad_norm")
    ema_objective = (
        objective_norm
        if previous_objective is None
        else ema * previous_objective + (1.0 - ema) * objective_norm
    )
    ema_anchor = (
        raw_anchor_norm
        if previous_anchor is None
        else ema * previous_anchor + (1.0 - ema) * raw_anchor_norm
    )
    state["ema_objective_grad_norm"] = ema_objective
    state["ema_raw_anchor_grad_norm"] = ema_anchor

    if target_ratio == 0.0 or ema_objective <= epsilon or ema_anchor <= epsilon:
        beta_unclipped = 0.0
        beta = 0.0
    else:
        beta_unclipped = target_ratio * ema_objective / max(ema_anchor, epsilon)
        beta = min(max(beta_unclipped, beta_min), beta_max)

    with torch.no_grad():
        for parameter, objective, raw_anchor in zip(
            parameters, objective_gradients, raw_anchor_gradients
        ):
            if objective is None and (raw_anchor is None or beta == 0.0):
                parameter.grad = None
                continue
            if objective is None:
                optimized = beta * raw_anchor
            elif raw_anchor is None:
                optimized = objective
            else:
                optimized = objective + beta * raw_anchor
            if parameter.grad is None:
                parameter.grad = optimized.clone()
            else:
                parameter.grad.copy_(optimized)

    applied_anchor_norm = beta * raw_anchor_norm
    achieved_ratio = (
        applied_anchor_norm / objective_norm if objective_norm > epsilon else 0.0
    )
    return {
        "beta": beta,
        "beta_unclipped": beta_unclipped,
        "beta_clipped": float(beta != beta_unclipped),
        "objective_grad_norm": objective_norm,
        "raw_anchor_grad_norm": raw_anchor_norm,
        "applied_anchor_grad_norm": applied_anchor_norm,
        "achieved_ratio": achieved_ratio,
        "ema_objective_grad_norm": ema_objective,
        "ema_raw_anchor_grad_norm": ema_anchor,
    }


def _gradient_geometry_from_snapshots(
    supervised: list[torch.Tensor],
    after_labelled_em: list[torch.Tensor],
    total: list[torch.Tensor],
) -> dict:
    """Recover objective-component gradient norms and pairwise cosines."""

    if not (len(supervised) == len(after_labelled_em) == len(total)):
        raise ValueError("gradient snapshots must contain the same parameters")
    components = {
        "B_sup": supervised,
        "B_prime_unsup": [
            labelled - sup for labelled, sup in zip(after_labelled_em, supervised)
        ],
        "B_unsup": [
            combined - labelled for combined, labelled in zip(total, after_labelled_em)
        ],
        "total": total,
    }

    norms = {}
    for name, tensors in components.items():
        squared = sum(float(torch.sum(tensor * tensor)) for tensor in tensors)
        norms[name] = (
            math.sqrt(max(squared, 0.0)) if math.isfinite(squared) else None
        )

    cosines = {}
    for left, right in (
        ("B_sup", "B_prime_unsup"),
        ("B_sup", "B_unsup"),
        ("B_prime_unsup", "B_unsup"),
    ):
        if norms[left] is None or norms[right] is None:
            cosine = None
        else:
            denominator = norms[left] * norms[right]
            dot = sum(
                float(torch.sum(a * b))
                for a, b in zip(components[left], components[right])
            )
            cosine = (
                min(max(dot / denominator, -1.0), 1.0)
                if denominator > 0 and math.isfinite(dot)
                else None
            )
        cosines[f"{left}__{right}"] = cosine

    return {"norms": norms, "cosines": cosines}


def _add_policy_anchor_gradient_geometry(
    geometry: dict,
    objective_total: list[torch.Tensor],
    optimized_total: list[torch.Tensor],
) -> dict:
    """Add the KL-anchor norm and its cosine with the unanchored loss gradient."""

    if len(objective_total) != len(optimized_total):
        raise ValueError("policy-anchor gradient snapshots must contain the same parameters")
    anchor = [
        optimized - objective
        for optimized, objective in zip(optimized_total, objective_total)
    ]

    def norm(tensors):
        squared = sum(float(torch.sum(tensor * tensor)) for tensor in tensors)
        return math.sqrt(max(squared, 0.0)) if math.isfinite(squared) else None

    objective_norm = geometry["norms"].get("total")
    anchor_norm = norm(anchor)
    optimized_norm = norm(optimized_total)
    geometry["norms"]["policy_anchor"] = anchor_norm
    geometry["norms"]["optimized_total"] = optimized_norm

    denominator = (
        objective_norm * anchor_norm
        if objective_norm is not None and anchor_norm is not None
        else 0.0
    )
    dot = sum(
        float(torch.sum(objective * policy))
        for objective, policy in zip(objective_total, anchor)
    )
    geometry["cosines"]["total__policy_anchor"] = (
        min(max(dot / denominator, -1.0), 1.0)
        if denominator > 0 and math.isfinite(dot)
        else None
    )
    return geometry


def _refresh_minibatch_weights(
    model,
    tok,
    buffers: dict[int, list[TraceRow]],
    labelled_pids: list[int],
    answer_only_pids: list[int],
    responsibility_score: str = "joint",
    responsibility_posterior: str = "softmax_entropy",
    responsibility_temperature: float = 1.0,
    responsibility_ess_floor: float = 0.0,
    responsibility_abstention: str = "none",
    responsibility_rejection_threshold: float = 0.0,
    responsibility_null_log_evidence: float = 0.0,
    responsibility_null_prior: float = 0.5,
    responsibility_policy: str = "current",
    responsibility_answer_policy: str = "current",
    variational_estimator: str = "delta_joint",
    labelled_numeric_constraint: str = "off",
    numeric_penalty: float = 2.0,
    numeric_contradiction_penalty: float = 0.0,
    numeric_missing_penalty: float = 0.0,
    record_joint_logprobs: bool = False,
    task=None,
    responsibility_verifier_rollouts: int = 0,
    responsibility_verifier_temperature: float = 1.0,
    responsibility_verifier_max_new_tokens: int = 64,
    responsibility_verifier_batch_size: int = 16,
    responsibility_verifier_smoothing_alpha: float = 0.5,
    responsibility_verifier_seed: int = 0,
    responsibility_verifier_diagnostics: dict[str, Any] | None = None,
    verifier_calibration_path: str | None = None,
    sampled_support_outer_initial: bool = False,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Recompute detached trace responsibilities for the current minibatches.

    Args:
        model: Language model after the latest gradient step.
        tok: Tokenizer.
        buffers: Dictionary mapping question id to list of TraceRow objects.
        labelled_pids: Labelled question ids currently in L'.
        answer_only_pids: Answer-only question ids currently in U'.
        responsibility_score: E-step logit definition.
        responsibility_posterior: Ordinary soft posterior, entropy-free
            single-delta posterior, or the registered two-witness coefficients.
        responsibility_temperature: Positive E-step softmax temperature.
        responsibility_ess_floor: One-sided minimum ESS fraction.
        responsibility_abstention: Optional question-level real update mass.
        responsibility_policy: Current adapter or frozen-base E-step scorer.
        responsibility_answer_policy: Current or frozen-base answer-reader factor.
        variational_estimator: Declared finite-sample Barber approximation.
        labelled_numeric_constraint: Local or calculation-graph arithmetic
            potential applied only to L'.
        numeric_penalty: Per-invalid-equation log penalty in soft mode.
        numeric_contradiction_penalty: Per-gold-contradiction soft penalty.
        numeric_missing_penalty: Per-missing graph item soft penalty.
        record_joint_logprobs: Persist the scoring logits for diagnostics.
        task: Answer-aware task required by rollout-value responsibilities.
        responsibility_verifier_rollouts: Continuations sampled per trace.
        responsibility_verifier_temperature: Continuation sampling temperature.
        responsibility_verifier_max_new_tokens: Continuation token limit.
        responsibility_verifier_batch_size: Continuation generation microbatch.
        responsibility_verifier_smoothing_alpha: Symmetric Beta smoothing.
        responsibility_verifier_seed: Base seed for deterministic rollout streams.
        responsibility_verifier_diagnostics: Optional mutable rollout counters.
        sampled_support_outer_initial: Enforce d=0 on the immediate
            post-sampling E-step for the exact sampled-support estimator.

    Returns:
        labelled_weights: Responsibilities over B(q) for q in labelled_pids.
        answer_only_weights: Responsibilities over B(q) for q in answer_only_pids.
    """

    labelled_weights = _buffer_weights_for_questions(
        model,
        tok,
        buffers,
        labelled_pids,
        responsibility_score=responsibility_score,
        responsibility_posterior=responsibility_posterior,
        responsibility_temperature=responsibility_temperature,
        responsibility_ess_floor=responsibility_ess_floor,
        responsibility_abstention=responsibility_abstention,
        responsibility_rejection_threshold=responsibility_rejection_threshold,
        responsibility_null_log_evidence=responsibility_null_log_evidence,
        responsibility_null_prior=responsibility_null_prior,
        responsibility_policy=responsibility_policy,
        responsibility_answer_policy=responsibility_answer_policy,
        variational_estimator=variational_estimator,
        labelled_numeric_constraint=labelled_numeric_constraint,
        numeric_penalty=numeric_penalty,
        numeric_contradiction_penalty=numeric_contradiction_penalty,
        numeric_missing_penalty=numeric_missing_penalty,
        record_joint_logprobs=record_joint_logprobs,
        task=task,
        responsibility_verifier_rollouts=responsibility_verifier_rollouts,
        responsibility_verifier_temperature=responsibility_verifier_temperature,
        responsibility_verifier_max_new_tokens=(
            responsibility_verifier_max_new_tokens
        ),
        responsibility_verifier_batch_size=responsibility_verifier_batch_size,
        responsibility_verifier_smoothing_alpha=(
            responsibility_verifier_smoothing_alpha
        ),
        responsibility_verifier_seed=responsibility_verifier_seed,
        responsibility_verifier_diagnostics=responsibility_verifier_diagnostics,
        verifier_calibration_path=verifier_calibration_path,
        sampled_support_outer_initial=sampled_support_outer_initial,
    )
    answer_only_weights = _buffer_weights_for_questions(
        model,
        tok,
        buffers,
        answer_only_pids,
        responsibility_score=responsibility_score,
        responsibility_posterior=responsibility_posterior,
        responsibility_temperature=responsibility_temperature,
        responsibility_ess_floor=responsibility_ess_floor,
        responsibility_abstention=responsibility_abstention,
        responsibility_rejection_threshold=responsibility_rejection_threshold,
        responsibility_null_log_evidence=responsibility_null_log_evidence,
        responsibility_null_prior=responsibility_null_prior,
        responsibility_policy=responsibility_policy,
        responsibility_answer_policy=responsibility_answer_policy,
        variational_estimator=variational_estimator,
        record_joint_logprobs=record_joint_logprobs,
        task=task,
        responsibility_verifier_rollouts=responsibility_verifier_rollouts,
        responsibility_verifier_temperature=responsibility_verifier_temperature,
        responsibility_verifier_max_new_tokens=(
            responsibility_verifier_max_new_tokens
        ),
        responsibility_verifier_batch_size=responsibility_verifier_batch_size,
        responsibility_verifier_smoothing_alpha=(
            responsibility_verifier_smoothing_alpha
        ),
        responsibility_verifier_seed=responsibility_verifier_seed + 1_000_003,
        responsibility_verifier_diagnostics=responsibility_verifier_diagnostics,
        verifier_calibration_path=verifier_calibration_path,
        sampled_support_outer_initial=sampled_support_outer_initial,
    )
    return labelled_weights, answer_only_weights


def _responsibility_refresh_total_variations(
    before: dict[int, torch.Tensor],
    after: dict[int, torch.Tensor],
) -> list[float]:
    """Return per-question total variation across one E-step refresh.

    Support is fixed inside an AC-ALG1 inner loop, so a changed question key or
    vector width is a bookkeeping error rather than a quantity to average away.
    Empty partitions return an empty list.
    """

    if set(before) != set(after):
        raise RuntimeError(
            "responsibility refresh changed the question support"
        )
    variations = []
    for pid in sorted(before):
        left = before[pid].detach().float().cpu()
        right = after[pid].detach().float().cpu()
        if left.shape != right.shape:
            raise RuntimeError(
                f"responsibility refresh changed support width for pid {pid}"
            )
        variation = 0.5 * float(torch.sum(torch.abs(left - right)).item())
        if not math.isfinite(variation):
            raise FloatingPointError(
                f"non-finite responsibility refresh variation for pid {pid}"
            )
        variations.append(min(max(variation, 0.0), 1.0))
    return variations


def _sampled_support_diagnostics(
    buffers: dict[int, list[TraceRow]],
    pids: list[int],
    weights: dict[int, torch.Tensor],
) -> dict[str, Any]:
    """Audit every cached identity for the exact sampled-support estimator."""

    questions: list[dict[str, Any]] = []
    maximum_q_sum_error = 0.0
    maximum_r_sum_error = 0.0
    maximum_signed_sum_error = 0.0
    maximum_initial_uniform_error = 0.0
    positive_coefficients = 0
    negative_coefficients = 0
    zero_coefficients = 0
    for pid in pids:
        rows = list(buffers[int(pid)])
        q_tensor = weights.get(int(pid))
        if not rows or q_tensor is None:
            continue
        if q_tensor.ndim != 1 or q_tensor.numel() != len(rows):
            raise RuntimeError("sampled-support diagnostic q does not match support")
        prior_values = [row.sampled_support_prior_mass for row in rows]
        marginal_values = [row.sampled_support_log_marginal for row in rows]
        initial_values = [row.sampled_support_outer_initial for row in rows]
        if not all(value is not None and math.isfinite(value) for value in prior_values):
            raise RuntimeError("sampled-support diagnostic is missing finite r")
        if not all(value is not None and math.isfinite(value) for value in marginal_values):
            raise RuntimeError("sampled-support diagnostic is missing its log marginal")
        if len({bool(value) for value in initial_values}) != 1:
            raise RuntimeError("sampled-support support mixes initial and refreshed rows")
        if max(float(value) for value in marginal_values) - min(
            float(value) for value in marginal_values
        ) > 1e-6:
            raise RuntimeError("sampled-support rows disagree on the question marginal")

        q = q_tensor.detach().float().cpu()
        r = torch.tensor([float(value) for value in prior_values], dtype=torch.float32)
        signed = q - r
        q_sum_error = abs(float(q.sum()) - 1.0)
        r_sum_error = abs(float(r.sum()) - 1.0)
        signed_sum_error = abs(float(signed.sum()))
        maximum_q_sum_error = max(maximum_q_sum_error, q_sum_error)
        maximum_r_sum_error = max(maximum_r_sum_error, r_sum_error)
        maximum_signed_sum_error = max(maximum_signed_sum_error, signed_sum_error)
        if max(q_sum_error, r_sum_error, signed_sum_error) > 1e-6:
            raise RuntimeError("sampled-support q/r coefficient identity failed")
        initial = bool(initial_values[0])
        initial_uniform_error = (
            float(torch.max(torch.abs(r - 1.0 / len(rows))))
            if initial else 0.0
        )
        maximum_initial_uniform_error = max(
            maximum_initial_uniform_error,
            initial_uniform_error,
        )
        if initial_uniform_error > 1e-7:
            raise RuntimeError("outer sampled-support r is not exactly uniform")
        positive_coefficients += int((signed > 1e-8).sum())
        negative_coefficients += int((signed < -1e-8).sum())
        zero_coefficients += int((torch.abs(signed) <= 1e-8).sum())
        behaviour = [float(row.proposal_trace_logprob) for row in rows]
        current = [float(row.trace_logprob) for row in rows]
        answer = [float(row.answer_logprob) for row in rows]
        if not all(math.isfinite(value) for value in [*behaviour, *current, *answer]):
            raise RuntimeError("sampled-support diagnostic scores must be finite")
        questions.append({
            "pid": int(pid),
            "support_size": len(rows),
            "trace_ids": [row.trace_id for row in rows],
            "outer_initial": initial,
            "q": [float(value) for value in q.tolist()],
            "r": [float(value) for value in r.tolist()],
            "q_minus_r": [float(value) for value in signed.tolist()],
            "behaviour_trace_logprobs": behaviour,
            "current_trace_logprobs": current,
            "answer_logprobs": answer,
            "log_marginal": float(marginal_values[0]),
            "q_sum_error": q_sum_error,
            "r_sum_error": r_sum_error,
            "signed_sum_error": signed_sum_error,
            "initial_uniform_error": initial_uniform_error,
        })
    return {
        "question_count": len(questions),
        "questions": questions,
        "maximum_q_sum_error": maximum_q_sum_error,
        "maximum_r_sum_error": maximum_r_sum_error,
        "maximum_signed_sum_error": maximum_signed_sum_error,
        "maximum_initial_uniform_error": maximum_initial_uniform_error,
        "positive_rationale_coefficients": positive_coefficients,
        "negative_rationale_coefficients": negative_coefficients,
        "zero_rationale_coefficients": zero_coefficients,
    }


def _fixed_surrogate_objective_values(
    model,
    tok,
    task,
    buffers: dict[int, list[TraceRow]],
    labelled_pids: list[int],
    answer_only_pids: list[int],
    labelled_weights: dict[int, torch.Tensor],
    answer_only_weights: dict[int, torch.Tensor],
    supervised_weight: float,
    labelled_em_weight: float,
    answer_only_em_weight: float,
    labelled_supervision: str,
    compact_gold_weight: float,
    digit_token_weight: float,
    answer_target_termination: str,
    latent_mstep_objective: str,
) -> dict[str, float]:
    """Evaluate the three M-step terms without refreshing responsibilities.

    These values define the generalized-EM rollback test.  Holding the E-step
    responsibilities fixed is essential: refreshing them before the comparison
    would compare two different surrogate functions and invalidate the
    monotonicity claim.
    """

    was_training = model.training
    model.eval()
    try:
        B_sup = (
            supervised_weight * _B_sup(
                model,
                tok,
                task,
                labelled_pids,
                labelled_supervision=labelled_supervision,
                compact_gold_weight=compact_gold_weight,
                digit_token_weight=digit_token_weight,
                answer_target_termination=answer_target_termination,
                grad=False,
            )
            if supervised_weight > 0 else torch.zeros((), device=DEV)
        )
        B_prime_unsup = (
            labelled_em_weight * _B_unsup_for_questions(
                model,
                tok,
                buffers,
                labelled_pids,
                labelled_weights,
                grad=False,
                latent_mstep_objective=latent_mstep_objective,
            )
            if labelled_em_weight > 0 else torch.zeros((), device=DEV)
        )
        B_unsup = (
            answer_only_em_weight * _B_unsup_for_questions(
                model,
                tok,
                buffers,
                answer_only_pids,
                answer_only_weights,
                grad=False,
                latent_mstep_objective=latent_mstep_objective,
            )
            if answer_only_em_weight > 0 else torch.zeros((), device=DEV)
        )
        return {
            "B_sup": float(B_sup.detach()),
            "B_prime_unsup": float(B_prime_unsup.detach()),
            "B_unsup": float(B_unsup.detach()),
        }
    finally:
        model.train(was_training)


def _snapshot_trainable_parameters(model) -> list[torch.Tensor]:
    """Clone trainable parameters before a candidate optimizer step."""

    with torch.no_grad():
        return [
            parameter.detach().clone()
            for parameter in model.parameters()
            if parameter.requires_grad
        ]


def _restore_trainable_parameters(model, snapshots: list[torch.Tensor]) -> None:
    """Restore a trainable-parameter snapshot after a rejected candidate step."""

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(parameters) != len(snapshots):
        raise ValueError("parameter rollback snapshot has the wrong length")
    with torch.no_grad():
        for parameter, snapshot in zip(parameters, snapshots):
            parameter.copy_(snapshot)


def _question_gradient_attribution(
    model,
    tok,
    task,
    buffers: dict[int, list[TraceRow]],
    labelled_pids: list[int],
    answer_only_pids: list[int],
    labelled_weights: dict[int, torch.Tensor],
    answer_only_weights: dict[int, torch.Tensor],
    aggregate_gradients: list[torch.Tensor | None],
    *,
    limit: int,
    supervised_weight: float,
    labelled_em_weight: float,
    answer_only_em_weight: float,
    labelled_supervision: str,
    compact_gold_weight: float,
    digit_token_weight: float,
    answer_target_termination: str,
    latent_mstep_objective: str,
) -> dict[str, object]:
    """Recompute a bounded subset of question-local gradients at one fixed point.

    The optimizer is untouched and CPU/CUDA RNG states are restored, so this
    opt-in diagnostic cannot change subsequent stochastic training.
    """

    if limit <= 0:
        return {
            "enabled": False,
            "selection": "disabled",
            "question_limit": 0,
            "questions": [],
        }
    candidates = []
    for partition, pids, weights_by_pid in (
        ("labelled", labelled_pids, labelled_weights),
        ("answer_only", answer_only_pids, answer_only_weights),
    ):
        for pid in pids:
            weights = weights_by_pid.get(int(pid))
            if weights is None or not buffers[int(pid)]:
                continue
            candidates.append((
                -float(torch.max(weights.detach()).item()),
                int(pid),
                partition,
                weights,
            ))
    selected = sorted(candidates)[:limit]
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    records = []
    try:
        for _negative_mass, pid, partition, weights in selected:
            model.zero_grad(set_to_none=True)
            if partition == "labelled":
                if supervised_weight > 0:
                    B_sup = supervised_weight * _B_sup(
                        model,
                        tok,
                        task,
                        [pid],
                        labelled_supervision=labelled_supervision,
                        compact_gold_weight=compact_gold_weight,
                        digit_token_weight=digit_token_weight,
                        answer_target_termination=answer_target_termination,
                    )
                    if B_sup.requires_grad:
                        (-B_sup).backward()
                if labelled_em_weight > 0:
                    _backward_B_unsup_for_questions(
                        model,
                        tok,
                        buffers,
                        [pid],
                        {pid: weights},
                        coefficient=labelled_em_weight,
                        latent_mstep_objective=latent_mstep_objective,
                    )
            elif answer_only_em_weight > 0:
                _backward_B_unsup_for_questions(
                    model,
                    tok,
                    buffers,
                    [pid],
                    {pid: weights},
                    coefficient=answer_only_em_weight,
                    latent_mstep_objective=latent_mstep_objective,
                )
            question_gradients = _snapshot_native_trainable_gradients(model)
            values = [float(value) for value in weights.detach().cpu()]
            rows = buffers[pid]
            records.append({
                "pid": pid,
                "partition": partition,
                "trace_count": len(rows),
                "max_responsibility": max(values),
                "correct_trace_mass": sum(
                    value
                    for row, value in zip(rows, values)
                    if row.proposal_correct is True
                ),
                "gradient_l2_norm": tensor_list_norm(question_gradients),
                "cosine_with_aggregate": tensor_list_cosine(
                    question_gradients,
                    aggregate_gradients,
                ),
            })
    finally:
        assign_trainable_gradients(model, aggregate_gradients)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    return {
        "enabled": True,
        "selection": "highest_max_responsibility",
        "question_limit": int(limit),
        "questions": records,
    }


def _m_step_support_diagnostics(
    tok,
    task,
    buffers: dict[int, list[TraceRow]],
    labelled_pids: list[int],
    answer_only_pids: list[int],
    labelled_weights: dict[int, torch.Tensor],
    answer_only_weights: dict[int, torch.Tensor],
    *,
    supervised_weight: float,
    labelled_em_weight: float,
    answer_only_em_weight: float,
    labelled_supervision: str,
    answer_target_termination: str,
    latent_mstep_objective: str,
) -> dict[str, int]:
    """Count exact question, trace and target-token work for one inner step."""

    active_labelled = [
        int(pid)
        for pid in labelled_pids
        if (
            supervised_weight > 0
            or (
                labelled_em_weight > 0
                and buffers[int(pid)]
                and labelled_weights.get(int(pid)) is not None
            )
        )
    ]
    active_answer_only = [
        int(pid)
        for pid in answer_only_pids
        if (
            answer_only_em_weight > 0
            and buffers[int(pid)]
            and answer_only_weights.get(int(pid)) is not None
        )
    ]
    buffer_rows = []
    if labelled_em_weight > 0:
        buffer_rows.extend(
            row for pid in labelled_pids for row in buffers[int(pid)]
            if labelled_weights.get(int(pid)) is not None
        )
    if answer_only_em_weight > 0:
        buffer_rows.extend(
            row for pid in answer_only_pids for row in buffers[int(pid)]
            if answer_only_weights.get(int(pid)) is not None
        )

    supervised_tokens = 0
    supervised_eos_tokens = 0
    if supervised_weight > 0 and labelled_pids:
        gold_solutions = [task.gold_solution[int(pid)] for pid in labelled_pids]
        if labelled_supervision == "gold_answer":
            gold_solutions = [
                f"#### {task.gold_answer[int(pid)]}"
                for pid in labelled_pids
            ]
        elif labelled_supervision == "gold_graph_factorized":
            gold_solutions = [
                _structured_gold_solution(solution, task.gold_answer[int(pid)])
                for pid, solution in zip(labelled_pids, gold_solutions)
            ]
        gold_batch = _build_supervised_batch(
            tok,
            task,
            labelled_pids,
            solutions=(
                gold_solutions
                if labelled_supervision in ("gold_answer", "gold_graph_factorized")
                else None
            ),
            answer_target_termination=answer_target_termination,
        )
        if gold_batch is not None:
            supervised_tokens += int(gold_batch[1].sum().item())
            supervised_eos_tokens += int(
                ((gold_batch[0] == tok.eos_token_id) & gold_batch[1]).sum().item()
            )
        if labelled_supervision in ("gold_compact_mix", "gold_compact_set"):
            compact_solutions = [
                _compact_gold_solution(
                    task.gold_solution[int(pid)],
                    task.gold_answer[int(pid)],
                )
                for pid in labelled_pids
            ]
            compact_batch = _build_supervised_batch(
                tok,
                task,
                labelled_pids,
                solutions=compact_solutions,
                answer_target_termination=answer_target_termination,
            )
            if compact_batch is not None:
                supervised_tokens += int(compact_batch[1].sum().item())
                supervised_eos_tokens += int(
                    (
                        (compact_batch[0] == tok.eos_token_id)
                        & compact_batch[1]
                    ).sum().item()
                )

    buffer_masks = [
        _latent_mstep_mask(
            row.span,
            row.ans,
            latent_mstep_objective,
        )
        for row in buffer_rows
    ]
    buffer_tokens = sum(int(mask.sum().item()) for mask in buffer_masks)
    buffer_eos_tokens = sum(
        int(((row.ids == tok.eos_token_id) & mask).sum().item())
        for row, mask in zip(buffer_rows, buffer_masks)
    )
    return {
        "active_questions": len(set(active_labelled + active_answer_only)),
        "active_labelled_questions": len(set(active_labelled)),
        "active_answer_only_questions": len(set(active_answer_only)),
        "active_traces": len(buffer_rows),
        "supervised_backward_tokens": supervised_tokens,
        "supervised_backward_eos_tokens": supervised_eos_tokens,
        "buffer_backward_tokens": buffer_tokens,
        "buffer_backward_eos_tokens": buffer_eos_tokens,
        "backward_tokens": supervised_tokens + buffer_tokens,
        "backward_eos_tokens": supervised_eos_tokens + buffer_eos_tokens,
    }


def _unweighted_trace_nll(
    model,
    tok,
    buffers: dict[int, list[TraceRow]],
    pids: list[int],
    latent_mstep_objective: str = "joint",
) -> float | None:
    """Mean trace NLL with uniform within-question weights."""

    active = [int(pid) for pid in pids if buffers[int(pid)]]
    if not active:
        return None
    weights = {
        pid: torch.full(
            (len(buffers[pid]),),
            1.0 / len(buffers[pid]),
            dtype=torch.float32,
        )
        for pid in active
    }
    value = _B_unsup_for_questions(
        model,
        tok,
        buffers,
        active,
        weights,
        grad=False,
        latent_mstep_objective=latent_mstep_objective,
    )
    return -float(value.detach())


def _run_diagnostic_probe(model, probe_fn) -> float:
    """Backward-compatible wrapper around the shared diagnostic probe."""

    return run_diagnostic_probe(model, probe_fn)


def _inner_weighted_em_steps(
    model,
    tok,
    opt,
    task,
    buffers: dict[int, list[TraceRow]],
    labelled_pids: list[int],
    answer_only_pids: list[int],
    labelled_weights: dict[int, torch.Tensor],
    answer_only_weights: dict[int, torch.Tensor],
    inner_steps: int,
    responsibility_score: str = "joint",
    responsibility_posterior: str = "softmax_entropy",
    responsibility_temperature: float = 1.0,
    responsibility_ess_floor: float = 0.0,
    responsibility_abstention: str = "none",
    responsibility_rejection_threshold: float = 0.0,
    responsibility_null_log_evidence: float = 0.0,
    responsibility_null_prior: float = 0.5,
    responsibility_policy: str = "current",
    responsibility_answer_policy: str = "current",
    responsibility_refresh: str = "inner_step",
    variational_estimator: str = "delta_joint",
    labelled_em_weight: float = 1.0,
    answer_only_em_weight: float = 1.0,
    policy_kl_coef: float | None = None,
    supervised_weight: float = 1.0,
    policy_anchor_mode: str = "fixed",
    policy_anchor_target_ratio: float | None = None,
    policy_anchor_beta_min: float = 0.0,
    policy_anchor_beta_max: float = 10.0,
    policy_anchor_ema: float = 0.9,
    policy_anchor_token_scope: str = "objective",
    policy_anchor_state: dict[str, float] | None = None,
    labelled_numeric_constraint: str = "off",
    numeric_penalty: float = 2.0,
    numeric_contradiction_penalty: float = 0.0,
    numeric_missing_penalty: float = 0.0,
    labelled_supervision: str = "gold",
    compact_gold_weight: float = 0.5,
    digit_token_weight: float = 1.0,
    answer_target_termination: str = "none",
    latent_mstep_objective: str = "joint",
    update_geometry: str = "sum",
    step_acceptance: str = "none",
    rollback_tolerance: float = 1e-6,
    rollback_max_backtracks: int = 0,
    rollback_shrink: float = 0.5,
    record_joint_logprobs: bool = False,
    record_gradient_geometry: bool = False,
    diagnostics_level: str = "standard",
    diagnostics_gradient_questions: int = 0,
    diagnostics_probe_fn=None,
    diagnostic_state: dict[str, object] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, object]]:
    """Run repeated objective ascent and responsibility refreshes on one minibatch.

    Args:
        model: Trainable language model.
        tok: Tokenizer.
        opt: Optimizer for trainable model parameters.
        task: GSM8K-style task.
        buffers: Dictionary mapping question id to trace rows.
        labelled_pids: Labelled question ids in L'.
        answer_only_pids: Answer-only question ids in U'.
        labelled_weights: Initial detached responsibilities for labelled buffers.
        answer_only_weights: Initial detached responsibilities for answer-only buffers.
        inner_steps: Number of gradient/refresh iterations.
        responsibility_score: E-step logit definition.
        responsibility_posterior: Entropy-regularised soft posterior or
            entropy-free single-delta posterior.
        responsibility_temperature: Positive E-step softmax temperature.
        responsibility_ess_floor: One-sided minimum ESS fraction.
        responsibility_abstention: Optional question-level real update mass.
        responsibility_policy: Current adapter or frozen-base E-step scorer.
        responsibility_answer_policy: Current or frozen-base answer-reader factor.
        responsibility_refresh: Refresh after every gradient step or hold the
            outer-round E-step fixed throughout the approximate M-step.
        variational_estimator: Declared finite-sample Barber approximation.
        labelled_em_weight: Coefficient on B'_unsup; zero removes that term.
        answer_only_em_weight: Coefficient on B_unsup; zero removes that term.
        policy_kl_coef: KL-to-frozen-base coefficient. ``None`` disables both
            measurement and anchoring; zero measures the matched control.
        supervised_weight: Coefficient on B_sup; zero removes the direct
            gold-rationale supervision term.
        policy_anchor_mode: ``fixed`` uses policy_kl_coef; ``grad_ratio``
            adapts beta so the applied KL-gradient norm tracks a target
            fraction of the unanchored objective-gradient norm.
        policy_anchor_target_ratio: Target applied-anchor/objective gradient
            norm ratio in grad_ratio mode.
        policy_anchor_beta_min: Lower clip for adaptive beta.
        policy_anchor_beta_max: Upper clip for adaptive beta.
        policy_anchor_ema: EMA decay for the two gradient norms.
        policy_anchor_token_scope: ``objective`` anchors every M-step target
            token; ``reasoning`` excludes the answer marker, answer and EOS.
        policy_anchor_state: Mutable EMA state shared across outer rounds.
        labelled_numeric_constraint: Local or calculation-graph arithmetic
            potential applied only to L'.
        numeric_penalty: Per-invalid-equation log penalty in soft mode.
        numeric_contradiction_penalty: Per-gold-contradiction soft penalty.
        numeric_missing_penalty: Per-missing graph item soft penalty.
        labelled_supervision: Full gold or full/compact mixed B_sup target.
        compact_gold_weight: Compact-target mixture weight.
        digit_token_weight: Scale-preserving relative weight on digit tokens.
        latent_mstep_objective: Joint, answer-only, or rationale-only target
            span for both latent objective terms.
        update_geometry: ``sum`` for the original gradient, ``mgda`` for a
            raw common-descent direction, ``normalized_mgda`` for its
            scale-invariant counterpart, or ``answer_primary`` for a
            B_unsup-dominant direction with non-conflicting auxiliaries.
        step_acceptance: Post-Adam fixed-surrogate acceptance rule.
        rollback_tolerance: Absolute no-harm tolerance for the fixed surrogate.
        rollback_max_backtracks: Smaller candidate steps tried after rejection.
        rollback_shrink: Multiplicative learning-rate shrink per backtrack.
        record_joint_logprobs: Persist logits from the final refresh for diagnostics.
        record_gradient_geometry: Decompose existing final-inner-step gradients
            without another forward or backward pass.
        diagnostics_level: ``standard`` records values already produced by
            training. ``deep`` additionally evaluates the fixed-responsibility
            surrogate after otherwise-unchecked optimizer steps.
        diagnostics_gradient_questions: Number of question-local gradients to
            attribute per inner step. Zero disables the additional backwards.
        diagnostics_probe_fn: Optional fixed-probe evaluator called after each
            accepted inner M-step. The caller must keep its prompt set fixed.
        diagnostic_state: Mutable accepted/rejected-step counters shared across
            outer rounds.

    Returns:
        labelled_weights: Final labelled responsibilities under the configured
            refresh policy.
        answer_only_weights: Final answer-only responsibilities under the
            configured refresh policy.
        stats: Mean objective values and number of optimizer steps taken.
    """

    if responsibility_refresh not in RESPONSIBILITY_REFRESH_MODES:
        raise ValueError(
            "unknown AC-ALG1 responsibility_refresh "
            f"{responsibility_refresh!r}"
        )
    if variational_estimator not in VARIATIONAL_ESTIMATORS:
        raise ValueError(
            f"unknown AC-ALG1 variational_estimator {variational_estimator!r}"
        )
    if responsibility_score == "token_mean" and variational_estimator != "delta_joint":
        raise ValueError(
            "token_mean responsibilities require variational_estimator='delta_joint'"
        )
    if (
        variational_estimator != "delta_joint"
        and responsibility_posterior
        not in {
            "softmax_entropy",
            "two_witness",
            *MULTI_VERIFIER_POSTERIORS,
            "verifier_bayesian",
        }
    ):
        raise ValueError(
            "Monte Carlo and importance estimators require softmax_entropy or "
            "a registered derived posterior"
        )
    if (
        responsibility_posterior == "two_witness"
        and variational_estimator != "prior_importance"
    ):
        raise ValueError("two_witness requires prior_importance evidence")
    diagnostics_level = validate_diagnostic_level(diagnostics_level)
    if diagnostics_gradient_questions < 0:
        raise ValueError("diagnostics_gradient_questions must be nonnegative")
    diagnostic_state = diagnostic_state if diagnostic_state is not None else {}
    diagnostic_state.setdefault("accepted_steps", 0)
    diagnostic_state.setdefault("consecutive_rejections", 0)

    totals = {
        "B_sup": 0.0,
        "B_prime_unsup": 0.0,
        "B_unsup": 0.0,
        "F": 0.0,
        "policy_kl": 0.0,
        "policy_kl_penalty": 0.0,
        "F_anchored": 0.0,
        "steps": 0.0,
    }
    gradient_geometry = None
    update_geometry_diagnostics = None
    update_metric_sums = {
        "direction_norm": 0.0,
        "B_sup_coefficient": 0.0,
        "B_prime_unsup_coefficient": 0.0,
        "B_unsup_coefficient": 0.0,
        "count": 0.0,
    }
    safeguard_totals = {
        "candidate_steps": 0.0,
        "rolled_back_candidates": 0.0,
        "backtracks": 0.0,
        "accepted_scale": 0.0,
        "accepted_surrogate_total_delta": 0.0,
        "accepted_B_sup_delta": 0.0,
        "accepted_B_prime_unsup_delta": 0.0,
        "accepted_B_unsup_delta": 0.0,
        "accepted_steps": 0.0,
    }
    responsibility_refresh_variations: list[float] = []
    inner_step_records: list[dict[str, object]] = []
    posterior_refresh_records: list[dict[str, object]] = []
    policy_anchor_state = policy_anchor_state if policy_anchor_state is not None else {}
    policy_anchor_measured = (
        policy_kl_coef is not None or policy_anchor_mode == "grad_ratio"
    )
    anchor_metric_sums = {
        "beta": 0.0,
        "beta_unclipped": 0.0,
        "beta_clipped": 0.0,
        "objective_grad_norm": 0.0,
        "raw_anchor_grad_norm": 0.0,
        "applied_anchor_grad_norm": 0.0,
        "achieved_ratio": 0.0,
        "ema_objective_grad_norm": 0.0,
        "ema_raw_anchor_grad_norm": 0.0,
    }
    support_diagnostics = (
        _m_step_support_diagnostics(
            tok,
            task,
            buffers,
            labelled_pids,
            answer_only_pids,
            labelled_weights,
            answer_only_weights,
            supervised_weight=supervised_weight,
            labelled_em_weight=labelled_em_weight,
            answer_only_em_weight=answer_only_em_weight,
            labelled_supervision=labelled_supervision,
            answer_target_termination=answer_target_termination,
            latent_mstep_objective=latent_mstep_objective,
        )
        if record_gradient_geometry
        else {
            "active_questions": 0,
            "active_labelled_questions": 0,
            "active_answer_only_questions": 0,
            "active_traces": 0,
            "backward_tokens": 0,
            "supervised_backward_tokens": 0,
            "supervised_backward_eos_tokens": 0,
            "buffer_backward_tokens": 0,
            "buffer_backward_eos_tokens": 0,
            "backward_eos_tokens": 0,
        }
    )

    for inner_index in range(inner_steps):
        step_started = time.perf_counter()
        sampled_support_before = (
            _sampled_support_diagnostics(
                buffers,
                [*labelled_pids, *answer_only_pids],
                {**labelled_weights, **answer_only_weights},
            )
            if (
                record_gradient_geometry
                and variational_estimator == "sampled_support_importance"
            )
            else None
        )
        collect_gradient_geometry = record_gradient_geometry
        parameter_before_step = (
            _snapshot_trainable_parameters(model)
            if record_gradient_geometry else None
        )
        step_attempts: list[dict[str, object]] = []
        model.train()
        opt.zero_grad()

        B_sup_took_grad = False
        if supervised_weight == 0:
            B_sup_value = 0.0
        else:
            B_sup = supervised_weight * _B_sup(
                model,
                tok,
                task,
                labelled_pids,
                labelled_supervision=labelled_supervision,
                compact_gold_weight=compact_gold_weight,
                digit_token_weight=digit_token_weight,
                answer_target_termination=answer_target_termination,
            )
            B_sup_value = float(B_sup.detach())
            if B_sup.requires_grad:
                (-B_sup).backward()
                B_sup_took_grad = True
        supervised_gradients = (
            _snapshot_trainable_gradients(model) if collect_gradient_geometry else None
        )
        supervised_native_gradients = (
            _snapshot_native_trainable_gradients(model)
            if update_geometry != "sum" else None
        )

        if labelled_em_weight == 0:
            B_prime_unsup_value, B_prime_took_grad = 0.0, False
        else:
            B_prime_unsup_value, B_prime_took_grad = _backward_B_unsup_for_questions(
                model,
                tok,
                buffers,
                labelled_pids,
                labelled_weights,
                coefficient=labelled_em_weight,
                latent_mstep_objective=latent_mstep_objective,
            )
        after_labelled_gradients = (
            _snapshot_trainable_gradients(model) if collect_gradient_geometry else None
        )
        after_labelled_native_gradients = (
            _snapshot_native_trainable_gradients(model)
            if update_geometry != "sum" else None
        )
        if answer_only_em_weight == 0:
            B_unsup_value, B_unsup_took_grad = 0.0, False
        else:
            B_unsup_value, B_unsup_took_grad = _backward_B_unsup_for_questions(
                model,
                tok,
                buffers,
                answer_only_pids,
                answer_only_weights,
                coefficient=answer_only_em_weight,
                latent_mstep_objective=latent_mstep_objective,
            )
        objective_gradients = (
            _snapshot_trainable_gradients(model) if collect_gradient_geometry else None
        )
        raw_objective_native_gradients = (
            _snapshot_native_trainable_gradients(model)
            if record_gradient_geometry else None
        )
        summed_native_gradients = (
            _snapshot_native_trainable_gradients(model)
            if update_geometry == "sum" and hasattr(model, "parameters") else []
        )
        update_geometry_diagnostics = {
            "mode": "sum",
            "active_components": [
                name for name, active in (
                    ("B_sup", B_sup_took_grad),
                    ("B_prime_unsup", B_prime_took_grad),
                    ("B_unsup", B_unsup_took_grad),
                )
                if active
            ],
            "coefficients": {
                name: 1.0 for name, active in (
                    ("B_sup", B_sup_took_grad),
                    ("B_prime_unsup", B_prime_took_grad),
                    ("B_unsup", B_unsup_took_grad),
                )
                if active
            },
            "direction_norm": _gradient_norm(summed_native_gradients),
        }
        objective_took_grad = (
            B_sup_took_grad or B_prime_took_grad or B_unsup_took_grad
        )
        if update_geometry != "sum":
            objective_native_gradients = _snapshot_native_trainable_gradients(model)
            component_gradients = component_gradients_from_cumulative(
                supervised_native_gradients,
                after_labelled_native_gradients,
                objective_native_gradients,
            )
            optimized_gradients, update_geometry_diagnostics = (
                combine_component_gradients(
                    component_gradients,
                    mode=update_geometry,
                )
            )
            assign_trainable_gradients(model, optimized_gradients)
            objective_took_grad = (
                float(update_geometry_diagnostics["direction_norm"]) > 0.0
            )
        projected_objective_gradients = (
            _snapshot_native_trainable_gradients(model)
            if record_gradient_geometry else None
        )
        update_metric_sums["direction_norm"] += float(
            update_geometry_diagnostics.get("direction_norm", 0.0)
        )
        for component_name in ("B_sup", "B_prime_unsup", "B_unsup"):
            update_metric_sums[f"{component_name}_coefficient"] += float(
                update_geometry_diagnostics["coefficients"].get(component_name, 0.0)
            )
        update_metric_sums["count"] += 1.0

        adaptive_objective_gradients = (
            _snapshot_native_trainable_gradients(model)
            if policy_anchor_mode == "grad_ratio" else None
        )

        policy_kl_value = 0.0
        policy_kl_took_grad = False
        effective_beta = 0.0
        anchor_metrics = None
        if policy_anchor_measured:
            beta = (
                1.0 if policy_anchor_mode == "grad_ratio"
                else float(policy_kl_coef)
            )
            labelled_buffer_available = any(
                buffers[int(pid)] and labelled_weights.get(int(pid)) is not None
                for pid in labelled_pids
            )
            answer_only_buffer_available = any(
                buffers[int(pid)] and answer_only_weights.get(int(pid)) is not None
                for pid in answer_only_pids
            )
            component_weight = (
                (supervised_weight if labelled_pids else 0.0)
                + (labelled_em_weight if labelled_buffer_available else 0.0)
                + (answer_only_em_weight if answer_only_buffer_available else 0.0)
            )
            if component_weight > 0:
                supervised_kl_value = 0.0
                if supervised_weight > 0 and labelled_pids:
                    supervised_kl = _supervised_reference_policy_kl(
                        model,
                        tok,
                        task,
                        labelled_pids,
                        labelled_supervision=labelled_supervision,
                        answer_target_termination=answer_target_termination,
                        grad=beta > 0,
                    )
                    supervised_kl_value = float(supervised_kl.detach())
                    if beta > 0 and supervised_kl.requires_grad:
                        (
                            beta * supervised_weight * supervised_kl
                            / component_weight
                        ).backward()
                        policy_kl_took_grad = True

                labelled_kl_value, labelled_kl_took_grad = (
                    _backward_reference_policy_kl_for_questions(
                        model,
                        tok,
                        buffers,
                        labelled_pids,
                        labelled_weights,
                        backward_scale=(
                            beta * labelled_em_weight / component_weight
                            if labelled_buffer_available else 0.0
                        ),
                        latent_mstep_objective=latent_mstep_objective,
                        token_scope=policy_anchor_token_scope,
                    )
                )
                answer_only_kl_value, answer_only_kl_took_grad = (
                    _backward_reference_policy_kl_for_questions(
                        model,
                        tok,
                        buffers,
                        answer_only_pids,
                        answer_only_weights,
                        backward_scale=(
                            beta * answer_only_em_weight / component_weight
                            if answer_only_buffer_available else 0.0
                        ),
                        latent_mstep_objective=latent_mstep_objective,
                        token_scope=policy_anchor_token_scope,
                    )
                )
                policy_kl_took_grad = (
                    policy_kl_took_grad
                    or labelled_kl_took_grad
                    or answer_only_kl_took_grad
                )
                policy_kl_value = (
                    supervised_weight * supervised_kl_value
                    + labelled_em_weight * labelled_kl_value
                    + answer_only_em_weight * answer_only_kl_value
                ) / component_weight

            if policy_anchor_mode == "grad_ratio":
                anchor_metrics = _apply_gradient_ratio_policy_anchor(
                    model,
                    adaptive_objective_gradients,
                    policy_anchor_state,
                    target_ratio=float(policy_anchor_target_ratio),
                    beta_min=policy_anchor_beta_min,
                    beta_max=policy_anchor_beta_max,
                    ema=policy_anchor_ema,
                )
                effective_beta = anchor_metrics["beta"]
                policy_kl_took_grad = policy_kl_took_grad and effective_beta > 0
            else:
                effective_beta = beta

        applied_gradients = (
            _snapshot_native_trainable_gradients(model)
            if record_gradient_geometry else None
        )
        question_gradient_attribution = (
            _question_gradient_attribution(
                model,
                tok,
                task,
                buffers,
                labelled_pids,
                answer_only_pids,
                labelled_weights,
                answer_only_weights,
                applied_gradients,
                limit=diagnostics_gradient_questions,
                supervised_weight=supervised_weight,
                labelled_em_weight=labelled_em_weight,
                answer_only_em_weight=answer_only_em_weight,
                labelled_supervision=labelled_supervision,
                compact_gold_weight=compact_gold_weight,
                digit_token_weight=digit_token_weight,
                answer_target_termination=answer_target_termination,
                latent_mstep_objective=latent_mstep_objective,
            )
            if record_gradient_geometry
            and diagnostics_level == "deep"
            and diagnostics_gradient_questions > 0
            else {
                "enabled": False,
                "selection": "disabled",
                "question_limit": int(diagnostics_gradient_questions),
                "questions": [],
            }
        )

        if collect_gradient_geometry:
            gradient_geometry = _gradient_geometry_from_snapshots(
                supervised_gradients,
                after_labelled_gradients,
                objective_gradients,
            )
            if policy_anchor_measured:
                gradient_geometry = _add_policy_anchor_gradient_geometry(
                    gradient_geometry,
                    objective_gradients,
                    _snapshot_trainable_gradients(model),
                )
        took_grad = objective_took_grad or policy_kl_took_grad
        before_values = {
            "B_sup": B_sup_value,
            "B_prime_unsup": B_prime_unsup_value,
            "B_unsup": B_unsup_value,
        }
        after_values = None
        unweighted_trace_nll_before = None
        unweighted_trace_nll_after = None
        accepted = False
        rejection_reason = None
        if took_grad:
            if diagnostics_level == "deep" and record_gradient_geometry:
                unweighted_trace_nll_before = _unweighted_trace_nll(
                    model,
                    tok,
                    buffers,
                    [*labelled_pids, *answer_only_pids],
                    latent_mstep_objective=latent_mstep_objective,
                )
            if step_acceptance == "none":
                if diagnostics_level == "deep" and record_gradient_geometry:
                    before_values = _fixed_surrogate_objective_values(
                        model,
                        tok,
                        task,
                        buffers,
                        labelled_pids,
                        answer_only_pids,
                        labelled_weights,
                        answer_only_weights,
                        supervised_weight=supervised_weight,
                        labelled_em_weight=labelled_em_weight,
                        answer_only_em_weight=answer_only_em_weight,
                        labelled_supervision=labelled_supervision,
                        compact_gold_weight=compact_gold_weight,
                        digit_token_weight=digit_token_weight,
                        answer_target_termination=answer_target_termination,
                        latent_mstep_objective=latent_mstep_objective,
                    )
                attempt_started = time.perf_counter()
                opt.step()
                accepted = True
                if diagnostics_level == "deep" and record_gradient_geometry:
                    after_values = _fixed_surrogate_objective_values(
                        model,
                        tok,
                        task,
                        buffers,
                        labelled_pids,
                        answer_only_pids,
                        labelled_weights,
                        answer_only_weights,
                        supervised_weight=supervised_weight,
                        labelled_em_weight=labelled_em_weight,
                        answer_only_em_weight=answer_only_em_weight,
                        labelled_supervision=labelled_supervision,
                        compact_gold_weight=compact_gold_weight,
                        digit_token_weight=digit_token_weight,
                        answer_target_termination=answer_target_termination,
                        latent_mstep_objective=latent_mstep_objective,
                    )
                    unweighted_trace_nll_after = _unweighted_trace_nll(
                        model,
                        tok,
                        buffers,
                        [*labelled_pids, *answer_only_pids],
                        latent_mstep_objective=latent_mstep_objective,
                    )
                step_attempts.append({
                    "attempt": 0,
                    "learning_rate_scale": 1.0,
                    "accepted": True,
                    "failed_gates": [],
                    "rejection_reason": None,
                    "objective_after": after_values,
                    "elapsed_seconds": time.perf_counter() - attempt_started,
                    "parameter_update_norm": (
                        parameter_delta_norm(
                            parameter_before_step,
                            _snapshot_trainable_parameters(model),
                        )
                        if parameter_before_step is not None else None
                    ),
                })
            else:
                before_values = _fixed_surrogate_objective_values(
                    model,
                    tok,
                    task,
                    buffers,
                    labelled_pids,
                    answer_only_pids,
                    labelled_weights,
                    answer_only_weights,
                    supervised_weight=supervised_weight,
                    labelled_em_weight=labelled_em_weight,
                    answer_only_em_weight=answer_only_em_weight,
                    labelled_supervision=labelled_supervision,
                    compact_gold_weight=compact_gold_weight,
                    digit_token_weight=digit_token_weight,
                    answer_target_termination=answer_target_termination,
                    latent_mstep_objective=latent_mstep_objective,
                )
                parameter_snapshot = _snapshot_trainable_parameters(model)
                optimizer_snapshot = copy.deepcopy(opt.state_dict())
                base_learning_rates = [
                    float(group["lr"]) for group in opt.param_groups
                ]
                for backtrack_index in range(rollback_max_backtracks + 1):
                    attempt_started = time.perf_counter()
                    step_scale = rollback_shrink ** backtrack_index
                    for group, base_learning_rate in zip(
                        opt.param_groups, base_learning_rates
                    ):
                        group["lr"] = base_learning_rate * step_scale
                    opt.step()
                    after_values = _fixed_surrogate_objective_values(
                        model,
                        tok,
                        task,
                        buffers,
                        labelled_pids,
                        answer_only_pids,
                        labelled_weights,
                        answer_only_weights,
                        supervised_weight=supervised_weight,
                        labelled_em_weight=labelled_em_weight,
                        answer_only_em_weight=answer_only_em_weight,
                        labelled_supervision=labelled_supervision,
                        compact_gold_weight=compact_gold_weight,
                        digit_token_weight=digit_token_weight,
                        answer_target_termination=answer_target_termination,
                        latent_mstep_objective=latent_mstep_objective,
                    )
                    if diagnostics_level == "deep" and record_gradient_geometry:
                        unweighted_trace_nll_after = _unweighted_trace_nll(
                            model,
                            tok,
                            buffers,
                            [*labelled_pids, *answer_only_pids],
                            latent_mstep_objective=latent_mstep_objective,
                        )
                    accepted, safeguard_diagnostics = fixed_surrogate_acceptance(
                        before_values,
                        after_values,
                        mode=step_acceptance,
                        active_components=update_geometry_diagnostics[
                            "active_components"
                        ],
                        tolerance=rollback_tolerance,
                    )
                    failed_gates = [
                        name
                        for name, passed in safeguard_diagnostics["checks"].items()
                        if not passed
                    ]
                    attempt_update_norm = (
                        parameter_delta_norm(
                            parameter_snapshot,
                            _snapshot_trainable_parameters(model),
                        )
                        if record_gradient_geometry else None
                    )
                    step_attempts.append({
                        "attempt": backtrack_index,
                        "learning_rate_scale": step_scale,
                        "accepted": bool(accepted),
                        "failed_gates": failed_gates,
                        "rejection_reason": (
                            None
                            if accepted
                            else "failed:" + ",".join(failed_gates)
                        ),
                        "objective_after": dict(after_values),
                        "objective_deltas": dict(
                            safeguard_diagnostics["deltas"]
                        ),
                        "objective_total_delta": float(
                            safeguard_diagnostics["total_delta"]
                        ),
                        "elapsed_seconds": time.perf_counter() - attempt_started,
                        "parameter_update_norm": attempt_update_norm,
                    })
                    safeguard_totals["candidate_steps"] += 1.0
                    safeguard_totals["backtracks"] += float(backtrack_index > 0)
                    if accepted:
                        safeguard_totals["accepted_steps"] += 1.0
                        safeguard_totals["accepted_scale"] += step_scale
                        safeguard_totals["accepted_surrogate_total_delta"] += float(
                            safeguard_diagnostics["total_delta"]
                        )
                        for component_name in (
                            "B_sup",
                            "B_prime_unsup",
                            "B_unsup",
                        ):
                            safeguard_totals[
                                f"accepted_{component_name}_delta"
                            ] += float(
                                safeguard_diagnostics["deltas"][component_name]
                            )
                        break

                    safeguard_totals["rolled_back_candidates"] += 1.0
                    _restore_trainable_parameters(model, parameter_snapshot)
                    opt.load_state_dict(optimizer_snapshot)

                for group, base_learning_rate in zip(
                    opt.param_groups, base_learning_rates
                ):
                    group["lr"] = base_learning_rate
                took_grad = accepted
                if not accepted:
                    rejection_reason = (
                        step_attempts[-1]["rejection_reason"]
                        if step_attempts else "all_backtracking_attempts_rejected"
                    )
                    after_values = None
                    unweighted_trace_nll_after = None
        else:
            rejection_reason = "no_gradient"
            step_attempts.append({
                "attempt": 0,
                "learning_rate_scale": 0.0,
                "accepted": False,
                "failed_gates": ["gradient_available"],
                "rejection_reason": rejection_reason,
                "objective_after": None,
                "elapsed_seconds": time.perf_counter() - step_started,
                "parameter_update_norm": 0.0,
            })

        behavioural_probe = {
            "evaluated": False,
            "accuracy": None,
            "change_from_previous": None,
            "elapsed_seconds": 0.0,
        }
        if accepted and diagnostics_probe_fn is not None:
            probe_started = time.perf_counter()
            previous_probe = diagnostic_state.get("probe_previous_accuracy")
            probe_accuracy = _run_diagnostic_probe(model, diagnostics_probe_fn)
            probe_elapsed = time.perf_counter() - probe_started
            diagnostic_state["probe_previous_accuracy"] = probe_accuracy
            diagnostic_state["probe_elapsed_seconds"] = (
                float(diagnostic_state.get("probe_elapsed_seconds", 0.0))
                + probe_elapsed
            )
            behavioural_probe = {
                "evaluated": True,
                "accuracy": probe_accuracy,
                "change_from_previous": (
                    probe_accuracy - float(previous_probe)
                    if previous_probe is not None else None
                ),
                "elapsed_seconds": probe_elapsed,
            }

        F_value = B_sup_value + B_prime_unsup_value + B_unsup_value
        policy_kl_penalty = effective_beta * policy_kl_value
        F_anchored_value = F_value - policy_kl_penalty
        totals["B_sup"] += B_sup_value
        totals["B_prime_unsup"] += B_prime_unsup_value
        totals["B_unsup"] += B_unsup_value
        totals["F"] += F_value
        totals["policy_kl"] += policy_kl_value
        totals["policy_kl_penalty"] += policy_kl_penalty
        totals["F_anchored"] += F_anchored_value
        totals["steps"] += float(took_grad)
        anchor_metric_sums["beta"] += effective_beta
        anchor_metric_sums["beta_unclipped"] += (
            anchor_metrics["beta_unclipped"]
            if anchor_metrics is not None else effective_beta
        )
        if anchor_metrics is not None:
            for key in (
                "beta_clipped",
                "objective_grad_norm",
                "raw_anchor_grad_norm",
                "applied_anchor_grad_norm",
                "achieved_ratio",
                "ema_objective_grad_norm",
                "ema_raw_anchor_grad_norm",
            ):
                anchor_metric_sums[key] += anchor_metrics[key]

        current_inner_record = None
        if record_gradient_geometry:
            parameter_after_step = _snapshot_trainable_parameters(model)
            applied_update_norm = (
                parameter_delta_norm(
                    parameter_before_step,
                    parameter_after_step,
                )
                if parameter_before_step is not None else None
            )
            raw_gradient_norm = tensor_list_norm(
                raw_objective_native_gradients
            )
            projected_gradient_norm = tensor_list_norm(
                projected_objective_gradients
            )
            combined_gradient_norm = tensor_list_norm(applied_gradients)
            anchor_gradients = []
            for applied, projected in zip(
                applied_gradients,
                projected_objective_gradients,
            ):
                if applied is None and projected is None:
                    anchor_gradients.append(None)
                elif applied is None:
                    anchor_gradients.append(-projected)
                elif projected is None:
                    anchor_gradients.append(applied.clone())
                else:
                    anchor_gradients.append(applied - projected)
            anchor_gradient_norm = tensor_list_norm(anchor_gradients)
            policy_anchor_cosine = tensor_list_cosine(
                projected_objective_gradients,
                anchor_gradients,
            )
            before_total = sum(before_values.values())
            after_total = (
                sum(after_values.values()) if after_values is not None else None
            )
            objective_gain = (
                after_total - before_total
                if after_total is not None else None
            )
            step_elapsed = time.perf_counter() - step_started
            backward_tokens = support_diagnostics["backward_tokens"]
            if accepted:
                diagnostic_state["accepted_steps"] += 1
                diagnostic_state["consecutive_rejections"] = 0
            else:
                diagnostic_state["consecutive_rejections"] += 1
            moment_diagnostics = optimizer_moment_diagnostics(opt)
            current_inner_record = {
                "inner_step": inner_index,
                "status": "accepted" if accepted else "rejected",
                "rejection_reason": rejection_reason,
                "objective": {
                    "weighted_before": {
                        **before_values,
                        "total": before_total,
                    },
                    "weighted_after": (
                        {**after_values, "total": after_total}
                        if after_values is not None else None
                    ),
                    "marginal_gain": objective_gain,
                    "gain_per_backward_token": (
                        objective_gain / backward_tokens
                        if objective_gain is not None and backward_tokens > 0
                        else None
                    ),
                    "gain_per_gpu_second": (
                        objective_gain / step_elapsed
                        if objective_gain is not None and step_elapsed > 0
                        else None
                    ),
                    "unweighted_trace_nll_before": unweighted_trace_nll_before,
                    "unweighted_trace_nll_after": unweighted_trace_nll_after,
                },
                "trust": {
                    "safety_nll_before": None,
                    "safety_nll_after": None,
                    "history_nll_before": None,
                    "history_nll_after": None,
                    "availability": "not_configured_for_ac_alg1",
                },
                "policy_kl": {
                    "nonnegative_before": policy_kl_value,
                    "signed_after": None,
                    "nonnegative_after": None,
                    "after_availability": "requires_extra_reference_forward",
                },
                "gradient": {
                    "raw_objective_l2_norm": raw_gradient_norm,
                    "projected_objective_l2_norm": projected_gradient_norm,
                    "raw_anchor_l2_norm": (
                        anchor_metrics["raw_anchor_grad_norm"]
                        if anchor_metrics is not None else anchor_gradient_norm
                    ),
                    "applied_anchor_l2_norm": anchor_gradient_norm,
                    "combined_l2_norm": combined_gradient_norm,
                    "policy_anchor_cosine": policy_anchor_cosine,
                    "clipping_fraction": 0.0,
                    "clipping_mode": "none",
                },
                "component_gradient_geometry": gradient_geometry,
                "update": {
                    "applied_parameter_l2_norm": applied_update_norm,
                    "effective_step_size": (
                        applied_update_norm / combined_gradient_norm
                        if applied_update_norm is not None
                        and combined_gradient_norm > 0
                        else None
                    ),
                    "update_to_gradient_ratio": (
                        applied_update_norm / combined_gradient_norm
                        if applied_update_norm is not None
                        and combined_gradient_norm > 0
                        else None
                    ),
                    **moment_diagnostics,
                },
                "anchor": {
                    "coefficient": effective_beta,
                    "unclipped_coefficient": (
                        anchor_metrics["beta_unclipped"]
                        if anchor_metrics is not None else effective_beta
                    ),
                    "coefficient_clipped": (
                        bool(anchor_metrics["beta_clipped"])
                        if anchor_metrics is not None else False
                    ),
                    "target_gradient_ratio": policy_anchor_target_ratio,
                    "achieved_gradient_ratio": (
                        anchor_metrics["achieved_ratio"]
                        if anchor_metrics is not None else None
                    ),
                },
                "support": dict(support_diagnostics),
                "gradient_attribution": question_gradient_attribution,
                "behavioural_probe": behavioural_probe,
                "attempts": step_attempts,
                "elapsed_seconds": step_elapsed,
                "rejected_attempt_elapsed_seconds": sum(
                    float(attempt["elapsed_seconds"])
                    for attempt in step_attempts
                    if not attempt["accepted"]
                ),
                "consecutive_rejections": int(
                    diagnostic_state["consecutive_rejections"]
                ),
                "cumulative_accepted_steps": int(
                    diagnostic_state["accepted_steps"]
                ),
                "posterior_churn": None,
            }
            if sampled_support_before is not None:
                current_inner_record["sampled_support"] = {
                    "before": sampled_support_before,
                    "after": None,
                }

        if responsibility_refresh == "inner_step":
            trace_ids_by_pid = (
                {
                    int(pid): [row.trace_id for row in buffers[int(pid)]]
                    for pid in [*labelled_pids, *answer_only_pids]
                    if (
                        int(pid) in labelled_weights
                        or int(pid) in answer_only_weights
                    )
                }
                if record_gradient_geometry
                else {}
            )
            refreshed_labelled, refreshed_answer_only = _refresh_minibatch_weights(
                model,
                tok,
                buffers,
                labelled_pids,
                answer_only_pids,
                responsibility_score=responsibility_score,
                responsibility_posterior=responsibility_posterior,
                responsibility_temperature=responsibility_temperature,
                responsibility_ess_floor=responsibility_ess_floor,
                responsibility_abstention=responsibility_abstention,
                responsibility_rejection_threshold=(
                    responsibility_rejection_threshold
                ),
                responsibility_null_log_evidence=(
                    responsibility_null_log_evidence
                ),
                responsibility_null_prior=responsibility_null_prior,
                responsibility_policy=responsibility_policy,
                responsibility_answer_policy=responsibility_answer_policy,
                variational_estimator=variational_estimator,
                labelled_numeric_constraint=labelled_numeric_constraint,
                numeric_penalty=numeric_penalty,
                numeric_contradiction_penalty=numeric_contradiction_penalty,
                numeric_missing_penalty=numeric_missing_penalty,
                record_joint_logprobs=(
                    variational_estimator == "sampled_support_importance"
                    or (record_joint_logprobs and inner_index == inner_steps - 1)
                ),
            )
            responsibility_refresh_variations.extend(
                _responsibility_refresh_total_variations(
                    labelled_weights,
                    refreshed_labelled,
                )
            )
            responsibility_refresh_variations.extend(
                _responsibility_refresh_total_variations(
                    answer_only_weights,
                    refreshed_answer_only,
                )
            )
            if record_gradient_geometry:
                labelled_churn = posterior_churn(
                    labelled_weights,
                    refreshed_labelled,
                    trace_ids=trace_ids_by_pid,
                )
                answer_only_churn = posterior_churn(
                    answer_only_weights,
                    refreshed_answer_only,
                    trace_ids=trace_ids_by_pid,
                )
                refresh_record = {
                    "inner_step": inner_index,
                    "labelled": labelled_churn,
                    "answer_only": answer_only_churn,
                }
                posterior_refresh_records.append(refresh_record)
                if current_inner_record is not None:
                    current_inner_record["posterior_churn"] = refresh_record
                    if variational_estimator == "sampled_support_importance":
                        current_inner_record["sampled_support"]["after"] = (
                            _sampled_support_diagnostics(
                                buffers,
                                [*labelled_pids, *answer_only_pids],
                                {**refreshed_labelled, **refreshed_answer_only},
                            )
                        )
            labelled_weights = refreshed_labelled
            answer_only_weights = refreshed_answer_only
        if current_inner_record is not None:
            inner_step_records.append(current_inner_record)

    denom = max(inner_steps, 1)
    stats = {
        key: value / denom
        for key, value in totals.items()
        if key != "steps"
    }
    stats["B_unsup_label"] = stats["B_prime_unsup"]
    stats["B_unsup_answer_only"] = stats["B_unsup"]
    stats["steps"] = totals["steps"]
    stats["gradient_geometry"] = gradient_geometry
    stats["update_geometry_diagnostics"] = update_geometry_diagnostics
    stats["update_geometry"] = update_geometry
    stats["step_acceptance"] = step_acceptance
    stats["responsibility_refresh"] = responsibility_refresh
    stats["responsibility_refresh_total_variation_mean"] = _mean_or_none(
        responsibility_refresh_variations
    )
    stats["responsibility_refresh_total_variation_max"] = (
        max(responsibility_refresh_variations)
        if responsibility_refresh_variations else None
    )
    update_count = max(update_metric_sums["count"], 1.0)
    stats["update_direction_norm"] = (
        update_metric_sums["direction_norm"] / update_count
    )
    for component_name in ("B_sup", "B_prime_unsup", "B_unsup"):
        stats[f"update_{component_name}_coefficient"] = (
            update_metric_sums[f"{component_name}_coefficient"] / update_count
        )
    stats["candidate_steps"] = safeguard_totals["candidate_steps"]
    stats["rolled_back_candidates"] = safeguard_totals["rolled_back_candidates"]
    stats["rollback_backtracks"] = safeguard_totals["backtracks"]
    accepted_safeguarded_steps = safeguard_totals["accepted_steps"]
    stats["safeguard_acceptance_fraction"] = (
        accepted_safeguarded_steps / safeguard_totals["candidate_steps"]
        if safeguard_totals["candidate_steps"] else None
    )
    stats["accepted_step_scale"] = (
        safeguard_totals["accepted_scale"] / accepted_safeguarded_steps
        if accepted_safeguarded_steps else None
    )
    stats["accepted_surrogate_total_delta"] = (
        safeguard_totals["accepted_surrogate_total_delta"]
        / accepted_safeguarded_steps
        if accepted_safeguarded_steps else None
    )
    for component_name in ("B_sup", "B_prime_unsup", "B_unsup"):
        stats[f"accepted_{component_name}_delta"] = (
            safeguard_totals[f"accepted_{component_name}_delta"]
            / accepted_safeguarded_steps
            if accepted_safeguarded_steps else None
        )
    stats["policy_anchor_mode"] = policy_anchor_mode
    stats["policy_anchor_target_ratio"] = policy_anchor_target_ratio
    stats["policy_anchor_beta"] = (
        anchor_metric_sums["beta"] / denom if policy_anchor_measured else None
    )
    stats["policy_anchor_beta_unclipped"] = (
        anchor_metric_sums["beta_unclipped"] / denom
        if policy_anchor_measured else None
    )
    for key in (
        "beta_clipped",
        "objective_grad_norm",
        "raw_anchor_grad_norm",
        "applied_anchor_grad_norm",
        "achieved_ratio",
        "ema_objective_grad_norm",
        "ema_raw_anchor_grad_norm",
    ):
        stats[f"policy_anchor_{key}"] = (
            anchor_metric_sums[key] / denom
            if policy_anchor_mode == "grad_ratio" else None
        )
    stats["inner_step_diagnostics"] = inner_step_records
    stats["posterior_refresh_diagnostics"] = posterior_refresh_records
    stats["diagnostics_level"] = diagnostics_level
    stats["diagnostics_gradient_questions"] = diagnostics_gradient_questions

    return labelled_weights, answer_only_weights, stats


def _finite_or_none(value):
    """Return a JSON-safe finite float, or None for NaN/Inf/missing values."""

    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean_or_none(values) -> float | None:
    """Mean of a non-empty numeric iterable, otherwise None."""

    values = [float(value) for value in values
              if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def _answer_parse_diagnostics(text: str, *, task=None) -> dict[str, object]:
    """Use the task's symbolic parser on chat MATH; retain GSM replay otherwise."""

    if task is not None and bool(
        getattr(task, "rendered_chat_prompts", False)
    ):
        parser = getattr(task, "parse_answer_event", None)
        if parser is None:
            raise ValueError("chat task has no answer-event parser")
        parsed_event = parser(text, mode="legacy")
        strict_event = parser(text, mode="strict_terminal_marker")
        parsed = parsed_event.answer
        mode = parsed_event.parse_mode
        trailing_contradiction = None
        format_compliant = bool(strict_event.strict_valid)
    else:
        marker = re.search(r"####\s*(-?\d[\d,]*)", text)
        numbers = re.findall(r"-?\d[\d,]*", text)
        if marker:
            raw = marker.group(1)
            mode = "marker"
        elif numbers:
            raw = numbers[-1]
            mode = "fallback_last_integer"
        else:
            raw = None
            mode = "unparsed"
        try:
            parsed = int(raw.replace(",", "")) if raw is not None else None
        except ValueError:
            parsed = None
            mode = "unparsed"
        trailing_contradiction = False
        if marker and numbers:
            try:
                trailing_contradiction = (
                    int(numbers[-1].replace(",", "")) != parsed
                )
            except ValueError:
                trailing_contradiction = False
        format_compliant = mode == "marker"
    words = text.split()
    repeated_unigrams = (
        1.0 - len(set(words)) / len(words) if words else None
    )
    fourgrams = [
        tuple(words[index:index + 4])
        for index in range(max(len(words) - 3, 0))
    ]
    repeated_fourgrams = (
        1.0 - len(set(fourgrams)) / len(fourgrams)
        if fourgrams else None
    )
    return {
        "parsed_answer": parsed,
        "parse_mode": mode,
        "format_compliant": format_compliant,
        "trailing_numeric_contradiction": trailing_contradiction,
        "repeated_token_fraction": repeated_unigrams,
        "repeated_fourgram_fraction": repeated_fourgrams,
    }


def _numeric_responsibility_metrics(weighted_rows) -> dict:
    """Summarise arithmetic and graph status under supplied responsibilities."""

    weighted_rows = list(weighted_rows)
    graph_available = [
        (row, float(weight))
        for row, weight in weighted_rows
        if row.numeric_audit.graph_fully_covered is not None
    ]
    graph_available_mass = sum(weight for _row, weight in graph_available)
    return {
        "numeric_valid_mass": sum(
            weight
            for row, weight in weighted_rows
            if row.numeric_audit.parsed_equations > 0
            and row.numeric_audit.invalid_equations == 0
            and row.numeric_audit.gold_contradictions == 0
        ),
        "numeric_invalid_mass": sum(
            weight
            for row, weight in weighted_rows
            if row.numeric_audit.invalid_equations > 0
            or row.numeric_audit.gold_contradictions > 0
        ),
        "numeric_unparsed_mass": sum(
            weight
            for row, weight in weighted_rows
            if row.numeric_audit.parsed_equations == 0
        ),
        "numeric_graph_available_mass": graph_available_mass,
        "numeric_graph_compatible_mass": sum(
            weight
            for row, weight in graph_available
            if row.numeric_audit.graph_fully_covered is True
        ),
        "numeric_graph_incomplete_mass": sum(
            weight
            for row, weight in graph_available
            if row.numeric_audit.graph_fully_covered is False
        ),
        "numeric_graph_unavailable_mass": sum(
            weight
            for row, weight in weighted_rows
            if row.numeric_audit.graph_fully_covered is None
        ),
        "numeric_graph_node_coverage": (
            sum(
                weight * float(row.numeric_audit.graph_node_coverage)
                for row, weight in graph_available
            ) / graph_available_mass
            if graph_available_mass > 0
            else None
        ),
        "numeric_graph_edge_coverage": (
            sum(
                weight * float(row.numeric_audit.graph_edge_coverage)
                for row, weight in graph_available
            ) / graph_available_mass
            if graph_available_mass > 0
            else None
        ),
    }


def _sample_diagnostics(
    task,
    pid_rows: list[int],
    texts: list[str],
    token_counts: list[int],
    sources: list[str],
    *,
    trace_ids: list[str] | None = None,
    rewards: list[float] | None = None,
    admitted: list[bool] | None = None,
    retained_after_insertion: list[bool] | None = None,
    verifier_calls: int | None = None,
) -> dict:
    """Summarise already-generated completions without another model call."""

    optional = (trace_ids, rewards, admitted, retained_after_insertion)
    if not (len(pid_rows) == len(texts) == len(token_counts) == len(sources)):
        raise ValueError("sample diagnostic inputs must have equal lengths")
    if any(values is not None and len(values) != len(texts) for values in optional):
        raise ValueError("optional sample diagnostic inputs must match text count")

    if not texts:
        return {
            "count": 0,
            "verifier_calls": 0,
            "correct_fraction": None,
            "format_fraction": None,
            "mean_tokens": None,
            "mean_characters": None,
            "mean_words": None,
            "unique_fraction_within_question": None,
            "duplicate_fraction_within_question": None,
            "question_outcomes": {"all_wrong": 0, "mixed": 0, "all_correct": 0},
            "samples": [],
        }

    trace_ids = trace_ids if trace_ids is not None else [None] * len(texts)
    admitted = admitted if admitted is not None else [True] * len(texts)
    retained_after_insertion = (
        retained_after_insertion
        if retained_after_insertion is not None
        else list(admitted)
    )
    if rewards is None:
        rewards = [float(value) for value in task.reward(texts, pids=pid_rows)]
        verifier_calls = len(texts)
    else:
        rewards = [float(value) for value in rewards]
        verifier_calls = 0 if verifier_calls is None else int(verifier_calls)
    correct = [bool(float(reward) > 0.5) for reward in rewards]
    formatted = [
        "####" in text or ("<answer>" in text and "</answer>" in text)
        for text in texts
    ]
    by_pid: dict[int, list[bool]] = defaultdict(list)
    for pid, is_correct in zip(pid_rows, correct):
        by_pid[int(pid)].append(is_correct)

    outcomes = Counter()
    for values in by_pid.values():
        if all(values):
            outcomes["all_correct"] += 1
        elif any(values):
            outcomes["mixed"] += 1
        else:
            outcomes["all_wrong"] += 1

    unique = len({(int(pid), text) for pid, text in zip(pid_rows, texts)})
    samples = [
        {
            "pid": int(pid),
            "trace_id": trace_id,
            "source": source,
            "text": text,
            "reward": _finite_or_none(reward),
            "correct": is_correct,
            "admitted": bool(was_admitted),
            "retained_after_insertion": bool(was_retained),
            "has_answer_marker": has_marker,
            "tokens": int(n_tokens),
            "characters": len(text),
            "words": len(text.split()),
            "gold_answer": (
                (
                    str(task.gold_answer[int(pid)])
                    if bool(getattr(task, "rendered_chat_prompts", False))
                    else int(task.gold_answer[int(pid)])
                )
                if hasattr(task, "gold_answer") else None
            ),
            **_answer_parse_diagnostics(text, task=task),
            "eos_reached": None,
            "truncated": None,
            "termination": "not_recorded_by_sampler",
        }
        for (pid, trace_id, source, text, reward, is_correct, was_admitted,
             was_retained, has_marker, n_tokens) in zip(
            pid_rows, trace_ids, sources, texts, rewards, correct, admitted,
            retained_after_insertion, formatted, token_counts
        )
    ]
    return {
        "count": len(texts),
        "verifier_calls": verifier_calls,
        "correct_fraction": float(np.mean(correct)),
        "format_fraction": float(np.mean(formatted)),
        "mean_tokens": float(np.mean(token_counts)),
        "mean_characters": float(np.mean([len(text) for text in texts])),
        "mean_words": float(np.mean([len(text.split()) for text in texts])),
        "unique_fraction_within_question": unique / len(texts),
        "duplicate_fraction_within_question": 1.0 - unique / len(texts),
        "question_outcomes": {
            "all_wrong": int(outcomes["all_wrong"]),
            "mixed": int(outcomes["mixed"]),
            "all_correct": int(outcomes["all_correct"]),
        },
        "samples": samples,
    }


def _buffer_diagnostics(
    buffers: dict[int, list[TraceRow]],
    buffer_limit: int,
    buffer_strategy: str,
    rows_added: int,
    rows_evicted: int,
    round_index: int | None = None,
) -> dict:
    """Describe retained trace composition after this round's pruning."""

    active = {int(pid): rows for pid, rows in buffers.items() if rows}
    all_rows = [row for rows in active.values() for row in rows]
    source_counts = Counter(row.source for row in all_rows)
    unique_rows = len({(row.pid, tuple(row.ids.tolist())) for row in all_rows})
    cluster_counts = {
        int(pid): len({row.calculation_path_signature for row in rows})
        for pid, rows in active.items()
    }
    n_rows = len(all_rows)
    saturated = (
        sum(len(rows) >= buffer_limit for rows in active.values()) / len(active)
        if buffer_limit > 0 and active
        else None
    )
    ages = (
        [max(int(round_index) - row.round_added, 0) for row in all_rows]
        if round_index is not None else []
    )
    age_counts = Counter(ages)
    correct_by_age: dict[str, dict[str, int]] = {}
    if round_index is not None:
        for age in sorted(age_counts):
            rows_at_age = [
                row
                for row in all_rows
                if max(int(round_index) - row.round_added, 0) == age
            ]
            correct_by_age[str(age)] = {
                "rows": len(rows_at_age),
                "correct": sum(
                    row.is_gold or row.proposal_correct is True
                    for row in rows_at_age
                ),
                "incorrect": sum(
                    row.proposal_correct is False for row in rows_at_age
                ),
                "unknown": sum(
                    not row.is_gold and row.proposal_correct is None
                    for row in rows_at_age
                ),
            }
    return {
        "strategy": buffer_strategy,
        "limit_per_question": int(buffer_limit),
        "rows": n_rows,
        "gold_rows": sum(row.is_gold for row in all_rows),
        "model_rows": sum(not row.is_gold for row in all_rows),
        "active_questions": len(active),
        "mean_rows_per_active_question": _mean_or_none(len(rows) for rows in active.values()),
        "max_rows_per_question": max((len(rows) for rows in active.values()), default=0),
        "saturated_question_fraction": saturated,
        "exact_duplicate_fraction": (1.0 - unique_rows / n_rows) if n_rows else None,
        "source_counts": dict(sorted(source_counts.items())),
        "rows_added_this_round": int(rows_added),
        "rows_evicted_this_round": int(rows_evicted),
        "rows_retained": n_rows,
        "unique_trace_fraction": unique_rows / n_rows if n_rows else None,
        "calculation_path_clusters": sum(cluster_counts.values()),
        "mean_calculation_path_clusters_per_question": _mean_or_none(
            cluster_counts.values()
        ),
        "mean_calculation_path_diversity_fraction": _mean_or_none(
            cluster_counts[int(pid)] / len(rows)
            for pid, rows in active.items()
            if rows
        ),
        "mean_trace_age": _mean_or_none(ages),
        "max_trace_age": max(ages, default=None),
        "age_counts": {
            str(age): int(count) for age, count in sorted(age_counts.items())
        },
        "correctness_by_age": correct_by_age,
        "per_question": [
            {
                "pid": pid,
                "rows": len(rows),
                "saturated": bool(buffer_limit > 0 and len(rows) >= buffer_limit),
                "unique_fraction": (
                    len({tuple(row.ids.tolist()) for row in rows}) / len(rows)
                    if rows else None
                ),
                "calculation_path_clusters": cluster_counts[int(pid)],
                "calculation_path_diversity_fraction": (
                    cluster_counts[int(pid)] / len(rows) if rows else None
                ),
            }
            for pid, rows in sorted(active.items())
        ],
    }


def _counterfactual_responsibility_diagnostics(rows: list[TraceRow]) -> dict:
    """Replay score/temperature combinations from already-computed joint logits."""

    schemes = (
        ("joint_tau1", "joint", 1.0),
        ("joint_tau2", "joint", 2.0),
        ("token_mean_tau1", "token_mean", 1.0),
        ("token_mean_tau2", "token_mean", 2.0),
    )
    joint = [float(row.joint_logprob) for row in rows]
    if not rows or any(not math.isfinite(value) for value in joint):
        return {
            name: {"available": False}
            for name, _score, _temperature in schemes
        }

    results = {}
    for name, score, temperature in schemes:
        logits = [
            value / max(int(row.span.sum()), 1) if score == "token_mean" else value
            for row, value in zip(rows, joint)
        ]
        scaled = [value / temperature for value in logits]
        maximum = max(scaled)
        unnormalized = [math.exp(value - maximum) for value in scaled]
        denominator = sum(unnormalized)
        weights = [value / denominator for value in unnormalized]
        order = sorted(range(len(weights)), key=weights.__getitem__, reverse=True)
        gold_indices = [index for index, row in enumerate(rows) if row.is_gold]
        gold_ranks = [order.index(index) + 1 for index in gold_indices]
        entropy = -sum(value * math.log(max(value, 1e-300)) for value in weights)
        top1_index = order[0]
        audited_model_rows = [
            (row, weight)
            for row, weight in zip(rows, weights)
            if not row.is_gold and row.numeric_audit is not None
        ]
        results[name] = {
            "available": True,
            "score": score,
            "temperature": temperature,
            "weights": weights,
            "gold_mass": sum(weights[index] for index in gold_indices),
            "gold_rank": min(gold_ranks) if gold_ranks else None,
            "gold_is_top1": bool(rows[top1_index].is_gold),
            "top1_index": top1_index,
            "top1_trace_id": rows[top1_index].trace_id,
            "top1_source": rows[top1_index].source,
            "max_responsibility": weights[top1_index],
            "normalized_entropy": entropy / math.log(len(rows)) if len(rows) > 1 else 0.0,
            "effective_sample_size_fraction": (
                1.0 / sum(value * value for value in weights) / len(rows)
            ),
            **_numeric_responsibility_metrics(audited_model_rows),
        }

    faithful_top1 = results["joint_tau1"]["top1_index"]
    for result in results.values():
        result["top1_changed_from_joint_tau1"] = result["top1_index"] != faithful_top1
    return results


def _responsibility_diagnostics(
    buffers: dict[int, list[TraceRow]],
    labelled_weights: dict[int, torch.Tensor],
    answer_only_weights: dict[int, torch.Tensor],
    round_index: int,
    *,
    tok=None,
    include_trace_tape: bool = False,
) -> dict:
    """Record final E-step responsibilities already computed for this minibatch."""

    traces = []
    questions = []
    partition_summaries = {}
    for partition, weights_by_pid in (
        ("labelled", labelled_weights),
        ("answer_only", answer_only_weights),
    ):
        partition_questions = []
        for pid, weights in weights_by_pid.items():
            rows = list(buffers[int(pid)])
            values = [float(value) for value in weights.detach().cpu().tolist()]
            if len(rows) != len(values):
                raise RuntimeError(
                    f"responsibility length mismatch for pid {pid}: {len(rows)} rows vs {len(values)} weights"
                )

            nonfinite = sum(not math.isfinite(value) for value in values)
            order = sorted(
                range(len(values)),
                key=lambda index: values[index] if math.isfinite(values[index]) else float("-inf"),
                reverse=True,
            )
            ranks = {index: rank + 1 for rank, index in enumerate(order)}
            if nonfinite:
                gold_mass = max_responsibility = entropy = normalized_entropy = None
                ess = ess_fraction = weighted_age = None
                gold_is_top1 = None
                gini = top_margin = answer_correct_mass = None
                weighted_joint_plus_entropy = joint_support_logsumexp = None
                real_coverage = null_mass = real_log_mean_evidence = None
                calculation_path_clusters = cluster_ess = cluster_ess_fraction = None
                weighted_verifier_value = None
            else:
                real_coverage = min(max(sum(values), 0.0), 1.0)
                null_mass = 1.0 - real_coverage
                conditional_values = (
                    [value / real_coverage for value in values]
                    if real_coverage > 0.0
                    else [0.0 for value in values]
                )
                evidence_values = {
                    row.responsibility_real_log_mean_evidence for row in rows
                }
                real_log_mean_evidence = (
                    evidence_values.pop() if len(evidence_values) == 1 else None
                )
                gold_mass = min(max(
                    sum(value for row, value in zip(rows, values) if row.is_gold), 0.0
                ), 1.0)
                max_responsibility = min(max(max(values, default=0.0), 0.0), 1.0)
                entropy = max(
                    -sum(
                        value * math.log(max(value, 1e-300))
                        for value in conditional_values
                    ),
                    0.0,
                )
                normalized_entropy = (
                    min(max(entropy / math.log(len(values)), 0.0), 1.0)
                    if len(values) > 1 else 0.0
                )
                squared_conditional_mass = sum(
                    value * value for value in conditional_values
                )
                ess = (
                    1.0 / squared_conditional_mass
                    if squared_conditional_mass > 0.0 else 0.0
                )
                ess_fraction = min(max(ess / len(values), 0.0), 1.0)
                weighted_age = sum(value * max(round_index - row.round_added, 0)
                                   for row, value in zip(rows, values))
                gold_is_top1 = bool(rows[order[0]].is_gold) if rows else False
                gini = responsibility_gini(values)
                top_margin = responsibility_margin(values)
                answer_correct_mass = sum(
                    value
                    for row, value in zip(rows, values)
                    if row.is_gold or row.proposal_correct is True
                )
                cluster_masses: dict[str, float] = defaultdict(float)
                for row, value in zip(rows, conditional_values):
                    cluster_masses[row.calculation_path_signature] += value
                calculation_path_clusters = len(cluster_masses)
                cluster_squared_mass = sum(
                    value * value for value in cluster_masses.values()
                )
                cluster_ess = (
                    1.0 / cluster_squared_mass if cluster_squared_mass > 0.0 else 0.0
                )
                cluster_ess_fraction = (
                    cluster_ess / calculation_path_clusters
                    if calculation_path_clusters else None
                )
                weighted_verifier_value = (
                    sum(
                        value * float(row.verifier_value)
                        for row, value in zip(rows, conditional_values)
                    )
                    if rows and all(row.verifier_value is not None for row in rows)
                    else None
                )
                joint_values = [float(row.joint_logprob) for row in rows]
                if joint_values and all(math.isfinite(value) for value in joint_values):
                    weighted_joint_plus_entropy = sum(
                        weight * (
                            joint
                            - math.log(max(weight, 1e-300))
                        )
                        for joint, weight in zip(joint_values, values)
                    )
                    joint_maximum = max(joint_values)
                    joint_support_logsumexp = joint_maximum + math.log(
                        sum(
                            math.exp(value - joint_maximum)
                            for value in joint_values
                        )
                    )
                else:
                    weighted_joint_plus_entropy = joint_support_logsumexp = None

            audited_model_rows = [
                (row, value)
                for row, value in zip(rows, values)
                if not row.is_gold and row.numeric_audit is not None
            ]
            numeric_metrics = _numeric_responsibility_metrics(audited_model_rows)
            before_values = [row.responsibility_before_potential for row in rows]
            before_available = bool(before_values) and all(
                value is not None and math.isfinite(value) for value in before_values
            )
            if before_available and not nonfinite:
                gold_mass_before_potential = min(max(
                    sum(
                        float(value)
                        for row, value in zip(rows, before_values)
                        if row.is_gold
                    ),
                    0.0,
                ), 1.0)
                gold_mass_change_from_potential = gold_mass - gold_mass_before_potential
                responsibility_total_variation_from_potential = 0.5 * sum(
                    abs(value - float(before_value))
                    for value, before_value in zip(values, before_values)
                )
            else:
                gold_mass_before_potential = None
                gold_mass_change_from_potential = None
                responsibility_total_variation_from_potential = None

            structured_rows = [
                (row, value)
                for row, value in zip(rows, values)
                if row.structured_audit is not None
            ]
            latent_verifier_rows = [
                row for row in rows if row.latent_verifier_audit is not None
            ]
            if latent_verifier_rows:
                verifier_modes = {
                    row.latent_verifier_mode for row in latent_verifier_rows
                }
                verifier_questions = {
                    row.latent_verifier_audit.raw_question
                    for row in latent_verifier_rows
                }
                verifier_targets = {
                    row.latent_verifier_audit.target_answer
                    for row in latent_verifier_rows
                }
                global_null_masses = {
                    row.latent_verifier_global_null_mass
                    for row in latent_verifier_rows
                }
                invalid_masses = {
                    row.latent_verifier_invalid_mass
                    for row in latent_verifier_rows
                }
                if (
                    len(verifier_modes) != 1
                    or None in verifier_modes
                    or len(verifier_questions) != 1
                    or len(verifier_targets) != 1
                    or len(global_null_masses) != 1
                    or None in global_null_masses
                    or len(invalid_masses) != 1
                    or None in invalid_masses
                ):
                    raise RuntimeError(
                        "latent-verifier question diagnostics are inconsistent"
                    )
                applied_arithmetic_counts = Counter(
                    row.latent_verifier_applied_arithmetic
                    for row in latent_verifier_rows
                )
                applied_graph_counts = Counter(
                    row.latent_verifier_applied_graph
                    for row in latent_verifier_rows
                )
                raw_validity_probabilities = [
                    joint_validity_probability(
                        row.latent_verifier_audit.arithmetic_observation,
                        row.latent_verifier_audit.graph_observation,
                        posterior=(
                            "verifier_joint"
                            if row.latent_verifier_mode == "verifier_bayesian"
                            else row.latent_verifier_mode
                        ),
                    )
                    for row in latent_verifier_rows
                ]
                latent_verifier_summary = {
                    "mode": next(iter(verifier_modes)),
                    "raw_question": next(iter(verifier_questions)),
                    "target_answer": next(iter(verifier_targets)),
                    "mean_validity_probability": _mean_or_none(
                        row.latent_verifier_validity_probability
                        for row in latent_verifier_rows
                    ),
                    "global_null_mass": next(iter(global_null_masses)),
                    "verifier_invalid_mass": next(iter(invalid_masses)),
                    "arithmetic_observation_counts": {
                        name: int(applied_arithmetic_counts.get(name, 0))
                        for name in ("pass", "fail", "missing")
                    },
                    "graph_observation_counts": {
                        name: int(applied_graph_counts.get(name, 0))
                        for name in ("pass", "fail", "missing")
                    },
                    "shuffled_trace_fraction": _mean_or_none(
                        row.latent_verifier_source_index != index
                        for index, row in enumerate(rows)
                    ),
                    "changed_validity_fraction": _mean_or_none(
                        abs(
                            float(row.latent_verifier_validity_probability)
                            - raw_validity
                        )
                        > 1e-12
                        for row, raw_validity in zip(
                            latent_verifier_rows,
                            raw_validity_probabilities,
                        )
                    ),
                }
            else:
                latent_verifier_summary = None
            question = {
                "pid": int(pid),
                "partition": partition,
                "trace_count": len(rows),
                "real_coverage": real_coverage,
                "null_mass": null_mass,
                "real_log_mean_evidence": real_log_mean_evidence,
                "insufficient_witness": bool(
                    rows
                    and all(
                        row.responsibility_insufficient_witness for row in rows
                    )
                ),
                "latent_verifier": latent_verifier_summary,
                "gold_mass": gold_mass,
                "gold_mass_before_potential": gold_mass_before_potential,
                "gold_mass_change_from_potential": gold_mass_change_from_potential,
                "responsibility_total_variation_from_potential": (
                    responsibility_total_variation_from_potential
                ),
                "gold_is_top1": gold_is_top1,
                "max_responsibility": max_responsibility,
                "top_one_two_margin": top_margin,
                "gini_coefficient": gini,
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "effective_sample_size": ess,
                "effective_sample_size_fraction": ess_fraction,
                "responsibility_temperature_used": _mean_or_none(
                    row.responsibility_temperature_used for row in rows
                ),
                "responsibility_weighted_age": weighted_age,
                "answer_correct_mass": answer_correct_mass,
                "calculation_path_clusters": calculation_path_clusters,
                "calculation_path_effective_sample_size": cluster_ess,
                "calculation_path_effective_sample_size_fraction": (
                    cluster_ess_fraction
                ),
                "responsibility_weighted_verifier_value": weighted_verifier_value,
                "weighted_joint_plus_entropy": weighted_joint_plus_entropy,
                "joint_support_logsumexp": joint_support_logsumexp,
                "nonfinite_responsibilities": nonfinite,
                "numeric_audited_model_traces": len(audited_model_rows),
                "structured_traces": len(structured_rows),
                "structured_well_formed_fraction": _mean_or_none(
                    row.structured_audit.well_formed for row, _value in structured_rows
                ),
                "structured_well_formed_mass": (
                    sum(
                        value
                        for row, value in structured_rows
                        if row.structured_audit.well_formed
                    )
                    if structured_rows and not nonfinite else None
                ),
                **numeric_metrics,
                "numeric_hard_rejected_traces": sum(
                    math.isinf(row.numeric_log_potential)
                    and row.numeric_log_potential < 0
                    for row, _value in audited_model_rows
                ),
                "numeric_graph_rejected_traces": sum(
                    math.isinf(row.numeric_log_potential)
                    and row.numeric_log_potential < 0
                    and row.numeric_audit.graph_fully_covered is False
                    for row, _value in audited_model_rows
                ),
                "equation_valid_mass": (
                    sum(
                        value
                        for row, value in audited_model_rows
                        if row.numeric_audit.parsed_equations > 0
                        and row.numeric_audit.invalid_equations == 0
                        and row.numeric_audit.gold_contradictions == 0
                    )
                    if not nonfinite else None
                ),
                "graph_valid_mass": (
                    sum(
                        value
                        for row, value in audited_model_rows
                        if row.numeric_audit.graph_fully_covered is True
                    )
                    if not nonfinite else None
                ),
                "correlations": {
                    "responsibility_vs_length_spearman": spearman_correlation(
                        values,
                        [float(row.span.sum().item()) for row in rows],
                    ),
                    "responsibility_vs_correctness_spearman": (
                        spearman_correlation(
                            values,
                            [
                                float(row.is_gold or row.proposal_correct is True)
                                for row in rows
                            ],
                        )
                        if any(
                            row.is_gold or row.proposal_correct is not None
                            for row in rows
                        )
                        else None
                    ),
                    "responsibility_vs_age_spearman": spearman_correlation(
                        values,
                        [
                            float(max(round_index - row.round_added, 0))
                            for row in rows
                        ],
                    ),
                    "responsibility_vs_trace_logprob_spearman": (
                        spearman_correlation(
                            [
                                value
                                for row, value in zip(rows, values)
                                if math.isfinite(row.trace_logprob)
                            ],
                            [
                                float(row.trace_logprob)
                                for row in rows
                                if math.isfinite(row.trace_logprob)
                            ],
                        )
                    ),
                    "responsibility_vs_verifier_value_spearman": (
                        spearman_correlation(
                            [
                                value
                                for row, value in zip(rows, values)
                                if row.verifier_value is not None
                            ],
                            [
                                float(row.verifier_value)
                                for row in rows
                                if row.verifier_value is not None
                            ],
                        )
                        if any(
                            row.verifier_value is not None for row in rows
                        )
                        else None
                    ),
                },
                "counterfactuals": _counterfactual_responsibility_diagnostics(rows),
            }
            questions.append(question)
            partition_questions.append(question)

            for index, (row, value) in enumerate(zip(rows, values)):
                audit = row.numeric_audit
                structured_audit = row.structured_audit
                latent_audit = row.latent_verifier_audit
                trace_record = {
                    "pid": int(pid),
                    "partition": partition,
                    "buffer_index": index,
                    "trace_id": row.trace_id,
                    "source": row.source,
                    "is_gold": bool(row.is_gold),
                    "proposal_correct": row.proposal_correct,
                    "proposal_tokens": row.proposal_tokens,
                    "calculation_path_signature": row.calculation_path_signature,
                    "round_added": int(row.round_added),
                    "age": max(round_index - row.round_added, 0),
                    "joint_logprob": _finite_or_none(row.joint_logprob),
                    "trace_logprob": _finite_or_none(row.trace_logprob),
                    "answer_logprob": _finite_or_none(row.answer_logprob),
                    "proposal_trace_logprob": _finite_or_none(
                        row.proposal_trace_logprob
                    ),
                    "log_importance_correction": _finite_or_none(
                        row.log_importance_correction
                    ),
                    "responsibility_logit": _finite_or_none(row.responsibility_logit),
                    "responsibility_before_potential": _finite_or_none(
                        row.responsibility_before_potential
                    ),
                    "responsibility_temperature_used": _finite_or_none(
                        row.responsibility_temperature_used
                    ),
                    "verifier_successes": row.verifier_successes,
                    "verifier_trials": row.verifier_trials,
                    "verifier_raw_rate": _finite_or_none(
                        row.verifier_raw_rate
                    ),
                    "verifier_value": _finite_or_none(row.verifier_value),
                    "verifier_policy": row.verifier_policy,
                    "verifier_generated_tokens": row.verifier_generated_tokens,
                    "verifier_outputs": (
                        list(row.verifier_outputs) if include_trace_tape else None
                    ),
                    "verifier_correct": (
                        list(row.verifier_correct) if include_trace_tape else None
                    ),
                    "responsibility": _finite_or_none(value),
                    "responsibility_real_coverage": _finite_or_none(
                        row.responsibility_real_coverage
                    ),
                    "responsibility_null_mass": _finite_or_none(
                        row.responsibility_null_mass
                    ),
                    "responsibility_real_log_mean_evidence": _finite_or_none(
                        row.responsibility_real_log_mean_evidence
                    ),
                    "responsibility_insufficient_witness": bool(
                        row.responsibility_insufficient_witness
                    ),
                    "latent_verifier": (
                        None
                        if latent_audit is None
                        else {
                            "mode": row.latent_verifier_mode,
                            "raw_rationale": latent_audit.raw_rationale,
                            "raw_arithmetic_observation": (
                                latent_audit.arithmetic_observation
                            ),
                            "raw_graph_observation": (
                                latent_audit.graph_observation
                            ),
                            "applied_arithmetic_observation": (
                                row.latent_verifier_applied_arithmetic
                            ),
                            "applied_graph_observation": (
                                row.latent_verifier_applied_graph
                            ),
                            "validity_probability": (
                                row.latent_verifier_validity_probability
                            ),
                            "source_buffer_index": (
                                row.latent_verifier_source_index
                            ),
                            "global_null_mass": (
                                row.latent_verifier_global_null_mass
                            ),
                            "verifier_invalid_mass": (
                                row.latent_verifier_invalid_mass
                            ),
                            "equation_mentions": latent_audit.equation_mentions,
                            "parsed_equations": latent_audit.parsed_equations,
                            "invalid_equations": latent_audit.invalid_equations,
                            "candidate_graph_nodes": (
                                latent_audit.candidate_graph_nodes
                            ),
                            "candidate_graph_edges": (
                                latent_audit.candidate_graph_edges
                            ),
                            "target_nodes": latent_audit.target_nodes,
                            "target_ancestor_nodes": (
                                latent_audit.target_ancestor_nodes
                            ),
                            "binary_operations": (
                                latent_audit.binary_operations
                            ),
                            "question_numbers": latent_audit.question_numbers,
                            "input_leaf_numbers": (
                                latent_audit.input_leaf_numbers
                            ),
                            "grounded_input_leaf_numbers": (
                                latent_audit.grounded_input_leaf_numbers
                            ),
                        }
                    ),
                    "rank": ranks[index],
                    "total_tokens": int(row.ids.numel()),
                    "objective_tokens": int(row.span.sum()),
                    "answer_tokens": int(row.ans.sum()),
                    "reasoning_token_count": row.reasoning_token_count,
                    "segment_responsibility_deltas": [
                        _finite_or_none(delta)
                        for delta in row.segment_responsibility_deltas
                    ],
                    "trace_representation": row.trace_representation,
                    "structured_audit": (
                        None if structured_audit is None else {
                            "has_calculation_block": structured_audit.has_calculation_block,
                            "has_reasoning_block": structured_audit.has_reasoning_block,
                            "calculation_precedes_reasoning": (
                                structured_audit.calculation_precedes_reasoning
                            ),
                            "well_formed": structured_audit.well_formed,
                            "calculation_characters": (
                                structured_audit.calculation_characters
                            ),
                            "reasoning_characters": structured_audit.reasoning_characters,
                        }
                    ),
                    "numeric_audit": None if audit is None else {
                        "equation_mentions": int(audit.equation_mentions),
                        "parsed_equations": int(audit.parsed_equations),
                        "parse_failures": int(
                            audit.equation_mentions - audit.parsed_equations
                        ),
                        "invalid_equations": int(audit.invalid_equations),
                        "gold_matches": int(audit.gold_matches),
                        "gold_contradictions": int(audit.gold_contradictions),
                        "gold_graph_available": bool(audit.gold_graph_available),
                        "gold_graph_nodes": int(audit.gold_graph_nodes),
                        "gold_graph_edges": int(audit.gold_graph_edges),
                        "candidate_graph_nodes": int(audit.candidate_graph_nodes),
                        "candidate_graph_edges": int(audit.candidate_graph_edges),
                        "graph_node_matches": int(audit.graph_node_matches),
                        "graph_edge_matches": int(audit.graph_edge_matches),
                        "graph_node_coverage": audit.graph_node_coverage,
                        "graph_edge_coverage": audit.graph_edge_coverage,
                        "graph_fully_covered": audit.graph_fully_covered,
                        "equations": list(audit.equations),
                        "invalid": list(audit.invalid),
                        "contradictions": list(audit.contradictions),
                        "missing_graph_nodes": list(audit.missing_graph_nodes),
                        "missing_graph_edges": list(audit.missing_graph_edges),
                        "log_potential": _finite_or_none(row.numeric_log_potential),
                        "hard_rejected": bool(
                            math.isinf(row.numeric_log_potential)
                            and row.numeric_log_potential < 0
                        ),
                    },
                }
                if row.sampled_support_prior_mass is not None:
                    trace_record.update({
                        "sampled_support_prior_mass": _finite_or_none(
                            row.sampled_support_prior_mass
                        ),
                        "sampled_support_log_marginal": _finite_or_none(
                            row.sampled_support_log_marginal
                        ),
                        "sampled_support_outer_initial": (
                            row.sampled_support_outer_initial
                        ),
                    })
                if include_trace_tape:
                    trace_record["trace_tape"] = {
                        "full_token_ids": [
                            int(token) for token in row.ids.detach().cpu().tolist()
                        ],
                        "objective_token_indices": [
                            int(index)
                            for index in row.span.detach().cpu().nonzero(
                                as_tuple=False
                            ).flatten().tolist()
                        ],
                        "answer_token_indices": [
                            int(index)
                            for index in row.ans.detach().cpu().nonzero(
                                as_tuple=False
                            ).flatten().tolist()
                        ],
                        "decoded_sequence": (
                            tok.decode(row.ids.detach().cpu().tolist())
                            if tok is not None else None
                        ),
                        "decoded_objective": (
                            tok.decode(
                                row.ids[row.span].detach().cpu().tolist()
                            )
                            if tok is not None else None
                        ),
                        "decoded_answer": (
                            tok.decode(
                                row.ids[row.ans].detach().cpu().tolist()
                            )
                            if tok is not None else None
                        ),
                    }
                traces.append(trace_record)

        counterfactual_summaries = {}
        for scheme in ("joint_tau1", "joint_tau2", "token_mean_tau1", "token_mean_tau2"):
            available = [
                question["counterfactuals"][scheme]
                for question in partition_questions
                if question["counterfactuals"][scheme].get("available")
            ]
            counterfactual_summaries[scheme] = {
                "question_count": len(available),
                "mean_gold_mass": _mean_or_none(item["gold_mass"] for item in available),
                "gold_top1_fraction": _mean_or_none(item["gold_is_top1"] for item in available),
                "mean_gold_rank": _mean_or_none(item["gold_rank"] for item in available),
                "mean_effective_sample_size_fraction": _mean_or_none(
                    item["effective_sample_size_fraction"] for item in available
                ),
                "top1_change_fraction_from_joint_tau1": _mean_or_none(
                    item["top1_changed_from_joint_tau1"] for item in available
                ),
                "mean_numeric_valid_mass": _mean_or_none(
                    item["numeric_valid_mass"] for item in available
                ),
                "mean_numeric_invalid_mass": _mean_or_none(
                    item["numeric_invalid_mass"] for item in available
                ),
                "mean_numeric_unparsed_mass": _mean_or_none(
                    item["numeric_unparsed_mass"] for item in available
                ),
                "mean_numeric_graph_available_mass": _mean_or_none(
                    item["numeric_graph_available_mass"] for item in available
                ),
                "mean_numeric_graph_compatible_mass": _mean_or_none(
                    item["numeric_graph_compatible_mass"] for item in available
                ),
                "mean_numeric_graph_incomplete_mass": _mean_or_none(
                    item["numeric_graph_incomplete_mass"] for item in available
                ),
                "mean_numeric_graph_unavailable_mass": _mean_or_none(
                    item["numeric_graph_unavailable_mass"] for item in available
                ),
                "mean_numeric_graph_node_coverage": _mean_or_none(
                    item["numeric_graph_node_coverage"] for item in available
                ),
                "mean_numeric_graph_edge_coverage": _mean_or_none(
                    item["numeric_graph_edge_coverage"] for item in available
                ),
            }

        partition_summaries[partition] = {
            "question_count": len(partition_questions),
            "mean_real_coverage": _mean_or_none(
                q["real_coverage"] for q in partition_questions
            ),
            "mean_null_mass": _mean_or_none(
                q["null_mass"] for q in partition_questions
            ),
            "mean_real_log_mean_evidence": _mean_or_none(
                q["real_log_mean_evidence"] for q in partition_questions
            ),
            "mean_trace_count": _mean_or_none(q["trace_count"] for q in partition_questions),
            "mean_gold_mass": _mean_or_none(q["gold_mass"] for q in partition_questions),
            "mean_gold_mass_before_potential": _mean_or_none(
                q["gold_mass_before_potential"] for q in partition_questions
            ),
            "mean_gold_mass_change_from_potential": _mean_or_none(
                q["gold_mass_change_from_potential"] for q in partition_questions
            ),
            "mean_responsibility_total_variation_from_potential": _mean_or_none(
                q["responsibility_total_variation_from_potential"]
                for q in partition_questions
            ),
            "gold_top1_fraction": _mean_or_none(q["gold_is_top1"] for q in partition_questions),
            "mean_max_responsibility": _mean_or_none(
                q["max_responsibility"] for q in partition_questions
            ),
            "mean_top_one_two_margin": _mean_or_none(
                q["top_one_two_margin"] for q in partition_questions
            ),
            "mean_gini_coefficient": _mean_or_none(
                q["gini_coefficient"] for q in partition_questions
            ),
            "mean_answer_correct_mass": _mean_or_none(
                q["answer_correct_mass"] for q in partition_questions
            ),
            "mean_weighted_joint_plus_entropy": _mean_or_none(
                q["weighted_joint_plus_entropy"] for q in partition_questions
            ),
            "mean_joint_support_logsumexp": _mean_or_none(
                q["joint_support_logsumexp"] for q in partition_questions
            ),
            "mean_effective_sample_size_fraction": _mean_or_none(
                q["effective_sample_size_fraction"] for q in partition_questions
            ),
            "mean_normalized_entropy": _mean_or_none(
                q["normalized_entropy"] for q in partition_questions
            ),
            "mean_responsibility_weighted_age": _mean_or_none(
                q["responsibility_weighted_age"] for q in partition_questions
            ),
            "nonfinite_responsibilities": sum(
                q["nonfinite_responsibilities"] for q in partition_questions
            ),
            "mean_numeric_valid_mass": _mean_or_none(
                q["numeric_valid_mass"] for q in partition_questions
            ),
            "mean_numeric_invalid_mass": _mean_or_none(
                q["numeric_invalid_mass"] for q in partition_questions
            ),
            "mean_numeric_unparsed_mass": _mean_or_none(
                q["numeric_unparsed_mass"] for q in partition_questions
            ),
            "mean_numeric_graph_available_mass": _mean_or_none(
                q["numeric_graph_available_mass"] for q in partition_questions
            ),
            "mean_numeric_graph_compatible_mass": _mean_or_none(
                q["numeric_graph_compatible_mass"] for q in partition_questions
            ),
            "mean_numeric_graph_incomplete_mass": _mean_or_none(
                q["numeric_graph_incomplete_mass"] for q in partition_questions
            ),
            "mean_numeric_graph_unavailable_mass": _mean_or_none(
                q["numeric_graph_unavailable_mass"] for q in partition_questions
            ),
            "mean_numeric_graph_node_coverage": _mean_or_none(
                q["numeric_graph_node_coverage"] for q in partition_questions
            ),
            "mean_numeric_graph_edge_coverage": _mean_or_none(
                q["numeric_graph_edge_coverage"] for q in partition_questions
            ),
            "mean_equation_valid_mass": _mean_or_none(
                q["equation_valid_mass"] for q in partition_questions
            ),
            "mean_graph_valid_mass": _mean_or_none(
                q["graph_valid_mass"] for q in partition_questions
            ),
            "numeric_hard_rejected_traces": sum(
                q["numeric_hard_rejected_traces"] for q in partition_questions
            ),
            "numeric_graph_rejected_traces": sum(
                q["numeric_graph_rejected_traces"] for q in partition_questions
            ),
            "structured_traces": sum(q["structured_traces"] for q in partition_questions),
            "mean_structured_well_formed_fraction": _mean_or_none(
                q["structured_well_formed_fraction"] for q in partition_questions
            ),
            "mean_structured_well_formed_mass": _mean_or_none(
                q["structured_well_formed_mass"] for q in partition_questions
            ),
            "correlations": {
                key: _mean_or_none(
                    question["correlations"][key]
                    for question in partition_questions
                )
                for key in (
                    "responsibility_vs_length_spearman",
                    "responsibility_vs_correctness_spearman",
                    "responsibility_vs_age_spearman",
                    "responsibility_vs_trace_logprob_spearman",
                    "responsibility_vs_verifier_value_spearman",
                )
            },
            "reader_calibration": binary_score_calibration(
                [
                    float(row.answer_logprob)
                    for pid in weights_by_pid
                    for row in buffers[int(pid)]
                    if math.isfinite(row.answer_logprob)
                    and (row.is_gold or row.proposal_correct is not None)
                ],
                [
                    bool(row.is_gold or row.proposal_correct is True)
                    for pid in weights_by_pid
                    for row in buffers[int(pid)]
                    if math.isfinite(row.answer_logprob)
                    and (row.is_gold or row.proposal_correct is not None)
                ],
            ),
            "counterfactuals": counterfactual_summaries,
        }

    return {
        "partitions": partition_summaries,
        "questions": questions,
        "traces": traces,
        "trace_tape": {
            "enabled": bool(include_trace_tape),
            "contains_full_token_ids": bool(include_trace_tape),
            "contains_per_token_proposal_logprobs": False,
            "contains_per_token_reader_logprobs": False,
        },
    }


def _optimizer_diagnostics(
    model,
    initial_parameters: list[torch.Tensor] | None = None,
) -> dict:
    """Measure trainable parameter, displacement, and gradient health."""

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if initial_parameters is not None and len(initial_parameters) != len(trainable):
        raise ValueError("initial parameter snapshot has the wrong length")
    parameter_sq = 0.0
    parameter_drift_sq = 0.0
    gradient_sq = 0.0
    parameter_nonfinite = 0
    gradient_nonfinite = 0
    gradient_tensors = 0
    with torch.no_grad():
        for parameter_index, parameter in enumerate(trainable):
            values = parameter.detach()
            parameter_sq += float(torch.sum(values.float() ** 2))
            if initial_parameters is not None:
                reference = initial_parameters[parameter_index]
                if reference.shape != values.shape:
                    raise ValueError(
                        "initial parameter snapshot has a mismatched shape"
                    )
                parameter_drift_sq += float(torch.sum(
                    (values.float() - reference.to(values.device).float()) ** 2
                ))
            parameter_nonfinite += int((~torch.isfinite(values)).sum())
            if parameter.grad is None:
                continue
            gradients = parameter.grad.detach()
            gradient_tensors += 1
            gradient_sq += float(torch.sum(gradients.float() ** 2))
            gradient_nonfinite += int((~torch.isfinite(gradients)).sum())
    return {
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "parameter_l2_norm": _finite_or_none(math.sqrt(parameter_sq)),
        "parameter_l2_drift_from_initial": (
            _finite_or_none(math.sqrt(parameter_drift_sq))
            if initial_parameters is not None else None
        ),
        "gradient_tensors": gradient_tensors,
        "gradient_l2_norm": _finite_or_none(math.sqrt(gradient_sq)),
        "nonfinite_parameter_values": parameter_nonfinite,
        "nonfinite_gradient_values": gradient_nonfinite,
    }


def _validate_ac_alg1_run_config(
    config: ACAlg1RunConfig,
    *,
    diagnostics_fn,
    diagnostics_probe_fn,
) -> tuple[ACAlg1RunConfig, bool]:
    algorithm_profile = config.algorithm_profile
    U_batch = config.U_batch
    length_norm = config.length_norm
    buffer_strategy = config.buffer_strategy
    buffer_semantics = config.buffer_semantics
    buffer_lifecycle = config.buffer_lifecycle
    buffer_max_age = config.buffer_max_age
    proposal_prompt = config.proposal_prompt
    labelled_proposal_prompt = config.labelled_proposal_prompt
    answer_only_proposal_prompt = config.answer_only_proposal_prompt
    proposal_mixture = config.proposal_mixture
    proposal_filter = config.proposal_filter
    proposal_policy = config.proposal_policy
    proposal_temperature = config.proposal_temperature
    proposal_allocation_mode = config.proposal_allocation_mode
    proposal_initial_traces = config.proposal_initial_traces
    proposal_allocation_max_traces = config.proposal_allocation_max_traces
    responsibility_score = config.responsibility_score
    responsibility_posterior = config.responsibility_posterior
    responsibility_temperature = config.responsibility_temperature
    responsibility_ess_floor = config.responsibility_ess_floor
    responsibility_abstention = config.responsibility_abstention
    responsibility_rejection_threshold = config.responsibility_rejection_threshold
    responsibility_null_log_evidence = config.responsibility_null_log_evidence
    responsibility_null_prior = config.responsibility_null_prior
    responsibility_policy = config.responsibility_policy
    responsibility_answer_policy = config.responsibility_answer_policy
    responsibility_refresh = config.responsibility_refresh
    responsibility_verifier_rollouts = config.responsibility_verifier_rollouts
    responsibility_verifier_temperature = (
        config.responsibility_verifier_temperature
    )
    responsibility_verifier_max_new_tokens = (
        config.responsibility_verifier_max_new_tokens
    )
    responsibility_verifier_batch_size = (
        config.responsibility_verifier_batch_size
    )
    responsibility_verifier_smoothing_alpha = (
        config.responsibility_verifier_smoothing_alpha
    )
    verifier_calibration_path = config.verifier_calibration_path
    reuse_fresh_traces = config.reuse_fresh_traces
    reuse_importance_min = config.reuse_importance_min
    reuse_importance_max = config.reuse_importance_max
    variational_estimator = config.variational_estimator
    labelled_em_weight = config.labelled_em_weight
    answer_only_em_weight = config.answer_only_em_weight
    policy_kl_coef = config.policy_kl_coef
    supervised_weight = config.supervised_weight
    policy_anchor_mode = config.policy_anchor_mode
    policy_anchor_target_ratio = config.policy_anchor_target_ratio
    policy_anchor_beta_min = config.policy_anchor_beta_min
    policy_anchor_beta_max = config.policy_anchor_beta_max
    policy_anchor_ema = config.policy_anchor_ema
    policy_anchor_token_scope = config.policy_anchor_token_scope
    labelled_numeric_constraint = config.labelled_numeric_constraint
    numeric_penalty = config.numeric_penalty
    numeric_contradiction_penalty = config.numeric_contradiction_penalty
    numeric_missing_penalty = config.numeric_missing_penalty
    labelled_supervision = config.labelled_supervision
    compact_gold_weight = config.compact_gold_weight
    digit_token_weight = config.digit_token_weight
    trace_representation = config.trace_representation
    latent_mstep_objective = config.latent_mstep_objective
    answer_event_mode = config.answer_event_mode
    answer_target_termination = config.answer_target_termination
    update_geometry = config.update_geometry
    step_acceptance = config.step_acceptance
    rollback_tolerance = config.rollback_tolerance
    rollback_max_backtracks = config.rollback_max_backtracks
    rollback_shrink = config.rollback_shrink
    optimizer_state_scope = config.optimizer_state_scope
    diagnostics_level = config.diagnostics_level
    diagnostics_trace_tape = config.diagnostics_trace_tape
    diagnostics_gradient_questions = config.diagnostics_gradient_questions
    checkpoint_every = config.checkpoint_every

    if algorithm_profile not in ALGORITHM_PROFILES:
        raise ValueError(f"unknown AC-ALG1 algorithm_profile {algorithm_profile!r}")
    if length_norm:
        raise ValueError("AC-ALG1 is faithful only with length_norm=False")
    if buffer_strategy not in ("fifo", "hybrid", "calculation_diverse"):
        raise ValueError(f"unknown AC-ALG1 buffer_strategy {buffer_strategy!r}")
    if buffer_semantics not in BUFFER_SEMANTICS:
        raise ValueError(f"unknown AC-ALG1 buffer_semantics {buffer_semantics!r}")
    if buffer_lifecycle not in BUFFER_LIFECYCLES:
        raise ValueError(f"unknown AC-ALG1 buffer_lifecycle {buffer_lifecycle!r}")
    if int(buffer_max_age) < -1:
        raise ValueError("buffer_max_age must be -1 or nonnegative")
    if proposal_prompt not in PROPOSAL_PROMPTS:
        raise ValueError(f"unknown AC-ALG1 proposal_prompt {proposal_prompt!r}")
    if not math.isfinite(proposal_temperature) or proposal_temperature <= 0:
        raise ValueError(
            "proposal_temperature must be finite and positive, got "
            f"{proposal_temperature}"
        )
    labelled_proposal_prompt = labelled_proposal_prompt or proposal_prompt
    answer_only_proposal_prompt = answer_only_proposal_prompt or proposal_prompt
    if labelled_proposal_prompt not in PROPOSAL_PROMPTS:
        raise ValueError(
            "unknown AC-ALG1 labelled_proposal_prompt "
            f"{labelled_proposal_prompt!r}"
        )
    if answer_only_proposal_prompt not in PROPOSAL_PROMPTS:
        raise ValueError(
            "unknown AC-ALG1 answer_only_proposal_prompt "
            f"{answer_only_proposal_prompt!r}"
        )
    if answer_only_proposal_prompt in (
        "tagged_gold_rationale",
        "answer_graph_derive",
    ):
        raise ValueError(
            "answer_only_proposal_prompt cannot expose a gold rationale or graph to U'"
        )
    if proposal_mixture not in PROPOSAL_MIXTURES:
        raise ValueError(f"unknown AC-ALG1 proposal_mixture {proposal_mixture!r}")
    _proposal_components(
        labelled_proposal_prompt,
        proposal_mixture,
        "labelled_sample",
    )
    _proposal_components(
        answer_only_proposal_prompt,
        proposal_mixture,
        "answer_only_sample",
    )
    if proposal_filter not in PROPOSAL_FILTERS:
        raise ValueError(f"unknown AC-ALG1 proposal_filter {proposal_filter!r}")
    if proposal_policy not in ADAPTER_POLICY_MODES:
        raise ValueError(f"unknown AC-ALG1 proposal_policy {proposal_policy!r}")
    if proposal_allocation_mode not in PROPOSAL_ALLOCATION_MODES:
        raise ValueError(
            "unknown AC-ALG1 proposal_allocation_mode "
            f"{proposal_allocation_mode!r}"
        )
    if int(proposal_initial_traces) < 0:
        raise ValueError("proposal_initial_traces must be nonnegative")
    if int(proposal_allocation_max_traces) < 0:
        raise ValueError("proposal_allocation_max_traces must be nonnegative")
    if int(reuse_fresh_traces) < 0:
        raise ValueError("reuse_fresh_traces must be nonnegative")
    if (
        not math.isfinite(reuse_importance_min)
        or not math.isfinite(reuse_importance_max)
        or reuse_importance_min <= 0.0
        or reuse_importance_max < reuse_importance_min
    ):
        raise ValueError(
            "reuse importance bounds must be finite, positive, and ordered"
        )
    if algorithm_profile != "l2r_uncertainty_allocation_pilot" and (
        proposal_allocation_mode != "uniform"
        or proposal_initial_traces != 0
        or proposal_allocation_max_traces != 0
    ):
        raise ValueError(
            "adaptive proposal allocation is isolated to its registered pilot"
        )
    if algorithm_profile not in {
        "l2r_age_one_reuse_pilot",
        "l2r_small_group_replay_pilot",
    } and (
        buffer_max_age != -1
        or reuse_fresh_traces != 0
        or reuse_importance_min != 0.5
        or reuse_importance_max != 2.0
    ):
        raise ValueError("age-one reuse controls are isolated to their pilot")
    if algorithm_profile != "l2r_bayesian_fusion_pilot" and (
        verifier_calibration_path is not None
    ):
        raise ValueError(
            "Bayesian verifier calibration is isolated to its registered pilot"
        )
    if responsibility_score not in RESPONSIBILITY_SCORES:
        raise ValueError(f"unknown AC-ALG1 responsibility_score {responsibility_score!r}")
    if responsibility_posterior not in RESPONSIBILITY_POSTERIORS:
        raise ValueError(
            "unknown AC-ALG1 responsibility_posterior "
            f"{responsibility_posterior!r}"
        )
    if (
        responsibility_posterior == "hard_delta_no_entropy"
        and responsibility_ess_floor != 0.0
    ):
        raise ValueError(
            "hard_delta_no_entropy requires responsibility_ess_floor=0"
        )
    if not math.isfinite(responsibility_temperature) or responsibility_temperature <= 0:
        raise ValueError(
            "responsibility_temperature must be finite and positive, "
            f"got {responsibility_temperature}"
        )
    if (
        not math.isfinite(responsibility_ess_floor)
        or not 0.0 <= responsibility_ess_floor <= 1.0
    ):
        raise ValueError(
            "responsibility_ess_floor must be finite and in [0, 1], "
            f"got {responsibility_ess_floor}"
        )
    if responsibility_abstention not in RESPONSIBILITY_ABSTENTION_MODES:
        raise ValueError(
            "unknown AC-ALG1 responsibility_abstention "
            f"{responsibility_abstention!r}"
        )
    if not math.isfinite(responsibility_rejection_threshold):
        raise ValueError("responsibility_rejection_threshold must be finite")
    if not math.isfinite(responsibility_null_log_evidence):
        raise ValueError("responsibility_null_log_evidence must be finite")
    if (
        not math.isfinite(responsibility_null_prior)
        or not 0.0 < responsibility_null_prior < 1.0
    ):
        raise ValueError(
            "responsibility_null_prior must be finite and strictly between zero and one"
        )
    if algorithm_profile not in {
        "l2r_abstention_pilot",
        "l2r_multi_verifier_pilot",
        "l2r_bayesian_fusion_pilot",
        "l2r_uncertainty_allocation_pilot",
        "l2r_age_one_reuse_pilot",
        "l2r_small_group_replay_pilot",
    } and (
        responsibility_abstention != "none"
        or responsibility_rejection_threshold != 0.0
        or responsibility_null_log_evidence != 0.0
        or responsibility_null_prior != 0.5
    ):
        raise ValueError(
            f"{algorithm_profile} forbids responsibility abstention controls"
        )
    if responsibility_policy not in ADAPTER_POLICY_MODES:
        raise ValueError(
            f"unknown AC-ALG1 responsibility_policy {responsibility_policy!r}"
        )
    if responsibility_answer_policy not in ADAPTER_POLICY_MODES:
        raise ValueError(
            "unknown AC-ALG1 responsibility_answer_policy "
            f"{responsibility_answer_policy!r}"
        )
    if responsibility_refresh not in RESPONSIBILITY_REFRESH_MODES:
        raise ValueError(
            "unknown AC-ALG1 responsibility_refresh "
            f"{responsibility_refresh!r}"
        )
    if int(responsibility_verifier_rollouts) < 0:
        raise ValueError(
            "responsibility_verifier_rollouts must be nonnegative, got "
            f"{responsibility_verifier_rollouts}"
        )
    for name, value in (
        (
            "responsibility_verifier_max_new_tokens",
            responsibility_verifier_max_new_tokens,
        ),
        (
            "responsibility_verifier_batch_size",
            responsibility_verifier_batch_size,
        ),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    for name, value in (
        (
            "responsibility_verifier_temperature",
            responsibility_verifier_temperature,
        ),
        (
            "responsibility_verifier_smoothing_alpha",
            responsibility_verifier_smoothing_alpha,
        ),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
    if responsibility_score == "rollout_value":
        if responsibility_refresh != "outer_round":
            raise ValueError(
                "rollout_value requires responsibility_refresh='outer_round' "
                "so each generalized E-step uses one fixed Monte Carlo estimate"
            )
        if variational_estimator not in {"delta_joint", "prior_importance"}:
            raise ValueError(
                "rollout_value supports only delta_joint or prior_importance"
            )
        if responsibility_verifier_rollouts < 1:
            raise ValueError(
                "responsibility_verifier_rollouts must be positive for "
                "rollout_value"
            )
    elif responsibility_verifier_rollouts != 0:
        raise ValueError(
            "responsibility_verifier_rollouts requires "
            "responsibility_score='rollout_value'"
        )
    if variational_estimator not in VARIATIONAL_ESTIMATORS:
        raise ValueError(
            f"unknown AC-ALG1 variational_estimator {variational_estimator!r}"
        )
    if variational_estimator == "sampled_support_importance" and (
        algorithm_profile != "l2r_exact_signed_factorial"
        or latent_mstep_objective != "exact_signed_trace_answer"
    ):
        raise ValueError(
            "sampled_support_importance is isolated to the exact signed factorial"
        )
    if (
        responsibility_score == "token_mean"
        and variational_estimator != "delta_joint"
    ):
        raise ValueError(
            "token_mean responsibilities require variational_estimator='delta_joint'"
        )
    if (
        variational_estimator != "delta_joint"
        and responsibility_posterior
        not in {
            "softmax_entropy",
            "two_witness",
            *MULTI_VERIFIER_POSTERIORS,
            "verifier_bayesian",
        }
    ):
        raise ValueError(
            "Monte Carlo and importance estimators require softmax_entropy or "
            "a registered derived posterior"
        )
    if responsibility_posterior == "two_witness" and (
        algorithm_profile != "l2r_two_witness_pilot"
        or variational_estimator != "prior_importance"
        or responsibility_temperature != 1.0
        or responsibility_ess_floor != 0.0
        or responsibility_abstention != "none"
    ):
        raise ValueError(
            "two_witness is isolated to the registered prior-importance pilot "
            "with unit temperature, no ESS floor, and no abstention"
        )
    if responsibility_posterior in MULTI_VERIFIER_POSTERIORS and (
        algorithm_profile != "l2r_multi_verifier_pilot"
        or variational_estimator != "prior_importance"
        or responsibility_temperature != 1.0
        or responsibility_ess_floor != 0.0
        or responsibility_abstention != "null_latent"
    ):
        raise ValueError(
            "multi-verifier posteriors are isolated to the registered "
            "prior-importance pilot with unit temperature, no ESS floor, and "
            "the frozen null latent"
        )
    if responsibility_posterior == "verifier_bayesian" and (
        algorithm_profile != "l2r_bayesian_fusion_pilot"
        or variational_estimator != "prior_importance"
        or responsibility_temperature != 1.0
        or responsibility_ess_floor != 0.0
        or responsibility_abstention != "null_latent"
    ):
        raise ValueError(
            "verifier_bayesian is isolated to its registered prior-importance "
            "pilot with the frozen null latent"
        )
    if variational_estimator in {
        "uniform_mc",
        "prior_importance",
        "answer_conditioned_importance",
        "sampled_support_importance",
    }:
        if buffer_lifecycle not in {"fresh_round", "fixed_bank"}:
            raise ValueError(
                f"{variational_estimator} requires fresh_round or fixed_bank support"
            )
        if buffer_semantics != "multiset_legacy":
            raise ValueError(
                f"{variational_estimator} requires empirical multiset semantics"
            )
    if variational_estimator == "prior_importance":
        if proposal_policy != "current":
            raise ValueError("prior_importance requires current-prior proposals")
        if (
            labelled_proposal_prompt != "question"
            or answer_only_proposal_prompt != "question"
        ):
            raise ValueError(
                "prior_importance requires question-only prior proposals"
            )
    if variational_estimator == "sampled_support_importance":
        if buffer_lifecycle != "fresh_round":
            raise ValueError(
                "sampled_support_importance requires fresh behaviour support"
            )
        if buffer_semantics != "multiset_legacy":
            raise ValueError(
                "sampled_support_importance requires empirical multiset semantics"
            )
        if proposal_policy != "current":
            raise ValueError(
                "sampled_support_importance requires current-policy proposals"
            )
        if responsibility_policy != "current" or responsibility_answer_policy != "current":
            raise ValueError(
                "sampled_support_importance requires current numerator policies"
            )
        if (
            labelled_proposal_prompt != "question"
            or answer_only_proposal_prompt != "question"
        ):
            raise ValueError(
                "sampled_support_importance requires question-only proposals"
            )
        if (
            responsibility_score != "joint"
            or responsibility_posterior != "softmax_entropy"
            or responsibility_temperature != 1.0
            or responsibility_ess_floor != 0.0
            or responsibility_abstention != "none"
        ):
            raise ValueError(
                "sampled_support_importance requires the unprojected unit-temperature "
                "joint posterior without abstention"
            )
    if variational_estimator == "frozen_prior_importance":
        if buffer_lifecycle != "fresh_round":
            raise ValueError(
                "frozen_prior_importance requires fresh frozen-prior draws"
            )
        if buffer_semantics != "multiset_legacy":
            raise ValueError(
                "frozen_prior_importance requires empirical multiset semantics"
            )
        if proposal_policy != "frozen_base":
            raise ValueError(
                "frozen_prior_importance requires frozen_base proposals"
            )
        if (
            responsibility_policy != "current"
            or responsibility_answer_policy != "current"
        ):
            raise ValueError(
                "frozen_prior_importance requires current numerator policies"
            )
        if (
            labelled_proposal_prompt != "question"
            or answer_only_proposal_prompt != "question"
        ):
            raise ValueError(
                "frozen_prior_importance requires question-only prior proposals"
            )
    if variational_estimator == "answer_conditioned_importance":
        if buffer_lifecycle != "fresh_round":
            raise ValueError(
                "answer_conditioned_importance requires fresh proposal draws"
            )
        if buffer_semantics != "multiset_legacy":
            raise ValueError(
                "answer_conditioned_importance requires empirical multiset semantics"
            )
        if proposal_policy != "current":
            raise ValueError(
                "answer_conditioned_importance requires current-policy proposals"
            )
        if (
            responsibility_policy != "current"
            or responsibility_answer_policy != "current"
        ):
            raise ValueError(
                "answer_conditioned_importance requires current numerator policies"
            )
        if (
            labelled_proposal_prompt != "answer_derive"
            or answer_only_proposal_prompt != "answer_derive"
        ):
            raise ValueError(
                "answer_conditioned_importance requires answer_derive proposals"
            )
    if variational_estimator == "persistent_answer_conditioned_importance":
        if buffer_lifecycle != "persistent":
            raise ValueError(
                "persistent_answer_conditioned_importance requires persistent "
                "proposal replay"
            )
        if buffer_semantics != "multiset_legacy":
            raise ValueError(
                "persistent_answer_conditioned_importance requires empirical "
                "multiset semantics"
            )
        if proposal_policy != "current":
            raise ValueError(
                "persistent_answer_conditioned_importance requires current-policy "
                "proposals"
            )
        if (
            responsibility_policy != "current"
            or responsibility_answer_policy != "current"
        ):
            raise ValueError(
                "persistent_answer_conditioned_importance requires current "
                "numerator policies"
            )
        if (
            labelled_proposal_prompt != "answer_derive"
            or answer_only_proposal_prompt != "answer_derive"
        ):
            raise ValueError(
                "persistent_answer_conditioned_importance requires answer_derive "
                "proposals"
            )
    if variational_estimator == "persistent_prior_importance":
        if buffer_lifecycle != "persistent":
            raise ValueError(
                "persistent_prior_importance requires persistent proposal replay"
            )
        if buffer_semantics != "multiset_legacy":
            raise ValueError(
                "persistent_prior_importance requires empirical multiset semantics"
            )
        if proposal_policy != "current":
            raise ValueError(
                "persistent_prior_importance requires current-policy proposals"
            )
        if (
            responsibility_policy != "current"
            or responsibility_answer_policy != "current"
        ):
            raise ValueError(
                "persistent_prior_importance requires current numerator policies"
            )
        if (
            labelled_proposal_prompt != "question"
            or answer_only_proposal_prompt != "question"
        ):
            raise ValueError(
                "persistent_prior_importance requires question-only proposals"
            )
    if buffer_lifecycle == "fixed_bank" and proposal_policy != "frozen_base":
        raise ValueError("fixed_bank requires frozen_base proposal sampling")
    if not math.isfinite(labelled_em_weight) or labelled_em_weight < 0:
        raise ValueError(
            f"labelled_em_weight must be finite and nonnegative, got {labelled_em_weight}"
        )
    if not math.isfinite(answer_only_em_weight) or answer_only_em_weight < 0:
        raise ValueError(
            "answer_only_em_weight must be finite and nonnegative, "
            f"got {answer_only_em_weight}"
        )
    if not math.isfinite(supervised_weight) or supervised_weight < 0:
        raise ValueError(
            f"supervised_weight must be finite and nonnegative, got {supervised_weight}"
        )
    if policy_kl_coef is not None and (
        not math.isfinite(policy_kl_coef) or policy_kl_coef < 0
    ):
        raise ValueError(
            "policy_kl_coef must be finite and nonnegative when supplied, "
            f"got {policy_kl_coef}"
        )
    if policy_anchor_mode not in POLICY_ANCHOR_MODES:
        raise ValueError(f"unknown AC-ALG1 policy_anchor_mode {policy_anchor_mode!r}")
    if policy_anchor_token_scope not in POLICY_ANCHOR_TOKEN_SCOPES:
        raise ValueError(
            "unknown AC-ALG1 policy_anchor_token_scope "
            f"{policy_anchor_token_scope!r}"
        )
    if policy_anchor_token_scope == "reasoning" and policy_anchor_mode != "grad_ratio":
        raise ValueError(
            "policy_anchor_token_scope='reasoning' requires "
            "policy_anchor_mode='grad_ratio'"
        )
    if (
        policy_anchor_token_scope == "reasoning"
        and algorithm_profile not in {
            "barber_q5_control",
            "l2r_pis_rationale_kl_followup",
        }
    ):
        raise ValueError(
            "reasoning-only AC-ALG1 anchoring is isolated to registered Q5/PIS profiles"
        )
    if policy_anchor_mode == "fixed":
        if policy_anchor_target_ratio is not None:
            raise ValueError(
                "policy_anchor_target_ratio is only valid when "
                "policy_anchor_mode='grad_ratio'"
            )
    else:
        if policy_kl_coef is not None:
            raise ValueError(
                "policy_kl_coef and policy_anchor_mode='grad_ratio' are mutually exclusive"
            )
        if policy_anchor_target_ratio is None or (
            not math.isfinite(policy_anchor_target_ratio)
            or policy_anchor_target_ratio < 0
        ):
            raise ValueError(
                "policy_anchor_target_ratio must be finite and nonnegative in "
                "grad_ratio mode"
            )
    for name, value in (
        ("policy_anchor_beta_min", policy_anchor_beta_min),
        ("policy_anchor_beta_max", policy_anchor_beta_max),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    if policy_anchor_beta_min > policy_anchor_beta_max:
        raise ValueError("policy_anchor_beta_min cannot exceed policy_anchor_beta_max")
    if not math.isfinite(policy_anchor_ema) or not 0 <= policy_anchor_ema < 1:
        raise ValueError(
            f"policy_anchor_ema must be finite and in [0,1), got {policy_anchor_ema}"
        )
    if labelled_numeric_constraint not in LABELLED_NUMERIC_CONSTRAINTS:
        raise ValueError(
            "unknown AC-ALG1 labelled_numeric_constraint "
            f"{labelled_numeric_constraint!r}"
        )
    if not math.isfinite(numeric_penalty) or numeric_penalty < 0:
        raise ValueError(f"numeric_penalty must be finite and nonnegative, got {numeric_penalty}")
    for name, value in (
        ("numeric_contradiction_penalty", numeric_contradiction_penalty),
        ("numeric_missing_penalty", numeric_missing_penalty),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    if labelled_supervision not in LABELLED_SUPERVISION_MODES:
        raise ValueError(f"unknown AC-ALG1 labelled_supervision {labelled_supervision!r}")
    policy_anchor_measured = (
        policy_kl_coef is not None or policy_anchor_mode == "grad_ratio"
    )
    if update_geometry not in UPDATE_GEOMETRIES:
        raise ValueError(f"unknown AC-ALG1 update_geometry {update_geometry!r}")
    if step_acceptance not in STEP_ACCEPTANCE_MODES:
        raise ValueError(f"unknown AC-ALG1 step_acceptance {step_acceptance!r}")
    if not math.isfinite(rollback_tolerance) or rollback_tolerance < 0:
        raise ValueError(
            "rollback_tolerance must be finite and nonnegative, "
            f"got {rollback_tolerance}"
        )
    if rollback_max_backtracks < 0:
        raise ValueError(
            "rollback_max_backtracks must be nonnegative, "
            f"got {rollback_max_backtracks}"
        )
    if not math.isfinite(rollback_shrink) or not 0 < rollback_shrink < 1:
        raise ValueError(
            "rollback_shrink must be finite and strictly between zero and one, "
            f"got {rollback_shrink}"
        )
    if step_acceptance == "none" and rollback_max_backtracks:
        raise ValueError(
            "rollback_max_backtracks requires an active step_acceptance rule"
        )
    if optimizer_state_scope not in OPTIMIZER_STATE_SCOPES:
        raise ValueError(
            "unknown AC-ALG1 optimizer_state_scope "
            f"{optimizer_state_scope!r}"
        )
    if (
        update_geometry != "sum" or step_acceptance != "none"
    ) and policy_anchor_measured:
        raise ValueError(
            "safeguarded update geometry and policy anchoring are isolated "
            "experimental factors and cannot be enabled in the same cell"
        )
    if (
        update_geometry == "answer_primary"
        or step_acceptance == "answer_primary"
    ) and (answer_only_em_weight <= 0 or U_batch <= 0):
        raise ValueError(
            "answer-primary updates require a nonzero B_unsup term and "
            "at least one answer-only question per round"
        )
    if policy_anchor_measured and labelled_supervision not in (
        "gold",
        "gold_graph_factorized",
    ):
        raise ValueError(
            "policy anchoring currently requires gold or gold_graph_factorized "
            "labelled supervision so its target distribution is unambiguous"
        )
    if not math.isfinite(compact_gold_weight) or not 0.0 <= compact_gold_weight <= 1.0:
        raise ValueError(
            "compact_gold_weight must be finite and in [0,1], "
            f"got {compact_gold_weight}"
        )
    if not math.isfinite(digit_token_weight) or digit_token_weight < 1.0:
        raise ValueError(
            "digit_token_weight must be finite and at least 1, "
            f"got {digit_token_weight}"
        )
    if trace_representation not in TRACE_REPRESENTATIONS:
        raise ValueError(f"unknown AC-ALG1 trace_representation {trace_representation!r}")
    if latent_mstep_objective not in LATENT_MSTEP_OBJECTIVES:
        raise ValueError(
            f"unknown AC-ALG1 latent_mstep_objective {latent_mstep_objective!r}"
        )
    if latent_mstep_objective == "exact_signed_trace_answer" and (
        algorithm_profile != "l2r_exact_signed_factorial"
        or variational_estimator != "sampled_support_importance"
    ):
        raise ValueError(
            "exact_signed_trace_answer is isolated to sampled_support_importance"
        )
    if answer_event_mode not in ANSWER_EVENT_MODES:
        raise ValueError(f"unknown answer event mode {answer_event_mode!r}")
    if answer_target_termination not in ANSWER_TARGET_TERMINATIONS:
        raise ValueError(
            "unknown answer target termination "
            f"{answer_target_termination!r}"
        )
    graph_supervision = labelled_supervision == "gold_graph_factorized"
    graph_representation = trace_representation == "calculation_graph"
    if graph_supervision != graph_representation:
        raise ValueError(
            "gold_graph_factorized supervision and calculation_graph trace_representation "
            "must be enabled together"
        )
    if graph_representation and (
        labelled_proposal_prompt.startswith("tagged_")
        or answer_only_proposal_prompt.startswith("tagged_")
    ):
        raise ValueError(
            "calculation_graph traces cannot be combined with tagged proposal prompts"
        )
    if checkpoint_every < 0:
        raise ValueError(f"checkpoint_every must be nonnegative, got {checkpoint_every}")
    validate_diagnostic_level(diagnostics_level)
    if diagnostics_gradient_questions < 0:
        raise ValueError("diagnostics_gradient_questions must be nonnegative")
    if diagnostics_gradient_questions and diagnostics_level != "deep":
        raise ValueError(
            "diagnostics_gradient_questions requires diagnostics_level='deep'"
        )
    if diagnostics_probe_fn is not None and diagnostics_level != "deep":
        raise ValueError("diagnostics_probe_fn requires diagnostics_level='deep'")
    if (
        diagnostics_fn is None
        and (
            diagnostics_level != "standard"
            or diagnostics_trace_tape
            or diagnostics_gradient_questions
            or diagnostics_probe_fn is not None
        )
    ):
        raise ValueError(
            "deep diagnostic options require save_training_diagnostics"
        )

    if algorithm_profile == "l2r_common_factorial":
        common_requirements = {
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "answer_target_termination": (answer_target_termination, "eos"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "labelled_proposal_prompt": (
                labelled_proposal_prompt,
                "question",
            ),
            "answer_only_proposal_prompt": (
                answer_only_proposal_prompt,
                "question",
            ),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_filter": (proposal_filter, "all"),
            "proposal_policy": (proposal_policy, "current"),
            "responsibility_score": (responsibility_score, "joint"),
            "responsibility_posterior": (
                responsibility_posterior,
                "softmax_entropy",
            ),
            "responsibility_temperature": (
                responsibility_temperature,
                1.0,
            ),
            "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_answer_policy": (
                responsibility_answer_policy,
                "current",
            ),
            "responsibility_verifier_rollouts": (
                responsibility_verifier_rollouts,
                0,
            ),
            "labelled_numeric_constraint": (
                labelled_numeric_constraint,
                "off",
            ),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "latent_mstep_objective": (latent_mstep_objective, "joint"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (
                optimizer_state_scope,
                "persistent",
            ),
            "question_sampling": (
                config.question_sampling,
                "epoch_shuffle",
            ),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in common_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                "l2r_common_factorial rejected an undeclared algorithm "
                f"change: {mismatches}"
            )
        if policy_kl_coef is not None or policy_anchor_mode != "fixed":
            raise ValueError("l2r_common_factorial forbids policy anchoring")
        if variational_estimator == "delta_joint":
            if (
                buffer_lifecycle != "persistent"
                or buffer_semantics != "unique_set"
            ):
                raise ValueError(
                    "l2r_common_factorial delta_joint requires a persistent "
                    "token-unique set"
                )
            if responsibility_refresh not in {"inner_step", "outer_round"}:
                raise ValueError(
                    "l2r_common_factorial delta_joint requires inner_step or "
                    "outer_round responsibility refresh"
                )
        elif variational_estimator in {"uniform_mc", "prior_importance"}:
            if (
                buffer_lifecycle != "fresh_round"
                or buffer_semantics != "multiset_legacy"
            ):
                raise ValueError(
                    "l2r_common_factorial Monte Carlo estimators require a "
                    "fresh empirical multiset"
                )
            if responsibility_refresh != "outer_round":
                raise ValueError(
                    "l2r_common_factorial Monte Carlo estimators require a "
                    "fixed outer-round E-step"
                )
        else:
            raise ValueError(
                "l2r_common_factorial permits only delta_joint, uniform_mc, "
                "or prior_importance"
            )
    elif algorithm_profile == "l2r_answer_conditioned_importance":
        importance_requirements = {
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "answer_target_termination": (answer_target_termination, "eos"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "buffer_semantics": (buffer_semantics, "multiset_legacy"),
            "buffer_lifecycle": (buffer_lifecycle, "fresh_round"),
            "labelled_proposal_prompt": (
                labelled_proposal_prompt,
                "answer_derive",
            ),
            "answer_only_proposal_prompt": (
                answer_only_proposal_prompt,
                "answer_derive",
            ),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_filter": (proposal_filter, "all"),
            "proposal_policy": (proposal_policy, "current"),
            "responsibility_score": (responsibility_score, "joint"),
            "responsibility_posterior": (
                responsibility_posterior,
                "softmax_entropy",
            ),
            "responsibility_temperature": (
                responsibility_temperature,
                1.0,
            ),
            "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_answer_policy": (
                responsibility_answer_policy,
                "current",
            ),
            "responsibility_refresh": (
                responsibility_refresh,
                "outer_round",
            ),
            "responsibility_verifier_rollouts": (
                responsibility_verifier_rollouts,
                0,
            ),
            "variational_estimator": (
                variational_estimator,
                "answer_conditioned_importance",
            ),
            "labelled_numeric_constraint": (
                labelled_numeric_constraint,
                "off",
            ),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "latent_mstep_objective": (latent_mstep_objective, "joint"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (
                optimizer_state_scope,
                "persistent",
            ),
            "question_sampling": (
                config.question_sampling,
                "epoch_shuffle",
            ),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in importance_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                "l2r_answer_conditioned_importance rejected an undeclared "
                f"algorithm change: {mismatches}"
            )
        if policy_kl_coef is not None or policy_anchor_mode != "fixed":
            raise ValueError(
                "l2r_answer_conditioned_importance forbids policy anchoring"
            )
    elif algorithm_profile == "l2r_curated_buffer_pilot":
        curated_requirements = {
            "rounds": (config.rounds, 32),
            "L_batch": (config.L_batch, 0),
            "U_batch": (U_batch, 8),
            "G_label": (config.G_label, config.G_answer_only),
            "inner_steps": (config.inner_steps, {1, 4}),
            "buffer_limit": (config.buffer_limit, 8),
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "answer_target_termination": (answer_target_termination, "eos"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "buffer_semantics": (buffer_semantics, "multiset_legacy"),
            "buffer_lifecycle": (buffer_lifecycle, "fresh_round"),
            "labelled_proposal_prompt": (labelled_proposal_prompt, "question"),
            "answer_only_proposal_prompt": (
                answer_only_proposal_prompt,
                "question",
            ),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_policy": (proposal_policy, "current"),
            "responsibility_score": (responsibility_score, "joint"),
            "responsibility_posterior": (
                responsibility_posterior,
                "softmax_entropy",
            ),
            "responsibility_temperature": (responsibility_temperature, 1.0),
            "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
            "responsibility_abstention": (responsibility_abstention, "none"),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_answer_policy": (
                responsibility_answer_policy,
                "current",
            ),
            "responsibility_refresh": (responsibility_refresh, "outer_round"),
            "responsibility_verifier_rollouts": (
                responsibility_verifier_rollouts,
                0,
            ),
            "labelled_numeric_constraint": (labelled_numeric_constraint, "off"),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "latent_mstep_objective": (latent_mstep_objective, "joint"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (optimizer_state_scope, "persistent"),
            "question_sampling": (config.question_sampling, "epoch_shuffle"),
            "diagnostics_level": (diagnostics_level, "standard"),
            "diagnostics_trace_tape": (diagnostics_trace_tape, True),
            "diagnostics_gradient_questions": (
                diagnostics_gradient_questions,
                0,
            ),
            "checkpoint_every": (checkpoint_every, 0),
        }
        mismatches = {}
        for name, (actual, required) in curated_requirements.items():
            if isinstance(required, set):
                matches = actual in required
            else:
                matches = actual == required
            if not matches:
                mismatches[name] = {"actual": actual, "required": required}
        if mismatches:
            raise ValueError(
                "l2r_curated_buffer_pilot rejected an undeclared change: "
                f"{mismatches}"
            )
        if policy_kl_coef is not None or policy_anchor_mode != "fixed":
            raise ValueError("l2r_curated_buffer_pilot forbids policy anchoring")
        generation_width = int(config.G_answer_only)
        registered_cell = (
            generation_width,
            proposal_filter,
            variational_estimator,
        )
        permitted_cells = {
            (32, "all", "prior_importance"),
            (32, "all", "uniform_mc"),
            (32, "answer_correct", "uniform_mc"),
        }
        if registered_cell not in permitted_cells:
            raise ValueError(
                "l2r_curated_buffer_pilot permits only the registered "
                "candidate-pool, admission, and estimator combinations; got "
                f"{registered_cell!r}"
            )
    elif algorithm_profile in {
        "l2r_reader_ess_closure",
        "l2r_credit_pilot",
        "l2r_abstention_pilot",
        "l2r_two_witness_pilot",
        "l2r_multi_verifier_pilot",
        "l2r_bayesian_fusion_pilot",
        "l2r_uncertainty_allocation_pilot",
        "l2r_age_one_reuse_pilot",
        "l2r_small_group_replay_pilot",
        "l2r_exact_signed_factorial",
    }:
        focused_requirements = {
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "answer_target_termination": (answer_target_termination, "eos"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "labelled_proposal_prompt": (labelled_proposal_prompt, "question"),
            "answer_only_proposal_prompt": (
                answer_only_proposal_prompt,
                "question",
            ),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_filter": (proposal_filter, "all"),
            "proposal_policy": (proposal_policy, "current"),
            "responsibility_score": (responsibility_score, "joint"),
            "responsibility_temperature": (responsibility_temperature, 1.0),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_verifier_rollouts": (
                responsibility_verifier_rollouts,
                0,
            ),
            "labelled_numeric_constraint": (labelled_numeric_constraint, "off"),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (optimizer_state_scope, "persistent"),
            "question_sampling": (config.question_sampling, "epoch_shuffle"),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in focused_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                f"{algorithm_profile} rejected an undeclared algorithm "
                f"change: {mismatches}"
            )
        if policy_kl_coef is not None or policy_anchor_mode != "fixed":
            raise ValueError(f"{algorithm_profile} forbids policy anchoring")
        if algorithm_profile == "l2r_two_witness_pilot":
            if responsibility_posterior not in {
                "softmax_entropy",
                "two_witness",
            }:
                raise ValueError(
                    "l2r_two_witness_pilot permits only the one-witness control "
                    "and two-witness update"
                )
        elif algorithm_profile in {
            "l2r_multi_verifier_pilot",
            "l2r_bayesian_fusion_pilot",
        }:
            if responsibility_posterior not in {
                "softmax_entropy",
                *MULTI_VERIFIER_POSTERIORS,
                "verifier_bayesian",
            }:
                raise ValueError(
                    f"{algorithm_profile} received an unregistered verifier "
                    "posterior"
                )
        elif responsibility_posterior != "softmax_entropy":
            raise ValueError(
                f"{algorithm_profile} requires softmax_entropy responsibilities"
            )
        if algorithm_profile == "l2r_reader_ess_closure":
            closure_requirements = {
                "buffer_semantics": (buffer_semantics, "unique_set"),
                "buffer_lifecycle": (buffer_lifecycle, "persistent"),
                "responsibility_refresh": (
                    responsibility_refresh,
                    "inner_step",
                ),
                "variational_estimator": (variational_estimator, "delta_joint"),
                "latent_mstep_objective": (latent_mstep_objective, "joint"),
            }
            mismatches = {
                name: {"actual": actual, "required": required}
                for name, (actual, required) in closure_requirements.items()
                if actual != required
            }
            if mismatches:
                raise ValueError(
                    "l2r_reader_ess_closure rejected a non-reader/ESS change: "
                    f"{mismatches}"
                )
            if responsibility_answer_policy not in {"current", "frozen_base"}:
                raise ValueError("closure reader must be current or frozen_base")
            if responsibility_ess_floor not in {0.0, 0.5}:
                raise ValueError("closure ESS floor must be exactly 0 or 0.5")
        elif algorithm_profile == "l2r_credit_pilot":
            pilot_requirements = {
                "buffer_semantics": (buffer_semantics, "multiset_legacy"),
                "buffer_lifecycle": (buffer_lifecycle, "fresh_round"),
                "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
                "responsibility_answer_policy": (
                    responsibility_answer_policy,
                    "current",
                ),
                "responsibility_refresh": (
                    responsibility_refresh,
                    "outer_round",
                ),
                "variational_estimator": (
                    variational_estimator,
                    "prior_importance",
                ),
            }
            mismatches = {
                name: {"actual": actual, "required": required}
                for name, (actual, required) in pilot_requirements.items()
                if actual != required
            }
            if mismatches:
                raise ValueError(
                    "l2r_credit_pilot rejected a non-credit change: "
                    f"{mismatches}"
                )
            if latent_mstep_objective not in {
                "joint",
                "centered_trace_answer",
                "segment_responsibility_flow",
            }:
                raise ValueError(
                    "l2r_credit_pilot permits only its three registered "
                    "M-step credit objectives"
                )
        elif algorithm_profile == "l2r_exact_signed_factorial":
            exact_common = {
                "buffer_semantics": (buffer_semantics, "multiset_legacy"),
                "buffer_lifecycle": (buffer_lifecycle, "fresh_round"),
                "buffer_limit": (int(config.buffer_limit), 8),
                "candidate_size": (int(config.G_answer_only), 8),
                "questions_per_round": (int(config.U_batch), 8),
                "outer_rounds": (int(config.rounds), 32),
                "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
                "responsibility_abstention": (responsibility_abstention, "none"),
                "responsibility_answer_policy": (
                    responsibility_answer_policy,
                    "current",
                ),
                "diagnostics_level": (diagnostics_level, "standard"),
                "diagnostics_trace_tape": (diagnostics_trace_tape, True),
                "diagnostics_gradient_questions": (
                    diagnostics_gradient_questions,
                    0,
                ),
                "checkpoint_every": (checkpoint_every, 0),
            }
            mismatches = {
                name: {"actual": actual, "required": required}
                for name, (actual, required) in exact_common.items()
                if actual != required
            }
            if mismatches:
                raise ValueError(
                    "l2r_exact_signed_factorial rejected a substrate change: "
                    f"{mismatches}"
                )
            registered_update = (
                variational_estimator,
                responsibility_refresh,
                latent_mstep_objective,
                int(config.inner_steps),
            )
            if registered_update not in {
                ("prior_importance", "outer_round", "joint", 1),
                ("prior_importance", "outer_round", "joint", 4),
                (
                    "sampled_support_importance",
                    "inner_step",
                    "exact_signed_trace_answer",
                    1,
                ),
                (
                    "sampled_support_importance",
                    "inner_step",
                    "exact_signed_trace_answer",
                    4,
                ),
            }:
                raise ValueError(
                    "l2r_exact_signed_factorial permits only ordinary answer-weighted "
                    "and exact dynamic signed updates at U in {1,4}; got "
                    f"{registered_update!r}"
                )
        elif algorithm_profile == "l2r_abstention_pilot":
            abstention_requirements = {
                "buffer_semantics": (buffer_semantics, "multiset_legacy"),
                "buffer_lifecycle": (buffer_lifecycle, "fresh_round"),
                "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
                "responsibility_answer_policy": (
                    responsibility_answer_policy,
                    "current",
                ),
                "responsibility_refresh": (
                    responsibility_refresh,
                    "outer_round",
                ),
                "variational_estimator": (
                    variational_estimator,
                    "prior_importance",
                ),
                "latent_mstep_objective": (latent_mstep_objective, "joint"),
            }
            mismatches = {
                name: {"actual": actual, "required": required}
                for name, (actual, required) in abstention_requirements.items()
                if actual != required
            }
            if mismatches:
                raise ValueError(
                    "l2r_abstention_pilot rejected a non-abstention change: "
                    f"{mismatches}"
                )
            expected_threshold = -1.2508713810019798
            expected_null_log_evidence = -2.1191687253117015
            if responsibility_abstention not in RESPONSIBILITY_ABSTENTION_MODES:
                raise ValueError("unknown abstention mode")
            if responsibility_abstention == "hard_threshold":
                if responsibility_rejection_threshold != expected_threshold:
                    raise ValueError(
                        "hard-threshold arm requires the frozen controller value"
                    )
                if (
                    responsibility_null_log_evidence != 0.0
                    or responsibility_null_prior != 0.5
                ):
                    raise ValueError("hard-threshold arm forbids null controls")
            elif responsibility_abstention == "null_latent":
                if (
                    responsibility_null_log_evidence
                    != expected_null_log_evidence
                    or responsibility_null_prior != 0.5
                    or responsibility_rejection_threshold != 0.0
                ):
                    raise ValueError(
                        "null-latent arm requires the frozen controller values"
                    )
            elif (
                responsibility_rejection_threshold != 0.0
                or responsibility_null_log_evidence != 0.0
                or responsibility_null_prior != 0.5
            ):
                raise ValueError("forced arm forbids inactive abstention controls")
        elif algorithm_profile in {
            "l2r_multi_verifier_pilot",
            "l2r_bayesian_fusion_pilot",
        }:
            multi_verifier_requirements = {
                "buffer_semantics": (buffer_semantics, "multiset_legacy"),
                "buffer_lifecycle": (buffer_lifecycle, "fresh_round"),
                "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
                "responsibility_abstention": (
                    responsibility_abstention,
                    "null_latent",
                ),
                "responsibility_rejection_threshold": (
                    responsibility_rejection_threshold,
                    0.0,
                ),
                "responsibility_null_log_evidence": (
                    responsibility_null_log_evidence,
                    -2.1191687253117015,
                ),
                "responsibility_null_prior": (
                    responsibility_null_prior,
                    0.5,
                ),
                "responsibility_answer_policy": (
                    responsibility_answer_policy,
                    "current",
                ),
                "responsibility_refresh": (
                    responsibility_refresh,
                    "outer_round",
                ),
                "variational_estimator": (
                    variational_estimator,
                    "prior_importance",
                ),
                "latent_mstep_objective": (latent_mstep_objective, "joint"),
            }
            mismatches = {
                name: {"actual": actual, "required": required}
                for name, (actual, required) in multi_verifier_requirements.items()
                if actual != required
            }
            if mismatches:
                raise ValueError(
                    f"{algorithm_profile} rejected a non-verifier change: "
                    f"{mismatches}"
                )
            if algorithm_profile == "l2r_bayesian_fusion_pilot":
                if responsibility_posterior != "verifier_bayesian":
                    raise ValueError(
                        "the Bayesian fusion profile requires verifier_bayesian"
                    )
                if not verifier_calibration_path:
                    raise ValueError(
                        "the Bayesian fusion profile requires a calibration artifact"
                    )
            elif responsibility_posterior == "verifier_bayesian":
                raise ValueError(
                    "verifier_bayesian is isolated to its registered profile"
                )
        elif algorithm_profile == "l2r_uncertainty_allocation_pilot":
            allocation_requirements = {
                "buffer_semantics": (buffer_semantics, "multiset_legacy"),
                "buffer_lifecycle": (buffer_lifecycle, "fresh_round"),
                "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
                "responsibility_abstention": (
                    responsibility_abstention,
                    "null_latent",
                ),
                "responsibility_null_log_evidence": (
                    responsibility_null_log_evidence,
                    -2.1191687253117015,
                ),
                "responsibility_null_prior": (responsibility_null_prior, 0.5),
                "responsibility_answer_policy": (
                    responsibility_answer_policy,
                    "current",
                ),
                "responsibility_refresh": (
                    responsibility_refresh,
                    "outer_round",
                ),
                "variational_estimator": (variational_estimator, "prior_importance"),
                "latent_mstep_objective": (latent_mstep_objective, "joint"),
                "proposal_initial_traces": (proposal_initial_traces, 4),
                "proposal_allocation_max_traces": (
                    proposal_allocation_max_traces,
                    12,
                ),
            }
            mismatches = {
                name: {"actual": actual, "required": required}
                for name, (actual, required) in allocation_requirements.items()
                if actual != required
            }
            if mismatches:
                raise ValueError(
                    "l2r_uncertainty_allocation_pilot rejected an undeclared "
                    f"change: {mismatches}"
                )
            if proposal_allocation_mode not in {
                "posterior_uncertainty",
                "posterior_uncertainty_shifted",
            }:
                raise ValueError(
                    "the uncertainty pilot requires an aligned or shifted allocation"
                )
        elif algorithm_profile in {
            "l2r_age_one_reuse_pilot",
            "l2r_small_group_replay_pilot",
        }:
            reuse_requirements = {
                "buffer_semantics": (buffer_semantics, "multiset_legacy"),
                "buffer_lifecycle": (buffer_lifecycle, "persistent"),
                "buffer_max_age": (buffer_max_age, 1),
                "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
                "responsibility_answer_policy": (
                    responsibility_answer_policy,
                    "current",
                ),
                "responsibility_refresh": (
                    responsibility_refresh,
                    "outer_round",
                ),
                "variational_estimator": (
                    variational_estimator,
                    "persistent_prior_importance",
                ),
                "latent_mstep_objective": (latent_mstep_objective, "joint"),
                "reuse_fresh_traces": (reuse_fresh_traces, 4),
                "reuse_importance_min": (reuse_importance_min, 0.5),
                "reuse_importance_max": (reuse_importance_max, 2.0),
            }
            if algorithm_profile == "l2r_age_one_reuse_pilot":
                reuse_requirements.update(
                    {
                        "responsibility_abstention": (
                            responsibility_abstention,
                            "null_latent",
                        ),
                        "responsibility_null_log_evidence": (
                            responsibility_null_log_evidence,
                            -2.1191687253117015,
                        ),
                        "responsibility_null_prior": (
                            responsibility_null_prior,
                            0.5,
                        ),
                    }
                )
            else:
                reuse_requirements.update(
                    {
                        "responsibility_abstention": (
                            responsibility_abstention,
                            "none",
                        ),
                        "responsibility_null_log_evidence": (
                            responsibility_null_log_evidence,
                            0.0,
                        ),
                        "responsibility_null_prior": (
                            responsibility_null_prior,
                            0.5,
                        ),
                    }
                )
            mismatches = {
                name: {"actual": actual, "required": required}
                for name, (actual, required) in reuse_requirements.items()
                if actual != required
            }
            if mismatches:
                raise ValueError(
                    f"{algorithm_profile} rejected an undeclared change: "
                    f"{mismatches}"
                )
        else:
            two_witness_requirements = {
                "buffer_semantics": (buffer_semantics, "multiset_legacy"),
                "buffer_lifecycle": (buffer_lifecycle, "fresh_round"),
                "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
                "responsibility_abstention": (
                    responsibility_abstention,
                    "none",
                ),
                "responsibility_answer_policy": (
                    responsibility_answer_policy,
                    "current",
                ),
                "responsibility_refresh": (
                    responsibility_refresh,
                    "outer_round",
                ),
                "variational_estimator": (
                    variational_estimator,
                    "prior_importance",
                ),
                "latent_mstep_objective": (latent_mstep_objective, "joint"),
            }
            mismatches = {
                name: {"actual": actual, "required": required}
                for name, (actual, required) in two_witness_requirements.items()
                if actual != required
            }
            if mismatches:
                raise ValueError(
                    "l2r_two_witness_pilot rejected a non-witness change: "
                    f"{mismatches}"
                )
    elif algorithm_profile in {"barber_source", "barber_fixed_kl_ablation"}:
        source_requirements = {
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_filter": (proposal_filter, "all"),
            "responsibility_score": (responsibility_score, "joint"),
            "responsibility_posterior": (
                responsibility_posterior,
                "softmax_entropy",
            ),
            "responsibility_temperature": (
                responsibility_temperature,
                1.0,
            ),
            "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_answer_policy": (
                responsibility_answer_policy,
                "current",
            ),
            "responsibility_refresh": (
                responsibility_refresh,
                "inner_step",
            ),
            "responsibility_verifier_rollouts": (
                responsibility_verifier_rollouts,
                0,
            ),
            "labelled_numeric_constraint": (
                labelled_numeric_constraint,
                "off",
            ),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (
                optimizer_state_scope,
                "persistent",
            ),
            "question_sampling": (
                config.question_sampling,
                "epoch_shuffle",
            ),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in source_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                f"{algorithm_profile} profile rejected hidden algorithm changes: "
                f"{mismatches}"
            )
        if algorithm_profile == "barber_source":
            if policy_kl_coef is not None or policy_anchor_mode != "fixed":
                raise ValueError("barber_source forbids policy anchoring")
        else:
            if policy_anchor_mode != "fixed":
                raise ValueError(
                    "barber_fixed_kl_ablation requires fixed policy anchoring"
                )
            if policy_kl_coef is None or policy_kl_coef <= 0:
                raise ValueError(
                    "barber_fixed_kl_ablation requires a positive policy_kl_coef"
                )
        if labelled_supervision not in {"gold", "gold_answer"}:
            raise ValueError(
                "barber_source permits only gold or direct-answer supervision"
            )
        if labelled_supervision == "gold_answer" and (
            labelled_em_weight > 0 or answer_only_em_weight > 0
        ):
            raise ValueError(
                "gold_answer is the isolated Equation-1 SFT control and "
                "cannot include latent terms"
            )
        if variational_estimator == "delta_joint":
            if (
                buffer_lifecycle != "persistent"
                or buffer_semantics != "unique_set"
            ):
                raise ValueError(
                    "source delta_joint requires a persistent token-unique set"
                )
        elif buffer_semantics != "multiset_legacy":
            raise ValueError(
                "source Monte Carlo/importance estimators must preserve draw "
                "multiplicity"
            )
    elif algorithm_profile == "barber_stability_ablation":
        stability_requirements = {
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "answer_target_termination": (answer_target_termination, "eos"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "buffer_semantics": (buffer_semantics, "unique_set"),
            "buffer_lifecycle": (buffer_lifecycle, "persistent"),
            "labelled_proposal_prompt": (
                labelled_proposal_prompt,
                "question",
            ),
            "answer_only_proposal_prompt": (
                answer_only_proposal_prompt,
                "question",
            ),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_filter": (proposal_filter, "all"),
            "proposal_policy": (proposal_policy, "current"),
            "responsibility_score": (responsibility_score, "joint"),
            "responsibility_posterior": (
                responsibility_posterior,
                "softmax_entropy",
            ),
            "responsibility_temperature": (
                responsibility_temperature,
                1.0,
            ),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_answer_policy": (
                responsibility_answer_policy,
                "current",
            ),
            "responsibility_refresh": (
                responsibility_refresh,
                "inner_step",
            ),
            "responsibility_verifier_rollouts": (
                responsibility_verifier_rollouts,
                0,
            ),
            "variational_estimator": (variational_estimator, "delta_joint"),
            "labelled_numeric_constraint": (
                labelled_numeric_constraint,
                "off",
            ),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "latent_mstep_objective": (latent_mstep_objective, "joint"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (
                optimizer_state_scope,
                "persistent",
            ),
            "question_sampling": (
                config.question_sampling,
                "epoch_shuffle",
            ),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in stability_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                "barber_stability_ablation rejected a non-stability change: "
                f"{mismatches}"
            )
        if policy_kl_coef is not None:
            raise ValueError(
                "barber_stability_ablation uses only the gradient-ratio KL "
                "controller; policy_kl_coef must be omitted"
            )
    elif algorithm_profile in {
        "barber_q5_control",
        "barber_q5_token_mean_followup",
    }:
        expected_responsibility_score = (
            "token_mean"
            if algorithm_profile == "barber_q5_token_mean_followup"
            else "joint"
        )
        q5_requirements = {
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "answer_target_termination": (answer_target_termination, "eos"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "buffer_semantics": (buffer_semantics, "unique_set"),
            "buffer_lifecycle": (buffer_lifecycle, "persistent"),
            "labelled_proposal_prompt": (
                labelled_proposal_prompt,
                "answer_derive",
            ),
            "answer_only_proposal_prompt": (
                answer_only_proposal_prompt,
                "answer_derive",
            ),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_filter": (proposal_filter, "all"),
            "proposal_policy": (proposal_policy, "current"),
            "responsibility_score": (
                responsibility_score,
                expected_responsibility_score,
            ),
            "responsibility_posterior": (
                responsibility_posterior,
                "softmax_entropy",
            ),
            "responsibility_temperature": (
                responsibility_temperature,
                1.0,
            ),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_refresh": (responsibility_refresh, "inner_step"),
            "responsibility_verifier_rollouts": (
                responsibility_verifier_rollouts,
                0,
            ),
            "variational_estimator": (variational_estimator, "delta_joint"),
            "labelled_numeric_constraint": (
                labelled_numeric_constraint,
                "off",
            ),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "latent_mstep_objective": (latent_mstep_objective, "joint"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (
                optimizer_state_scope,
                "persistent",
            ),
            "question_sampling": (
                config.question_sampling,
                "epoch_shuffle",
            ),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in q5_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                f"{algorithm_profile} rejected a non-Q5 change: "
                f"{mismatches}"
            )
        if algorithm_profile == "barber_q5_token_mean_followup":
            if responsibility_answer_policy != "current":
                raise ValueError("token-mean Q5 requires the moving answer reader")
            if responsibility_ess_floor != 0.0 or proposal_temperature != 1.0:
                raise ValueError("token-mean Q5 forbids simultaneous ESS or temperature changes")
            if (
                policy_kl_coef is not None
                or policy_anchor_mode != "fixed"
                or policy_anchor_target_ratio is not None
                or policy_anchor_token_scope != "objective"
            ):
                raise ValueError("token-mean Q5 forbids simultaneous policy anchoring")
        if responsibility_answer_policy not in {"current", "frozen_base"}:
            raise ValueError(
                "barber_q5_control permits only current or frozen_base readers"
            )
        if responsibility_ess_floor not in {0.0, 0.5}:
            raise ValueError(
                "barber_q5_control permits only ESS floors 0.0 or 0.5"
            )
        if proposal_temperature not in {1.0, 1.2}:
            raise ValueError(
                "barber_q5_control permits only proposal temperatures 1.0 or 1.2"
            )
        anchored = policy_anchor_mode == "grad_ratio"
        interventions = sum(
            (
                responsibility_answer_policy == "frozen_base",
                responsibility_ess_floor == 0.5,
                proposal_temperature == 1.2,
                anchored,
            )
        )
        if interventions > 1:
            raise ValueError(
                "barber_q5_control permits at most one reader, ESS, proposal-"
                "temperature or KL intervention per cell"
            )
        if anchored:
            if policy_kl_coef is not None:
                raise ValueError("adaptive Q5 anchoring forbids policy_kl_coef")
            if policy_anchor_target_ratio != 0.03:
                raise ValueError("adaptive Q5 anchoring requires target ratio 0.03")
            if policy_anchor_token_scope != "reasoning":
                raise ValueError("adaptive Q5 anchoring must target reasoning tokens")
        elif (
            policy_kl_coef is not None
            or policy_anchor_mode != "fixed"
            or policy_anchor_target_ratio is not None
            or policy_anchor_token_scope != "objective"
        ):
            raise ValueError("unanchored Q5 controls must disable every anchor setting")
    elif algorithm_profile == "l2r_pis_rationale_kl_followup":
        pis_requirements = {
            "rounds": (int(config.rounds), 32),
            "inner_steps": (int(config.inner_steps), 4),
            "buffer_limit": (int(config.buffer_limit), 8),
            "candidate_size": (int(config.G_answer_only), 8),
            "questions_per_round": (int(config.U_batch), 8),
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "answer_target_termination": (answer_target_termination, "eos"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "buffer_semantics": (buffer_semantics, "multiset_legacy"),
            "buffer_lifecycle": (buffer_lifecycle, "fresh_round"),
            "labelled_proposal_prompt": (labelled_proposal_prompt, "question"),
            "answer_only_proposal_prompt": (answer_only_proposal_prompt, "question"),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_filter": (proposal_filter, "all"),
            "proposal_policy": (proposal_policy, "current"),
            "responsibility_score": (responsibility_score, "joint"),
            "responsibility_posterior": (responsibility_posterior, "softmax_entropy"),
            "responsibility_temperature": (responsibility_temperature, 1.0),
            "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_answer_policy": (responsibility_answer_policy, "current"),
            "responsibility_refresh": (responsibility_refresh, "outer_round"),
            "responsibility_verifier_rollouts": (responsibility_verifier_rollouts, 0),
            "variational_estimator": (variational_estimator, "prior_importance"),
            "labelled_numeric_constraint": (labelled_numeric_constraint, "off"),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "latent_mstep_objective": (latent_mstep_objective, "joint"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (optimizer_state_scope, "persistent"),
            "question_sampling": (config.question_sampling, "epoch_shuffle"),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in pis_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                "l2r_pis_rationale_kl_followup rejected an undeclared change: "
                f"{mismatches}"
            )
        if policy_kl_coef is not None:
            raise ValueError("adaptive PIS anchoring forbids policy_kl_coef")
        if policy_anchor_mode != "grad_ratio":
            raise ValueError("PIS KL follow-up requires adaptive gradient-ratio anchoring")
        if policy_anchor_target_ratio != 0.03:
            raise ValueError("PIS KL follow-up requires target ratio 0.03")
        if policy_anchor_token_scope != "reasoning":
            raise ValueError("PIS KL follow-up must anchor rationale tokens only")
    elif algorithm_profile in {
        "barber_reader_ablation",
        "barber_refresh_ablation",
    }:
        reader_requirements = {
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "answer_target_termination": (answer_target_termination, "eos"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "buffer_semantics": (buffer_semantics, "unique_set"),
            "buffer_lifecycle": (buffer_lifecycle, "persistent"),
            "labelled_proposal_prompt": (
                labelled_proposal_prompt,
                "answer_derive",
            ),
            "answer_only_proposal_prompt": (
                answer_only_proposal_prompt,
                "answer_derive",
            ),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_filter": (proposal_filter, "all"),
            "proposal_policy": (proposal_policy, "current"),
            "responsibility_score": (responsibility_score, "joint"),
            "responsibility_posterior": (
                responsibility_posterior,
                "softmax_entropy",
            ),
            "responsibility_temperature": (
                responsibility_temperature,
                1.0,
            ),
            "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_verifier_rollouts": (
                responsibility_verifier_rollouts,
                0,
            ),
            "variational_estimator": (
                variational_estimator,
                "delta_joint",
            ),
            "labelled_numeric_constraint": (
                labelled_numeric_constraint,
                "off",
            ),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "latent_mstep_objective": (latent_mstep_objective, "joint"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (
                optimizer_state_scope,
                "persistent",
            ),
            "question_sampling": (
                config.question_sampling,
                "epoch_shuffle",
            ),
        }
        if algorithm_profile == "barber_reader_ablation":
            reader_requirements["responsibility_refresh"] = (
                responsibility_refresh,
                "inner_step",
            )
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in reader_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                f"{algorithm_profile} profile rejected hidden algorithm "
                f"changes: {mismatches}"
            )
        if (
            algorithm_profile == "barber_reader_ablation"
            and responsibility_answer_policy not in {"current", "frozen_base"}
        ):
            raise ValueError(
                "barber_reader_ablation permits only current or frozen_base "
                "answer readers"
            )
        if algorithm_profile == "barber_refresh_ablation":
            if responsibility_answer_policy != "current":
                raise ValueError(
                    "barber_refresh_ablation requires the moving answer reader"
                )
            if responsibility_refresh not in {"inner_step", "outer_round"}:
                raise ValueError(
                    "barber_refresh_ablation permits only inner_step or outer_round"
                )
        if policy_kl_coef is not None or policy_anchor_mode != "fixed":
            raise ValueError(f"{algorithm_profile} forbids policy anchoring")
    elif algorithm_profile == "barber_importance_ablation":
        importance_requirements = {
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "answer_target_termination": (answer_target_termination, "eos"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "buffer_semantics": (buffer_semantics, "multiset_legacy"),
            "buffer_lifecycle": (buffer_lifecycle, "fresh_round"),
            "labelled_proposal_prompt": (
                labelled_proposal_prompt,
                "answer_derive",
            ),
            "answer_only_proposal_prompt": (
                answer_only_proposal_prompt,
                "answer_derive",
            ),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_filter": (proposal_filter, "all"),
            "proposal_policy": (proposal_policy, "current"),
            "responsibility_score": (responsibility_score, "joint"),
            "responsibility_posterior": (
                responsibility_posterior,
                "softmax_entropy",
            ),
            "responsibility_temperature": (
                responsibility_temperature,
                1.0,
            ),
            "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_answer_policy": (
                responsibility_answer_policy,
                "current",
            ),
            "responsibility_refresh": (
                responsibility_refresh,
                "inner_step",
            ),
            "responsibility_verifier_rollouts": (
                responsibility_verifier_rollouts,
                0,
            ),
            "variational_estimator": (
                variational_estimator,
                "answer_conditioned_importance",
            ),
            "labelled_numeric_constraint": (
                labelled_numeric_constraint,
                "off",
            ),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "latent_mstep_objective": (latent_mstep_objective, "joint"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (
                optimizer_state_scope,
                "persistent",
            ),
            "question_sampling": (
                config.question_sampling,
                "epoch_shuffle",
            ),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in importance_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                "barber_importance_ablation profile rejected hidden algorithm "
                f"changes: {mismatches}"
            )
        if policy_kl_coef is not None or policy_anchor_mode != "fixed":
            raise ValueError("barber_importance_ablation forbids policy anchoring")
    elif algorithm_profile == "barber_persistent_bridge":
        bridge_requirements = {
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "answer_target_termination": (answer_target_termination, "eos"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "buffer_lifecycle": (buffer_lifecycle, "persistent"),
            "labelled_proposal_prompt": (
                labelled_proposal_prompt,
                "answer_derive",
            ),
            "answer_only_proposal_prompt": (
                answer_only_proposal_prompt,
                "answer_derive",
            ),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_filter": (proposal_filter, "all"),
            "proposal_policy": (proposal_policy, "current"),
            "responsibility_score": (responsibility_score, "joint"),
            "responsibility_posterior": (
                responsibility_posterior,
                "softmax_entropy",
            ),
            "responsibility_temperature": (
                responsibility_temperature,
                1.0,
            ),
            "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_answer_policy": (
                responsibility_answer_policy,
                "current",
            ),
            "responsibility_refresh": (
                responsibility_refresh,
                "inner_step",
            ),
            "responsibility_verifier_rollouts": (
                responsibility_verifier_rollouts,
                0,
            ),
            "labelled_numeric_constraint": (
                labelled_numeric_constraint,
                "off",
            ),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (
                optimizer_state_scope,
                "persistent",
            ),
            "question_sampling": (
                config.question_sampling,
                "epoch_shuffle",
            ),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in bridge_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                "barber_persistent_bridge rejected an undeclared change: "
                f"{mismatches}"
            )
        if policy_kl_coef is not None or policy_anchor_mode != "fixed":
            raise ValueError("barber_persistent_bridge forbids policy anchoring")
        if buffer_semantics == "unique_set":
            if variational_estimator != "delta_joint":
                raise ValueError(
                    "barber_persistent_bridge unique-set cells require delta_joint"
                )
        elif buffer_semantics == "multiset_legacy":
            if variational_estimator not in {
                "delta_joint",
                "persistent_answer_conditioned_importance",
            }:
                raise ValueError(
                    "barber_persistent_bridge multiset cells require joint or "
                    "persistent answer-conditioned importance weights"
                )
        else:  # guarded earlier, retained here as a fail-closed profile check
            raise ValueError(
                "barber_persistent_bridge requires unique-set or multiset semantics"
            )
    elif algorithm_profile == "q5_support_reallocation":
        q5_requirements = {
            "rounds": (config.rounds, 32),
            "L_batch": (config.L_batch, 0),
            "inner_steps": (config.inner_steps, 1),
            "lr": (config.lr, 1.0e-5),
            "buffer_limit": (config.buffer_limit, 16),
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "answer_target_termination": (answer_target_termination, "eos"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_semantics": (buffer_semantics, "unique_set"),
            "buffer_lifecycle": (buffer_lifecycle, "persistent"),
            "buffer_max_age": (buffer_max_age, -1),
            "labelled_proposal_prompt": (
                labelled_proposal_prompt,
                "answer_derive",
            ),
            "answer_only_proposal_prompt": (
                answer_only_proposal_prompt,
                "answer_derive",
            ),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_filter": (proposal_filter, "all"),
            "proposal_policy": (proposal_policy, "current"),
            "proposal_allocation_mode": (proposal_allocation_mode, "uniform"),
            "responsibility_posterior": (
                responsibility_posterior,
                "softmax_entropy",
            ),
            "responsibility_temperature": (responsibility_temperature, 1.0),
            "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
            "responsibility_abstention": (responsibility_abstention, "none"),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_answer_policy": (
                responsibility_answer_policy,
                "current",
            ),
            "variational_estimator": (variational_estimator, "delta_joint"),
            "labelled_numeric_constraint": (labelled_numeric_constraint, "off"),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "latent_mstep_objective": (latent_mstep_objective, "joint"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (optimizer_state_scope, "persistent"),
            "question_sampling": (config.question_sampling, "epoch_shuffle"),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in q5_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                "q5_support_reallocation rejected hidden algorithm changes: "
                f"{mismatches}"
            )
        if config.G_label != config.G_answer_only:
            raise ValueError(
                "q5_support_reallocation requires matched labelled and answer-only G"
            )
        registered_cell = (
            int(config.U_batch),
            int(config.G_answer_only),
            buffer_strategy,
            responsibility_score,
            responsibility_refresh,
            int(responsibility_verifier_rollouts),
        )
        # U_batch is the number of questions after the sweep adapter divides
        # the proposal budget by G; it is not the sweep-level proposal budget.
        registered_cells = {
            (4, 16, "fifo", "joint", "inner_step", 0),
            (4, 32, "fifo", "joint", "inner_step", 0),
            (8, 16, "fifo", "joint", "inner_step", 0),
            (4, 32, "calculation_diverse", "joint", "inner_step", 0),
            (4, 16, "fifo", "rollout_value", "outer_round", 1),
        }
        if registered_cell not in registered_cells:
            raise ValueError(
                "q5_support_reallocation permits only its five registered cells; "
                f"got {registered_cell!r}"
            )
        if policy_kl_coef is not None or policy_anchor_mode != "fixed":
            raise ValueError("q5_support_reallocation forbids policy anchoring")
        if responsibility_score == "rollout_value":
            verifier_requirements = {
                "temperature": (responsibility_verifier_temperature, 1.0),
                "max_new_tokens": (responsibility_verifier_max_new_tokens, 64),
                "batch_size": (responsibility_verifier_batch_size, 16),
                "smoothing_alpha": (responsibility_verifier_smoothing_alpha, 0.5),
            }
            verifier_mismatches = {
                name: {"actual": actual, "required": required}
                for name, (actual, required) in verifier_requirements.items()
                if actual != required
            }
            if verifier_mismatches:
                raise ValueError(
                    "q5_support_reallocation verifier settings changed: "
                    f"{verifier_mismatches}"
                )
    elif algorithm_profile == "q5_revisit_concise":
        q5_requirements = {
            "rounds": (config.rounds, 32),
            "L_batch": (config.L_batch, 0),
            "inner_steps": (config.inner_steps, 1),
            "lr": (config.lr, 1.0e-5),
            "buffer_limit": (config.buffer_limit, 16),
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "answer_target_termination": (answer_target_termination, "eos"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "buffer_semantics": (buffer_semantics, "unique_set"),
            "buffer_lifecycle": (buffer_lifecycle, "persistent"),
            "buffer_max_age": (buffer_max_age, -1),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_policy": (proposal_policy, "current"),
            "proposal_allocation_mode": (proposal_allocation_mode, "uniform"),
            "responsibility_score": (responsibility_score, "joint"),
            "responsibility_posterior": (
                responsibility_posterior,
                "softmax_entropy",
            ),
            "responsibility_temperature": (responsibility_temperature, 1.0),
            "responsibility_abstention": (responsibility_abstention, "none"),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_answer_policy": (
                responsibility_answer_policy,
                "current",
            ),
            "responsibility_refresh": (responsibility_refresh, "inner_step"),
            "responsibility_verifier_rollouts": (
                responsibility_verifier_rollouts,
                0,
            ),
            "variational_estimator": (variational_estimator, "delta_joint"),
            "labelled_numeric_constraint": (labelled_numeric_constraint, "off"),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (optimizer_state_scope, "persistent"),
            "question_sampling": (config.question_sampling, "epoch_shuffle"),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in q5_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                "q5_revisit_concise rejected hidden algorithm changes: "
                f"{mismatches}"
            )
        if config.G_label != config.G_answer_only:
            raise ValueError("q5_revisit_concise requires matched G values")
        registered_cell = (
            int(config.U_batch),
            int(config.G_answer_only),
            answer_only_proposal_prompt,
            proposal_filter,
            latent_mstep_objective,
            float(responsibility_ess_floor),
        )
        registered_cells = {
            (4, 16, "answer_derive", "all", "joint", 0.0),
            (4, 16, "answer_derive_concise", "all", "joint", 0.0),
            (
                4,
                16,
                "answer_derive_concise",
                "all",
                "joint_token_mean",
                0.0,
            ),
            (16, 4, "answer_derive", "all", "joint", 0.0),
            (
                16,
                4,
                "answer_derive",
                "answer_correct_numeric",
                "joint",
                0.5,
            ),
        }
        if registered_cell not in registered_cells:
            raise ValueError(
                "q5_revisit_concise permits only its five registered cells; "
                f"got {registered_cell!r}"
            )
        if labelled_proposal_prompt != answer_only_proposal_prompt:
            raise ValueError("q5_revisit_concise requires matched proposal prompts")
        if policy_kl_coef is not None or policy_anchor_mode != "fixed":
            raise ValueError("q5_revisit_concise forbids policy anchoring")
    elif algorithm_profile == "barber_verifier":
        verifier_requirements = {
            "answer_event_mode": (answer_event_mode, "strict_terminal_marker"),
            "labelled_frac": (config.labelled_frac, 0.0),
            "labelled_em_weight": (labelled_em_weight, 0.0),
            "answer_only_em_weight": (answer_only_em_weight, 1.0),
            "supervised_weight": (supervised_weight, 0.0),
            "buffer_strategy": (buffer_strategy, "fifo"),
            "labelled_proposal_prompt": (
                labelled_proposal_prompt,
                "question",
            ),
            "answer_only_proposal_prompt": (
                answer_only_proposal_prompt,
                "question",
            ),
            "proposal_mixture": (proposal_mixture, "single"),
            "proposal_filter": (proposal_filter, "all"),
            "proposal_policy": (proposal_policy, "current"),
            "responsibility_posterior": (
                responsibility_posterior,
                "softmax_entropy",
            ),
            "responsibility_temperature": (
                responsibility_temperature,
                1.0,
            ),
            "responsibility_ess_floor": (responsibility_ess_floor, 0.0),
            "responsibility_policy": (responsibility_policy, "current"),
            "responsibility_refresh": (
                responsibility_refresh,
                "outer_round",
            ),
            "labelled_numeric_constraint": (
                labelled_numeric_constraint,
                "off",
            ),
            "labelled_supervision": (labelled_supervision, "gold"),
            "digit_token_weight": (digit_token_weight, 1.0),
            "trace_representation": (trace_representation, "reasoning"),
            "latent_mstep_objective": (latent_mstep_objective, "joint"),
            "update_geometry": (update_geometry, "sum"),
            "step_acceptance": (step_acceptance, "none"),
            "optimizer_state_scope": (
                optimizer_state_scope,
                "persistent",
            ),
            "question_sampling": (
                config.question_sampling,
                "epoch_shuffle",
            ),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in verifier_requirements.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                "barber_verifier profile rejected hidden algorithm changes: "
                f"{mismatches}"
            )
        if policy_kl_coef is not None or policy_anchor_mode != "fixed":
            raise ValueError("barber_verifier forbids policy anchoring")
        if responsibility_score not in {"joint", "rollout_value"}:
            raise ValueError(
                "barber_verifier permits only joint or rollout_value evidence"
            )
        if responsibility_score == "joint":
            if responsibility_answer_policy != "current":
                raise ValueError(
                    "barber_verifier joint controls require the current answer reader"
                )
            if responsibility_verifier_rollouts != 0:
                raise ValueError(
                    "barber_verifier joint controls cannot sample verifier rollouts"
                )
        if variational_estimator == "delta_joint":
            if (
                buffer_lifecycle != "persistent"
                or buffer_semantics != "unique_set"
            ):
                raise ValueError(
                    "barber_verifier delta_joint requires a persistent "
                    "token-unique set"
                )
        elif variational_estimator == "prior_importance":
            if (
                buffer_lifecycle != "fresh_round"
                or buffer_semantics != "multiset_legacy"
            ):
                raise ValueError(
                    "barber_verifier prior_importance requires fresh empirical "
                    "multiset draws"
                )
        else:
            raise ValueError(
                "barber_verifier supports only delta_joint or prior_importance"
            )

    return (
        replace(
            config,
            labelled_proposal_prompt=labelled_proposal_prompt,
            answer_only_proposal_prompt=answer_only_proposal_prompt,
        ),
        policy_anchor_measured,
    )


def _validate_q5_support_task_contract(
    config: ACAlg1RunConfig,
    task,
) -> None:
    """Bind the support-reallocation profile to its registered question pool."""

    if config.algorithm_profile not in {
        "q5_support_reallocation",
        "q5_revisit_concise",
    }:
        return
    if config.algorithm_profile == "q5_revisit_concise":
        if len(task.prompts) != 128:
            raise ValueError(
                "q5_revisit_concise task pool changed: expected 128, found "
                f"{len(task.prompts)}"
            )
        return
    breadth_cell = config.U_batch == 8 and config.G_answer_only == 16
    expected_questions = 256 if breadth_cell else 128
    actual_questions = len(task.prompts)
    if actual_questions != expected_questions:
        raise ValueError(
            "q5_support_reallocation task pool changed: "
            f"expected {expected_questions}, found {actual_questions}"
        )


def _current_trace_prior_log_densities(
    model,
    tok,
    rows: list[TraceRow],
    *,
    policy: str,
) -> dict[str, float]:
    """Score retained rationale factors without changing model state."""

    if not rows:
        return {}
    if any(row.trace_id is None for row in rows):
        raise ValueError("age-one reuse requires stable trace identifiers")
    ids, span, ans = _pad_trace_rows(tok, rows)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad(), _adapter_policy_context(model, policy):
            values = seq_logprobs(
                model,
                ids,
                span & ~ans,
                micro=16,
                length_norm=False,
            )
    finally:
        model.train(was_training)
    return {
        str(row.trace_id): float(value)
        for row, value in zip(rows, values.detach().cpu().tolist())
    }


def _prepare_age_one_reuse_buffers(
    model,
    tok,
    buffers: dict[int, list[TraceRow]],
    pids: list[int],
    *,
    round_index: int,
    target_support_size: int,
    maximum_reused_rows: int,
    policy: str,
    fresh_source: str,
) -> tuple[list[int], int, dict[str, Any]]:
    """Keep safe age-one rows and return the exact fresh-backfill schedule."""

    sampling_pids: list[int] = []
    evicted = 0
    audits: dict[str, Any] = {}
    for pid in pids:
        existing = [row for row in buffers[int(pid)] if not row.is_gold]
        current = (
            _current_trace_prior_log_densities(
                model,
                tok,
                existing,
                policy=policy,
            )
            if round_index > 0
            else {
                str(row.trace_id): float(row.proposal_trace_logprob)
                for row in existing
                if row.trace_id is not None
            }
        )
        selection = select_age_one_reuse(
            existing,
            current_round=round_index,
            target_support_size=target_support_size,
            maximum_reused_rows=maximum_reused_rows,
            fresh_source=fresh_source,
            round_added=lambda row: int(row.round_added),
            trace_id=lambda row: str(row.trace_id),
            source=lambda row: str(row.source),
            proposal_log_density=lambda row: float(
                row.proposal_trace_logprob
            ),
            current_log_density=lambda row: float(
                current.get(str(row.trace_id), float("nan"))
            ),
        )
        selected = list(selection.selected_rows)
        accepted_current = {
            decision.trace_id: float(decision.current_log_density)
            for decision in selection.audit.decisions
            if decision.accepted
        }
        if set(accepted_current) != {
            str(row.trace_id) for row in selected
        }:
            raise RuntimeError("age-one selected rows do not match their density audit")
        for row in selected:
            row.reuse_admission_trace_logprob = accepted_current[
                str(row.trace_id)
            ]
        gold = [row for row in buffers[int(pid)] if row.is_gold]
        before = len(buffers[int(pid)])
        buffers[int(pid)][:] = gold + selected
        evicted += before - len(buffers[int(pid)])
        sampling_pids.extend(
            [int(pid)] * int(selection.audit.fresh_backfill_count)
        )
        audits[str(int(pid))] = asdict(selection.audit)
    return sampling_pids, evicted, {
        "mode": "age_one_importance_reuse",
        "target_support_size": int(target_support_size),
        "maximum_reused_rows": int(maximum_reused_rows),
        "fresh_draws": len(sampling_pids),
        "rollouts_saved": sum(
            int(audit["rollouts_saved"]) for audit in audits.values()
        ),
        "questions": audits,
    }


def _execute_ac_alg1_update(
    *,
    config: ACAlg1RunConfig,
    state: _ACAlg1RuntimeState,
    task,
    t: int,
    policy_anchor_measured: bool,
    eval_fn,
    diagnostics_fn,
    diagnostics_probe_fn,
) -> _ACAlg1RoundOutcome:
    rounds = config.rounds
    L_batch = config.L_batch
    U_batch = config.U_batch
    G_label = config.G_label
    G_answer_only = config.G_answer_only
    inner_steps = config.inner_steps
    lr = config.lr
    buffer_limit = config.buffer_limit
    buffer_strategy = config.buffer_strategy
    buffer_semantics = config.buffer_semantics
    buffer_lifecycle = config.buffer_lifecycle
    buffer_max_age = config.buffer_max_age
    labelled_proposal_prompt = config.labelled_proposal_prompt
    answer_only_proposal_prompt = config.answer_only_proposal_prompt
    proposal_mixture = config.proposal_mixture
    proposal_filter = config.proposal_filter
    proposal_policy = config.proposal_policy
    proposal_temperature = config.proposal_temperature
    proposal_allocation_mode = config.proposal_allocation_mode
    proposal_initial_traces = config.proposal_initial_traces
    proposal_allocation_max_traces = config.proposal_allocation_max_traces
    responsibility_score = config.responsibility_score
    responsibility_posterior = config.responsibility_posterior
    responsibility_temperature = config.responsibility_temperature
    responsibility_ess_floor = config.responsibility_ess_floor
    responsibility_abstention = config.responsibility_abstention
    responsibility_rejection_threshold = config.responsibility_rejection_threshold
    responsibility_null_log_evidence = config.responsibility_null_log_evidence
    responsibility_null_prior = config.responsibility_null_prior
    responsibility_policy = config.responsibility_policy
    responsibility_answer_policy = config.responsibility_answer_policy
    responsibility_refresh = config.responsibility_refresh
    responsibility_verifier_rollouts = config.responsibility_verifier_rollouts
    responsibility_verifier_temperature = (
        config.responsibility_verifier_temperature
    )
    responsibility_verifier_max_new_tokens = (
        config.responsibility_verifier_max_new_tokens
    )
    responsibility_verifier_batch_size = (
        config.responsibility_verifier_batch_size
    )
    responsibility_verifier_smoothing_alpha = (
        config.responsibility_verifier_smoothing_alpha
    )
    verifier_calibration_path = config.verifier_calibration_path
    reuse_fresh_traces = config.reuse_fresh_traces
    variational_estimator = config.variational_estimator
    labelled_em_weight = config.labelled_em_weight
    answer_only_em_weight = config.answer_only_em_weight
    policy_kl_coef = config.policy_kl_coef
    supervised_weight = config.supervised_weight
    policy_anchor_mode = config.policy_anchor_mode
    policy_anchor_target_ratio = config.policy_anchor_target_ratio
    policy_anchor_beta_min = config.policy_anchor_beta_min
    policy_anchor_beta_max = config.policy_anchor_beta_max
    policy_anchor_ema = config.policy_anchor_ema
    policy_anchor_token_scope = config.policy_anchor_token_scope
    labelled_numeric_constraint = config.labelled_numeric_constraint
    numeric_penalty = config.numeric_penalty
    numeric_contradiction_penalty = config.numeric_contradiction_penalty
    numeric_missing_penalty = config.numeric_missing_penalty
    labelled_supervision = config.labelled_supervision
    compact_gold_weight = config.compact_gold_weight
    digit_token_weight = config.digit_token_weight
    trace_representation = config.trace_representation
    latent_mstep_objective = config.latent_mstep_objective
    answer_event_mode = config.answer_event_mode
    answer_target_termination = config.answer_target_termination
    update_geometry = config.update_geometry
    step_acceptance = config.step_acceptance
    rollback_tolerance = config.rollback_tolerance
    rollback_max_backtracks = config.rollback_max_backtracks
    rollback_shrink = config.rollback_shrink
    optimizer_state_scope = config.optimizer_state_scope
    eval_every = config.eval_every
    eval_rounds = config.eval_rounds
    diagnostics_level = config.diagnostics_level
    diagnostics_gradient_questions = config.diagnostics_gradient_questions
    model = state.model
    tok = state.tok
    opt = state.opt
    labelled_pool = state.labelled_pool
    answer_only_pool = state.answer_only_pool
    labelled_sampler = state.labelled_sampler
    answer_only_sampler = state.answer_only_sampler
    buffers = state.buffers
    total_generated = state.total_generated
    total_steps = state.total_steps
    total_buffer_evictions = state.total_buffer_evictions
    total_set_duplicates = state.total_set_duplicates
    total_filter_verifier_calls = state.total_filter_verifier_calls
    total_responsibility_verifier_calls = (
        state.total_responsibility_verifier_calls
    )
    total_responsibility_verifier_tokens = (
        state.total_responsibility_verifier_tokens
    )
    policy_anchor_state = state.policy_anchor_state
    training_diagnostic_state = state.training_diagnostic_state
    initial_trainable_parameters = state.initial_trainable_parameters

    round_started = time.perf_counter()
    if opt is None or optimizer_state_scope == "outer_round":
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )
    labelled_pids = labelled_sampler.sample(L_batch).tolist()
    answer_only_pids = answer_only_sampler.sample(U_batch).tolist()
    active_labelled_pids = (
        labelled_pids if labelled_em_weight > 0 else []
    )
    active_answer_only_pids = (
        answer_only_pids if answer_only_em_weight > 0 else []
    )

    generation_started = time.perf_counter()
    labelled_sample_pids, labelled_lifecycle_evicted = _prepare_active_buffers(
        buffers,
        active_labelled_pids,
        buffer_lifecycle,
    )
    (
        answer_only_sample_pids,
        answer_only_lifecycle_evicted,
    ) = _prepare_active_buffers(
        buffers,
        active_answer_only_pids,
        buffer_lifecycle,
    )
    sampling_intervention: dict[str, Any] | None = None
    answer_traces_per_question = G_answer_only
    if config.algorithm_profile in {
        "l2r_age_one_reuse_pilot",
        "l2r_small_group_replay_pilot",
    }:
        (
            answer_only_sample_pids,
            reuse_evicted,
            sampling_intervention,
        ) = _prepare_age_one_reuse_buffers(
            model,
            tok,
            buffers,
            active_answer_only_pids,
            round_index=t,
            target_support_size=G_answer_only,
            maximum_reused_rows=G_answer_only - reuse_fresh_traces,
            policy=responsibility_policy,
            fresh_source="answer_only_sample",
        )
        answer_only_lifecycle_evicted += reuse_evicted
        answer_traces_per_question = 1
    elif proposal_allocation_mode != "uniform":
        answer_traces_per_question = proposal_initial_traces
    gold_added, gold_evicted = _add_gold_traces_to_buffer(
        tok, task, buffers, active_labelled_pids, round_added=t,
        buffer_limit=buffer_limit, buffer_strategy=buffer_strategy,
        trace_representation=trace_representation,
        answer_event_mode=answer_event_mode,
        answer_target_termination=answer_target_termination,
    )

    (labelled_sample_pid_row, labelled_sample_texts, labelled_sample_tokens,
     labelled_added, labelled_evicted, labelled_filter_stats) = _add_model_traces_to_buffer(
        model,
        tok,
        task,
        buffers,
        labelled_sample_pids,
        traces_per_question=G_label,
        round_added=t,
        source="labelled_sample",
        buffer_limit=buffer_limit,
        buffer_strategy=buffer_strategy,
        buffer_semantics=buffer_semantics,
        proposal_prompt=labelled_proposal_prompt,
        proposal_mixture=proposal_mixture,
        proposal_filter=proposal_filter,
        proposal_policy=proposal_policy,
        proposal_temperature=proposal_temperature,
        trace_representation=trace_representation,
        answer_event_mode=answer_event_mode,
        answer_target_termination=answer_target_termination,
        collect_token_counts=diagnostics_fn is not None,
        collect_proposal_outcomes=diagnostics_fn is not None,
        record_proposal_density=(
            variational_estimator in {
                "answer_conditioned_importance",
                "persistent_answer_conditioned_importance",
                "persistent_prior_importance",
                "sampled_support_importance",
            }
        ),
    )
    (answer_only_sample_pid_row, answer_only_sample_texts, answer_only_sample_tokens,
     answer_only_added, answer_only_evicted,
     answer_only_filter_stats) = _add_model_traces_to_buffer(
        model,
        tok,
        task,
        buffers,
        answer_only_sample_pids,
        traces_per_question=answer_traces_per_question,
        round_added=t,
        source="answer_only_sample",
        buffer_limit=buffer_limit,
        buffer_strategy=buffer_strategy,
        buffer_semantics=buffer_semantics,
        proposal_prompt=answer_only_proposal_prompt,
        proposal_mixture=proposal_mixture,
        proposal_filter=proposal_filter,
        proposal_policy=proposal_policy,
        proposal_temperature=proposal_temperature,
        trace_representation=trace_representation,
        answer_event_mode=answer_event_mode,
        answer_target_termination=answer_target_termination,
        collect_token_counts=diagnostics_fn is not None,
        collect_proposal_outcomes=diagnostics_fn is not None,
        record_proposal_density=(
            variational_estimator in {
                "answer_conditioned_importance",
                "persistent_answer_conditioned_importance",
                "persistent_prior_importance",
                "sampled_support_importance",
            }
        ),
    )
    if proposal_allocation_mode != "uniform":
        _unused_labelled, provisional_weights = _refresh_minibatch_weights(
            model,
            tok,
            buffers,
            [],
            active_answer_only_pids,
            responsibility_score=responsibility_score,
            responsibility_posterior=responsibility_posterior,
            responsibility_temperature=responsibility_temperature,
            responsibility_ess_floor=responsibility_ess_floor,
            responsibility_abstention=responsibility_abstention,
            responsibility_rejection_threshold=(
                responsibility_rejection_threshold
            ),
            responsibility_null_log_evidence=(
                responsibility_null_log_evidence
            ),
            responsibility_null_prior=responsibility_null_prior,
            responsibility_policy=responsibility_policy,
            responsibility_answer_policy=responsibility_answer_policy,
            variational_estimator=variational_estimator,
            record_joint_logprobs=True,
            task=task,
            verifier_calibration_path=verifier_calibration_path,
        )
        provisional_null_masses = []
        for pid in active_answer_only_pids:
            if int(pid) not in provisional_weights:
                provisional_null_masses.append(1.0)
                continue
            real_mass = float(provisional_weights[int(pid)].sum().item())
            if not -1e-6 <= real_mass <= 1.0 + 1e-6:
                raise RuntimeError(
                    "provisional real responsibility mass escaped [0, 1]"
                )
            provisional_null_masses.append(
                1.0 - min(max(real_mass, 0.0), 1.0)
            )
        allocation = allocate_uncertainty_budget(
            active_answer_only_pids,
            provisional_null_masses,
            [
                (
                    []
                    if int(pid) not in provisional_weights
                    else provisional_weights[int(pid)].detach().cpu().tolist()
                )
                for pid in active_answer_only_pids
            ],
        )
        allocation_mode = (
            "aligned"
            if proposal_allocation_mode == "posterior_uncertainty"
            else "placebo"
        )
        allocated_counts = allocation.counts_by_question(allocation_mode)
        if max(allocated_counts.values()) > proposal_allocation_max_traces:
            raise RuntimeError("uncertainty allocation exceeded its configured cap")
        extra_pids = [
            int(pid)
            for pid in active_answer_only_pids
            for _ in range(
                int(allocated_counts[int(pid)]) - proposal_initial_traces
            )
        ]
        (
            extra_pid_row,
            extra_texts,
            extra_tokens,
            extra_added,
            extra_evicted,
            extra_filter_stats,
        ) = _add_model_traces_to_buffer(
            model,
            tok,
            task,
            buffers,
            extra_pids,
            traces_per_question=1,
            round_added=t,
            source="answer_only_sample",
            buffer_limit=buffer_limit,
            buffer_strategy=buffer_strategy,
            buffer_semantics=buffer_semantics,
            proposal_prompt=answer_only_proposal_prompt,
            proposal_mixture=proposal_mixture,
            proposal_filter=proposal_filter,
            proposal_policy=proposal_policy,
            proposal_temperature=proposal_temperature,
            trace_representation=trace_representation,
            answer_event_mode=answer_event_mode,
            answer_target_termination=answer_target_termination,
            collect_token_counts=diagnostics_fn is not None,
            collect_proposal_outcomes=diagnostics_fn is not None,
            record_proposal_density=False,
            trace_index_offset=len(answer_only_sample_pid_row),
        )
        answer_only_sample_pid_row += extra_pid_row
        answer_only_sample_texts += extra_texts
        answer_only_sample_tokens += extra_tokens
        answer_only_added += extra_added
        answer_only_evicted += extra_evicted
        answer_only_filter_stats = _merge_trace_filter_stats(
            answer_only_filter_stats,
            extra_filter_stats,
        )
        sampling_intervention = {
            "mode": proposal_allocation_mode,
            "initial_traces_per_question": proposal_initial_traces,
            "selected_count_field": allocation_mode,
            "audit": allocation.as_dict(),
        }
    generation_elapsed = time.perf_counter() - generation_started

    total_generated += len(labelled_sample_pid_row) + len(answer_only_sample_pid_row)
    round_rows_added = gold_added + labelled_added + answer_only_added
    round_evictions = (
        labelled_lifecycle_evicted
        + answer_only_lifecycle_evicted
        + gold_evicted
        + labelled_evicted
        + answer_only_evicted
    )
    round_filter_attempted = int(
        labelled_filter_stats["attempted"] + answer_only_filter_stats["attempted"]
    )
    round_filter_accepted = int(
        labelled_filter_stats["accepted"] + answer_only_filter_stats["accepted"]
    )
    round_filter_rejected = int(
        labelled_filter_stats["rejected"] + answer_only_filter_stats["rejected"]
    )
    round_filter_verifier_calls = int(
        labelled_filter_stats["verifier_calls"]
        + answer_only_filter_stats["verifier_calls"]
    )
    round_set_duplicates = int(
        labelled_filter_stats["set_duplicate_count"]
        + answer_only_filter_stats["set_duplicate_count"]
    )
    total_set_duplicates += round_set_duplicates
    total_filter_verifier_calls += round_filter_verifier_calls

    e_step_started = time.perf_counter()
    responsibility_verifier_stats: dict[str, Any] = {
        "policy": (
            responsibility_answer_policy
            if responsibility_score == "rollout_value"
            else "off"
        ),
        "calls": 0,
        "generated_tokens": 0,
        "successes": 0,
        "traces": 0,
    }
    labelled_weights, answer_only_weights = _refresh_minibatch_weights(
        model,
        tok,
        buffers,
        active_labelled_pids,
        active_answer_only_pids,
        responsibility_score=responsibility_score,
        responsibility_posterior=responsibility_posterior,
        responsibility_temperature=responsibility_temperature,
        responsibility_ess_floor=responsibility_ess_floor,
        responsibility_abstention=responsibility_abstention,
        responsibility_rejection_threshold=responsibility_rejection_threshold,
        responsibility_null_log_evidence=responsibility_null_log_evidence,
        responsibility_null_prior=responsibility_null_prior,
        responsibility_policy=responsibility_policy,
        responsibility_answer_policy=responsibility_answer_policy,
        variational_estimator=variational_estimator,
        labelled_numeric_constraint=labelled_numeric_constraint,
        numeric_penalty=numeric_penalty,
        numeric_contradiction_penalty=numeric_contradiction_penalty,
        numeric_missing_penalty=numeric_missing_penalty,
        record_joint_logprobs=diagnostics_fn is not None,
        task=task,
        responsibility_verifier_rollouts=responsibility_verifier_rollouts,
        responsibility_verifier_temperature=responsibility_verifier_temperature,
        responsibility_verifier_max_new_tokens=(
            responsibility_verifier_max_new_tokens
        ),
        responsibility_verifier_batch_size=responsibility_verifier_batch_size,
        responsibility_verifier_smoothing_alpha=(
            responsibility_verifier_smoothing_alpha
        ),
        responsibility_verifier_seed=(
            int(config.seed) * 10_000_019 + int(t) * 10_007 + 97
        ),
        responsibility_verifier_diagnostics=responsibility_verifier_stats,
        verifier_calibration_path=verifier_calibration_path,
        sampled_support_outer_initial=(
            variational_estimator == "sampled_support_importance"
        ),
    )
    if latent_mstep_objective == "segment_responsibility_flow":
        _cache_segment_responsibility_flow(
            model,
            tok,
            buffers,
            active_labelled_pids,
            labelled_weights,
            answer_policy=responsibility_answer_policy,
        )
        _cache_segment_responsibility_flow(
            model,
            tok,
            buffers,
            active_answer_only_pids,
            answer_only_weights,
            answer_policy=responsibility_answer_policy,
        )
    e_step_elapsed = time.perf_counter() - e_step_started
    total_responsibility_verifier_calls += int(
        responsibility_verifier_stats["calls"]
    )
    total_responsibility_verifier_tokens += int(
        responsibility_verifier_stats["generated_tokens"]
    )

    probe_elapsed_before_round = float(
        training_diagnostic_state["probe_elapsed_seconds"]
    )
    m_step_started = time.perf_counter()
    labelled_weights, answer_only_weights, stats = _inner_weighted_em_steps(
        model,
        tok,
        opt,
        task,
        buffers,
        labelled_pids,
        answer_only_pids,
        labelled_weights,
        answer_only_weights,
        inner_steps,
        responsibility_score=responsibility_score,
        responsibility_posterior=responsibility_posterior,
        responsibility_temperature=responsibility_temperature,
        responsibility_ess_floor=responsibility_ess_floor,
        responsibility_abstention=responsibility_abstention,
        responsibility_rejection_threshold=responsibility_rejection_threshold,
        responsibility_null_log_evidence=responsibility_null_log_evidence,
        responsibility_null_prior=responsibility_null_prior,
        responsibility_policy=responsibility_policy,
        responsibility_answer_policy=responsibility_answer_policy,
        responsibility_refresh=responsibility_refresh,
        variational_estimator=variational_estimator,
        labelled_em_weight=labelled_em_weight,
        answer_only_em_weight=answer_only_em_weight,
        policy_kl_coef=policy_kl_coef,
        supervised_weight=supervised_weight,
        policy_anchor_mode=policy_anchor_mode,
        policy_anchor_target_ratio=policy_anchor_target_ratio,
        policy_anchor_beta_min=policy_anchor_beta_min,
        policy_anchor_beta_max=policy_anchor_beta_max,
        policy_anchor_ema=policy_anchor_ema,
        policy_anchor_token_scope=policy_anchor_token_scope,
        policy_anchor_state=policy_anchor_state,
        labelled_numeric_constraint=labelled_numeric_constraint,
        numeric_penalty=numeric_penalty,
        numeric_contradiction_penalty=numeric_contradiction_penalty,
        numeric_missing_penalty=numeric_missing_penalty,
        labelled_supervision=labelled_supervision,
        compact_gold_weight=compact_gold_weight,
        digit_token_weight=digit_token_weight,
        answer_target_termination=answer_target_termination,
        latent_mstep_objective=latent_mstep_objective,
        update_geometry=update_geometry,
        step_acceptance=step_acceptance,
        rollback_tolerance=rollback_tolerance,
        rollback_max_backtracks=rollback_max_backtracks,
        rollback_shrink=rollback_shrink,
        record_joint_logprobs=diagnostics_fn is not None,
        record_gradient_geometry=diagnostics_fn is not None,
        diagnostics_level=diagnostics_level,
        diagnostics_gradient_questions=diagnostics_gradient_questions,
        diagnostics_probe_fn=diagnostics_probe_fn,
        diagnostic_state=training_diagnostic_state,
    )
    m_step_elapsed = time.perf_counter() - m_step_started
    diagnostic_probe_elapsed = (
        float(training_diagnostic_state["probe_elapsed_seconds"])
        - probe_elapsed_before_round
    )
    training_m_step_elapsed = max(
        m_step_elapsed - diagnostic_probe_elapsed,
        0.0,
    )
    gradient_geometry = stats.pop("gradient_geometry", None)
    update_geometry_diagnostics = stats.pop(
        "update_geometry_diagnostics", None
    )
    inner_step_diagnostics = stats.pop("inner_step_diagnostics", [])
    posterior_refresh_diagnostics = stats.pop(
        "posterior_refresh_diagnostics", []
    )
    stats.pop("diagnostics_level", None)
    stats.pop("diagnostics_gradient_questions", None)

    total_steps += int(stats["steps"])
    for pid in set(labelled_pids) | set(answer_only_pids):
        round_evictions += _enforce_buffer_limit(
            buffers[int(pid)], buffer_limit, buffer_strategy
        )
    total_buffer_evictions += round_evictions
    evaluation_started = time.perf_counter()
    test_acc = maybe_eval(
        model,
        t,
        rounds,
        eval_every,
        eval_fn,
        eval_rounds=eval_rounds,
    )
    evaluation_elapsed = time.perf_counter() - evaluation_started


    state.opt = opt
    state.total_generated = total_generated
    state.total_steps = total_steps
    state.total_buffer_evictions = total_buffer_evictions
    state.total_set_duplicates = total_set_duplicates
    state.total_filter_verifier_calls = total_filter_verifier_calls
    state.total_responsibility_verifier_calls = (
        total_responsibility_verifier_calls
    )
    state.total_responsibility_verifier_tokens = (
        total_responsibility_verifier_tokens
    )
    return _ACAlg1RoundOutcome(
        round_started=round_started,
        labelled_pids=labelled_pids,
        answer_only_pids=answer_only_pids,
        labelled_sample_pid_row=labelled_sample_pid_row,
        labelled_sample_texts=labelled_sample_texts,
        labelled_sample_tokens=labelled_sample_tokens,
        labelled_filter_stats=labelled_filter_stats,
        answer_only_sample_pid_row=answer_only_sample_pid_row,
        answer_only_sample_texts=answer_only_sample_texts,
        answer_only_sample_tokens=answer_only_sample_tokens,
        answer_only_filter_stats=answer_only_filter_stats,
        labelled_weights=labelled_weights,
        answer_only_weights=answer_only_weights,
        generation_elapsed=generation_elapsed,
        e_step_elapsed=e_step_elapsed,
        m_step_elapsed=m_step_elapsed,
        diagnostic_probe_elapsed=diagnostic_probe_elapsed,
        training_m_step_elapsed=training_m_step_elapsed,
        evaluation_elapsed=evaluation_elapsed,
        gradient_geometry=gradient_geometry,
        update_geometry_diagnostics=update_geometry_diagnostics,
        inner_step_diagnostics=inner_step_diagnostics,
        posterior_refresh_diagnostics=posterior_refresh_diagnostics,
        stats=stats,
        round_rows_added=round_rows_added,
        round_evictions=round_evictions,
        round_set_duplicates=round_set_duplicates,
        round_filter_attempted=round_filter_attempted,
        round_filter_accepted=round_filter_accepted,
        round_filter_rejected=round_filter_rejected,
        round_filter_verifier_calls=round_filter_verifier_calls,
        responsibility_verifier_stats=responsibility_verifier_stats,
        test_acc=test_acc,
        sampling_intervention=sampling_intervention,
    )


def _record_ac_alg1_round(
    *,
    config: ACAlg1RunConfig,
    state: _ACAlg1RuntimeState,
    outcome: _ACAlg1RoundOutcome,
    task,
    t: int,
    policy_anchor_measured: bool,
    diagnostics_fn,
) -> None:
    algorithm_profile = config.algorithm_profile
    buffer_limit = config.buffer_limit
    labelled_frac = config.labelled_frac
    buffer_strategy = config.buffer_strategy
    buffer_semantics = config.buffer_semantics
    buffer_lifecycle = config.buffer_lifecycle
    proposal_prompt = config.proposal_prompt
    labelled_proposal_prompt = config.labelled_proposal_prompt
    answer_only_proposal_prompt = config.answer_only_proposal_prompt
    proposal_mixture = config.proposal_mixture
    proposal_filter = config.proposal_filter
    proposal_policy = config.proposal_policy
    proposal_temperature = config.proposal_temperature
    responsibility_score = config.responsibility_score
    responsibility_posterior = config.responsibility_posterior
    responsibility_temperature = config.responsibility_temperature
    responsibility_ess_floor = config.responsibility_ess_floor
    responsibility_abstention = config.responsibility_abstention
    responsibility_rejection_threshold = config.responsibility_rejection_threshold
    responsibility_null_log_evidence = config.responsibility_null_log_evidence
    responsibility_null_prior = config.responsibility_null_prior
    responsibility_policy = config.responsibility_policy
    responsibility_answer_policy = config.responsibility_answer_policy
    responsibility_refresh = config.responsibility_refresh
    responsibility_verifier_rollouts = config.responsibility_verifier_rollouts
    responsibility_verifier_temperature = (
        config.responsibility_verifier_temperature
    )
    responsibility_verifier_max_new_tokens = (
        config.responsibility_verifier_max_new_tokens
    )
    responsibility_verifier_batch_size = (
        config.responsibility_verifier_batch_size
    )
    responsibility_verifier_smoothing_alpha = (
        config.responsibility_verifier_smoothing_alpha
    )
    variational_estimator = config.variational_estimator
    labelled_em_weight = config.labelled_em_weight
    answer_only_em_weight = config.answer_only_em_weight
    policy_kl_coef = config.policy_kl_coef
    supervised_weight = config.supervised_weight
    policy_anchor_mode = config.policy_anchor_mode
    policy_anchor_target_ratio = config.policy_anchor_target_ratio
    policy_anchor_beta_min = config.policy_anchor_beta_min
    policy_anchor_beta_max = config.policy_anchor_beta_max
    policy_anchor_ema = config.policy_anchor_ema
    policy_anchor_token_scope = config.policy_anchor_token_scope
    labelled_numeric_constraint = config.labelled_numeric_constraint
    numeric_penalty = config.numeric_penalty
    numeric_contradiction_penalty = config.numeric_contradiction_penalty
    numeric_missing_penalty = config.numeric_missing_penalty
    labelled_supervision = config.labelled_supervision
    compact_gold_weight = config.compact_gold_weight
    digit_token_weight = config.digit_token_weight
    trace_representation = config.trace_representation
    latent_mstep_objective = config.latent_mstep_objective
    answer_event_mode = config.answer_event_mode
    answer_target_termination = config.answer_target_termination
    update_geometry = config.update_geometry
    step_acceptance = config.step_acceptance
    rollback_tolerance = config.rollback_tolerance
    rollback_max_backtracks = config.rollback_max_backtracks
    rollback_shrink = config.rollback_shrink
    optimizer_state_scope = config.optimizer_state_scope
    question_sampling = config.question_sampling
    buffers = state.buffers
    records = state.records
    total_generated = state.total_generated
    total_steps = state.total_steps
    total_buffer_evictions = state.total_buffer_evictions
    total_set_duplicates = state.total_set_duplicates
    total_filter_verifier_calls = state.total_filter_verifier_calls
    total_diagnostic_verifier_calls = state.total_diagnostic_verifier_calls
    total_responsibility_verifier_calls = (
        state.total_responsibility_verifier_calls
    )
    total_responsibility_verifier_tokens = (
        state.total_responsibility_verifier_tokens
    )
    labelled_pool = state.labelled_pool
    answer_only_pool = state.answer_only_pool
    labelled_pids = outcome.labelled_pids
    answer_only_pids = outcome.answer_only_pids
    labelled_sample_pid_row = outcome.labelled_sample_pid_row
    labelled_sample_texts = outcome.labelled_sample_texts
    labelled_sample_tokens = outcome.labelled_sample_tokens
    labelled_filter_stats = outcome.labelled_filter_stats
    answer_only_sample_pid_row = outcome.answer_only_sample_pid_row
    answer_only_sample_texts = outcome.answer_only_sample_texts
    answer_only_sample_tokens = outcome.answer_only_sample_tokens
    answer_only_filter_stats = outcome.answer_only_filter_stats
    responsibility_verifier_stats = outcome.responsibility_verifier_stats
    stats = outcome.stats
    round_filter_accepted = outcome.round_filter_accepted
    round_filter_rejected = outcome.round_filter_rejected
    test_acc = outcome.test_acc

    sample_diagnostics = None
    if diagnostics_fn is not None:
        sample_diagnostics = _sample_diagnostics(
            task,
            labelled_sample_pid_row + answer_only_sample_pid_row,
            labelled_sample_texts + answer_only_sample_texts,
            labelled_sample_tokens + answer_only_sample_tokens,
            (
                labelled_filter_stats["sources"]
                + answer_only_filter_stats["sources"]
            ),
            trace_ids=(
                labelled_filter_stats["trace_ids"]
                + answer_only_filter_stats["trace_ids"]
            ),
            rewards=(
                labelled_filter_stats["rewards"]
                + answer_only_filter_stats["rewards"]
            ),
            admitted=(
                labelled_filter_stats["admitted"]
                + answer_only_filter_stats["admitted"]
            ),
            retained_after_insertion=(
                labelled_filter_stats["retained_after_insertion"]
                + answer_only_filter_stats["retained_after_insertion"]
            ),
            verifier_calls=int(
                labelled_filter_stats["diagnostic_verifier_calls"]
                + answer_only_filter_stats["diagnostic_verifier_calls"]
            ),
        )
        total_diagnostic_verifier_calls += sample_diagnostics["verifier_calls"]

    record = {
        "round": t,
        "oracle": 0,
        "gen": total_generated,
        "llm_gen": total_generated + total_responsibility_verifier_calls,
        "gsteps": total_steps,
        "algorithm_profile": algorithm_profile,
        "buffer_limit": buffer_limit,
        "buffer_strategy": buffer_strategy,
        "buffer_semantics": buffer_semantics,
        "buffer_lifecycle": buffer_lifecycle,
        "buffer_max_age": config.buffer_max_age,
        "buffer_set_duplicates": total_set_duplicates,
        "proposal_prompt": proposal_prompt,
        "labelled_proposal_prompt": labelled_proposal_prompt,
        "answer_only_proposal_prompt": answer_only_proposal_prompt,
        "proposal_mixture": proposal_mixture,
        "proposal_filter": proposal_filter,
        "proposal_policy": proposal_policy,
        "proposal_temperature": proposal_temperature,
        "proposal_allocation_mode": config.proposal_allocation_mode,
        "proposal_initial_traces": config.proposal_initial_traces,
        "proposal_allocation_max_traces": (
            config.proposal_allocation_max_traces
        ),
        "responsibility_score": responsibility_score,
        "responsibility_posterior": responsibility_posterior,
        "responsibility_temperature": responsibility_temperature,
        "responsibility_ess_floor": responsibility_ess_floor,
        "responsibility_abstention": responsibility_abstention,
        "responsibility_rejection_threshold": responsibility_rejection_threshold,
        "responsibility_null_log_evidence": responsibility_null_log_evidence,
        "responsibility_null_prior": responsibility_null_prior,
        "responsibility_policy": responsibility_policy,
        "responsibility_answer_policy": responsibility_answer_policy,
        "responsibility_refresh": responsibility_refresh,
        "responsibility_verifier_rollouts": responsibility_verifier_rollouts,
        "responsibility_verifier_temperature": (
            responsibility_verifier_temperature
        ),
        "responsibility_verifier_max_new_tokens": (
            responsibility_verifier_max_new_tokens
        ),
        "responsibility_verifier_batch_size": (
            responsibility_verifier_batch_size
        ),
        "responsibility_verifier_smoothing_alpha": (
            responsibility_verifier_smoothing_alpha
        ),
        "responsibility_verifier_calls": (
            total_responsibility_verifier_calls
        ),
        "responsibility_verifier_generated_tokens": (
            total_responsibility_verifier_tokens
        ),
        "verifier_calibration_path": config.verifier_calibration_path,
        "responsibility_verifier_calls_this_round": int(
            responsibility_verifier_stats["calls"]
        ),
        "responsibility_verifier_successes_this_round": int(
            responsibility_verifier_stats["successes"]
        ),
        "variational_estimator": variational_estimator,
        "reuse_fresh_traces": config.reuse_fresh_traces,
        "reuse_importance_min": config.reuse_importance_min,
        "reuse_importance_max": config.reuse_importance_max,
        "labelled_em_weight": labelled_em_weight,
        "answer_only_em_weight": answer_only_em_weight,
        "supervised_weight": supervised_weight,
        "policy_kl_coef": policy_kl_coef,
        "policy_kl_measured": policy_anchor_measured,
        "policy_anchor_mode": policy_anchor_mode,
        "policy_anchor_target_ratio": policy_anchor_target_ratio,
        "policy_anchor_beta_min": policy_anchor_beta_min,
        "policy_anchor_beta_max": policy_anchor_beta_max,
        "policy_anchor_ema": policy_anchor_ema,
        "policy_anchor_token_scope": policy_anchor_token_scope,
        "labelled_numeric_constraint": labelled_numeric_constraint,
        "numeric_penalty": numeric_penalty,
        "numeric_contradiction_penalty": numeric_contradiction_penalty,
        "numeric_missing_penalty": numeric_missing_penalty,
        "labelled_supervision": labelled_supervision,
        "compact_gold_weight": compact_gold_weight,
        "digit_token_weight": digit_token_weight,
        "trace_representation": trace_representation,
        "latent_mstep_objective": latent_mstep_objective,
        "answer_event_mode": answer_event_mode,
        "update_geometry": update_geometry,
        "step_acceptance": step_acceptance,
        "rollback_tolerance": rollback_tolerance,
        "rollback_max_backtracks": rollback_max_backtracks,
        "rollback_shrink": rollback_shrink,
        "optimizer_state_scope": optimizer_state_scope,
        "question_sampling": question_sampling,
        "labelled_frac": labelled_frac,
        "buffer_rows": sum(len(rows) for rows in buffers.values()),
        "buffer_evictions": total_buffer_evictions,
        "proposal_filter_verifier_calls": total_filter_verifier_calls,
        "proposal_filter_accepted": round_filter_accepted,
        "proposal_filter_rejected": round_filter_rejected,
        "diagnostic_verifier_calls": total_diagnostic_verifier_calls,
        "labelled_pool": len(labelled_pool),
        "answer_only_pool": len(answer_only_pool),
        "labelled_questions": len(labelled_pids),
        "answer_only_questions": len(answer_only_pids),
        "test_acc": test_acc,
        **stats,
    }
    if outcome.sampling_intervention is not None:
        record["sampling_intervention_mode"] = outcome.sampling_intervention[
            "mode"
        ]
    if sample_diagnostics is not None:
        record.update(
            frac_correct=sample_diagnostics["correct_fraction"],
            fmt=sample_diagnostics["format_fraction"],
            gen_len=sample_diagnostics["mean_tokens"],
        )
    records.append(record)

    outcome.sample_diagnostics = sample_diagnostics
    outcome.record = record
    state.total_diagnostic_verifier_calls = total_diagnostic_verifier_calls


def _emit_ac_alg1_round_diagnostics(
    *,
    config: ACAlg1RunConfig,
    state: _ACAlg1RuntimeState,
    outcome: _ACAlg1RoundOutcome,
    t: int,
    policy_anchor_measured: bool,
    eval_fn,
    diagnostics_fn,
) -> None:
    algorithm_profile = config.algorithm_profile
    buffer_limit = config.buffer_limit
    buffer_strategy = config.buffer_strategy
    buffer_semantics = config.buffer_semantics
    buffer_lifecycle = config.buffer_lifecycle
    proposal_prompt = config.proposal_prompt
    labelled_proposal_prompt = config.labelled_proposal_prompt
    answer_only_proposal_prompt = config.answer_only_proposal_prompt
    proposal_mixture = config.proposal_mixture
    proposal_filter = config.proposal_filter
    proposal_policy = config.proposal_policy
    proposal_temperature = config.proposal_temperature
    responsibility_score = config.responsibility_score
    responsibility_posterior = config.responsibility_posterior
    responsibility_temperature = config.responsibility_temperature
    responsibility_ess_floor = config.responsibility_ess_floor
    responsibility_abstention = config.responsibility_abstention
    responsibility_rejection_threshold = config.responsibility_rejection_threshold
    responsibility_null_log_evidence = config.responsibility_null_log_evidence
    responsibility_null_prior = config.responsibility_null_prior
    responsibility_policy = config.responsibility_policy
    responsibility_answer_policy = config.responsibility_answer_policy
    responsibility_refresh = config.responsibility_refresh
    responsibility_verifier_rollouts = config.responsibility_verifier_rollouts
    responsibility_verifier_temperature = (
        config.responsibility_verifier_temperature
    )
    responsibility_verifier_max_new_tokens = (
        config.responsibility_verifier_max_new_tokens
    )
    responsibility_verifier_batch_size = (
        config.responsibility_verifier_batch_size
    )
    responsibility_verifier_smoothing_alpha = (
        config.responsibility_verifier_smoothing_alpha
    )
    variational_estimator = config.variational_estimator
    labelled_em_weight = config.labelled_em_weight
    answer_only_em_weight = config.answer_only_em_weight
    policy_kl_coef = config.policy_kl_coef
    supervised_weight = config.supervised_weight
    policy_anchor_mode = config.policy_anchor_mode
    policy_anchor_target_ratio = config.policy_anchor_target_ratio
    policy_anchor_beta_min = config.policy_anchor_beta_min
    policy_anchor_beta_max = config.policy_anchor_beta_max
    policy_anchor_ema = config.policy_anchor_ema
    policy_anchor_token_scope = config.policy_anchor_token_scope
    labelled_numeric_constraint = config.labelled_numeric_constraint
    numeric_penalty = config.numeric_penalty
    numeric_contradiction_penalty = config.numeric_contradiction_penalty
    numeric_missing_penalty = config.numeric_missing_penalty
    labelled_supervision = config.labelled_supervision
    compact_gold_weight = config.compact_gold_weight
    digit_token_weight = config.digit_token_weight
    trace_representation = config.trace_representation
    latent_mstep_objective = config.latent_mstep_objective
    answer_event_mode = config.answer_event_mode
    answer_target_termination = config.answer_target_termination
    update_geometry = config.update_geometry
    step_acceptance = config.step_acceptance
    rollback_tolerance = config.rollback_tolerance
    rollback_max_backtracks = config.rollback_max_backtracks
    rollback_shrink = config.rollback_shrink
    optimizer_state_scope = config.optimizer_state_scope
    diagnostics_level = config.diagnostics_level
    diagnostics_trace_tape = config.diagnostics_trace_tape
    model = state.model
    tok = state.tok
    buffers = state.buffers
    total_generated = state.total_generated
    total_steps = state.total_steps
    total_buffer_evictions = state.total_buffer_evictions
    total_set_duplicates = state.total_set_duplicates
    total_filter_verifier_calls = state.total_filter_verifier_calls
    total_diagnostic_verifier_calls = state.total_diagnostic_verifier_calls
    total_responsibility_verifier_calls = (
        state.total_responsibility_verifier_calls
    )
    total_responsibility_verifier_tokens = (
        state.total_responsibility_verifier_tokens
    )
    training_diagnostic_state = state.training_diagnostic_state
    initial_trainable_parameters = state.initial_trainable_parameters
    round_started = outcome.round_started
    labelled_pids = outcome.labelled_pids
    answer_only_pids = outcome.answer_only_pids
    labelled_sample_pid_row = outcome.labelled_sample_pid_row
    labelled_sample_tokens = outcome.labelled_sample_tokens
    labelled_filter_stats = outcome.labelled_filter_stats
    answer_only_sample_pid_row = outcome.answer_only_sample_pid_row
    answer_only_sample_tokens = outcome.answer_only_sample_tokens
    answer_only_filter_stats = outcome.answer_only_filter_stats
    labelled_weights = outcome.labelled_weights
    answer_only_weights = outcome.answer_only_weights
    generation_elapsed = outcome.generation_elapsed
    e_step_elapsed = outcome.e_step_elapsed
    m_step_elapsed = outcome.m_step_elapsed
    diagnostic_probe_elapsed = outcome.diagnostic_probe_elapsed
    training_m_step_elapsed = outcome.training_m_step_elapsed
    evaluation_elapsed = outcome.evaluation_elapsed
    gradient_geometry = outcome.gradient_geometry
    update_geometry_diagnostics = outcome.update_geometry_diagnostics
    inner_step_diagnostics = outcome.inner_step_diagnostics
    posterior_refresh_diagnostics = outcome.posterior_refresh_diagnostics
    stats = outcome.stats
    round_rows_added = outcome.round_rows_added
    round_evictions = outcome.round_evictions
    round_set_duplicates = outcome.round_set_duplicates
    round_filter_attempted = outcome.round_filter_attempted
    round_filter_accepted = outcome.round_filter_accepted
    round_filter_rejected = outcome.round_filter_rejected
    round_filter_verifier_calls = outcome.round_filter_verifier_calls
    responsibility_verifier_stats = outcome.responsibility_verifier_stats
    test_acc = outcome.test_acc
    sample_diagnostics = outcome.sample_diagnostics
    record = outcome.record
    if record is None:
        raise RuntimeError("AC-ALG1 round must be recorded before diagnostics")

    if diagnostics_fn is not None:
        rejected_step_elapsed = sum(
            float(step["rejected_attempt_elapsed_seconds"])
            for step in inner_step_diagnostics
        )
        generated_tokens = sum(
            labelled_sample_tokens + answer_only_sample_tokens
        )
        scored_tokens = sum(
            int(row.span.sum().item())
            for pid in [*labelled_pids, *answer_only_pids]
            for row in buffers[int(pid)]
        )
        backward_tokens = sum(
            int(step["support"]["backward_tokens"])
            for step in inner_step_diagnostics
        )
        backward_eos_tokens = sum(
            int(step["support"].get("backward_eos_tokens", 0))
            for step in inner_step_diagnostics
        )
        diagnostics_fn({
            "schema_version": (
                12
                if algorithm_profile == "l2r_exact_signed_factorial"
                else 11
                if algorithm_profile in {
                    "l2r_bayesian_fusion_pilot",
                    "l2r_uncertainty_allocation_pilot",
                    "l2r_age_one_reuse_pilot",
                    "l2r_small_group_replay_pilot",
                }
                else 10
                if algorithm_profile == "l2r_multi_verifier_pilot"
                else 9
            ),
            "algorithm_profile": algorithm_profile,
            "answer_event_mode": answer_event_mode,
            "answer_target_termination": answer_target_termination,
            "diagnostics_level": diagnostics_level,
            "round": t,
            "completed_rounds": t + 1,
            "minibatch": {
                "labelled_pids": [int(pid) for pid in labelled_pids],
                "answer_only_pids": [int(pid) for pid in answer_only_pids],
            },
            "generation": {
                "proposal_prompt": proposal_prompt,
                "labelled_proposal_prompt": labelled_proposal_prompt,
                "answer_only_proposal_prompt": answer_only_proposal_prompt,
                "proposal_mixture": proposal_mixture,
                "proposal_filter": proposal_filter,
                "proposal_policy": proposal_policy,
                "proposal_temperature": proposal_temperature,
                "trace_representation": trace_representation,
                "this_round": len(labelled_sample_pid_row) + len(answer_only_sample_pid_row),
                "cumulative": total_generated,
                "filter": {
                    "attempted": round_filter_attempted,
                    "accepted": round_filter_accepted,
                    "rejected": round_filter_rejected,
                    "acceptance_fraction": (
                        round_filter_accepted / round_filter_attempted
                        if round_filter_attempted else None
                    ),
                    "verifier_calls_this_round": round_filter_verifier_calls,
                    "verifier_calls_cumulative": total_filter_verifier_calls,
                    "labelled_component_attempted": (
                        labelled_filter_stats["component_attempted"]
                    ),
                    "labelled_component_accepted": (
                        labelled_filter_stats["component_accepted"]
                    ),
                    "answer_only_component_attempted": (
                        answer_only_filter_stats["component_attempted"]
                    ),
                    "answer_only_component_accepted": (
                        answer_only_filter_stats["component_accepted"]
                        ),
                        "boundary_rejected_this_round": int(
                            labelled_filter_stats.get("boundary_rejected_count", 0)
                            + answer_only_filter_stats.get(
                                "boundary_rejected_count", 0
                            )
                        ),
                },
                "samples": sample_diagnostics,
                "diagnostic_verifier_calls_cumulative": total_diagnostic_verifier_calls,
                "sampling_intervention": outcome.sampling_intervention,
            },
            "buffer": _buffer_diagnostics(
                buffers,
                buffer_limit,
                buffer_strategy,
                round_rows_added,
                round_evictions,
                round_index=t,
            ),
            "buffer_evictions_cumulative": total_buffer_evictions,
            "buffer_semantics": buffer_semantics,
            "buffer_lifecycle": buffer_lifecycle,
            "buffer_set_duplicates_this_round": round_set_duplicates,
            "buffer_set_duplicates_cumulative": total_set_duplicates,
            "responsibilities": {
                "score": responsibility_score,
                "posterior": responsibility_posterior,
                "temperature": responsibility_temperature,
                "ess_floor_fraction": responsibility_ess_floor,
                "abstention": {
                    "mode": responsibility_abstention,
                    "rejection_threshold": responsibility_rejection_threshold,
                    "null_log_evidence": responsibility_null_log_evidence,
                    "null_prior": responsibility_null_prior,
                },
                "policy": responsibility_policy,
                "answer_policy": responsibility_answer_policy,
                "refresh": responsibility_refresh,
                "verifier": {
                    "rollouts_per_trace": responsibility_verifier_rollouts,
                    "temperature": responsibility_verifier_temperature,
                    "max_new_tokens": responsibility_verifier_max_new_tokens,
                    "batch_size": responsibility_verifier_batch_size,
                    "smoothing_alpha": (
                        responsibility_verifier_smoothing_alpha
                    ),
                    "policy": responsibility_verifier_stats["policy"],
                    "calls_this_round": int(
                        responsibility_verifier_stats["calls"]
                    ),
                    "calls_cumulative": (
                        total_responsibility_verifier_calls
                    ),
                    "traces_this_round": int(
                        responsibility_verifier_stats["traces"]
                    ),
                    "successes_this_round": int(
                        responsibility_verifier_stats["successes"]
                    ),
                    "generated_tokens_this_round": int(
                        responsibility_verifier_stats["generated_tokens"]
                    ),
                    "generated_tokens_cumulative": (
                        total_responsibility_verifier_tokens
                    ),
                },
                "variational_estimator": variational_estimator,
                "inter_refresh_total_variation_mean": _finite_or_none(
                    stats["responsibility_refresh_total_variation_mean"]
                ),
                "inter_refresh_total_variation_max": _finite_or_none(
                    stats["responsibility_refresh_total_variation_max"]
                ),
                "labelled_numeric_constraint": labelled_numeric_constraint,
                "numeric_penalty": numeric_penalty,
                "numeric_contradiction_penalty": numeric_contradiction_penalty,
                "numeric_missing_penalty": numeric_missing_penalty,
                **_responsibility_diagnostics(
                    buffers,
                    labelled_weights,
                    answer_only_weights,
                    t,
                    tok=tok,
                    include_trace_tape=diagnostics_trace_tape,
                ),
            },
            "posterior_dynamics": {
                "refreshes": posterior_refresh_diagnostics,
                "refresh_count": len(posterior_refresh_diagnostics),
            },
            "inner_m_step": {
                "steps": inner_step_diagnostics,
                "attempted_steps": len(inner_step_diagnostics),
                "accepted_steps": sum(
                    step["status"] == "accepted"
                    for step in inner_step_diagnostics
                ),
                "rejected_steps": sum(
                    step["status"] == "rejected"
                    for step in inner_step_diagnostics
                ),
                "cumulative_accepted_steps": int(
                    training_diagnostic_state["accepted_steps"]
                ),
                "consecutive_rejections": int(
                    training_diagnostic_state["consecutive_rejections"]
                ),
                "rejected_attempt_elapsed_seconds": rejected_step_elapsed,
            },
            "optimizer": _optimizer_diagnostics(
                model,
                initial_parameters=initial_trainable_parameters,
            ),
            "gradient_geometry": gradient_geometry,
            "update_geometry": update_geometry_diagnostics,
            "objective": {
                "B_sup": _finite_or_none(stats["B_sup"]),
                "B_prime_unsup": _finite_or_none(stats["B_prime_unsup"]),
                "B_unsup": _finite_or_none(stats["B_unsup"]),
                "F": _finite_or_none(stats["F"]),
                "policy_kl": _finite_or_none(stats["policy_kl"]),
                "policy_kl_penalty": _finite_or_none(stats["policy_kl_penalty"]),
                "F_anchored": _finite_or_none(stats["F_anchored"]),
                "policy_kl_coef": policy_kl_coef,
                "policy_kl_measured": policy_anchor_measured,
                "policy_anchor_mode": policy_anchor_mode,
                "policy_anchor_target_ratio": policy_anchor_target_ratio,
                "policy_anchor_token_scope": policy_anchor_token_scope,
                "policy_anchor_beta": _finite_or_none(
                    stats["policy_anchor_beta"]
                ),
                "policy_anchor_beta_unclipped": _finite_or_none(
                    stats["policy_anchor_beta_unclipped"]
                ),
                "policy_anchor_beta_clip_fraction": _finite_or_none(
                    stats["policy_anchor_beta_clipped"]
                ),
                "policy_anchor_objective_grad_norm": _finite_or_none(
                    stats["policy_anchor_objective_grad_norm"]
                ),
                "policy_anchor_raw_grad_norm": _finite_or_none(
                    stats["policy_anchor_raw_anchor_grad_norm"]
                ),
                "policy_anchor_applied_grad_norm": _finite_or_none(
                    stats["policy_anchor_applied_anchor_grad_norm"]
                ),
                "policy_anchor_achieved_ratio": _finite_or_none(
                    stats["policy_anchor_achieved_ratio"]
                ),
                "policy_anchor_ema_objective_grad_norm": _finite_or_none(
                    stats["policy_anchor_ema_objective_grad_norm"]
                ),
                "policy_anchor_ema_raw_grad_norm": _finite_or_none(
                    stats["policy_anchor_ema_raw_anchor_grad_norm"]
                ),
                "policy_anchor_beta_min": policy_anchor_beta_min,
                "policy_anchor_beta_max": policy_anchor_beta_max,
                "policy_anchor_ema": policy_anchor_ema,
                "optimizer_state_scope": optimizer_state_scope,
                "labelled_em_weight": labelled_em_weight,
                "answer_only_em_weight": answer_only_em_weight,
                "supervised_weight": supervised_weight,
                "labelled_supervision": labelled_supervision,
                "compact_gold_weight": compact_gold_weight,
                "digit_token_weight": digit_token_weight,
                "trace_representation": trace_representation,
                "latent_mstep_objective": latent_mstep_objective,
                "update_geometry": update_geometry,
                "step_acceptance": step_acceptance,
                "rollback_tolerance": rollback_tolerance,
                "rollback_max_backtracks": rollback_max_backtracks,
                "rollback_shrink": rollback_shrink,
                "update_direction_norm": _finite_or_none(
                    stats["update_direction_norm"]
                ),
                "update_B_sup_coefficient": _finite_or_none(
                    stats["update_B_sup_coefficient"]
                ),
                "update_B_prime_unsup_coefficient": _finite_or_none(
                    stats["update_B_prime_unsup_coefficient"]
                ),
                "update_B_unsup_coefficient": _finite_or_none(
                    stats["update_B_unsup_coefficient"]
                ),
                "candidate_steps": int(stats["candidate_steps"]),
                "rolled_back_candidates": int(
                    stats["rolled_back_candidates"]
                ),
                "rollback_backtracks": int(stats["rollback_backtracks"]),
                "safeguard_acceptance_fraction": _finite_or_none(
                    stats["safeguard_acceptance_fraction"]
                ),
                "accepted_step_scale": _finite_or_none(
                    stats["accepted_step_scale"]
                ),
                "accepted_surrogate_total_delta": _finite_or_none(
                    stats["accepted_surrogate_total_delta"]
                ),
                "accepted_B_sup_delta": _finite_or_none(
                    stats["accepted_B_sup_delta"]
                ),
                "accepted_B_prime_unsup_delta": _finite_or_none(
                    stats["accepted_B_prime_unsup_delta"]
                ),
                "accepted_B_unsup_delta": _finite_or_none(
                    stats["accepted_B_unsup_delta"]
                ),
                "gradient_steps_this_round": int(stats["steps"]),
                "gradient_steps_cumulative": total_steps,
                "test_acc": _finite_or_none(test_acc),
            },
            "behavioural_utility": {
                "fixed_probe_baseline_accuracy": _finite_or_none(
                    training_diagnostic_state["probe_baseline_accuracy"]
                ),
                "fixed_probe_latest_accuracy": _finite_or_none(
                    training_diagnostic_state["probe_previous_accuracy"]
                ),
                "fixed_probe_evaluations_this_round": sum(
                    bool(step["behavioural_probe"]["evaluated"])
                    for step in inner_step_diagnostics
                ),
                "full_validation_accuracy": _finite_or_none(test_acc),
                "full_validation_evaluated": (
                    _finite_or_none(test_acc) is not None
                ),
                "full_validation_policy": (
                    "configured_checkpoint_schedule"
                    if eval_fn is not None else "disabled"
                ),
            },
            "compute": {
                "timings_seconds": {
                    "generation": generation_elapsed,
                    "e_step": e_step_elapsed,
                    "m_step": m_step_elapsed,
                    "m_step_excluding_diagnostic_probe": (
                        training_m_step_elapsed
                    ),
                    "diagnostic_probe": diagnostic_probe_elapsed,
                    "diagnostic_probe_baseline": (
                        float(
                            training_diagnostic_state[
                                "probe_baseline_elapsed_seconds"
                            ]
                        )
                        if t == 0 else 0.0
                    ),
                    "evaluation": evaluation_elapsed,
                    "round_total_before_serialization": (
                        time.perf_counter() - round_started
                    ),
                    "rejected_attempts": rejected_step_elapsed,
                },
                "tokens": {
                    "generated": int(generated_tokens),
                    "verifier_generated": int(
                        responsibility_verifier_stats["generated_tokens"]
                    ),
                    "llm_generated_total": int(
                        generated_tokens
                        + int(
                            responsibility_verifier_stats["generated_tokens"]
                        )
                    ),
                    "scored": int(scored_tokens),
                    "backward": int(backward_tokens),
                    "backward_eos": int(backward_eos_tokens),
                    "forward": None,
                },
                "throughput": {
                    "generated_tokens_per_second": (
                        generated_tokens / generation_elapsed
                        if generation_elapsed > 0 else None
                    ),
                    "verifier_tokens_per_second": (
                        int(responsibility_verifier_stats["generated_tokens"])
                        / e_step_elapsed
                        if e_step_elapsed > 0 else None
                    ),
                    "backward_tokens_per_second": (
                        backward_tokens / training_m_step_elapsed
                        if training_m_step_elapsed > 0 else None
                    ),
                },
                "cuda_memory": cuda_memory_diagnostics(),
                "gpu_utilization_percent": None,
                "power_watts": None,
                "energy_joules": None,
            },
        })



def _run_ac_alg1_round(
    *,
    config: ACAlg1RunConfig,
    state: _ACAlg1RuntimeState,
    task,
    t: int,
    policy_anchor_measured: bool,
    eval_fn,
    diagnostics_fn,
    diagnostics_probe_fn,
    checkpoint_fn,
    log,
) -> None:
    rounds = config.rounds
    L_batch = config.L_batch
    U_batch = config.U_batch
    G_label = config.G_label
    G_answer_only = config.G_answer_only
    inner_steps = config.inner_steps
    lr = config.lr
    buffer_limit = config.buffer_limit
    labelled_frac = config.labelled_frac
    buffer_strategy = config.buffer_strategy
    proposal_prompt = config.proposal_prompt
    labelled_proposal_prompt = config.labelled_proposal_prompt
    answer_only_proposal_prompt = config.answer_only_proposal_prompt
    proposal_mixture = config.proposal_mixture
    proposal_filter = config.proposal_filter
    proposal_policy = config.proposal_policy
    responsibility_score = config.responsibility_score
    responsibility_temperature = config.responsibility_temperature
    responsibility_ess_floor = config.responsibility_ess_floor
    responsibility_policy = config.responsibility_policy
    responsibility_answer_policy = config.responsibility_answer_policy
    responsibility_refresh = config.responsibility_refresh
    labelled_em_weight = config.labelled_em_weight
    answer_only_em_weight = config.answer_only_em_weight
    policy_kl_coef = config.policy_kl_coef
    supervised_weight = config.supervised_weight
    policy_anchor_mode = config.policy_anchor_mode
    policy_anchor_target_ratio = config.policy_anchor_target_ratio
    policy_anchor_beta_min = config.policy_anchor_beta_min
    policy_anchor_beta_max = config.policy_anchor_beta_max
    policy_anchor_ema = config.policy_anchor_ema
    labelled_numeric_constraint = config.labelled_numeric_constraint
    numeric_penalty = config.numeric_penalty
    numeric_contradiction_penalty = config.numeric_contradiction_penalty
    numeric_missing_penalty = config.numeric_missing_penalty
    labelled_supervision = config.labelled_supervision
    compact_gold_weight = config.compact_gold_weight
    digit_token_weight = config.digit_token_weight
    trace_representation = config.trace_representation
    update_geometry = config.update_geometry
    step_acceptance = config.step_acceptance
    rollback_tolerance = config.rollback_tolerance
    rollback_max_backtracks = config.rollback_max_backtracks
    rollback_shrink = config.rollback_shrink
    optimizer_state_scope = config.optimizer_state_scope
    question_sampling = config.question_sampling
    eval_every = config.eval_every
    eval_rounds = config.eval_rounds
    diagnostics_level = config.diagnostics_level
    diagnostics_trace_tape = config.diagnostics_trace_tape
    diagnostics_gradient_questions = config.diagnostics_gradient_questions
    checkpoint_every = config.checkpoint_every
    model = state.model
    tok = state.tok
    opt = state.opt
    labelled_pool = state.labelled_pool
    answer_only_pool = state.answer_only_pool
    labelled_sampler = state.labelled_sampler
    answer_only_sampler = state.answer_only_sampler
    buffers = state.buffers
    records = state.records
    total_generated = state.total_generated
    total_steps = state.total_steps
    total_buffer_evictions = state.total_buffer_evictions
    total_filter_verifier_calls = state.total_filter_verifier_calls
    total_diagnostic_verifier_calls = state.total_diagnostic_verifier_calls
    policy_anchor_state = state.policy_anchor_state
    training_diagnostic_state = state.training_diagnostic_state
    initial_trainable_parameters = state.initial_trainable_parameters

    outcome = _execute_ac_alg1_update(
        config=config,
        state=state,
        task=task,
        t=t,
        policy_anchor_measured=policy_anchor_measured,
        eval_fn=eval_fn,
        diagnostics_fn=diagnostics_fn,
        diagnostics_probe_fn=diagnostics_probe_fn,
    )
    round_started = outcome.round_started
    labelled_pids = outcome.labelled_pids
    answer_only_pids = outcome.answer_only_pids
    labelled_sample_pid_row = outcome.labelled_sample_pid_row
    labelled_sample_texts = outcome.labelled_sample_texts
    labelled_sample_tokens = outcome.labelled_sample_tokens
    labelled_filter_stats = outcome.labelled_filter_stats
    answer_only_sample_pid_row = outcome.answer_only_sample_pid_row
    answer_only_sample_texts = outcome.answer_only_sample_texts
    answer_only_sample_tokens = outcome.answer_only_sample_tokens
    answer_only_filter_stats = outcome.answer_only_filter_stats
    labelled_weights = outcome.labelled_weights
    answer_only_weights = outcome.answer_only_weights
    generation_elapsed = outcome.generation_elapsed
    e_step_elapsed = outcome.e_step_elapsed
    m_step_elapsed = outcome.m_step_elapsed
    diagnostic_probe_elapsed = outcome.diagnostic_probe_elapsed
    training_m_step_elapsed = outcome.training_m_step_elapsed
    evaluation_elapsed = outcome.evaluation_elapsed
    gradient_geometry = outcome.gradient_geometry
    update_geometry_diagnostics = outcome.update_geometry_diagnostics
    inner_step_diagnostics = outcome.inner_step_diagnostics
    posterior_refresh_diagnostics = outcome.posterior_refresh_diagnostics
    stats = outcome.stats
    round_rows_added = outcome.round_rows_added
    round_evictions = outcome.round_evictions
    round_filter_attempted = outcome.round_filter_attempted
    round_filter_accepted = outcome.round_filter_accepted
    round_filter_rejected = outcome.round_filter_rejected
    round_filter_verifier_calls = outcome.round_filter_verifier_calls
    test_acc = outcome.test_acc
    opt = state.opt
    total_generated = state.total_generated
    total_steps = state.total_steps
    total_buffer_evictions = state.total_buffer_evictions
    total_filter_verifier_calls = state.total_filter_verifier_calls

    _record_ac_alg1_round(
        config=config,
        state=state,
        outcome=outcome,
        task=task,
        t=t,
        policy_anchor_measured=policy_anchor_measured,
        diagnostics_fn=diagnostics_fn,
    )
    sample_diagnostics = outcome.sample_diagnostics
    record = outcome.record
    total_diagnostic_verifier_calls = state.total_diagnostic_verifier_calls

    _emit_ac_alg1_round_diagnostics(
        config=config,
        state=state,
        outcome=outcome,
        t=t,
        policy_anchor_measured=policy_anchor_measured,
        eval_fn=eval_fn,
        diagnostics_fn=diagnostics_fn,
    )

    if (checkpoint_fn is not None and checkpoint_every > 0
            and (t + 1) % checkpoint_every == 0 and (t + 1) < rounds):
        checkpoint_fn(model, t + 1)

    policy_suffix = (
        f" KL={record['policy_kl']:.4f} "
        f"beta={record['policy_anchor_beta']:.4g}"
        + (
            f" rho={record['policy_anchor_achieved_ratio']:.3f}"
            if policy_anchor_mode == "grad_ratio" else ""
        )
        if policy_anchor_measured else ""
    )
    log(
        f"  [AC-ALG1 r{t:>3}] "
        f"gen={total_generated:>6} "
        f"F={record['F']:.3f} "
        f"Bsup={record['B_sup']:.3f} "
        f"B'unsup={record['B_prime_unsup']:.3f} "
        f"Bunsup={record['B_unsup']:.3f} "
        f"accept={round_filter_accepted}/{round_filter_attempted} "
        f"H={record['buffer_rows']}"
        f"{policy_suffix}"
    )


    state.opt = opt
    state.total_generated = total_generated
    state.total_steps = total_steps
    state.total_buffer_evictions = total_buffer_evictions
    state.total_filter_verifier_calls = total_filter_verifier_calls
    state.total_diagnostic_verifier_calls = total_diagnostic_verifier_calls


def run_ac_alg1(
    task,
    algorithm_profile: str = "legacy",
    rounds: int = 40,
    L_batch: int = 32,
    U_batch: int = 32,
    G_label: int = 1,
    G_answer_only: int = 1,
    inner_steps: int = 8,
    seed: int = 0,
    lr: float = 1e-4,
    model_name: str = MODEL_NAME,
    model_tok=None,
    length_norm: bool = False,
    buffer_limit: int = 0,
    labelled_frac: float = 0.5,
    buffer_strategy: str = "fifo",
    buffer_semantics: str = "multiset_legacy",
    buffer_lifecycle: str = "persistent",
    buffer_max_age: int = -1,
    proposal_prompt: str = "question",
    labelled_proposal_prompt: str | None = None,
    answer_only_proposal_prompt: str | None = None,
    proposal_mixture: str = "single",
    proposal_filter: str = "all",
    proposal_policy: str = "current",
    proposal_temperature: float = 1.0,
    proposal_allocation_mode: str = "uniform",
    proposal_initial_traces: int = 0,
    proposal_allocation_max_traces: int = 0,
    responsibility_score: str = "joint",
    responsibility_posterior: str = "softmax_entropy",
    responsibility_temperature: float = 1.0,
    responsibility_ess_floor: float = 0.0,
    responsibility_abstention: str = "none",
    responsibility_rejection_threshold: float = 0.0,
    responsibility_null_log_evidence: float = 0.0,
    responsibility_null_prior: float = 0.5,
    responsibility_policy: str = "current",
    responsibility_answer_policy: str = "current",
    responsibility_refresh: str = "inner_step",
    responsibility_verifier_rollouts: int = 0,
    responsibility_verifier_temperature: float = 1.0,
    responsibility_verifier_max_new_tokens: int = 64,
    responsibility_verifier_batch_size: int = 16,
    responsibility_verifier_smoothing_alpha: float = 0.5,
    verifier_calibration_path: str | None = None,
    reuse_fresh_traces: int = 0,
    reuse_importance_min: float = 0.5,
    reuse_importance_max: float = 2.0,
    variational_estimator: str = "delta_joint",
    labelled_em_weight: float = 1.0,
    answer_only_em_weight: float = 1.0,
    policy_kl_coef: float | None = None,
    supervised_weight: float = 1.0,
    policy_anchor_mode: str = "fixed",
    policy_anchor_target_ratio: float | None = None,
    policy_anchor_beta_min: float = 0.0,
    policy_anchor_beta_max: float = 10.0,
    policy_anchor_ema: float = 0.9,
    policy_anchor_token_scope: str = "objective",
    labelled_numeric_constraint: str = "off",
    numeric_penalty: float = 2.0,
    numeric_contradiction_penalty: float = 0.0,
    numeric_missing_penalty: float = 0.0,
    labelled_supervision: str = "gold",
    compact_gold_weight: float = 0.5,
    digit_token_weight: float = 1.0,
    trace_representation: str = "reasoning",
    latent_mstep_objective: str = "joint",
    answer_event_mode: str = "legacy",
    answer_target_termination: str = "none",
    update_geometry: str = "sum",
    step_acceptance: str = "none",
    rollback_tolerance: float = 1e-6,
    rollback_max_backtracks: int = 0,
    rollback_shrink: float = 0.5,
    optimizer_state_scope: str = "persistent",
    question_sampling: str = "random",
    eval_every: int = 0,
    eval_rounds=None,
    eval_fn=None,
    diagnostics_fn=None,
    diagnostics_level: str = "standard",
    diagnostics_trace_tape: bool = False,
    diagnostics_gradient_questions: int = 0,
    diagnostics_probe_fn=None,
    checkpoint_every: int = 0,
    checkpoint_fn=None,
    log=print,
) -> list[dict]:
    """Train with the faithful three-term buffer objective.

    Args:
        task: GSM8K-style task with prompts, gold_answer, and gold_solution.
        algorithm_profile: Replay-compatible legacy behaviour, fail-closed
            source semantics, the common-factorial estimator contract, or a
            named controlled ablation.
        rounds: Number of outer rounds.
        L_batch: Number of labelled questions in each labelled minibatch.
        U_batch: Number of answer-only questions in each answer-only minibatch.
        G_label: Number of model traces sampled per labelled question.
        G_answer_only: Number of model traces sampled per answer-only question.
        inner_steps: Number of objective/weight-refresh steps per outer minibatch.
        seed: Random seed for minibatch sampling.
        lr: Optimizer learning rate.
        model_name: Model name passed to load_model if model_tok is not supplied.
        model_tok: Optional preloaded (model, tokenizer) pair.
        length_norm: Must be False; AC-ALG1 keeps Barber's unnormalised joint log-prob objective.
        buffer_limit: Per-question trace-buffer cap. Values <= 0 mean unbounded.
        labelled_frac: Fraction of answer-known prompts assigned to labelled L rather than answer-only U.
        buffer_strategy: "fifo", "hybrid", or the registered
            "calculation_diverse" support-compression intervention.
        buffer_semantics: Historical multiset replay or a token-unique set B(q).
        buffer_lifecycle: Persistent archive, fresh per-round empirical draws,
            or a frozen fixed proposal bank.
        buffer_max_age: Registered replay horizon for the isolated age-one
            pilot; ``-1`` disables this control everywhere else.
        proposal_prompt: Candidate-generation prompt mode: "question",
            "derive_only", "answer_hint", "answer_derive",
            "answer_derive_concise", "answer_graph_derive",
            "tagged_zero_shot", or "tagged_gold_rationale". Proposal-only
            instructions are not retained in the buffered sequence used for
            scoring or training. This is the fallback for both data pools.
        labelled_proposal_prompt: Optional candidate-generation override for
            labelled L' questions.
        answer_only_proposal_prompt: Optional candidate-generation override for
            answer-only U' questions. Gold-rationale prompting is forbidden.
        proposal_mixture: Candidate-support recipe: ``single``,
            ``question_answer``, or ``question_answer_graph``. Mixture recipes
            preserve an unguided question-only component and affect support
            construction only; the finite-buffer E-step remains unchanged.
        proposal_filter: ``all`` retains every proposal; ``answer_correct``
            retains only proposals that decode to the known final answer;
            ``answer_correct_numeric`` additionally requires at least one
            parsed equation, no invalid equation, and no contradiction with
            the reference calculation.
        proposal_policy: ``current`` samples with the trained adapter;
            ``frozen_base`` disables it during proposal generation.
        proposal_temperature: Positive sampling temperature used for rationale
            proposals. The canonical Q5 setting is 1.0.
        proposal_allocation_mode: Uniform draws, posterior-uncertainty
            allocation, or its cyclically shifted placebo.
        proposal_initial_traces: Per-question first-stage draw count for the
            uncertainty-allocation pilot.
        proposal_allocation_max_traces: Per-question cap after allocation.
        responsibility_score: ``joint`` uses the faithful unnormalised E-step
            logit; ``token_mean`` divides it by the scored sequence length;
            ``rollout_value`` substitutes a free-decoding estimate of
            p(a* | q,h_s) for the teacher-forced reader factor.
        responsibility_posterior: ``softmax_entropy`` uses the ordinary
            one-witness posterior; ``hard_delta_no_entropy`` selects one trace;
            ``two_witness`` uses the registered replicated-answer marginal.
        responsibility_temperature: Positive softmax temperature applied to
            the E-step logits. Values above one smooth responsibilities.
        responsibility_ess_floor: One-sided minimum effective-sample-size
            fraction over finite support. Zero disables adaptive smoothing.
        responsibility_abstention: ``none`` forces unit question mass;
            ``hard_threshold`` skips weak questions; ``null_latent`` assigns
            their unused mass to a frozen null state.
        responsibility_rejection_threshold: Frozen real-log-mean-evidence
            cutoff for ``hard_threshold``.
        responsibility_null_log_evidence: Frozen log-evidence baseline for the
            null state.
        responsibility_null_prior: Frozen prior probability of the null state.
        responsibility_policy: ``current`` scores responsibilities with the
            trained adapter; ``frozen_base`` uses the pretrained base.
        responsibility_answer_policy: Policy used for p(a* | h_s, q).
            ``frozen_base`` freezes only the answer reader while leaving the
            trace-prior factor under responsibility_policy.
        responsibility_refresh: ``inner_step`` recomputes the E-step after
            every optimizer step; ``outer_round`` holds it fixed for the
            complete generalized-EM M-step.
        responsibility_verifier_rollouts: Independent answer continuations
            sampled per trace for ``rollout_value``.
        responsibility_verifier_temperature: Sampling temperature for those
            verifier continuations.
        responsibility_verifier_max_new_tokens: Maximum generated answer
            tokens per verifier continuation.
        responsibility_verifier_batch_size: Generation microbatch for verifier
            continuations.
        responsibility_verifier_smoothing_alpha: Symmetric Beta prior used to
            keep finite-sample verifier logits defined when success count is zero.
        verifier_calibration_path: Frozen count-only calibration artifact for
            the Bayesian verifier-fusion pilot.
        reuse_fresh_traces: Minimum current-policy draws per question after
            the age-one bootstrap round.
        reuse_importance_min: Minimum admitted current-to-behaviour density ratio.
        reuse_importance_max: Maximum admitted current-to-behaviour density ratio.
        variational_estimator: Delta-set joint posterior, uniform Monte Carlo,
            current-prior importance sampling, or frozen-prior importance
            sampling, including the isolated persistent prior correction.
        labelled_em_weight: Coefficient on B'_unsup. Zero removes the labelled
            latent-buffer term while retaining B_sup and B_unsup.
        answer_only_em_weight: Coefficient on B_unsup. Zero removes the
            answer-only latent-buffer term while retaining B_sup and B'_unsup.
        policy_kl_coef: Coefficient on the empirical token-level KL penalty to
            the frozen pretrained policy. ``None`` preserves the original
            implementation without extra scoring; zero measures policy drift
            without changing gradients.
        supervised_weight: Coefficient on B_sup. Zero removes direct
            gold-rationale supervision while retaining both latent-buffer terms.
        policy_anchor_mode: ``fixed`` uses policy_kl_coef; ``grad_ratio``
            chooses a detached per-step beta from gradient norms.
        policy_anchor_target_ratio: Target norm ratio between the applied KL
            gradient and the unanchored objective gradient in grad_ratio mode.
        policy_anchor_beta_min: Lower clip for adaptive beta.
        policy_anchor_beta_max: Upper clip for adaptive beta.
        policy_anchor_ema: EMA decay applied to objective and unit-KL norms.
        policy_anchor_token_scope: ``objective`` anchors every token selected
            by the M-step; ``reasoning`` excludes answer and EOS tokens.
        labelled_numeric_constraint: ``off``, ``hard``, ``soft``, or
            ``graph_hard`` fixed arithmetic potential applied to labelled
            E-step logits only.
        numeric_penalty: Lambda_false, the per-invalid-equation soft penalty.
        numeric_contradiction_penalty: Lambda_contra, the per-gold-contradiction
            soft penalty.
        numeric_missing_penalty: Lambda_miss, the per-missing graph node or edge
            soft penalty.
        labelled_supervision: Gold, fixed compact mixture, set-valued compact
            evidence, or explicit graph-factorized B_sup target.
        compact_gold_weight: Weight on the compact target in the mixed B_sup.
        digit_token_weight: Scale-preserving relative B_sup weight on digit tokens.
        trace_representation: Plain reasoning or explicit z then h serialization.
        latent_mstep_objective: Train the joint h,a continuation, its per-token
            mean variant, the answer factor only, or the rationale factor only
            in latent terms.
        answer_event_mode: Replay-compatible or strict terminal-marker answer
            event shared with evaluation.
        answer_target_termination: ``none`` preserves historical answer targets;
            ``eos`` includes tokenizer EOS in the answer-factor loss.
        update_geometry: Geometry used to combine the three M-step gradients.
        step_acceptance: Fixed-responsibility post-Adam no-harm rule.
        rollback_tolerance: Absolute tolerance in the no-harm comparison.
        rollback_max_backtracks: Number of reduced-size retries after rollback.
        rollback_shrink: Learning-rate multiplier for each retry.
        optimizer_state_scope: ``persistent`` keeps Adam moments across outer
            rounds; ``outer_round`` resets them before each round's M-step.
        question_sampling: Independent random minibatches or reproducible
            shuffled epochs over each labelled/answer-only pool.
        eval_every: Run eval_fn every N rounds and on the final round; zero disables it.
        eval_rounds: Optional explicit completed-round evaluation schedule.
        eval_fn: Optional function mapping the current model to a held-out metric.
        diagnostics_fn: Optional callback receiving one observational diagnostics record per round.
        diagnostics_level: ``standard`` reuses training values; ``deep`` adds
            fixed-surrogate and unweighted-NLL evaluations.
        diagnostics_trace_tape: Include full token ids and masks in the
            compressed per-round diagnostics stream.
        diagnostics_gradient_questions: In deep mode, recompute at most this
            many highest-responsibility question gradients per inner step.
            Enabling it reuses sampled texts and E-step weights; it does not generate or score again.
        diagnostics_probe_fn: Optional fixed held-out probe evaluated before
            training and after every accepted inner M-step in deep mode.
        checkpoint_every: Save an intermediate adapter every N rounds through checkpoint_fn. The
            final round is omitted because the sweep runner saves the final adapter separately.
        checkpoint_fn: Optional callback accepting ``(model, completed_rounds)``.
        log: Logging function.

    Returns:
        List of per-round metric dictionaries.
    """

    run_config = ACAlg1RunConfig.from_call(locals())
    run_config, policy_anchor_measured = _validate_ac_alg1_run_config(
        run_config,
        diagnostics_fn=diagnostics_fn,
        diagnostics_probe_fn=diagnostics_probe_fn,
    )
    _validate_q5_support_task_contract(run_config, task)
    labelled_proposal_prompt = run_config.labelled_proposal_prompt
    answer_only_proposal_prompt = run_config.answer_only_proposal_prompt

    model, tok = model_tok if model_tok is not None else load_model(seed=seed, model=model_name)
    task_answer_event_mode = getattr(task, "answer_event_mode", answer_event_mode)
    if task_answer_event_mode != answer_event_mode:
        raise ValueError(
            "AC-ALG1 and task answer-event modes must match, got "
            f"{answer_event_mode!r} and {task_answer_event_mode!r}"
        )
    if answer_event_mode == "strict_terminal_marker":
        _validate_strict_answer_event_tokenization(tok)
    opt = None
    rng = np.random.default_rng(seed)

    labelled_pool, answer_only_pool = _labelled_answer_only_pools(task, labelled_frac=labelled_frac)
    labelled_sampler = QuestionSampler(labelled_pool, rng, mode=question_sampling)
    answer_only_sampler = QuestionSampler(answer_only_pool, rng, mode=question_sampling)
    buffers: dict[int, list[TraceRow]] = {pid: [] for pid in range(len(task.prompts))}
    records = []
    total_generated = 0
    total_steps = 0
    total_buffer_evictions = 0
    total_set_duplicates = 0
    total_filter_verifier_calls = 0
    total_diagnostic_verifier_calls = 0
    total_responsibility_verifier_calls = 0
    total_responsibility_verifier_tokens = 0
    policy_anchor_state: dict[str, float] = {}
    training_diagnostic_state = {
        "accepted_steps": 0,
        "consecutive_rejections": 0,
        "probe_previous_accuracy": None,
        "probe_baseline_accuracy": None,
        "probe_baseline_elapsed_seconds": 0.0,
        "probe_elapsed_seconds": 0.0,
    }
    initial_trainable_parameters = (
        _snapshot_trainable_parameters(model)
        if diagnostics_fn is not None else None
    )
    if diagnostics_probe_fn is not None:
        probe_started = time.perf_counter()
        probe_baseline = _run_diagnostic_probe(model, diagnostics_probe_fn)
        training_diagnostic_state["probe_previous_accuracy"] = probe_baseline
        training_diagnostic_state["probe_baseline_accuracy"] = probe_baseline
        training_diagnostic_state["probe_baseline_elapsed_seconds"] = (
            time.perf_counter() - probe_started
        )

    state = _ACAlg1RuntimeState(
        model=model,
        tok=tok,
        opt=opt,
        labelled_pool=labelled_pool,
        answer_only_pool=answer_only_pool,
        labelled_sampler=labelled_sampler,
        answer_only_sampler=answer_only_sampler,
        buffers=buffers,
        records=records,
        total_generated=total_generated,
        total_steps=total_steps,
        total_buffer_evictions=total_buffer_evictions,
        total_set_duplicates=total_set_duplicates,
        total_filter_verifier_calls=total_filter_verifier_calls,
        total_diagnostic_verifier_calls=total_diagnostic_verifier_calls,
        total_responsibility_verifier_calls=(
            total_responsibility_verifier_calls
        ),
        total_responsibility_verifier_tokens=(
            total_responsibility_verifier_tokens
        ),
        policy_anchor_state=policy_anchor_state,
        training_diagnostic_state=training_diagnostic_state,
        initial_trainable_parameters=initial_trainable_parameters,
    )
    for t in range(rounds):
        _run_ac_alg1_round(
            config=run_config,
            state=state,
            task=task,
            t=t,
            policy_anchor_measured=policy_anchor_measured,
            eval_fn=eval_fn,
            diagnostics_fn=diagnostics_fn,
            diagnostics_probe_fn=diagnostics_probe_fn,
            checkpoint_fn=checkpoint_fn,
            log=log,
        )

    return state.records
