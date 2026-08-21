"""Numerical kernels for variational reasoning experiments."""

from .em import (
    NullLatentPosterior,
    UniqueFIFOSupport,
    centered_trace_credit,
    importance_weights,
    joint_weights,
    null_latent_weights,
    pis_weights,
    q5_weights,
    uniform_weights,
    weighted_joint_loss,
)
from .self_training import Candidate, select_correct, star_examples
from .policy_gradient import (
    grpo_advantages,
    grpo_loss,
    rloo_advantages,
    rloo_loss,
)
from .trice import (
    Chain,
    Proposal,
    ScoreTerm,
    Transition,
    control_variate_terms,
    trice_step,
)

__all__ = [
    "Chain",
    "Candidate",
    "NullLatentPosterior",
    "Proposal",
    "ScoreTerm",
    "Transition",
    "UniqueFIFOSupport",
    "centered_trace_credit",
    "control_variate_terms",
    "grpo_advantages",
    "grpo_loss",
    "importance_weights",
    "joint_weights",
    "null_latent_weights",
    "pis_weights",
    "q5_weights",
    "rloo_advantages",
    "rloo_loss",
    "select_correct",
    "star_examples",
    "trice_step",
    "uniform_weights",
    "weighted_joint_loss",
]
