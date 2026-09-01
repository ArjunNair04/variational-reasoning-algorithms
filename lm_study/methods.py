"""Backward-compat facade. Each trainer lives in its own file; this re-exports them so existing
`from methods import ...` callers keep working.

    AC-EM (active)   ac_em.run_ac_em             answer-conditioned: RW (eq8) / AW / AAW weights (strategy-param)
    AC-ALG1          ac_alg1.run_ac_alg1         Barber L2R Algorithm 1 three-term buffer objective
    scalable L2R     l2r.run_l2r                 isolated reasoning generator + optional frozen answer reader
    conditional VRO  conditional_vro             Barber VRO equations 58--62 for LM responses
    weighted-EM      weighted_em.run_weighted_em LEGACY verifier-reward family (AW/AAW/RW), not run
    GRPO / RLOO      grpo / rloo                 RL  (flat functions)
    RAFT             raft                        best-of-n SFT  (flat function)
    DPO / APL        dpo / apl                   PO  (share preference.PreferenceTrainer: the DPO loss + ref)
Shared sampling / round loop / preference pairing live in common.py.
"""
from common import sample_multi, sample_round, pairs_by_reward, MAX_NEW  # noqa: F401 (re-export)
from ac_em import run_ac_em        # noqa: F401
from ac_alg1 import run_ac_alg1    # noqa: F401
from l2r import run_l2r            # noqa: F401
from ac_dpo import run_ac_dpo      # noqa: F401  (answer-conditioned DPO: verifier-free contrastive M-step)
from conditional_vro import run_conditional_vro  # noqa: F401
from weighted_em import run_weighted_em   # noqa: F401
from grpo import run_grpo        # noqa: F401
from raft import run_raft        # noqa: F401
from rloo import run_rloo        # noqa: F401
from dpo import run_dpo          # noqa: F401
from apl import run_apl          # noqa: F401
