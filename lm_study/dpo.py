"""DPO -- direct preference optimisation (Rafailov et al. 2023; ref TRL DPOTrainer, sigmoid loss).

Preferences SYNTHESISED from the reward oracle (chosen=best, rejected=worst of the G samples).
offline=False -> online iterative (regenerate each round); offline=True -> one pair set from the BASE
policy, reused. The loss / reference / update live in PreferenceTrainer; this only gathers the pairs.
"""
from __future__ import annotations
from contextlib import nullcontext

from common import (MODEL_NAME, sample_round, pairs_by_reward, comp_len, frac_correct,
                    format_rate, mean_token_lp)
from preference import PreferenceTrainer


class _DPOTrainer(PreferenceTrainer):
    def __init__(self, *args, offline=False, **kw):
        super().__init__(*args, **kw)
        self.offline, self.cache = offline, None

    def gather_round(self, t, B, G):
        if self.offline and self.cache is not None:                # reuse the fixed offline pair set
            return self.cache
        ctx = self.model.disable_adapter() if self.offline else nullcontext()   # offline: pairs from BASE
        with ctx:
            pids, pid_row, ids, mask, rew, texts = sample_round(self.model, self.tok, self.task, B, G, self.rng)
        self.oracle += B; self.gen += B                            # online DPO: every sample scored
        info = dict(mean_reward=float(rew.mean()), frac_correct=frac_correct(rew), gen_len=comp_len(mask),
                    fmt=format_rate(texts),                        # parseable-'####' rate on the round's samples
                    samp_lp=mean_token_lp(self.model, ids, mask))  # policy-entropy proxy (current policy on them)
        ci, li = pairs_by_reward(pids, pid_row, rew)
        if len(ci) == 0:
            return None, info
        cw_ids, cw_mask = ids[ci.copy()], mask[ci.copy()]
        cl_ids, cl_mask = ids[li.copy()], mask[li.copy()]
        ref_c, ref_l = self.ref_logprobs(cw_ids, cw_mask), self.ref_logprobs(cl_ids, cl_mask)
        out = ((cw_ids, cw_mask, cl_ids, cl_mask, ref_c, ref_l), info)
        if self.offline:
            self.cache = out
        return out


def run_dpo(task, rounds=40, B=64, G=4, seed=0, lr=1e-6, beta=0.1, epochs=1, offline=False,
            model_name=MODEL_NAME, model_tok=None, micro=2,
            eval_every=0, eval_fn=None, log=print):    # micro=2: chosen+rejected graphs
    tr = _DPOTrainer(task, model_tok, seed=seed, lr=lr, beta=beta, epochs=epochs, micro=micro,
                     model_name=model_name, eval_every=eval_every, eval_fn=eval_fn, log=log, offline=offline)
    return tr.run(rounds, B, G, tag="DPO")
