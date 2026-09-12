# Q5 prior exponent and answer reader

Results: [three-seed screen, 13 September 2026](results_2026-09-13.md).

Run `e97c3a20` tests four settings on paired seeds 1201, 1213 and 1217:

| Tasks | Prior exponent | Answer reader |
|---|---:|---|
| 1-3 | 0.5 | moving |
| 4-6 | 0 | moving |
| 7-9 | 0.5 | frozen base |
| 10-12 | 0 | frozen base |

For each retained rationale h, the E-step logit is
`answer_logprob(reader, answer+EOS | question,h) + tau * trace_logprob(current, h | question)`.
The proposal remains answer-derived. Only the answer factor is frozen in the
frozen-reader arms. The generator and complete-data joint M-step remain trainable.
This is neither proposal importance correction nor a sampling-temperature sweep.

Controls are the matching moving and frozen Q5 cells in validated run `68078ecc`.
They are not resubmitted. The screen preserves their three-shot prompt, question
selection, 128-question optimisation pool, 400-question train-derived validation
pool, 32 rounds, 2,048 proposals, 32 optimiser steps, buffer size 16, persistent
unique buffer and rank-16 attention-plus-MLP LoRA. The official test is not used.

Report Final extracted Acc@1 first, strict final accuracy second and normalized
trajectory AUC third. Pair by seed and reader; retain all four outcomes and each
seed's difference. Three-seed bootstrap intervals are descriptive, not confirmation.
Nominate at most one setting with positive mean final improvement and no mean
strict-final loss against its reader control. Rank eligible settings by final
improvement, breaking ties with strict final and AUC. Confirmation requires a
separate decision; the scheduler does not launch it automatically.

Record E-step factors, responsibility concentration, length correlation and
whether the exponent changes the top trace on the same buffer. Also retain
generation counts, optimiser steps, backward-token diagnostics and accelerator
hours. Historical controls are receipt-verified, and the default tau=1 path is
identity-tested; reuse does not promise bitwise cross-CUDA reproduction.

## Execution

- Generator: `lm_study/generate_qwen3_17b_q5_prior_exponent.py`
- Frozen YAML: `lm_study/experiments_qwen3_17b_q5_prior_exponent.yaml`
- Submitter: `lm_study/submit_qwen3_17b_q5_prior_exponent_ucl.sh`
- Validator: `lm_study/validate_qwen3_17b_q5_prior_exponent.py`
- Analyzer: `analysis/analyze_qwen3_q5_prior_exponent.py`

The submitter verifies control receipts before creating an uncapped, held
12-task H100 array and dependent validator. Release follows the tracking
transaction. Existing results and other jobs must remain untouched.
