# Q5 fixed-weight continuation kernels

Run `b6c4e921` is a three-seed development screen, separate from the prefix
proposal study `f812ca40`. Only `Q5-KERNEL25` is new. The control is the existing
`Q5-INDEPENDENT16` on seeds 1201, 1213 and 1217 (7483302 tasks 1-3).
No learned posterior network is implemented by this study.

## Question

Does training on a distribution of nearby continuations help more than placing
all of each Q5 component's mass on its single stored trace?

For question x and known answer a, let h_s be an ordinary Q5 buffer trace and
w_s its normalised joint-score weight, computed before the parameter update.
Let a+ include the answer and EOS. The original distribution is

    q(h) = sum_s w_s delta(h = h_s).

The treatment is

    q_epsilon(h) = sum_s w_s [(1-epsilon) delta(h = h_s) + epsilon K_s(h)],
    epsilon = 0.25.

K_s retains the first floor(n_s/2) native reasoning tokens of h_s, where n_s
excludes the answer marker, answer and EOS. It samples the remaining tokens
from the current model under the same answer-derived proposal prompt as Q5.
The model is held fixed while all these continuations are generated.

The full sampled completion is converted to a rationale using Q5's existing
first-marker native-token mapping. An unusable boundary maps back to h_s,
without retrying or conditioning on answer correctness. An empty usable prefix
also gives K_s = delta(h = h_s). Thus K_s is the normalised distribution induced
by generation followed by this mapping, not the raw suffix probability.

For one draw H_s from each nonzero-weight K_s, the M-step estimator is

    J_hat(theta) = sum_s w_s [0.75 log p_theta(a+, h_s | x)
                            + 0.25 log p_theta(a+, H_s | x)].

The outer average is over the same training questions as Q5. The inherited
weights and sampled continuations are detached and fixed for the one Adam
step. This is an unbiased estimator of the fixed-distribution objective, not
an exact evaluation of its expectation. The entropy of q_epsilon is constant
with respect to this M-step's theta; it is not estimated or claimed to improve.
Choosing the kernel at the E-step does not guarantee an improved bound.

## What remains fixed

- Qwen3-1.7B-Base; rank-16 attention-plus-MLP LoRA; same three-shot prompts.
- 128 training questions, 32 rounds, four questions/round, 16 ordinary proposals.
- Persistent unique FIFO buffer of capacity 16 per question.
- Answer-derived proposal; current joint E-step; no ESS or KL modification.
- One Adam step/round at 1e-5; rationale, marker, known answer and EOS targets.
- Question-only scoring, training and validation contexts.
- Same 400 training-derived validation questions; official test is not opened.

Kernel children never enter the persistent buffer and never participate in a
Q5 softmax. Duplicate children keep the mass of the components that drew them.
The original buffer remains the support for the subsequent E-step refresh.

## Cost and randomness

The treatment adds at most one suffix draw per positive-weight buffer trace
per visited question: up to 2048 extra draws over the run. It trains both parent
and child targets, so backward work can approach twice the original. Reused
prefix tokens count toward backward work but not newly generated tokens.
This tests a richer distribution with additional computation; it does not
establish a compute-matched efficiency improvement.

Kernel generation uses a dedicated SHA-256-derived seed for each outer round.
CPU and CUDA RNG states are restored afterwards. Epsilon zero takes the original
path without sampling, additional model calls, changed rows or changed weights.
The default M-step is compared against the deployed control source in tests.

## Validation and analysis

The generated YAML and its hash are frozen before submission. The prepared
analyzer checks both studies' receipts, source revisions, same-seed question
schedules, software/CUDA versions, validation support, all checkpoints, final
adapters, terminal logs and untruncated artifacts. It rejects changed parent
mass, child rescoring, child persistence, missing generation cost and broken
native-token lineage. The new array contains only three tasks; the validator
waits for both the new payload and existing control array.

Report Final extracted Acc@1, strict final accuracy and extracted trajectory
AUC, in that order. Include all three paired differences and descriptive paired
95% t intervals, plus strict AUC, fallback mass, actual changed mass, parent
weight concentration, generated tokens, backward tokens and accelerator hours.
The frozen historical round-zero values are used only to initialise AUC.

A positive mean final difference with no mean strict-final loss nominates this
variant for discussion. All outcomes and costs are reported, with no automatic
confirmation jobs. Three reused development seeds are not a new confirmation.

## Files

- `lm_study/ac_alg1_component_kernel.py`: isolated component construction.
- `lm_study/generate_qwen3_17b_q5_component_kernel.py`: protocol generator.
- `lm_study/experiments_qwen3_17b_q5_component_kernel.yaml`: frozen settings.
- `analysis/analyze_qwen3_q5_component_kernel.py`: validation and paired analysis.
- `lm_study/submit_qwen3_17b_q5_component_kernel_ucl.sh`: held submission.

Run the generator with `--check ... --runtime-check`, then the analyzer with
`--validate-design-only`. The submitter rechecks all coordinates, immutable
source and control identity. Publish the held job IDs before releasing only
the new payload. Never modify the existing prefix jobs to run this treatment.
