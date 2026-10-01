# Q5 Prefix-Continuation Screen

## Question

Can Q5 learn more from its candidate budget by revisiting the beginning of a
sampled solution and drawing a different continuation?

This is the first test motivated by a richer distribution over traces. It
changes how candidate traces are found. Q5 still represents its training
distribution as weighted point masses on the traces in its buffer.

## Two Cells

| Cell | Candidates per question | Seeds |
| --- | --- | --- |
| Q5-INDEPENDENT16 | 16 independent full generations | 1201, 1213, 1217 |
| Q5-PREFIX8x2 | 8 independent full roots, plus 1 continuation per root | 1201, 1213, 1217 |

Both use the answer-derived proposal prompt, current answer reader, unchanged
joint Q5 weights and full joint M-step. The buffer is a persistent, token-unique
FIFO with capacity 16 per question. There are 32 rounds, four questions per
round, one Adam update per round at learning rate 1e-5, and 128 training
questions. Both use Qwen3-1.7B-Base, rank-16 attention-plus-MLP LoRA, three-shot
prompts, answer-plus-EOS targets and the existing 400-question train-derived
validation pool. No official test access is permitted.

The treatment keeps every root, regardless of its answer or score. For each
root it retains the first floor(n/2) native reasoning tokens, where n excludes
the first answer marker and everything after it. A token overlapping the marker
is excluded. It appends these token IDs directly to the same proposal prompt,
then samples a fresh suffix. Missing markers or empty usable prefixes produce
an independent fallback draw. The full root is generated and charged before
this split; this is not an online stopping rule or a semantic phase detector.

## Unchanged Objective

For every buffered trace h_s, the E-step uses

    w_s = softmax_s(log p_theta(h_s | q) + log p_theta(a, EOS | q, h_s)).

The M-step maximizes the weighted joint log probability of the complete
rationale and the known answer followed by EOS, treating the weights as fixed
within the update. Both the copied prefix and the newly sampled suffix remain
inside the scored and trained rationale. There is no proposal-density division,
KL addition, ESS floor, correctness filter or frozen-reader change.

Shared-prefix children are correlated draws. The experiment therefore measures
a candidate-discovery strategy, not a new unbiased importance estimator.

## Budget and Analysis

Each cell produces 2,048 complete candidates. Each child keeps the original
whole-completion cap: suffix allowance equals cap minus retained-prefix length.
Copied tokens are not new generation, but still incur prompt processing and
backpropagation cost. Report generated suffix/root tokens, copied tokens,
backward tokens, accelerator hours and timing separately. This is not an
equal-compute comparison. Different suffix caps can reduce batching efficiency.

Report Final extracted Acc@1, strict final accuracy, and normalized Acc@1 AUC
over rounds 0-32, in that order. The frozen YAML contains receipt-checked
historical baseline values for round zero, their evaluation hashes and the
validation-question-set hash. Both training cells run concurrently. Report all
three paired seed differences and descriptive 95% paired t intervals. A
positive mean final change without a mean strict-final loss nominates the
treatment for discussion only. AUC and compute trade-offs remain visible.

The analyzer also checks all six terminals and hashed completion receipts,
adapters, checkpoint schedule, 32 diagnostic rounds, budget, unchanged scoring,
native-prefix lineage and generated/copied-token identities. Parent hashes,
fallbacks, buffer duplicates and posterior concentration are logged without
additional random draws. Invalid or incomplete output fails validation.

## Execution

The generated YAML is `lm_study/experiments_qwen3_17b_q5_prefix_continuations.yaml`.
Edit its generator, not the YAML. The submitter performs runtime validation and
all six task-coordinate dry runs, then submits a held H100 array and dependent
CPU validator/analyzer. There is no artificial concurrency cap. Publish the
immutable commit, YAML hash and job IDs before releasing the payload. A
successful local test is not evidence that cluster training has run.
