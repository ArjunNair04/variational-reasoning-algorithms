# Q5 positional rationale-prior weighting

Status: [token-resolved scoring audit completed and validated](scoring_results_2026-09-14.md);
the user approved the nine-task marker-fixed training screen on 14 September.
Payload **7413571** (tasks 1-9) has completed. Validator **7413572** passed at
20:36 BST on 14 September, confirming all nine tasks and adapters. All payload
logs have terminal markers and no failure signatures. Approved compact
collection and the frozen paired analysis are complete; see the
[completed results](results_2026-09-14.md). Second-half attenuation is the sole
screen nominee, with an unresolved +1.08 pp final gain over historical Q5.
The user explicitly approved publication of job IDs and AMN paths, and both
tracking pushes succeeded before release. Execution remains
at `037b1d3`; there is no artificial concurrency limit. The prepared analyzer
reports all six paired contrasts after complete task-level validation.
The preceding
[pre-training audit](pretraining_findings_2026-09-14.md) records the semantic
review and historical saved-score comparison. Job 7412300 adds fixed-final-
checkpoint scoring on all 384 saved supports. The original partial manifest
is preserved, including its blocked-training boolean. The new submission gate
revalidates all three scoring receipts, logs and token tapes before submission.

## Question

Does reducing the prior contribution of early versus late rationale tokens
change which traces Q5 learns from, and improve final answer accuracy?

This is a positional experiment. It does not assume that the first half of a
rationale represents understanding and the second half represents calculation.
The initial text-only audit inspected 300 saved final evaluation completions
from historical moving Q5, across seeds 1201, 1213 and 1217. In 232 completions,
the first line already contained an equation sign; 176 contained an explicit
numeric operation, and 32 rationales occupied one line. These are syntactic
counts, not semantic labels or an audit of training buffers. The source hashes
and per-completion records are in `existing_trace_audit.json`.
The subsequent outcome-blind review labels 24 complete traces: seven have a
separate initial planning/interpretation block; seventeen calculate on the first
line. On 384 archived training supports, uniform075 changes only five winners.
These findings support treating the proposed split as positional, not semantic.

## Fixed design

| Cell | Early exponent | Late exponent | New tasks |
|---|---:|---:|---:|
| Historical moving Q5 | 1 | 1 | 0 |
| Q5-EARLY-HALF | 0.5 | 1 | 3 |
| Q5-LATE-HALF | 1 | 0.5 | 3 |
| Q5-UNIFORM075 | 0.75 | 0.75 | 3 |

Run `c7e32a91` contains nine new tasks, using seeds 1201, 1213 and 1217.
It supersedes the unsubmitted `a2f46c91` draft, whose halves included the marker.
The control is `Q5-AD-M-LR1e-5-U1-K16` in validated historical run `68078ecc`.
It is a historical comparator, not a contemporaneous or bit-exact GPU replay.

All cells retain the moving reader, answer-derived proposals, persistent unique
FIFO buffer of 16 traces, complete joint weighted M-step, learning rate 1e-5,
128 optimization questions, the fixed 400-question train-derived validation
pool, three-shot prompts, rank-16 attention-plus-MLP LoRA, 32 rounds, 2048
training generations and 32 optimizer steps. No official test data are used.
There is no new ESS adjustment, proposal temperature, KL penalty, prompt format,
M-step masking or generation stopping rule.

## Scoring

For a trace with n reasoning positions, the early segment contains the first
floor(n/2); the late segment contains the remainder. Prompt, padding, `####`,
answer digits and EOS positions are excluded from both halves. The historical
prior mask includes `####`; the new rule separates that suffix using the native
stored reasoning-token count and retains its probability at full weight.
The boundary is computed from rationale positions, not padded sequence length.
A one-token rationale belongs entirely to the late segment; an empty rationale
has two empty segments. No text or token is removed from the conditioning
prefix, and neither segment is independently retokenized.

Let H and T be the sums of token log probabilities in the early and late
segments. Let A be the answer-reader log probability of the known answer
target (space, answer digits and EOS). Let M be the marker log probability.
The E-step uses:

    score = A + M + alpha * H + beta * T
    weight = exp(score - max_score) / sum(exp(score - max_score))

Weights are detached before the unchanged joint M-step. This is a modified
finite-buffer weighting rule, not an importance correction or an exact
posterior. The neutral (1,1) setting takes the original branch without an
additional model forward. Equal non-unit exponents retain the existing
uniform-reasoning-prior arithmetic, with M unchanged. All three new cells log
both segment factors and M. The two additional scoring passes preserve the
PyTorch CPU/CUDA RNG states. Neither pass changes the M-step token mask.

Uniform075 matches the nominal average exponent exactly at even lengths.
For odd lengths, early and late differ by one token. It also does not match
the realized sum of log probabilities: that difference is the intervention.
The analyzer reports segment counts, factor sums and odd-length frequency.

## Scoring-only continuation

`scoring_manifest.json` binds three final adapters and the archived prompt and
training-diagnostic files. No model state from intermediate training rounds was
saved. The audit therefore retokenizes saved candidate text and scores every
support under its seed's final checkpoint, held fixed. It does not replay the
historical per-round scores or generate new candidates.

`analysis/score_q5_prior_segments.py` reconstructs each retained candidate with
the native strict-Q5 prompt/gold-answer/EOS builder. One unpadded forward per
candidate yields all token log probabilities. Both mask scopes compare the same
four settings: historical prior (including the marker), and pure reasoning with
marker probability added unchanged. All 384 supports and 2,987 trace rows are
included; the three-task H100 array is uncapped. There is no optimizer, backward
pass, dataset loader or generation call. The original adapters remain unchanged.

`analysis/analyze_q5_segment_scoring.py` requires three complete receipts and
terminal logs, verifies every token-mask/factor/softmax identity, checks source
support coverage, and reports all six pairwise ranking contrasts and marker
sensitivity. Its completion does not automatically release the training array.

## Analysis and decisions

Report Final extracted Acc@1, strict final Acc@1, then normalized extracted
AUC over rounds 0-32. Include strict AUC, backward tokens, training generations,
optimizer steps and accelerator-hours. Evaluation time is included in the
recorded accelerator-hours, and the additional scoring passes may add runtime.

The six paired contrasts are each new cell versus historical moving Q5,
early versus late, and each positional cell versus concurrent uniform075.
Report all three seed differences and descriptive seed-bootstrap intervals.
These reused development seeds cannot establish a confirmatory improvement.

At most one setting can be nominated for discussion: it must improve mean
final extracted accuracy without reducing mean strict final accuracy against
historical Q5. Rank eligible settings by final accuracy, then strict final,
then AUC, with cell ID resolving an exact tie. AUC is not a veto. A positional
effect specifically requires examining the contrasts with uniform075, not
only the historical control. No further training is launched automatically.

The validator requires nine terminal task logs, complete hashed receipts,
expected adapters, the exact execution/configuration identity, the frozen
checkpoint schedule, 32 diagnostic rounds and budget checks. For every logged
trace, it checks that segment counts partition the reasoning, reasoning and
marker factors sum to the historical prior, logits follow the declared exponents, and recorded weights
match the softmax. Diagnostics include effective support, largest weight,
weight-length correlation and top-trace changes against joint and uniform075
scores on the same buffer.

## Reproduce

The read-only pre-training audit requires the archived dump/diagnostic root
and its completion-receipt root. It creates a new directory and refuses to
overwrite an earlier audit:

```bash
python analysis/audit_q5_prior_segments_pretraining.py \
  --artifact-dir "$ARTIFACT_DIR" --receipt-dir "$RECEIPT_DIR" \
  --annotations docs/experiments/qwen3_q5_prior_segments/semantic_annotations.json \
  --output-dir "$NEW_AUDIT_DIR"
```

From the repository root, with the verified experiment environment:

```bash
export PYTHONPATH="$PWD/src:$PWD/lm_study:$PWD/analysis:${PYTHONPATH:-}"
python lm_study/generate_qwen3_17b_q5_prior_segments.py \
  --check lm_study/experiments_qwen3_17b_q5_prior_segments.yaml --runtime-check
python -m pytest
bash lm_study/submit_qwen3_17b_q5_prior_segments_ucl.sh
```

Before submission, the checksum-bound training gate reruns the completed
token-resolved audit against its original inputs and compares receipt identities.
The frozen configuration records the explicit marker-fixed choice separately
from the old partial audit. The archived total scores cannot identify
early and late scores; do not manufacture them by dividing by token count.
Submission is AMN-only after a pushed account-specific coordination lease,
immutable source deployment, live quota/scheduler checks and receipt-bound
control verification. The submitter creates a held nine-task payload and a
dependent validator, without a concurrency ceiling. Record their IDs and push
tracking before releasing the payload. Do not change the execution checkout
while jobs are active. After validation, invoke
`analysis/analyze_qwen3_q5_prior_segments.py` with the config, result and control
roots, both validator markers, expected execution commit and source job, and a
new timestamped output directory.
