# Frozen-reader replication of positional Q5 weighting

Run `7a31c9e2` contains nine new tasks, authorized on 15 September 2026.
No standard-Q5 or moving-reader controls are rerun.

Submitted and released as payload **7414007** (tasks 1-9), with dependent validator
**7414008**, at immutable execution `2927112`. No concurrency cap. All local
tests and cluster preflight gates passed, including revalidation of the nine
moving comparison tasks and adapters. Both tracking pushes and canonical MLflow
refresh succeeded before release. All nine tasks were verified ordinary queued;
the validator remains dependency-held. This is not a completion claim.

| Tasks | Cell | First-half weight | Second-half weight | Seeds |
|---|---|---:|---:|---|
| 1-3 | Q5-EARLY-HALF-F | 0.5 | 1 | 1201, 1213, 1217 |
| 4-6 | Q5-LATE-HALF-F | 1 | 0.5 | 1201, 1213, 1217 |
| 7-9 | Q5-UNIFORM075-F | 0.75 | 0.75 | 1201, 1213, 1217 |

## Single change

The E-step answer likelihood now comes from the frozen base model. The
current model still generates proposals and scores the rationale and marker.
The complete joint M-step still trains on both rationale and answer-plus-EOS.
Freezing the reader does not freeze generation or remove answer training.

For each candidate, let H and T be current-policy log probabilities summed
over the first floor-half and remaining reasoning tokens. Let M be the
current-policy log probability of the retained `####` marker, and A0 the
frozen-base log probability of the gold-answer digits and EOS. Then:

    score = A0 + M + alpha*H + beta*T
    responsibility = softmax(score within this question's buffer)

The responsibilities are detached before the original joint weighted-MLE
update. Marker, answer and EOS factors retain unit weight. This is a positional
split, not evidence for separate understanding and calculation phases.

All other settings match `c7e32a91`: answer-derived proposals, current-policy
rationale scores, 16-trace persistent token-unique FIFO buffer, one M-step per
round, learning rate 1e-5, 32 rounds, 2,048 training generations, 32 optimizer
steps, 128 optimization questions, 400 train-derived validation questions,
three-shot prompts and rank-16 attention-plus-MLP LoRA. No KL, ESS adjustment,
sampling-temperature change, early stopping or official-test access.

## Controls and prepared analysis

The exact frozen-reader control is `Q5-AD-F-LR1e-5-U1-K16` in validated run
`68078ecc`. The same seeds' moving-reader control is retained for context.
The three completed moving positional cells come from validated `c7e32a91`.
Source commits, configuration hashes, receipts, adapters, terminal logs and
validation question identities are verified before using those comparisons.

The shared analyzer resolves only the two exact generated positional YAMLs.
It reports Final extracted Acc@1 first, strict final second, normalized
Acc@1 AUC over rounds 0-32 third, plus strict AUC and efficiency. Its ten
paired contrasts comprise six within the frozen-reader panel (each new cell
versus frozen Q5, early versus late, each versus uniform075) and four
frozen-minus-moving comparisons (three positional settings and standard Q5).
All per-seed differences and descriptive seed-bootstrap intervals are shown.

At most one frozen positional setting is nominated: mean final accuracy
must improve without a mean strict-final loss versus frozen Q5. Eligible
settings rank by final, strict, AUC, then cell ID. AUC is not a veto. No
follow-up launches automatically. These are three reused development seeds
and historical controls, not fresh-question or bit-exact replay evidence.

Diagnostics retain segment/marker factor and count identities, actual reader
and prior policy, responsibility concentration, weight-length correlation,
top-trace changes, backward tokens, generations, optimizer steps and total
accelerator time. Frozen-base answer scoring can add cost; measure it.

## Execution and validation

Local verification: 257 tests pass, including factor-level reader routing,
RNG preservation, exact historical-configuration parity outside the declared
change, invalid-profile rejection, task mapping and analysis contracts.

Use `lm_study/submit_qwen3_17b_q5_prior_segments_frozen_ucl.sh`. It submits
an immutable held nine-task H100 array and a dependent validator, uncapped.
The wrappers delegate from the explicit checkout, never the SGE spool path.
Publish tracking and the prepared analyzer before releasing the payload.
The validator checks nine terminal logs, receipts and adapter hashes, complete
evaluation/diagnostic records, budgets, factor identities and frozen-reader
identity in every diagnostic round. Existing moving artifacts remain untouched.

The analysis command uses `analysis/analyze_qwen3_q5_prior_segments.py` with
`--config` pointing to the frozen-reader YAML and the usual result/control
paths, markers and execution identity. Additionally require `--moving-dir`
and `--moving-marker` for the recorded `c7e32a91` comparison. It refuses to
overwrite an existing output directory. Training outcomes are not yet available.
