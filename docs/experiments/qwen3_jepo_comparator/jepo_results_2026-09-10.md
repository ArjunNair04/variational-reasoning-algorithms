# JEPO: Seven-Seed Results and Thesis Handoff

## Main Finding

JEPO reached **77.46% Final Acc@1, 75.07% strict final accuracy and 76.83%
accuracy AUC**. Compared with the same-seed frozen model, Final Acc@1 increased
by **1.29 percentage points**, with a paired 95% interval of **[-1.47, +4.05]**.
The evidence therefore does not establish an improvement in answer accuracy.
Strict accuracy increased by **17.57 points [5.45, 29.69]**, improving on all
seven seeds. The clearest result is better completion formatting and
termination, rather than a demonstrated gain in mathematical accuracy.

This is a common-protocol implementation of JEPO, an external comparator.
It is not a new algorithm from this project or a reproduction of the original
paper's full-model, larger-scale experiment.

## Comparison Under the AMN Replay Protocol

All values are seven-seed means. Historical controls come from run `68078ecc`;
JEPO comes from the fresh, fully completed run `4124d5c8`. The local verification
checked matching optimization-question IDs, demonstration IDs and the same
400 validation questions for every seed across these methods.

| Method | Final Acc@1 | Strict final | Acc@1 AUC | Training generations | Optimizer steps | H100 hours |
|---|---:|---:|---:|---:|---:|---:|
| Frozen base | 76.18% | 57.50% | 76.18% | 0 | 0 | 0.112 |
| Q5, moving reader | 80.32% | 70.89% | 79.38% | 2,048 | 32 | 0.560 |
| PIS | 80.00% | 79.36% | 78.37% | 2,048 | 128 | 0.630 |
| TRICE-CV | 83.64% | 83.00% | 82.83% | 2,176 | 32 | 0.553 |
| RLOO | 80.64% | 80.11% | 80.61% | 4,096 | 128 | 0.974 |
| **JEPO-MS4** | **77.46%** | **75.07%** | **76.83%** | **2,048** | **32** | **0.684** |

JEPO's mean Final Acc@1 is 2.86 points below Q5, 2.54 below PIS, 6.18 below
TRICE-CV and 3.18 below RLOO. These are descriptive gaps, not newly declared
significance tests. The prepared analysis specified paired inference against
the frozen base and descriptive context for the trained comparators.

The table uses the AMN replay controls, not the earlier thesis headline panel
whose Q5/PIS scores were 80.57%/76.89%. Reusing the same seeds does not create
additional independent replications of those historical results.

## What Was Run

- Qwen3-1.7B-Base, rank-16 LoRA on attention and MLP projections, three-shot
  question-only generation and evaluation prompts.
- Seven seeds: 1201, 1213, 1217, 1223, 1229, 1231 and 1237.
- 128 optimization questions and a disjoint fixed 400-question validation set
  drawn from GSM8K training data. The official test split was not loaded.
- 32 rounds, 16 questions per round, four sampled completions per question,
  and one optimizer step per round: 2,048 training generations per seed.
- JEPO's four-sample evidence objective, leave-one-out trace credit, format
  penalty 10, supervised-answer coefficient 0.01, frozen-base KL coefficient
  0.001 and learning rate 1e-5. The answer target includes EOS.
- Strict-valid but token-boundary-unsegmentable completions retain format
  credit but do not contribute to the evidence terms; the denominator remains
  the fixed four samples. This repair affected only one of 14,336 training
  generations in the completed run, as measured by the coverage diagnostics.

Final Acc@1 uses the existing extracted-answer evaluator. Strict final requires
the strict terminal-answer convention; it is not identical to requiring EOS.
Natural EOS is reported separately. AUC is the trapezoidal average of accuracy
over rounds 0-32, divided by 32, using the paired frozen control at round zero.
The final checkpoint was fixed at round 32, not selected from the curves.

The paired intervals are the prespecified Student-t intervals over seven
per-seed differences, with six degrees of freedom. They describe variability
across training seeds on this fixed, repeatedly used validation set, not a
fresh test-question population.

## Learning Curve and Diagnostics

| Completed rounds | Acc@1 | Strict accuracy |
|---:|---:|---:|
| 0 | 76.18% | 57.50% |
| 1 | 75.82% | 57.64% |
| 2 | 75.46% | 57.46% |
| 4 | 75.64% | 58.89% |
| 8 | 76.07% | 61.64% |
| 16 | 77.04% | 69.25% |
| 24 | 77.75% | 74.21% |
| 32 | 77.46% | 75.07% |

The model gradually learns to return a cleaner final answer. Strict accuracy
continues improving while extracted accuracy changes little. The 0.29-point
mean decline from round 24 to 32 is small; this run does not show the severe
late collapse seen in some earlier L2R studies.

During training, valid-completion coverage rose from 53.04% in rounds 1-8 to
79.85% in rounds 25-32. Sampled correctness rose from 33.90% to 49.64%, while
the effective number of answer-weighted samples rose from 1.73 to 2.31 out of
four. Thus more sampled traces contributed usable evidence later in training,
but this did not translate into a clear held-out answer-accuracy gain.

At the final checkpoint, natural EOS occurred in 98.46% of completions,
strict format failures in 6.18%, and nonterminal markers in 2.96%. There were
no multiple-marker or direct-answer-only completions. Mean generation length
to EOS was 101.70 tokens. Pass@8 was 96.43% on the separate 100-question
sampled-evaluation subset; it is not the greedy Final Acc@1 metric.

Normalized trace advantages were clipped on 8.77% of their logged eligible
samples. The logged signed sampled-policy log-ratio averaged -0.00278;
this noisy diagnostic is not an exact nonnegative KL divergence. It should
not be used to claim that policy drift was eliminated. The training loss uses
the separately implemented nonnegative k3 KL penalty.

## Compute

Per seed, JEPO used 262,778 generated training tokens and 412,078 backward
tokens. Backward tokens count token positions processed by gradient-bearing
passes; repeated processing is counted again. They are not a count of unique
text or a FLOP measurement. JEPO took 0.684 H100-hours per seed, or 4.790
hours summed over seven seeds. These timings include the recorded evaluation
work; training-generation counts exclude evaluation.

At the same 2,048-generation budget as Q5 and PIS, JEPO neither improved their
mean Final Acc@1 nor reduced measured accelerator time. RLOO used twice as
many training generations, so its raw accuracy lead is not an equal-budget
causal comparison. Older Q5/PIS token counters are absent in these archived
cell records and remain blank in the machine-readable table, not filled with
zeros.

## Verification and Provenance

Payload `7340568` completed all seven tasks; validator `7340569` recorded
success on 7 September. Execution revision was
`b3ae5bf53bd22113b7a5ab0563e85c524e826aca`; configuration SHA-256 was
`433aa325d1358f43e7ddc51e0c8b2f65f70c37513f490947d40aec78f873b95c`.

The prepared analyzer initially rejected the different seed-specific ordering
of the same 400 questions. Canonicalising question IDs fixed that check without
changing any metric, training artifact or comparison. Reordered, changed and
duplicate-ID regression tests pass; the complete repository suite passes
189 tests. The corrected analyzer then passed remotely against all original
receipt-bound artifacts, including the seven adapters.

The local compact mirror has all 77 top-level JEPO files, seven task logs,
the validator log and marker, 49 checkpoint-evaluation records and 224
training-diagnostic records. All receipt-bound non-adapter files match their
SHA-256 hashes; JSON and gzip files parse, and task logs contain terminal
markers with no failure signatures. Same-seed comparator files and analysis
are included. Adapter weights and model checkpoints are deliberately absent.
The `checkpoint_eval` files are small metric records and are retained.
Seven archived `.running` lock files are retained as raw provenance; they are
not evidence of an active job.

Earlier failed attempts `7452ba96` and `6c94a797` are not pooled with these
results. The latter's six-seed preliminary numbers are superseded. No further
training, deletion of cluster files or thesis-text edits are part of this handoff.

## Files for Thesis Writing

- `analysis/method_summary.csv`, `seed_metrics.csv`, `paired_contrasts.csv`:
  the prespecified JEPO results and paired base contrasts.
- `analysis/comparator_context.csv`, `checkpoint_metrics.csv`,
  `diagnostic_phase_summary.csv`, `final_format_diagnostics.json`:
  descriptive companion tables used above.
- `results/`: evaluations, completion dumps, pass@8 results, trajectories,
  prompt contracts, diagnostics, sweep rows and original completion receipts.
- `controls/`: same-seed AMN control artifacts and original analysis. Historical
  control bootstrap outputs are preserved but are not the source of the JEPO
  Student-t intervals.
- `provenance/`: frozen YAML, analyzer and regression test, validator source,
  receipt/hash verification, and the analysis-only repair patch.
- `SHA256SUMS`: checksums for the handoff package.

Use this as additional comparator evidence. Integrating it into approved thesis
prose remains a separate paragraph-level review; the printed-copy PDF is unchanged.
