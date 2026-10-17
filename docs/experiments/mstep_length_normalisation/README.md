# Q5 and PIS: M-step Length Normalisation

## Status

The existing-trace audit is complete for three paired historical seeds. The
M-step arithmetic has been checked against a direct tiny-Qwen calculation.
The new six-task training screen is configured, but has not been submitted.
The frozen result analyzer is prepared and reconstructs all 192 historical
diagnostic rounds. Cluster access, deployment wiring and full control runtime
provenance checks must be completed before submission. No historical run or
training result has been changed.

## Existing Traces

Source: completed run `68078ecc`, seeds 1201, 1213 and 1217. The audit reads
the NAS files in place and checks each diagnostics file against its completion
receipt. All 192 round records parse, including gzip end checks. Every retained
trace joins to its original generated response. No model inference, dataset
loading or optimizer steps are performed by the audit.

| Diagnostic | Q5 | PIS |
|---|---:|---:|
| Question-round supports | 384 | 768 |
| Retained trace records | 2,987 | 4,881 |
| Weight on originally wrong extracted answers | 8.30% | 14.49% |
| Weight on original format failures | 20.47% | 9.14% |
| Largest trace weight, mean | 95.90% | 39.90% |
| Effective support size, mean | 1.10 | 3.82 |
| Weight-length rank correlation, mean | -0.439 | 0.023 |
| Posterior-weighted target length | 127.79 | 116.43 |

Each number first averages question-round supports within a seed, then averages
the three seeds. These are training-support diagnostics, not validation accuracy.
Q5's saved weights follow its end-of-round refresh, so they are not the exact
weights used in the preceding M-step. PIS saves the outer-round weights reused
through its four M-steps. The two methods also visit different numbers of
question-round supports under their existing budgets.

Q5 already strongly prefers short traces and usually places nearly all weight
on one candidate. Token averaging will not fix that concentration because the
E-step is unchanged. It instead tests whether the M-step benefits from giving
each weighted sequence a token-averaged loss. PIS distributes its weights more
broadly, making relative changes in the contributions of unequal-length traces
potentially more important.

The audit also confirms that replacing a generated answer with the known answer
can retain an incorrect argument. For example, a Q5 trace with unit saved weight
states `$7000 - $40,000 = $3000`, yet ends at the known answer. This is a concrete
failure example, not an estimate of the frequency of semantic reasoning errors.

### Arithmetic Parser Review

Automated equation flags must not be presented as verified errors. Inspection
found that the existing parser sometimes truncates expressions containing units
or words: `10% of $200 = $20` becomes `$200 = $20`, while
`25 flavors * 2 = 50 flavors` becomes `2 = 50`. Exact rational comparisons also
flag benign rounding such as `25/60 = 0.4167`. The audit reports these as
`arithmetic_parser_flag`, separately from answer correctness and format.
No parser-derived filtering or penalty is introduced into training.

## Intervention

Let `w_s` be the existing detached posterior weight for a trace and `T_s` its
number of predicted target tokens. The target includes the rationale, answer
marker, known answer and EOS; prompt and padding positions are excluded.

\[
J_{\rm old}(\theta)=\frac{1}{|\mathcal Q|}\sum_{q\in\mathcal Q}
\sum_s w_s\log p_\theta(h_s,a^+\mid q),
\]
\[
J_{\rm mean}(\theta)=\frac{1}{|\mathcal Q|}\sum_{q\in\mathcal Q}
\sum_s w_s\frac{\log p_\theta(h_s,a^+\mid q)}{T_s}.
\]

This is a per-trace mean, not a batch-wide division by total tokens. The
posterior vector is not divided by length or renormalised. In particular, this
does not repeat the earlier Q5 token-mean *E-step* experiment.

| New cell | Proposal | Posterior | Buffer | Inner M-steps |
|---|---|---|---|---:|
| Q5-MSTEP-TOKENMEAN | Answer-derived | Existing joint Q5 weights | Persistent unique FIFO, 16 | 1 |
| PIS-MSTEP-TOKENMEAN | Question-only | Existing prior-IS answer weights | Fresh multiset, 8 | 4 |

Each cell uses seeds 1201, 1213 and 1217, with the existing 32-round protocol,
current reader, learning rate, attention-plus-MLP LoRA, EOS target and validation
partition. Only `latent_mstep_objective: joint_token_mean` changes scientifically.
The historical profile validators remain unchanged; the new profile delegates
to them after checking the permitted M-step change.

The plan reuses the six corresponding historical controls after provenance and
runtime checks. It does not rerun the frozen base or add temperature, reader,
KL or learning-rate variants. The learned policies will diverge after training,
so later posterior values can differ even though the weighting rule is identical.

## Interpretation and Analysis

Report Final Acc@1 first, strict final accuracy second, and normalised Acc@1 AUC
third. Use round 32 rather than selecting the best checkpoint. Show every seed
and each paired treatment-minus-control difference with descriptive uncertainty.

Record generated and backward tokens, gradient norm, applied update norm,
response length, formatting and posterior concentration. Length normalisation
changes both relative trace contributions and overall loss scale; Adam does not
necessarily reduce the parameter update in proportion to that scale. A positive
screen would motivate a separately approved scale-matched follow-up, not an
immediate claim that length bias was the cause. This is a modified M-step
objective, not an exact M-step for the original unnormalised likelihood bound.

## Reproduction

`analysis/analyze_qwen3_mstep_length_normalisation.py` fixes the paired contrasts
before training. It requires immutable execution identity, successful task logs,
generic artifact validation, checksum-valid completion receipts, the historical
validator marker, consistent runtime/question schedules, intact posterior
formulas, EOS support and complete trajectories before writing a success marker.
Full historical controls now reside on the NAS; deployment must preserve this
analysis path rather than assuming deleted AMN control files still exist.

The `audit_sources.json` file lists the six explicit NAS input paths. Run:

```sh
PYTHONPATH=src:lm_study:analysis .venv/bin/python analysis/audit_mstep_length_normalisation.py \
  --sources docs/experiments/mstep_length_normalisation/audit_sources.json \
  --output-dir /tmp/mstep-length-audit

PYTHONPATH=src:lm_study:analysis .venv/bin/python lm_study/generate_qwen3_17b_mstep_length_normalisation.py \
  --check lm_study/experiments_qwen3_17b_mstep_length_normalisation.yaml --runtime-check
```

`audit_2026-10-16_v2` contains the reviewed-schema audit. The first local audit
directory is preserved as preliminary output and excluded from Git. Raw trace
and support exports remain local; compact summaries, source hashes and the
manifest are retained for provenance. No output is itself permission to release
jobs, and no official-test performance is inferred from these files.
