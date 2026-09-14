# Q5: what comes before calculation?

## Result

An initial interpretation or planning phase is visible in some saved traces,
but it is not a reliable first-half or first-line division. The proposed
halfway experiment should therefore be described as **early versus late token
weighting**, not as a separation of understanding from answering.

The semantic review and the uniform-weight ranking check are complete. Actual
early-versus-late rankings remain unmeasured. The nine training tasks are
prepared but not submitted, and the submitter now rejects an incomplete
pre-training audit.

### Token-boundary clarification

The archived Q5 prior totals include the retained `####` marker; the answer
factor covers the space, digits and EOS that follow it. The uniform comparison
below reweights that historical prior. A separate scoring-only audit will report
both this convention and pure-reasoning halves with the marker held fixed, using
the same archived candidates under each seed's final adapter. That audit is new
fixed-checkpoint scoring, not reconstruction of earlier model states.

## Reading the traces

The review covers 24 saved final-evaluation traces from moving Q5, eight from
each of seeds 1201, 1213 and 1217. Within each seed's 100 saved completions,
traces were divided into four length groups and two selected from each group
by a fixed hash, without consulting correctness. Every nonempty pre-answer
line was labelled for interpretation, planning, calculation, summary,
independent verification, or other content. Lines can have several labels.

This is one Codex-authored qualitative review, not independently adjudicated
human annotation. The counts describe these 24 texts, not the entire dataset
or the training buffers.

| Observation | Reviewed traces |
|---|---:|
| At least one separate interpretation/planning line before calculation | 7/24 |
| That initial block contains interpretation, rather than planning alone | 4/24 |
| Calculation already appears on the first line | 17/24 |
| First line mixes calculation with interpretation or planning | 16/24 |
| Interpretation reappears on a later line after calculation has begun | 8/24 |
| Planning reappears on a later line after calculation has begun | 14/24 |
| A setup-to-solution transition exists inside a single paragraph | 3/24 |
| Independent verification of the solution is visible | 0/24 |

Three examples explain the distinction:

- **Shopping total, seed 1201, question 5833:** the model first says to
  multiply each item's quantity by its price and add the results. It then
  calculates the individual costs. There is a separate plan here, although
  it does not occupy half the rationale.
- **Fuel-to-distance proportion, question 5433:** it explains the ratio,
  sets up a proportion, and then solves it. All of this is in one paragraph.
  A line-break rule would miss the transition entirely.
- **Weekly newspaper deliveries, seed 1201, question 6059:** it calculates
  weekday deliveries before interpreting the different Sunday rules.
  Understanding the conditions is interleaved with using them.

Among the seven traces with a separate initial block, that block spans
6.8%-68.3% of the rationale's whitespace-separated words. This is a text-length
description, not a model-token boundary. In addition, the earlier syntactic
audit found 32 single-line rationales among 300 saved completions. Neither
newlines nor a halfway cut consistently identifies a semantic phase.

## Reweighting the saved buffers

The second audit uses actual training diagnostics, separately from the
24-text review: 32 rounds, four questions per round and three seeds, giving
384 question-round supports and 2,987 recorded trace rows. A trace row is a
scored candidate occurrence, not an additional generated sample. All 384
supports were present. The recorded scores reproduce the original weights
within the fixed numerical tolerance; the recorded answer and prior factors
sum exactly to the joint scores in this archive.
All six input files (three evaluation dumps and three diagnostic streams)
match their original completion-receipt sizes and SHA-256 values. This check
does not redownload or revalidate the full checkpoint archives.

Let A be the saved answer log probability, and P the saved rationale-prior log
probability. The original score is A + P; uniform attenuation uses
A + 0.75 P. Both are normalized over the same question's saved candidates.
This changes no prompts, candidates, reader, M-step, buffer policy or EOS
handling. It is arithmetic on archived scores, not a new training run.

| Measure, averaged over question-rounds | Original (1,1) | Uniform (0.75,0.75) |
|---|---:|---:|
| Effective number of weighted traces | 1.098 | 1.134 |
| Largest trace weight | 95.90% | 94.55% |
| Score-length rank correlation | -0.441 | -0.438 |
| Highest-weight trace changes versus original | -- | 5/384 (1.30%) |

Uniform attenuation moves about 2.53% of probability mass on average between
traces, measured by total variation. The score-length correlations exclude
five supports where the rank correlation is undefined. There is one tied
maximum under each rule; none of the five changed winners involves a tie.

| Seed | Supports | Changed highest-weight trace | Effective traces: original -> uniform |
|---|---:|---:|---:|
| 1201 | 128 | 2 | 1.112 -> 1.151 |
| 1213 | 128 | 0 | 1.096 -> 1.139 |
| 1217 | 128 | 3 | 1.086 -> 1.112 |

Thus uniform 0.75 is a useful control: it makes the weights slightly less
concentrated but almost always retains the same preferred trace. These are
within-buffer ranking findings; they do not establish an accuracy benefit or
predict the outcome of repeated training.

## Why the positional comparison is still pending

The historical diagnostics save the whole rationale score, but not its
per-token log probabilities. They include sampled text, yet lack the exact
token-resolved scoring record and historical model state needed to recover
each recorded early/late score. Knowing P alone does not identify H and T
where P = H + T.

There is a limited check available without inference. Because log
probabilities are nonpositive, either half-attenuated score lies between
A + P and A + 0.5 P. Those bounds guarantee the original winner survives
either positional change in 55/384 supports. The remaining 329 are
**undetermined**, not 329 predicted changes. A regression test constructs two
different early/late allocations with the same saved total that give opposite
rankings, demonstrating why filling in missing scores would be invalid.

AMN was reachable earlier in this session, but the subsequent approved helper
call to inspect checkpoint availability returned exit 255. No checkpoint
scoring, new generations, optimizer steps, remote changes or submissions were
performed. Current remote checkpoint availability remains unverified.

## Next step before training

1. Verify whether an archived model state and the exact stored candidate tokens
   can be joined. If so, score those unchanged candidates under that state.
2. Otherwise, explicitly declare a new fixed-checkpoint scoring audit. Use
   existing saved candidate texts, reconstruct the canonical prompt/answer
   contract once, and record the newly tokenized support. This is not a replay
   of the historical per-round scores. There is no optimizer and no comparison
   of accuracy outcomes.
3. On that single common support, compute A, H and T once and compare (1,1),
   (0.5,1), (1,0.5), and (0.75,0.75). Save token IDs, score masks, checkpoint
   identity, prompt hashes, factor sums, all four rankings and weights.
4. Report changed winners, probability-mass movement, effective support and
   length relationships for early versus late and each versus uniform. Check
   segment lengths and reconstructed factor identities before releasing any
   training. No model-quality gate can be inferred from ranking changes alone.

The existing nine-task design remains intact and blocked on this audit.
Prompts, M-step, reader, buffer and EOS rules remain unchanged.

## Reproducibility files

- [Annotation rules and all 24 line labels](semantic_annotations.json)
- [Reviewed traces, questions, notes and source hashes](pretraining_audit/semantic_review.json)
- [All 384 saved-buffer comparisons and source hashes](pretraining_audit/saved_buffer_rankings.json)
- [Audit manifest and training block](pretraining_audit/audit_manifest.json)
- [Reusable audit program](../../../analysis/audit_q5_prior_segments_pretraining.py)

The official GSM8K test was not opened by this audit. Its evaluation inputs
explicitly record train-only provenance; the training inputs are the existing
Q5 optimization diagnostics. Historical result files were read without edits.

Verification: all 234 curated-repository tests pass, including 12 focused
audit tests. Report links and manifest hashes pass. Canonical registry views
are current, nine registry/importer tests pass, and the existing study's local
MLflow projection refreshed. No new training run was registered as completed.
