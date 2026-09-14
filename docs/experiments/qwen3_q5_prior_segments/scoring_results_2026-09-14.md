# Q5: does the position of a token affect trace selection?

## Main finding

Weakening the first or second half of a rationale changes Q5's preferred trace
much more often than weakening the whole rationale equally. The two positional
rules disagree on 74 of 384 saved question buffers (19.27%). However, all four
rules still place most of the weight on one trace. This is evidence that the
position of attenuation matters, not evidence of improved answer accuracy.

All three tasks in job **7412300** completed and passed the prepared validator.
The audit covers **2,987 candidate rows and 384 buffers**, using seeds 1201,
1213 and 1217. No model was trained and no new answer was generated.

## Four scoring rules

Each seed's existing final Q5 adapter was held fixed. Every saved candidate
was scored once under its original canonical question prompt. Those token
scores were then reused for every weighting rule, keeping the known-answer
and EOS contribution unchanged. The following table splits reasoning tokens
in half and holds the `####` marker's score fixed.

| Rule: beginning / ending weight | Changed preferred trace vs original | Mean weight on preferred trace | Mean effective number of traces | Weight-length correlation |
|---|---:|---:|---:|---:|
| Original: 1 / 1 | 0 / 384 | 95.48% | 1.106 | -0.421 |
| Weaken beginning: 0.5 / 1 | 38 / 384 (9.90%) | 92.71% | 1.188 | -0.399 |
| Weaken ending: 1 / 0.5 | 43 / 384 (11.20%) | 94.46% | 1.137 | -0.420 |
| Weaken both: 0.75 / 0.75 | 7 / 384 (1.82%) | 94.09% | 1.144 | -0.418 |

The effective number of traces is `1 / sum(weight^2)`: a value near one means
one trace dominates. Correlations are within-buffer Spearman correlations,
averaged over the 379 buffers where both ranks vary. Negative values mean
longer traces tend to receive less weight. Buffers are weighted equally;
these 384 buffers are not 384 independent training replications.

Weakening the beginning spreads the weights most in every seed. It also
slightly weakens the preference for shorter traces. The mean length of the
preferred rationale rises from 123.60 to 125.27 tokens, so this is not a large
shift towards longer solutions. Weakening the ending changes a similar number
of winners but provides less consistent relief from weight concentration.

## All paired ranking contrasts

The historical mask includes `####` in the trace prior. The marker-fixed mask
excludes it from the halfway split and adds its score unchanged. These are two
analyses of the same token scores, not additional model runs.

| Comparison | Marker fixed: changed winners | Historical mask: changed winners | Marker fixed: mean weight mass redistributed |
|---|---:|---:|---:|
| Weaken beginning vs original | 38 / 384 | 37 / 384 | 11.33% |
| Weaken ending vs original | 43 / 384 | 41 / 384 | 9.78% |
| Weaken both vs original | 7 / 384 | 4 / 384 | 2.74% |
| Weaken beginning vs weaken ending | 74 / 384 | 73 / 384 | 19.45% |
| Weaken beginning vs weaken both | 42 / 384 | 38 / 384 | 11.36% |
| Weaken ending vs weaken both | 38 / 384 | 39 / 384 | 8.89% |

Redistributed weight mass is half the sum of absolute weight differences,
averaged over buffers. It captures changes beyond just the top-ranked trace.

Changing only the marker convention changes the winner in 0, 3, 8 and 3 buffers
for original, beginning, ending and uniform rules respectively. The broad
positional result survives either convention; individual selections can differ.

| Seed | Beginning vs original | Ending vs original | Both vs original | Beginning vs ending |
|---|---:|---:|---:|---:|
| 1201 | 17 / 128 | 18 / 128 | 4 / 128 | 31 / 128 |
| 1213 | 13 / 128 | 14 / 128 | 1 / 128 | 26 / 128 |
| 1217 | 8 / 128 | 11 / 128 | 2 / 128 | 17 / 128 |

## What this suggests

1. **Where we weaken the score matters.** Either positional rule changes
   substantially more selections than the uniform control on the same support.
2. **We have not solved concentration.** Even the most diffuse rule leaves
   92.71% of the weight on the leading trace, on average. A larger effective
   support would not by itself establish better learning.
3. **The split is positional, not semantic.** The preceding 24-trace review
   found a distinct initial planning/interpretation block in only seven traces,
   while seventeen already calculated on the first line. We should call this
   early-versus-late weighting, not separation of understanding and solving.

The narrow training comparison remains worth discussing: beginning, ending and
uniform attenuation, with all other training choices unchanged. For a reasoning-
position experiment, the recommended convention is to hold `####` fixed. That
would require an explicit protocol/mask update and tests before submission;
the current nine-task draft uses the historical mask. Neither convention has
been selected by this ranking audit, and no training job was released.

## Verification and limits

- Three original adapter hashes and archived prompt/diagnostic hashes passed
  preflight. The returned receipts bind the exact execution, manifest, model
  revision, seed and scheduler task.
- All three token tapes match their receipt sizes and SHA-256 hashes, decompress
  fully, contain finite scores and cover the expected candidates. The prepared
  analyzer reconstructs every segment sum, joint score and four-way softmax,
  and verifies source text and canonical prompt hashes.
- Saved text was retokenized once; it was not claimed to be historical token
  replay. Seventeen candidate rows (0.57%) differ in objective-token count:
  fourteen by -1 token, and one each by -2, -3 and +1. Each occurs in a distinct
  buffer. Excluding those buffers after seeing this diagnostic leaves 367:
  beginning, ending and uniform change 38, 40 and 7 winners; beginning and ending
  disagree on 71 (19.35%). This sensitivity check does not replace the primary
  all-buffer result. Equal token counts alone do not prove identical token IDs.
- The models are final checkpoints, not the intermediate checkpoints that
  originally generated each buffer. Consequently, the earlier saved-score
  uniform comparison (5/384 changes) and this fixed-final historical-mask
  comparison (4/384) are different audits, not a reproduction discrepancy.
- The recorded model-loading/scoring phases lasted 59.57, 62.80 and 68.79 seconds,
  totalling 0.0531 GPU-hours. These timings exclude imports, reconstruction and
  scheduler/node startup. Scheduler starts were 15:37:36 BST and copied terminal
  logs were last modified at about 16:11:53-55 BST, roughly 34 minutes later.
  File modification times are not authoritative scheduler end times. `qacct`
  could not read its accounting file, so full allocated GPU-hours are unavailable.
- No optimisation step, new generation, validation evaluation or official-test
  access occurred. Final Acc@1, strict final accuracy and AUC are therefore not
  outcomes of this audit.

## Evidence and reproduction

- Machine-readable [validated scoring summary](scoring_summary_2026-09-14.json).
- Worker execution: `802c174e0e1a20842b757ce8b25aea70d8730b6d`.
- Prepared analyzer: `41b964e747d08c6588e9158615b642b02ebc2bc7`.
- Frozen manifest: [scoring_manifest.json](scoring_manifest.json), SHA-256
  `292f80620f6e90e34b5051778e2a1b28f70c7efa30d06572512356f88568caa9`.
- Compact local archive:
  `ResultsArchive/beaker-amanojna/20260914T154218Z__q5_segment_scoring_7412300`.
- The archive contains all token tapes, receipts and logs, `analysis/summary.json`,
  the separately labelled post-result sensitivity and `SHA256SUMS`. No checkpoints
  were copied and no remote artifacts were moved or deleted.

From the curated repository with `PYTHONPATH=src:lm_study:analysis`:

```bash
python analysis/analyze_q5_segment_scoring.py \
  --source "$HISTORICAL_REPLAY_68078ECC" \
  --results "$SCORING_ARCHIVE" \
  --manifest docs/experiments/qwen3_q5_prior_segments/scoring_manifest.json \
  --logs "$SCORING_ARCHIVE/logs" --output "$NEW_ANALYSIS_DIRECTORY" \
  --execution-commit 802c174e0e1a20842b757ce8b25aea70d8730b6d --job 7412300
python "$SCORING_ARCHIVE/provenance/audit_extra_checks.py"
```
