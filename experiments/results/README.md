# Frozen numerical evidence

These 41 CSV exports contain the numerical data used in the thesis. Values and
identifiers are unchanged; `manifest.json` binds every file by SHA-256 and row
count. No question text, generated response text or model weights are included.

The top-level files cover the initial objective comparisons, prompt/EOS/joint
fitting studies, proposals, the full common grid, selected and equal-count
comparisons, sample-budget trajectories and transfer. `*_seed.csv` files retain
training-seed measurements; `*_summary.csv` files contain the reported means.
Column suffixes `_pct`, `_percent` and `_pp` specify percentages and percentage
points. In the selected-schedule and answer-guided-importance summaries the
accuracy columns are percentages; common-grid, equal-schedule and transfer
accuracy columns are fractions. The plotting code applies these conversions
explicitly.

The first historical objective table combines saved aggregate curves with
dated numerical records for latent-only arms whose original question-level
artifacts were unavailable during the thesis audit. It is not presented as a
new reconstruction of every individual response. Later confirmations and
transfer use their separately recorded seed/question designs. Do not pool
these cohorts or infer one common confidence-interval construction.

`diagnostics/` contains the two retrospective checks added in the final review:

- Fixed-weight delta at S8/M8/J4, seeds 31/47/73: consecutive pre-update objective
  values reveal the first three optimizer steps of each round. The fourth is
  unobserved. Counts describe this fixed-weight diagnostic, not direct changes
  in represented log probability or the refreshed delta objective.
- Frozen transfer: 56 receipt-verified method/dataset/seed files representing
  42,000 responses were classified with the unchanged original parsers. The
  exports retain all category/EOS/token-limit counts, all seed metrics and paired
  base-versus-trained gaps. Here the compact verifier rebuilds those metrics
  from frozen counts; it does not reclassify absent raw response text.

Run `python experiments/results.py --verify` from the repository root. This
checks all file bindings, recomputes entropy and ESS from actual saved weights,
checks fixed-objective differences and continuity, and recovers the published
transfer means from the 56 seed-level category tables. The optional plots show
these existing measurements without a new model run, model selection or
confidence interval. Original analysis source entry points are listed in
`../studies.json` and are available through `../reproduce.py --prepare`.
