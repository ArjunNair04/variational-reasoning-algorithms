# Verification for the thesis v3 code release

The release was checked on 7 September 2026. No new model training, GPU job,
cluster command or official GSM8K test access was performed.

## Original source and configuration

- Independently compared 617 shipped source/configuration blobs with their Git
  objects at the recorded original revisions. Every byte matches.
- Checked all 18 archive manifests (17 execution versions and one later analysis
  version), every member hash, safe file membership and Python syntax.
- Expanded all 17 training-study groups through their original YAML drivers in
  offline dry-run mode. Each passed, including all 163 common-grid cells at the
  first configured seed. Transfer preparation is checked separately because
  evaluation requires receipt-bound trained adapters.
- Rechecked the selected delta example after adding explicit replay metadata.
  No scientific field changes during runtime preparation: only output placement
  and selection of an existing cell are permitted. A new run has its own
  provenance record; it is not represented as an original completed experiment.

The common-grid continuation archive selects a source revision, not a historical
scheduler task range. A GPU replay would still need the archived dependencies,
pinned model/data assets and sufficient accelerator resources. Dry-run success
establishes configuration and interface consistency, not a new numerical
reproduction of the reported training outcomes.

## CPU behavior checks

The full 40-test suite passed under Python 3.12.14, NumPy 1.26.4 and CPU
PyTorch 2.2.2 on Intel macOS. The local wheel availability differs from the
archived Linux/CUDA environment; its dependency specifications are preserved
separately in `environment/`.

The tests execute original source operations alongside the clean references:

- delta, prior-importance, corrected-proposal and uniform weights, including
  detached responsibilities;
- exact original-token prefix admission, the internal delta proposal prompt,
  EOS appended once, and the independent extracted/terminal-format parsers;
- FIFO behavior on unlabelled buffers, sequence-sum M-step values and gradients,
  usable-question averaging, empty-buffer behavior and negative-infinity exclusions;
- ESS temperature control, hard exclusions and entropy/effective-count relations;
- GRPO's clipped token objective with batch-level microbatch accumulation,
  RLOO's KL shaping before leave-one-out subtraction, and TRICE's original
  leave-one-out control-variate scales;
- fixed disjoint training/validation index construction on synthetic row indices,
  archive/config bindings and restricted runtime cell selection.

The minimal `.[test]` installation runs the NumPy, parser and standard-library
checks. It may skip the tests marked with `pytest.importorskip('torch')` or
`pytest.importorskip('yaml')`; installing CPU PyTorch and PyYAML enables those
comparisons. The full 40-test result above includes them, with no skips.

## Numeric evidence and compute

The independent compact result verifier passes for 41 CSV exports. It checks
3,430 saved weight allocations, all 288 observed fixed-weight objective
transitions, and all 56 transfer method/dataset/seed cells accounting for 42,000
responses. The transfer means are recovered from category counts and compared
with the published summaries. The checker does not re-read absent raw responses
or regenerate confidence intervals from aggregate means.

The separate compute verifier passes for 970 distinct completed timing records
and 808.589722222 H100 GPU-hours. This remains a lower bound on the wider project,
not just the final comparison panel. The source/receipt audit is documented
inside `resources/compute/`.

All three portable plots were generated from the shipped CSVs and visually
checked for readable units, axes, legends and labels. They retain separate
answer and terminal-format accuracy and identify the fixed-Q observation scope.
The archive and working-tree scan found no credentials, personal filesystem
paths, private correspondence, raw response files or model checkpoints.
