# Repository operations

- Keep the NumPy package independent from the model-training stack.
- Treat `lm_study/experiments_qwen3_17b_selected_method_posterity.yaml` and
  `lm_study/experiments_qwen3_17b_pis_temperature_posterity.yaml` as generated,
  frozen protocols. Edit their generators, never the generated YAML directly.
- Do not inspect or score the official GSM8K test split in development runs.
- Submit SGE arrays without an artificial concurrency ceiling.
- Bind every submitted job to an immutable Git commit and YAML SHA-256.
- Prepare analyzers before outcomes and require validators before claiming
  completion.
- Beaker uses legacy Git. Avoid `git -C`, `git switch`, `git worktree` and
  `git branch --show-current` in cluster-facing commands.
- Load the shared Python runtime before invoking the AMN virtual environment.

## Change record

- **2026-09-13, Q5 exponent ingestion:** all 12 tasks, adapters and frozen paired
  analysis passed after the isolated dependency repair; compact outputs and
  hashes are mirrored locally. Moving tau0.5 is the sole three-seed nominee,
  with unresolved +0.58 pp final gain; tau0 spreads support but reduces accuracy.
  Full local suite: 204 passed. No training rerun or automatic confirmation.

- **2026-09-13, Q5 exponent validator dependency repair:** compute Spearman
  correlation as Pearson correlation of average ranks, avoiding optional SciPy
  in the AMN runtime. Training, frozen YAML, metrics and contrasts are unchanged.
  Regression coverage compares tied and untied ranks with SciPy while blocking
  SciPy imports inside the mechanism calculation. See registry run `e97c3a20`.

- **2026-09-12, Q5 released:** all 12 tasks of 7380685 are ordinary queued
  without a concurrency cap; 7380686 is dependency-held. The execution
  checkout stays at 5b4e32d; subsequent commits are tracking-only.

- **2026-09-12, Q5 held submission:** exact uncapped payload 7380685 and
  dependent validator 7380686 bind execution 5b4e32d and run e97c3a20.
  Nine historical control receipts/hashes and all cluster coordinates pass;
  the local full suite passes 201 tests. Release follows the tracking push.

- **2026-09-12, Q5 submission-path preflight repair:** remove a per-command
  `PYTHONPATH` override that discarded `src` during analyzer preflight. Keep
  the exported source path for every subprocess. Cluster preflight failed
  before qsub; no payload was created. Added a wrapper regression assertion;
  scientific configuration and model code are unchanged.

- **2026-09-12, Q5 prior-exponent reader screen:** added the isolated
  `q5_prior_exponent` profile and `responsibility_prior_exponent` YAML knob.
  Only the E-step rationale-prior factor changes; tau=1 preserves the legacy
  scoring path and the M-step is unchanged. Run `e97c3a20` contains tau=0/0.5
  crossed with moving/frozen answer readers on three paired seeds, reusing
  receipt-bound tau=1 controls from `68078ecc`. Verification includes actual
  mocked E-step factor/reader tests, identity tests, runtime-profile rejection,
  exact task mapping, frozen analysis and validator tests, and the full suite.
  See `docs/experiments/qwen3_q5_prior_exponent/README.md` and the registry.
  Final local gate: 200 tests, all 12 dry-run coordinates, runtime validation
  for all four cells and a 128-question-round historical diagnostic replay pass.

- **2026-09-10, JEPO result ingestion:** canonicalise validation question IDs
  before comparing support across seeds; the evaluator deliberately permutes
  the same 400-question pool. This analysis-only repair changes no training,
  evaluation, metric or contrast. Regression tests cover reordered, changed and
  duplicate IDs. Run `4124d5c8` is collected without model weights for the thesis
  handoff; provenance and completion are recorded in `docs/experiment_registry.json`.

- **2026-09-07, JEPO token-boundary robustness repair:** retain strict decoded
  format scoring when a rare completion cannot be segmented at an exact sampled-
  token answer boundary, but mask that sample from JEPO's lower-bound and answer
  terms while preserving the fixed four-sample denominator. Diagnostics now
  distinguish strict-valid from segmentable support and explicitly declare the
  natural-EOS reward contract. The incomplete six-of-seven run remains untouched;
  the repair requires a fresh seven-seed run ID. Verification covers row masking,
  diagnostic schema, estimator normalization, generated-YAML equality, task-
  specific runtime preflight, Python compilation, shell syntax and the full test
  suite.

- **2026-09-05, JEPO GSM8K runtime repair and isolated retry:** use the shared
  GSM8K answer-event parser when the task does not expose a parser method, and
  export the source package path in the dependent validator. The failed
  pre-training artifacts from run `7452ba96` remain untouched; the scientifically
  identical retry uses a new run ID and output root. Verification adds an
  executable GSM8K parser preflight and regression test alongside Python
  compilation, shell syntax, generated-YAML equality, all task coordinates and
  the full repository test suite.

- **2026-09-04, JEPO cluster import-path repair:** prepend the repository's
  `src` and `lm_study` directories to `PYTHONPATH` in the JEPO submitter and
  GPU runner. This is an infrastructure-only repair for AMN's uninstalled
  source layout; the JEPO estimator, generated configuration, seeds and task
  mapping are unchanged. Verification covers a scheduler regression assertion,
  Python compilation, shell syntax, generated-YAML equality, all seven dry-run
  coordinates and the full repository test suite.

- **2026-09-04, seven-seed JEPO comparator:** added the multi-sample Jensen
  evidence policy objective as one new common-protocol comparator cell on the
  established seven paired seeds. The implementation uses four
  question-conditioned samples, leave-one-out trace credit, a log-mean gold
  answer term, invalid-format masking with leave-one-out format credit and a
  frozen-policy KL penalty. No historical comparator is rerun. The generated
  protocol remains validation-only and binds each task to its configuration
  hash and execution commit. Verification covers numerical estimator tests,
  fail-closed runtime axes, generated-YAML equality, all seven task mappings,
  frozen analysis, artifact/receipt validation, Python compilation, shell
  syntax and the repository test suite.

- **2026-09-03, Q5 support-allocation follow-up:** added a generated,
  seven-seed comparison between 64 proposals with the exact support-32 M-step
  and 32 proposals with an exact dominant posterior term plus 15 residual
  draws. The latter is an unbiased finite-support estimator that prevents the
  dominant trace from consuming repeated Monte Carlo draws. Both cells retain
  the canonical Q5 prompt, posterior, buffer, question breadth, one-step update,
  EOS target and attention-plus-MLP rank-16 LoRA. The study reuses both
  validated support-32 controls and remains validation-only. Verification
  covers estimator determinism and unbiasedness, fail-closed profile rejection,
  generated YAML, all task coordinates, frozen paired analysis, artifact
  validation, Python compilation, shell syntax and the repository test suite.

- **2026-09-03, Q5 buffer-sampling analysis schema repair:** corrected the
  prepared analyzer to read per-step support and posterior-sampling records
  from the persisted `inner_m_step.steps` schema, require complete 32-round
  coverage, and report full-support rows alongside sampled rows. This changes
  no training artifact or scientific method. Verification covers a nested-
  schema regression test, the focused analyzer tests, Python compilation,
  formatting checks and rerunning analysis against all 14 validated receipts.

- **2026-09-02, Q5 large-support M-step sampling:** added a generated,
  seven-seed comparison that retains 32 unique answer-guided Q5 traces and
  either updates over the full posterior or 16 categorical posterior draws.
  The zero-sample path preserves the existing full-support objective by object
  identity; positive sampling is isolated to one fail-closed profile and does
  not alter proposal generation, responsibility refresh or question order.
  The study reuses the validated support-16 Q5-MORE control, remains
  validation-only and records support coverage, M-step uniqueness and compute.
  Verification covers deterministic sampling tests, identity-path regression,
  runtime-profile rejection, generated YAML, all task coordinates, frozen
  analysis, validator contracts, Python compilation and shell syntax.

- **2026-09-02, same-seed reproducibility audit:** added a fail-closed
  historical-versus-posterity analyzer that checks complete paired-seed
  coverage, reports Final Acc@1 before strict final and AUC, verifies identical
  validation support at every checkpoint, compares final generations, and
  locates the first observable Q5/PIS divergence across question selection,
  sampled traces, posterior weights and optimizer drift. This changes no
  experiment or model behavior. Verification covers focused tests, Python
  compilation, formatting checks and execution against the mirrored source
  and replay artifacts.

- **2026-09-02, AMN validator Python-runtime repair:** initialise the shared
  Python runtime and invoke the experiment virtual environment's interpreter
  explicitly in all three posterity validator wrappers. This repairs
  post-training validation only; payload code, YAMLs, task mappings and result
  artifacts are unchanged. Verification covers focused wrapper assertions,
  shell syntax, Python tests and rerunning the validators against the completed
  payload artifacts.

- **2026-09-01, posterior-update posterity follow-ups:** added a generated
  six-cell, seven-seed study for Q5 support depth, Q5 token-mean
  responsibilities, PIS update reuse, rationale-only adaptive PIS KL, and the
  exact signed sampled-support update at one and four passes. New validation
  profiles isolate the two previously unsupported interventions without
  loosening historical profiles. The 42-task array is dependency-gated on the
  selected-method posterity validator and has no concurrency cap. Verification
  covers generated-YAML equality, direct runtime-profile validation, every
  cell/seed dry run, frozen paired analysis, fail-closed artifact validation,
  shell syntax and the repository test suite.

- **2026-09-01, thesis reproduction evidence gate:** added an immutable
  model/dataset/config/job manifest and a fail-closed verifier for the two AMN
  posterity studies. The verifier requires exact task/log coverage,
  receipt-bound artifact hashes, consistent runtime provenance, structured
  validator markers, frozen analysis and the registered metric order before it
  writes `THESIS_EVIDENCE.json`. The exact glibc-2.17 AMN dependency lock is
  retained and checksum-bound. This changes no model, data, prompt, method,
  seed, training or evaluation behavior. Verification covers manifest/config/
  environment-lock consistency and rejection tests for incomplete or mixed
  runtime provenance.

- **2026-09-01, AMN batch Python resolution repair:** changed the two posterity
  payload runners to invoke the experiment virtual environment's Python by
  absolute path. AMN's login shell had cached `/usr/bin/python` 2.7 before
  `PATH` and virtual-environment setup, causing tasks to fail before training
  at the `importlib.metadata` dependency gate. This is an infrastructure-only
  repair; YAML, task mapping, model, prompts, objectives and training code are
  unchanged. Verification covers a batch-order reproduction, focused runner
  tests, shell syntax, Python compilation, all-coordinate dry runs and a clean
  GPU smoke start before releasing the replacement arrays.

- **2026-09-01, AMN posterity execution port:** imported the minimal frozen
  Qwen3/GSM8K execution, SGE, validation and analysis surface from source
  revision `3472a14`. Added AMN Python-runtime initialisation and an independent
  reference-kernel audit. The two generated protocols retain their original
  91-task and 14-task coordinates; no method, seed, prompt, objective, update,
  adapter or evaluation setting changed. Verification covers generated-YAML
  checks, all-coordinate dry runs, Python compilation, shell syntax, focused
  tests and numerical agreement between the training and reference kernels.
