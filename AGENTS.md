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
