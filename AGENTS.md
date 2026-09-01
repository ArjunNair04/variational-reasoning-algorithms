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
