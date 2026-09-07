# Public implementation instructions

Keep the short NumPy functions readable. The files under `experiments/frozen` are
source snapshots of the executed studies; change them only with an explicit
source and configuration audit, preserving original hashes and listing every
portability change. A clean reference function is not a substitute for the
executed model-training procedure.

Do not add thesis source, personal correspondence, credentials, scheduler/SSH
scripts, personal filesystem paths, model weights, or raw dataset responses.
Use only the GSM8K training/validation partition in experiment checks; never
load the official GSM8K test split. Preparation and dry-run checks must not
load models or datasets. Do not run costly training as part of the test suite.

Every change must update the record below, with its scope and actual checks.

## Recent verified changes

- 2026-09-07: added source-bound local replay for the thesis studies, with exact
  execution revisions, separate archived analyses and frozen numeric summaries.
  Kept the compact reference kernels distinct from the executed trainer.
  CSV attributes preserve the frozen exports' original record endings and hashes.
  Verification is recorded in `experiments/VERIFICATION.md`; no new model
  training, cluster operation or official GSM8K test access was performed.
