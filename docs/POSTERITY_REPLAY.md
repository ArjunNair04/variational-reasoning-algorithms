# Selected-method posterity replay

## Purpose

This port makes the frozen Qwen3-1.7B/GSM8K replay executable from this
repository without changing its scientific design. The original model-training
implementation is retained under `lm_study/`. The compact NumPy package remains
an independent statement of the update rules and is used as a deployment-time
numerical oracle.

## Studies

| Run | Design | Tasks | Seeds |
|---|---|---:|---:|
| `68078ecc` | Frozen base; Q5 prompt, reader, ESS, KL and temperature controls; PIS prompt controls; ReST-EM; STaR; TRICE-CV; RLOO | 91 | 7 |
| `f3950b2e` | PIS temperature-one control versus exact 50:50 temperature-one/temperature-1.2 mixture | 14 | 7 |
| `f20c9e17` | Q5 support-depth and token-mean follow-ups; PIS update-reuse and rationale-only KL; exact signed sampled-support updates | 42 | 7 |

All studies use the fixed training-derived 400-question validation partition.
The official GSM8K test split remains unused. Results are reported as Final
Acc@1, strict final accuracy and normalized trajectory AUC, in that order.

## Provenance boundary

- The execution files under `lm_study/` and `analysis/` originate from source
  revision `3472a14`.
- The generated YAML files retain their original run IDs, source hashes, seeds
  and output contracts.
- `lm_study/reference_kernel_audit.py` compares Q5, PIS and off-policy
  importance weights against `src/variational_reasoning/` without entering the
  training path.
- AMN runtime initialization and submitter preflight support are infrastructure
  changes only.

## Local verification

```bash
python -m pytest
python lm_study/generate_qwen3_17b_selected_method_posterity.py \
  --check lm_study/experiments_qwen3_17b_selected_method_posterity.yaml
python lm_study/generate_qwen3_17b_pis_temperature_posterity.py \
  --check lm_study/experiments_qwen3_17b_pis_temperature_posterity.yaml
python lm_study/generate_qwen3_17b_posterity_followups.py \
  --check lm_study/experiments_qwen3_17b_posterity_followups.yaml
PYTHONPATH=src:lm_study python lm_study/reference_kernel_audit.py
```

## AMN submission

The submitters create user-held payload arrays and dependency-held validators.
Inspect the returned job IDs and array contracts before releasing the payloads.

```bash
cd "$HOME/vrl_hpc/checkouts/<immutable-checkout>/lm_study"
bash submit_qwen3_17b_selected_method_posterity_ucl.sh
bash submit_qwen3_17b_pis_temperature_posterity_ucl.sh
CONTROL_VALIDATOR_JOB_ID=<selected-method-validator-job> \
  bash submit_qwen3_17b_posterity_followups_ucl.sh
```

The array scripts omit `-tc`; SGE resource availability controls concurrency.
Disappearance from `qstat` is not evidence of success. Completion requires all
task receipts, artifacts, logs and validator markers.

## Thesis evidence gate

The run is provisional until the independent evidence gate succeeds. The gate
checks the immutable model and dataset revisions, YAML checksums, exact task
coverage, task logs, completion-receipt hashes, runtime package/GPU provenance,
validator marker and frozen analysis before writing `THESIS_EVIDENCE.json`.
The full AMN dependency graph is retained with package hashes in
`requirements/cluster_qwen3_glibc217.lock`; its checksum is part of the
reproducibility manifest.

```bash
python analysis/verify_thesis_reproduction.py \
  --study selected_methods \
  --validate-contract-only
python analysis/verify_thesis_reproduction.py \
  --study temperature_mixture \
  --validate-contract-only
```

For a completed run, supply the result root, validator marker, log directory
and a new analysis output directory. A selected-method replay may also receive
the checksum-bound historical `method_summary.csv` through `--source-summary`.
Without that source summary, the replay result may be cited after the gate
passes, but it must not be described as reproducing the historical estimate.
