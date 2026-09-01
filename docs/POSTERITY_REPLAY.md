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

Both studies use the fixed training-derived 400-question validation partition.
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
PYTHONPATH=src:lm_study python lm_study/reference_kernel_audit.py
```

## AMN submission

The submitters create user-held payload arrays and dependency-held validators.
Inspect the returned job IDs and array contracts before releasing the payloads.

```bash
cd "$HOME/vrl_hpc/checkouts/<immutable-checkout>/lm_study"
bash submit_qwen3_17b_selected_method_posterity_ucl.sh
bash submit_qwen3_17b_pis_temperature_posterity_ucl.sh
```

The array scripts omit `-tc`; SGE resource availability controls concurrency.
Disappearance from `qstat` is not evidence of success. Completion requires all
task receipts, artifacts, logs and validator markers.
