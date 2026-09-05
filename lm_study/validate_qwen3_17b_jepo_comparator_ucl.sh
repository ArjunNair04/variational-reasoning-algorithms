#!/bin/bash -l
#$ -N vrl_q3_jepv
#$ -l h_rt=02:00:00
#$ -l tmem=8G
#$ -cwd -j y -o logs/qwen3_jepo_comparator_validator.$JOB_ID.log

set -euo pipefail

PROJ=${PROJ:-$HOME/vrl_hpc/vrl}
PAYLOAD_PROJ=${PAYLOAD_PROJ:-$PROJ}
VENV=${VENV:-$HOME/vrl_hpc/po_venv}
YAML=${YAML:-$PAYLOAD_PROJ/lm_study/experiments_qwen3_17b_jepo_comparator.yaml}
MARKER=${MARKER:-$HOME/po_results/auto_state/qwen3_jepo_comparator_6c94a797.ok}
SOURCE_JOB_ID=${SOURCE_JOB_ID:?SOURCE_JOB_ID is required}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?VALIDATOR_COMMIT is required}
EXECUTION_COMMIT=${EXECUTION_COMMIT:?EXECUTION_COMMIT is required}
EXPECTED_CONFIG_SHA256=${EXPECTED_CONFIG_SHA256:?EXPECTED_CONFIG_SHA256 is required}

source "$PROJ/lm_study/ucl_python_env.sh"
export PYTHONPATH="$PROJ/src:$PROJ/lm_study${PYTHONPATH:+:$PYTHONPATH}"
test "$(cd "$PROJ" && git rev-parse HEAD)" = "$VALIDATOR_COMMIT" || exit 2
test -z "$(cd "$PROJ" && git status --porcelain --untracked-files=no)" || exit 2
"$VENV/bin/python" -m py_compile \
  "$PROJ/lm_study/validate_qwen3_17b_jepo_comparator.py"
"$VENV/bin/python" "$PROJ/lm_study/validate_qwen3_17b_jepo_comparator.py" \
  --config "$YAML" \
  --log-glob "$PAYLOAD_PROJ/lm_study/logs/qwen3_jepo_comparator.${SOURCE_JOB_ID}.*.log" \
  --marker "$MARKER" \
  --expected-commit "$EXECUTION_COMMIT" \
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256" \
  --source-job-id "$SOURCE_JOB_ID"
