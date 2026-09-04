#!/bin/bash -l
#$ -N vrl_q3_gpv
#$ -l h_rt=02:00:00
#$ -l tmem=8G
#$ -cwd -j y -o logs/qwen3_gsm8k_panel_robustness_validator.$JOB_ID.log

set -euo pipefail
PROJ=${PROJ:-$HOME/vrl_hpc/vrl}
VENV=${VENV:-$HOME/vrl_hpc/po_venv}
CONFIG=${CONFIG:?CONFIG is required}
RESULT_ROOT=${RESULT_ROOT:?RESULT_ROOT is required}
MARKER=${MARKER:?MARKER is required}
SOURCE_JOB_ID=${SOURCE_JOB_ID:?SOURCE_JOB_ID is required}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}
EXPECTED_CONFIG_SHA256=${EXPECTED_CONFIG_SHA256:?EXPECTED_CONFIG_SHA256 is required}
source "$PROJ/lm_study/ucl_python_env.sh"
test "$(cd "$PROJ" && git rev-parse HEAD)" = "$EXPECTED_COMMIT" || exit 2
cd "$PROJ/lm_study"
"$VENV/bin/python" validate_qwen3_17b_gsm8k_panel_robustness.py \
  --config "$CONFIG" --result-root "$RESULT_ROOT" \
  --log-glob "$PROJ/lm_study/logs/qwen3_gsm8k_panel_robustness.${SOURCE_JOB_ID}.*.log" \
  --marker "$MARKER" --expected-commit "$EXPECTED_COMMIT" \
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256" \
  --source-job-id "$SOURCE_JOB_ID"
PYTHONPATH="$PROJ/lm_study:$PROJ/analysis" "$VENV/bin/python" \
  "$PROJ/analysis/analyze_qwen3_gsm8k_panel_robustness.py" \
  --config "$CONFIG" --result-root "$RESULT_ROOT"
