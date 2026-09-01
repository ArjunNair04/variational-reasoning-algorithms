#!/bin/bash -l
#$ -N vrl_q3_focusv
#$ -l h_rt=04:00:00
#$ -l tmem=8G
#$ -cwd -j y -o logs/qwen3_l2r_focused_followup_validate.$JOB_ID.log

set -euo pipefail

export VRL_ROOT=${VRL_ROOT:-$HOME/vrl_hpc}
export PROJ=${PROJ:-$VRL_ROOT/vrl}
export PAYLOAD_PROJ=${PAYLOAD_PROJ:-$PROJ}
export VENV=${VENV:-$VRL_ROOT/po_venv}

for required in SOURCE_JOB_ID VALIDATOR_COMMIT EXECUTION_COMMIT EXPECTED_CONFIG_SHA256 YAML MARKER RUN_ID LOG_STEM; do
  test -n "${!required:-}" || { echo "ERROR: $required missing" >&2; exit 3; }
done
test "$(cd "$PROJ" && git rev-parse HEAD)" = "$VALIDATOR_COMMIT" || exit 3
test "$(cd "$PAYLOAD_PROJ" && git rev-parse HEAD)" = "$EXECUTION_COMMIT" || exit 3
test -z "$(cd "$PROJ" && git status --porcelain --untracked-files=no)" || exit 3
test -z "$(cd "$PAYLOAD_PROJ" && git status --porcelain --untracked-files=no)" || exit 3

cd "$PROJ/lm_study"
source "$VENV/bin/activate"
python validate_yaml_run.py "$YAML" \
  --log-glob "$PAYLOAD_PROJ/lm_study/logs/${LOG_STEM}.${SOURCE_JOB_ID}.*.log" \
  --marker "$MARKER" \
  --marker-run-id "$RUN_ID" \
  --marker-commit "$EXECUTION_COMMIT" \
  --marker-config-sha256 "$EXPECTED_CONFIG_SHA256" \
  --marker-source-job-id "$SOURCE_JOB_ID"
