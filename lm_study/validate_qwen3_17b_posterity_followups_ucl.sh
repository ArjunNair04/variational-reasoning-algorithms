#!/bin/bash -l
#$ -N vrl_q3_pfv
#$ -l h_rt=04:00:00
#$ -l tmem=8G
#$ -cwd -j y -o logs/qwen3_posterity_followups_validate.$JOB_ID.log

set -euo pipefail

export VRL_ROOT=${VRL_ROOT:-$HOME/vrl_hpc}
export PROJ=${PROJ:-$VRL_ROOT/vrl}
export PAYLOAD_PROJ=${PAYLOAD_PROJ:-$PROJ}
export VENV=${VENV:-$VRL_ROOT/po_venv}

for required in SOURCE_JOB_ID VALIDATOR_COMMIT EXECUTION_COMMIT EXPECTED_CONFIG_SHA256 YAML MARKER; do
  test -n "${!required:-}" || { echo "ERROR: $required missing" >&2; exit 3; }
done
test "$(cd "$PROJ" && git rev-parse HEAD)" = "$VALIDATOR_COMMIT" || exit 3
test "$(cd "$PAYLOAD_PROJ" && git rev-parse HEAD)" = "$EXECUTION_COMMIT" || exit 3
test -z "$(cd "$PROJ" && git status --porcelain --untracked-files=no)" || exit 3

cd "$PROJ/lm_study"
source "$VENV/bin/activate"
"$VENV/bin/python" generate_qwen3_17b_posterity_followups.py --check "$YAML"
"$VENV/bin/python" validate_qwen3_17b_posterity_followups.py \
  --config "$YAML" \
  --log-glob "$PAYLOAD_PROJ/lm_study/logs/qwen3_posterity_followups.${SOURCE_JOB_ID}.*.log" \
  --marker "$MARKER" \
  --expected-commit "$EXECUTION_COMMIT" \
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256" \
  --source-job-id "$SOURCE_JOB_ID"
