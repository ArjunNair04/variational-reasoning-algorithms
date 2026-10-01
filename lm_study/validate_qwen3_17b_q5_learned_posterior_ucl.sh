#!/bin/bash -l
#$ -N vrl_q5_pov
#$ -l h_rt=04:00:00
#$ -l tmem=8G
#$ -cwd -j y -o logs/qwen3_q5_posterior_validate.$JOB_ID.log

set -euo pipefail
: "${PROJ:?immutable checkout required}"
: "${EXPECTED_COMMIT:?execution commit required}"
: "${EXPECTED_CONFIG_SHA256:?configuration hash required}"
: "${SOURCE_JOB_ID:?source payload required}"
: "${CONTROL_PROJ:?existing control checkout required}"
VENV=${VENV:-$HOME/vrl_hpc/po_venv}
export PYTHONPATH="$PROJ/src:$PROJ/lm_study:$PROJ/analysis:${PYTHONPATH:-}"
source "$PROJ/lm_study/ucl_python_env.sh"
test "$(cd "$PROJ" && git rev-parse HEAD)" = "$EXPECTED_COMMIT" || exit 2
test -z "$(cd "$PROJ" && git status --porcelain --untracked-files=no)" || exit 2
"$VENV/bin/python" "$PROJ/analysis/analyze_qwen3_q5_learned_posterior.py" \
  --config "$PROJ/lm_study/experiments_qwen3_17b_q5_learned_posterior.yaml" \
  --logs "$PROJ/lm_study/logs" --control-logs "$CONTROL_PROJ/lm_study/logs" \
  --control-config "$CONTROL_PROJ/lm_study/experiments_qwen3_17b_q5_prefix_continuations.yaml" \
  --marker "$HOME/po_results/auto_state/qwen3_q5_posterior_2d7c8a61.ok" \
  --output-dir "$HOME/po_results/2026-10-02/analysis/qwen3_q5_posterior_2d7c8a61" \
  --expected-commit "$EXPECTED_COMMIT" --expected-config-sha256 "$EXPECTED_CONFIG_SHA256" \
  --source-job-id "$SOURCE_JOB_ID"
