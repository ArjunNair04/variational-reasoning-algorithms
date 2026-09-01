#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ=${PROJ:-$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)}
VENV=${VENV:-$HOME/vrl_hpc/po_venv}
YAML=${YAML:-$SCRIPT_DIR/experiments_qwen3_17b_posterity_followups.yaml}
RESULT_OUT=${RESULT_OUT:-$HOME/po_results/2026-09-01/reproducibility/qwen3-posterity-followups__f20c9e17}
MARKER=${MARKER:-$HOME/po_results/auto_state/qwen3_posterity_followups_f20c9e17.ok}
CLAIM_DIR=${CLAIM_DIR:-$HOME/po_results/auto_state/qwen3_posterity_followups_20260901.submit.claim}
CONTROL_VALIDATOR_JOB_ID=${CONTROL_VALIDATOR_JOB_ID:?CONTROL_VALIDATOR_JOB_ID is required}

source "$SCRIPT_DIR/ucl_python_env.sh"

test "$CONTROL_VALIDATOR_JOB_ID" -eq "$CONTROL_VALIDATOR_JOB_ID" 2>/dev/null || {
  echo "ERROR: invalid control validator job ID" >&2
  exit 2
}
test -z "$(cd "$PROJ" && git status --porcelain --untracked-files=no)" || {
  echo "ERROR: tracked worktree is dirty" >&2
  exit 2
}
expected_commit=$(cd "$PROJ" && git rev-parse HEAD)
test -x "$VENV/bin/python" || { echo "ERROR: experiment Python missing" >&2; exit 2; }
for path in "$RESULT_OUT" "$MARKER"; do
  test ! -e "$path" || { echo "ERROR: output or marker exists: $path" >&2; exit 2; }
done
mkdir "$CLAIM_DIR" 2>/dev/null || { echo "ERROR: submission claim owned" >&2; exit 2; }
trap 'rmdir "$CLAIM_DIR" 2>/dev/null || true' EXIT

"$VENV/bin/python" -m py_compile \
  "$SCRIPT_DIR/ac_alg1.py" \
  "$SCRIPT_DIR/run_sweep_lm.py" \
  "$SCRIPT_DIR/run_yaml.py" \
  "$SCRIPT_DIR/generate_qwen3_17b_posterity_followups.py" \
  "$SCRIPT_DIR/validate_qwen3_17b_posterity_followups.py" \
  "$PROJ/analysis/analyze_qwen3_posterity_followups.py"
bash -n "$SCRIPT_DIR/run_qwen3_17b_posterity_followups_ucl.sh"
bash -n "$SCRIPT_DIR/validate_qwen3_17b_posterity_followups_ucl.sh"
"$VENV/bin/python" "$SCRIPT_DIR/generate_qwen3_17b_posterity_followups.py" --check "$YAML"
PYTHONPATH="$SCRIPT_DIR:$PROJ/analysis" "$VENV/bin/python" \
  "$PROJ/analysis/analyze_qwen3_posterity_followups.py" \
  --config "$YAML" --validate-design-only >/dev/null

for cell in $(seq 0 5); do
  for seed in $(seq 0 6); do
    "$VENV/bin/python" "$SCRIPT_DIR/run_yaml.py" "$YAML" \
      --dry-run --seed "$seed" --shard "$cell" --nshard 6 \
      --expect-cells 1 >/dev/null
  done
done

config_sha=$("$VENV/bin/python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$YAML")
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  printf 'preflight_ok=1\nexecution_commit=%s\nconfig_sha256=%s\n' "$expected_commit" "$config_sha"
  exit 0
fi

mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"
payload_output=$(qsub -h -hold_jid "$CONTROL_VALIDATOR_JOB_ID" \
  -v "PROJ=$PROJ,EXPECTED_COMMIT=$expected_commit,EXPECTED_CONFIG_SHA256=$config_sha,YAML=$YAML" \
  run_qwen3_17b_posterity_followups_ucl.sh)
payload_job=$(printf '%s\n' "$payload_output" | sed -nE 's/.*job-array ([0-9]+).*/\1/p')
test -n "$payload_job" || { echo "ERROR: cannot parse payload: $payload_output" >&2; exit 3; }

set +e
validator_output=$(qsub -hold_jid "$payload_job" \
  -v "PROJ=$PROJ,PAYLOAD_PROJ=$PROJ,SOURCE_JOB_ID=$payload_job,VALIDATOR_COMMIT=$expected_commit,EXECUTION_COMMIT=$expected_commit,EXPECTED_CONFIG_SHA256=$config_sha,YAML=$YAML,MARKER=$MARKER" \
  validate_qwen3_17b_posterity_followups_ucl.sh 2>&1)
validator_status=$?
set -e
if [ "$validator_status" -ne 0 ]; then
  qdel "$payload_job" >/dev/null 2>&1 || true
  echo "ERROR: validator submission failed; cancelled held payload $payload_job" >&2
  echo "$validator_output" >&2
  exit 4
fi
validator_job=$(printf '%s\n' "$validator_output" | sed -nE 's/.*job ([0-9]+).*/\1/p')
test -n "$validator_job" || { qdel "$payload_job" >/dev/null 2>&1 || true; exit 4; }
printf 'execution_commit=%s\nrun_id=f20c9e17\npayload_job=%s\nvalidator_job=%s\ncontrol_validator_job=%s\nconfig_sha256=%s\nresult_out=%s\nmarker=%s\nstate=user_held_and_dependency_held_pending_release\n' \
  "$expected_commit" "$payload_job" "$validator_job" "$CONTROL_VALIDATOR_JOB_ID" "$config_sha" "$RESULT_OUT" "$MARKER"
