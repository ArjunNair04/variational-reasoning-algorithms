#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ=${PROJ:-$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)}
VENV=${VENV:-$HOME/vrl_hpc/po_venv}
YAML=${YAML:-$SCRIPT_DIR/experiments_qwen3_17b_jepo_comparator.yaml}
RESULT_OUT=${RESULT_OUT:-$HOME/po_results/2026-09-04/jepo-comparator/qwen3-jepo-ms4__7452ba96}
MARKER=${MARKER:-$HOME/po_results/auto_state/qwen3_jepo_comparator_7452ba96.ok}
CONTROL_MARKER=${CONTROL_MARKER:-$HOME/po_results/auto_state/qwen3_selected_method_posterity_68078ecc.ok}
CLAIM_DIR=${CLAIM_DIR:-$HOME/po_results/auto_state/qwen3_jepo_comparator_20260904.submit.claim}

source "$SCRIPT_DIR/ucl_python_env.sh"

test -z "$(cd "$PROJ" && git status --porcelain --untracked-files=no)" || {
  echo "ERROR: tracked worktree is dirty" >&2
  exit 2
}
expected_commit=$(cd "$PROJ" && git rev-parse HEAD)
test -x "$VENV/bin/python" || { echo "ERROR: experiment Python missing" >&2; exit 2; }
for path in "$RESULT_OUT" "$MARKER"; do
  test ! -e "$path" || { echo "ERROR: output or marker exists: $path" >&2; exit 2; }
done
"$VENV/bin/python" - "$CONTROL_MARKER" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
required = {"status": "ok", "run_id": "68078ecc", "task_count": 91, "official_test_used": False}
for key, value in required.items():
    if payload.get(key) != value:
        raise SystemExit("invalid control marker field %s" % key)
PY
mkdir "$CLAIM_DIR" 2>/dev/null || { echo "ERROR: submission claim owned" >&2; exit 2; }
trap 'rmdir "$CLAIM_DIR" 2>/dev/null || true' EXIT

"$VENV/bin/python" -m py_compile \
  "$SCRIPT_DIR/jepo.py" \
  "$SCRIPT_DIR/run_sweep_lm.py" \
  "$SCRIPT_DIR/run_yaml.py" \
  "$SCRIPT_DIR/generate_qwen3_17b_jepo_comparator.py" \
  "$SCRIPT_DIR/validate_qwen3_17b_jepo_comparator.py" \
  "$PROJ/analysis/analyze_qwen3_jepo_comparator.py"
bash -n "$SCRIPT_DIR/run_qwen3_17b_jepo_comparator_ucl.sh"
bash -n "$SCRIPT_DIR/validate_qwen3_17b_jepo_comparator_ucl.sh"
"$VENV/bin/python" "$SCRIPT_DIR/generate_qwen3_17b_jepo_comparator.py" --check "$YAML"
PYTHONPATH="$SCRIPT_DIR:$PROJ/analysis:$PROJ/src" "$VENV/bin/python" \
  "$PROJ/analysis/analyze_qwen3_jepo_comparator.py" \
  --config "$YAML" --validate-design-only >/dev/null

for seed in $(seq 0 6); do
  "$VENV/bin/python" "$SCRIPT_DIR/run_yaml.py" "$YAML" \
    --dry-run --seed "$seed" --shard 0 --nshard 1 \
    --expect-cells 1 >/dev/null
done

config_sha=$("$VENV/bin/python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$YAML")
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  printf 'preflight_ok=1\nexecution_commit=%s\nconfig_sha256=%s\n' "$expected_commit" "$config_sha"
  exit 0
fi

mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"
payload_output=$(qsub -h \
  -v "PROJ=$PROJ,EXPECTED_COMMIT=$expected_commit,EXPECTED_CONFIG_SHA256=$config_sha,YAML=$YAML" \
  run_qwen3_17b_jepo_comparator_ucl.sh)
payload_job=$(printf '%s\n' "$payload_output" | sed -nE 's/.*job-array ([0-9]+).*/\1/p')
test -n "$payload_job" || { echo "ERROR: cannot parse payload: $payload_output" >&2; exit 3; }

set +e
validator_output=$(qsub -hold_jid "$payload_job" \
  -v "PROJ=$PROJ,PAYLOAD_PROJ=$PROJ,SOURCE_JOB_ID=$payload_job,VALIDATOR_COMMIT=$expected_commit,EXECUTION_COMMIT=$expected_commit,EXPECTED_CONFIG_SHA256=$config_sha,YAML=$YAML,MARKER=$MARKER" \
  validate_qwen3_17b_jepo_comparator_ucl.sh 2>&1)
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
printf 'execution_commit=%s\nrun_id=7452ba96\npayload_job=%s\nvalidator_job=%s\nconfig_sha256=%s\nresult_out=%s\nmarker=%s\nstate=user_held_pending_release\n' \
  "$expected_commit" "$payload_job" "$validator_job" "$config_sha" "$RESULT_OUT" "$MARKER"
