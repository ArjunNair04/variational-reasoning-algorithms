#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ=${PROJ:-$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)}
VENV=${VENV:-$HOME/vrl_hpc/po_venv}
YAML=${YAML:-$SCRIPT_DIR/experiments_qwen3_17b_q5_prior_segments.yaml}
GENERATOR=${GENERATOR:-$SCRIPT_DIR/generate_qwen3_17b_q5_prior_segments.py}
PAYLOAD_RUNNER=${PAYLOAD_RUNNER:-run_qwen3_17b_q5_prior_segments_ucl.sh}
VALIDATOR_RUNNER=${VALIDATOR_RUNNER:-validate_qwen3_17b_q5_prior_segments_ucl.sh}
LOG_PREFIX=${LOG_PREFIX:-qwen3_q5_prior_segments}
RESULT_OUT=${RESULT_OUT:-$HOME/po_results/2026-09-14/q5-prior-segments/qwen3-q5-prior-segments__c7e32a91}
MARKER=${MARKER:-$HOME/po_results/auto_state/qwen3_q5_prior_segments_c7e32a91.ok}
CONTROL_MARKER=${CONTROL_MARKER:-$HOME/po_results/auto_state/qwen3_selected_method_posterity_68078ecc.ok}
CLAIM_DIR=${CLAIM_DIR:-$HOME/po_results/auto_state/qwen3_q5_prior_segments_c7e32a91.submit.claim}
SCORING_RESULTS=${SCORING_RESULTS:-$HOME/po_results/2026-09-14/q5-segment-scoring/q5-fixed-final-scoring__802c174}
SCORING_LOGS=${SCORING_LOGS:-$HOME/vrl_hpc/audits/q5-segment-scoring-802c174/lm_study/logs}

source "$SCRIPT_DIR/ucl_python_env.sh"
export PYTHONPATH="$PROJ/src:$SCRIPT_DIR:$PROJ/analysis:${PYTHONPATH:-}"

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
  "$SCRIPT_DIR/ac_alg1.py" \
  "$SCRIPT_DIR/ac_alg1_prior_segments.py" \
  "$SCRIPT_DIR/run_sweep_lm.py" \
  "$SCRIPT_DIR/run_yaml.py" \
  "$SCRIPT_DIR/generate_qwen3_17b_q5_prior_segments.py" \
  "$SCRIPT_DIR/validate_qwen3_17b_q5_prior_segments.py" \
  "$PROJ/analysis/analyze_qwen3_q5_prior_segments.py" \
  "$PROJ/analysis/check_q5_segment_training_gate.py"
bash -n "$SCRIPT_DIR/$PAYLOAD_RUNNER"
bash -n "$SCRIPT_DIR/$VALIDATOR_RUNNER"
"$VENV/bin/python" "$GENERATOR" --check "$YAML" --runtime-check
"$VENV/bin/python" \
  "$PROJ/analysis/analyze_qwen3_q5_prior_segments.py" \
  --config "$YAML" --validate-design-only >/dev/null
"$VENV/bin/python" "$PROJ/analysis/analyze_qwen3_q5_prior_segments.py" \
  --config "$YAML" --validate-controls-only --control-marker "$CONTROL_MARKER" \
  --control-dir "$HOME/po_results/2026-08-29/reproducibility/qwen3-selected-method-posterity__68078ecc"

for cell in 0 1 2; do
  for seed in $(seq 0 2); do
    "$VENV/bin/python" "$SCRIPT_DIR/run_yaml.py" "$YAML" \
      --dry-run --seed "$seed" --shard "$cell" --nshard 3 \
      --expect-cells 1 >/dev/null
  done
done

config_sha=$("$VENV/bin/python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$YAML")
"$VENV/bin/python" "$PROJ/analysis/check_q5_segment_training_gate.py" \
  --check-training-gate "$YAML" \
  --source "$HOME/po_results/2026-08-29/reproducibility/qwen3-selected-method-posterity__68078ecc" \
  --scoring-results "$SCORING_RESULTS" --scoring-logs "$SCORING_LOGS"
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  printf 'preflight_ok=1\nexecution_commit=%s\nconfig_sha256=%s\n' "$expected_commit" "$config_sha"
  exit 0
fi

mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"
payload_output=$(qsub -h \
  -v "PROJ=$PROJ,EXPECTED_COMMIT=$expected_commit,EXPECTED_CONFIG_SHA256=$config_sha,YAML=$YAML" \
  "$PAYLOAD_RUNNER")
payload_job=$(printf '%s\n' "$payload_output" | sed -nE 's/.*job-array ([0-9]+).*/\1/p')
test -n "$payload_job" || { echo "ERROR: cannot parse payload: $payload_output" >&2; exit 3; }

set +e
validator_output=$(qsub -hold_jid "$payload_job" \
  -v "PROJ=$PROJ,PAYLOAD_PROJ=$PROJ,SOURCE_JOB_ID=$payload_job,VALIDATOR_COMMIT=$expected_commit,EXECUTION_COMMIT=$expected_commit,EXPECTED_CONFIG_SHA256=$config_sha,YAML=$YAML,MARKER=$MARKER,GENERATOR=$GENERATOR,LOG_PREFIX=$LOG_PREFIX" \
  "$VALIDATOR_RUNNER" 2>&1)
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
run_id=$("$VENV/bin/python" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["run_id"])' "$YAML")
printf 'execution_commit=%s\nrun_id=%s\npayload_job=%s\nvalidator_job=%s\nconfig_sha256=%s\nresult_out=%s\nmarker=%s\nstate=user_held_pending_release\n' \
  "$expected_commit" "$run_id" "$payload_job" "$validator_job" "$config_sha" "$RESULT_OUT" "$MARKER"
