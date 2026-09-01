#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ=${PROJ:-$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)}
VENV=${VENV:-$HOME/vrl_hpc/po_venv}
YAML=${YAML:-$SCRIPT_DIR/experiments_qwen3_17b_final_method_confirmation.yaml}
RESULT_OUT=${RESULT_OUT:-$HOME/po_results/2026-08-18/final-comparison/seven-seed-confirmation__978b99c8}
MARKER=${MARKER:-$HOME/po_results/auto_state/qwen3_final_method_confirmation_978b99c8.ok}
CLAIM_DIR=${CLAIM_DIR:-$HOME/po_results/auto_state/qwen3_final_method_confirmation_20260818.submit.claim}

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
  "$SCRIPT_DIR/common.py" \
  "$SCRIPT_DIR/self_training.py" \
  "$SCRIPT_DIR/ac_alg1_trice.py" \
  "$SCRIPT_DIR/run_sweep_lm.py" \
  "$SCRIPT_DIR/run_yaml.py" \
  "$SCRIPT_DIR/validate_yaml_run.py" \
  "$SCRIPT_DIR/generate_qwen3_17b_final_method_confirmation.py" \
  "$SCRIPT_DIR/validate_qwen3_17b_final_method_confirmation.py" \
  "$PROJ/analysis/analyze_qwen3_final_method_confirmation.py"
bash -n "$SCRIPT_DIR/run_qwen3_17b_final_method_confirmation_ucl.sh"
bash -n "$SCRIPT_DIR/validate_qwen3_17b_final_method_confirmation_ucl.sh"
"$VENV/bin/python" "$SCRIPT_DIR/generate_qwen3_17b_final_method_confirmation.py" --check "$YAML"

mapping=$(PYTHONPATH="$SCRIPT_DIR" "$VENV/bin/python" -c '
import sys, yaml
from run_yaml import _prepare_cells
payload = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
cells = _prepare_cells(payload, only=None, run_id=str(payload["run_id"]), defaults=payload["defaults"])
expected_methods = ("base", "AC-ALG1", "AC-ALG1", "Gold-CoT-SFT", "RFT-Source", "ReST-EM", "STaR", "TRICE", "GRPO", "RLOO")
observed_methods = tuple(cell.method for cell in cells)
if len(cells) != 10 or observed_methods != expected_methods:
    raise SystemExit(f"unexpected final-confirmation cells: {len(cells)} {observed_methods}")
if tuple(payload["defaults"]["seed_values"]) != (1201, 1213, 1217, 1223, 1229, 1231, 1237):
    raise SystemExit("unexpected final-confirmation seeds")
if payload["defaults"].get("save_adapter") is not True:
    raise SystemExit("trained adapters are not retained")
print("10 7 70")
' "$YAML")
test "$mapping" = "10 7 70" || exit 2

for cell in $(seq 0 9); do
  for seed in $(seq 0 6); do
    "$VENV/bin/python" "$SCRIPT_DIR/run_yaml.py" "$YAML" \
      --dry-run --seed "$seed" --shard "$cell" --nshard 10 \
      --expect-cells 1 >/dev/null
  done
done

config_sha=$("$VENV/bin/python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$YAML")
mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"
payload_output=$(qsub -h -v "PROJ=$PROJ,EXPECTED_COMMIT=$expected_commit,EXPECTED_CONFIG_SHA256=$config_sha,YAML=$YAML" run_qwen3_17b_final_method_confirmation_ucl.sh)
payload_job=$(printf '%s\n' "$payload_output" | sed -nE 's/.*job-array ([0-9]+).*/\1/p')
test -n "$payload_job" || { echo "ERROR: cannot parse payload: $payload_output" >&2; exit 3; }

set +e
validator_output=$(qsub -hold_jid "$payload_job" -v "PROJ=$PROJ,PAYLOAD_PROJ=$PROJ,SOURCE_JOB_ID=$payload_job,VALIDATOR_COMMIT=$expected_commit,EXECUTION_COMMIT=$expected_commit,EXPECTED_CONFIG_SHA256=$config_sha,YAML=$YAML,MARKER=$MARKER" validate_qwen3_17b_final_method_confirmation_ucl.sh 2>&1)
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
printf 'execution_commit=%s\npayload_job=%s\nvalidator_job=%s\nconfig_sha256=%s\nstate=user_held_pending_tracking\n' "$expected_commit" "$payload_job" "$validator_job" "$config_sha"
