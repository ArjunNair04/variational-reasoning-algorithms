#!/bin/bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ=${PROJ:-$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)}
VENV=${VENV:-$HOME/vrl_hpc/po_venv}
CONFIG=${CONFIG:-$SCRIPT_DIR/experiments_qwen3_17b_gsm8k_panel_robustness.yaml}
RESULT_ROOT=${RESULT_ROOT:-$HOME/po_results/2026-09-04/evaluation/gsm8k-panel-robustness__ec4c38db}
ADAPTER_ROOT=${ADAPTER_ROOT:-$HOME/po_results/2026-08-18/final-comparison/seven-seed-confirmation__978b99c8}
MARKER=${MARKER:-$HOME/po_results/auto_state/qwen3_gsm8k_panel_robustness_ec4c38db.ok}
CLAIM_DIR=${CLAIM_DIR:-$HOME/po_results/auto_state/qwen3_gsm8k_panel_robustness_20260904.submit.claim}

source "$SCRIPT_DIR/ucl_python_env.sh"
test -z "$(cd "$PROJ" && git status --porcelain --untracked-files=no)" || exit 2
expected_commit=$(cd "$PROJ" && git rev-parse HEAD)
test ! -e "$RESULT_ROOT" || { echo "ERROR: result root exists" >&2; exit 2; }
test ! -e "$MARKER" || { echo "ERROR: marker exists" >&2; exit 2; }
mkdir "$CLAIM_DIR" 2>/dev/null || { echo "ERROR: submission claim owned" >&2; exit 2; }
trap 'rmdir "$CLAIM_DIR" 2>/dev/null || true' EXIT

"$VENV/bin/python" -m py_compile \
  "$SCRIPT_DIR/generate_qwen3_17b_gsm8k_panel_robustness.py" \
  "$SCRIPT_DIR/evaluate_qwen3_17b_gsm8k_panel_robustness.py" \
  "$SCRIPT_DIR/validate_qwen3_17b_gsm8k_panel_robustness.py" \
  "$PROJ/analysis/analyze_qwen3_gsm8k_panel_robustness.py"
bash -n "$SCRIPT_DIR/run_qwen3_17b_gsm8k_panel_robustness_ucl.sh"
bash -n "$SCRIPT_DIR/validate_qwen3_17b_gsm8k_panel_robustness_ucl.sh"
"$VENV/bin/python" "$SCRIPT_DIR/generate_qwen3_17b_gsm8k_panel_robustness.py" --check "$CONFIG"
PYTHONPATH="$SCRIPT_DIR:$PROJ/analysis" "$VENV/bin/python" \
  "$PROJ/analysis/analyze_qwen3_gsm8k_panel_robustness.py" \
  --config "$CONFIG" --validate-design-only >/dev/null
PYTHONPATH="$SCRIPT_DIR" "$VENV/bin/python" - "$CONFIG" "$ADAPTER_ROOT" <<'PY'
import sys
from pathlib import Path
import yaml
from evaluate_qwen3_17b_gsm8k_panel_robustness import _adapter_path
config = yaml.safe_load(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2]).expanduser()
count = 0
for method in config["methods"]:
    fragment = method["adapter_name_fragment"]
    if fragment is None:
        continue
    for seed in config["design"]["seeds"]:
        _adapter_path(root, fragment, int(seed))
        count += 1
if count != 35:
    raise SystemExit(f"expected 35 trained adapters, found {count}")
print(f"adapter_preflight={count}")
PY

config_sha=$("$VENV/bin/python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$CONFIG")
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  printf 'preflight_ok=1\nexecution_commit=%s\nconfig_sha256=%s\n' "$expected_commit" "$config_sha"
  exit 0
fi
mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"
payload_output=$(qsub -h \
  -v "PROJ=$PROJ,EXPECTED_COMMIT=$expected_commit,EXPECTED_CONFIG_SHA256=$config_sha,CONFIG=$CONFIG" \
  run_qwen3_17b_gsm8k_panel_robustness_ucl.sh)
payload_job=$(printf '%s\n' "$payload_output" | sed -nE 's/.*job-array ([0-9]+).*/\1/p')
test -n "$payload_job" || { echo "ERROR: cannot parse payload: $payload_output" >&2; exit 3; }
set +e
validator_output=$(qsub -hold_jid "$payload_job" \
  -v "PROJ=$PROJ,SOURCE_JOB_ID=$payload_job,EXPECTED_COMMIT=$expected_commit,EXPECTED_CONFIG_SHA256=$config_sha,CONFIG=$CONFIG,RESULT_ROOT=$RESULT_ROOT,MARKER=$MARKER" \
  validate_qwen3_17b_gsm8k_panel_robustness_ucl.sh 2>&1)
validator_status=$?
set -e
if [ "$validator_status" -ne 0 ]; then
  qdel "$payload_job" >/dev/null 2>&1 || true
  echo "$validator_output" >&2
  exit 4
fi
validator_job=$(printf '%s\n' "$validator_output" | sed -nE 's/.*job ([0-9]+).*/\1/p')
test -n "$validator_job" || { qdel "$payload_job" >/dev/null 2>&1 || true; exit 4; }
printf 'execution_commit=%s\nrun_id=ec4c38db\npayload_job=%s\nvalidator_job=%s\nconfig_sha256=%s\nstate=user_held_pending_release\n' \
  "$expected_commit" "$payload_job" "$validator_job" "$config_sha"
