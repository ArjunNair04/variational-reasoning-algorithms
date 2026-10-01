#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)
VENV=${VENV:-$HOME/vrl_hpc/po_venv}
YAML="$SCRIPT_DIR/experiments_qwen3_17b_q5_prefix_continuations.yaml"
STATE="$HOME/po_results/auto_state"
CLAIM="$STATE/qwen3_q5_prefix_f812ca40.submit.claim"
RECEIPT="$STATE/qwen3_q5_prefix_f812ca40.submission.txt"
export PYTHONPATH="$PROJ/src:$SCRIPT_DIR:$PROJ/analysis:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/vrl_hpc/hf_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
source "$SCRIPT_DIR/ucl_python_env.sh"
test -x "$VENV/bin/python" || { echo "ERROR: missing experiment Python"; exit 2; }
test -z "$(cd "$PROJ" && git status --porcelain --untracked-files=no)" || exit 2
commit=$(cd "$PROJ" && git rev-parse HEAD)
sha=$("$VENV/bin/python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$YAML")
for path in "$RECEIPT" "$STATE/qwen3_q5_prefix_f812ca40.ok" \
  "$HOME/po_results/2026-10-01/q5-prefix-continuations/qwen3-q5-prefix-continuations__f812ca40"; do
  test ! -e "$path" || { echo "ERROR: study already has output or tracking: $path"; exit 2; }
done
"$VENV/bin/python" "$SCRIPT_DIR/generate_qwen3_17b_q5_prefix_continuations.py" --check "$YAML" --runtime-check
"$VENV/bin/python" "$PROJ/analysis/analyze_qwen3_q5_prefix_continuations.py" --config "$YAML" --validate-design-only
for cell in 0 1; do
  for seed in 0 1 2; do
    "$VENV/bin/python" "$SCRIPT_DIR/run_yaml.py" "$YAML" --dry-run --seed "$seed" --shard "$cell" --nshard 2 --expect-cells 1 >/dev/null
  done
done
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  printf 'preflight_ok=1\nexecution_commit=%s\nconfig_sha256=%s\n' "$commit" "$sha"
  exit 0
fi
mkdir -p "$STATE" "$SCRIPT_DIR/logs"
mkdir "$CLAIM" || { echo "ERROR: submission claim already owned"; exit 2; }
trap 'rmdir "$CLAIM" 2>/dev/null || true' EXIT
test ! -e "$RECEIPT" || exit 2
cd "$SCRIPT_DIR"
payload_output=$(qsub -h -v "PROJ=$PROJ,EXPECTED_COMMIT=$commit,EXPECTED_CONFIG_SHA256=$sha" run_qwen3_17b_q5_prefix_continuations_ucl.sh)
payload=$(printf '%s\n' "$payload_output" | sed -nE 's/.*job-array ([0-9]+).*/\1/p')
test -n "$payload" || { echo "ERROR: cannot parse payload: $payload_output"; exit 3; }
# Preserve the held job ID even if validator submission or tracking fails.
printf 'execution_commit=%s\nrun_id=f812ca40\npayload_job=%s\nconfig_sha256=%s\nstate=user_held\n' "$commit" "$payload" "$sha" > "$RECEIPT"
validator_output=$(qsub -hold_jid "$payload" -v "PROJ=$PROJ,EXPECTED_COMMIT=$commit,EXPECTED_CONFIG_SHA256=$sha,SOURCE_JOB_ID=$payload" validate_qwen3_17b_q5_prefix_continuations_ucl.sh)
validator=$(printf '%s\n' "$validator_output" | sed -nE 's/.*job ([0-9]+).*/\1/p')
test -n "$validator" || { echo "ERROR: cannot parse validator: $validator_output"; exit 4; }
printf 'validator_job=%s\n' "$validator" >> "$RECEIPT"
cat "$RECEIPT"
echo "Payload remains held. Publish tracking, then release this payload only."
