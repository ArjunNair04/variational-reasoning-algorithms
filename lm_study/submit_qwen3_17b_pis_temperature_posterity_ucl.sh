#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ=${PROJ:-$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)}
VENV=${VENV:-$HOME/vrl_hpc/po_venv}
YAML=${YAML:-$SCRIPT_DIR/experiments_qwen3_17b_pis_temperature_posterity.yaml}
RESULT_OUT=${RESULT_OUT:-$HOME/po_results/2026-08-29/reproducibility/pis-temperature-mixture-posterity__f3950b2e}
MARKER=${MARKER:-$HOME/po_results/auto_state/qwen3_pis_temperature_posterity_f3950b2e.ok}
CLAIM_DIR=${CLAIM_DIR:-$HOME/po_results/auto_state/qwen3_pis_temperature_posterity_20260829.submit.claim}

# shellcheck disable=SC1091
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
mkdir "$CLAIM_DIR" 2>/dev/null || { echo "ERROR: submission claim owned" >&2; exit 2; }
trap 'rmdir "$CLAIM_DIR" 2>/dev/null || true' EXIT

"$VENV/bin/python" -m py_compile \
  "$SCRIPT_DIR/l2r.py" \
  "$SCRIPT_DIR/trainer_config.py" \
  "$SCRIPT_DIR/run_yaml.py" \
  "$SCRIPT_DIR/validate_yaml_run.py" \
  "$SCRIPT_DIR/generate_qwen3_17b_pis_temperature_posterity.py" \
  "$SCRIPT_DIR/reference_kernel_audit.py" \
  "$PROJ/analysis/analyze_l2r_temperature_mixture.py"
bash -n "$SCRIPT_DIR/ucl_python_env.sh"
bash -n "$SCRIPT_DIR/run_qwen3_17b_pis_temperature_posterity_ucl.sh"
bash -n "$SCRIPT_DIR/validate_qwen3_17b_l2r_focused_followup_ucl.sh"
PYTHONPATH="$PROJ/src:$SCRIPT_DIR" "$VENV/bin/python" \
  "$SCRIPT_DIR/reference_kernel_audit.py" >/dev/null
"$VENV/bin/python" "$SCRIPT_DIR/generate_qwen3_17b_pis_temperature_posterity.py" --check "$YAML"
config_sha=$("$VENV/bin/python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$YAML")
PYTHONPATH="$SCRIPT_DIR:$PROJ" "$VENV/bin/python" \
  "$PROJ/analysis/analyze_l2r_temperature_mixture.py" \
  --config "$YAML" \
  --run-id f3950b2e \
  --expected-commit "$expected_commit" \
  --source-job pending \
  --config-sha256 "$config_sha" \
  --tag-prefix q3_l2r_temperature_mixture \
  --log-stem qwen3_17b_pis_temperature_posterity \
  --validate-design-only >/dev/null

PYTHONPATH="$SCRIPT_DIR" "$VENV/bin/python" - "$YAML" <<'PY'
import sys, yaml
from run_yaml import _prepare_cells
p=yaml.safe_load(open(sys.argv[1], encoding="utf-8")); d=p["defaults"]
c=_prepare_cells(p, only=None, run_id=str(p["run_id"]), defaults=d)
assert tuple(x.tag.rsplit("_",1)[-1] for x in c)==("PIS-T1","PIS-TMIX1.2")
assert tuple(map(int,d["seed_values"]))==(1481,1483,1487,1489,1499,1511,1523)
a={**d,**c[0].axes}; b={**d,**c[1].axes}
assert (a["G"],a["batch"],a["iters"],a["proposal_temperature"])==(8,64,4,1.0)
assert (b["proposal_mixture"],b["proposal_prior_fraction"],b["proposal_temperature"])==("question_temperature",0.5,1.2)
PY
for cell in 0 1; do for seed in 0 1 2 3 4 5 6; do
  "$VENV/bin/python" "$SCRIPT_DIR/run_yaml.py" "$YAML" --dry-run --seed "$seed" --shard "$cell" --nshard 2 --expect-cells 1 >/dev/null
done; done

if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  printf 'preflight_ok=1\nexecution_commit=%s\nconfig_sha256=%s\n' "$expected_commit" "$config_sha"
  exit 0
fi
mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"
payload_output=$(qsub -h -v "PROJ=$PROJ,EXPECTED_COMMIT=$expected_commit,EXPECTED_CONFIG_SHA256=$config_sha,YAML=$YAML" run_qwen3_17b_pis_temperature_posterity_ucl.sh)
payload_job=$(printf '%s\n' "$payload_output" | sed -nE 's/.*job-array ([0-9]+).*/\1/p')
test -n "$payload_job" || { echo "ERROR: cannot parse payload: $payload_output" >&2; exit 3; }
set +e
validator_output=$(qsub -hold_jid "$payload_job" -v "PROJ=$PROJ,PAYLOAD_PROJ=$PROJ,SOURCE_JOB_ID=$payload_job,VALIDATOR_COMMIT=$expected_commit,EXECUTION_COMMIT=$expected_commit,EXPECTED_CONFIG_SHA256=$config_sha,YAML=$YAML,MARKER=$MARKER,RUN_ID=f3950b2e,LOG_STEM=qwen3_17b_pis_temperature_posterity" validate_qwen3_17b_l2r_focused_followup_ucl.sh 2>&1)
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
printf 'execution_commit=%s\nrun_id=f3950b2e\npayload_job=%s\nvalidator_job=%s\nconfig_sha256=%s\nresult_out=%s\nmarker=%s\nstate=user_held_pending_release\n' "$expected_commit" "$payload_job" "$validator_job" "$config_sha" "$RESULT_OUT" "$MARKER"
