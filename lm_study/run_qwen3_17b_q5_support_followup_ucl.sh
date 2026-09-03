#!/bin/bash -l
#$ -N vrl_q3_qsf
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 1
#$ -l h_rt=24:00:00
#$ -l tmem=24G
#$ -t 1-14
#$ -cwd -j y -o logs/qwen3_q5_support_followup.$JOB_ID.$TASK_ID.log

set -euo pipefail

export VRL_ROOT=${VRL_ROOT:-$HOME/vrl_hpc}
export PROJ=${PROJ:-$VRL_ROOT/vrl}
export VENV=${VENV:-$VRL_ROOT/po_venv}
export HF_HOME=${HF_HOME:-$VRL_ROOT/hf_cache}
export YAML=${YAML:-$PROJ/lm_study/experiments_qwen3_17b_q5_support_followup.yaml}
export PYROOT=${PYROOT:-/share/apps/python-3.11.9-shared}
export PYTHON_SOURCE=${PYTHON_SOURCE:-/share/apps/source_files/python/python-3.11.9.source}
export OPENSSL_ROOT=${OPENSSL_ROOT:-/share/apps/openssl-3.0.13}
export LIBFFI_ROOT=${LIBFFI_ROOT:-/share/apps/libffi-3.4.6}
export CUDA_SOURCE=${CUDA_SOURCE:-/share/apps/source_files/cuda/cuda-11.8.source}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

for required in EXPECTED_COMMIT EXPECTED_CONFIG_SHA256; do
  test -n "${!required:-}" || { echo "ERROR: $required missing" >&2; exit 2; }
done
actual_commit=$(cd "$PROJ" && git rev-parse HEAD)
test "$actual_commit" = "$EXPECTED_COMMIT" || { echo "ERROR: checkout drift" >&2; exit 2; }
test -z "$(cd "$PROJ" && git status --porcelain --untracked-files=no)" || {
  echo "ERROR: tracked cluster worktree is dirty" >&2
  exit 2
}

if [ -f "$PYTHON_SOURCE" ]; then
  set +u
  source "$PYTHON_SOURCE"
  set -u
fi
if [ -x "$PYROOT/bin/python3" ]; then
  export PATH="$PYROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$PYROOT/lib:${LD_LIBRARY_PATH:-}"
  for runtime_lib in "$OPENSSL_ROOT/lib64" "$OPENSSL_ROOT/lib" "$LIBFFI_ROOT/lib64" "$LIBFFI_ROOT/lib"; do
    test ! -d "$runtime_lib" || export LD_LIBRARY_PATH="$runtime_lib:$LD_LIBRARY_PATH"
  done
else
  module load python/3.8.5 || true
fi
test ! -f "$CUDA_SOURCE" || source "$CUDA_SOURCE"

cd "$PROJ/lm_study"
source "$VENV/bin/activate"
actual_config_sha256=$("$VENV/bin/python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$YAML")
test "$actual_config_sha256" = "$EXPECTED_CONFIG_SHA256" || { echo "ERROR: configuration drift" >&2; exit 2; }

task_id=${SGE_TASK_ID:-0}
cell_count=2
seed_count=7
test "$task_id" -ge 1 -a "$task_id" -le 14 || exit 4
zero_index=$((task_id - 1))
cell_index=$((zero_index / seed_count))
seed_index=$((zero_index % seed_count))

echo "== Q5 support follow-up | cell $((cell_index + 1))/$cell_count | seed $((seed_index + 1))/$seed_count | $(hostname) | $(date) =="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
"$VENV/bin/python" run_yaml.py "$YAML" \
  --seed "$seed_index" \
  --shard "$cell_index" \
  --nshard "$cell_count" \
  --expect-cells 1
