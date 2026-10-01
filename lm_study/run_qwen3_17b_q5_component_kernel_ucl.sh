#!/bin/bash -l
#$ -N vrl_q5_ker
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 1
#$ -l h_rt=24:00:00
#$ -l tmem=24G
#$ -t 1-3
#$ -cwd -j y -o logs/qwen3_q5_kernel.$JOB_ID.$TASK_ID.log

set -euo pipefail
: "${PROJ:?immutable checkout required}"
: "${EXPECTED_COMMIT:?execution commit required}"
: "${EXPECTED_CONFIG_SHA256:?configuration hash required}"
export VENV=${VENV:-$HOME/vrl_hpc/po_venv}
export HF_HOME=${HF_HOME:-$HOME/vrl_hpc/hf_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PROJ/src:$PROJ/lm_study:$PROJ/analysis:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
YAML="$PROJ/lm_study/experiments_qwen3_17b_q5_component_kernel.yaml"
source "$PROJ/lm_study/ucl_python_env.sh"
CUDA_SOURCE=${CUDA_SOURCE:-/share/apps/source_files/cuda/cuda-11.8.source}
if [ -f "$CUDA_SOURCE" ]; then
  set +u
  source "$CUDA_SOURCE"
  set -u
fi
test "$(cd "$PROJ" && git rev-parse HEAD)" = "$EXPECTED_COMMIT" || exit 2
test -z "$(cd "$PROJ" && git status --porcelain --untracked-files=no)" || exit 2
actual_sha=$("$VENV/bin/python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$YAML")
test "$actual_sha" = "$EXPECTED_CONFIG_SHA256" || exit 2
task_id=${SGE_TASK_ID:-0}
test "$task_id" -ge 1 -a "$task_id" -le 3 || exit 4
cd "$PROJ/lm_study"
echo "== Q5 component kernel | task $task_id | $(hostname) | $(date) =="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
"$VENV/bin/python" run_yaml.py "$YAML" --seed "$((task_id - 1))" --expect-cells 1
