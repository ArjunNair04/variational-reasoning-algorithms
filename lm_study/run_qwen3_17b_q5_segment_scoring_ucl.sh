#!/bin/bash -l
#$ -N vrl_q5_score
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 1
#$ -l h_rt=02:00:00
#$ -l tmem=24G
#$ -t 1-3
#$ -cwd -j y -o logs/qwen3_q5_segment_scoring.$JOB_ID.$TASK_ID.log

set -euo pipefail
for name in PROJ EXPECTED_COMMIT EXPECTED_MANIFEST_SHA256 RESULT_OUT; do
  test -n "${!name:-}" || { echo "ERROR: missing $name" >&2; exit 2; }
done
test "$(cd "$PROJ" && git rev-parse HEAD)" = "$EXPECTED_COMMIT" || exit 3
test -z "$(cd "$PROJ" && git status --porcelain --untracked-files=no)" || exit 3
source "$PROJ/lm_study/ucl_python_env.sh"
VENV=${VENV:-$HOME/vrl_hpc/po_venv}
export PYTHONPATH="$PROJ/src:$PROJ/lm_study:$PROJ/analysis:${PYTHONPATH:-}"
export HF_HOME="$HOME/vrl_hpc/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
MANIFEST="$PROJ/docs/experiments/qwen3_q5_prior_segments/scoring_manifest.json"
actual=$("$VENV/bin/python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$MANIFEST")
test "$actual" = "$EXPECTED_MANIFEST_SHA256" || exit 3
case "${SGE_TASK_ID:-0}" in
  1) seed=1201;;
  2) seed=1213;;
  3) seed=1217;;
  *) exit 4;;
esac
cd "$PROJ"
echo "Q5_SEGMENT_SCORING_START seed=$seed job=$JOB_ID task=$SGE_TASK_ID host=$(hostname) commit=$EXPECTED_COMMIT"
"$VENV/bin/python" analysis/score_q5_prior_segments.py \
  --source "$HOME/po_results/2026-08-29/reproducibility/qwen3-selected-method-posterity__68078ecc" \
  --manifest "$MANIFEST" --seed "$seed" --output "$RESULT_OUT/seed_$seed"
