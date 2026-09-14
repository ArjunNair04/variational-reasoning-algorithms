#!/bin/bash -l
#$ -N vrl_q5_psf
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 1
#$ -l h_rt=24:00:00
#$ -l tmem=24G
#$ -t 1-9
#$ -cwd -j y -o logs/qwen3_q5_prior_segments_frozen.$JOB_ID.$TASK_ID.log
set -euo pipefail
: "${PROJ:?immutable execution checkout required}"
: "${YAML:?frozen-reader configuration required}"
exec bash "$PROJ/lm_study/run_qwen3_17b_q5_prior_segments_ucl.sh"
