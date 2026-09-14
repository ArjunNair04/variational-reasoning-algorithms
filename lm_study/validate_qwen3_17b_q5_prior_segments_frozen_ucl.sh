#!/bin/bash -l
#$ -N vrl_q5_pfv
#$ -l h_rt=04:00:00
#$ -l tmem=8G
#$ -cwd -j y -o logs/qwen3_q5_prior_segments_frozen_validate.$JOB_ID.log
set -euo pipefail
: "${PROJ:?immutable execution checkout required}"
: "${GENERATOR:?frozen-reader generator required}"
: "${LOG_PREFIX:?frozen-reader log prefix required}"
exec bash "$PROJ/lm_study/validate_qwen3_17b_q5_prior_segments_ucl.sh"
