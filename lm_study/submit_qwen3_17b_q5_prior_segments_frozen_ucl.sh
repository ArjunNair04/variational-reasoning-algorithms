#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export YAML="$SCRIPT_DIR/experiments_qwen3_17b_q5_prior_segments_frozen.yaml"
export GENERATOR="$SCRIPT_DIR/generate_qwen3_17b_q5_prior_segments_frozen.py"
export PAYLOAD_RUNNER=run_qwen3_17b_q5_prior_segments_frozen_ucl.sh
export VALIDATOR_RUNNER=validate_qwen3_17b_q5_prior_segments_frozen_ucl.sh
export LOG_PREFIX=qwen3_q5_prior_segments_frozen
export RESULT_OUT="$HOME/po_results/2026-09-15/q5-prior-segments-frozen/qwen3-q5-prior-segments-frozen__7a31c9e2"
export MARKER="$HOME/po_results/auto_state/qwen3_q5_prior_segments_frozen_7a31c9e2.ok"
export CLAIM_DIR="$HOME/po_results/auto_state/qwen3_q5_prior_segments_frozen_7a31c9e2.submit.claim"
exec bash "$SCRIPT_DIR/submit_qwen3_17b_q5_prior_segments_ucl.sh"
