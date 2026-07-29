#!/usr/bin/env bash
set -euo pipefail

BASE_PATH=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${BASE_PATH}/results/mechanism/gpt2-0.1B-1.5B-distillm"}
NUM_EXAMPLES=${NUM_EXAMPLES:-500}
BATCH_SIZE=${BATCH_SIZE:-1}
DEVICE=${DEVICE:-auto}
DTYPE=${DTYPE:-auto}

TEACHER=${TEACHER:-"bachthetrollface/gpt2-1.5B-teacher-dolly"}
DISTILLM=${DISTILLM:-"bachthetrollface/gpt2-120M-distillm-dolly"}
CONTRA=${CONTRA:-"bachthetrollface/gpt2-120M-contra-distillm-dolly"}
VELOCITY=${VELOCITY:-"bachthetrollface/velocity-field-gpt2"}
DATA_PATH=${DATA_PATH:-"hf://dvtiendat/contra-data/dolly/valid.jsonl"}

COMMON_ARGS=(
  --teacher-path "${TEACHER}"
  --velocity-source "${VELOCITY}"
  --data-path "${DATA_PATH}"
  --output-dir "${OUTPUT_ROOT}"
  --num-examples "${NUM_EXAMPLES}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
)

python3 "${BASE_PATH}/tools/evaluate_mechanism.py" \
  "${COMMON_ARGS[@]}" \
  --student-label contra-distillm \
  --student-path "${CONTRA}" \
  --baseline-label distillm \
  --baseline-path "${DISTILLM}"
