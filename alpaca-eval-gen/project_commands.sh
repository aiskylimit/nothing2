#!/usr/bin/env bash
set -euo pipefail

BASE_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MASTER_PORT=${MASTER_PORT:-$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")}
GPUS_PER_NODE=1
CUDA_VISIBLE_DEVICES=4
HF_HOME=${HF_HOME:-"${BASE_PATH}/.cache/huggingface"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"no-v-theta"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.10"}

export BASE_PATH
export CUDA_VISIBLE_DEVICES
export HF_HOME
export HF_HUB_DISABLE_TELEMETRY=1
export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="${BASE_PATH}"

conda_base=$(conda info --base)
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"
if conda tos --help >/dev/null 2>&1; then
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
fi
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${CONDA_ENV_NAME}"
pip install alpaca-eval

mkdir -p logs


if [[ ! -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  conda install -y -c nvidia -c conda-forge cuda-toolkit=13.0
fi
export CUDA_HOME="${CONDA_PREFIX}"


# Each entry: "model_path|peft_path"
# Leave peft_path empty for checkpoints without a LoRA adapter.
CHECKPOINTS=(
  "bachthetrollface/gpt2-1.5B-teacher-dolly"
  "HoangTran223/MCW_KD_GPT2_SFT-1"
  "bachthetrollface/gpt2-120M-distillm-dolly"
  "bachthetrollface/gpt2-120M-contra-distillm-dolly"
  "bachthetrollface/gpt2-120M-fdd-dolly"
  "bachthetrollface/gpt2-120M-contra-fdd-dolly"
  "bachthetrollface/gpt2-120M-distillm2-dolly"
  "bachthetrollface/gpt2-120M-contra-distillm2-dolly"
  "bachthetrollface/gpt2-120M-csd-dolly"
  "bachthetrollface/gpt2-120M-contra-csd-dolly"
  "openlm-research/open_llama_7b_v2|bachthetrollface/openllama2-7B-teacher-dolly-lora"
  "openlm-research/open_llama_3b_v2|bachthetrollface/openllama2-3B-distillm-dolly-lora"
  "openlm-research/open_llama_3b_v2|bachthetrollface/openllama2-3B-contra-distillm-dolly-lora"
  "openlm-research/open_llama_3b_v2|bachthetrollface/openllama2-3B-fdd-dolly-lora"
  "openlm-research/open_llama_3b_v2|bachthetrollface/openllama2-3B-contra-fdd-dolly-lora"
  "openlm-research/open_llama_3b_v2|bachthetrollface/openllama2-3B-distillm2-dolly-lora"
  "openlm-research/open_llama_3b_v2|bachthetrollface/openllama2-3B-contra-distillm2-dolly-lora"
  "openlm-research/open_llama_3b_v2|bachthetrollface/openllama2-3B-csd-dolly-lora"
  "openlm-research/open_llama_3b_v2|bachthetrollface/openllama2-3B-contra-csd-dolly-lora"
)

for ENTRY in "${CHECKPOINTS[@]}"; do
  IFS='|' read -r MODEL_PATH PEFT_PATH <<< "$ENTRY"

#   echo "=============================="
#   echo "Generating for: $GEN_NAME"
#   echo "  model: $MODEL_PATH"
#   echo "  peft:  ${PEFT_PATH:-<none>}"
#   echo "=============================="

  if [ -n "$PEFT_PATH" ]; then
    python generate.py \
      --model_name "$MODEL_PATH" \
      --peft "$PEFT_PATH" \
      2>&1 | tee "logs/${PEFT_PATH}.log"
  else
    python generate.py \
      --model_name "$MODEL_PATH" \
      2>&1 | tee "logs/${MODEL_PATH}.log"
  fi
done

echo "All checkpoints done. Outputs in ./outputs/"