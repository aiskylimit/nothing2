export TF_CPP_MIN_LOG_LEVEL=3


PYTHONPATH=. python ./distillm/process_data.py \
    --processed-data-dir ./processed_data/MetaMathQA-50k \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --data-process-workers 8 \
    --max-prompt-length 256 \
    --model-type qwen