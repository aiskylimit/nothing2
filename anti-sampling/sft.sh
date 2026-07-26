# accelerate launch --config_file acc_config_0.yaml --main_process_port 0 sft.py \
#     hydra.run.dir=experiments_sft/metadata/sft/gsm8k \
#     student=meta-llama/Llama-3.2-3B \
#     tokenizer=meta-llama/Llama-3.2-3B-Instruct \
#     exp_dir=experiments_sft \
#     train_traces=gsm8k_train \
#     holdout_traces=traces_holdout \
#     model_name=student_gsm8k_sft max_length=1025

# accelerate launch --config_file acc_config_0.yaml --main_process_port 0 gentraces.py  \
#     hydra.run.dir=experiments_sft/metadata/eval/gsm8k \
#     teacher=experiments_sft/models/student_gsm8k_sft/final \
#     is_teacher=false exp_dir=experiments_sft answer_force=true \
#     data_split=gsm8k_test batch_size=192 max_samples=2880 \
#     trace_name=eval_student_gsm8k_sft seed=42

# accelerate launch --config_file acc_config_0.yaml --main_process_port 0 sft.py \
#     hydra.run.dir=experiments_sft/metadata/sft/hendrycks_math \
#     student=meta-llama/Llama-3.2-3B \
#     tokenizer=meta-llama/Llama-3.2-3B-Instruct \
#     exp_dir=experiments_sft \
#     train_traces=hendrycks_math_train \
#     holdout_traces=traces_holdout \
#     model_name=student_hendrycks_math_sft max_length=1025 max_samples=2880

accelerate launch --config_file acc_config_0.yaml --main_process_port 0 gentraces.py  \
    hydra.run.dir=experiments_sft/metadata/eval/hendrycks_math \
    teacher=experiments_sft/models/student_hendrycks_math_sft/final \
    is_teacher=false exp_dir=experiments_sft answer_force=true \
    data_split=hendrycks_math_test batch_size=384 max_samples=2880 \
    trace_name=eval_student_hendrycks_math_sft seed=42 max_prompt_length=256

# accelerate launch --config_file acc_config_0.yaml --main_process_port 0 sft.py \
#     hydra.run.dir=experiments_sft/metadata/sft/mmlu \
#     student=meta-llama/Llama-3.2-3B \
#     tokenizer=meta-llama/Llama-3.2-3B-Instruct \
#     exp_dir=experiments_sft \
#     train_traces=mmlu_train \
#     holdout_traces=traces_holdout \
#     model_name=student_mmlu_sft max_length=1025 max_samples=2880

accelerate launch --config_file acc_config_0.yaml --main_process_port 0 gentraces.py  \
    hydra.run.dir=experiments_sft/metadata/eval/mmlu \
    teacher=experiments_sft/models/student_mmlu_sft/final \
    is_teacher=false exp_dir=experiments_sft answer_force=true \
    data_split=mmlu_test batch_size=384 max_samples=2880 \
    trace_name=eval_student_mmlu_sft seed=42 max_prompt_length=256