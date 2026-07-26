mkdir -p experiments_gpqa

hf download VoCuc/anti-sampling-ckpt \
    --include "experiments_mmlu_1/models/**" \
    --local-dir experiments_gpqa


echo ">>> [1/4] Eval student_tau1.00e+00_lam3.95e-02_eps1.00e-02 ..."
accelerate launch --config_file acc_config_4.yaml --main_process_port 0 gentraces.py  \
    hydra.run.dir=experiments_gpqa/metadata/eval/student_mmlu \
    teacher=experiments_gpqa/experiments_mmlu_1/models/student_tau1.00e+00_lam3.95e-02_eps1.00e-02/final \
    is_teacher=false exp_dir=experiments_gpqa answer_force=true \
    data_split=gpqa_test batch_size=256 max_samples=2880 \
    trace_name=eval_student_tau1.00e+00_lam3.95e-02_eps1.00e-02 \
    seed=42 max_length=1024 max_prompt_length=512 &

echo ">>> [2/4] Eval student_tau9.00e-01_lam1.58e-02_eps1.00e-02 ..."
accelerate launch --config_file acc_config_6.yaml --main_process_port 0 gentraces.py  \
    hydra.run.dir=experiments_gpqa/metadata/eval/student_mmlu \
    teacher=experiments_gpqa/experiments_mmlu_1/models/student_tau9.00e-01_lam1.58e-02_eps1.00e-02/final \
    is_teacher=false exp_dir=experiments_gpqa answer_force=true \
    data_split=gpqa_test batch_size=256 max_samples=2880 \
    trace_name=eval_student_tau9.00e-01_lam1.58e-02_eps1.00e-02 \
    seed=42 max_length=1024 max_prompt_length=512 &

wait

echo ">>> [3/4] Eval student_tau9.00e-01_lam2.37e-02_eps1.00e-02 ..."
accelerate launch --config_file acc_config_4.yaml --main_process_port 0 gentraces.py  \
    hydra.run.dir=experiments_gpqa/metadata/eval/student_mmlu \
    teacher=experiments_gpqa/experiments_mmlu_1/models/student_tau9.00e-01_lam2.37e-02_eps1.00e-02/final \
    is_teacher=false exp_dir=experiments_gpqa answer_force=true \
    data_split=gpqa_test batch_size=256 max_samples=2880 \
    trace_name=eval_student_tau9.00e-01_lam2.37e-02_eps1.00e-02 \
    seed=42 max_length=1024 max_prompt_length=512 &

echo ">>> [4/4] Eval student_tau9.00e-01_lam3.16e-02_eps1.00e-02 ..."
accelerate launch --config_file acc_config_6.yaml --main_process_port 0 gentraces.py  \
    hydra.run.dir=experiments_gpqa/metadata/eval/student_mmlu \
    teacher=experiments_gpqa/experiments_mmlu_1/models/student_tau9.00e-01_lam3.16e-02_eps1.00e-02/final \
    is_teacher=false exp_dir=experiments_gpqa answer_force=true \
    data_split=gpqa_test batch_size=256 max_samples=2880 \
    trace_name=eval_student_tau9.00e-01_lam3.16e-02_eps1.00e-02 \
    seed=42 max_length=1024 max_prompt_length=512 &

wait

echo ">>> Eval Teacher ..."
bash ./pipeline_gpqa_4.sh


echo "🎉 DONE!"