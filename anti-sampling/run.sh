# bash sft.sh
# bash pipeline_gsm8k_0.sh
# bash pipeline_mmlu_0.sh
# bash pipeline_math_0.sh
# bash sft.sh
# cd ./_distill
# bash script/train/train_teacher_lora_3B.sh
# bash script/train/train_teacher_lora_no_entropy.sh
# bash script/train/train_teacher_lora_no_fluency_topk.sh
# bash script/train/train_teacher_lora_no_fluency.sh
# bash script/eval/mmlu/run_eval_0.sh
# bash script/eval/gsm8k/run_eval_3B.sh
# bash script/eval/gsm8k/run_eval_no_entropy.sh
# bash script/eval/gsm8k/run_eval_no_fluency_topk.sh
# bash script/eval/gsm8k/run_eval_no_fluency.sh
# bash script/eval/mmlu/run_eval_0.sh
# bash script/train/train_teacher_lora_3B.sh
# bash script/eval/mmlu/run_eval_0.sh
# bash script/eval/gsm8k/run_eval_3B.sh
# bash script/eval/gsm8k/run_eval_no_entropy.sh
# bash script/eval/gsm8k/run_eval_no_fluency_topk.sh
# bash script/eval/gsm8k/run_eval_no_fluency.sh
# bash run.sh
# cd ..
bash pipeline_mmlu_0.sh
# bash pipeline_gsm8k_proxy_1.5B.sh
cd ./_distill
bash script/eval/gsm8k/run_eval_no_fluency.sh
bash script/eval/mmlu/run_eval_0.sh
bash script/eval/gsm8k/run_eval_no_topk.sh
bash script/eval/gsm8k/run_eval_no_fluency.sh
bash script/eval/gsm8k/run_eval_no_topk.sh
cd ..
bash pipeline_mmlu.sh
cd ./_distill
bash script/eval/gsm8k/run_eval_no_fluency_topk.sh
bash script/eval/gsm8k/run_eval_no_fluency.sh