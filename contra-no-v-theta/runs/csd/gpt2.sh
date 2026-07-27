BASE_PATH=${1-"."}
export HF_HOME="/mnt/data/huggingface"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
DEVICE=0 # set gpu id here, 1 gpu only

CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/gpt2/csd/csd_base.sh ${BASE_PATH} ${MASTER_PORT} 1

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/gpt2/eval/eval_main_${benchmark}.sh ${BASE_PATH} ${MASTER_PORT} 1 csd/0.1B_1.5B --seed $seed --eval-batch-size 64
    done
done


CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/gpt2/csd/contra_base.sh ${BASE_PATH} ${MASTER_PORT} 1

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/gpt2/eval/eval_main_${benchmark}.sh ${BASE_PATH} ${MASTER_PORT} 1 contra/csd/0.1B_1.5B --seed $seed --eval-batch-size 64
    done
done

