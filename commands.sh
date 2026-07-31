# 1 +120
#sql
#v1

#2 -f-/home/ubuntu/aiskylimit_nothing2/anti-sampling/results_yaml/ +a

# sudo reboot

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt-get install -y cuda-toolkit-13-0
# sudo apt install -y zip unzip
# echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# source ~/.bashrc
# bash install_miniconda.sh

# cd gpu_burn
# make CUDAPATH=/usr/local/cuda-13.0
# ./gpu_burn 36000000000


kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
sleep 5
nvidia-smi

# sudo systemctl stop nvidia-fabricmanager
# sudo apt purge -y nvidia-fabricmanager
# sudo apt install -y nvidia-fabricmanager=595.71.05-1ubuntu1

# sudo apt-mark hold nvidia-fabricmanager

# sudo systemctl daemon-reload
# sudo systemctl enable nvidia-fabricmanager
# sudo systemctl start nvidia-fabricmanager

# sudo reboot

# systemctl status nvidia-fabricmanager
# /usr/bin/nv-fabricmanager --version

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# cd anti-sampling
# ls experiments_gsm8k_seed62 -R
# rm -rf results_yaml
# bash ./collect_results.sh
# rm -rf as_gema_experiments_gsm8k experiments_gsm8k_seed62
# bash ./project_commands.sh


# cd contra-no-v-theta
# zip -r results_gpt2_eval_main.zip results/gpt2/eval_main
# du -sh results_gpt2_eval_main.zip
# bash ./project_commands.sh

# kill -9 1947988
# cd contra-mse-hsm
# bash ./project_commands.sh

# cd contra-delta-matching
# # zip -r results_gpt2_eval_main.zip results/gpt2/eval_main
# # du -sh results_gpt2_eval_main.zip
# bash ./project_commands.sh


# cd contra-fdd-curriculum
# bash ./project_commands.sh

# cd contra-only
# bash ./project_commands.sh

# cd contra-velocity-field-design
# bash ./project_commands.sh

# cd contra-analysis
# bash ./project_commands.sh

# cd regkd-vlm2vec
# bash ./project_commands.sh

# cd ./text2sql_distillation_draft

QWEN25_GPU_LIST=0,1,2,3,4,5,6,7 GPUS_PER_JOB=2 RUN_MODE=parallel \
BATCH_SIZE=4 GRAD_ACC=4 bash project_commands_qwen2.5.sh
