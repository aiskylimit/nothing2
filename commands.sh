#1 +120+a
#contra
#v1

#2 -f-/home/ubuntu/aiskylimit_nothing2/anti-sampling/results_yaml/ +a


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

# kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
# sleep 5
nvidia-smi

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

# cd alpaca-eval-gen
# bash ./project_commands.sh

# cd contra-fdd-curriculum
# bash ./project_commands.sh

cd contra-only
bash ./project_commands.sh

# cd contra-velocity-field-design
# bash ./project_commands.sh
