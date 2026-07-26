uv sync
source .venv/bin/activate

RAW="TwXoDncyxhVSoTFklwVXpsPaXzVipMJavD"
export HF_TOKEN="hf_${RAW}"
hf auth login --token "hf_${RAW}"

# hf download VoCuc/anti-data \
#   --repo-type dataset \
#   --local-dir .
# unzip -o anti_data.zip


bash ./pipeline_gsm8k_0.sh &
bash ./pipeline_gsm8k_2.sh &

wait

bash ./collect_results.sh