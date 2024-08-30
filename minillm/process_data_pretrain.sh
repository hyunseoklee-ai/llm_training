BASE_PATH=${1-"$(pwd)"}

MAX_LENGTH=4096

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir /opt/dlami/nvme/data/english/ \
    --processed-data-dir /opt/dlami/nvme/data_llama/english/ \
    --model-path meta-llama/Meta-Llama-3.1-8B \
    --max-length ${MAX_LENGTH} \
    --train-num 2000000000 \
    --data-process-workers 16 \
    --dev-num 10 \