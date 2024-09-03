BASE_PATH=${1-"$(pwd)"}

MAX_LENGTH=8192

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir /home/ubuntu/alinllm/data/english \
    --processed-data-dir ${BASE_PATH}/processed_data/newnewnew/english/${MAX_LENGTH} \
    --model-path meta-llama/Meta-Llama-3.1-8B \
    --max-length ${MAX_LENGTH} \
    --train-num 10000000000 \
    --data-process-workers 128 \
    --dev-num 1000 \