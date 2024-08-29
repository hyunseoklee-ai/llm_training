BASE_PATH=${1-"$(pwd)"}

MAX_LENGTH=8192

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/removed/textbook1 \
    --processed-data-dir ${BASE_PATH}/processed_data/google/gemma-2-2b/${MAX_LENGTH}/korean/ \
    --model-path google/gemma-2-2B \
    --max-length ${MAX_LENGTH} \
    --train-num 2000000 \
    --data-process-workers 32 \
    --dev-num 100 \