#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT "

# " --deepspeed_config ${BASE_PATH}/configs/deepspeed/deepspeed_zero3.yaml"

# model
BASE_PATH="/opt/dlami/nvme/llm_training/minillm"
# CKPT_NAME="gemma"
# CKPT="${BASE_PATH}/test/${CKPT_NAME}/"
CKPT_NAME="meta-llama/Meta-Llama-3.1-8B"
CKPT="${CKPT_NAME}"
TEACHER_CKPT_NAME="gemma"
# google/gemma-2-2B
TEACHER_CKPT="${BASE_PATH}/test/${CKPT_NAME}/"
TEACHER_CKPT_NAME="meta-llama/Meta-Llama-3.1-8B"
TEACHER_CKPT="${TEACHER_CKPT_NAME}"
TEACHER_PEFT_CKPT_NAME="lora-2B"
TEACHER_PEFT_CKPT="${BASE_PATH}/results/llama/train/${TEACHER_PEFT_CKPT_NAME}/"
MP_SIZE=2
# data
DATA_DIR="/opt/dlami/nvme/data_llama/korean/2B/"
ENG_DATA_DIR="/opt/dlami/nvme/data_llama/english/2B/"
# hp
BATCH_SIZE=1
LR=0.00001
GRAD_ACC=4
EVAL_BATCH_SIZE=1
# length
MAX_LENGTH=4096
# runtime
SAVE_PATH="${BASE_PATH}/results/kd_llama_flash"
# seed
SEED=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
# OPTS+=" --teacher-model-w16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type llama"
OPTS+=" --gradient-checkpointing"
# OPTS+=" --model-parallel"
# OPTS+=" --model-parallel-size ${MP_SIZE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --eng-data-dir ${ENG_DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 100"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 100"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 1"
OPTS+=" --kd-ratio 0.5"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
# OPTS+=" --eval-gen"
OPTS+=" --save-interval 200"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 1"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# lora
# OPTS+=" --peft lora"
# OPTS+=" --teacher-peft-name ${TEACHER_PEFT_CKPT_NAME}"
# OPTS+=" --peft_lora_r 16"
# OPTS+=" --peft_lora_alpha 32"
# OPTS+=" --peft_lora_dropout 0.05"
# OPTS+=" --teacher-peft-path ${TEACHER_PEFT_CKPT}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
# OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero3.json"

# type
OPTS+=" --type kd"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

# export WANDB_MODE=online
export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
# export PYTHONPATH=${BASE_PATH}
# export CUDA_VISIBLE_DEVICES=4,5
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"
# CMD="python3 -m torch.distributed.run ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}

    # // "fp16": {
    # //     "enabled": "auto",
    # //     "loss_scale": 0,
    # //     "loss_scale_window": 1000,
    # //     "initial_scale_power": 16,
    # //     "hysteresis": 2,
    # //     "min_loss_scale": 1
    # // },