export NCCL_ALGO=Ring               # 你已设置
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NRANK=1
TAG="8B_bf16_${NRANK}"

NPROC=4

MASTER_ADDR=10.247.211.104
PORT=13304

CONFIG_DIR="/inspire/hdd/project/yunweiyuhuifu/p-shangli/Qwen3-8B"
CKPT_DIR="/inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/ckpt0313/$TAG"
# DATA_DIR="/mnt/workspace/dclm/global-shard_01_of_10"
DATA_DIR="/inspire/hdd/global_user/p-shangli/dclm-muon"
LOG_DIR="/inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/logs0313/$TAG"

NUM_WORKERS=4

LR=1.5e-4
LOCAL_BATCH_SIZE=8
GLOBAL_BATCH_SIZE_WARMUP=128
GLOBAL_BATCH_SIZE_TRAIN=128
BATCH_WARMUP_STEP=200
TRAIN_STEP=400000
SAVE_INTERVAL=5000
SEQ_LEN=1024
OPT_NAME="adam"

# Construct CLI args to pass into the Python script
ARGS=""
ARGS+=" --local-batch-size ${LOCAL_BATCH_SIZE}"
ARGS+=" --global-batch-size-warmup ${GLOBAL_BATCH_SIZE_WARMUP}"
ARGS+=" --global-batch-size-train ${GLOBAL_BATCH_SIZE_TRAIN}"
ARGS+=" --batch-warmup-step ${BATCH_WARMUP_STEP}"
ARGS+=" --save-interval ${SAVE_INTERVAL}"
ARGS+=" --train-steps ${TRAIN_STEP}"
ARGS+=" --seq-len ${SEQ_LEN}"
ARGS+=" --num-workers ${NUM_WORKERS}"
ARGS+=" --config-dir ${CONFIG_DIR}"
ARGS+=" --ckpt-dir ${CKPT_DIR}"
ARGS+=" --data-dir ${DATA_DIR}"
ARGS+=" --log-dir ${LOG_DIR}"
ARGS+=" --lr ${LR}"
ARGS+=" --optimizer-name ${OPT_NAME}"
ARGS+=" --pp-size 2"
ARGS+=" --pp-chunks 8"
ARGS+=" --pp-checkpoint never"

WORK_DIR="/inspire/hdd/project/yunweiyuhuifu/p-shangli/Metis-quantization-muon/qwen3_train"
echo "Launched training with TAG=${TAG}, PORT=${PORT}. Logs -> ${WORK_DIR}/nohupout/${TAG}.log"
torchrun \
    --nproc_per_node=${NPROC} \
    --nnodes=2 \
    --node_rank=${NRANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} /inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/pp_pretrain_qwen3.py ${ARGS} \
    > /inspire/hdd/project/yunweiyuhuifu/p-shangli/Metis-quantization-muon/qwen3_train/nohupout/${TAG}.log 2>&1