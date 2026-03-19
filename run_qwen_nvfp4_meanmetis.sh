export NCCL_DEBUG=INFO
export NCCL_ALGO=Ring             
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

TAG="Qwen0.6B-nvfp4-metis-mean-mean-fix"

NPROC=8

MASTER_ADDR=localhost
PORT=13300

CONFIG_DIR="/inspire/hdd/project/yunweiyuhuifu/p-shangli/Qwen3-0.6B"
CKPT_DIR="/inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/ckpt0224/$TAG"
DATA_DIR="/inspire/hdd/global_user/p-shangli/dclm-muon"
LOG_DIR="/inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/logs0224/$TAG"
LOAD_DIR="/inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/ckpt0224/Qwen0.6B-nvfp4-metis-mean-mean/checkpoint_45000.pth"

NUM_WORKERS=8

LR=1e-4
LOCAL_BATCH_SIZE=16
GLOBAL_BATCH_SIZE_WARMUP=256
GLOBAL_BATCH_SIZE_TRAIN=512
BATCH_WARMUP_STEP=2000
# TRAIN_STEP=97098
TRAIN_STEP=191735
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
ARGS+=" --reg-lambda 0"
ARGS+=" --forward-svd-warmup-steps 0"
ARGS+=" --forward-svd-merge-steps -1"
ARGS+=" --q-forward-input nvfp4e2m1bnosr"
ARGS+=" --q-forward-weight nvfp4e2m1bnosr"
ARGS+=" --q-backward-input nvfp4e2m1bnosr"
ARGS+=" --q-backward-weight nvfp4e2m1bnosr"
ARGS+=" --q-backward-outputgrad nvfp4e2m1b"
ARGS+=" --enable-lowbit"
ARGS+=" --activation-metis-mode mean"
ARGS+=" --gout-metis-mode mean"
ARGS+=" --enable-forward-svd"
ARGS+=" --forward-svd-rank 16"
ARGS+=" --enable-activation-svd"
ARGS+=" --activation-lowrank-svd 16"
ARGS+=" --activation-lowrank-niter 2"
ARGS+=" --activation-broadcast-dim -1"
ARGS+=" --enable-backward-svd"
ARGS+=" --backward-lowrank-svd 16"
ARGS+=" --backward-lowrank-niter 2"
ARGS+=" --backward-broadcast-dim -1"
# ARGS+=" --load-from ${LOAD_DIR}"


WORK_DIR="/inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4"
echo "Launched training with TAG=${TAG}, PORT=${PORT}. Logs -> ${WORK_DIR}/nohupout/${TAG}.log"
nohup torchrun \
    --nproc_per_node=${NPROC} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} dp_pretrain_qwen3.py ${ARGS} \
    > /inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/nohupout/${TAG}.log 2>&1