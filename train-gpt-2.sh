LOCAL_BATCH_SIZE=64
GRAD_ACC=8
NPROC=1

TAG=your-tag
PORT=13222

nohup python dp_main.py > /inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/nohupout/$TAG.log \
    --chkpt-dir /your/checkpoint/path \
    --dataset-path ./dataset \
    --log-dir /your/log/path \
    --tokenizer-path /your/tokenizer/path \
    --device 5 \
    --tag $TAG \
    --reg-lambda 0 \
    --layers 12 \
    --embed-dim 768 \
    --max-epochs 4 \
    --heads 12 \
    --lr-warmup-steps 50 \
    --grad-clipping 2.0 \
    --win-size 256 \
    --forward-svd-warmup-steps 0 \
    --forward-svd-merge-steps -1 \
    --batch-size $LOCAL_BATCH_SIZE \
    --lr 1e-4 \
    --merged-lr 1e-4 \
    --grad-acc $GRAD_ACC \
    --train-steps 400000 \
    --q-forward-input nvfp4e2m1bnosr \
    --q-forward-weight nvfp4e2m1bnosr \
    --q-backward-input nvfp4e2m1bnosr \
    --q-backward-weight nvfp4e2m1bnosr \
    --q-backward-outputgrad nvfp4e2m1b \
    --enable-lowbit \
    --save-steps 10000 \
    --enable-forward-svd-intime \
    --forward-lowrank-svd-intime 16 \
    --enable-backward-svd \
    --backward-lowrank-svd 16 \
    --backward-lowrank-niter 2 \
    --backward-broadcast-dim -1 \
