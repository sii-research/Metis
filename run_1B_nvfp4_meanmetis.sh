
LOCAL_BATCH_SIZE=8
GRAD_ACC=8

# total batchsize = LOCAL_BATCH_SIZE * GRAD_ACC * NPROC

TAG=1B-nvfp4-metis-mean-mean
PORT=13200
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# # export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch --nproc_per_node 8 --master-port $PORT dp_main.py > /inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/nohupout_new/$TAG.log \
    --chkpt-dir /inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/ckpt0224 \
    --dataset-path /inspire/hdd/global_user/p-shangli/DCLM/ \
    --log-dir /inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/logs0224 \
    --tokenizer-path /inspire/hdd/global_user/p-shangli/tokenizers/r50k_base.tiktoken \
    --tag $TAG \
    --reg-lambda 0 \
    --layers 32 \
    --embed-dim 1024 \
    --max-epochs 4 \
    --heads 32 \
    --lr-warmup-steps 40 \
    --grad-clipping 8 \
    --win-size 1024 \
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
    --activation-metis-mode mean \
    --gout-metis-mode mean \
    --enable-forward-svd \
    --forward-svd-rank 16 \
    --enable-activation-svd \
    --activation-lowrank-svd 16 \
    --activation-lowrank-niter 2 \
    --activation-broadcast-dim -1 \
    --enable-backward-svd \
    --backward-lowrank-svd 16 \
    --backward-lowrank-niter 2 \
    --backward-broadcast-dim -1 \
    # --use-power-iter
    
    
    
    






    
    
    
    
