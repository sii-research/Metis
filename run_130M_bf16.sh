
LOCAL_BATCH_SIZE=64
GRAD_ACC=8
NPROC=1
# total batchsize = LOCAL_BATCH_SIZE * GRAD_ACC * NPROC

TAG=130M-bf16
PORT=13200
export CUDA_VISIBLE_DEVICES=0
# # export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch --nproc_per_node $NPROC --master-port $PORT dp_main.py > /inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/nohupout/$TAG.log \
    --chkpt-dir /inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4/checkpoint \
    --dataset-path /inspire/hdd/global_user/p-shangli/DCLM-cleaned/ \
    --log-dir /inspire/hdd/project/yunweiyuhuifu/p-shangli/cmy/Metis-Hif4//log \
    --tokenizer-path /inspire/hdd/global_user/p-shangli/tokenizers/r50k_base.tiktoken \
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
    --save-steps 10000 \
    
    
    
    






    
    
    
    
