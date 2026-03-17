import json
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

from utils import DCLMDataset, MyDistributedSampler
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup, AutoModelForCausalLM, Qwen3ForCausalLM
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import os
from Metis.bitlinear import BitLinear

def args_2_json(args, verbose=True):
    d = dir(args)
    dic = {}
    for p in d:
        if not p.startswith("__") and not callable(getattr(args, p)):
            dic[p] = getattr(args, p)
    if verbose:
        print("********  Training Args  ********")
        print(json.dumps(dic, indent=2))
        print("*********************************")
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(f"{args.log_dir}/args.{int(time.time())}.json", "w") as fd:
        json.dump(dic, fd, indent=2)

def get_available_cuda_device() -> int:
    max_devs = torch.cuda.device_count()
    for i in range(max_devs):
        try:
            mem = torch.cuda.mem_get_info(i)
        except:
            continue
        if mem[0] / mem[1] > 0.85:
            return i
    return -1

def build_parser():
    parser = argparse.ArgumentParser(description="Distributed pretraining launcher for Qwen3 student model")

    # batching and sequence
    parser.add_argument('--local-batch-size', type=int, default=4)
    parser.add_argument('--global-batch-size-warmup', type=int, default=256)
    parser.add_argument('--global-batch-size-train', type=int, default=512)
    parser.add_argument('--batch-warmup-step', type=int, default=2000)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--train-steps', type=int, default=10)
    # parser.add_argument('--heterogeneous-opt', type=bool, default=False)

    # data / training flags
    parser.add_argument('--data-shuffle', dest='data_shuffle', action='store_true')
    parser.set_defaults(data_shuffle=True)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument("--adam-beta1", type=int, default=0.9)
    parser.add_argument("--adam-beta2", type=int, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument('--optimizer-name', type=str, default='adam')
    # parser.add_argument("--orthM", type=bool, default=True)
    # parser.add_argument("--power-iters", type=int, default=1)
    # parser.add_argument("--k-ratio", type=float, default=0.015)
    parser.add_argument("--continue-steps", type=int, default=0)

    # IO / devices
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--config-dir', type=str, default='../../Qwen-family/Qwen3-0.6B')
    parser.add_argument('--ckpt-dir', type=str, default='../checkpoints/qwen/Mytrain-Qwen3-0.8B-2.8B')
    parser.add_argument('--data-dir', type=str, default='../../washed_data_0520/train')
    parser.add_argument('--log-dir', type=str, default='../logs/qwen/Mytrain-Qwen3-0.8B-2.8B')
    parser.add_argument('--load-from', type=str, default='')

    # metis transfer
    parser.add_argument("--reg-alpha1", type=float, default=1.0)
    parser.add_argument("--reg-alpha2", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=0.0)
    parser.add_argument("--reg-beta", type=int, default=2)
    parser.add_argument("--enable-nv-recipe", action="store_true")
    parser.add_argument("--enable-lowbit", action="store_true")
    parser.add_argument("--activation-metis-mode", type=str, default="svd")
    parser.add_argument("--gout-metis-mode", type=str, default="svd")
    parser.add_argument("--use-power-iter", action="store_true")
    parser.add_argument("--enable-forward-svd", action="store_true")
    parser.add_argument("--forward-svd-rank", type=int, default=-1)
    parser.add_argument("--forward-svd-warmup-steps", type=int, default=5000)
    parser.add_argument("--forward-svd-merge-steps", type=int, default=5000)
    parser.add_argument("--enable-backward-svd", action="store_true")
    parser.add_argument("--enable-activation-svd", action="store_true")
    parser.add_argument("--q-forward-input", type=str, default="fp4e2m1")
    parser.add_argument("--q-forward-weight", type=str, default="fp4e2m1")
    parser.add_argument("--q-backward-input", type=str, default="fp4e2m1")
    parser.add_argument("--q-backward-weight", type=str, default="fp4e2m1")
    parser.add_argument("--q-backward-outputgrad", type=str, default="fp4e2m1")
    parser.add_argument("--q-scalar", type=float, default=1.0)
    parser.add_argument("--enable-te", action="store_true")
    parser.add_argument("--merged-lr", type=float, default=2e-5)
    parser.add_argument("--backward-lowrank-svd", type=int, default=-1)
    parser.add_argument("--backward-lowrank-niter", type=int, default=0)
    parser.add_argument("--activation-lowrank-svd", type=int, default=-1)
    parser.add_argument("--activation-lowrank-niter", type=int, default=0)
    parser.add_argument("--enable-forward-svd-intime", action="store_true")
    parser.add_argument("--forward-lowrank-svd-intime", type=int, default=0)
    parser.add_argument("--backward-longtail-schedule", type=str, default="none")
    parser.add_argument("--activation-longtail-schedule", type=str, default="none")
    parser.add_argument("--backward-broadcast-dim", type=int, default=-1)
    parser.add_argument("--activation-broadcast-dim", type=int, default=-1)
    parser.add_argument("--gradacc-broadcast", action="store_true")
    parser.add_argument("--gradacc-broadcast-steps", type=int, default=1)
    return parser



def replace_linear_with_custom(model, args):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if "lm_head" in name:
                continue
            # 获取原线性层的属性
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            
            new_layer = BitLinear(in_features, out_features, bias=bias, args=args)
            setattr(model, name, new_layer)
        else:
            # 递归处理子模块
            replace_linear_with_custom(child, args)
            
@torch.no_grad()
def prepare_model(local_rank, world_size, device, args):
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code = True)

    model = Qwen3ForCausalLM(config).to(device)
    model.train()
    if local_rank == 0:
        config.to_json_file(f"{args.ckpt_dir}/config.json")
        
    
    if local_rank == 0:
        print(id(model.model.embed_tokens.weight) == id(model.lm_head.weight)) # 如果输出 True，说明是同一个东西
        print(dict(model.named_children()).keys())
        for name, param in model.named_parameters():
            print(name)
        print('-----------------------')
    replace_linear_with_custom(model, args)
    if local_rank == 0:
        for name, param in model.named_parameters():
            print(name)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    print(f'rank {local_rank} model ok, params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B/{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B') # 
    return model


def prepare_data(local_rank, world_size, args, resume_step):
    tokenizer = AutoTokenizer.from_pretrained(args.config_dir, trust_remote_code=True)
    dataset = DCLMDataset(args.data_dir, args.seq_len, tokenizer)

    sampler = MyDistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=args.data_shuffle, cont_step=resume_step, warmup_steps=args.batch_warmup_step, bs_warmup=args.global_batch_size_warmup, bs_normal=args.global_batch_size_train)
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, num_workers=args.num_workers, sampler=sampler)

    print(f"rank {local_rank} data ok. Data Length {len(dataset)}.")
    return dataloader

def get_optimizer(args, model):
    if args.optimizer_name == "adam":
        return optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            betas=(args.adam_beta1, args.adam_beta2), 
            eps=1e-8, 
            weight_decay=args.weight_decay
        )
    else:
        assert 0, "optimizer not supported"


def prepare_loss_optimizer(local_rank, model, args):
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=151643, reduction='mean')
    optimizer = get_optimizer(args, model)
    # optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 2000, args.train_steps)
    
    return token_loss_fn, optimizer, lr_scheduler


def forward_step(device, args, source, target, model, token_loss_fn):
    source, target = source.to(device), target.to(device)
    batch_size, seq_length = source.shape[:2]
    past_key_values_length = 0
    position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).to(device)

    output = model(source, position_ids=position_ids, output_hidden_states = False)
    logits = output.logits

    loss = token_loss_fn(logits.view(-1, logits.size(-1)), target.reshape(-1))

    return loss


def update_step(optimizer, scheduler, world_size):
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

def thread_main(local_rank, world_size, args, device):
    if local_rank == 0:
        writer = SummaryWriter(args.log_dir)
    print(f"running on {device}")


    resume_step = 0
    resume_epoch = 0
    if args.load_from:
        # 先在 CPU 加载以获取 metadata
        states = torch.load(args.load_from, map_location='cpu')
        resume_step = states['global_step']
        # 假设你在保存时也保存了 epoch，如果没有，可以根据 step 和数据集大小估算
        resume_epoch = states.get('epoch', 0) 
        
    model = prepare_model(local_rank, world_size, device, args)
    dataloader = prepare_data(local_rank, world_size, args, resume_step)
    token_loss_fn, optimizer, lr_scheduler = prepare_loss_optimizer(local_rank, model, args)
    
    gradient_accumulation_steps_warmup = args.global_batch_size_warmup // args.local_batch_size // world_size
    gradient_accumulation_steps_train = args.global_batch_size_train // args.local_batch_size // world_size
    
    global_step = resume_step
    accumulated_loss = 0.0
    accumulated_steps = 0

    if args.load_from:
        if world_size == 1:
            model.load_state_dict(states['model_state_dict'])
        else:
            model.module.load_state_dict(states['model_state_dict'])
        # if not args.heterogeneous_opt:
        #     optimizer.load_state_dict(states['optimizer_state_dict'])
        lr_scheduler.load_state_dict(states['lr_scheduler_state_dict'])
    
    model.train()
    for epoch in range(100):
        for local_batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
            if global_step < args.batch_warmup_step:
                gradient_accumulation_steps = gradient_accumulation_steps_warmup
            else:
                gradient_accumulation_steps = gradient_accumulation_steps_train
                if args.enable_forward_svd and (local_batch_idx - 1) >= args.forward_svd_warmup_steps and accumulated_steps == 0:
                    if (local_batch_idx - 1) == args.forward_svd_warmup_steps or \
                    (args.forward_svd_merge_steps > 0 and ((local_batch_idx - 1) - args.forward_svd_warmup_steps) % args.forward_svd_merge_steps == 0):
                        print("split")
                        # for m in model.modules():
                        #     if isinstance(m, BitLinear):
                        #         m.split()
                                
                        m_to_split = model.module if dist.is_initialized() else model
                        for m in m_to_split.modules():
                            if isinstance(m, BitLinear): m.split()
                        
                        # 重新包装 DDP 和重新定义 Optimizer
                        if dist.is_initialized():
                            model = DDP(m_to_split, device_ids=[local_rank])
                        if local_rank == 0:
                            for name, param in model.named_parameters():
                                print(name)
                        
                        optimizer = optim.AdamW(
                            model.parameters(), 
                            lr=args.merged_lr, 
                            betas=(args.adam_beta1, args.adam_beta2), 
                            eps=1e-8, 
                            weight_decay=args.weight_decay
                        )  
                        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 2000, args.train_steps)
                
            loss = forward_step(device, args, source, target, model, token_loss_fn)

            accumulated_loss += loss.item()
            accumulated_steps += 1


            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if local_batch_idx % gradient_accumulation_steps == 0:
                # 只在梯度累积结束时记录一次（每个 global batch）
                if local_rank == 0:
                    avg_loss = accumulated_loss / accumulated_steps
                    writer.add_scalar("loss", avg_loss, global_step)
                    print(f"global batch: {global_step}, loss: {avg_loss:.3f}", flush=True)

                # 重置累积变量
                accumulated_loss = 0.0
                accumulated_steps = 0    
                
                update_step(optimizer, lr_scheduler, world_size)
                global_step += 1
                
                if global_step >= args.train_steps:
                    if local_rank == 0:
                        print("\nReached maximum traning steps. Stopped.")
                    break

                if local_rank == 0 and global_step % args.save_interval == 0:
                    if hasattr(model, 'module'):  # DDP包装的模型
                        model_to_save = model.module
                    else:
                        model_to_save = model
                    
                    checkpoint_path = f"{args.ckpt_dir}/checkpoint_{global_step}.pth"
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    }, checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    parser = build_parser()
    args = parser.parse_args()
    args.device = f"{device}"
    print(args.device)
    if local_rank == 0:
        args_2_json(args)
        os.makedirs(args.ckpt_dir, exist_ok=True)

    thread_main(local_rank, dist.get_world_size(), args, device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
    
