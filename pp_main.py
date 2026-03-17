import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from utils import Tokenized_data, MyDistributedSampler
from models import TransformerSeq
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from Metis import BitLinear
from utils import parse
from torchgpipe import GPipe
import torch.distributed as dist
from opti import NewMuon, Muon, Dion, ProfilingDion, ProfilingMuon, ProfilingNewMuon, ProfilingAdam, ProfilingMyDion, Myopt, MomentumMuon, NewMuon_SM, SGDMuon, Oron, Newmuon_SM_iter, Newmuon_iter
from torch.nn.parallel import DistributedDataParallel as DDP


def load_model(args, local_rank):
    # if args.local_rank >= 0:
    #     torch.cuda.set_device(args.local_rank)
    #     dist.init_process_group(backend='nccl') 
    #     args.device = torch.device("cuda", args.local_rank)
    # model = TransformerSeq(args).to(args.device)
    start_gpu = local_rank * 4  # 0->0, 1->4
    dev_ids = list(range(start_gpu, start_gpu + 4)) # [0,1,2,3] 或 [4,5,6,7]
    devices = [torch.device(f"cuda:{i}") for i in dev_ids]
    
    # 设置当前进程的“主显卡”，这是解决 NCCL 报错的核心
    torch.cuda.set_device(devices[0])
    args.device = devices[0]
    model = TransformerSeq(args).to(devices[0])


    # You need to change the balance here if you want to change the pp configuration or the model size.
    # model = GPipe(model, balance=[8,9,9,9], chunks=min(args.grad_acc, 8))
    # model = GPipe(model, balance=[4,4,5,5,5,5,5,2], chunks=args.grad_acc)
    model = GPipe(model, balance=[10,10,10,5], chunks=args.grad_acc, devices=devices)
    
    if args.load_from:
        state_dict = torch.load(args.load_from)

        try:
            model.load_state_dict(state_dict)
        except:
            for m in model.modules():
                if isinstance(m, BitLinear):
                    m.split()
            model.load_state_dict(state_dict)
            
        if args.local_rank == 0:
            print(f"model loaded from {args.load_from}")

    if args.local_rank == 0:
        print(f'Model ok on device {args.device}. params: {sum(p.numel() for p in model.parameters())}')
        print("******** Model Parameters *******")
        for name, p in model.named_parameters():
            print(name, p.shape)
        print("*********************************")
    return model

def load_dataset(args, local_rank, world_size, resume_step):
    dataset = Tokenized_data(args)
    sampler = MyDistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=args.shuffle, cont_step=resume_step, warmup_steps=args.lr_warmup_steps, bs_warmup=512, bs_normal=512)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.dataset_workers, sampler=sampler)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.dataset_workers)
    # dataloader = DataLoader(
    #     dataset, 
    #     batch_size=args.batch_size, 
    #     shuffle=args.shuffle, 
    #     num_workers=args.dataset_workers
    # )
    print(f'Data ok on device {args.device}.')

    return dataloader, sampler

def get_optimizer(args, model):
    if args.optimizer_name == "adam":
        return optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            betas=(args.adam_beta1, args.adam_beta2), 
            eps=1e-8, 
            weight_decay=args.weight_decay
        ) 
    elif args.optimizer_name == "newmuon":
        muon_params = [
            p for name, p in model.named_parameters() 
            if p.ndim >= 2 and "emb" not in name and "lm" not in name and p.requires_grad
        ]
        adamw_params = [
            p for name, p in model.named_parameters() 
            if not(p.ndim >= 2 and "emb" not in name and "lm" not in name) and p.requires_grad
        ]
        # muon_params = [
        #     p
        #     for name, p in model.named_parameters()
        #     if p.ndim >= 2 and "emb" not in name and "fc" not in name
        # ]
        # adamw_params = [
        #     p
        #     for name, p in model.named_parameters()
        #     if not (
        #         p.ndim >= 2 and "emb" not in name and "fc" not in name
        #     )
        # ]

        return NewMuon(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2)
        )
    elif args.optimizer_name == "muon":
        muon_params = [
            p for name, p in model.named_parameters() 
            if p.ndim >= 2 and "emb" not in name and "lm" not in name and p.requires_grad
        ]
        adamw_params = [
            p for name, p in model.named_parameters() 
            if not(p.ndim >= 2 and "emb" not in name and "lm" not in name) and p.requires_grad
        ]

        return Muon(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2)
        )
    elif args.optimizer_name == "newmuon_sm":
        muon_params = [
            p for name, p in model.named_parameters() 
            if p.ndim >= 2 and "emb" not in name and "lm" not in name and p.requires_grad
        ]
        adamw_params = [
            p for name, p in model.named_parameters() 
            if not(p.ndim >= 2 and "emb" not in name and "lm" not in name) and p.requires_grad
        ]

        return NewMuon_SM(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2),
            orth_Momentum=args.orthM
        )
    elif args.optimizer_name == "newmuon_sm_iter":
        muon_params = [
            p for name, p in model.named_parameters() 
            if p.ndim >= 2 and "emb" not in name and "lm" not in name and p.requires_grad
        ]
        adamw_params = [
            p for name, p in model.named_parameters() 
            if not(p.ndim >= 2 and "emb" not in name and "lm" not in name) and p.requires_grad
        ]
        return Newmuon_SM_iter(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2),
            orth_Momentum=args.orthM,
            power_iters=args.power_iters
        )
    elif args.optimizer_name == "newmuon_iter":
        muon_params = [
            p for name, p in model.named_parameters() 
            if p.ndim >= 2 and "emb" not in name and "lm" not in name and p.requires_grad
        ]
        adamw_params = [
            p for name, p in model.named_parameters() 
            if not(p.ndim >= 2 and "emb" not in name and "lm" not in name) and p.requires_grad
        ]
        return Newmuon_iter(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2),
            power_iters=args.power_iters
        )
    else:
        assert 0, "optimizer not supported"

def train(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # --- 重要：先 set_device，再 init_process_group ---
    # 计算当前进程的主卡索引
    main_gpu_for_this_process = local_rank * 4
    torch.cuda.set_device(main_gpu_for_this_process)
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    dp_rank = dist.get_rank()
    dp_world_size = dist.get_world_size()
    
    if dp_rank == 0:
        writer = SummaryWriter(args.log_dir)
    
    resume_step = 0
    resume_epoch = 0
    if args.load_from:
        # 先在 CPU 加载以获取 metadata
        states = torch.load(args.load_from, map_location='cpu')
        resume_step = states['global_step']
        resume_epoch = states.get('epoch', 0) 

    dataloader, sampler = load_dataset(args, dp_rank, dp_world_size, resume_step)
    model = load_model(args, local_rank)
    model = DDP(model, device_ids=None, static_graph=True) 
    loss_fn = nn.CrossEntropyLoss(ignore_index = args.vocab_size - 1)
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.lr_warmup_steps, args.train_steps)
    
    if args.load_from:
        optimizer.load_state_dict(states['optimizer_state_dict'])
        lr_scheduler.load_state_dict(states['lr_scheduler_state_dict'])

    model.train()

    train_steps = resume_step
    acc_steps = 1
    acc_loss = 0
    optimizer.zero_grad()
    
    
    for epoch in range(resume_epoch, args.max_epochs):
        sampler.set_epoch(epoch)
        for batch, (source, target, _) in enumerate(dataloader):
            
            source, target = source.to(model.module.devices[0]), target.to(model.module.devices[-1])
            optimizer.zero_grad()
            logit = model(source)
            loss = loss_fn(logit.view(-1, args.vocab_size), target.view(-1)).to(model.module.devices[-1])
            if dp_rank <= 0:
                writer.add_scalar("loss", loss, train_steps)
            # if dp_rank % 4 == 0:
            # print(f"{dp_rank}, Logit sample: {logit[0, 0, :5]}") # 打印前几个输出
            print(f"{dp_rank}, Target sample: {target[0, :20]}") # 打印前几个标签
            loss.backward()
                
            # logit = model(source)
            # loss = loss_fn(logit.view(-1, args.vocab_size), target.view(-1)) / args.grad_acc
            # acc_loss += loss.item()
        
            # if acc_steps == args.grad_acc:
            #     if dp_rank <= 0:
            #         writer.add_scalar("loss", acc_loss, train_steps)
                
            #     loss.backward()
                
            # else:
            #     print(loss)
            #     loss.backward()
            #     acc_steps += 1
            #     continue
            

            if dp_rank <= 0:
                print(f"rank: {dp_rank}, "
                    f"epoch: {epoch}, "
                    f"batch: {train_steps}, "
                    f"loss: {loss:.3f}, "
                    )
            
            # g = 0
            # for name, p in model.named_parameters():
            #     if not (p.grad is None):
            #         g += p.grad.norm().item()
            # clip_thres = 1 if args.grad_clipping > g else args.grad_clipping / g
            # for name, p in model.named_parameters():
            #     if not (p.grad is None):
            #         p.grad *= clip_thres

            optimizer.step()
            lr_scheduler.step()
            
            torch.cuda.synchronize() 

            if train_steps % args.save_steps == 0 and train_steps > 0 and dp_rank == 0:
                    
                checkpoint_path = f"{args.ckpt_dir}/checkpoint_{train_steps}.pth"
                torch.save({
                    'global_step': train_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
                    
                # torch.save(model.state_dict(), f"{args.chkpt_dir}/{args.tag}/{epoch}_{batch}.pth")
                # print(f"model saved at {args.chkpt_dir}/{args.tag}/{epoch}_{batch}.pth")
            
            acc_loss = 0
            acc_steps = 1
            if args.train_steps == train_steps:
                break
            train_steps += 1


if __name__ == "__main__":
    args = parse()
    train(args)
