import json
import os
import argparse
import time
import inspect
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup, Qwen3ForCausalLM

from torchgpipe import GPipe  # <- 关键：gpipe

from utils import DCLMDataset, MyDistributedSampler

from Metis.bitlinear import BitLinear

# import sys
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# opti_path = os.path.join(project_root, 'opti')
# sys.path.append(project_root)

# from opti import (
#     NewMuon, Muon, Dion, ProfilingDion, ProfilingMuon, ProfilingNewMuon, ProfilingAdam,
#     ProfilingMyDion, Myopt, MomentumMuon, NewMuon_SM, SGDMuon, Oron,
#     Newmuon_SM_iter, Newmuon_iter
# )


# =========================
# misc
# =========================
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
        os.makedirs(args.log_dir, exist_ok=True)
    with open(f"{args.log_dir}/args.{int(time.time())}.json", "w") as fd:
        json.dump(dic, fd, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(description="PP+DP pretraining launcher for Qwen3 model (torchgpipe + DDP)")

    # batching and sequence
    parser.add_argument('--local-batch-size', type=int, default=4)
    parser.add_argument('--global-batch-size-warmup', type=int, default=256)
    parser.add_argument('--global-batch-size-train', type=int, default=512)
    parser.add_argument('--batch-warmup-step', type=int, default=2000)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--train-steps', type=int, default=10)
    parser.add_argument('--heterogeneous-opt', type=bool, default=False)

    # PP config (torchgpipe)
    parser.add_argument('--pp-size', type=int, default=2, help="pipeline stages per DP process (GPUs per process)")
    parser.add_argument('--pp-chunks', type=int, default=8, help="micro-batches inside GPipe (chunks)")
    parser.add_argument('--pp-checkpoint', type=str, default="except_last",
                        choices=["never", "always", "except_last"],
                        help="activation checkpoint policy for GPipe")
    parser.add_argument('--dtype', type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument('--untie-embeddings', action='store_true',
                        help="If embeddings and lm_head are tied, untie them so GPipe can place them on different GPUs.")
    parser.set_defaults(untie_embeddings=True)

    # data / training flags
    parser.add_argument('--data-shuffle', dest='data_shuffle', action='store_true')
    parser.set_defaults(data_shuffle=True)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument('--optimizer-name', type=str, default='adam')
    parser.add_argument("--orthM", type=bool, default=True)
    parser.add_argument("--power-iters", type=int, default=1)
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


def _dtype_from_str(s: str):
    if s == "fp32":
        return torch.float32
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    raise ValueError(s)

def _as_cos_sin(x):
    # 返回 (cos, sin) 或 None
    if x is None:
        return None
    if isinstance(x, (tuple, list)) and len(x) >= 2:
        cos, sin = x[0], x[1]
        if torch.is_tensor(cos) and torch.is_tensor(sin):
            return (cos, sin)
    # 有些实现可能返回对象
    if hasattr(x, "cos") and hasattr(x, "sin"):
        cos, sin = x.cos, x.sin
        if torch.is_tensor(cos) and torch.is_tensor(sin):
            return (cos, sin)
    return None


def _infer_position_embeddings(block, hidden_states, position_ids, rotary):
    """
    兼容 Qwen3 的 rotary_emb：尽可能计算出 (cos, sin)。
    如果失败，会 raise 并打印 rotary_emb.forward 的 signature，方便你定位。
    """
    # print(type(block))
    # attn = getattr(block, "self_attn", None)
    # if attn is None:
    #     return None
    # rotary = getattr(attn, "rotary_emb", None)
    # if rotary is None:
    #     return None

    seqlen = hidden_states.shape[1]
    pid2d = position_ids
    pid1d = position_ids[0] if position_ids.dim() == 2 else position_ids  # 有些实现只吃 [S]

    # 多种候选调用方式
    cands = []

    # 1) 最常见：rotary(x, position_ids[, seq_len])
    cands += [
        lambda: rotary(hidden_states, pid2d),
        lambda: rotary(hidden_states, pid1d),
        lambda: rotary(hidden_states, pid2d, seqlen),
        lambda: rotary(hidden_states, pid1d, seqlen),
        lambda: rotary(hidden_states, pid2d, seq_len=seqlen),
        lambda: rotary(hidden_states, pid1d, seq_len=seqlen),
    ]

    # 2) 有些实现：rotary(position_ids[, seq_len])
    cands += [
        lambda: rotary(pid2d),
        lambda: rotary(pid1d),
        lambda: rotary(pid2d, seqlen),
        lambda: rotary(pid1d, seqlen),
        lambda: rotary(pid2d, seq_len=seqlen),
        lambda: rotary(pid1d, seq_len=seqlen),
    ]

    # 3) 有些实现：rotary(seq_len=..., device=..., dtype=...)
    cands += [
        lambda: rotary(seq_len=seqlen, device=hidden_states.device, dtype=hidden_states.dtype),
        lambda: rotary(seqlen, hidden_states.device, hidden_states.dtype),
        lambda: rotary(seqlen),
    ]

    last_err = None
    for fn in cands:
        try:
            out = fn()
            cs = _as_cos_sin(out)
            if cs is not None:
                return cs
        except Exception as e:
            last_err = e
            continue

    # 最终失败：打印 rotary 的 forward 签名，强制报错（比默默 None 好排查）
    try:
        sig = inspect.signature(rotary.forward)
    except Exception:
        sig = "<cannot get signature>"


    raise RuntimeError(
        f"Failed to compute position_embeddings (cos,sin) for Qwen3.\n"
        f"rotary_emb = {type(rotary)}\n"
        f"rotary_emb.forward signature = {sig}\n"
        f"last_err = {repr(last_err)}\n"
        f"position_ids.shape={tuple(position_ids.shape)} seqlen={seqlen} dtype={hidden_states.dtype} device={hidden_states.device}"
    )

def _call_decoder_layer(layer, hidden_states, position_ids, position_embeddings=None):
    """
    只传 layer.forward 支持的 kwargs，并补上 position_embeddings。
    """
    sig = inspect.signature(layer.forward)
    kwargs = {}

    if "position_ids" in sig.parameters:
        kwargs["position_ids"] = position_ids

    if "position_embeddings" in sig.parameters:
        kwargs["position_embeddings"] = position_embeddings

    if "attention_mask" in sig.parameters:
        kwargs["attention_mask"] = None
    if "output_attentions" in sig.parameters:
        kwargs["output_attentions"] = False
    if "use_cache" in sig.parameters:
        kwargs["use_cache"] = False
    if "past_key_value" in sig.parameters:
        kwargs["past_key_value"] = None
    if "cache_position" in sig.parameters:
        kwargs["cache_position"] = None

    out = layer(hidden_states, **kwargs)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class QwenEmbedAndBlocks(nn.Module):
    def __init__(self, embed_tokens: nn.Module, blocks: nn.ModuleList, rotary_emb):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.blocks = nn.ModuleList(list(blocks))
        self.rotary_emb = rotary_emb

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.embed_tokens(input_ids)
        bsz, seqlen = input_ids.shape
        position_ids = torch.arange(seqlen, device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        pos_emb = None
        if len(self.blocks) > 0:
            pos_emb = _infer_position_embeddings(self.blocks[0], hidden_states, position_ids, self.rotary_emb)
            # print("infer position embeddings")

        for blk in self.blocks:
            # print("="*20)
            # print(pos_emb)
            hidden_states = _call_decoder_layer(blk, hidden_states, position_ids, position_embeddings=pos_emb)
        return hidden_states


class QwenBlocksOnly(nn.Module):
    def __init__(self, blocks: nn.ModuleList, rotary_emb):
        super().__init__()
        self.blocks = nn.ModuleList(list(blocks))
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states: torch.Tensor):
        bsz, seqlen = hidden_states.shape[0], hidden_states.shape[1]
        position_ids = torch.arange(seqlen, device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        pos_emb = None
        if len(self.blocks) > 0:
            pos_emb = _infer_position_embeddings(self.blocks[0], hidden_states, position_ids, self.rotary_emb)
            # print("="*20)
            # print(pos_emb)

        for blk in self.blocks:
            hidden_states = _call_decoder_layer(blk, hidden_states, position_ids, position_embeddings=pos_emb)
        return hidden_states


class QwenBlocksNormHead(nn.Module):
    def __init__(self, blocks: nn.ModuleList, norm: nn.Module, lm_head: nn.Module, rotary_emb):
        super().__init__()
        self.blocks = nn.ModuleList(list(blocks))
        self.norm = norm
        self.lm_head = lm_head
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states: torch.Tensor):
        bsz, seqlen = hidden_states.shape[0], hidden_states.shape[1]
        position_ids = torch.arange(seqlen, device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        pos_emb = None
        if len(self.blocks) > 0:
            # print("infer position embeddings")
            pos_emb = _infer_position_embeddings(self.blocks[0], hidden_states, position_ids, self.rotary_emb)

        for blk in self.blocks:
            # print("="*20)
            # print(pos_emb)
            hidden_states = _call_decoder_layer(blk, hidden_states, position_ids, position_embeddings=pos_emb)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


def _split_layers(n_layers: int, pp_size: int):
    """把 n_layers 均匀切成 pp_size 段，返回每段的 [start,end)"""
    base = n_layers // pp_size
    rem = n_layers % pp_size
    sizes = [base + (1 if i < rem else 0) for i in range(pp_size)]
    bounds = []
    s = 0
    for sz in sizes:
        bounds.append((s, s + sz))
        s += sz
    return bounds

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
def prepare_model(dp_rank, dp_world_size, local_proc_rank, args):
    # ---- device mapping: one DP process uses pp_size GPUs: [base, base+1, ...]
    n_gpus = torch.cuda.device_count()
    assert args.pp_size >= 1
    # assert (local_proc_rank + 1) * args.pp_size <= n_gpus, \
    #     f"Not enough GPUs on this node. local_proc_rank={local_proc_rank}, pp_size={args.pp_size}, n_gpus={n_gpus}"

    base = local_proc_rank * args.pp_size
    pp_devices = [torch.device(f"cuda:{base + i}") for i in range(args.pp_size)]
    torch.cuda.set_device(pp_devices[0])  # NCCL init / default current device

    # ---- build hf model (CPU first), then turn into GPipe sequential
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    dtype = _dtype_from_str(args.dtype)

    # 重要：减少 CPU 内存峰值（可选：你也可以改成先 to(bf16) 再切）
    hf_model = Qwen3ForCausalLM(config)
    replace_linear_with_custom(hf_model, args)
    hf_model.train()

    # 如果 embed 和 lm_head 绑在一起，GPipe 无法把它们放在不同 GPU：默认 untie
    # 注意：untie 会多一份 vocab projection 权重，显著增加显存（尤其是 8B + 大 vocab）
    try:
        tied = (hf_model.lm_head.weight is hf_model.model.embed_tokens.weight)
    except Exception:
        tied = False

    if tied and args.untie_embeddings:
        # clone 一个新的 lm_head，权重初始化为 embed 权重
        vocab = hf_model.lm_head.weight.shape[0]
        hidden = hf_model.lm_head.weight.shape[1]
        new_head = nn.Linear(hidden, vocab, bias=False)
        new_head.weight.data.copy_(hf_model.lm_head.weight.data)
        hf_model.lm_head = new_head
        config.tie_word_embeddings = False
        if dp_rank == 0:
            print("[PP] Detected tied embeddings. Untied lm_head to enable GPipe placement.", flush=True)
    elif tied and (not args.untie_embeddings):
        raise RuntimeError(
            "Detected tied embeddings (embed_tokens.weight is lm_head.weight). "
            "GPipe cannot split tied params across devices. Enable --untie-embeddings."
        )

    # cast dtype before gpipe partitioning (parameters still on CPU here)
    hf_model = hf_model.to(dtype=dtype)

    if dp_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        config.to_json_file(f"{args.ckpt_dir}/config.json")

    # Extract backbone pieces
    embed_tokens = hf_model.model.embed_tokens
    blocks = hf_model.model.layers
    norm = hf_model.model.norm
    lm_head = hf_model.lm_head
    rotary_emb = hf_model.model.rotary_emb
    
    n_layers = len(blocks)

    bounds = _split_layers(n_layers, args.pp_size)

    stages = []
    # stage0: embed + first block range
    s0, e0 = bounds[0]
    stages.append(QwenEmbedAndBlocks(embed_tokens, blocks[s0:e0], rotary_emb))

    # middle stages
    for i in range(1, args.pp_size - 1):
        si, ei = bounds[i]
        stages.append(QwenBlocksOnly(blocks[si:ei], rotary_emb))

    # last stage
    sl, el = bounds[-1]
    stages.append(QwenBlocksNormHead(blocks[sl:el], norm, lm_head, rotary_emb))

    seq = nn.Sequential(*stages)

    # Now build GPipe
    balance = [1] * args.pp_size  # each stage module per partition
    gpipe_model = GPipe(
        seq,
        balance=balance,
        devices=pp_devices,
        chunks=max(1, args.pp_chunks),
        checkpoint=args.pp_checkpoint,
    )

    # Drop hf_model reference to avoid duplicated module trees
    del hf_model

    # Wrap with DDP across DP processes
    if dp_world_size > 1:
        # multi-device module: device_ids 必须 None
        gpipe_model = DDP(
            gpipe_model,
            device_ids=None,
            output_device=None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # params count
    if dp_rank == 0:
        total = sum(p.numel() for p in gpipe_model.parameters())
        trainable = sum(p.numel() for p in gpipe_model.parameters() if p.requires_grad)
        print(f"[PP+DP] model ok. trainable={trainable/1e9:.2f}B total={total/1e9:.2f}B "
              f"pp_size={args.pp_size} dp_size={dp_world_size}", flush=True)

    return gpipe_model, pp_devices


def prepare_data(dp_rank, dp_world_size, args, resume_step):
    tokenizer = AutoTokenizer.from_pretrained(args.config_dir, trust_remote_code=True)
    dataset = DCLMDataset(args.data_dir, args.seq_len, tokenizer)

    # 注意：数据并行的 world_size 现在是 dp_world_size（进程数），不是总 GPU 数
    sampler = MyDistributedSampler(
        dataset,
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=args.data_shuffle,
        cont_step=resume_step,
        warmup_steps=args.batch_warmup_step,
        bs_warmup=args.global_batch_size_warmup,
        bs_normal=args.global_batch_size_train
    )
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, num_workers=args.num_workers, sampler=sampler)

    if dp_rank == 0:
        print(f"[DP] data ok. Data Length {len(dataset)}. dp_world_size={dp_world_size}", flush=True)
    return dataloader


def get_optimizer(args, model):
    # model may be DDP(GPipe(...))
    named_params = model.named_parameters()

    if args.optimizer_name == "adam":
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=1e-8,
            weight_decay=args.weight_decay
        )

    raise AssertionError("optimizer not supported")


def prepare_loss_optimizer(model, args):
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=151643, reduction='mean')
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 2000, args.train_steps)
    return token_loss_fn, optimizer, lr_scheduler


def forward_step(pp_devices, source, target, model, token_loss_fn):
    """
    source 放到 pipeline 第一个 device；target 放到最后一个 device；
    logits 会在最后一个 device 上产出（GPipe）。
    """
    dev0 = pp_devices[0]
    dev_last = pp_devices[-1]

    source = source.to(dev0, non_blocking=True)
    target = target.to(dev_last, non_blocking=True)

    logits = model(source)  # GPipe: input -> logits on last device
    loss = token_loss_fn(logits.view(-1, logits.size(-1)), target.reshape(-1))
    return loss


def update_step(optimizer, scheduler):
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


def thread_main(dp_rank, dp_world_size, local_proc_rank, args):
    # build model (PP inside process, DP across processes)
    model, pp_devices = prepare_model(dp_rank, dp_world_size, local_proc_rank, args)

    writer = SummaryWriter(args.log_dir) if dp_rank == 0 else None

    # resume
    resume_step = 0
    if args.load_from:
        states = torch.load(args.load_from, map_location='cpu')
        resume_step = states['global_step']

    dataloader = prepare_data(dp_rank, dp_world_size, args, resume_step)
    token_loss_fn, optimizer, lr_scheduler = prepare_loss_optimizer(model, args)

    # grad accumulation computed on DP world size (processes)
    ga_warm = args.global_batch_size_warmup // args.local_batch_size // dp_world_size
    ga_train = args.global_batch_size_train // args.local_batch_size // dp_world_size

    global_step = resume_step
    accumulated_loss = 0.0
    accumulated_steps = 0

    # load states
    if args.load_from:
        # model might be DDP-wrapped
        target_model = model.module if hasattr(model, "module") else model
        missing, unexpected = target_model.load_state_dict(states['model_state_dict'], strict=False)
        if dp_rank == 0:
            print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

        if not args.heterogeneous_opt and 'optimizer_state_dict' in states:
            optimizer.load_state_dict(states['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in states:
            lr_scheduler.load_state_dict(states['lr_scheduler_state_dict'])

    model.train()

    for epoch in range(100):
        for local_batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
            ga = ga_warm if global_step < args.batch_warmup_step else ga_train
            ga = max(1, ga)

            loss = forward_step(pp_devices, source, target, model, token_loss_fn)

            accumulated_loss += loss.item()
            accumulated_steps += 1

            loss = loss / ga

            # DDP 梯度累积：非最后一次 backward 用 no_sync 省 allreduce
            if hasattr(model, "no_sync") and (local_batch_idx % ga != 0):
                sync_ctx = model.no_sync()
            else:
                sync_ctx = nullcontext()

            with sync_ctx:
                loss.backward()

            if local_batch_idx % ga == 0:
                if dp_rank == 0:
                    avg_loss = accumulated_loss / max(1, accumulated_steps)
                    writer.add_scalar("loss", avg_loss, global_step)
                    print(f"global batch: {global_step}, loss: {avg_loss:.3f}", flush=True)

                accumulated_loss = 0.0
                accumulated_steps = 0

                update_step(optimizer, lr_scheduler)
                global_step += 1

                if global_step >= args.train_steps:
                    if dp_rank == 0:
                        print("\nReached maximum training steps. Stopped.", flush=True)
                    break

                if dp_rank == 0 and (global_step % args.save_interval == 0):
                    model_to_save = model.module if hasattr(model, "module") else model
                    checkpoint_path = f"{args.ckpt_dir}/checkpoint_{global_step}.pth"
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    }, checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}", flush=True)

        if global_step >= args.train_steps:
            break


def main():
    dist.init_process_group(backend="nccl")
    parser = build_parser()
    args = parser.parse_args()

    # torchrun 环境变量
    local_proc_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_proc_rank}")
    args.device = f"{device}"
    print(args.device)
    # 先根据 local_proc_rank 设置默认 device（NCCL init 会用到）
    # 注意：一个进程会用 pp_size 张 GPU，base = local_proc_rank * pp_size
    base = local_proc_rank * args.pp_size
    torch.cuda.set_device(base)

    dp_rank = dist.get_rank()
    dp_world_size = dist.get_world_size()

    if dp_rank == 0:
        args_2_json(args)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

    thread_main(dp_rank, dp_world_size, local_proc_rank, args)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
