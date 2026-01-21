import argparse
import os
import torch
import numpy as np
import random
import json
import time

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

def check_args(args):
    if args.device is None:
        args.device = get_available_cuda_device()
    args.device = f"cuda:{args.device}"


    if args.load_args_from:
        args = json_2_args(args)
    else:
        args_2_json(args)
    
    chkpt_dir = f"{args.chkpt_dir}/{args.tag}"
    if not os.path.exists(chkpt_dir):
        try:
            os.makedirs(chkpt_dir)
            
        except:
            pass

    if args.tag is None:
        setattr(args, "tag", f"seed_{args.seed}_amp_{args.amp}")

    args.dim = args.embed_dim
    args.n_layers = args.layers
    args.n_heads = args.heads
    args.max_seq_len = args.win_size

    
    
    set_seed(args.seed)

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
    
    if not os.path.exists(f"{args.log_dir}/{args.tag}"):
        os.makedirs(f"{args.log_dir}/{args.tag}")
    with open(f"{args.log_dir}/{args.tag}/args.{int(time.time())}.json", "w") as fd:
        json.dump(dic, fd, indent=2)

def json_2_args(args, verbose=True):
    json.dump("", args.load_args_from)

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--load-args-from", type=str, default="")
    parser.add_argument("--new-tag", type=str, default="")
    
    parser.add_argument("--local-rank", type=int, default=-1)


    # trainer config
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=200000)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--chkpt-dir", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--device", type=int)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--dataset-workers", type=int, default=1)
    parser.add_argument("--adam-beta1", type=int, default=0.9)
    parser.add_argument("--adam-beta2", type=int, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-from", type=str, default="")
    parser.add_argument("--grad-clipping", type=float, default=1.0)
    parser.add_argument("--grad-acc", type=int, default=1)

    parser.add_argument("--reg-alpha1", type=float, default=1.0)
    parser.add_argument("--reg-alpha2", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=0.0)
    parser.add_argument("--reg-beta", type=int, default=2)
    
    parser.add_argument("--enable-nv-recipe", action="store_true")

    # gpt config
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--win-size", type=int, default=512)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    parser.add_argument("--vocab-size", type=int, default=50258)
    
    # llama config
    parser.add_argument("--multiple-of", type=int, default=256)
    parser.add_argument("--ffn-dim-multiplier", type=int, default=1)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=1e-5)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--model", type=str, default="gpt")
    

    parser.add_argument("--enable-lowbit", action="store_true")


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
    
    
    args = parser.parse_args()

    check_args(args)
    return args
