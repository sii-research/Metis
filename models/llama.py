
import torch
from torch import nn
from torch.nn import functional as F

# from torch.nn import RMSNorm
import math
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd
from matplotlib import pyplot as plt
from Metis.bitlinear import *

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device: str="cuda"):
        super().__init__()
        self.device = device
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim = -1, keepdim=True) + self.eps)
    
    def forward(self, x):
        #Shape: x[bs, seq,dim]
        output = self._norm(x.float()).type_as(x)
        
        #Shape: x[bs,seq,dim] -> x_norm[bs,seq,dim]
        return output * self.weight

def precompute_freqs_cis(dim:int, seq_len: int, device: str, theta: float=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[:(dim//2)].float()/dim))
    
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    
    freqs = torch.outer(t, freqs).to(device)
    
    freqs_cis = torch.polar(torch.ones_like(freqs).to(device), freqs).to(device)
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), "the last two dimension of freqs_cis, x must match"
    shape = [d if i==1 or i==ndim-1 else 1 for i,d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, device)->Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(device) #xq_:[bsz, seq_len, n_heads, head_dim/2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(device) #xk_:[bsz, seq_len, n_heads, head_dim/2]
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).to(device) #xq_out:[bsz, seq_len, n_heads, head_dim]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).to(device) #xk_out:[bsz, seq_len, n_heads, head_dim]
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.dim = args.dim
    self.n_heads = args.n_heads
    self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    self.head_dim = args.dim // args.n_heads
    self.n_rep = args.n_heads // args.n_kv_heads

    self.wq = BitLinear(self.dim, self.n_heads * self.head_dim, args=args, bias=False)
    self.wk = BitLinear(self.dim, self.n_kv_heads * self.head_dim, args=args, bias=False)
    self.wv = BitLinear(self.dim, self.n_kv_heads * self.head_dim, args=args, bias=False)
    self.wo = BitLinear(self.n_heads * self.head_dim, self.dim, args=args, bias=False)

  def forward(self, x: torch.Tensor, start_pos, inference):
    bsz, seq_len, _ = x.shape
    mask = None

    xq = self.wq(x)  #x[bsz,seq_len,dim]*wq[dim,n_heads * head_dim] -> q[bsz,seq_len,n_heads * head_dim]
    xk = self.wk(x)  #x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> k[bsz,seq_len,n_kv_heads * head_dim]
    xv = self.wv(x)  #x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> v[bsz,seq_len,n_kv_heads * head_dim]

    xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)      #xq[bsz,seq_len,n_heads, head_dim]
    xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   #xk[bsz,seq_len,n_kv_heads, head_dim]
    xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   #xv[bsz,seq_len,n_kv_heads, head_dim]

    inference = False
    if inference:
      freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len * 2, device=xq.device)
      freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis, xq.device)

      self.cache_k = self.cache_k.to(xq)
      self.cache_v = self.cache_v.to(xq)
      self.cache_k[:bsz, start_pos:start_pos + seq_len] = xk
      self.cache_v[:bsz, start_pos:start_pos + seq_len] = xv

      keys = self.cache_k[:bsz, :start_pos + seq_len]
      values = self.cache_v[:bsz, :start_pos + seq_len]

      keys = repeat_kv(keys, self.n_rep)      #keys[bsz,seq_len,n_heads,head_dim]
      values = repeat_kv(values, self.n_rep)  #values[bsz,seq_len,n_heads,head_dim]

    # Mode - Training mode: KV-Cache not implemented
    else:
      freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len, device=xq.device)

      #xq[bsz,seq_len,n_heads, head_dim], xk[bsz,seq_len,n_heads, head_dim]
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis, xq.device)

      #keys[bsz,seq_len,n_heads,head_dim], #values[bsz,seq_len,n_heads,head_dim]
      keys = repeat_kv(xk, self.n_rep)
      values = repeat_kv(xv, self.n_rep)

      mask = torch.full((seq_len, seq_len),float("-inf"),device=xq.device)
      mask = torch.triu(mask, diagonal=1).to(xq.device)

   
    xq = xq.transpose(1,2)                  #xq[bsz,n_heads,seq_len,head_dim]
    keys = keys.transpose(1,2)              #keys[bsz,n_heads,seq_len,head_dim]
    values = values.transpose(1,2)          #values[bsz,n_heads,seq_len,head_dim]

    # Computing attention score
    scores = torch.matmul(xq, keys.transpose(2,3))/math.sqrt(self.head_dim)
    if mask is not None:
      scores = scores + mask

    # Apply softmax to the attention score
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    # Matrix multiplication of attention score with the values
    output = torch.matmul(scores, values).to(xq.device)

    output = output.transpose(1,2).contiguous().view(bsz, seq_len, -1)

    # shape: output [bsz,seq_len,dim]
    return self.wo(output)

def repeat_kv(x:torch.Tensor, n_rep: int)-> torch.Tensor:
  bsz, seq_len, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
    return x
  return (
      x[:,:,:,None,:]
      .expand(bsz,seq_len,n_kv_heads,n_rep, head_dim)
      .reshape(bsz,seq_len,n_kv_heads * n_rep, head_dim)
  )

class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        
        hidden_dim = int(8 * args.dim/3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        # self.w1 = nn.Linear(self.dim, hidden_dim, bias=False, device=args.device)
        # self.w2 = nn.Linear(hidden_dim, self.dim, bias=False, device=args.device)
        # self.w3 = nn.Linear(self.dim, hidden_dim, bias=False, device=args.device)
        
        self.w1 = BitLinear(self.dim, hidden_dim, bias=False, args=args)
        self.w2 = BitLinear(hidden_dim, self.dim, bias=False, args=args)
        self.w3 = BitLinear(self.dim, hidden_dim, bias=False, args=args)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.attention_norm = RMSNorm(args.dim, eps = args.norm_eps)
    self.attention = Attention(args)
    self.ff_norm = RMSNorm(args.dim, eps = args.norm_eps)
    self.feedforward = FeedForward(args)

  def forward(self, x, start_pos=0, inference=False):
    h = x + self.attention(self.attention_norm(x), start_pos, inference)

    out = h + self.feedforward(self.ff_norm(h))
    # Shape: [bsz,seq_len,dim]
    return out


class Transformer(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.params = args
    self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

    self.layers = nn.ModuleList()
    for layer_id in range(args.n_layers):
      self.layers.append(TransformerBlock(args=args))
    self.norm = RMSNorm(args.dim, eps = args.norm_eps)
  
    self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

  def forward(self, x, start_pos=0, targets=None):
    
    # x[bsz, seq_len] -> h[bsz, seq_len, dim]
    h = self.tok_embeddings(x)

    if targets is None:
      inference = True
    else:
      inference = False

    for layer in self.layers:
      h = layer(h, start_pos, inference)
    h = self.norm(h)

    logits = self.output(h).float()
    loss = None
    if targets is None:
      loss = None
    else:
      loss = F.cross_entropy(logits.view(-1, self.params.vocab_size), targets.view(-1))

    return logits, loss

# for torchgpipe
class TransformerSeq(nn.Sequential):
  def __init__(self, args):
    super().__init__()
    self.params = args
    
    self.append(nn.Embedding(args.vocab_size, args.dim))
    for layer_id in range(args.n_layers):
      self.append(TransformerBlock(args=args))

    self.append(RMSNorm(args.dim, eps = args.norm_eps))
    
    self.append(nn.Linear(args.dim, args.vocab_size, bias=False))

