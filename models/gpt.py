import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import torch.nn.init as init

from Metis.bitlinear import *


        

class MultiheadAttention(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.embed_dim = args.embed_dim
        self.heads_num = args.heads
        self.window_size = args.win_size
        assert args.embed_dim % args.heads == 0, 'Embedding dimension must be divisible by number of heads.'

        self.key = BitLinear(args.embed_dim, args.embed_dim, args=args)
        self.query = BitLinear(args.embed_dim, args.embed_dim, args=args)
        self.value = BitLinear(args.embed_dim, args.embed_dim, args=args)
        self.proj = BitLinear(args.embed_dim, args.embed_dim, args=args)
        # self.key = nn.Linear(embed_dim, embed_dim)
        # self.query = nn.Linear(embed_dim, embed_dim)
        # self.value = nn.Linear(embed_dim, embed_dim)
        # self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(args.dropout_prob)
        self.proj_dropout = nn.Dropout(args.dropout_prob)
        self.register_buffer('mask',
            torch.tril(torch.ones(1, 1, self.window_size, self.window_size, device=args.device), diagonal=0)
        )

        self.mask_zero = torch.zeros(1, device=args.device)

    def forward(self, x):
        bs = x.size(0)
        seq_len = x.size(1)

        # x = [bs, seq_len, embed_dim]
        k = self.key(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        q = self.query(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        v = self.value(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        # k, q, v = [bs, heads_num, seq_len, embed_dim // heads_num]

        # [b, h, n, d] * [b, h, d, n] = [b, h, n, n]
        attn = (torch.matmul(q, k.transpose(-2, -1))) / math.sqrt(self.embed_dim // self.heads_num)
        mask = self.mask[:, :, :seq_len, :seq_len] #[1, 1, n, n]
        attn = attn.masked_fill(mask == self.mask_zero, float('-inf')) 

        # attn[b, 0, n] = q[b, 0, d] * k[b, d, n] * mask_fill
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # [b, h, n, n] * [b, h, n, d] = [b, h, n, d]     x[b, 0, d] = attn[b, 0, n] * v[b, n, d]
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(bs, seq_len, self.embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.feed_fwd = nn.Sequential(
            BitLinear(args.embed_dim, 4 * args.embed_dim, args=args),
            nn.GELU(),
            BitLinear(4 * args.embed_dim, args.embed_dim, args=args),
            nn.Dropout(args.dropout_prob)
        )

    def forward(self, x):
        return self.feed_fwd(x)


class Decoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(args.embed_dim, device=args.device)
        self.ln2 = nn.LayerNorm(args.embed_dim, device=args.device)
        self.attn = MultiheadAttention(args)
        self.feed_fwd = FeedForward(args)
        
        self.get_attn_output_hook = lambda x, y, z: None
        self.get_ffn_output_hook = lambda x, y, z: None

    def forward(self, x):
        if isinstance(x, tuple):
            x, _ = x
        x = self.get_attn_output(x)
        x = self.get_ffn_output(x)

        return x
    
    def get_attn_output(self, x):
        if isinstance(x, tuple):
            x, _ = x
        attn_out = self.attn(x)
        out = attn_out + x
        out = self.ln1(out)
        self.get_attn_output_hook(attn_out, x, out)
        return out
    
    def get_ffn_output(self, x):
        ffn_out = self.get_ffn_output_wo_ln(x)
        out = ffn_out + x
        self.get_ffn_output_hook(ffn_out, x, out)
        out = self.ln2(out)
        return out
    
    def get_ffn_output_wo_ln(self, x):
        if isinstance(x, tuple):
            x, _ = x
        x = self.feed_fwd(x)
        return x
    
    def ffn_ln(self, x):
        return self.ln2(x)

class GPT(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(
            args.vocab_size, 
            args.embed_dim, 
            padding_idx=args.vocab_size-1, 
            device=args.device
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, args.win_size, args.embed_dim).to(device=args.device))
        self.dropout = nn.Dropout(args.dropout_prob)
        self.decoders = nn.Sequential(*[Decoder(args) for _ in range(args.layers)])
        self.ln = nn.LayerNorm(args.embed_dim, device=args.device)
        self.fc = nn.Linear(args.embed_dim, args.vocab_size, bias=False, device=args.device)


    def forward(self, x):
        x = self.get_decoder_output(x, len(self.decoders) - 1)
        x = self.decode(x)

        return x

    def get_decoder_output(self, x, i, prev = None):
        if prev is None:
            x = self.embed(x)
            for j in range(i + 1):
                x = self.decoders[j](x)
            return x
        else:
            return self.decoders[i](prev)

    def get_attn_output(self, x, layer):
        x = self.get_decoder_output(x, layer - 1)
        x = self.decoders[layer].get_attn_output(x)
        return x

    def decode(self, x):
        x = self.fc(self.ln(x))
        return x

    def embed(self, x):
        seq_len = x.size(1)
        tok_x = self.tok_emb(x)
        pos_emb = self.pos_emb[:, :seq_len, :]
        x = self.dropout(tok_x) + pos_emb
        return x

