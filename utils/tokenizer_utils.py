import os
import base64
import tiktoken

ENDOFTEXT = "<|endoftext|>"
PADDING = "<|padding|>" 
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"

def load_tiktoken_bpe(bpe_file):
    with open(bpe_file, 'rb') as f:
        contents =  f.read()

    mergeable_ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }
    return mergeable_ranks


def r50k_base():
    mergeable_ranks = load_tiktoken_bpe('../tokenizers/r50k_base.tiktoken')
    constructor = {
        "name": "r50k_base",
        "explicit_n_vocab": 50258,
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {
            "<|endoftext|>": 50256,
            PADDING : 50257,
            },
    }

    enc = tiktoken.Encoding(**constructor)
    return enc


def p50k_base():
    mergeable_ranks = load_tiktoken_bpe('../tokenizers/p50k_base.tiktoken')
    constructor = {
        "name": "p50k_base",
        "explicit_n_vocab": 50281,
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }
    enc = tiktoken.Encoding(**constructor)
    return enc


def p50k_edit():
    mergeable_ranks = load_tiktoken_bpe('../tokenizers/p50k_base.tiktoken')
    special_tokens = {ENDOFTEXT: 50256, FIM_PREFIX: 50281, FIM_MIDDLE: 50282, FIM_SUFFIX: 50283}
    constructor = {
        "name": "p50k_edit",
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }
    enc = tiktoken.Encoding(**constructor)
    return enc


def cl100k_base():
    mergeable_ranks = load_tiktoken_bpe('../tokenizers/cl100k_base.tiktoken')
    special_tokens = {
        ENDOFTEXT: 100257,
        FIM_PREFIX: 100258,
        FIM_MIDDLE: 100259,
        FIM_SUFFIX: 100260,
        ENDOFPROMPT: 100276,
    }
    constructor = {
        "name": "cl100k_base",
        "pat_str": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }
    enc = tiktoken.Encoding(**constructor)
    return enc


def o200k_base():
    mergeable_ranks = load_tiktoken_bpe('../tokenizers/o200k_base.tiktoken')
    special_tokens = {ENDOFTEXT: 199999, ENDOFPROMPT: 200018}
    # This regex could be made more efficient. If I was the one working on this encoding, I would
    # have done a few other things differently too, e.g. I think you can allocate tokens more
    # efficiently across languages.
    pat_str = "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )
    constructor = {
        "name": "o200k_base",
        "pat_str": pat_str,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }
    enc = tiktoken.Encoding(**constructor)
    return enc


TOKENIZERS = {
    "r50k_base": r50k_base,
    "p50k_base": p50k_base,
    "p50k_edit": p50k_edit,
    "cl100k_base": cl100k_base,
    "o200k_base": o200k_base,
}
