import torch
import json
import tiktoken

from torch.utils.data import Dataset
import base64
import tiktoken

from pathlib import Path

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

def r50k_base(args):
    mergeable_ranks = load_tiktoken_bpe(args.tokenizer_path)
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

def find_files(root_dir):
    root_path = Path(root_dir)
    return [str(file) for file in root_path.rglob('local-shard*/*.jsonl')]
    # return [str(file) for file in root_path.rglob('cut*/*.jsonl')]

class Tokenized_data(Dataset):
    def __init__(self, args, start_from = 0) -> None:
        super().__init__()
        self.dataset_dir = args.dataset_path
        
        self.files = find_files(self.dataset_dir)
        
        self._load_file(0)
        self.lines_per_file = len(self.file_texts)
        self.tokenizer = r50k_base(args=args)
        self.max_seq_len = args.win_size
        self.total_len = len(self.files) * self.lines_per_file
        self.file_index = 0
        self.line_index = 0

    def __len__(self):
        return self.total_len

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(f'{self.files[self.cur_file_idx]}', 'r') as f:
            self.file_texts = f.readlines()

    def __getitem__(self, index):
        if self.line_index >= len(self.file_texts):
            self._load_file(self.file_index + 1)
            self.file_index += 1
            self.line_index = 0
        text = self.file_texts[self.line_index].replace('<|', '').replace('|>', '')
        text = json.loads(json.loads(text))

        self.line_index += 1

        tokens = self.tokenizer.encode(text) + [self.tokenizer.eot_token]
        source, target = tokens[:self.max_seq_len], tokens[1:self.max_seq_len + 1]
        if self.max_seq_len != -1:
            source += [self.tokenizer._special_tokens['<|padding|>']] * (self.max_seq_len - len(source))
            target += [self.tokenizer._special_tokens['<|padding|>']] * (self.max_seq_len - len(target))
        
        source = torch.tensor(source).long()
        target = torch.tensor(target).long()

        return source, target, 0

def find_files_qwen(root_dir):
    root_path = Path(root_dir)
    return [str(file) for file in root_path.rglob('local-shard*/*.jsonl')]

class DCLMDataset(Dataset):
    def __init__(self, dataset_dir, max_seq_len, tokenizer, padding = True) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.files = find_files_qwen(self.dataset_dir)
        self._load_file(0)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.lines_per_file = len(self.file_texts)
        self.padding = padding

    def __len__(self):
        return len(self.files) * self.lines_per_file

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(f'{self.files[self.cur_file_idx]}', 'r') as f:
            self.file_texts = f.readlines()

    def __getitem__(self, index):
        file_idx = index // self.lines_per_file
        if file_idx != self.cur_file_idx:
            self._load_file(file_idx)
        line_idx = index % self.lines_per_file
        if line_idx >= len(self.file_texts):
            # print(f'{index}, {line_idx}, {len(self.file_texts)}, {self.dataset_dir}/{self.files[self.cur_file_idx]}')
            line_idx = len(self.file_texts) - 1

        text = self.file_texts[line_idx]
        text = json.loads(json.loads(text))

        if self.padding:
            token_ids = self.tokenizer(text, padding='max_length', 
                                   max_length = self.max_seq_len + 1, padding_side='right', 
                                   truncation=True, return_tensors='pt').input_ids
            real_len = len(self.tokenizer(text).input_ids)
        else:
            token_ids = self.tokenizer(text, return_tensors='pt').input_ids
            real_len = len(token_ids[0])
        return token_ids[0, :-1], token_ids[0, 1:], real_len