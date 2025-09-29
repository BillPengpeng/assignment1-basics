import json
import time
import numpy as np

from tqdm import tqdm
import cProfile
import pstats
from memory_profiler import profile

import json
import pathlib
from functools import lru_cache
from tests.adapters import run_train_bpe, BPETokenizer, find_chunk_boundaries
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"

def calc(vocab_size, seq_len, num_layers, d_model, num_heads, d_ff):
    proj_flops = num_layers * (8*seq_len*d_model*d_model)
    qkv_flops = num_layers * (4*d_model*seq_len*seq_len)
    ffn_flops = num_layers * (6*seq_len*d_ff*d_model)
    norm_flops = 2*seq_len*vocab_size*d_model
    total_flops = proj_flops + qkv_flops + ffn_flops + norm_flops
    result = dict()
    result['total_flops'] = total_flops
    result['proj_flops'] = [proj_flops, proj_flops / total_flops]
    result['qkv_flops'] = [qkv_flops, qkv_flops / total_flops]
    result['ffn_flops'] = [ffn_flops, ffn_flops / total_flops]
    return result

#  num_layers * (8*seq_len*d_model^2 + 4*d_model*seq_len^2 + 6*seq*d_ff*d_model) + 2*seq_len*vocab_size*d_model 
#       = 48 * (8*1024*1600*1600 + 4*1600*1024*1024 + 6*1024*6400*1600) + 2*1024*50257*1600

if __name__ == "__main__":
    vocab_size = 50257
    seq_len = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25
    d_ff = 6400
    result = calc(vocab_size, seq_len, num_layers, d_model, num_heads, d_ff)
    print(result)

    num_layers = 12
    d_model = 768
    num_heads = 12
    d_ff = d_model * 8 // 3
    result = calc(vocab_size, seq_len, num_layers, d_model, num_heads, d_ff)
    print(result)

    num_layers = 24
    d_model = 1024
    num_heads = 16
    d_ff = d_model * 8 // 3
    result = calc(vocab_size, seq_len, num_layers, d_model, num_heads, d_ff)
    print(result)

    num_layers = 36
    d_model = 1280
    num_heads = 20
    d_ff = d_model * 8 // 3
    result = calc(vocab_size, seq_len, num_layers, d_model, num_heads, d_ff)
    print(result)

