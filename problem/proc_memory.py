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
    params_memory = num_layers * (4*d_model^2 + 2*d_model + 3*d_model*d_ff) + d_model + d_model*vocab_size
    act_memory = num_layers * (seq_len*3*d_model + num_heads*seq_len*seq_len + seq_len*d_model + seq_len*d_model) + num_layers * (2*seq_len*d_ff + 2*seq_len*d_model) + seq_len*d_model + seq_len*vocab_size + seq_len*vocab_size
    result = dict()
    result['params_memory'] = params_memory #* 4 / (1024 * 1024 * 1024)
    result['act_memory'] = act_memory * 4 / (1024 * 1024 * 1024)

    proj_flops = num_layers * (8*seq_len*d_model*d_model)
    qkv_flops = num_layers * (4*d_model*seq_len*seq_len)
    ffn_flops = num_layers * (6*seq_len*d_ff*d_model)
    norm_flops = 2*seq_len*vocab_size*d_model
    total_flops = proj_flops + qkv_flops + ffn_flops + norm_flops
    adam_flops = 10*result['params_memory']
    result['total_flops'] = total_flops / 10e12
    result['adam_flops'] = adam_flops / 10e12
    result['total_flops_v2'] = 6 * params_memory * seq_len / 10e12
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


