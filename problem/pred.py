import os
import sys
import json
import time
import logging
import numpy as np
import yaml
import argparse

import random
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from cs336_basics.module import linear, embedding, rmsnorm, silu, swiglu, rope
from cs336_basics.module import softmax, softmax_func, scaled_dot_product_attention_func
from cs336_basics.module import causal_multihead_self_attention, transformer_block
from cs336_basics.module import transformer_lm
from cs336_basics.optim import cross_entropy_func, AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.data import DataLoader
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint

# tests
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

# train
from problem.train import load_config, build_model, build_optimizer

# device
device_str = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device_str)


def apply_top_p_sampling(logits: torch.Tensor, top_p: float):
    """
    应用Top-p（核）采样
    
    Args:
        logits: 模型输出的logits [batch_size, vocab_size]
        top_p: 累积概率阈值
    
    Returns:
        filtered_logits: 过滤后的logits
    """
    # 计算概率分布
    # probs = F.softmax(logits, dim=-1)
    probs = softmax_func(logits, dim=-1)
    
    # 对概率进行排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 创建掩码：移除累积概率超过top_p的token
    mask = cumulative_probs <= top_p
    
    # 确保至少选择一个token
    if not mask.any():
        mask[0] = True
    
    # 将掩码应用到原始索引
    mask = mask.scatter(1, sorted_indices, mask)
    
    # 过滤logits：将不在top-p中的token设为负无穷
    filtered_logits = logits.clone()
    filtered_logits[~mask] = -float('Inf')
    
    return filtered_logits

def decode_from_lm(
    model,  # 您的语言模型
    tokenizer,  # 分词器
    prompt: str,  # 用户提供的提示文本
    max_new_tokens: int = 50,  # 最大生成词元数
    temperature: float = 1.0,  # 温度参数
    top_p: float = None,  # Top-p采样阈值
    eos_token: str = "<|endoftext|>",  # 结束标记
):
    # 设置模型为评估模式
    model.eval()
    
    # 编码提示文本
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(np.array(input_ids).reshape(1, -1), dtype=torch.int32, device=device)
    generated_ids = input_ids.clone()
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    # print("generated_text:", generated_text, generated_ids)
    
    # 开始生成循环
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # 前向传播（只关注最后一个词元的输出）
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :] if hasattr(outputs, 'logits') else outputs[:, -1, :]
            
            # 应用温度缩放
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 应用Top-p采样（如果指定）
            if top_p is not None and top_p < 1.0:
                next_token_logits = apply_top_p_sampling(next_token_logits, top_p)
            
            # 计算概率分布
            # probs = F.softmax(next_token_logits, dim=-1)
            probs = softmax_func(next_token_logits, dim=-1)
            
            # 从分布中采样下一个词元
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 检查是否生成结束标记
            if next_token.item() in tokenizer.special_tokens_dict.values():
                break
            
            # 将新词元添加到序列中
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # 如果输入序列太长，可以截断（可选）
            # if input_ids.size(-1) > model.config.max_position_embeddings:
            #     input_ids = input_ids[:, -model.config.max_position_embeddings:]
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text

def parse_args():
    parser = argparse.ArgumentParser(description='Convert list of images to COCO format JSON')
    parser.add_argument('--yaml_path', type=str, help='yaml path')
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--prompt', type=str, help='prompt')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.yaml_path)
    # workdirs
    if not os.path.exists(cfg['workdir']):
        os.makedirs(cfg['workdir'])

    # model
    model = build_model(
        cfg['dataset']['vocab_size'],
        cfg['model']['seq_len'],
        cfg['model']['d_model'],
        cfg['model']['num_layers'],
        cfg['model']['num_heads'],
        cfg['model']['d_ff'],
        cfg['model']['rope_theta']
    )

    # tokenizer
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=cfg['dataset']['vocab_path'],
        merges_path=cfg['dataset']['merges_path'],
    )

    # load_checkpoint
    assert os.path.exists(args.model_path)
    optimizer = build_optimizer(model, cfg['scheduler']['max_learning_rate'])
    _ = load_checkpoint(args.model_path, model, optimizer)

    # decode_from_lm
    result_str = decode_from_lm(model, tokenizer, args.prompt)
    print(result_str)


    


    
