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

from cs336_basics.module import linear, embedding, rmsnorm, silu, swiglu, rope
from cs336_basics.module import softmax, softmax_func, scaled_dot_product_attention_func
from cs336_basics.module import causal_multihead_self_attention, transformer_block
from cs336_basics.module import transformer_lm
from cs336_basics.optim import cross_entropy_func, AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.data import DataLoader
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint

# device
device_str = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device_str)

# cudnn
torch.backends.cudnn.benchmark = True 
torch.backends.cudnn.enabled = True 

def parse_args():
    parser = argparse.ArgumentParser(description='Convert list of images to COCO format JSON')
    parser.add_argument('--yaml_path', type=str, help='yaml path')
    return parser.parse_args()

def load_config(yaml_path: str):
    """从YAML文件加载配置"""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def build_model(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float
):
    model = transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=device)
    return model

def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3, 
    weight_decay: float = 0.01, 
    betas: tuple = (0.9, 0.999), 
    eps: float = 1e-8
):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    return optimizer

def get_numpy_shape(path):
    """安全获取numpy数组的真实形状"""
    with open(path, 'rb') as f:
        # 只读取头部信息，不加载全部数据
        version = np.lib.format.read_magic(f)
        shape, dtype, _ = np.lib.format.read_array_header_1_0(f)
    return shape, dtype

def build_dataset(
    path: str,
    dtype: np.dtype,
    expected_shape: tuple | None = None
):
    assert os.path.exists(path)
    # file_size = os.path.getsize(path)
    # itemsize = np.dtype(dtype).itemsize
    
    # if expected_shape:
    #     # 验证文件大小是否匹配预期shape
    #     expected_size = np.prod(expected_shape) * itemsize
    #     if file_size != expected_size:
    #         raise ValueError(f"文件大小不匹配: 预期 {expected_size}, 实际 {file_size}")
    #     return np.memmap(path, dtype=dtype, mode='r', shape=expected_shape)
    # else:
    #     M = np.load(path)
    #     # 自动计算1D数组的shape
    #     shape, _ = get_numpy_shape(path)
    #     import pdb;pdb.set_trace()
        # return np.memmap(path, dtype=dtype, mode='r', shape=shape)
    return np.load(path, mmap_mode='r')

def lr_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    iter: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
): 
    lr = get_lr_cosine_schedule(iter, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def calc_num_iters(
    dataset_size: int,
    batchsize: int,
    context_length: int
):
    num_iters = (dataset_size - context_length) // batchsize
    return num_iters

def eval(
    valid_dataset: np,
    model: nn.Module,
    batchsize: int,
    context_length: int,
    print_freq: int,
    eval_iters: int
):
    valid_loader = DataLoader() #random=False)
    dataset_size = valid_dataset.shape[0]
    # num_iters = (dataset_size - context_length + batchsize - 1) // batchsize
    num_iters = calc_num_iters(dataset_size, batchsize, context_length)
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for iter in range(1, eval_iters + 1):
            data, labels = valid_loader.get_batch(valid_dataset, batchsize, context_length, device_str)
            pred = model(data)
            loss = cross_entropy_func(pred, labels, reduction = 'sum')      
            non_padding_mask = (labels >= 0)  # 假设-100是padding值
            num_tokens = non_padding_mask.sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens
            if (iter % print_freq == 0) or (iter == num_iters):
                average_loss = total_loss / total_tokens
                perplexity = torch.exp(torch.tensor(average_loss))
                logging.info("[Eval] iter: {}/{} average_loss: {} total_tokens: {} perplexity: {}".format(iter, num_iters, average_loss, total_tokens, perplexity))


def train(
    train_dataset: np,
    valid_dataset: np,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    max_l2_norm: float,
    start_iter: int, 
    num_iters: int,
    num_iters_per_epoch: int,
    batchsize: int,
    context_length: int,
    print_freq: int,
    save_freq: int,
    eval_freq: int,
    eval_iters: int,
    work_dir: str
):
    train_loader = DataLoader()
    for iter in range(start_iter + 1, num_iters + 1):
        optimizer.zero_grad()
        if iter % num_iters_per_epoch == 0:
            train_loader.dataset_idx = 0
        data, labels = train_loader.get_batch(train_dataset, batchsize, context_length, device_str)
        pred = model(data)
        lr = lr_cosine_schedule(optimizer, iter, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        loss = cross_entropy_func(pred, labels)      
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm)
        optimizer.step()
        if iter % print_freq == 0:
            logging.info("[Train] iter: {}/{} lr: {} loss: {}".format(iter, num_iters, lr, loss))
        if iter % eval_freq == 0:
            eval(valid_dataset, model, batchsize, context_length, print_freq, eval_iters)
            model.train()
        if iter % save_freq == 0:
            dst_path = os.path.join(work_dir, "iter_{}.pth".format(iter))
            save_checkpoint(model, optimizer, iter, dst_path)
        # break
    

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.yaml_path)
    # workdirs
    if not os.path.exists(cfg['workdir']):
        os.makedirs(cfg['workdir'])

    # logger
    log_path = os.path.join(cfg['workdir'], 'log.txt')
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # dataset
    train_dataset = build_dataset(cfg['dataset']['train'], np.uint16)
    valid_dataset = build_dataset(cfg['dataset']['valid'], np.uint16)
    logging.info("train_dataset: {} shape: {}".format(cfg['dataset']['train'], train_dataset.shape))
    logging.info("valid_dataset: {} shape: {}".format(cfg['dataset']['valid'], valid_dataset.shape))

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
    logging.info("create model num_layers: {} num_heads: {}".format(cfg['model']['num_layers'], cfg['model']['num_heads']))

    # optimizer 
    optimizer = build_optimizer(model, lr=cfg['scheduler']['max_learning_rate'], weight_decay=cfg['scheduler']['weight_decay'])
    dataset_size = train_dataset.shape[0]
    num_iters_per_epoch = calc_num_iters(dataset_size, cfg['scheduler']['batchsize'], cfg['model']['seq_len'])
    assert 'total_tokens' in cfg['scheduler'].keys() or 'num_epochs' in cfg['scheduler'].keys()
    if 'total_tokens' in cfg['scheduler'].keys():
        num_tokens_per_iter = cfg['scheduler']['batchsize'] * cfg['model']['seq_len']
        num_iters = (cfg['scheduler']['total_tokens'] + num_tokens_per_iter - 1) // num_tokens_per_iter
        logging.info("num_tokens_per_iter: {} total_tokens: {} num_iters: {}".format(num_tokens_per_iter, cfg['scheduler']['total_tokens'], num_iters))
    elif 'num_epochs' in cfg['scheduler'].keys():
        num_iters = int(cfg['scheduler']['num_epochs'] * num_iters_per_epoch)
        logging.info("num_iters_per_epoch: {} num_epochs: {} num_iters: {}".format(num_iters_per_epoch, cfg['scheduler']['num_epochs'], num_iters))

    warmup_iters = cfg['scheduler']['warmup_iters']
    cosine_cycle_iters = int(cfg['scheduler']['cosine_cycle_iter_ratio'] * num_iters)
    logging.info("create optimizer num_iters: {} warmup_iters: {} cosine_cycle_iters: {}".format(num_iters, warmup_iters, cosine_cycle_iters))

    # load_checkpoint
    if 'resume' in cfg['model'].keys():
        start_iter = load_checkpoint(cfg['model']['resume'], model, optimizer)
        logging.info("load_checkpoint: {} start_iter: {}".format(cfg['model']['resume'], start_iter))
    else:
        start_iter = 0

    # train
    train(
        train_dataset,
        valid_dataset,
        model,
        optimizer,
        cfg['scheduler']['max_learning_rate'],
        cfg['scheduler']['min_learning_rate'],
        warmup_iters,
        cosine_cycle_iters,
        cfg['scheduler']['max_l2_norm'],
        start_iter,
        num_iters,
        num_iters_per_epoch,
        cfg['scheduler']['batchsize'],
        cfg['model']['seq_len'],
        cfg['print_freq'],
        cfg['save_freq'],
        cfg['eval_freq'],
        cfg['eval_iters'],
        cfg['workdir']
    )
    # import pdb;pdb.set_trace()

    
