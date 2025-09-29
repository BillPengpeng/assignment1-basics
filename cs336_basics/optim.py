import math
import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange,einsum
from collections.abc import Callable, Iterable
from cs336_basics.module import softmax_func

def cross_entropy_func(pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # num_classes = pred.shape[-1]
    # probs = softmax_func(pred, dim=-1)
    # one_hot = torch.zeros(targets.size(0), num_classes, dtype=torch.float32)
    # one_hot.scatter_(1, targets.unsqueeze(1), 1)
    # y = torch.sum(probs * one_hot, dim=-1)
    # y = -torch.mean(torch.log(y))

    num_classes = pred.shape[-1]
    max_val, _ = torch.max(pred, dim=-1, keepdim=True)
    exp_val = torch.exp(pred - max_val)
    sum_val = torch.sum(exp_val, dim=-1, keepdim=True)

    one_hot = torch.zeros(targets.size(0), num_classes, dtype=torch.float32)
    one_hot.scatter_(1, targets.unsqueeze(1), 1)
    pred_sum = torch.sum(pred * one_hot, dim=-1)
    y = -torch.mean(pred_sum - torch.log(sum_val) - max_val)
    return y

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data-= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.      
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # lr = group["lr"] # Get the learning rate.
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                # p.data-= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.

                if 'first_order' not in state.keys():
                    state['first_order'] = (1 - betas[0]) * grad
                else:
                    # import pdb;pdb.set_trace()
                    state['first_order'] = betas[0]*state['first_order'] + (1 - betas[0]) * grad
                first_order = state['first_order']

                if 'second_order' not in state.keys():
                    state['second_order'] = (1 - betas[1]) * grad * grad
                else:
                    state['second_order'] = betas[1]*state['second_order'] + (1 - betas[1]) * grad * grad
                second_order = state['second_order']

                lr = group["lr"] * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)
                # import pdb;pdb.set_trace()
                p.data -= lr * first_order / (torch.sqrt(second_order) + eps)
                p.data -= group["lr"] * weight_decay * p.data
                state["t"] = t + 1 # Increment iteration number.      
        return loss

def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # 收集所有参数的梯度
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad)
    
    if not gradients:
        return  # 如果没有梯度，直接返回
    
    # 计算所有梯度的L2范数（使用PyTorch内置函数）
    total_norm = torch.norm(
        torch.stack([torch.norm(grad) for grad in gradients]), 
        p=2
    )
    
    # 如果范数超过最大值，进行裁剪
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for grad in gradients:
            grad.mul_(clip_coef)  # 原地修改梯度值