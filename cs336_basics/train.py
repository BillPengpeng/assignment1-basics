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

                if 'first_order' not in self.state.keys():
                    self.state['first_order'] = (1 - betas[0]) * grad
                else:
                    self.state['first_order'] = betas[0]*self.state['first_order'] + (1 - betas[0]) * grad
                first_order = self.state['first_order']

                if 'second_order' not in self.state.keys():
                    self.state['second_order'] = (1 - betas[1]) * grad * grad
                else:
                    self.state['second_order'] = betas[1]*self.state['second_order'] + (1 - betas[1]) * grad * grad
                second_order = self.state['second_order']

                lr = group["lr"] * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)
                # import pdb;pdb.set_trace()
                p.data -= lr * first_order / (torch.sqrt(second_order) + eps)
                p.data -= group["lr"] * weight_decay * p.data
                state["t"] = t + 1 # Increment iteration number.      
        return loss