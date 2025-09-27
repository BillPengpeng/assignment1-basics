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