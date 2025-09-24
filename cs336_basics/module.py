import torch
import torch.nn as nn
from einops import rearrange,einsum

class linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 device: torch.device | None=None, 
                 dtype: torch.dtype| None=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
        return y

class embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 device: torch.device | None=None, 
                 dtype: torch.dtype| None=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        y = self.weight[token_ids]
        return y

class rmsnorm(nn.Module): 
    def __init__(self, d_model: int, eps: float = 1e-5,
                 device: torch.device | None=None, 
                 dtype: torch.dtype| None=None):
        super().__init__()
        self.eps = eps
        # 增益参数 g 被定义为形状为 (d_model,) 的可学习参数
        self.g = nn.Parameter(torch.ones(d_model)) # 初始化为1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        input = x.type(torch.float32)
        rms = torch.sqrt(torch.mean(input.pow(2), dim=-1, keepdim=True) + self.eps)
        y = input / rms * self.g
        y = y.type(dtype)
        return y

class silu(nn.Module): 
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * torch.sigmoid(x)
        return y