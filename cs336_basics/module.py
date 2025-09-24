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

class swiglu(nn.Module): 
    def __init__(self, d_model: int, d_ff: int,
                 device: torch.device | None=None, 
                 dtype: torch.dtype| None=None):
        super().__init__()
        self.act = silu()
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, dtype=dtype, device=device))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, dtype=dtype, device=device))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, dtype=dtype, device=device))
        std = (2.0 / (d_model + d_ff)) ** 0.5
        nn.init.trunc_normal_(self.w1, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w2, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w3, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape, self.w1.shape, self.w2.shape)
        x1 = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        x3 = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        y = self.act(x1) * x3
        y = einsum(self.w2, y, "d_model d_ff, ... d_ff -> ... d_model")
        return y