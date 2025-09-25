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
    
class rope(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, 
                 device: torch.device | None=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 关键：初始化一个None缓冲区，persistent=False
        # 我们不在这里计算，而是延迟到第一次前向传播时计算（更高效）
        self.register_buffer("freqs", None, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

        # 计算
        t = torch.arange(max_seq_len, device=device)
        freps = 1.0 / (theta**(torch.arange(start=0, end=d_k, step=2, device=device) / d_k))
        self.freqs = einsum(t, freps, "t_len, freps_len -> t_len freps_len")
        self.cos_cached = self.freqs.cos()
        self.sin_cached = self.freqs.sin()


    def forward(self, x:torch.Tensor, token_positions:torch.Tensor)->torch.Tensor:
        # # 1. 将输入x的最后一维拆分为二维向量对
        # # [..., d_model] -> [..., d_model//2, 2]
        # # x_shaped = x.view(*x.shape[:-1], -1, 2)
        # x_shaped = rearrange(x, '... (d_model_div2 two) -> ... d_model_div2 two', two=2)

        # # 2. 分离出x1和x2，即每个向量对的两个元素
        # x1 = x_shaped[..., 0] # 对应所有a
        # x2 = x_shaped[..., 1] # 对应所有b
        x1, x2 = rearrange(x, '... (d two) -> two ... d', two=2)

        # 3. 根据当前序列位置，获取对应的cos和sin值
        # cos_vals 形状 (seq_len, d_model//2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 4. 应用旋转公式（这步就是在“填充”和计算！）
        x1_rotated = einsum(x1, cos, "... seq_len d_model_div2, seq_len d_model_div2 -> ... seq_len d_model_div2") - \
                     einsum(x2, sin, "... seq_len d_model_div2, seq_len d_model_div2 -> ... seq_len d_model_div2")
        x2_rotated = einsum(x1, sin, "... seq_len d_model_div2, seq_len d_model_div2 -> ... seq_len d_model_div2") + \
                     einsum(x2, cos, "... seq_len d_model_div2, seq_len d_model_div2 -> ... seq_len d_model_div2")

        # # 5. 将旋转后的结果拼接回去
        # x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1)
        # # print(x1_rotated.shape, x2_rotated.shape, x_rotated.shape)

        # # 6. 展平回原始形状
        # y = rearrange(x_rotated, "... seq_len d_model_div2 two -> ... seq_len (d_model_div2 two)")
        y = rearrange([x1_rotated, x2_rotated], 'two ... d_model_div2  -> ... (d_model_div2 two)')

        return y