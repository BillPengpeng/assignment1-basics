import math
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
        self.g = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device)) # 初始化为1
    
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
        self.register_buffer("inv_freq", None, persistent=False)
        self.register_buffer("freqs", None, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

        # 计算
        t = torch.arange(max_seq_len, device=device)
        self.inv_freq = 1.0 / (theta**(torch.arange(start=0, end=d_k, step=2, device=device) / d_k))
        # self.freqs = einsum(t, freq, "t_len, freps_len -> t_len freps_len")
        self.freqs = torch.outer(t, self.inv_freq)
        # import pdb;pdb.set_trace()
        self.cos_cached = self.freqs.cos()
        self.sin_cached = self.freqs.sin()


    def forward(self, x:torch.Tensor, token_positions:torch.Tensor, conj:bool=False)->torch.Tensor:
        # # 2. 分离出x1和x2，即每个向量对的两个元素
        x1, x2 = rearrange(x, '... (d two) -> two ... d', two=2)

        # 3. 根据当前序列位置，获取对应的cos和sin值
        # cos_vals 形状 (seq_len, d_model//2)
        # 20250925 add torch.squeeze
        seq_len = x.shape[-2]
        with torch.no_grad():
            if token_positions is not None:
                cos = self.cos_cached[token_positions]
                sin = self.sin_cached[token_positions]
            else:
                cos = self.cos_cached[:seq_len, ]
                sin = self.sin_cached[:seq_len, ]

            cos = torch.squeeze(cos)
            sin = torch.squeeze(sin)

        # 4. 应用旋转公式（这步就是在“填充”和计算！）
        x1_rotated = einsum(x1, cos, "... seq_len d_model_div2, seq_len d_model_div2 -> ... seq_len d_model_div2") - \
                     einsum(x2, sin, "... seq_len d_model_div2, seq_len d_model_div2 -> ... seq_len d_model_div2")
        x2_rotated =   einsum(x1, sin, "... seq_len d_model_div2, seq_len d_model_div2 -> ... seq_len d_model_div2") + \
                       einsum(x2, cos, "... seq_len d_model_div2, seq_len d_model_div2 -> ... seq_len d_model_div2")

        # # 6. 展平回原始形状
        y = rearrange([x1_rotated, x2_rotated], 'two ... d_model_div2  -> ... (d_model_div2 two)')
        return y

        
        # seq_len = x.shape[-2]
        # # 将x视为复数 (实部x1,虚部x2)
        # x = rearrange(x, '... (d two) -> ... d two', two=2)
        # x_complex = torch.view_as_complex(x.contiguous())

        # # 获取对应位置的旋转角度 [seq_len,dim//2]
        # with torch.no_grad():
        #     if token_positions is not None:
        #         angles = self.freqs[token_positions,]
        #     else:
        #         angles = self.freqs[:seq_len, ]

        # # 构造复数旋转因子 e^{i*theta}
        # rot_factor = torch.polar(
        #     torch.ones_like(angles),  # 模为1
        #     angles                   # 角度
        # )  # [seq_len,dim//2]

        # # 复数乘法旋转
        # x_rotated = x_complex * rot_factor
        
        # # 转换回实数表示
        # return torch.view_as_real(x_rotated).flatten(-2)
        

def softmax_func(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    max_val, _ = torch.max(in_features, dim=dim, keepdim=True)
    x1 = torch.exp(in_features - max_val)
    x2 = torch.sum(x1, dim=dim, keepdim=True)
    y = x1 / x2
    return y

class softmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, in_features: torch.Tensor, dim: int) -> torch.Tensor:
        return softmax_func(in_features, dim)

def scaled_dot_product_attention_func(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                      mask: torch.Tensor | None=None) -> torch.Tensor:
    d_k = Q.shape[-1]
    QK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        QK = QK.masked_fill(mask == 0, float("-inf"))
    y = softmax_func(QK, dim=-1)
    y = einsum(y, V, "... queries keys, ... keys d_v -> ... queries d_v") 
    return y

class causal_multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 max_seq_len: int | None=None,  
                 theta: float | None=None, 
                 device: torch.device | None=None, 
                 dtype: torch.dtype| None=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_weight = nn.Parameter(torch.empty(3 * d_model, d_model, dtype=dtype, device=device))
        self.o_weight = nn.Parameter(torch.empty(self.d_model, self.d_model, dtype=dtype, device=device))
        std = (2.0 / (4 * self.head_dim)) ** 0.5
        nn.init.trunc_normal_(self.qkv_weight, mean=0, std=std, a=-3*std, b=3*std)
        std = (2.0 / (2 * self.d_model)) ** 0.5
        nn.init.trunc_normal_(self.o_weight, mean=0, std=std, a=-3*std, b=3*std)

        # rope
        if max_seq_len is not None and theta is not None:
            self.rope = rope(theta, self.head_dim, max_seq_len, device)
        else:
            self.rope = None

    def create_causal_mask(self, seq_len: int, device: torch.device = None):
        row_indices = torch.arange(seq_len, device=device)
        col_indices = torch.arange(seq_len, device=device)
        mask = row_indices.unsqueeze(1) >= col_indices
        return mask


    def forward(self, in_features: torch.Tensor,
                token_positions:torch.Tensor | None=None) -> torch.Tensor:
        seq_len = in_features.shape[-2]
        qkv = einsum(self.qkv_weight, in_features, "three_out_dim out_dim, ... seq_len out_dim -> ... seq_len three_out_dim")
        q, k, v = rearrange(qkv, "... seq_len (three num_heads head_dim) -> three ... num_heads seq_len head_dim", three=3, num_heads=self.num_heads)
        if (self.rope is not None):
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        mask = self.create_causal_mask(seq_len, in_features.device)
        out = scaled_dot_product_attention_func(q, k, v, mask)
        out = rearrange(out, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")
        out = einsum(self.o_weight, out, "out_dim in_dim, ... seq_len in_dim -> ... seq_len out_dim")
        return out

class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int | None=None,  
                 theta: float | None=None, 
                 device: torch.device | None=None, 
                 dtype: torch.dtype| None=None):
        super().__init__()
        self.ln1 = rmsnorm(d_model, device=device, dtype=dtype)
        self.ln2 = rmsnorm(d_model, device=device, dtype=dtype)
        self.attention = causal_multihead_self_attention(d_model, num_heads, max_seq_len=max_seq_len, \
                                                         theta=theta, device=device, dtype=dtype)
        self.ffn = swiglu(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, in_features: torch.Tensor):
        y1 = self.attention(self.ln1(in_features))
        y2 = in_features + y1
        y3 = self.ffn(self.ln2(y2))
        y4 = y2 + y3
        return y4

