from __future__ import annotations

import heapq
import os
import regex
import time, json
from collections import defaultdict
# 20250919 add deque
from collections import deque
# 20250916 add ABC, dataclass
from abc import ABC
from dataclasses import dataclass

# 20250918 add partial, multiprocessing
from functools import partial
import multiprocessing
from multiprocessing import Pool, cpu_count
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

# 20250921
from tests.common import gpt2_bytes_to_unicode

# 20250923
from einops import rearrange,einsum
from cs336_basics.module import linear, embedding, rmsnorm, silu, swiglu, rope
from cs336_basics.module import softmax, softmax_func, scaled_dot_product_attention_func
from cs336_basics.module import causal_multihead_self_attention

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    model = linear(d_in, d_out)
    with torch.no_grad():
        model.weight.copy_(weights)
    y = model(in_features)
    return y

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    model = embedding(vocab_size, d_model)
    with torch.no_grad():
        model.weight.copy_(weights)
    y = model(token_ids)
    return y

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    # raise NotImplementedError
    model = swiglu(d_model, d_ff)
    with torch.no_grad():
        model.w1.copy_(w1_weight)
        model.w2.copy_(w2_weight)
        model.w3.copy_(w3_weight)
    y = model(in_features)
    return y


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return scaled_dot_product_attention_func(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    model = causal_multihead_self_attention(d_model, num_heads)
    with torch.no_grad():
        model.qkv_weight[:d_model,].copy_(q_proj_weight)
        model.qkv_weight[d_model:2*d_model,].copy_(k_proj_weight)
        model.qkv_weight[2*d_model:3*d_model,].copy_(v_proj_weight)
        model.o_weight.copy_(o_proj_weight)
    y = model(in_features)
    return y


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    model = causal_multihead_self_attention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta)
    with torch.no_grad():
        model.qkv_weight[:d_model,].copy_(q_proj_weight)
        model.qkv_weight[d_model:2*d_model,].copy_(k_proj_weight)
        model.qkv_weight[2*d_model:3*d_model,].copy_(v_proj_weight)
        model.o_weight.copy_(o_proj_weight)
    y = model(in_features, token_positions=token_positions)
    return y


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    model = rope(theta, d_k, max_seq_len)
    y = model(in_query_or_key, token_positions)
    return y


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    model = rmsnorm(d_model)
    with torch.no_grad():
        model.g.copy_(weights)
    y = model(in_features)
    return y



def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    model = silu()
    y = model(in_features)
    return y


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    # model = softmax()
    # y = model(in_features, dim)
    # return y
    return softmax_func(in_features, dim)



def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError

GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    # i = 0  # @inspect i
    # if pair[0] not in indices or pair[1] not in indices:
    #     return indices

    # indices_len = len(indices)
    # while i < indices_len - 1:
    #     if indices[i] == pair[0] and indices[i + 1] == pair[1]:
    #         new_indices.append(new_index)
    #         i += 2
    #     else:
    #         new_indices.append(indices[i])
    #         i += 1
    # if i < indices_len: 
    #     new_indices.append(indices[i])
    # return new_indices

    # 直接返回
    # if pair[0] not in indices or pair[1] not in indices:
    #     return
    
    # 原地修改
    indices_len = len(indices)
    
    i = 0
    while i < indices_len - 1:
        # match success
        if indices[i] == pair[0] and indices[i + 1] == pair[1]:
            indices[i] = new_index
            del indices[i + 1]
            indices_len -= 1
        i += 1

    # 删除结尾
    del indices[indices_len:]
    return indices_len

def process_segment_shared(
    segment: str,
    shared_dict: multiprocessing.managers.DictProxy,
    # merges: Dict[Tuple[int, int], int],
    pair: tuple[int, int], 
    new_index: int,
    special_tokens: Dict[str, int]
):
    """处理单个segment（共享内存版本）"""
    if segment in special_tokens.keys():
        return

    induces = shared_dict[segment]
    if len(induces) <= 1:
        return
    induces = merge(induces, pair, new_index)

    # print(segment, induces)
    # for pair, new_index in merges.items():
    #     try:
    #         print(induces, segment)
    #         if len(induces) <= 1:
    #             break
    #         induces = merge(induces, pair, new_index)
            
    #     except:
            
    #         import pdb;pdb.set_trace()
    
    # # 更新共享字典
    # shared_dict[segment] = induces

def extract_segments_before_special_tokens(input_str: str, special_tokens: list[str], filter=True) -> list[str]:
    """
    删除输入字符串中的特殊标记，并提取删除后的前两段文本。
    
    Args:
        input_str: 待处理的输入字符串（如 "hello [PAD] world [UNK]"）。
        special_tokens: 需要删除的特殊标记列表（如 ["[PAD]", "[UNK]"]）。
    
    Returns:
        删除特殊标记后的前两段文本（如 ["hello ", " world "]）。
    """
    # 处理空标记列表（直接返回原字符串的前两段，按空格分割）
    if not special_tokens:
        return [input_str]  # 按空格分割，取前两段（可根据需求调整分割逻辑）

    split_parts = [input_str] 
    # aim to ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    special_tokens = sorted(special_tokens, key=lambda s: (-len(s), s))
    for token in special_tokens:
        new_split_parts = list()
        for parts in split_parts:
            if token not in parts or (parts in special_tokens):
                new_split_parts.append(parts)
            else:
                # import pdb;pdb.set_trace()
                parts = parts.split(token)
                for part in parts:
                    if len(part) > 0:
                        new_split_parts.append(part)
                    new_split_parts.append(token)
                    # import pdb;pdb.set_trace()
                del new_split_parts[-1]
        split_parts = new_split_parts

    desired_num_chunks = 4
    file_size = len(input_str)
    chunk_size = file_size // desired_num_chunks

    if not filter:
        return split_parts
    
    # 提取所有非标记的文本部分（过滤掉特殊标记）
    non_marker_parts = [part for part in split_parts if part not in special_tokens]
    
    # 返回前两段非标记文本（若不足两段，返回所有存在的部分）
    return non_marker_parts


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,  
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# @dataclass(frozen=True)是 Python 中用于定义不可变数据类的装饰器，确保类的实例属性在初始化后无法被修改
@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index
    special_tokens: list[str]

# 抽象基类（ABC）是 Python 中通过 abc模块实现的特殊类，​​不能被直接实例化​​，只能被继承
class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # init
        self.vocab = vocab
        self.merges_dict: dict[tuple[int, int], int] = dict()
        self.special_tokens = special_tokens
        vocab_index_dict = {v: k for k, v in vocab.items()}
        for byte1, byte2 in merges:
            find_index_byte1 = vocab_index_dict[byte1]
            find_index_byte2 = vocab_index_dict[byte2]
            find_index_merge = vocab_index_dict[byte1+byte2]
            self.merges_dict[(find_index_byte1, find_index_byte2)] = find_index_merge

        # special_tokens
        self.special_tokens_dict = dict()
        self.vocab_index_dict = {v: k for k, v in self.vocab.items()}
        if self.special_tokens is not None:
            for special_token in self.special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                self.special_tokens_dict[special_token] = self.vocab_index_dict[byte_encoded_special_token]


        # encode_iterable
        self.max_special_token_len = 0
        if self.special_tokens is not None:
            for special_token in self.special_tokens:
                self.max_special_token_len = len(special_token) if len(special_token) > self.max_special_token_len else 0

        # chunk_size
        self.chunk_size = 4096  # 4KB块

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):  
        # 反转 gpt2_bytes_to_unicode得到 gpt2_byte_decoder（Unicode 字符到字节的映射），用于后续将词表中的 Unicode 字符串还原为原始字节。
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        # vocab_path指向 GPT-2 格式的词表文件（如 vocab.json），内容为 JSON 对象，键是子词字符串（如 "hello"），值是对应的索引（如 123）
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        # 最终 vocab的键是原子词索引，值是原始字节（如 123: b'low'）
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        # 将合并规则中的 Unicode 子词对还原为原始字节对，与词表的字节级数据对齐
        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_tokens)
 

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        buffer_len = 0
        for text_chunk in iterable:
            buffer += text_chunk
            buffer_len += len(text_chunk)

            # special_token
            check_token_len = self.max_special_token_len
            if self.special_tokens is not None and buffer_len >= check_token_len:
                for special_token in self.special_tokens:
                    pos = buffer.find(special_token, -check_token_len, -1)
                    if pos > 0:
                        sub_buffer = buffer[:pos]
                        buffer = buffer[pos:]
                        buffer_len -= pos
                        yield from iter(self.encode(sub_buffer))

        # Last buffer
        if buffer_len > 0:
            # self.encode_buffer = ""
            yield from iter(self.encode(buffer))
        

    def encode(self, string: str) -> list[int]:
        # indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # 切割字符串
        split_content_list = extract_segments_before_special_tokens(string, self.special_tokens, filter=False)
        segments = list()
        for split_content in split_content_list:
            if split_content in self.special_tokens_dict.keys() or ("\n" not in split_content): 
                segments.append(split_content)
            else:
                # 将换行符替换为独立标记 <newline> 
                parts = split_content.split("\n")
                for part in parts:
                    if len(part) > 0:
                        segments.append(part)
                    segments.append("\n")
                del segments[-1]

        # 合并\n
        segments_size = len(segments)
        i = 1
        while i < segments_size:
            if segments[i - 1].endswith("\n") and segments[i] == "\n" and \
               (i == segments_size - 1 or segments[i + 1] == "\n"):
               segments[i - 1] += "\n"
               del segments[i]
               segments_size -= 1
            else:
                i += 1

        # str => utf-8 => int
        segment_indice_dict = dict()
        for segment in set(segments):
            if segment in self.special_tokens_dict.keys():
                segment_indice_dict[segment] = [self.special_tokens_dict[segment]]
            else:
                # segment_induce_dict[segment] = list(map(int, segment.encode("utf-8")))
                segment_indice_dict[segment] = list()
                # import pdb;pdb.set_trace()
                for ch in segment.encode("utf-8"):
                    # unicode => byte
                    cur_bytes = bytes([ch])
                    segment_indice_dict[segment].append(self.vocab_index_dict[cur_bytes])
                
        # merge
        for segment in segment_indice_dict.keys():
            if segment in self.special_tokens_dict.keys():
                continue

            indices = segment_indice_dict[segment]
            for pair, new_index in self.merges_dict.items():  # @inspect pair, @inspect new_index
                if len(indices) <= 1:
                    break
                 # 原地修改
                if pair[0] not in indices or pair[1] not in indices:
                    continue
                merge(indices, pair, new_index)

        # result
        result_indices = list()
        for segment in segments:
            result_indices += segment_indice_dict[segment]
        return result_indices

        # indices_generator = self._encode_generator(segments)
        # return list(indices_generator)  # 最终必须返回列表


    def _encode_generator(self, segments: list[int]):
        """生成器实现，逐步产生 token 索引"""

        for segment in segments:
            if segment in self.special_tokens_dict.keys():
                yield self.special_tokens_dict[segment]
            else:
                indices = list()
                for ch in segment.encode("utf-8"):
                    cur_bytes = bytes([ch])
                    cur_index = self.vocab_index_dict[cur_bytes]
                    indices.append(cur_index)

                # other
                for pair, new_index in self.merges.items():  # @inspect pair, @inspect new_index
                    if len(indices) <= 1:
                        break
                    # 原地修改
                    if pair[0] not in indices or pair[1] not in indices:
                        continue
                    merge(indices, pair, new_index)

                # 逐步产生索引
                for idx in indices:
                    yield idx

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8", errors='ignore')  # @inspect string
        return string

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return BPETokenizer(vocab, merges, special_tokens)
    # merges_dict: dict[tuple[int, int], int] = dict()
    # vocab_index_dict = {v: k for k, v in vocab.items()}

    # for byte1, byte2 in merges:
    #     find_index_byte1 = vocab_index_dict[byte1]
    #     find_index_byte2 = vocab_index_dict[byte2]
    #     find_index_merge = vocab_index_dict[byte1+byte2]
    #     merges_dict[(find_index_byte1, find_index_byte2)] = find_index_merge

    # return BPETokenizer(BPETokenizerParams(vocab, merges_dict, special_tokens))
    # raise NotImplementedError


def count_pretoken_frequency_by_chunks(input_path, boundaries, special_tokens):
    chunk_count_dict = defaultdict(int)
    with open(input_path, "rb") as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            content = f.read(end - start).decode("utf-8", errors="ignore")
            split_content_list = extract_segments_before_special_tokens(content, special_tokens)
            # chunk_count_dict
            pattern = GPT2_TOKENIZER_REGEX  # @inspect pattern
            segments = list()
            for split_content in split_content_list:
                segments += regex.findall(pattern, split_content)  # @inspect segments
            for segment in segments:
                chunk_count_dict[segment] += 1
    return chunk_count_dict

def calc_max_pair(segment_count_dict, segment_induce_dict, counts, vocab, last_max_pair=None, last_max_pair_new_index=None):
    max_count = 0
    max_pair = None
    # print("segment_count_dict.keys():", len(segment_count_dict.keys()))
    # time1 = time.time()
    for segment in segment_count_dict.keys():
        indices = segment_induce_dict[segment]

        # max_pair
        if last_max_pair is not None and \
            last_max_pair[0] not in indices and \
            last_max_pair[1] not in indices and \
            last_max_pair_new_index not in indices:
            continue

        # 相邻合并对
        indices_len = len(indices)
        for jdx in range(indices_len - 1):
            index1, index2 = indices[jdx], indices[jdx + 1]  
            if last_max_pair is not None and \
                index1 not in last_max_pair and \
                index2 not in last_max_pair and \
                index1 != last_max_pair_new_index and \
                index2 != last_max_pair_new_index:
                continue

            # update index_segment_dict
            cur_pair = (index1, index2)
            counts[cur_pair] += segment_count_dict[segment]
    # time2 = time.time()
    # print("calc_max_pair 1:", time2 - time1)
                
    # max_pair
    # print("counts.keys():", len(counts.keys()))
    for cur_pair in counts.keys():
        cur_count = counts[cur_pair]
        if cur_count < max_count or cur_count == 0:
            continue

        if cur_count > max_count:
            max_count = cur_count
            max_pair = cur_pair
            continue

        # cur_vocab
        index1, index2 = cur_pair[0], cur_pair[1] 
        # print(max_pair, cur_count, max_count)
        if (vocab[index1] > vocab[max_pair[0]]) or \
            (vocab[index1] == vocab[max_pair[0]] and vocab[index2] > vocab[max_pair[1]]):
            max_count = cur_count
            max_pair = cur_pair

    # time3 = time.time()
    # print("calc_max_pair 2:", time3 - time2)
    
    # reset count
    for cur_pair in counts.keys():
        if cur_pair[0] not in max_pair and \
            cur_pair[1] not in max_pair:
            continue
        counts[cur_pair] = 0

    # time4 = time.time()
    # print("calc_max_pair 3:", time4 - time3)
    
    return max_pair

class PairItem:
    def __init__(self, count, vocab1, vocab2, original_pair):
        self.count = count
        self.vocab1 = vocab1 # bytes
        self.vocab2 = vocab2 # bytes
        self.original_pair = original_pair

    # 定义比较规则，用于堆排序
    # 我们需要使【更大】的 PairItem 在比较中【更小】（因为是最小堆）
    def __lt__(self, other):
        # 比较规则与你的原始逻辑完全一致：
        if self.count != other.count:
            # 计数大的更“小”（因为我们要最大堆，计数大的应该在堆顶）
            return self.count > other.count
        # 计数相同，比较第一个词汇
        if self.vocab1 != other.vocab1:
            # 词汇字节串大的更“小”
            return self.vocab1 > other.vocab1
        # 前两个都相同，比较第二个词汇
        return self.vocab2 > other.vocab2

    def __eq__(self, other):
        return (self.count == other.count and 
                self.vocab1 == other.vocab1 and 
                self.vocab2 == other.vocab2)

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    merges: list[tuple[bytes, bytes]] = []
    
    # 合并special_tokens
    special_token_vocab_list = list()
    for special_token in special_tokens:
        # 编码为 UTF-8 字节（关键步骤）
        idx = len(vocab.keys())
        vocab[idx] = special_token.encode("utf-8")
        special_token_vocab_list.append(vocab[idx])

    # 切割字符串
    start_time = time.time()
    segment_count_dict = defaultdict(int)
    segment_induce_dict = dict()
    subproc_cnt = cpu_count() * 2 #
    print("subproc_cnt:", subproc_cnt)
    chunk_per_subproc = 5 # 100
    with open(input_path, "rb") as f:
        # 获取分块边界
        boundaries = find_chunk_boundaries(
            f, subproc_cnt * chunk_per_subproc, "<|endoftext|>".encode("utf-8")
        )
    end_time = time.time()
    print("find_chunk_boundaries:", end_time - start_time)

    start_time = time.time()
    with Pool() as pool:
        results = pool.starmap(
            count_pretoken_frequency_by_chunks,
            [
                (
                    input_path,
                    boundaries[i * chunk_per_subproc : (i + 1) * chunk_per_subproc + 1],
                    special_tokens
                )
                for i in range(subproc_cnt)
            ],
        )
    for res in results:
        for k,v in res.items():
            segment_count_dict[k] += v
    end_time = time.time()
    print("count_pretoken_frequency_by_chunks:", end_time - start_time)

      
    for segment in segment_count_dict.keys():
        segment_induce_dict[segment] = list(map(int, segment.encode("utf-8")))    
    # import pdb;pdb.set_trace()

    '''
    # max_pair
    max_pair = None
    max_pair_vocab = [0, 0]
    max_pair_new_index = None
    # 对于普通dict，如果访问不存在的键，会引发KeyError异常，而defaultdict则会返回默认值，并且自动将该键添加到字典中，值为默认值的结果。
    counts = defaultdict(int)
    cur_vocab_size = len(vocab.keys())

    while cur_vocab_size < vocab_size:
        # time1 = time.time()
        max_pair = calc_max_pair(segment_count_dict, segment_induce_dict, counts, vocab, \
                                 last_max_pair=max_pair, last_max_pair_new_index=max_pair_new_index)
        
        # update merges
        max_pair_vocab = (vocab[max_pair[0]], vocab[max_pair[1]])
        merges.append(max_pair_vocab)

        # time2 = time.time()
        # print("cur_vocab_size < vocab_size 1:", time2 - time1)

        # Merge that pair.
        max_pair_new_index = cur_vocab_size
        vocab[max_pair_new_index] = max_pair_vocab[0] + max_pair_vocab[1]
        segment_count_dict_key = list(segment_count_dict.keys())
        for segment in segment_count_dict_key:
            indices = segment_induce_dict[segment]
            # 原地修改
            if max_pair[0] not in indices or max_pair[1] not in indices:
                continue
            indices_len = merge(indices, max_pair, max_pair_new_index) 

        # time3 = time.time()
        # print("cur_vocab_size < vocab_size 2:", time3 - time2)

        # update vocab_size
        cur_vocab_size += 1
        # print("cur_vocab_size:", cur_vocab_size)
    '''

    ## init dict
    token_idx = 1
    token_idx_induce_dict = defaultdict(int)
    token_idx_count_dict  = defaultdict(int)
    induce_pre = defaultdict(int)
    induce_next = defaultdict(int)
    pair_count = defaultdict(int)
    pair_pos = defaultdict(set)
    start_time = time.time()
    for segment, segment_count in segment_count_dict.items():
        cur_induce = list(map(int, segment.encode("utf-8"))) 
        segment_induce_dict[segment] = cur_induce
        cur_induce_len = len(cur_induce)
        # for segment_idx in range(segment_count):
        for idx in range(cur_induce_len):
            token_idx_induce_dict[token_idx] = cur_induce[idx]
            token_idx_count_dict[token_idx] = segment_count
            if idx < cur_induce_len - 1:
                cur_pos = token_idx 
                cur_pair = (cur_induce[idx], cur_induce[idx+1])
                # pair_count[cur_pair] += 1
                pair_pos[cur_pair].add(cur_pos)
                pair_count[cur_pair] += segment_count
            
            induce_pre[token_idx] = 0 if idx == 0 else token_idx - 1
            induce_next[token_idx] = 0 if idx == cur_induce_len - 1 else token_idx + 1
            token_idx += 1
    end_time = time.time()
    # print("init dict:", end_time - start_time)

    ## heapq
    heap = []
    for cur_pair, cur_count in pair_count.items():
        index1, index2 = cur_pair
        # 构建堆的元素元组。
        # 元组结构决定了排序优先级：
        # 第一项: -cur_count (将计数取负，使得计数大的在堆顶)
        # 第二项: -vocab[index1], -vocab[index2] (将词汇取负，实现降序字典序比较)
        # 第三项: cur_pair (存储原始数据，用于最后返回)
        # heap_item = (-cur_count, -vocab[index1], -vocab[index2], cur_pair)
        heap_item = PairItem(cur_count, vocab[index1], vocab[index2], cur_pair)
        heapq.heappush(heap, heap_item)

    ## merge
    cur_vocab_size = len(vocab.keys())
    new_token_idx = token_idx
    while cur_vocab_size < vocab_size and heap:
        # calc max_pair
        # max_pair = None
        # max_count = 0
        # for cur_pair in pair_count.keys():
        #     cur_count = pair_count[cur_pair]
        #     if cur_count < max_count or cur_count == 0:
        #         continue

        #     if cur_count > max_count:
        #         max_count = cur_count
        #         max_pair = cur_pair
        #         continue

        #     # cur_vocab
        #     index1, index2 = cur_pair[0], cur_pair[1]
        #     max_index1, max_index2 = max_pair[0], max_pair[1]
        #     if (vocab[index1] > vocab[max_index1]) or \
        #         (vocab[index1] == vocab[max_index1] and vocab[index2] > vocab[max_index2]):
        #         max_count = cur_count
        #         max_pair = cur_pair
        # if 0 == max_count:
        #     break
        # neg_count, neg_vocab1, neg_vocab2, max_pair = heapq.heappop(heap) 
        best_item = heapq.heappop(heap)
        max_pair = best_item.original_pair
        max_count = best_item.count
        # 检查这个 pair 是否仍然有效
        if max_pair not in pair_count.keys() or \
           pair_count[max_pair] != max_count:
            continue 
        
        # update merges
        max_index1, max_index2 = max_pair[0], max_pair[1]
        max_pair_vocab = (vocab[max_index1], vocab[max_index2])
        merges.append(max_pair_vocab)
        max_pair_new_index = cur_vocab_size
        token_idx_induce_dict[new_token_idx] = max_pair_new_index
        vocab[max_pair_new_index] = max_pair_vocab[0] + max_pair_vocab[1]

        # update dict
        new_pair_set = set()
        pos_lst = list(pair_pos.get(max_pair, set()))
        for cur_pos in sorted(pos_lst):
            pre_token_idx = induce_pre[cur_pos]
            next_token_idx = induce_next[cur_pos]
            next_next_token_idx = induce_next[next_token_idx] if next_token_idx > 0 else 0
            if next_token_idx is None or \
               token_idx_induce_dict[cur_pos] != max_index1 or \
               token_idx_induce_dict[next_token_idx] != max_index2:
                continue

            if pre_token_idx > 0:
                cur_old_pair = (token_idx_induce_dict[pre_token_idx], max_index1)
                cur_new_pair = (token_idx_induce_dict[pre_token_idx], max_pair_new_index)
                # pair_count[cur_old_pair] -= 1
                assert pair_count[cur_old_pair] > 0
                pair_count[cur_old_pair] -= token_idx_count_dict[pre_token_idx]
                pair_pos[cur_old_pair].remove(pre_token_idx)

                pair_count[cur_new_pair] += token_idx_count_dict[pre_token_idx]
                pair_pos[cur_new_pair].add(pre_token_idx)
                induce_pre[new_token_idx] = pre_token_idx
                induce_next[pre_token_idx] = new_token_idx
                token_idx_count_dict[new_token_idx] = token_idx_count_dict[pre_token_idx]

                # update new_pair_set
                new_pair_set.add(cur_old_pair)
                new_pair_set.add(cur_new_pair)

            if next_next_token_idx > 0:
                cur_old_pair = (max_index2,         token_idx_induce_dict[next_next_token_idx])
                cur_new_pair = (max_pair_new_index, token_idx_induce_dict[next_next_token_idx])
                # pair_count[cur_old_pair] -= 1
                assert pair_count[cur_old_pair] > 0
                pair_count[cur_old_pair] -= token_idx_count_dict[next_next_token_idx]
                pair_pos[cur_old_pair].remove(next_token_idx)

                # pair_count[cur_new_pair] += 1
                pair_count[cur_new_pair] += token_idx_count_dict[next_next_token_idx]
                pair_pos[cur_new_pair].add(new_token_idx)
                induce_next[new_token_idx] = next_next_token_idx
                induce_pre[next_next_token_idx] = new_token_idx
                token_idx_count_dict[new_token_idx] = token_idx_count_dict[next_next_token_idx]

                # update new_pair_set
                new_pair_set.add(cur_old_pair)
                new_pair_set.add(cur_new_pair)

            # update token_idx_induce_dict
            token_idx_induce_dict[next_token_idx] = 0
            induce_pre[next_token_idx] = 0
            induce_next[next_token_idx] = 0
            token_idx_induce_dict[new_token_idx] = max_pair_new_index

            # update new_token_idx
            new_token_idx += 1

        # update heap
        for cur_pair in new_pair_set:
            # heap_item = (-pair_count[cur_pair], -vocab[cur_pair[0]], -vocab[cur_pair[1]], cur_pair)
            heap_item = PairItem(pair_count[cur_pair], vocab[cur_pair[0]], vocab[cur_pair[1]], cur_pair)
            heapq.heappush(heap, heap_item)


        # update vocab_size
        pair_count[max_pair] = 0
        del pair_count[max_pair]
        cur_vocab_size += 1
        if cur_vocab_size % 1000 == 0:
            print(cur_vocab_size)

    return vocab, merges
