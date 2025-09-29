import math
import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange,einsum
from collections.abc import Callable, Iterable
import numpy.typing as npt
import numpy as np


dataset_idx = 0

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    global dataset_idx

    dataset = dataset.reshape(-1)
    dataset_size = dataset.shape[0]
    cur_data = np.zeros(batch_size*context_length, )
    cur_label = np.zeros(batch_size*context_length, )

    for i in range(batch_size):
        cur_data_indices = np.arange(dataset_idx, dataset_idx + context_length)
        cur_data[i*context_length:(i+1)*context_length] = np.take(dataset, cur_data_indices, mode='wrap')
        cur_label_indices = np.arange(dataset_idx + 1, dataset_idx + context_length + 1)
        cur_label[i*context_length:(i+1)*context_length] = np.take(dataset, cur_label_indices, mode='wrap')
        # last label
        if dataset_idx + context_length == dataset_size:
            cur_label[(i+1)*context_length - 1] = dataset_size
        # reset idx
        dataset_idx += 1
        # 此处保证至少保留一个元素作为label
        if dataset_idx + context_length >= dataset_size:
            dataset_idx = 0

    return_tensor = (torch.Tensor(cur_data.reshape(batch_size, context_length), device=device), \
                     torch.Tensor(cur_label.reshape(batch_size, context_length), device=device))
    return return_tensor


