import json
import time
import numpy as np

import random
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 20250927
from cs336_basics.train import SGD

def toy_sgd(lr, epochs, num):
    all_losses = []
    for idx in range(num):
        losses = []
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=1)
        for t in range(epochs):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            # print(loss.cpu().item())
            losses.append(loss.item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.
        all_losses.append(losses)
    return all_losses

if __name__ == "__main__":
    num = 1
    epochs = 10000
    results = {}
    learning_rates = [1e1, 1e2, 1e3, 1e4]
    for lr in learning_rates:
        all_losses = toy_sgd(lr, epochs, num)
        results[lr] = np.mean(all_losses, axis=0)

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 绘制每条曲线
    lines = []
    labels = []

    for i, (lr, losses) in enumerate(results.items()):
        # 生成鲜艳的随机颜色，避免太暗或太亮的颜色
        r = random.random() * 0.8 + 0.2  # 0.3-1.0
        g = random.random() * 0.8 + 0.2
        b = random.random() * 0.8 + 0.2
        color = (r, g, b)
        line, = plt.plot(range(epochs), losses, 
                        color=color, 
                        linewidth=1, 
                        marker='o', 
                        markersize=1,
                        label=f'LR={lr}')
        lines.append(line)
        labels.append(f'LR={lr}')

    # 设置图表属性
    # plt.title('不同学习率下的损失变化趋势', fontsize=16, fontweight='bold', pad=20)
    # plt.xlabel('迭代次数', fontsize=14)
    # plt.ylabel('损失值 (Loss)', fontsize=14)
    plt.xlabel('t', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数坐标
    
    # 在右上角添加图例
    legend = plt.legend(lines, labels, 
                    loc='upper right',
                    title='lr',
                    fontsize=12,
                    title_fontsize=13,
                    framealpha=0.9,
                    shadow=True)

    # 设置图例背景颜色
    legend.get_frame().set_facecolor('lightgray')
    legend.get_frame().set_alpha(0.8)

    # 添加额外的统计信息框
    final_loss_text = "\n".join([f"LR={lr}: {results[lr][-1]:.2e}" for lr in learning_rates])
    plt.text(0.02, 0.98, f'Loss:\n{final_loss_text}', 
            transform=plt.gca().transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()

    # 保存图像
    plt.savefig('loss_lr.png', dpi=300, bbox_inches='tight')

