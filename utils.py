# -*- coding: utf-8 -*-
"""
utils.py —— 通用工具函数

把"训练和测试都要用到的杂活"集中放在这里，让 train.py / test.py 保持简洁。
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed=42):
    """固定随机种子，让每次实验的随机数都一样，结果可以复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """优先使用 GPU（cuda），没有 GPU 才退回 CPU。"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(path, model, optimizer, epoch, best_acc):
    """把当前模型状态保存到磁盘，方便以后接着训练或测试。

    保存的不只是参数，还包括优化器状态、训练到第几轮、当前最好准确率。
    """
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
    }
    torch.save(state, path)


def load_checkpoint(path, model, device):
    """加载 checkpoint 里的模型参数，返回 (保存时的轮次, 最好准确率)。"""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    return state.get('epoch', 0), state.get('best_acc', 0.0)


@torch.no_grad()   # 评估阶段不需要算梯度：省显存、速度快
def evaluate(model, dataloader, device):
    """在一个数据集（验证集或测试集）上算平均损失和准确率。

    注意三件事：
      1. 一定要 model.eval()：让 BatchNorm 等层切换到"评估模式"
      2. 一定要 torch.no_grad()：这里只是"看模型答得怎么样"，不更新参数
      3. 准确率 = 预测对的样本数 / 总样本数
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total = 0
    correct = 0
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)                        # 模型预测：每个样本 10 个得分
        loss = criterion(outputs, labels)              # 这一批的平均损失
        total_loss += loss.item() * images.size(0)     # 按样本数加权，累加总损失
        preds = outputs.argmax(dim=1)                  # 得分最高的那个类别 = 预测结果
        correct += (preds == labels).sum().item()      # 数一数这一批猜对几个
        total += labels.size(0)
    return total_loss / total, correct / total         # (平均损失, 准确率)


def plot_curves(train_losses, val_losses, val_accs, save_path):
    """把训练过程中的 loss / acc 画成曲线图，保存成 PNG。"""
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label='train loss')
    axes[0].plot(epochs, val_losses, label='val loss')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].set_title('Loss')

    axes[1].plot(epochs, val_accs, label='val acc', color='orange')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')
    axes[1].legend()
    axes[1].set_title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
