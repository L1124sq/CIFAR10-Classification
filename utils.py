# -*- coding: utf-8 -*-
"""
utils.py —— 通用工具函数（“工具箱”）

把训练和测试都要用到的小功能集中在这里，让 train.py / test.py 保持简洁。
"""

import csv
import json
import os
import random

import matplotlib

matplotlib.use("Agg")

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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(path, model, optimizer, epoch, best_acc, extra=None):
    """保存模型、优化器状态和关键元信息。"""
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(path, model, device):
    """加载 checkpoint 里的模型参数，返回 (保存时的轮次, 最好准确率)。"""
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return state.get("epoch", 0), state.get("best_acc", 0.0)


@torch.no_grad()
def evaluate(model, dataloader, device, label_smoothing=0.0):
    """在一个数据集（验证集或测试集）上算平均损失和准确率。"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total = 0
    correct = 0
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def plot_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """画三张并排曲线：loss、accuracy、train-val gap。"""
    epochs = range(1, len(train_losses) + 1)
    gaps = [t - v for t, v in zip(train_accs, val_accs)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(epochs, train_losses, label="train loss")
    axes[0].plot(epochs, val_losses, label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(epochs, train_accs, label="train acc")
    axes[1].plot(epochs, val_accs, label="val acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy")

    axes[2].plot(epochs, gaps, label="train acc - val acc", color="red")
    axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("gap")
    axes[2].legend()
    axes[2].set_title("Generalization Gap")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_json(path, data):
    """把字典保存成 UTF-8 JSON 文件。"""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_csv(path, row, fieldnames):
    """把一行字典追加到 CSV；文件不存在时自动写表头。"""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
