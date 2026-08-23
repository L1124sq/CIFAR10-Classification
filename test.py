# -*- coding: utf-8 -*-
"""
test.py —— 模型测试

加载训练时保存的最好模型，在 10000 张测试集图片上做详细评估：
  1. 整体准确率
  2. 每个类别的精确率 / 召回率 / F1
  3. 混淆矩阵图
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from dataset import get_cifar10_dataloader
from model import create_model
from utils import (
    append_csv,
    count_parameters,
    evaluate,
    get_device,
    load_checkpoint,
    save_json,
)


MODEL_CHOICES = [
    "cnn",
    "resnet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 分类测试")
    parser.add_argument("--model", type=str, default="cnn", choices=MODEL_CHOICES)
    parser.add_argument("--tag", type=str, default=None, help="训练时使用的实验标识")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型权重路径")
    parser.add_argument("--dropout", type=float, default=0.0, help="与训练时相同的 Dropout")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="与训练时相同的标签平滑")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    device = get_device()
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.tag:
        checkpoint_path = f"checkpoints/{args.tag}_best.pth"
    else:
        checkpoint_path = f"checkpoints/{args.model}_best.pth"

    output_name = args.tag or args.model

    _, test_loader = get_cifar10_dataloader(args.batch_size, args.data_root)
    model = create_model(args.model, dropout=args.dropout).to(device)
    total_params, trainable_params = count_parameters(model)
    epoch, best_acc = load_checkpoint(checkpoint_path, model, device)
    print(
        f"已加载模型: {checkpoint_path}（保存于 epoch {epoch}，"
        f"验证集最好 acc={best_acc:.4f}）"
    )
    print(f"模型参数量: total={total_params:,}, trainable={trainable_params:,}")

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        device,
        label_smoothing=args.label_smoothing,
    )
    print(f"\n测试集整体: loss {test_loss:.4f} | acc {test_acc:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    class_names = test_loader.dataset.classes
    print("\n逐类别指标:")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
    ))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), class_names, rotation=45)
    ax.set_yticks(range(10), class_names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    cm_path = f"logs/{output_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)

    metrics = {
        "tag": output_name,
        "model": args.model,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "checkpoint": checkpoint_path,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    save_json(f"results/metrics/{output_name}_test.json", metrics)
    append_csv(
        "results/test_metrics.csv",
        metrics,
        fieldnames=list(metrics.keys()),
    )
    print(f"\n混淆矩阵已保存: {cm_path}")
    print(f"测试结果已保存: results/metrics/{output_name}_test.json")


if __name__ == "__main__":
    main()
