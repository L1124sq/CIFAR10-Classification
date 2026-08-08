# -*- coding: utf-8 -*-
"""
test.py —— 模型测试

加载训练时保存的最好模型，在 10000 张测试集图片上做详细评估：
  1. 整体准确率
  2. 每个类别的精确率 / 召回率 / F1（scikit-learn 计算）
  3. 混淆矩阵图（哪些类别容易被搞混）

运行示例：
    python test.py --model cnn
    python test.py --model resnet
"""

import argparse

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix

# matplotlib 默认字体不含中文，设置成系统中文字体，否则图里的中文会变方块
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

from dataset import get_cifar10_dataloader
from model import create_model
from utils import get_device, load_checkpoint, evaluate


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 分类测试')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型权重路径，默认 checkpoints/{model}_best.pth')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()

    device = get_device()
    checkpoint_path = args.checkpoint or f'checkpoints/{args.model}_best.pth'

    # 加载数据（只用测试集）和模型
    _, test_loader = get_cifar10_dataloader(args.batch_size, args.data_root)
    model = create_model(args.model).to(device)
    epoch, best_acc = load_checkpoint(checkpoint_path, model, device)
    print(f'已加载模型: {checkpoint_path}（保存于 epoch {epoch}，验证集最好 acc={best_acc:.4f}）')

    # 1) 整体指标
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f'\n测试集整体: loss {test_loss:.4f} | acc {test_acc:.4f}')

    # 2) 逐类指标：把所有预测结果先收集起来
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    class_names = test_loader.dataset.classes
    print('\n逐类别指标:')
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 3) 混淆矩阵：第 i 行第 j 列 = "真实是 i、被预测成 j"的样本数
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(10), class_names, rotation=45)
    ax.set_yticks(range(10), class_names)
    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f'logs/{args.model}_confusion_matrix.png', dpi=150)
    print(f'\n混淆矩阵已保存: logs/{args.model}_confusion_matrix.png')


if __name__ == '__main__':
    main()
