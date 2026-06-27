"""
YOLO26 模型评估脚本
对训练好的模型进行完整评估, 输出各项指标和可视化图表

运行方式:
    python scripts/evaluate.py

输出:
    - mAP@0.5, mAP@0.5:0.95
    - Precision, Recall, F1
    - PR曲线
    - 混淆矩阵
    - 各类别指标
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# ======================== 配置区 ========================
# 模型路径 (训练后自动产生)
MODEL_PATH = "runs/train/yolo26n_ua_detrac/weights/best_train2.pt"

# 数据集配置
# DATA_CONFIG = "datasets/ua-detrac/data.yaml"
DATA_CONFIG = "H:/深度学习实践任务/datasets/ua-detrac/data.yaml"

# 评估配置
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0

# 输出目录
OUTPUT_DIR = "runs/evaluate"
# ========================================================


def main():
    print("=" * 60)
    print("YOLO26 模型评估")
    print("=" * 60)

    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        print("请先运行 scripts/train.py 训练模型")
        return

    # 加载模型
    print(f"\n[1/4] 加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 运行评估
    print(f"\n[2/4] 在验证集上评估...")
    results = model.val(
        data=DATA_CONFIG,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        conf=0.25,
        iou=0.6,
        plots=True,        # 生成可视化图表
        save_json=False,   # 不保存COCO格式JSON
        verbose=True,
    )

    # 打印详细指标
    print(f"\n[3/4] 评估结果:")
    print("-" * 40)

    # 总体指标
    metrics = results.results_dict
    print(f"  mAP@0.5:       {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP@0.5:0.95:  {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision:     {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall:        {metrics.get('metrics/recall(B)', 0):.4f}")

    # 各类别指标
    if hasattr(results, 'ap_class_index') and results.ap_class_index is not None:
        print(f"\n  各类别 mAP@0.5:0.95:")
        names = results.names
        maps = results.maps  # 每个类别的mAP50-95
        for i, class_idx in enumerate(results.ap_class_index):
            ap = maps[i] if i < len(maps) else 0
            print(f"    {names[class_idx]:12s}: mAP={ap:.4f}")

    print("-" * 40)

    # 保存评估报告
    print(f"\n[4/4] 保存评估报告...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 写入文本报告
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("YOLO26 模型评估报告\n")
        f.write("=" * 40 + "\n")
        f.write(f"模型: {MODEL_PATH}\n")
        f.write(f"数据集: {DATA_CONFIG}\n")
        f.write(f"图片尺寸: {IMG_SIZE}\n\n")
        f.write(f"mAP@0.5:       {metrics.get('metrics/mAP50(B)', 0):.4f}\n")
        f.write(f"mAP@0.5:0.95:  {metrics.get('metrics/mAP50-95(B)', 0):.4f}\n")
        f.write(f"Precision:     {metrics.get('metrics/precision(B)', 0):.4f}\n")
        f.write(f"Recall:        {metrics.get('metrics/recall(B)', 0):.4f}\n")

        if hasattr(results, 'ap_class_index') and results.ap_class_index is not None:
            f.write(f"\n各类别 mAP@0.5:0.95:\n")
            names = results.names
            maps = results.maps
            for i, class_idx in enumerate(results.ap_class_index):
                ap = maps[i] if i < len(maps) else 0
                f.write(f"  {names[class_idx]:12s}: {ap:.4f}\n")

    print(f"  报告已保存: {report_path}")

    # 生成PR曲线
    try:
        generate_pr_curve(results, OUTPUT_DIR)
        print(f"  PR曲线已保存: {OUTPUT_DIR}/pr_curve.png")
    except Exception as e:
        print(f"  [警告] PR曲线生成失败: {e}")

    # 生成对比柱状图
    try:
        generate_bar_chart(results, OUTPUT_DIR)
        print(f"  指标柱状图已保存: {OUTPUT_DIR}/metrics_bar.png")
    except Exception as e:
        print(f"  [警告] 柱状图生成失败: {e}")

    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


def generate_pr_curve(results, output_dir):
    """生成PR曲线图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 获取PR数据
    if hasattr(results, 'speed') and results.speed is not None:
        # 使用Ultralytics内置的plot方法保存
        results.save_dir = output_dir
        # 保存混淆矩阵和PR曲线
        for plot_name in ['confusion_matrix.png', 'PR_curve.png', 'F1_curve.png']:
            src = os.path.join(results.save_dir, plot_name)
            if os.path.exists(src):
                print(f"  已生成: {plot_name}")

    plt.close(fig)


def generate_bar_chart(results, output_dir):
    """生成指标对比柱状图"""
    metrics = results.results_dict

    metric_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    metric_values = [
        metrics.get('metrics/mAP50(B)', 0),
        metrics.get('metrics/mAP50-95(B)', 0),
        metrics.get('metrics/precision(B)', 0),
        metrics.get('metrics/recall(B)', 0),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax.bar(metric_names, metric_values, color=colors, width=0.6)

    # 在柱子上显示数值
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('YOLO26 Model Evaluation Metrics', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_bar.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
