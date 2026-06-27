"""
YOLO26 模型训练与评估脚本
基于 COCO 预训练权重 yolo26n.pt 在 UA-DETRAC 车辆检测数据集上开展迁移学习。
训练结束后输出 mAP、精确率(Precision)、召回率(Recall) 等全面评估指标。
"""

import os
import sys
import yaml
from pathlib import Path

# 当前工作目录
BASE_DIR = Path(__file__).parent.absolute()
DATA_YAML = BASE_DIR / "datasets" / "ua-detrac" / "data.yaml"
WEIGHTS = BASE_DIR / "yolo26n.pt"

# 训练超参数（可根据 GPU 显存调整）
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 4
DEVICE = "0"  # 使用 GPU 0；CPU 训练请改为 "cpu"

# 输出目录
PROJECT_NAME = "ua-detrac-yolo26"
RUN_NAME = "train2"


def check_environment():
    """检查必要的文件与依赖是否就绪。"""
    missing = []
    if not DATA_YAML.exists():
        missing.append(str(DATA_YAML))
    if not WEIGHTS.exists():
        missing.append(str(WEIGHTS))

    if missing:
        print("[错误] 以下必要文件缺失:")
        for m in missing:
            print(f"  - {m}")
        print("\n提示：请先运行 convert_to_yolo.py 完成数据集转换，并确保 yolo26n.pt 位于当前目录。")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 未安装 ultralytics 库。")
        print("请执行: pip install ultralytics -q")
        sys.exit(1)

    print("[环境检查通过]")
    print(f"  数据集配置: {DATA_YAML}")
    print(f"  预训练权重: {WEIGHTS}")
    print(f"  训练轮数: {EPOCHS}, 图像尺寸: {IMG_SIZE}, 批次大小: {BATCH_SIZE}")


def train_model():
    """加载 yolo26n.pt 在 UA-DETRAC 上训练，支持从 train2 断点续训。"""
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("阶段 1: 加载模型")
    print("=" * 60)

    LAST_PT = BASE_DIR / "runs" / "detect" / PROJECT_NAME / RUN_NAME / "weights" / "last.pt"

    if LAST_PT.exists():
        print(f"检测到断点权重，恢复训练: {LAST_PT}")
        model = YOLO(str(LAST_PT))
        print(f"成功加载: {LAST_PT}")

        print("\n" + "=" * 60)
        print("阶段 2: 断点续训")
        print("=" * 60)
        results = model.train(resume=True)
        return model, results

    print(f"加载 COCO 预训练权重: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))
    print(f"成功加载: {WEIGHTS}")

    print("\n" + "=" * 60)
    print("阶段 2: 开始迁移学习训练")
    print("=" * 60)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-2,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=10,
        save=True,
        save_period=-1,
        verbose=True,
        # 数据增强（针对车辆遮挡/粘连场景优化）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,          # 4图拼接，天然制造密集场景
        mixup=0.2,           # 提高至0.2：模拟半透明遮挡
        copy_paste=0.25,     # 提高至0.25：人工制造车叠车重叠
    )

    return model, results


def evaluate_model(model):
    """在验证集上全面评估模型性能。"""
    print("\n" + "=" * 60)
    print("阶段 3: 模型性能评估")
    print("=" * 60)

    # 默认在 data.yaml 指定的 val 上进行验证
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        split="val",
        save_json=True,
        save_hybrid=False,
        conf=0.001,
        iou=0.6,
        max_det=300,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("验证集评估结果汇总")
    print("=" * 60)

    # mAP 指标
    map50_95 = metrics.box.map   # mAP@0.5:0.95
    map50 = metrics.box.map50    # mAP@0.5
    map75 = metrics.box.map75    # mAP@0.75

    # 每个类别的 AP@0.5:0.95
    maps_per_class = metrics.box.maps

    # 精确率与召回率（在 conf=0.001 下的原始 PR 曲线近似值）
    # 若需特定置信度阈值下的 P/R，可通过 metrics.box.p / metrics.box.r 获取 PR 曲线数组
    precision_curve = metrics.box.p  # 各置信度下的精确率
    recall_curve = metrics.box.r     # 各置信度下的召回率
    f1_curve = metrics.box.f1        # 各置信度下的 F1

    # 取最高 F1 对应的 P 和 R 作为代表性值
    if f1_curve is not None and len(f1_curve) > 0:
        best_f1_idx = f1_curve.argmax()
        best_precision = precision_curve[best_f1_idx]
        best_recall = recall_curve[best_f1_idx]
        best_f1 = f1_curve[best_f1_idx]
    else:
        best_precision = best_recall = best_f1 = 0.0

    print(f"\n【整体性能】")
    print(f"  mAP@0.5:0.95 : {map50_95:.4f}")
    print(f"  mAP@0.5      : {map50:.4f}")
    print(f"  mAP@0.75     : {map75:.4f}")

    print(f"\n【最佳 F1 阈值处】")
    print(f"  Precision    : {best_precision:.4f}")
    print(f"  Recall       : {best_recall:.4f}")
    print(f"  F1-Score     : {best_f1:.4f}")

    # 加载类别名称
    class_names = []
    if DATA_YAML.exists():
        with open(DATA_YAML, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
        class_names = data_cfg.get("names", [])

    if maps_per_class is not None and len(maps_per_class) > 0:
        print(f"\n【各类别 AP@0.5:0.95】")
        for i, ap in enumerate(maps_per_class):
            name = class_names[i] if i < len(class_names) else f"class_{i}"
            print(f"  {name:12s}: {ap:.4f}")

    # 额外指标：在 conf=0.25 下重新验证，获取更贴近实际部署的 P/R
    print("\n" + "-" * 60)
    print("补充：在 conf=0.25 阈值下重新评估（更贴近实际部署场景）")
    print("-" * 60)
    metrics_pr = model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        split="val",
        conf=0.25,
        iou=0.6,
        verbose=False,
    )
    print(f"  mAP@0.5:0.95 : {metrics_pr.box.map:.4f}")
    print(f"  mAP@0.5      : {metrics_pr.box.map50:.4f}")

    # 保存评估结果摘要到文本
    summary_path = Path(PROJECT_NAME) / RUN_NAME / "evaluation_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("YOLO26 模型评估结果摘要\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"数据集: {DATA_YAML}\n")
        f.write(f"权重  : {WEIGHTS}\n")
        f.write(f"验证集划分: val split from data.yaml\n\n")
        f.write("[mAP 指标]\n")
        f.write(f"  mAP@0.5:0.95 : {map50_95:.4f}\n")
        f.write(f"  mAP@0.5      : {map50:.4f}\n")
        f.write(f"  mAP@0.75     : {map75:.4f}\n\n")
        f.write("[最佳 F1 阈值处]\n")
        f.write(f"  Precision    : {best_precision:.4f}\n")
        f.write(f"  Recall       : {best_recall:.4f}\n")
        f.write(f"  F1-Score     : {best_f1:.4f}\n\n")
        if class_names and maps_per_class is not None:
            f.write("[各类别 AP@0.5:0.95]\n")
            for i, ap in enumerate(maps_per_class):
                name = class_names[i] if i < len(class_names) else f"class_{i}"
                f.write(f"  {name:12s}: {ap:.4f}\n")
        f.write("\n")
        f.write("[conf=0.25 补充评估]\n")
        f.write(f"  mAP@0.5:0.95 : {metrics_pr.box.map:.4f}\n")
        f.write(f"  mAP@0.5      : {metrics_pr.box.map50:.4f}\n")

    print(f"\n评估摘要已保存至: {summary_path}")
    return metrics


def export_model(model):
    """导出 ONNX / TorchScript 等格式（可选）。"""
    print("\n" + "=" * 60)
    print("阶段 4: 模型导出 (ONNX)")
    print("=" * 60)
    try:
        export_path = model.export(format="onnx", imgsz=IMG_SIZE)
        print(f"ONNX 导出成功: {export_path}")
    except Exception as e:
        print(f"ONNX 导出失败（非致命）: {e}")


def main():
    check_environment()
    model, train_results = train_model()
    evaluate_model(model)
    export_model(model)
    print("\n" + "=" * 60)
    print("全部流程结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
