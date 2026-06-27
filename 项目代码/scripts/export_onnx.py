"""
YOLO26 模型导出脚本
将训练好的.pt模型导出为ONNX格式, 加速推理部署

运行方式:
    python scripts/export_onnx.py

导出格式:
    - ONNX: 通用格式, 支持多平台部署
    - TensorRT: NVIDIA GPU加速 (可选)

导出后的模型用于:
    - Web应用部署
    - CPU/GPU推理加速
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ======================== 配置区 ========================
# 模型路径
MODEL_PATH = "runs/train/yolo26n_ua_detrac/weights/last.pt"

# 导出配置
EXPORT_FORMAT = "onnx"           # 导出格式: onnx, engine, coreml, tflite等
IMG_SIZE = 640                   # 输入尺寸
OPSET_VERSION = 14               # ONNX opset版本 (建议12-17)
SIMPLIFY = True                  # 是否简化ONNX模型

# ONNX Runtime配置 (用于验证)
USE_ONNX_RUNTIME = True          # 是否用ONNX Runtime验证导出结果

# 输出目录
OUTPUT_DIR = "runs/export"
# ========================================================


def benchmark_models(pt_path, onnx_path, img_size=640, num_runs=50):
    """
    对比PT模型和ONNX模型的推理速度

    参数:
        pt_path: PyTorch模型路径
        onnx_path: ONNX模型路径
        img_size: 输入尺寸
        num_runs: 测试次数

    返回:
        dict: 包含两种模型的速度对比
    """
    print(f"\n  基准测试 (运行 {num_runs} 次)...")

    results = {}

    # 测试PyTorch模型
    model = YOLO(pt_path)
    dummy_input = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # 预热
    for _ in range(5):
        model.predict(dummy_input, verbose=False)

    # 正式测试
    start = time.time()
    for _ in range(num_runs):
        model.predict(dummy_input, verbose=False)
    pt_time = (time.time() - start) / num_runs * 1000  # ms

    results['pt_time'] = pt_time
    print(f"    PyTorch: {pt_time:.2f} ms/帧")

    # 测试ONNX模型
    if os.path.exists(onnx_path) and USE_ONNX_RUNTIME:
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name

            # 创建输入张量 (NCHW格式)
            dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

            # 预热
            for _ in range(5):
                session.run(None, {input_name: dummy})

            # 正式测试
            start = time.time()
            for _ in range(num_runs):
                session.run(None, {input_name: dummy})
            onnx_time = (time.time() - start) / num_runs * 1000  # ms

            results['onnx_time'] = onnx_time
            speedup = pt_time / onnx_time if onnx_time > 0 else 0
            results['speedup'] = speedup

            print(f"    ONNX:     {onnx_time:.2f} ms/帧")
            print(f"    加速比:   {speedup:.2f}x")

        except ImportError:
            print("    [警告] 未安装 onnxruntime, 跳过ONNX测试")
            print("    安装命令: pip install onnxruntime")
        except Exception as e:
            print(f"    [警告] ONNX测试失败: {e}")

    return results


def main():
    print("=" * 60)
    print("YOLO26 模型导出")
    print("=" * 60)

    # 检查模型
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        print("请先运行 scripts/train.py 训练模型")
        return

    # 加载模型
    print(f"\n[1/4] 加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 导出模型
    print(f"\n[2/4] 导出为 {EXPORT_FORMAT.upper()} 格式...")
    print(f"  输入尺寸: {IMG_SIZE}")
    print(f"  OPSET版本: {OPSET_VERSION}")
    print(f"  简化模型: {SIMPLIFY}")

    export_result = model.export(
        format=EXPORT_FORMAT,
        imgsz=IMG_SIZE,
        opset=OPSET_VERSION,
        simplify=SIMPLIFY,
        dynamic=False,        # 静态batch (兼容性更好)
        half=False,           # FP32 (兼容性好, 可改True用FP16)
    )

    # 获取导出文件路径
    export_path = export_result if isinstance(export_result, str) else str(export_result)

    # 如果export_result返回的不是完整路径, 尝试拼接
    if not os.path.exists(export_path):
        # 尝试常见的导出路径
        possible_paths = [
            os.path.join(os.path.dirname(MODEL_PATH), "..", "..", "yolo26n.onnx"),
            os.path.join("runs", "train", "yolo26n_ua_detrac", "yolo26n.onnx"),
            os.path.join("runs", "train", "yolo26n_ua_detrac", "model.onnx"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                export_path = p
                break

    print(f"  导出路径: {export_path}")

    # 基准测试
    print(f"\n[3/4] 推理速度对比...")
    if os.path.exists(export_path):
        benchmark_models(MODEL_PATH, export_path, IMG_SIZE)
    else:
        print(f"  [警告] 未找到导出文件, 跳过基准测试")

    # 保存导出信息
    print(f"\n[4/4] 保存导出信息...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    info_path = os.path.join(OUTPUT_DIR, "export_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("YOLO26 模型导出信息\n")
        f.write("=" * 40 + "\n")
        f.write(f"原始模型: {MODEL_PATH}\n")
        f.write(f"导出格式: {EXPORT_FORMAT}\n")
        f.write(f"导出文件: {export_path}\n")
        f.write(f"输入尺寸: {IMG_SIZE}\n")
        f.write(f"OPSET版本: {OPSET_VERSION}\n")

    print(f"  信息已保存: {info_path}")

    print("\n" + "=" * 60)
    print("导出完成!")
    print(f"  模型文件: {export_path}")
    print("  可用于Web应用部署")
    print("=" * 60)


if __name__ == "__main__":
    main()
