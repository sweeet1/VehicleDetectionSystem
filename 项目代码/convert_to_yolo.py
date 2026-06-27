"""
UA-DETRAC 数据集 → YOLO 格式转换脚本
类别: car(0), van(1), bus(2), others(3)
输出目录: datasets/ua-detrac/
划分: 80/20 按序列级别划分
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
import yaml
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data/UA-DETRAC(车辆检测数据集8250车辆)")
XML_DIR = os.path.join(DATASET_DIR, "DETRAC-Train-Annotations-XML")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "Insight-MVT_Annotation_Train")
TEST_IMG_DIR = os.path.join(DATASET_DIR, "Insight-MVT_Annotation_Test")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "ua-detrac")

IMG_W, IMG_H = 960, 540
CLASS_MAP = {"car": 0, "van": 1, "bus": 2, "others": 3}
CLASS_NAMES = {0: "car", 1: "van", 2: "bus", 3: "others"}
RANDOM_SEED = 42
VAL_RATIO = 0.2


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = {}
    for frame in root.findall("frame"):
        frame_num = int(frame.get("num"))
        boxes = []
        target_list = frame.find("target_list")
        if target_list is None:
            continue
        for target in target_list.findall("target"):
            attr = target.find("attribute")
            vehicle_type = attr.get("vehicle_type", "others") if attr is not None else "others"
            cls_id = CLASS_MAP.get(vehicle_type, 3)
            box = target.find("box")
            if box is None:
                continue
            left = float(box.get("left", 0))
            top = float(box.get("top", 0))
            width = float(box.get("width", 0))
            height = float(box.get("height", 0))
            if width <= 0 or height <= 0:
                continue
            cx = (left + width / 2) / IMG_W
            cy = (top + height / 2) / IMG_H
            nw = width / IMG_W
            nh = height / IMG_H
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            boxes.append((cls_id, cx, cy, nw, nh))
        if boxes:
            annotations[frame_num] = boxes
    return annotations


def frame_num_to_filename(frame_num):
    return f"img{frame_num:05d}.jpg"


def create_output_dirs():
    dirs = [
        os.path.join(OUTPUT_DIR, "images", "train"),
        os.path.join(OUTPUT_DIR, "images", "val"),
        os.path.join(OUTPUT_DIR, "images", "test"),
        os.path.join(OUTPUT_DIR, "labels", "train"),
        os.path.join(OUTPUT_DIR, "labels", "val"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    # 验证目录创建成功
    for d in dirs:
        if not os.path.isdir(d):
            raise RuntimeError(f"目录创建失败: {d}")
    print(f"输出目录已创建: {OUTPUT_DIR}")


def get_all_sequences():
    xml_files = sorted([f for f in os.listdir(XML_DIR) if f.endswith(".xml")])
    sequences = [f.replace(".xml", "") for f in xml_files]
    print(f"找到 {len(sequences)} 个XML标注序列")
    return sequences


def split_sequences(sequences):
    random.seed(RANDOM_SEED)
    shuffled = sequences.copy()
    random.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * VAL_RATIO))
    val_seqs = sorted(shuffled[:val_count])
    train_seqs = sorted(shuffled[val_count:])
    print(f"训练集: {len(train_seqs)} 个序列, 验证集: {len(val_seqs)} 个序列")
    return train_seqs, val_seqs


def process_sequence(seq_name, split):
    xml_path = os.path.join(XML_DIR, f"{seq_name}.xml")
    img_dir = os.path.join(TRAIN_IMG_DIR, seq_name)
    if not os.path.exists(xml_path):
        print(f"  [跳过] XML不存在: {seq_name}")
        return 0
    annotations = parse_xml(xml_path)
    if not annotations:
        print(f"  [跳过] 无有效标注: {seq_name}")
        return 0
    img_out_dir = os.path.join(OUTPUT_DIR, "images", split)
    lbl_out_dir = os.path.join(OUTPUT_DIR, "labels", split)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)
    count = 0
    for frame_num, boxes in annotations.items():
        img_name = frame_num_to_filename(frame_num)
        src_img = os.path.join(img_dir, img_name)
        if not os.path.exists(src_img):
            continue
        dst_img = os.path.join(img_out_dir, f"{seq_name}_{img_name}")
        shutil.copy2(src_img, dst_img)
        lbl_name = f"{seq_name}_{img_name.replace('.jpg', '.txt')}"
        dst_lbl = os.path.join(lbl_out_dir, lbl_name)
        with open(dst_lbl, "w") as f:
            for cls_id, cx, cy, w, h in boxes:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        count += 1
    return count


def process_test_sequences():
    if not os.path.exists(TEST_IMG_DIR):
        print("测试集目录不存在，跳过")
        return 0
    test_out_dir = os.path.join(OUTPUT_DIR, "images", "test")
    sequences = sorted(os.listdir(TEST_IMG_DIR))
    total = 0
    for seq_name in sequences:
        seq_dir = os.path.join(TEST_IMG_DIR, seq_name)
        if not os.path.isdir(seq_dir):
            continue
        img_files = sorted([f for f in os.listdir(seq_dir) if f.endswith(".jpg")])
        for img_name in img_files:
            src = os.path.join(seq_dir, img_name)
            dst = os.path.join(test_out_dir, f"{seq_name}_{img_name}")
            shutil.copy2(src, dst)
            total += 1
        print(f"  测试序列 {seq_name}: {len(img_files)} 张图片")
    return total


def generate_data_yaml():
    data = {
        "path": OUTPUT_DIR,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 4,
        "names": ["car", "van", "bus", "others"],
    }
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"data.yaml 已生成: {yaml_path}")


def generate_file_lists():
    for split in ["train", "val"]:
        img_dir = os.path.join(OUTPUT_DIR, "images", split)
        txt_path = os.path.join(OUTPUT_DIR, f"{split}.txt")
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        with open(txt_path, "w") as f:
            for img_name in img_files:
                f.write(os.path.join(img_dir, img_name) + "\n")
        print(f"{split}.txt: {len(img_files)} 张图片")


def print_statistics(train_seqs, val_seqs):
    print("\n" + "=" * 60)
    print("转换完成统计")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(OUTPUT_DIR, "images", split)
        lbl_dir = os.path.join(OUTPUT_DIR, "labels", split)
        img_count = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")]) if os.path.exists(img_dir) else 0
        lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith(".txt")]) if os.path.exists(lbl_dir) else 0
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        if os.path.exists(lbl_dir):
            for lbl_file in os.listdir(lbl_dir):
                if lbl_file.endswith(".txt"):
                    with open(os.path.join(lbl_dir, lbl_file)) as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(parts[0])
                                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        print(f"\n{split.upper()}:")
        print(f"  图片数: {img_count}, 标注文件数: {lbl_count}")
        if lbl_count > 0:
            total = sum(class_counts.values())
            for cls_id, name in CLASS_NAMES.items():
                cnt = class_counts[cls_id]
                pct = cnt / total * 100 if total > 0 else 0
                print(f"  {name}: {cnt} ({pct:.1f}%)")


def main():
    print("=" * 60)
    print("UA-DETRAC → YOLO 格式转换")
    print("=" * 60)
    create_output_dirs()
    sequences = get_all_sequences()
    train_seqs, val_seqs = split_sequences(sequences)
    print("\n处理训练序列...")
    train_total = 0
    for i, seq in enumerate(tqdm(train_seqs, desc="训练序列", unit="序列")):
        count = process_sequence(seq, "train")
        train_total += count
        if (i + 1) % 10 == 0:
            print(f"  进度: {i + 1}/{len(train_seqs)} 序列, {train_total} 张图片")
    print("\n处理验证序列...")
    val_total = 0
    for seq in tqdm(val_seqs, desc="验证序列", unit="序列"):
        count = process_sequence(seq, "val")
        val_total += count
    print("\n处理测试集...")
    test_total = process_test_sequences()
    print("\n生成配置文件...")
    generate_data_yaml()
    generate_file_lists()
    print_statistics(train_seqs, val_seqs)
    print(f"\n训练集序列: {train_seqs}")
    print(f"验证集序列: {val_seqs}")
    print("\n转换完成!")


if __name__ == "__main__":
    main()
