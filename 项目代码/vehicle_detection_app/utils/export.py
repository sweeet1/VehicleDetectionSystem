"""
统计导出模块：将车辆统计数据导出为 CSV 文件。
"""

import csv
import os
from datetime import datetime
from typing import Dict


class StatsExporter:
    """将检测统计数据导出为 CSV 格式"""

    @staticmethod
    def export_summary(stats: Dict, filepath: str = "") -> str:
        """导出一段时间内的汇总统计数据"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else "./exports", exist_ok=True)

        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"./exports/vehicle_stats_{timestamp}.csv"

        raw_cc = stats.get("class_counts", {})
        # 兼容两种格式：扁平 {car:n, ...} 或嵌套 {"entry":{...},"exit":{...}}
        if "entry" in raw_cc or "exit" in raw_cc:
            counts = {"car": 0, "bus": 0, "truck": 0, "moto": 0}
            model_to_key = {0: "car", 1: "truck", 2: "bus", 3: "moto"}
            for direction in ("entry", "exit"):
                for k, v in raw_cc.get(direction, {}).items():
                    key = model_to_key.get(k, "moto")
                    counts[key] = counts.get(key, 0) + v
        else:
            counts = raw_cc

        total = sum(counts.values()) or stats.get("total_count", 0)
        rows = [
            ["指标", "数值"],
            ["车辆总数", total],
            ["IN", stats.get("entry_count", "--")],
            ["OUT", stats.get("exit_count", "--")],
            ["轿车", counts.get("car", 0)],
            ["货车", counts.get("truck", 0)],
            ["公交车", counts.get("bus", 0)],
            ["其他", counts.get("moto", 0)],
            ["平均车速(km/h)", stats.get("avg_speed_kmh", "--")],
            ["帧率 FPS", f"{stats.get('fps', 0):.1f}"],
            ["导出时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ]

        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return filepath

    @staticmethod
    def export_frame_log(log_entries: list, filepath: str = "") -> str:
        """导出逐帧检测日志"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else "./exports", exist_ok=True)

        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"./exports/detection_log_{timestamp}.csv"

        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["帧号", "目标ID", "类别", "置信度", "x1", "y1", "x2", "y2"])
            for entry in log_entries:
                writer.writerow(entry)

        return filepath
