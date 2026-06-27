"""
自动车流分析：用光流分析前 N 帧，自动判断方向、检测线类型和位置。
"""

import cv2
import numpy as np


def analyze_traffic_flow(video_path: str, n_frames: int = 30) -> dict | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_step = max(1, total // (n_frames + 1))
    flows = []

    prev_gray = None
    count = 0
    for i in range(min(n_frames * 3, total)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * sample_step)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = flow.shape[:2]
            cy, cx = h // 2, w // 2
            region = flow[cy - h // 4:cy + h // 4, cx - w // 4:cx + w // 4]
            fx = float(region[:, :, 0].mean())
            fy = float(region[:, :, 1].mean())
            if abs(fx) > 0.5 or abs(fy) > 0.5:
                flows.append([fx, fy])
                count += 1
                if count >= n_frames:
                    break
        prev_gray = gray

    cap.release()
    if len(flows) < 5:
        return None

    avg_fx = sum(f[0] for f in flows) / len(flows)
    avg_fy = sum(f[1] for f in flows) / len(flows)
    abs_fx, abs_fy = abs(avg_fx), abs(avg_fy)

    if abs_fy > abs_fx * 1.2:
        direction = "vertical"
        primary = "down" if avg_fy > 0 else "up"
        entry_pct = 0.25 if avg_fy > 0 else 0.75
        exit_pct = 0.75 if avg_fy > 0 else 0.25
        line_type = "horizontal"
    elif abs_fx > abs_fy * 1.2:
        direction = "horizontal"
        primary = "right" if avg_fx > 0 else "left"
        entry_pct = 0.25 if avg_fx > 0 else 0.75
        exit_pct = 0.75 if avg_fx > 0 else 0.25
        line_type = "vertical"
    else:
        direction = "diagonal"
        if abs_fy >= abs_fx:
            primary = "down" if avg_fy > 0 else "up"
            entry_pct = 0.25 if avg_fy > 0 else 0.75
            exit_pct = 0.75 if avg_fy > 0 else 0.25
            line_type = "horizontal"
        else:
            primary = "right" if avg_fx > 0 else "left"
            entry_pct = 0.25 if avg_fx > 0 else 0.75
            exit_pct = 0.75 if avg_fx > 0 else 0.25
            line_type = "vertical"

    return {
        "entry_pct": entry_pct, "exit_pct": exit_pct,
        "line_type": line_type, "primary": primary,
        "direction": direction, "flow_x": avg_fx, "flow_y": avg_fy,
        "width": width, "height": height,
    }
