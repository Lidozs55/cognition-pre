"""M2 预处理：中心方形裁切 + 模型尺寸缩放。

裁切帧是全管线读色坐标系基准；缩放帧只喂模型。
已知代价：16:9 全景镜头两侧背景被裁，氛围色素材变少（见 docs/roadmap.md Phase 0 专项验证）。
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import PreprocessConfig


def center_square_crop(frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """中心方形裁切，边长 = min(H, W)。返回 (crop, (x0, y0))。"""
    h, w = frame_bgr.shape[:2]
    s = min(h, w)
    x0 = (w - s) // 2
    y0 = (h - s) // 2
    return frame_bgr[y0 : y0 + s, x0 : x0 + s], (x0, y0)


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        if not self.cfg.square_crop:
            return frame_bgr, (0, 0)
        return center_square_crop(frame_bgr)

    def to_model_rgb(self, crop_bgr: np.ndarray, size: int | None = None) -> np.ndarray:
        """裁切帧 → 模型输入尺寸的 RGB uint8（缩小用 INTER_AREA 抗锯齿）。"""
        size = size or self.cfg.model_input_size
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != size or rgb.shape[1] != size:
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        return rgb
