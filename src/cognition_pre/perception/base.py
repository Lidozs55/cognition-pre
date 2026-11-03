"""注意力后端抽象与热图后处理。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class AttentionBackend(ABC):
    """注意力后端接口：输入模型尺寸 RGB uint8，输出 (Ph, Pw) patch 级热图 [0,1]。"""

    name = "base"

    @abstractmethod
    def infer(self, rgb_u8: np.ndarray) -> np.ndarray: ...


class HeatmapPostprocessor:
    """热图后处理：截掉前 top_pct 极值 → 归一到 [0,1]。

    clip 先于归一（在原始值域上取分位数）：若先归一，单个极端离群会把其余结构压到
    0 附近，分位数随之失真。top-pct 截断线落在背景平台上时（注意力集中在少于
    top_pct 的 patch 上，属锐利而非病态的热图）退到最小非背景值，防止截断把唯一
    信号整幅抹掉。
    """

    def __init__(self, top_clip_pct: float = 0.05) -> None:
        self.pct = float(top_clip_pct)

    def __call__(self, attn: np.ndarray) -> np.ndarray:
        a = np.asarray(attn, dtype=np.float32)
        if float(a.max()) <= float(a.min()):
            return np.zeros_like(a)
        if self.pct > 0:
            thr = float(np.quantile(a, 1.0 - self.pct))
            if thr <= float(a.min()):
                above = a[a > a.min()]
                thr = float(above.min()) if above.size else float(a.max())
            a = np.minimum(a, thr)
        span = float(a.max()) - float(a.min())
        if span <= 0:
            return np.zeros_like(a)
        a = (a - a.min()) / span
        return a.astype(np.float32)


def upsample(attn: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """patch 网格热图双线性上采样到裁切坐标系 (H, W)。"""
    h, w = size_hw
    return cv2.resize(attn.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
