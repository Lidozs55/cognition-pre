"""双阈值分段线性 ramp：展示路调制与主色加权共用的同一份实现。

主色权重与调制掩码必须同形（所见即所得的计算侧对应），故不各自复制一份——
改这里即两处同时生效。语义：s<=lo → 0，s>=hi → 1，中间线性；hi<=lo 退化为阶跃。
"""

from __future__ import annotations

import numpy as np


def ramp(s: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """[lo, hi] 双阈值分段线性：<=lo → 0，>=hi → 1，中间线性；hi<=lo 时退化为阶跃。"""
    if hi <= lo:
        return (np.asarray(s) >= hi).astype(np.float32)
    s = np.asarray(s, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
