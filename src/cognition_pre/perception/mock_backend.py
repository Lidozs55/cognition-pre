"""Mock 注意力后端：确定性移动高斯斑。

用途：不下载权重跑通全链路（采集→热图→双主色→调制→输出）、CI 测试。
"""

from __future__ import annotations

import time

import numpy as np

from .base import AttentionBackend


class MockAttentionBackend(AttentionBackend):
    name = "mock"
    patch_size = 16

    def __init__(
        self,
        animated: bool = True,
        speed: float = 0.25,
        pos: tuple[float, float] = (0.5, 0.5),
        sigma: float = 0.16,
    ) -> None:
        self.animated = animated
        self.speed = speed
        self.pos = pos
        self.sigma = sigma

    def infer(self, rgb_u8: np.ndarray) -> np.ndarray:
        ph = max(1, rgb_u8.shape[0] // self.patch_size)
        pw = max(1, rgb_u8.shape[1] // self.patch_size)
        if self.animated:
            t = time.time() * self.speed * 2.0 * np.pi
            cx = self.pos[0] + 0.30 * np.sin(t)
            cy = self.pos[1] + 0.30 * np.cos(0.8 * t)
        else:
            cx, cy = self.pos
        ys = (np.arange(ph) + 0.5) / ph - cy
        xs = (np.arange(pw) + 0.5) / pw - cx
        g = np.exp(-(xs[None, :] ** 2 + ys[:, None] ** 2) / (2.0 * self.sigma**2))
        return g.astype(np.float32)
