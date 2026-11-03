"""M6 时域稳定：对 10 Hz 主色输出做 EMA 平滑，抑制簇切换抖动。

ema_alpha 是"每次更新"语义，默认值按 color.hz=10 调校（时间常数 τ≈2.0s）；
改 color.hz 时按 α = 1 − exp(−1/(hz·τ)) 重调（由 τ = 1/(hz·−ln(1−α)) 反推），
否则平滑强度随频率漂移。

大跳变重置（Phase 2）：新主色与 EMA 状态的 RGB 距离超过 jump_reset_dist 时直接
置入——EMA 会把场景切换"爬"成一段错误的中间色相，迟滞管同屏闪烁，重置管换景。
"""

from __future__ import annotations

import numpy as np


class ColorStabilizer:
    def __init__(
        self,
        ema_alpha: float = 0.35,
        conf_ema_alpha: float = 0.2,
        jump_reset_dist: float = 60.0,
    ) -> None:
        self.a = float(ema_alpha)
        self.ca = float(conf_ema_alpha)
        self.jump_dist = float(jump_reset_dist)
        self._rgb: np.ndarray | None = None
        self._conf = 0.0

    def update(self, rgb: tuple[float, float, float], conf: float) -> tuple[tuple[float, float, float], float]:
        """输入最新主色与置信度，返回平滑后的 (rgb, conf)。首值直通。"""
        rgb_arr = np.asarray(rgb, dtype=np.float64)
        if self._rgb is None:
            self._rgb = rgb_arr.copy()
            self._conf = float(conf)
            return tuple(float(v) for v in self._rgb), float(self._conf)
        if self.jump_dist > 0 and float(np.linalg.norm(rgb_arr - self._rgb)) > self.jump_dist:
            self._rgb = rgb_arr.copy()  # 大跳变：置入，不爬中间色相
        else:
            self._rgb += self.a * (rgb_arr - self._rgb)
        self._conf += self.ca * (float(conf) - self._conf)
        return tuple(float(v) for v in self._rgb), float(self._conf)
