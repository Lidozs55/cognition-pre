"""帧率统计与固定频率节拍器。"""

from __future__ import annotations

import time


class FpsMeter:
    """EMA 平滑的帧率计。"""

    def __init__(self, smooth: float = 0.85) -> None:
        self._smooth = smooth
        self._prev: float | None = None
        self._fps = 0.0

    def tick(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        if self._prev is not None:
            dt = now - self._prev
            if dt > 0:
                inst = 1.0 / dt
                self._fps = inst if self._fps <= 0 else self._smooth * self._fps + (1 - self._smooth) * inst
        self._prev = now
        return self._fps

    @property
    def value(self) -> float:
        return self._fps


class RateKeeper:
    """固定频率循环节拍器：每次循环开头调用 sleep_next()。

    落后超过一个周期则重置基准，避免"追赶连发"。
    """

    def __init__(self, fps: float) -> None:
        self.period = 1.0 / max(fps, 1e-6)
        self._next = time.monotonic()  # 立即触发首轮

    def sleep_next(self) -> None:
        now = time.monotonic()
        delay = self._next - now
        if delay > 0:
            time.sleep(delay)
        now = time.monotonic()
        if self._next < now - self.period:  # 落太多，重置
            self._next = now
        self._next += self.period
