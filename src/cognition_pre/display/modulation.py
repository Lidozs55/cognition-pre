"""M5 双流亮度调制（展示路）：它本身就是这套系统最好的解释图。

分段线性阈值化（ramp）：焦点流对 A(x) 做 lo/hi 双阈值——A<=lo 全黑（black_floor 残亮）、
A>=hi 原始亮度、中间线性过渡；氛围流对 1−A(x) 同样处理（阈值独立配置）。
比旧版 γ 线性更激进：注意力够低就看不见、够高就原样，明暗分离锐利，适合课堂展示。
逐像素乘法，全帧率跑（用最近热图，与热图更新解耦）。

铁律：输出只给人看，严禁进入任何颜色计算（展示/计算分离）。
录制（StreamRecorder）同属展示路：只写不读，与计算路零耦合。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import ModulationConfig


def _ramp(s: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """[lo, hi] 双阈值分段线性：<=lo → 0，>=hi → 1，中间线性；hi<=lo 时退化为阶跃。"""
    if hi <= lo:
        return (s >= hi).astype(np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


class DualStreamModulator:
    def __init__(self, cfg: ModulationConfig) -> None:
        self.focus_lo = float(cfg.focus_lo)
        self.focus_hi = float(cfg.focus_hi)
        self.ambient_lo = float(cfg.ambient_lo)
        self.ambient_hi = float(cfg.ambient_hi)
        self.floor = float(cfg.black_floor)
        self.enabled = bool(cfg.enabled)

    def apply(self, frame_u8: np.ndarray, s01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (focus_stream, ambient_stream)，与输入同尺寸同格式。s01 为 A(x) [0,1]。"""
        a = np.clip(np.asarray(s01, dtype=np.float32), 0.0, 1.0)
        m_focus = self.floor + (1.0 - self.floor) * _ramp(a, self.focus_lo, self.focus_hi)
        m_ambient = self.floor + (1.0 - self.floor) * _ramp(1.0 - a, self.ambient_lo, self.ambient_hi)
        img = frame_u8.astype(np.float32)
        # rint 后再转 uint8：astype 直接截断会把 200×0.5≈100.00004 这类边界值截成 99
        focus = np.clip(np.rint(img * m_focus[..., None]), 0, 255).astype(np.uint8)
        ambient = np.clip(np.rint(img * m_ambient[..., None]), 0, 255).astype(np.uint8)
        return focus, ambient


class StreamRecorder:
    """焦点/氛围两路调制流录制成两份 MJPG AVI（display.record_dir 非空时启用）。

    帧率取 capture.target_fps 作名义值：展示路按采集节拍跑，录制品只用于回看演示，
    不承诺精确时轴。文件由 close() 落盘收尾。
    """

    def __init__(self, out_dir: str, nominal_fps: float) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(nominal_fps)
        self._writers: dict[str, cv2.VideoWriter] = {}

    def write(self, name: str, frame: np.ndarray) -> None:
        w = self._writers.get(name)
        if w is None:
            h, wd = frame.shape[:2]
            w = cv2.VideoWriter(
                str(self.dir / f"{name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (wd, h)
            )
            if not w.isOpened():
                raise RuntimeError(f"无法创建录制文件: {self.dir / (name + '.avi')}")
            self._writers[name] = w
        w.write(frame)

    def close(self) -> None:
        for w in self._writers.values():
            w.release()
        self._writers.clear()
