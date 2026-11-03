"""M5 双流亮度调制（展示路）：它本身就是这套系统最好的解释图。

焦点流 mask = γ + (1−γ)·A(x)：低注意力区压暗；
氛围流 mask = γ + (1−γ)·(1−A(x))：高注意力区压暗。
逐像素乘法，全帧率跑（用最近热图，与热图更新解耦）。

铁律：输出只给人看，严禁进入任何颜色计算（展示/计算分离）。
录制（StreamRecorder）同属展示路：只写不读，与计算路零耦合。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import ModulationConfig


class DualStreamModulator:
    def __init__(self, cfg: ModulationConfig) -> None:
        self.gamma = float(cfg.gamma)
        self.enabled = bool(cfg.enabled)

    def apply(self, frame_u8: np.ndarray, s01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (focus_stream, ambient_stream)，与输入同尺寸同格式。s01 为 A(x) [0,1]。"""
        a = np.clip(np.asarray(s01, dtype=np.float32), 0.0, 1.0)
        m_focus = self.gamma + (1.0 - self.gamma) * a
        m_ambient = self.gamma + (1.0 - self.gamma) * (1.0 - a)
        img = frame_u8.astype(np.float32)
        focus = np.clip(img * m_focus[..., None], 0, 255).astype(np.uint8)
        ambient = np.clip(img * m_ambient[..., None], 0, 255).astype(np.uint8)
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
