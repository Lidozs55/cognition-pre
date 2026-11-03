"""M1 输入采集：统一帧源抽象。

- CameraFileSource：OBS 虚拟摄像头 / 本机摄像头 / 视频文件（cv2.VideoCapture 统一处理）。
- SyntheticSource："test:split" / "test:gradient" 合成源，无摄像头无素材时联调与测试用。

铁律：所有颜色统计在解码后的 RGB 上做——本模块只交出 BGR 帧（cv2 原生格式），
RGB 转换发生在下游边界（模型输入、颜色计算）。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import cv2
import numpy as np

from ..config import CaptureConfig

_CV_BACKENDS = {
    "auto": None,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "ffmpeg": cv2.CAP_FFMPEG,
}


class FrameSource(ABC):
    is_file = False

    @abstractmethod
    def open(self) -> bool: ...

    @abstractmethod
    def read(self) -> np.ndarray | None: ...

    def rewind(self) -> bool:
        return False

    @abstractmethod
    def close(self) -> None: ...


class CameraFileSource(FrameSource):
    """摄像头（索引）或视频文件（路径）。文件源按 target_fps 节拍播放以模拟实时。"""

    def __init__(self, source: str, backend: str = "auto", target_fps: float = 30.0) -> None:
        self._source = int(source) if source.isdigit() else source
        self._backend = _CV_BACKENDS.get(backend)
        self._fps = target_fps
        self.is_file = not source.isdigit()
        self._cap: cv2.VideoCapture | None = None
        self._last = 0.0

    def open(self) -> bool:
        if self._backend is None:
            self._cap = cv2.VideoCapture(self._source)
        else:
            self._cap = cv2.VideoCapture(self._source, self._backend)
        return self._cap.isOpened()

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if ok and self.is_file:
            now = time.monotonic()
            wait = 1.0 / max(self._fps, 1e-6) - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        return frame if ok else None

    def rewind(self) -> bool:
        if self._cap is not None and self.is_file:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True
        return False

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SyntheticSource(FrameSource):
    """合成源：test:split（左红右蓝）| test:gradient（水平渐变）。480x360@30fps。"""

    is_file = True

    def __init__(self, spec: str = "split", size: tuple[int, int] = (480, 360), fps: float = 30.0) -> None:
        self.spec = spec
        self.size = size
        self.fps = fps
        self._last = 0.0

    def open(self) -> bool:
        self._last = 0.0
        return True

    def read(self) -> np.ndarray | None:
        now = time.monotonic()
        wait = 1.0 / max(self.fps, 1e-6) - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

        h, w = self.size
        if self.spec == "gradient":
            ramp = np.linspace(0, 255, w, dtype=np.uint8)
            img = np.stack([ramp, 255 - ramp, np.full(w, 128, np.uint8)], axis=-1)  # BGR 列渐变
            return np.tile(img[None, :, :], (h, 1, 1))
        # 默认 split：BGR 左红右蓝（RGB: 左 (200,30,30)，右 (30,60,200)）
        img = np.zeros((h, w, 3), np.uint8)
        img[:, : w // 2] = (30, 30, 200)
        img[:, w // 2 :] = (200, 60, 30)
        return img

    def close(self) -> None: ...


def open_source(cap_cfg: CaptureConfig) -> FrameSource:
    s = cap_cfg.source
    if s.startswith("test:"):
        return SyntheticSource(spec=s[len("test:") :] or "split")
    return CameraFileSource(s, cap_cfg.backend, cap_cfg.target_fps)
