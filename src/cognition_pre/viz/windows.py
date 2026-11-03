"""预览画布：裁切帧 | 焦点流 | 氛围流 / 热图叠加 + 色板与状态。

cv2 窗口必须在主线程；画布文字用英文（Hershey 字体不渲染中文）。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import DisplayConfig
from ..output.results import FrameResult
from ..utils.fps import FpsMeter

_WINDOW = "cognition-pre  [q]quit  [h]heatmap"


class PreviewWindows:
    def __init__(self, cfg: DisplayConfig) -> None:
        self.cfg = cfg
        self.show_heat = cfg.show_heatmap
        cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)

    def draw(
        self,
        crop_bgr: np.ndarray,
        focus_bgr: np.ndarray,
        ambient_bgr: np.ndarray,
        attn01: np.ndarray | None,
        result: FrameResult | None,
        fps: dict[str, FpsMeter],
    ) -> int:
        """渲染一帧并返回 waitKey 按键码。"""
        w = self.cfg.window_width

        def tile(img: np.ndarray) -> np.ndarray:
            return cv2.resize(img, (w, w), interpolation=cv2.INTER_AREA)

        row1 = np.hstack([tile(crop_bgr), tile(focus_bgr), tile(ambient_bgr)])

        heat = np.zeros((w, w, 3), np.uint8)
        if self.show_heat and attn01 is not None:
            hm = cv2.applyColorMap((np.clip(attn01, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat = cv2.addWeighted(tile(crop_bgr), 0.55, cv2.resize(hm, (w, w)), 0.45, 0)

        bar = np.zeros((w, 2 * w, 3), np.uint8)
        if result is not None:
            self._swatch(bar, (10, 10, 2 * w - 20, w // 2 - 24), result.focus_rgb, "FOCUS", result.focus_conf)
            self._swatch(bar, (10, w // 2 - 4, 2 * w - 20, w // 2 - 24), result.ambient_rgb, "AMBIENT", result.ambient_conf)
        cv2.putText(
            bar,
            f"cap {fps['capture'].value:4.1f} | attn {fps['attention'].value:4.1f} | color {fps['color'].value:4.1f} Hz",
            (12, w - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

        canvas = np.vstack([row1, np.hstack([heat, bar])])
        cv2.imshow(_WINDOW, canvas)
        return cv2.waitKey(1) & 0xFF

    @staticmethod
    def _swatch(bar: np.ndarray, rect: tuple[int, int, int, int], rgb, label: str, conf: float) -> None:
        x, y, rw, rh = rect
        b, g, r = round(rgb[2]), round(rgb[1]), round(rgb[0])
        cv2.rectangle(bar, (x, y), (x + rw, y + rh), (b, g, r), thickness=-1)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        color = (0, 0, 0) if lum > 128 else (255, 255, 255)
        cv2.putText(bar, f"{label} rgb({r},{g},{b}) conf={conf:.2f}", (x + 10, y + rh // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def close(self) -> None:
        cv2.destroyAllWindows()
