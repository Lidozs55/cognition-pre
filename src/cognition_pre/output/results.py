"""管线最终输出结构：{焦点色RGB, 氛围色RGB, 置信度×2, 时间戳}。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameResult:
    ts: float  # epoch 秒
    frame_seq: int  # 生成热图时的帧序号；focus/ambient 颜色取自 >= 该序号的最新裁切帧（color 线程从两个 slot 各取最新，可能混帧）
    focus_rgb: tuple[float, float, float]
    ambient_rgb: tuple[float, float, float]
    focus_conf: float
    ambient_conf: float

    def to_dict(self) -> dict:
        return {
            "ts": round(self.ts, 3),
            "frame_seq": self.frame_seq,
            "focus_rgb": [round(v) for v in self.focus_rgb],
            "ambient_rgb": [round(v) for v in self.ambient_rgb],
            "focus_conf": round(self.focus_conf, 4),
            "ambient_conf": round(self.ambient_conf, 4),
        }
