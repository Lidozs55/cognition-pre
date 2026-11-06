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
    """热图后处理：轻度平滑（可选）→ 截掉前 top_pct 极值 → 背景平台抑制归一到 [0,1]。

    smooth_frac：高斯核边长 = 网格边长 × frac（取奇、≥1），核尺寸显式受控——不用
    sigma 语义（OpenCV 按 3σ 自动核，sigma=1 即 7×7，把主体尖角抹平，用户实测翻车）。
    默认 224² 网格不模糊、518² 3×3：背景散斑已由 norm_floor_pct 治理，平滑只做轻度
    并斑。CLS 注意力是"信息密度"图而非"主体"图。0 关闭。
    clip 先于归一（在原始值域上取分位数）：若先归一，单个极端离群会把其余结构压到
    0 附近，分位数随之失真。top-pct 截断线落在背景平台上时（注意力集中在少于
    top_pct 的 patch 上，属锐利而非病态的热图）退到最小非背景值，防止截断把唯一
    信号整幅抹掉。
    norm_floor_pct：可选背景平台抑制——把该分位的值压到 0（背景高注意力泄漏的对策；
    平滑治不了泄漏，泄漏要在归一化里治）。无直接论文依据，且隐含"主体 <50% 画面"
    假设（特写误伤主体），故默认 0=关闭（min-max 归一）。
    """

    def __init__(
        self,
        top_clip_pct: float = 0.05,
        smooth_frac: float = 0.05,  # 与 configs/default.yaml 一致；生产路径由 pipeline 按配置传入
        norm_floor_pct: float = 0.0,  # 默认关（同上）：类内默认只作独立使用兜底，勿另设一套
    ) -> None:
        self.pct = float(top_clip_pct)
        self.smooth_frac = float(smooth_frac)
        self.floor_pct = float(norm_floor_pct)

    def __call__(self, attn: np.ndarray) -> np.ndarray:
        a = np.asarray(attn, dtype=np.float32)
        if float(a.max()) <= float(a.min()):
            return np.zeros_like(a)
        if self.smooth_frac > 0:
            ph, pw = a.shape[:2]
            k = int(np.ceil(self.smooth_frac * min(ph, pw)))
            k = max(1, k + 1 if k % 2 == 0 else k)
            if k > 1:
                a = cv2.GaussianBlur(a, (k, k), 0.0)  # sigma 随核尺寸推导（OpenCV 标准）
        if self.pct > 0:
            thr = float(np.quantile(a, 1.0 - self.pct))
            if thr <= float(a.min()):
                above = a[a > a.min()]
                thr = float(above.min()) if above.size else float(a.max())
            a = np.minimum(a, thr)
        hi = float(a.max())
        lo = float(np.quantile(a, self.floor_pct)) if self.floor_pct > 0 else float(a.min())
        if hi <= lo:
            return np.zeros_like(a)
        a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        return a.astype(np.float32)


def upsample(attn: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """patch 网格热图双线性上采样到裁切坐标系 (H, W)。"""
    h, w = size_hw
    return cv2.resize(attn.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
