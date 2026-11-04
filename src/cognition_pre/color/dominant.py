"""M4 主色计算：加权 Mini-Batch K-Means + v1 打分器。

两个并行实例只差一个权重：焦点色 w = 1 + λ·A(x)，氛围色 w = 1 + λ·(1−A(x))。
权重经 sample_weight 进入 K-Means，影响簇心与打分（超预算时像素均匀抽样，
不按权重采样——避免"采样概率 ∝ w 再乘 w"的双重强调，见 compute 内注释）。

空间选择已在权重里完成，故聚类特征用 Lab（a/b 高权重、L 低权重——L 不必再兼职掩膜，
这正是"展示/计算分离"的红利）。最终 RGB 一律回到原图像素求加权平均——颜色永远从原图读。

打分器 v1（Phase 2，按 Itti & Koch 2001 细化，机制映射见 docs/design.md）：
  score = 质量 × 彩度 × 紧凑度 × 对比度
  - 质量/彩度/紧凑度：v0 遗留，回答"这个簇能不能代表它所在的世界"；
  - 对比度（center-surround / 空间竞争的近似映射）：簇色与全帧均色的归一 Lab 距离，
    回答"这个簇在整幅画面里是否醒目"——焦点世界的"主角色"由此胜出；
  - 迟滞选主（top-down attentional bias / 注视稳定性的时序类似物。注：论文的 IOR
    机制方向相反——抑制在位赢家、强制扫视前进，与本机制"保护在位主色"相反，
    故不标 IOR）：与上帧主色接近的"在位簇"仅在挑战簇有 (1+margin) 分数优势时
    让位，抑制同屏近似簇之间的主色闪烁。
  - warm-start：上轮簇心作本轮初始化（簇集合时域连贯，迟滞的前提），静态画面
    每轮的簇心不再从随机状态重聚。
置信度输出为支配度（胜者得分 / 全簇得分和），见 DominantResult.dominance。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ..config import ColorConfig

# 经验尺度常数：归一 Lab（L∈[0,1]、a/b∈[-1,1]）空间里"色相级差异"的量级（a/b 全幅 2.0）
_CONTRAST_NORM = 0.5


@dataclass
class ColorCandidate:
    rgb: tuple[float, float, float]
    lab01: tuple[float, float, float]  # 归一 Lab（未乘空间权重）
    score: float
    mass_frac: float  # 加权质量占比
    compactness: float  # 0-1，特征空间簇内紧凑度
    chroma: float  # 0-1，彩度（a/b 模长归一）
    contrast: float  # 0-1，与全帧均色的 Lab 距离归一


@dataclass
class DominantResult:
    best: ColorCandidate | None
    candidates: list[ColorCandidate]
    dominance: float = 0.0  # 胜者得分 / 全簇得分和；置信度的原始量（进 EMA 前的语义）


class WeightedDominantColor:
    """单个"世界"（焦点或氛围）的加权主色器。invert=True 表示用 1−A 做权重。"""

    def __init__(self, cfg: ColorConfig, invert: bool = False, name: str = "") -> None:
        self.cfg = cfg
        self.invert = invert
        self.name = name
        self.rng = np.random.default_rng(cfg.seed)
        self._w_scale = np.asarray(cfg.lab_weights, dtype=np.float32)
        self._prev_lab: np.ndarray | None = None  # 上帧选中的主色（迟滞基准）
        self._prev_centers: np.ndarray | None = None  # 上轮簇心（warm-start，簇集合时域连贯）

    def compute(self, rgb_u8: np.ndarray, s01: np.ndarray) -> DominantResult:
        """rgb_u8: 裁切帧 RGB uint8（读色基准）；s01: 同尺寸权重图 [0,1]（A 或 1−A）。"""
        cfg = self.cfg
        h, w = rgb_u8.shape[:2]
        n_all = h * w

        # 1) 网格降采样控制预算
        step = max(1, round((n_all / max(cfg.sample_budget, 1)) ** 0.5))
        rgb_d = rgb_u8[::step, ::step].reshape(-1, 3).astype(np.float32)
        s_d = s01[::step, ::step].reshape(-1).astype(np.float32)
        wts = 1.0 + cfg.lam * ((1.0 - s_d) if self.invert else s_d)
        n = rgb_d.shape[0]
        if n < max(8, cfg.n_clusters * 4):
            return DominantResult(None, [])

        # 2) 全网格 Lab 特征：对比度基准（全帧均值）与聚类采样共用一次转换
        lab01 = self._to_lab01(rgb_d)
        feat_grid = lab01 * self._w_scale
        scene_mean = lab01.mean(axis=0)

        # 3) 超预算时均匀无回退抽样，权重只走 sample_weight——保持 w 的线性语义
        #    （若按 p∝w 采样再乘 w 权重会产生 w² 双重强调，且与 n≤budget 分支不一致）
        if n > cfg.sample_budget:
            idx = self.rng.choice(n, size=cfg.sample_budget, replace=False)
            x_feat, x_rgb, ws = feat_grid[idx], rgb_d[idx], wts[idx]
        else:
            x_feat, x_rgb, ws = feat_grid, rgb_d, wts
        m = x_feat.shape[0]

        # 4) Mini-Batch K-Means（sample_weight 进入簇心与打分）。
        #    warm-start（防闪 A）：接得上上轮簇心时只做一次初始化——簇集合时域连贯，
        #    候选颜色跨帧可比（迟滞选主的前提），且收敛更快；簇数对不上（画面尺寸骤变）
        #    时退回默认初始化。
        k = min(cfg.n_clusters, max(2, m // 16))
        prev = self._prev_centers
        if prev is not None and prev.shape[0] == k:
            km = MiniBatchKMeans(
                n_clusters=k,
                init=prev,
                n_init=1,
                batch_size=min(cfg.kmeans_batch_size, m),
                random_state=cfg.seed,
            )
        else:
            km = MiniBatchKMeans(
                n_clusters=k,
                batch_size=min(cfg.kmeans_batch_size, m),
                n_init=cfg.kmeans_n_init,
                random_state=cfg.seed,
            )
        km.fit(x_feat, sample_weight=ws)
        self._prev_centers = km.cluster_centers_.copy()
        labels = km.labels_
        total = float(ws.sum())

        # 5) 簇统计：RGB 回原图加权均值，按 v1 公式打分
        candidates: list[ColorCandidate] = []
        for c in range(k):
            sel = labels == c
            if not sel.any():
                continue
            wc = ws[sel]
            wsum = float(wc.sum())
            if wsum <= 0:
                continue
            rgb_c = (x_rgb[sel] * wc[:, None]).sum(axis=0) / wsum
            labc = x_feat[sel].mean(axis=0) / self._w_scale
            dist = float(np.linalg.norm(x_feat[sel] - km.cluster_centers_[c], axis=1).mean())
            # 1.73 为经验尺度常数（≈无权重归一特征空间的跨度量级）：压缩值域防单因子归零
            compactness = 1.0 / (1.0 + 2.0 * dist / 1.7320508)
            aa, bb = float(labc[1]), float(labc[2])
            chroma = min(1.0, (aa * aa + bb * bb) ** 0.5 / 0.45)
            contrast = min(1.0, float(np.linalg.norm(labc - scene_mean)) / _CONTRAST_NORM)
            mass = wsum / total
            score = mass * (0.55 + 0.45 * chroma) * (0.6 + 0.4 * compactness) * (0.6 + 0.4 * contrast)
            candidates.append(
                ColorCandidate(
                    rgb=tuple(float(v) for v in rgb_c),
                    lab01=tuple(float(v) for v in labc),
                    score=float(score),
                    mass_frac=float(mass),
                    compactness=float(compactness),
                    chroma=float(chroma),
                    contrast=float(contrast),
                )
            )

        candidates = [c for c in candidates if c.mass_frac >= cfg.min_mass_frac] or candidates
        best = self._select(candidates)
        dominance = 0.0
        if best is not None:
            total_score = sum(c.score for c in candidates)
            dominance = best.score / total_score if total_score > 0 else 0.0
        return DominantResult(best, sorted(candidates, key=lambda c: -c.score), dominance)

    # ---- 内部 ------------------------------------------------------------

    def _select(self, candidates: list[ColorCandidate]) -> ColorCandidate | None:
        """迟滞选主：在位簇仅在挑战簇有 (1+margin) 分数优势时让位（注视稳定性时序类似物）。"""
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c.score)
        margin = self.cfg.hysteresis_margin
        if margin > 0 and self._prev_lab is not None:
            near = [
                c
                for c in candidates
                if float(np.linalg.norm(np.asarray(c.lab01) - self._prev_lab)) <= self.cfg.hysteresis_dist
            ]
            if near:
                incumbent = max(near, key=lambda c: c.score)
                if best.score <= incumbent.score * (1.0 + margin):
                    best = incumbent
        self._prev_lab = np.asarray(best.lab01, dtype=np.float64)
        return best

    @staticmethod
    def _to_lab01(rgb_f32: np.ndarray) -> np.ndarray:
        """RGB(N,3) float32 → 归一 Lab(N,3)；cvtColor 只认图像形状，像素列表摆成单行。"""
        u8 = np.clip(rgb_f32, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(u8.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        lab[..., 0] /= 255.0
        lab[..., 1:] = (lab[..., 1:] - 128.0) / 128.0
        return lab
