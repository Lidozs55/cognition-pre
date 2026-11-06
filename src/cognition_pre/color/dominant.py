"""M4 主色计算：加权 Mini-Batch K-Means + v1 打分器。

两个并行实例只差一个权重：焦点色 w = ramp(A; lo, hi)，氛围色 w = ramp(1−A; lo, hi)——
与展示路调制同形的双阈值分段线性（utils/ramp.py 同一份实现），A<lo 的像素从本世界
剔除（权重要能压到 0：旧式 w=1+λ·A 给背景保底权重，大面积背景按面积碾压主体，λ 排
不掉，实测被背景主导）。权重掩膜在全分辨率上先定（只做一次比较），抽样只发生在本
世界像素上、密度按本世界面积自适应（主体小也得满预算，见 _sample_world）；权重只在
抽样像素上计算、只乘一次（sample_weight），无 w² 双重强调，进 K-Means 影响簇心与打分。

空间选择已在权重里完成，故聚类特征用 Lab（a/b 高权重、L 低权重——L 不必再兼职掩膜）。
最终 RGB 一律回到原图像素求加权平均——颜色永远从原图读。

打分器 v1（Phase 2，按 Itti & Koch 2001 细化；机制映射与设计依据全文见 docs/design.md）：
  score = 质量 × 彩度 × 紧凑度 × 对比度
  - 彩度系数 (0.45+0.55c) 比其余两项陡（鲜艳度影响力略高，2026-11-05 反馈）；
  - 对比度：焦点实例基准 = 上一帧氛围主色（center-surround，贴背景色打折），
    无参照退回全帧均色；由 pipeline 单向接线，避免双实例互推的反馈环；
  - 迟滞选主：保护门槛 = margin × 动量，三态动力学见 _select；
  - warm-start：上轮簇心作本轮初始化（簇集合时域连贯，迟滞的前提）；
  - 近邻簇心合并：K 固定，簇心距离 < merge_dist 并族（动态类数），见 _merge_groups。
置信度输出为支配度（胜者得分 / 全簇得分和），见 DominantResult.dominance。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ..config import ColorConfig
from ..utils.ramp import ramp

# 经验尺度常数：归一 Lab（L∈[0,1]、a/b∈[-1,1]）空间里"色相级差异"的量级（a/b 全幅 2.0）
_CONTRAST_NORM = 0.5

# 迟滞动量动力学常数（三态语义与自锁复盘见 _select docstring / docs/design.md）
_MOM_BUILD = 0.2  # 连胜每帧攒 0.2，约 5 帧满
_MOM_LEAK = 0.85  # 被压制挑战者慢泄：持续小优势 ~0.5-1s 夺主（偶发噪声被连胜回填对冲）
_MOM_RELEASE = 0.6  # 换景快泄
_MOM_LAB_ALPHA = 0.3  # 参考色 = 赢家 Lab 慢 EMA（MoCo 式动量更新）


def _score(mass: float, chroma: float, compactness: float, contrast: float) -> float:
    """v1 打分公式。彩度系数 (0.45+0.55c) 特意比紧凑度/对比度的 (0.6+0.4x) 陡——
    鲜艳度影响力略高（2026-11-05 实测反馈），端点值由 test_score_formula_chroma_tilt 固定。"""
    return mass * (0.45 + 0.55 * chroma) * (0.6 + 0.4 * compactness) * (0.6 + 0.4 * contrast)


@dataclass
class ColorCandidate:
    rgb: tuple[float, float, float]
    lab01: tuple[float, float, float]  # 归一 Lab（未乘空间权重）
    score: float
    mass_frac: float  # 加权质量占比
    compactness: float  # 0-1，特征空间簇内紧凑度
    chroma: float  # 0-1，彩度（a/b 模长归一）
    contrast: float  # 0-1，与对比度基准（背景主色，缺省全帧均色）的 Lab 距离归一


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
        self._mom_lab: np.ndarray | None = None  # 动量参考色（赢家 Lab 慢 EMA）
        self._mom = 0.0  # 迟滞动量 [0,1]：连胜攒、断连胜泄，调制保护门槛
        self._prev_centers: np.ndarray | None = None  # 上轮簇心（warm-start，簇集合时域连贯）

    def compute(
        self, rgb_u8: np.ndarray, s01: np.ndarray, surround_ref: tuple | None = None
    ) -> DominantResult:
        """rgb_u8: 裁切帧 RGB uint8（读色基准）；s01: 同尺寸注意力图 [0,1]（两个实例
        都传 A，invert=True 的实例内部翻转为 1−A）。

        surround_ref: 对侧世界（背景）主色的 lab01，作对比度基准（center-surround：
        贴背景色的候选打折，保护色不显眼）；None 时退回全帧均色。仅焦点实例由
        pipeline 传入上一帧氛围赢家——单向，避免双实例互推的反馈环。"""
        cfg = self.cfg
        h, w = rgb_u8.shape[:2]
        n_all = h * w
        min_pts = max(8, cfg.n_clusters * 4)
        if n_all < min_pts:
            return DominantResult(None, [])
        step = max(1, round((n_all / max(cfg.sample_budget, 1)) ** 0.5))

        # 1) 本世界掩膜（全分辨率，只做一次比较；ramp 的 w>0 ⟺ s>lo 常规域）。预算分配
        #    以此为准——主体占画面小时，全帧等步长网格会把绝大部分预算漏给零权重背景
        #    格点（1080p 下 5% 主体旧路径只得 ~600 样本），两世界互相"饥饿"
        in_world = (s01 < 1.0 - cfg.weight_lo) if self.invert else (s01 > cfg.weight_lo)
        n_world = int(in_world.sum())

        # 2) 场景基准 = 全帧粗网格 Lab 均值（含两个世界；对比度缺参照时的回退基准）
        scene_mean = self._to_lab01(
            rgb_u8[::step, ::step].reshape(-1, 3).astype(np.float32)
        ).mean(axis=0)

        # 3) 本世界抽样（≤预算全取 / 超预算按本世界面积自适应步长 / 退化回退），权重只在
        #    抽样像素上算——只乘一次，无 w² 双重强调
        x_rgb, ws = self._sample_world(rgb_u8, s01, in_world, n_world, step)

        # 4) 抽样像素 Lab 特征
        lab01 = self._to_lab01(x_rgb)
        x_feat = lab01 * self._w_scale
        m = x_rgb.shape[0]

        # 5) Mini-Batch K-Means（sample_weight 进入簇心与打分）。
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
        self._prev_centers = km.cluster_centers_.copy()  # warm-start 存合并前原始簇心（K 恒定）
        labels = km.labels_
        total = float(ws.sum())

        # 6) 近邻簇心合并（动态类数，2026-11-06）：K 固定是 warm-start 的前提，动态性
        #    放在合并层——簇心距离 < merge_dist 的簇并为一族再统计，同色系被 K-Means
        #    拆出的深浅碎片合成一个候选；真多彩场景互距远、不丢分辨率。
        wsums = np.bincount(labels, weights=ws, minlength=k)  # 各簇加权质量（一次扫描）
        groups = self._merge_groups(km.cluster_centers_, wsums, cfg.merge_dist)

        # 7) 族统计：RGB 回原图加权均值，按 v1 公式打分。对比度基准 = 对侧世界主色
        #    （center-surround，由 pipeline 传入；缺省退回全帧均色）
        contrast_ref = (
            np.asarray(surround_ref, dtype=np.float64)
            if surround_ref is not None
            else scene_mean
        )
        candidates: list[ColorCandidate] = []
        for members in groups:
            sel = np.isin(labels, members)
            if not sel.any():
                continue
            wc = ws[sel]
            wsum = float(wc.sum())
            if wsum <= 0:
                continue
            rgb_c = (x_rgb[sel] * wc[:, None]).sum(axis=0) / wsum
            mu = x_feat[sel].mean(axis=0)
            labc = mu / self._w_scale
            dist = float(np.linalg.norm(x_feat[sel] - mu, axis=1).mean())
            # 1.73 为经验尺度常数（≈无权重归一特征空间的跨度量级）：压缩值域防单因子归零
            compactness = 1.0 / (1.0 + 2.0 * dist / 1.7320508)
            aa, bb = float(labc[1]), float(labc[2])
            chroma = min(1.0, (aa * aa + bb * bb) ** 0.5 / 0.45)
            contrast = min(1.0, float(np.linalg.norm(labc - contrast_ref)) / _CONTRAST_NORM)
            mass = wsum / total
            score = _score(mass, chroma, compactness, contrast)
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

    def _sample_world(
        self, rgb_u8: np.ndarray, s01: np.ndarray, in_world: np.ndarray, n_world: int, step: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """本世界抽样，返回 (rgb float32(N,3), ws(N))。

        预算按本世界面积自适应消费：n_world ≤ 预算全取（带内像素一个不漏，权重保留
        ramp 线性值）；超预算按 n_world 定步长（而非全帧面积）的网格交掩膜，样本数
        ≈预算，起点随机抖动抗周期内容走样。本世界几乎无合格像素（如平坦热图后处理
        全零）时退化为全帧等步长无权重，保证 10Hz 输出不断流。
        """
        cfg = self.cfg
        if n_world < max(8, cfg.n_clusters * 4):
            rgb_s = rgb_u8[::step, ::step].reshape(-1, 3)
            return rgb_s.astype(np.float32), np.ones(rgb_s.shape[0], np.float32)
        if n_world <= cfg.sample_budget:
            rgb_s, s_sel = rgb_u8[in_world], s01[in_world]
        else:
            step_w = max(1, int(np.ceil((n_world / cfg.sample_budget) ** 0.5)))
            oy, ox = (int(v) for v in self.rng.integers(0, step_w, size=2))
            g = in_world[oy::step_w, ox::step_w]
            rgb_s, s_sel = rgb_u8[oy::step_w, ox::step_w][g], s01[oy::step_w, ox::step_w][g]
        s_eff = (1.0 - s_sel) if self.invert else s_sel
        return rgb_s.astype(np.float32), ramp(s_eff, cfg.weight_lo, cfg.weight_hi)

    @staticmethod
    def _merge_groups(centers: np.ndarray, wsums: np.ndarray, merge_dist: float) -> list[list[int]]:
        """贪心近邻合并：簇心距离 < merge_dist 的簇反复并族，族心 = 加权质量均值。

        用族均值重算距离而非只在原始簇心上做连通性——链式近邻（A 贴 B、B 贴 C 而
        A/C 相远）不会把远簇拖进来。k<=5，至多 4 次合并，开销可忽略。返回原簇号的
        族划分（merge_dist<=0 时每簇自成一组，即关闭合并）。
        """
        groups = [[i] for i in range(centers.shape[0])]
        if merge_dist <= 0.0 or len(groups) <= 1:
            return groups
        means = [c.astype(np.float64).copy() for c in centers]
        wts = [float(w) for w in wsums]
        while len(groups) > 1:
            best = min(
                (float(np.linalg.norm(means[i] - means[j])), i, j)
                for i in range(len(groups))
                for j in range(i + 1, len(groups))
            )
            if best[0] >= merge_dist:
                break
            _, i, j = best
            w = wts[i] + wts[j]
            if w > 0:
                means[i] = (means[i] * wts[i] + means[j] * wts[j]) / w
            else:  # 双空簇（重复簇心无样本）：退化为算术平均，避免 0/0
                means[i] = (means[i] + means[j]) / 2.0
            wts[i] = w
            groups[i] = groups[i] + groups[j]
            del groups[j], means[j], wts[j]
        return groups

    def _select(self, candidates: list[ColorCandidate]) -> ColorCandidate | None:
        """动量化迟滞选主：有效保护门槛 = margin × 动量（注视稳定性时序类似物）。

        动量三态（复盘自锁缺陷后定案，2026-11-06）：旧式从第 2 帧起恒享全额 margin，
        起选错误会一直锁着；若只把泄放挂在"赢家换人"上，满动量下优势 < margin 的
        挑战者永远赢不了一帧、动量永远不泄——门槛自锁，弱化只作用于起选初期。
        三态修复：① 连胜（赢家贴参考色且本就是分数赢家）→ 攒，静态约 5 帧满，
        全额抑制同屏近分闪烁；② 被门槛压制（挑战者分数更高却被压下）→ 慢泄 ×0.85，
        持续小优势 ~0.5-1s 内夺主，偶发噪声的压制被连胜回填对冲；③ 换景（赢家远离
        参考色）→ 快泄 ×0.6 + 参考色跟新赢家。参考色本身是赢家 Lab 的 MoCo 式
        动量更新（慢 EMA）。
        """
        if not candidates:
            return None
        raw_best = max(candidates, key=lambda c: c.score)
        best = raw_best
        ref = self._mom_lab
        eff_margin = self.cfg.hysteresis_margin * self._mom
        if eff_margin > 0.0 and ref is not None:
            near = [
                c
                for c in candidates
                if float(np.linalg.norm(np.asarray(c.lab01) - ref)) <= self.cfg.hysteresis_dist
            ]
            if near:
                incumbent = max(near, key=lambda c: c.score)
                if best.score <= incumbent.score * (1.0 + eff_margin):
                    best = incumbent
        near_ref = ref is not None and (
            float(np.linalg.norm(np.asarray(best.lab01) - ref)) <= self.cfg.hysteresis_dist
        )
        if best is not raw_best:
            self._mom *= _MOM_LEAK
        elif near_ref:
            self._mom = min(1.0, self._mom + _MOM_BUILD)
            self._mom_lab = (1.0 - _MOM_LAB_ALPHA) * ref + _MOM_LAB_ALPHA * np.asarray(
                best.lab01, dtype=np.float64
            )
        else:
            self._mom *= _MOM_RELEASE
            self._mom_lab = np.asarray(best.lab01, dtype=np.float64)
        return best

    @staticmethod
    def _to_lab01(rgb_f32: np.ndarray) -> np.ndarray:
        """RGB(N,3) float32 → 归一 Lab(N,3)；cvtColor 只认图像形状，像素列表摆成单行。"""
        u8 = np.clip(rgb_f32, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(u8.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        lab[..., 0] /= 255.0
        lab[..., 1:] = (lab[..., 1:] - 128.0) / 128.0
        return lab
