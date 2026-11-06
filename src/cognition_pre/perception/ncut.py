"""NCut 谱分割（TokenCut-lite 核心）：patch 特征余弦亲和 → 归一化割第二特征向量。

单一实现：主色路后端（dinov2_backend，ncut_color=true）与诊断脚本
（scripts/compare_heatmaps.py / heatmap_stability.py）共用。2026-11-06 真素材
判读（注意力散斑/掩膜超纳入 54%，谱分割 5 图 4 胜）后集成——主色世界掩膜用
谱分割，展示路仍注意力（design.md M3/M4）。torch 惰性导入（core 安装不含 torch，
mock 路径与单测永不触碰）。

极性 = 第二特征向量的符号歧义怎么定向（三版演进，判读史见 roadmap Phase 1）：
- v1 中心先验：离中心主体整个翻反（dog 案）→ 弃；
- v2 注意力粗质量：文本/涂鸦密集场景注意力质量本身在背景上，自信判反
  （截图 1259 案，r=0.51）→ 弃；
- v3（现行）级联：|corr(sal, 度数)|>0.5 → 度数锚（高连通侧 = 同质大面积簇 =
  背景；Fiedler 向量对度数向量定向是谱聚类惯例），否则 → 紧凑度锚（空间方差
  小的一侧 = 主体，背景沿画面铺开）。6 图判读：生产尺度 224 全对；518 有 1 例
  （dog@518）各锚皆不可分——跨尺度分割几何不同，按侧锚定原理上难两全（见
  roadmap）。阈值区间 (0.44, 0.59) 取中，样本量小、裕度不厚，新场景极性可疑
  先跑 compare 工具核对 ncut 行。
"""

from __future__ import annotations

import numpy as np


def _spatial_var(mask_flat: np.ndarray, side: int) -> float:
    """一侧 patch 的空间方差（归一坐标）；不足 2 点视为极散。"""
    if mask_flat.sum() < 2:
        return float("inf")
    g = np.linspace(-1.0, 1.0, side)
    gy, gx = np.meshgrid(g, g, indexing="ij")
    return float(np.var(gy.ravel()[mask_flat]) + np.var(gx.ravel()[mask_flat]))


def align_polarity(sal: np.ndarray, degree: np.ndarray, side: int) -> np.ndarray:
    """定向第二特征向量：正侧=主体。纯 numpy（可单测）。

    度数锚自信（|corr|>0.5）时高连通侧判为背景（同质大面积簇），否则按紧凑度
    （散的一侧=背景）。返回值与输入等长（展平向量）。
    """
    r_deg = 0.0
    if float(degree.std()) > 1e-8:
        r_deg = float(np.corrcoef(sal, degree)[0, 1])
    if abs(r_deg) > 0.5:
        flip = r_deg > 0.0  # 正侧=高连通=背景
    else:
        flip = _spatial_var(sal > 0, side) > _spatial_var(sal < 0, side)  # 正侧更散=背景
    return -sal if flip else sal


def ncut_map(patch_feat, side: int) -> np.ndarray:
    """(N, D) patch 特征 → (side, side) 谱图（未归一；后续过 HeatmapPostprocessor）。

    松弛 NCut 的第二小特征向量 = 归一化亲和矩阵 D^-1/2 W D^-1/2 的第二大特征向量。
    """
    import torch  # 惰性导入：core 安装不含 torch

    f = torch.nn.functional.normalize(patch_feat.float(), dim=-1)
    w = f @ f.T
    w.fill_diagonal_(0)
    degree = w.sum(dim=1)
    d_inv_sqrt = torch.rsqrt(torch.clamp(degree, min=1e-8))
    norm_adj = d_inv_sqrt[:, None] * w * d_inv_sqrt[None, :]
    _vals, vecs = torch.linalg.eigh(norm_adj)  # 升序；[-1] 是平凡特征向量
    sal = vecs[:, -2].cpu().numpy().astype(np.float32)
    sal = align_polarity(sal, degree.cpu().numpy().astype(np.float32), side)
    return sal.reshape(side, side)
