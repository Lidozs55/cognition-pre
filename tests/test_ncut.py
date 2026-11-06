"""NCut 极性定向单测：align_polarity 是纯 numpy（ncut.py 的 torch 惰性导入，
mock 环境无 torch 也可测）。极性三版演进的回归锁：度数锚自信→度数；否则紧凑度。"""

import numpy as np

from cognition_pre.perception.ncut import align_polarity


def test_degree_anchor_confident_positive_flips():
    """度数强正相关：正侧=高连通同质簇=背景 → 翻（dog 案回归：度数 0.78 救回）。"""
    sal = np.linspace(-1.0, 1.0, 16).astype(np.float32)
    degree = np.linspace(-1.0, 1.0, 16).astype(np.float32)  # r=+1
    out = align_polarity(sal, degree, 4)
    np.testing.assert_allclose(out, -sal)


def test_degree_anchor_confident_negative_keeps():
    sal = np.linspace(-1.0, 1.0, 16).astype(np.float32)
    degree = np.linspace(1.0, -1.0, 16).astype(np.float32)  # r=-1
    out = align_polarity(sal, degree, 4)
    np.testing.assert_allclose(out, sal)


def test_weak_degree_falls_back_to_compactness():
    """度数不自信（常数度数）→ 紧凑度：正侧=散开环、负侧=中心块 → 正侧是背景 → 翻
    （截图 1259 案回归：涂鸦背景不同质、度数弱，紧凑度接手）。"""
    g = np.zeros((4, 4), np.float32)
    g[1:3, 1:3] = -1.0  # 负侧 = 中心 2x2（紧凑）
    g[g == 0.0] = 1.0  # 正侧 = 外环（散）
    sal = g.ravel()
    out = align_polarity(sal, np.full(16, 3.0, np.float32), 4)  # 度数方差 0 → r=0
    np.testing.assert_allclose(out, -sal)


def test_weak_degree_compact_positive_keeps():
    g = np.zeros((4, 4), np.float32)
    g[1:3, 1:3] = 1.0  # 正侧 = 中心块（紧凑）
    g[g == 0.0] = -1.0  # 负侧 = 外环（散）
    sal = g.ravel()
    out = align_polarity(sal, np.full(16, 3.0, np.float32), 4)
    np.testing.assert_allclose(out, sal)
