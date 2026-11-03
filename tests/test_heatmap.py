import numpy as np

from cognition_pre.perception import HeatmapPostprocessor, upsample
from cognition_pre.perception.mock_backend import MockAttentionBackend


def test_postprocess_clips_outliers_on_normal_map():
    """正常热图：离群极值被截到保留区最高水位，结构信号保留。"""
    a = np.full((14, 14), 0.5, np.float32)
    a[2:8, 2:8] = 1.0  # 大块主体（36/196 ≈ 18%）
    a[0, 0] = 100.0
    a[13, 13] = 100.0  # 极端离群（~1%）
    out = HeatmapPostprocessor(0.05)(a)
    assert 0.0 <= out.min() and out.max() <= 1.0 + 1e-5
    assert out[0, 0] == out.max()  # 离群被截到主体水位
    assert out[7, 7] == out.max()  # 主体峰保留
    assert out[10, 10] < out.max()  # 背景仍是背景


def test_postprocess_sparse_map_keeps_signal():
    """稀疏热图（注意力集中在 <top_pct 的 patch）：截断不得整幅清零（P2 回归）。"""
    a = np.zeros((10, 10), np.float32)
    a[4, 4] = a[4, 5] = 5.0  # 2% 的孤斑是唯一信号
    out = HeatmapPostprocessor(0.05)(a)
    assert out.sum() > 0
    assert out[4, 4] == out.max() and out.max() <= 1.0 + 1e-5
    assert out[0, 0] == 0.0  # 背景保持背景


def test_postprocess_constant_is_zero():
    out = HeatmapPostprocessor(0.05)(np.full((5, 5), 3.0))
    assert np.allclose(out, 0.0)


def test_upsample_shape():
    attn = np.random.default_rng(0).random((14, 14)).astype(np.float32)
    big = upsample(attn, (360, 360))
    assert big.shape == (360, 360)


def test_mock_backend_shape_and_range():
    be = MockAttentionBackend(animated=False, pos=(0.5, 0.5))
    out = be.infer(np.zeros((224, 224, 3), np.uint8))
    assert out.shape == (14, 14)
    assert 0.0 <= out.min() and out.max() <= 1.0
    center = out[7, 7]
    assert center == out.max()  # 高斯斑中心在正中
