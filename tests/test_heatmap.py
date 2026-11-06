import numpy as np

from cognition_pre.perception import HeatmapPostprocessor, upsample
from cognition_pre.perception.mock_backend import MockAttentionBackend


def test_postprocess_clips_outliers_on_normal_map():
    """正常热图：离群极值被截到保留区最高水位，结构信号保留（隔离平滑/平台项）。"""
    a = np.full((14, 14), 0.5, np.float32)
    a[2:8, 2:8] = 1.0  # 大块主体（36/196 ≈ 18%）
    a[0, 0] = 100.0
    a[13, 13] = 100.0  # 极端离群（~1%）
    out = HeatmapPostprocessor(0.05, smooth_frac=0.0, norm_floor_pct=0.0)(a)
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


def test_postprocess_smooth_consolidates_speckle():
    """轻度平滑（核尺寸语义）：峰的近邻分到质量（blob 化），3×3 核够不成抹平尖角。"""
    a = np.zeros((16, 16), np.float32)
    a[4, 4] = a[4, 11] = 10.0  # 两个孤峰（相距 7 patch，3×3 核不会把峰间填实）
    smooth = HeatmapPostprocessor(0.0, smooth_frac=0.12)(a)  # 16 网格 → 3×3 核
    raw = HeatmapPostprocessor(0.0, smooth_frac=0.0, norm_floor_pct=0.0)(a)  # 全关对照
    assert smooth[4, 5] > 0 and smooth[4, 10] > 0  # 峰的近邻分到质量（blob 化）
    assert smooth[4, 4] == smooth.max() and smooth.max() <= 1.0 + 1e-5
    assert smooth[4, 7] == 0.0  # 峰间仍是 0：核小，不抹平结构（旧 sigma 语义的教训）
    assert smooth[0, 0] == 0.0  # 远端背景不受影响
    assert raw[4, 5] == 0.0  # 对照：不平滑时近邻确为 0


def test_postprocess_floor_suppresses_background_leak():
    """背景平台抑制：floor 分位以下的值域压到 0——背景高注意力泄漏的对策。

    泄漏带高于真零背景：min-max 归一保不住（泄漏原样输出 0.4），floor 归零能治。
    """
    a = np.zeros((10, 10), np.float32)
    a[2:8, :] = 0.4  # 60% 面积的"背景泄漏"（高于真零，min-max 下原样保留）
    a[8:, :] = 1.0  # 底部 20% 主体；前两行是真零背景
    out = HeatmapPostprocessor(0.0, smooth_frac=0.0, norm_floor_pct=0.5)(a)
    assert out[2:8].max() == 0.0  # 泄漏带整片清零（60%>floor 分位，median 落在带上）
    assert out[9, 9] == 1.0  # 主体保持满值
    assert out[0, 0] == 0.0  # 真零背景不变
    legacy = HeatmapPostprocessor(0.0, smooth_frac=0.0, norm_floor_pct=0.0)(a)
    assert abs(float(legacy[4, 0]) - 0.4) < 1e-5  # floor=0 退回 min-max：泄漏原样保留（老行为对照）


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
