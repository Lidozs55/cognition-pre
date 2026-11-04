import numpy as np

from cognition_pre.config import ModulationConfig
from cognition_pre.display.modulation import DualStreamModulator


def test_full_attention_focus_bright_ambient_black():
    mod = DualStreamModulator(ModulationConfig())
    img = np.full((10, 10, 3), 200, np.uint8)
    a = np.ones((10, 10), np.float32)
    focus, ambient = mod.apply(img, a)
    assert np.allclose(focus, 200)  # A=1 >= hi → 原始亮度
    assert np.allclose(ambient, 0)  # 1−A=0 <= lo → 全黑（black_floor=0）


def test_zero_attention_reverses():
    mod = DualStreamModulator(ModulationConfig())
    img = np.full((10, 10, 3), 200, np.uint8)
    a = np.zeros((10, 10), np.float32)
    focus, ambient = mod.apply(img, a)
    assert np.allclose(focus, 0)
    assert np.allclose(ambient, 200)


def test_ramp_linear_between_thresholds():
    """lo/hi 之间线性过渡：A=中点 → 半亮。"""
    mod = DualStreamModulator(ModulationConfig(focus_lo=0.25, focus_hi=0.65))
    img = np.full((10, 10, 3), 200, np.uint8)
    a = np.full((10, 10), 0.45, np.float32)  # (0.45−0.25)/(0.65−0.25)=0.5
    focus, _ = mod.apply(img, a)
    assert np.allclose(focus, 100)


def test_ramp_clips_outside_band():
    mod = DualStreamModulator(ModulationConfig(focus_lo=0.25, focus_hi=0.65))
    img = np.full((10, 10, 3), 200, np.uint8)
    below, _ = mod.apply(img, np.full((10, 10), 0.1, np.float32))
    above, _ = mod.apply(img, np.full((10, 10), 0.9, np.float32))
    assert np.allclose(below, 0) and np.allclose(above, 200)


def test_black_floor_keeps_residue():
    mod = DualStreamModulator(ModulationConfig(black_floor=0.1))
    img = np.full((10, 10, 3), 200, np.uint8)
    focus, _ = mod.apply(img, np.zeros((10, 10), np.float32))
    assert np.allclose(focus, 20)  # 200 × 0.1 残亮，不是死黑


def test_degenerate_thresholds_become_step():
    """hi <= lo 退化为阶跃：避免除零，边界处硬切。"""
    mod = DualStreamModulator(ModulationConfig(focus_lo=0.5, focus_hi=0.5))
    img = np.full((10, 10, 3), 200, np.uint8)
    dark, _ = mod.apply(img, np.full((10, 10), 0.49, np.float32))
    bright, _ = mod.apply(img, np.full((10, 10), 0.5, np.float32))
    assert np.allclose(dark, 0) and np.allclose(bright, 200)
