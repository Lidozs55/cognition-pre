import numpy as np

from cognition_pre.config import ModulationConfig
from cognition_pre.display.modulation import DualStreamModulator


def test_full_attention_brightens_focus_damps_ambient():
    mod = DualStreamModulator(ModulationConfig(gamma=0.3))
    img = np.full((10, 10, 3), 200, np.uint8)
    a = np.ones((10, 10), np.float32)
    focus, ambient = mod.apply(img, a)
    assert np.allclose(focus, 200)          # A=1 → mask=1
    assert np.allclose(ambient, 200 * 0.3)  # A=1 → mask=γ


def test_zero_attention_reverses():
    mod = DualStreamModulator(ModulationConfig(gamma=0.3))
    img = np.full((10, 10, 3), 200, np.uint8)
    a = np.zeros((10, 10), np.float32)
    focus, ambient = mod.apply(img, a)
    assert np.allclose(focus, 60)
    assert np.allclose(ambient, 200)
