import pytest
from conftest import make_cfg

from cognition_pre.config import load_config, set_option


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        """
attention:
  backend: dinov2
  fps: 12
  layer_start_frac: 0.5
color:
  weight_lo: 0.1
  merge_dist: 0.07
  surround_contrast: false
  lab_weights: [0.4, 1.2, 1.2]
unknown_section:
  foo: 1
""",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="未知配置项"):
        cfg = load_config(p)
    assert cfg.attention.backend == "dinov2"
    assert cfg.attention.fps == 12
    assert cfg.attention.layer_start_frac == 0.5
    assert cfg.color.weight_lo == 0.1
    assert cfg.color.merge_dist == 0.07
    assert cfg.color.surround_contrast is False
    assert cfg.color.lab_weights == (0.4, 1.2, 1.2)  # yaml list → tuple
    assert cfg.capture.source == "0"  # 未覆盖的键保持默认


def test_set_option_dotted():
    cfg = make_cfg()
    set_option(cfg, "color.weight_lo", 0.15)
    set_option(cfg, "color.lab_weights", [0.3, 1.0, 1.0])
    assert cfg.color.weight_lo == 0.15
    assert cfg.color.lab_weights == (0.3, 1.0, 1.0)
    with pytest.raises(KeyError):
        set_option(cfg, "color.nope", 1)
