import pytest
from conftest import make_cfg

from cognition_pre.config import load_config, set_option


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        """
attention:
  backend: dinov3
  fps: 12
  layer_start_frac: 0.5
color:
  lam: 1.0
  lab_weights: [0.4, 1.2, 1.2]
unknown_section:
  foo: 1
""",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="未知配置项"):
        cfg = load_config(p)
    assert cfg.attention.backend == "dinov3"
    assert cfg.attention.fps == 12
    assert cfg.attention.layer_start_frac == 0.5
    assert cfg.color.lam == 1.0
    assert cfg.color.lab_weights == (0.4, 1.2, 1.2)  # yaml list → tuple
    assert cfg.capture.source == "0"  # 未覆盖的键保持默认


def test_set_option_dotted():
    cfg = make_cfg()
    set_option(cfg, "color.lam", 1.5)
    set_option(cfg, "color.lab_weights", [0.3, 1.0, 1.0])
    assert cfg.color.lam == 1.5
    assert cfg.color.lab_weights == (0.3, 1.0, 1.0)
    with pytest.raises(KeyError):
        set_option(cfg, "color.nope", 1)
