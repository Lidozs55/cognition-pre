"""测试公共配置工厂：全走 mock 后端 + 合成源，无摄像头无权重无窗口。"""

from __future__ import annotations

import pytest

from cognition_pre.config import AppConfig, set_option


def make_cfg(tmp_path=None, overrides: dict | None = None) -> AppConfig:
    cfg = AppConfig()
    cfg.display.show = False
    cfg.capture.source = "test:split"
    cfg.attention.fps = 20.0
    cfg.color.hz = 10.0
    cfg.color.sample_budget = 4000
    cfg.output.log_every = 0.0
    cfg.max_seconds = 2.5
    if tmp_path is not None:
        cfg.output.jsonl_path = str(tmp_path / "out.jsonl")
    for dotted, val in (overrides or {}).items():
        set_option(cfg, dotted, val)
    return cfg


@pytest.fixture
def cfg_factory():
    return make_cfg
