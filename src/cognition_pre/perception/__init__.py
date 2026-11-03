"""M3 感知层：注意力后端抽象 + 热图后处理 + DINOv3/mock 实现。"""

from __future__ import annotations

import logging
import os

from .base import AttentionBackend, HeatmapPostprocessor, upsample
from .mock_backend import MockAttentionBackend

__all__ = [
    "AttentionBackend",
    "HeatmapPostprocessor",
    "MockAttentionBackend",
    "build_backend",
    "upsample",
]


def build_backend(cfg) -> AttentionBackend:
    """按配置构建注意力后端。默认 mock（零依赖跑通全链路）。"""
    if cfg.backend == "mock":
        return MockAttentionBackend()
    if cfg.backend == "dinov3":
        from .dinov3_backend import DINOv3AttentionBackend

        token = os.environ.get(cfg.hf_token_env)
        if not token:
            logging.getLogger(__name__).warning(
                "未设置环境变量 %s：DINOv3 权重为 gated 仓库，拉取可能 401。"
                "先在 HF 页面接受许可并配置 token（见 README）。",
                cfg.hf_token_env,
            )
        return DINOv3AttentionBackend(
            hf_repo=cfg.hf_repo,
            device=cfg.device,
            token=token,
            layer_start_frac=cfg.layer_start_frac,
            layer_end_drop=cfg.layer_end_drop,
        )
    raise ValueError(f"未知注意力后端: {cfg.backend}")
