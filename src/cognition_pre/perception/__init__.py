"""M3 感知层：注意力后端抽象 + 热图后处理 + DINOv2/mock 实现。"""

from __future__ import annotations

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
    if cfg.backend == "dinov2":
        from .dinov2_backend import DINOv2AttentionBackend

        return DINOv2AttentionBackend(
            hf_repo=cfg.hf_repo,
            device=cfg.device,
            layer_start_frac=cfg.layer_start_frac,
            layer_end_drop=cfg.layer_end_drop,
        )
    raise ValueError(f"未知注意力后端: {cfg.backend}")
