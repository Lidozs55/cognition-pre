"""DINOv2 注意力后端：中后层 CLS→patch 自注意力多头平均（DINO 家族招牌特性：
注意力 ≈ 对象级分割，零训练直用）。

实现说明与 Phase 1 校验清单（见 docs/roadmap.md）：
- 权重为公开仓库（facebook/dinov2-small 等），无需 HF token。
- prefix tokens：transformers 的 Dinov2Model 只有 CLS、无 register token，CLS 行
  切片偏移取 config.num_prefix_tokens（该 config 无此字段，getattr 缺省按 1 处理，
  恰为正确值）；首次真跑打印并人工核对。
- patch 14：224/14=16、518/14=37 均整除，patch 数必为平方数；infer 内保留校验兜底。
- 层选择：取 [num_layers*layer_start_frac, num_layers-layer_end_drop) 的多头平均；
  深层易糊，末层默认丢弃。备选对照：跨层 rollout（Phase 1 实现）。
- 耗时：output_attentions=True 会物化全部层注意力。224² CPU 可实时；
  518² CPU 明显变慢，仅作离线对照。将来可改 hook 精取所选层（见 roadmap Phase 4）。
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from .base import AttentionBackend

log = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2AttentionBackend(AttentionBackend):
    name = "dinov2"

    def __init__(
        self,
        hf_repo: str,
        device: str = "cpu",
        layer_start_frac: float = 0.6,
        layer_end_drop: int = 1,
    ) -> None:
        from transformers import AutoModel  # 延迟导入：core 安装不含 transformers

        self.device = device
        # 注意力热图是核心输出：必须用 eager 实现物化注意力权重
        # （transformers 5.x 默认 SDPA，output_attentions=True 也不会返回 attentions）
        self.model = AutoModel.from_pretrained(hf_repo, attn_implementation="eager")
        self.model.eval().to(device)
        cfg = self.model.config
        self.prefix = int(getattr(cfg, "num_prefix_tokens", 1))
        n_layers = int(cfg.num_hidden_layers)
        start = int(n_layers * layer_start_frac)
        end = max(start + 1, n_layers - max(0, layer_end_drop))
        self.layers = list(range(start, end))
        self._mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1).to(device)
        log.info(
            "DINOv2 就绪: repo=%s patch_size=%s layers=%s prefix_tokens=%s device=%s",
            hf_repo,
            getattr(cfg, "patch_size", "?"),
            self.layers,
            self.prefix,
            device,
        )

    @torch.no_grad()
    def infer(self, rgb_u8: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(rgb_u8)).permute(2, 0, 1)[None].float().div_(255.0)
        x = (x - self._mean) / self._std
        out = self.model(pixel_values=x, output_attentions=True)
        attns = getattr(out, "attentions", None)
        if not attns:
            raise RuntimeError("模型未返回 attentions：检查 transformers 版本（需支持 DINOv2）")
        acc = None
        for i in self.layers:
            a = attns[i][0, :, 0, self.prefix :].mean(dim=0)  # CLS 行 → patch 列，多头平均
            acc = a if acc is None else acc + a
        acc = (acc / len(self.layers)).float().cpu().numpy()
        side = round(acc.shape[0] ** 0.5)
        if side * side != acc.shape[0]:
            raise RuntimeError(f"注意力 patch 数非平方数: {acc.shape[0]}")
        return acc.reshape(side, side).astype(np.float32)
