"""Phase 1 真权重校验：对指定源抓一帧，跑注意力后端并落盘热图叠加图。

用途：接入/更换注意力模型后的人工判读（roadmap Phase 1 清单）——
  - 打印后端关键参数（repo / patch size / 层区间 / prefix tokens）与单帧耗时；
  - 保存并排图：中心裁切帧 | 注意力热图叠加回裁切帧。
判读要点：热图应集中在画面主体（屏幕内容、人脸等），而非均匀一片或全部压在边缘；
层区间不合理时常见两种病态——无结构噪声 / 全图均亮。

用法：
  uv run python scripts/check_backend.py --source 1 --capture-backend dshow
  uv run python scripts/check_backend.py --source data/demo.avi --repo facebook/dinov2-small
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from cognition_pre.config import AppConfig
from cognition_pre.perception import HeatmapPostprocessor, build_backend, upsample
from cognition_pre.preprocess import Preprocessor


def grab_frame(source: str, cap_backend: str, warmup: int = 10) -> np.ndarray:
    """从摄像头索引或视频文件抓一帧；摄像头先丢 warmup 帧让自动曝光稳定。"""
    api = cv2.CAP_DSHOW if cap_backend == "dshow" else cv2.CAP_MSMF if cap_backend == "msmf" else cv2.CAP_ANY
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source, api)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频源: {source!r}")
    frame = None
    for i in range(warmup):
        ok, frame = cap.read()
        if not ok:
            frame = None
    cap.release()
    if frame is None:
        raise SystemExit(f"视频源无帧输出: {source!r}")
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description="注意力后端单帧校验（热图叠加落盘）")
    ap.add_argument("--source", default="1", help="摄像头索引 / 视频文件路径")
    ap.add_argument("--capture-backend", default="dshow", help="auto | dshow | msmf | ffmpeg")
    ap.add_argument("--backend", default="dinov2", help="注意力后端（缺省 dinov2；mock 仅用于对照）")
    ap.add_argument("--repo", default=None, help="HF 模型仓（缺省走 configs/default.yaml）")
    ap.add_argument("--size", type=int, default=None, help="模型输入边长（224 / 518）")
    ap.add_argument("--device", default=None, help="cpu | cuda")
    ap.add_argument("--out", default="outputs/check_backend.png", help="输出 PNG 路径")
    args = ap.parse_args()

    cfg = AppConfig()
    cfg.attention.backend = args.backend
    if args.repo:
        cfg.attention.hf_repo = args.repo
    if args.size:
        cfg.preprocess.model_input_size = args.size
    if args.device:
        cfg.attention.device = args.device

    frame = grab_frame(args.source, args.capture_backend)
    pre = Preprocessor(cfg.preprocess)
    crop, _ = pre.process(frame)
    model_rgb = pre.to_model_rgb(crop)

    backend = build_backend(cfg.attention)
    post = HeatmapPostprocessor(cfg.attention.top_clip_pct)

    backend.infer(model_rgb)  # 预热（首次调用含惰性初始化）
    t0 = time.perf_counter()
    n = 5
    for _ in range(n):
        raw = backend.infer(model_rgb)
    dt = (time.perf_counter() - t0) / n

    attn = upsample(post(raw), crop.shape[:2])
    heat = cv2.applyColorMap((attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(crop, 0.55, heat, 0.45, 0)
    gap = np.full((crop.shape[0], 8, 3), 255, np.uint8)
    canvas = np.concatenate([crop, gap, overlay], axis=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)

    print(f"repo={cfg.attention.hf_repo} size={cfg.preprocess.model_input_size} device={cfg.attention.device}")
    print(f"patch 热图 {raw.shape[0]}x{raw.shape[1]} | min={raw.min():.4f} max={raw.max():.4f}")
    print(f"单帧 {dt * 1000:.0f} ms（{1 / dt:.1f} fps，含 output_attentions）")
    print(f"已写入 {out}（左：裁切帧，右：热图叠加）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
