"""热图时域稳定性 A/B：同一视频源连续采帧，比较候选信号阈值化掩膜的帧间 IoU。

动机（2026-11-06 真素材判读反馈）：单帧对照里 NCut 空间连贯性最佳，但 2026-11-05
曾因"比主体多纳入背景/不稳"被否——单帧好看与时域稳定是两回事：
  - 注意力散斑 → 掩膜逐帧抖动（IoU 持续偏低）；
  - NCut 极性翻转/模式切换（特征向量近简并、中心先验定符号翻转）→ IoU 偶发崩。
判读口径：mean 高且无骤降 = 稳定；偶发 <0.5 = 极性/模式切换；持续低 = 信号抖。
掩膜占画面比例（area）同时量化"超出主体"的程度（2026-11-06 反馈：focusmask 全面
超出主体——该值大即超纳入）。

行（--rows 可选）：last-layer(current) / last-headmax / last-k3smooth / ncut
（定义与 compare_heatmaps.py 一致；k3 = 同现役信号 + 定核 3x3 平滑）。

用法：
  uv run python scripts/heatmap_stability.py --source 1 --capture-backend dshow
  uv run python scripts/heatmap_stability.py --source data/xxx.mp4 --capture-backend auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from compare_heatmaps import forward_signals, label, load_model, overlay

from cognition_pre.config import AppConfig
from cognition_pre.perception import HeatmapPostprocessor, upsample
from cognition_pre.preprocess import Preprocessor

# 行定义：(信号名, 后处理模式)，与 compare_heatmaps.py 的诊断行一一对应
ROW_SPECS: dict[str, tuple[str, str]] = {
    "last-layer(current)": ("last-layer(current)", "post"),
    "last-headmax": ("last-headmax", "post"),
    "last-k3smooth": ("last-layer(current)", "k3"),
    "ncut": ("ncut", "post"),
}


def collect_frames(source: str, backend: str, n: int, stride: int, warmup: int = 10) -> list[np.ndarray]:
    """连续采样 n 帧（相邻间隔 stride 帧读一帧）；摄像头先丢 warmup 帧稳定曝光。"""
    api = cv2.CAP_DSHOW if backend == "dshow" else cv2.CAP_MSMF if backend == "msmf" else cv2.CAP_ANY
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source, api)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频源: {source!r}")
    frames: list[np.ndarray] = []
    if source.isdigit():
        for _ in range(warmup):
            cap.read()
    ok, frame = cap.read()
    while len(frames) < n and ok:
        frames.append(frame)
        for _ in range(stride):
            ok, frame = cap.read()
    cap.release()
    if len(frames) < 2:
        raise SystemExit("采帧不足 2 帧（源太短或 stride 过大），无法算 IoU")
    return frames


def world_mask(raw: np.ndarray, crop_hw: tuple[int, int], cfg, post: HeatmapPostprocessor, mode: str) -> np.ndarray:
    """候选信号 → 焦世界二值掩膜（A > color.weight_lo，与 dominant.py 同一比较）。"""
    proc = post
    if mode == "k3":
        proc = HeatmapPostprocessor(cfg.attention.top_clip_pct, 3.0 / raw.shape[0], cfg.attention.norm_floor_pct)
    a01 = upsample(proc(raw), crop_hw)
    return (a01 > cfg.color.weight_lo).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return 1.0 if union == 0.0 else inter / union  # 双空掩膜视为一致


def main() -> int:
    ap = argparse.ArgumentParser(description="热图时域稳定性 A/B（候选信号掩膜帧间 IoU + 接触表）")
    ap.add_argument("--source", default="1", help="摄像头索引 / 视频文件路径")
    ap.add_argument("--capture-backend", default="dshow")
    ap.add_argument("--repo", default="facebook/dinov2-small")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", type=int, default=12, help="采样帧数")
    ap.add_argument("--stride", type=int, default=5, help="相邻采样帧间隔（30fps 源约 0.17s/帧）")
    ap.add_argument("--rows", default="last-layer(current),last-k3smooth,ncut",
                    help=f"逗号分隔，可选 {'/'.join(ROW_SPECS)}")
    ap.add_argument("--out", default="outputs/heatmap_stability.png")
    args = ap.parse_args()

    row_names = [r.strip() for r in args.rows.split(",") if r.strip()]
    unknown = [r for r in row_names if r not in ROW_SPECS]
    if unknown:
        raise SystemExit(f"未知行: {unknown}，可选 {'/'.join(ROW_SPECS)}")

    cfg = AppConfig()
    pre = Preprocessor(cfg.preprocess)
    post = HeatmapPostprocessor(cfg.attention.top_clip_pct, cfg.attention.smooth_frac, cfg.attention.norm_floor_pct)
    model = load_model(args.repo, args.device)

    frames = collect_frames(args.source, args.capture_backend, args.frames, args.stride)
    print(f"repo={args.repo} size={args.size} 采帧 {len(frames)} 张（间隔 {args.stride} 帧）")

    masks: dict[str, list[np.ndarray]] = {name: [] for name in row_names}
    for i, frame in enumerate(frames):
        crop, _ = pre.process(frame)
        signals = forward_signals(model, pre.to_model_rgb(crop, args.size), args.device)
        for name in row_names:
            sig, mode = ROW_SPECS[name]
            masks[name].append(world_mask(signals[sig], crop.shape[:2], cfg, post, mode))

    stats: list[str] = []
    for name in row_names:
        ms = masks[name]
        ious = [iou(a, b) for a, b in zip(ms, ms[1:])]
        area = float(np.mean([m.mean() for m in ms]))
        drops = sum(v < 0.5 for v in ious)
        stats.append(f"{name}: IoU mean={np.mean(ious):.3f} min={min(ious):.3f} "
                     f"低于0.5次数={drops}/{len(ious)} 掩膜占比均值={area:.2f}")
    print("\n".join(stats))

    # 接触表：行=信号，列=帧（掩膜叠加，直读抖动/翻转/超纳入）
    cell = 320
    grid_rows: list[np.ndarray] = []
    for name in row_names:
        cells = []
        for i, frame in enumerate(frames):
            crop, _ = pre.process(frame)
            img = overlay(crop, masks[name][i].astype(np.float32))
            img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
            cells.append(label(img, f"f{i}" if i else name))
        grid_rows.append(np.hstack(cells))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(grid_rows))
    print(f"已写入 {out}（行={row_names}，列=帧序号）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
