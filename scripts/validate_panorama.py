"""Phase 0 全景专项验证：中心方形裁切 vs 全画幅，对氛围色稳定性的影响。

背景：16:9 素材被中心裁切后两侧背景丢失，氛围色素材变少（docs/design.md M2 已知代价）。
本脚本隔离"裁切几何"这一个变量——不跑注意力（权重取均匀，等于纯主色统计），
同一帧分别按两种几何逐帧算氛围主色并做时域 EMA，输出对比指标：

  - jump  ：相邻帧稳定输出的 RGB 距离（越小越稳）
  - drift ：稳定输出各通道的时间标准差（越小越稳）
  - diver ：同一帧两种几何稳定输出的 RGB 距离（裁切带来的氛围色偏差）

判读：diver 明显偏大或裁切侧 jump 明显更差 → 按 roadmap 启动"全画幅可变分辨率输入"备选方案。

用法：
  uv run python scripts/validate_panorama.py --source data/wide.avi [--seconds 10] [--jsonl out.jsonl]
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import cv2
import numpy as np

from cognition_pre.color import WeightedDominantColor
from cognition_pre.config import AppConfig, PreprocessConfig
from cognition_pre.preprocess import Preprocessor
from cognition_pre.stability import ColorStabilizer


def make_machine(name: str, square_crop: bool, cfg: AppConfig):
    """一种几何的完整氛围色机：预处理 + 聚类 + 时域稳定。"""
    pre = Preprocessor(PreprocessConfig(square_crop=square_crop))
    cm = WeightedDominantColor(cfg.color, invert=True, name=name)
    stab = ColorStabilizer(cfg.stability.ema_alpha, cfg.stability.conf_ema_alpha)
    return pre, cm, stab


def ambient_color(machine, frame_bgr: np.ndarray) -> tuple[float, float, float] | None:
    """均匀权重（不跑注意力）下的氛围主色，经 EMA 后输出。"""
    pre, cm, stab = machine
    crop, _ = pre.process(frame_bgr)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = cm.compute(rgb, np.zeros(rgb.shape[:2], np.float32))
    if res.best is None:
        return None
    return stab.update(res.best.rgb, res.dominance)[0]


def stats(seq: list[tuple[float, float, float]]) -> dict[str, float]:
    a = np.asarray(seq)
    jumps = np.linalg.norm(np.diff(a, axis=0), axis=1) if len(a) > 1 else np.zeros(1)
    return {
        "jump_mean": float(jumps.mean()),
        "jump_p95": float(np.quantile(jumps, 0.95)),
        "drift": float(a.std(axis=0).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="宽幅视频文件路径")
    ap.add_argument("--seconds", type=float, default=0.0, help="只取前 N 秒（0=全部）")
    ap.add_argument("--jsonl", help="可选：逐帧稳定输出 JSONL（mode 字段区分两种几何）")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频源: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    limit = int(args.seconds * fps) if args.seconds > 0 else None

    cfg = AppConfig()
    crop_m = make_machine("crop", square_crop=True, cfg=cfg)
    full_m = make_machine("full", square_crop=False, cfg=cfg)

    # 每帧两种几何各一条稳定输出；None 表示该几何此帧聚类无结果（统计时对齐跳过）
    per_frame: list[dict[str, tuple[float, float, float] | None]] = []
    with contextlib.ExitStack() as stack:
        jsonl_fh = stack.enter_context(Path(args.jsonl).open("w", encoding="utf-8")) if args.jsonl else None
        stack.callback(cap.release)
        i = 0
        while limit is None or i < limit:
            ok, frame = cap.read()
            if not ok:
                break
            entry = {
                "crop": ambient_color(crop_m, frame),
                "full": ambient_color(full_m, frame),
            }
            per_frame.append(entry)
            if jsonl_fh is not None:
                for mode, rgb in entry.items():
                    if rgb is not None:
                        jsonl_fh.write(
                            json.dumps({"mode": mode, "frame": i, "rgb": [round(v) for v in rgb]}) + "\n"
                        )
            i += 1

    if len(per_frame) < 2:
        raise SystemExit(f"有效帧不足（读到 {len(per_frame)} 帧）")
    print(f"读入 {len(per_frame)} 帧 @ {fps:.1f}fps")

    both = [e for e in per_frame if e["crop"] is not None and e["full"] is not None]
    diver = np.linalg.norm(
        np.asarray([e["crop"] for e in both]) - np.asarray([e["full"] for e in both]), axis=1
    )

    print(f"\n{'指标':<10} {'中心裁切':>12} {'全画幅':>12}")
    sc = stats([e["crop"] for e in per_frame if e["crop"] is not None])
    sf = stats([e["full"] for e in per_frame if e["full"] is not None])
    for key, label in (("jump_mean", "jump 均值"), ("jump_p95", "jump p95"), ("drift", "drift")):
        print(f"{label:<10} {sc[key]:>12.2f} {sf[key]:>12.2f}")
    print(f"\n两几何氛围色偏差 diver: 均值 {diver.mean():.2f} / p95 {np.quantile(diver, 0.95):.2f}（RGB 0-441 尺度）")
    print("判读参考：diver 均值 > 30 或裁切侧 jump 明显更差 → 建议 Phase 0 立项全画幅备选方案")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
