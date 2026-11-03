"""生成本地演示视频：左暖右冷背景 + 移动高饱和块，用于无 OBS 联调。

用法：python scripts/make_demo_video.py --out data/demo.avi --seconds 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/demo.avi")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    w, h = args.width, args.height
    n = int(args.seconds * args.fps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # 背景一次生成：左暖 (RGB 190,120,60) 右冷 (RGB 70,120,190)，中间水平过渡（BGR）
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    warm = np.array([60, 120, 190], np.float32)  # BGR
    cool = np.array([190, 120, 70], np.float32)  # BGR
    row = (1 - xs)[:, None] * warm[None, :] + xs[:, None] * cool[None, :]  # (w, 3)
    bg = np.repeat(row.astype(np.uint8)[None, :, :], h, axis=0)  # (h, w, 3) contiguous

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # MJPG 编码跨平台最稳（XVID 在部分 OpenCV 构建上写不出）
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"无法创建视频: {args.out}")

    for i in range(n):
        frame = bg.copy()
        # 移动的黄色高饱和块（BGR: 40,200,230）
        cx = int(w / 2 + 0.35 * w * np.sin(2 * np.pi * i / n * 2))
        cy = int(h / 2 + 0.25 * h * np.cos(2 * np.pi * i / n))
        s = 140
        x0, y0 = max(0, cx - s // 2), max(0, cy - s // 2)
        frame[y0 : y0 + s, x0 : x0 + s] = (40, 200, 230)
        writer.write(frame)
    writer.release()

    # 写后校验：确实能读回帧才算成功
    check = cv2.VideoCapture(args.out)
    got, _ = check.read()
    check.release()
    if not got:
        raise SystemExit(f"写后校验失败：{args.out} 读不出帧（检查 OpenCV 编码器），可改用 ffmpeg 生成素材")
    print(f"已生成 {args.out}: {n} 帧 @ {args.fps}fps ({w}x{h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
