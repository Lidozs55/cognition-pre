"""热图质量对照：同一帧并排比较 DINOv2 saliency 信号与输入分辨率。

行（信号/诊断）：
  - last-layer(current) ：末层 CLS→patch 注意力（现役，官方 DINO 配方）
  - last-headmax       ：末层多头取最大（对照现役多头平均）——散斑归因：平均是否把
    "物体头"摊薄进"结构头"（2026-11-06 观感反馈"散、无聚集感"）
  - last-k3smooth      ：同现役信号 + 定核 3x3 平滑（224² 现役核=1 实际不模糊）——
    聚集感对照：轻度并斑能否替代换模型
  - focusmask / focusmask-closed ：焦世界二值掩膜（A > color.weight_lo，主色路真实
    看到的世界）与闭运算（核≈1 patch）填洞预览——直读"主体内低注意力空洞"
    （观感反馈假说 3：洞会把主体内部泄漏给氛围世界）
  - midlate / midlate+last ：层选择备选（中后层平均，层集合固定 [int(n*0.6), n-1)，不随配置变）
  - ncut ：TokenCut-lite（CVPR 2022 Wang et al.）——patch 特征余弦亲和 → 归一化割
    第二特征向量。真素材判读未通过（稳定比主体多纳入背景，末层@518 胜出），不集成
    backend；行保留供将来 full TokenCut（含 CRF 精修）复查
已判不可行（真素材人眼判读，不再保留实现）：PCA 前景图、feat-norm 特征范数。
列：--sizes 分辨率；热图行过同一后处理（平滑 + top-clip + floor 归一）与同一叠加方式。

用法：
  uv run python scripts/compare_heatmaps.py --source 1 --capture-backend dshow
  uv run python scripts/compare_heatmaps.py --source data/demo.avi --sizes 224,518
  uv run python scripts/compare_heatmaps.py --source "data/xxx.png"   # 静态图（真实场景判读）
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from check_backend import grab_frame

from cognition_pre.config import AppConfig
from cognition_pre.perception import HeatmapPostprocessor, upsample
from cognition_pre.perception.ncut import ncut_map
from cognition_pre.preprocess import Preprocessor

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(repo: str, device: str):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(repo, attn_implementation="eager")  # 5.x SDPA 不返回注意力
    model.eval().to(device)
    return model


def ncut_saliency(patch_feat, side: int) -> np.ndarray:
    """薄封装：与主色路后端共用 perception/ncut.py 的单一实现（含极性策略）。"""
    return ncut_map(patch_feat, side)


@torch.no_grad()
def forward_signals(model, rgb_u8: np.ndarray, device: str) -> dict[str, np.ndarray]:
    """一次前向，返回全部候选信号的 patch 网格热图（未上采样）。"""
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1).to(device)
    x = torch.from_numpy(np.ascontiguousarray(rgb_u8)).permute(2, 0, 1)[None].float().div_(255.0)
    x = (x - mean) / std

    cfg_m = model.config
    # 兼容两种字段：普通版 num_prefix_tokens（缺省 1=仅 CLS）；registers 版只有
    # num_register_tokens（CLS+registers 都在 patch 列之前）
    prefix = int(getattr(cfg_m, "num_prefix_tokens", 1 + getattr(cfg_m, "num_register_tokens", 0)))
    out = model(pixel_values=x, output_attentions=True)
    attns = out.attentions
    if not attns:
        raise RuntimeError("模型未返回 attentions（检查 eager 与 transformers 版本）")
    n_layers = len(attns)
    midlate = list(range(int(n_layers * 0.6), n_layers - 1))

    def cls_attn(idxs: list[int]) -> np.ndarray:
        acc = None
        for i in idxs:
            a = attns[i][0, :, 0, prefix:].mean(dim=0)
            acc = a if acc is None else acc + a
        return (acc / len(idxs)).cpu().numpy()

    patch_feat = out.last_hidden_state[0, prefix:]
    side = round(patch_feat.shape[0] ** 0.5)
    if side * side != patch_feat.shape[0]:
        raise RuntimeError(f"patch 数非平方数: {patch_feat.shape[0]}")

    last = cls_attn([n_layers - 1]).astype(np.float32)
    signals: dict[str, np.ndarray] = {
        "midlate": cls_attn(midlate).astype(np.float32).reshape(side, side),
        "midlate+last": cls_attn(midlate + [n_layers - 1]).astype(np.float32).reshape(side, side),
        "last-layer(current)": last.reshape(side, side),
        # 多头聚合对照：max 强调"任一头强烈注视"的 patch，mean 会被平坦结构头稀释
        "last-headmax": attns[n_layers - 1][0, :, 0, prefix:]
        .max(dim=0)
        .values.cpu()
        .numpy()
        .astype(np.float32)
        .reshape(side, side),
        # 极性策略（度数/紧凑度级联）内聚在 perception/ncut.py 单一实现里
        "ncut": ncut_saliency(patch_feat, side),
    }
    return signals


def overlay(crop: np.ndarray, attn01: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap((np.clip(attn01, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(crop, 0.55, heat, 0.45, 0)


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def load_source_frame(source: str, cap_backend: str) -> np.ndarray:
    """静态图直接读（真实场景判读素材）；视频/摄像头走 grab_frame。
    Windows 中文路径 imread 读不了，走 np.fromfile + imdecode。"""
    p = Path(source)
    if p.suffix.lower() in _IMAGE_EXTS and p.is_file():
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"图片读取失败: {source!r}")
        return img
    return grab_frame(source, cap_backend)


def render_cell(mode: str, raw: np.ndarray, crop: np.ndarray, cfg, post: HeatmapPostprocessor) -> np.ndarray:
    """按行模式出图。post=现役后处理热图；k3=定核 3x3 平滑对照（frac=3/side 精确到 3）；
    mask / mask-closed=焦世界二值掩膜（A>weight_lo，与 dominant.py 同一比较），
    closed 版用闭运算（核≈1 patch）填洞——主体空洞直读与填洞代价预览。"""
    if mode == "k3":
        proc = HeatmapPostprocessor(cfg.attention.top_clip_pct, 3.0 / raw.shape[0], cfg.attention.norm_floor_pct)
        return overlay(crop, upsample(proc(raw), crop.shape[:2]))
    a01 = upsample(post(raw), crop.shape[:2])
    if mode in ("mask", "mask-closed"):
        m = (a01 > cfg.color.weight_lo).astype(np.uint8)
        if mode == "mask-closed":
            p = max(1, round(crop.shape[0] / raw.shape[0]))  # 一个 patch 在裁切帧里的边长
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p + 1, 2 * p + 1))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return overlay(crop, m.astype(np.float32))
    return overlay(crop, a01)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="DINOv2 热图信号/分辨率对照（同帧网格落盘）")
    ap.add_argument("--source", default="1", help="摄像头索引 / 视频文件路径")
    ap.add_argument("--capture-backend", default="dshow")
    ap.add_argument("--repo", default="facebook/dinov2-small")
    ap.add_argument("--sizes", default="224,518", help="逗号分隔的模型输入边长")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/compare_heatmaps.png")
    args = ap.parse_args()

    cfg = AppConfig()
    frame = load_source_frame(args.source, args.capture_backend)
    pre = Preprocessor(cfg.preprocess)
    crop, _ = pre.process(frame)

    model = load_model(args.repo, args.device)
    cfg_model = model.config
    print(
        f"repo={args.repo} device={args.device} patch={getattr(cfg_model, 'patch_size', '?')} "
        f"layers={cfg_model.num_hidden_layers} device={args.device}"
    )

    post = HeatmapPostprocessor(cfg.attention.top_clip_pct, cfg.attention.smooth_frac, cfg.attention.norm_floor_pct)
    sizes = [int(s) for s in args.sizes.split(",")]
    # 行定义：(行名, 信号, 出图模式)——诊断组针对 2026-11-06 观感反馈三假说
    rows_def: list[tuple[str, str, str]] = [
        ("last-layer(current)", "last-layer(current)", "post"),
        ("last-headmax", "last-headmax", "post"),
        ("last-k3smooth", "last-layer(current)", "k3"),
        ("focusmask", "last-layer(current)", "mask"),
        ("focusmask-closed", "last-layer(current)", "mask-closed"),
        ("midlate", "midlate", "post"),
        ("midlate+last", "midlate+last", "post"),
        ("ncut", "ncut", "post"),
    ]
    row_names = [r[0] for r in rows_def]

    grid: dict[str, dict[int, np.ndarray]] = {name: {} for name in row_names}
    for size in sizes:
        rgb = pre.to_model_rgb(crop, size)
        t0 = time.perf_counter()
        signals = forward_signals(model, rgb, args.device)
        dt = (time.perf_counter() - t0) * 1000
        print(f"size={size}: 单帧 {dt:.0f} ms（含 output_attentions + ncut）")
        for label_name, sig, mode in rows_def:
            grid[label_name][size] = render_cell(mode, signals[sig], crop, cfg, post)

    # 网格：每行 = 原图 | 各分辨率叠加（自带标签，无表头行，列数天然对齐）
    cell = crop.shape[0]
    gap = np.full((cell, 6, 3), 255, np.uint8)
    rows: list[np.ndarray] = []
    for name in row_names:
        row = [label(crop, "crop")]
        for size in sizes:
            row.append(gap)
            row.append(label(grid[name][size], f"{name} @{size}"))
        rows.append(np.hstack(row))
        rows.append(np.full((4, rows[-1].shape[1], 3), 255, np.uint8))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"已写入 {out}（行={row_names}，列={sizes}，首列原图）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
