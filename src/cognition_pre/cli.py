"""命令行入口。

常用：
  cognition-pre doctor                                     # 环境自检（依赖/摄像头探测）
  cognition-pre run --mock --no-display --max-seconds 3    # 无权重冒烟
  cognition-pre run --source 0 --dinov3 --device cuda      # OBS 虚拟摄像头 + 真模型
  cognition-pre run --source data/demo.avi --mock          # 文件源联调
"""

from __future__ import annotations

import argparse
import importlib
import logging
import platform
from pathlib import Path

from .config import AppConfig, load_config, set_option
from .pipeline import RealtimePipeline

log = logging.getLogger("cognition-pre")

_DEFAULT_CONFIG_CANDIDATES = ("configs/default.yaml",)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cognition-pre",
        description="DINOv3 注意力驱动的实时双主色提取（焦点色/氛围色）",
        epilog="用法: cognition-pre run [flags...] | cognition-pre doctor",
    )
    p.add_argument("--config", help="YAML 配置路径（缺省依次找 configs/default.yaml、内置默认）")
    p.add_argument("--source", help='摄像头索引 / 视频文件路径 / "test:split" 合成源')
    p.add_argument("--backend", help="采集后端 auto | dshow | msmf | ffmpeg（Windows 打 OBS 虚拟摄像头建议 dshow）")
    p.add_argument("--mock", action="store_true", help="用 mock 注意力后端（无需权重）")
    p.add_argument("--dinov3", action="store_true", help="用 DINOv3 真模型（需 uv sync --extra model + HF_TOKEN）")
    p.add_argument("--size", type=int, choices=(224, 518), help="模型输入边长")
    p.add_argument("--device", help="cpu | cuda")
    p.add_argument("--attention-fps", type=float, help="热图更新频率（默认 10）")
    p.add_argument("--color-hz", type=float, help="主色计算频率（默认 10）")
    p.add_argument("--lambda", dest="lam", type=float, help="注意力权重强度 λ（默认 0.75）")
    p.add_argument("--gamma", type=float, help="调制下限 γ（默认 0.3）")
    p.add_argument("--no-display", action="store_true", help="无窗口运行（自检/服务器）")
    p.add_argument("--record-dir", help="录制焦点/氛围两路调制流到该目录（仅展示路，需有显示模式）")
    p.add_argument("--no-loop", action="store_true", help="文件源播完不循环")
    p.add_argument("--max-seconds", type=float, help="运行 N 秒后自动停止")
    p.add_argument("--jsonl", help="主色结果 JSONL 输出路径")
    p.add_argument("--udp", metavar="HOST:PORT", help="主色结果 UDP 外发")
    p.add_argument("--ws", metavar="URL", help="主色结果 WebSocket 外发，例 ws://127.0.0.1:8765")
    return p


def _load_cfg(path: str | None) -> AppConfig:
    if path:
        return load_config(path)
    for cand in _DEFAULT_CONFIG_CANDIDATES:
        if Path(cand).exists():
            return load_config(cand)
    return load_config(None)


def _apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> None:
    def set_or_die(dotted: str, value):
        try:
            set_option(cfg, dotted, value)
        except KeyError:
            raise SystemExit(f"未知配置项: {dotted}")

    if args.source is not None:
        set_or_die("capture.source", args.source)
    if args.backend is not None:
        set_or_die("capture.backend", args.backend)
    if args.mock:
        set_or_die("attention.backend", "mock")
    if args.dinov3:
        set_or_die("attention.backend", "dinov3")
    if args.size is not None:
        set_or_die("preprocess.model_input_size", args.size)
    if args.device is not None:
        set_or_die("attention.device", args.device)
    if args.attention_fps is not None:
        set_or_die("attention.fps", args.attention_fps)
    if args.color_hz is not None:
        set_or_die("color.hz", args.color_hz)
    if args.lam is not None:
        set_or_die("color.lam", args.lam)
    if args.gamma is not None:
        set_or_die("modulation.gamma", args.gamma)
    if args.no_display:
        set_or_die("display.show", False)
    if args.record_dir is not None:
        set_or_die("display.record_dir", args.record_dir)
    if args.no_loop:
        set_or_die("capture.loop_file", False)
    if args.max_seconds is not None:
        set_or_die("max_seconds", args.max_seconds)
    if args.jsonl is not None:
        set_or_die("output.jsonl_path", args.jsonl)
    if args.udp is not None:
        host, _, port = args.udp.rpartition(":")
        if not host or not port.isdigit():
            raise SystemExit("--udp 需要 HOST:PORT 形式")
        set_or_die("output.udp_host", host)
        set_or_die("output.udp_port", int(port))
    if args.ws is not None:
        set_or_die("output.ws_url", args.ws)


def doctor(cfg: AppConfig) -> int:
    """环境自检：核心/可选依赖版本 + 摄像头探测。"""
    print(f"python {platform.python_version()}  ({sys_executable()})")
    for mod in ("numpy", "cv2", "sklearn", "yaml", "torch", "transformers", "websockets"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:14s} OK   {getattr(m, '__version__', '?')}")
        except Exception as exc:
            print(f"  {mod:14s} --   ({exc.__class__.__name__}: {exc})")
    try:
        import torch

        print(f"  CUDA available: {torch.cuda.is_available()}")
    except Exception:
        pass

    print("摄像头探测（索引 0-3，dshow；个别驱动可能响应慢）:")
    import cv2

    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        got = False
        if ok:
            got, _ = cap.read()
        cap.release()
        print(f"  camera {i}: open={ok} frame={got}")
    print(f"当前源: {cfg.capture.source!r} | 注意力后端: {cfg.attention.backend}")
    print("提示：OBS 虚拟摄像头打开后通常出现在某个索引上；用上表找到它，再 --source <索引>")
    return 0


def sys_executable() -> str:
    import sys

    return sys.executable


def sys_argv() -> list[str]:
    import sys

    return sys.argv


def main(argv: list[str] | None = None) -> int:
    argv = list(sys_argv()[1:] if argv is None else argv)
    # 轻量"子命令"：run（默认）/ doctor；其余走统一 flag 解析
    cmd = "run"
    if argv and argv[0] in ("run", "doctor"):
        cmd = argv[0]
        argv = argv[1:]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    cfg = _load_cfg(args.config)
    _apply_overrides(cfg, args)

    if cmd == "doctor":
        return doctor(cfg)

    pipe = RealtimePipeline(cfg)
    try:
        pipe.run_blocking()
    except KeyboardInterrupt:
        log.info("中断退出")
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2
    finally:
        pipe.stop()

    if pipe.latest_result is not None:
        d = pipe.latest_result.to_dict()
        print(f"最后输出: focus={d['focus_rgb']} ambient={d['ambient_rgb']} (ts={d['ts']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
