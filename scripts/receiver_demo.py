"""外发联调接收端模拟器：把 cognition-pre 的主色输出画成终端"灯带"。

与 output/sinks.py 对接（协议 = FrameResult.to_dict() 的 JSON）：
  uv run cognition-pre run --mock --udp 127.0.0.1:9999          # 管线侧
  uv run python scripts/receiver_demo.py --udp 127.0.0.1:9999   # 本脚本（另一终端）

WebSocket 模式下本脚本是服务端（管线里的 WebSocketSink 是客户端）：
  uv run python scripts/receiver_demo.py --ws 127.0.0.1:8765    # 需 uv sync --extra ws
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time


class RateMeter:
    """1 秒滑动窗口的消息速率。"""

    def __init__(self) -> None:
        self.count = 0
        self.hz = 0.0
        self._t_win = time.monotonic()
        self._n_win = 0

    def tick(self) -> None:
        self.count += 1
        self._n_win += 1
        now = time.monotonic()
        if now - self._t_win >= 1.0:
            self.hz = self._n_win / (now - self._t_win)
            self._t_win, self._n_win = now, 0


def enable_ansi() -> None:
    if sys.platform == "win32":
        os.system("")  # 激活 Windows 控制台的 ANSI 转义支持


def render(d: dict, meter: RateMeter) -> str:
    def block(rgb) -> str:
        r, g, b = rgb
        return f"\033[48;2;{r};{g};{b}m   \033[0m rgb({r},{g},{b})"

    return (
        f"\rFOCUS {block(d['focus_rgb'])} conf {d['focus_conf']:.2f}   "
        f"AMBIENT {block(d['ambient_rgb'])} conf {d['ambient_conf']:.2f}   "
        f"{meter.hz:4.1f} Hz #{meter.count}"
    )


def print_line(line: str, prev_len: int) -> int:
    sys.stdout.write(line + " " * max(prev_len - len(line), 0))
    sys.stdout.flush()
    return len(line)


def run_udp(host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.5)
    print(f"UDP 监听 {host}:{port}，等待 cognition-pre ...（Ctrl+C 退出）")
    meter, line_len = RateMeter(), 0
    while True:
        try:
            data, _ = sock.recvfrom(65536)
        except TimeoutError:
            continue
        try:
            d = json.loads(data.decode("utf-8"))
            meter.tick()
            line = render(d, meter)
        except (UnicodeDecodeError, ValueError, KeyError):
            continue  # 非 JSON 或字段不全的消息跳过
        line_len = print_line(line, line_len)


def run_ws(host: str, port: int) -> None:
    try:
        from websockets.sync.server import serve
    except ImportError:
        raise SystemExit("未安装 websockets：先 uv sync --extra ws")

    def handler(ws) -> None:
        meter, line_len = RateMeter(), 0
        print(f"\nWebSocket 客户端已接入: {ws.remote_address}")
        for msg in ws:
            try:
                d = json.loads(msg)
                meter.tick()
                line = render(d, meter)
            except (ValueError, TypeError, KeyError):
                continue  # 非 JSON 或字段不全的消息跳过
            line_len = print_line(line, line_len)

    print(f"WebSocket 服务端 ws://{host}:{port}，等待 cognition-pre 接入 ...（Ctrl+C 退出）")
    with serve(handler, host, port) as server:
        server.serve_forever()


def parse_addr(spec: str) -> tuple[str, int]:
    host, _, port = spec.rpartition(":")
    if not host or not port.isdigit():
        raise SystemExit(f"地址需 HOST:PORT 形式: {spec!r}")
    return host, int(port)


def main() -> int:
    enable_ansi()
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--udp", metavar="HOST:PORT", help="以 UDP 接收端运行（绑定本机地址）")
    g.add_argument("--ws", metavar="HOST:PORT", help="以 WebSocket 服务端运行（管线为客户端）")
    args = ap.parse_args()

    try:
        if args.udp:
            run_udp(*parse_addr(args.udp))
        else:
            run_ws(*parse_addr(args.ws))
    except KeyboardInterrupt:
        print("\n退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
