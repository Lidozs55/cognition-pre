"""输出 sinks：控制台 / JSONL / UDP / WebSocket，任选组合（M6 外发口子）。

UDP/WebSocket 的下游是灯带、特效等消费方；协议就是 FrameResult.to_dict() 的 JSON。
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
from pathlib import Path

from ..config import OutputConfig
from .results import FrameResult

log = logging.getLogger(__name__)


class BaseSink:
    def emit(self, result: FrameResult) -> None: ...

    def close(self) -> None: ...


class LoggerSink(BaseSink):
    """控制台周期打印（默认 1 秒一条，避免刷屏）。"""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = float(interval)
        self._last = 0.0

    def emit(self, result: FrameResult) -> None:
        if self.interval <= 0:
            return
        now = time.monotonic()
        if now - self._last >= self.interval:
            self._last = now
            d = result.to_dict()
            log.info(
                "[out] focus=%s conf=%.2f | ambient=%s conf=%.2f",
                d["focus_rgb"],
                d["focus_conf"],
                d["ambient_rgb"],
                d["ambient_conf"],
            )


class JsonlSink(BaseSink):
    """每行一条 JSON，供离线分析（Phase 0 全景专项验证的数据出口）。"""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._fh = None
        self._failed = False  # 打开失败只告警一次，不按 10Hz 刷屏

    def emit(self, result: FrameResult) -> None:
        if self._failed:
            return
        if self._fh is None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            except OSError:
                self._failed = True
                log.warning("JSONL 输出打开失败，本轮降级关闭: %s", self.path)
                return
        self._fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class UdpSink(BaseSink):
    def __init__(self, host: str, port: int) -> None:
        self.addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def emit(self, result: FrameResult) -> None:
        try:
            self._sock.sendto(json.dumps(result.to_dict()).encode("utf-8"), self.addr)
        except OSError as exc:  # 单次丢包不致命
            log.debug("UDP 发送失败: %s", exc)

    def close(self) -> None:
        self._sock.close()


class WebSocketSink(BaseSink):
    """websockets.sync 客户端独立线程；断线自动重连（1s 退避），积压丢弃旧包。"""

    def __init__(self, url: str) -> None:
        from websockets.sync.client import connect  # noqa: F401  仅探测依赖是否可用

        self.url = url
        self._q: queue.Queue[str] = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ws-sink", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from websockets.sync.client import connect

        while not self._stop.is_set():
            try:
                with connect(self.url, open_timeout=3) as ws:
                    log.info("WebSocket 已连接: %s", self.url)
                    self._drain_queue()  # 断线期间积压的是过期数据，灯带下游要实时
                    while not self._stop.is_set():
                        try:
                            payload = self._q.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        ws.send(payload)
            except Exception as exc:
                log.warning("WebSocket 断开（%s），1s 后重连", exc)
                if self._stop.wait(1.0):
                    break

    def _drain_queue(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def emit(self, result: FrameResult) -> None:
        payload = json.dumps(result.to_dict())
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()  # 丢最旧，保实时
                self._q.put_nowait(payload)
            except (queue.Empty, queue.Full):
                pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.5)


class SinkGroup(BaseSink):
    def __init__(self, sinks: list[BaseSink]) -> None:
        self.sinks = sinks

    def emit(self, result: FrameResult) -> None:
        for s in self.sinks:
            try:
                s.emit(result)
            except Exception:
                log.exception("sink %s 发送失败", type(s).__name__)

    def close(self) -> None:
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                log.exception("sink %s 关闭失败", type(s).__name__)


def create_sinks(cfg: OutputConfig) -> SinkGroup:
    sinks: list[BaseSink] = [LoggerSink(cfg.log_every)]
    if cfg.jsonl_path:
        sinks.append(JsonlSink(cfg.jsonl_path))
    if cfg.udp_host and cfg.udp_port > 0:
        sinks.append(UdpSink(cfg.udp_host, cfg.udp_port))
    if cfg.ws_url:
        try:
            sinks.append(WebSocketSink(cfg.ws_url))
        except ImportError:
            log.warning("未安装 websockets，跳过 WS 输出（uv sync --extra ws）")
    return SinkGroup(sinks)
