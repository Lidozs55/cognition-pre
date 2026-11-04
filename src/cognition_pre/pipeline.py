"""实时管线编排。

线程模型（多频率解耦，全部走"最新者赢"单槽）：
  capture    ：源读帧 → 中心方形裁切 → crop_slot
  perception ：按 attention.fps 取最新裁切帧 → 后端热图 → 后处理 → 上采样 → attn_slot
  color      ：按 color.hz 取最新帧+热图 → 焦点/氛围双实例加权聚类 → EMA → sinks
  main       ：显示路（cv2 窗口必须主线程）；无显示时仅等待停止信号

展示路（调制视频）只在 main 线程存在，与计算路零耦合。
"""

from __future__ import annotations

import logging
import threading
import time

import cv2

from .buffers import LatestSlot
from .color import WeightedDominantColor
from .config import AppConfig
from .display import DualStreamModulator, StreamRecorder
from .input import open_source
from .output import FrameResult, create_sinks
from .perception import HeatmapPostprocessor, build_backend, upsample
from .preprocess import Preprocessor
from .stability import ColorStabilizer
from .utils.fps import FpsMeter, RateKeeper

log = logging.getLogger(__name__)


class RealtimePipeline:
    def __init__(self, cfg: AppConfig, sinks=None) -> None:
        self.cfg = cfg
        self.pre = Preprocessor(cfg.preprocess)
        self.backend = build_backend(cfg.attention)
        self.post = HeatmapPostprocessor(cfg.attention.top_clip_pct)
        self.focus_cm = WeightedDominantColor(cfg.color, invert=False, name="focus")
        self.ambient_cm = WeightedDominantColor(cfg.color, invert=True, name="ambient")
        self.mod = DualStreamModulator(cfg.modulation)
        self.stab_focus = ColorStabilizer(
            cfg.stability.ema_alpha, cfg.stability.conf_ema_alpha, cfg.stability.jump_reset_dist
        )
        self.stab_ambient = ColorStabilizer(
            cfg.stability.ema_alpha, cfg.stability.conf_ema_alpha, cfg.stability.jump_reset_dist
        )
        self.sinks = sinks if sinks is not None else create_sinks(cfg.output)

        self.crop_slot = LatestSlot()
        self.attn_slot = LatestSlot()  # 元素: (frame_seq, attn_at_crop_res)
        self.latest_result: FrameResult | None = None

        self.stop_evt = threading.Event()
        self.fps = {"capture": FpsMeter(), "attention": FpsMeter(), "color": FpsMeter()}
        self._source = None
        self._threads: list[threading.Thread] = []
        self._t0 = 0.0
        if cfg.display.record_dir and not cfg.display.show:
            log.warning("display.record_dir 已设置但 display.show=False：录制属展示路，无显示模式不会产出文件")

    # ---- 生命周期 -------------------------------------------------------

    def start(self) -> None:
        self._source = open_source(self.cfg.capture)
        if not self._source.open():
            raise RuntimeError(f"无法打开视频源: {self.cfg.capture.source!r}")
        self._t0 = time.monotonic()
        for target, name in (
            (self._capture_loop, "capture"),
            (self._perception_loop, "perception"),
            (self._color_loop, "color"),
        ):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        log.info(
            "管线启动: source=%s backend=%s attn=%.1ffps color=%.1fHz",
            self.cfg.capture.source,
            self.cfg.attention.backend,
            self.cfg.attention.fps,
            self.cfg.color.hz,
        )

    def run_blocking(self) -> None:
        self.start()
        try:
            if not self.cfg.display.show:
                while not self.stop_evt.wait(0.25):
                    pass
            else:
                self._display_loop()
        finally:
            self.stop()

    def stop(self) -> None:
        self.stop_evt.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        if self._source is not None:
            self._source.close()
            self._source = None
        self.sinks.close()

    def _timed_out(self) -> bool:
        return self.cfg.max_seconds > 0 and (time.monotonic() - self._t0) > self.cfg.max_seconds

    # ---- 工作线程 -------------------------------------------------------

    def _capture_loop(self) -> None:
        rewind_spins = 0
        try:
            while not self.stop_evt.is_set():
                if self._timed_out():
                    self.stop_evt.set()
                    break
                frame = self._source.read()
                if frame is None:
                    if self._source.is_file and self.cfg.capture.loop_file and self._source.rewind():
                        rewind_spins += 1
                        if rewind_spins > 10:  # 文件源坏掉时防止无限 rewind
                            log.error("文件源连续多次读不到帧，放弃: %s", self.cfg.capture.source)
                            self.stop_evt.set()
                            break
                        continue
                    log.info("视频源结束，管线停止")
                    self.stop_evt.set()  # 通知其余线程与主循环退出（OBS 掉线/文件播完）
                    break
                rewind_spins = 0
                crop, _ = self.pre.process(frame)
                self.crop_slot.push(crop)
                self.fps["capture"].tick()
        except Exception:
            log.exception("capture 线程异常")
            self.stop_evt.set()

    def _perception_loop(self) -> None:
        keeper = RateKeeper(self.cfg.attention.fps)
        last_seq = -1
        try:
            while not self.stop_evt.is_set():
                keeper.sleep_next()
                seq, crop = self.crop_slot.get()
                if crop is None or seq == last_seq:
                    continue
                last_seq = seq
                model_rgb = self.pre.to_model_rgb(crop)
                raw = self.backend.infer(model_rgb)
                attn = upsample(self.post(raw), crop.shape[:2])
                self.attn_slot.push((seq, attn))
                self.fps["attention"].tick()
        except Exception:
            log.exception("perception 线程异常")
            self.stop_evt.set()

    def _color_loop(self) -> None:
        keeper = RateKeeper(self.cfg.color.hz)
        try:
            while not self.stop_evt.is_set():
                keeper.sleep_next()
                _, crop = self.crop_slot.get()
                _, apair = self.attn_slot.get()
                if crop is None or apair is None:
                    continue
                attn = apair[1]
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                fres = self.focus_cm.compute(rgb, attn)
                ares = self.ambient_cm.compute(rgb, attn)
                if fres.best is None or ares.best is None:
                    continue
                # 置信度 = 主色支配度（胜者得分占全簇总分比例），给下游灯带做淡入淡出依据
                fr, fc = self.stab_focus.update(fres.best.rgb, fres.dominance)
                ar, ac = self.stab_ambient.update(ares.best.rgb, ares.dominance)
                result = FrameResult(
                    ts=time.time(),
                    frame_seq=apair[0],
                    focus_rgb=fr,
                    ambient_rgb=ar,
                    focus_conf=fc,
                    ambient_conf=ac,
                )
                self.latest_result = result
                self.sinks.emit(result)
                self.fps["color"].tick()
        except Exception:
            log.exception("color 线程异常")
            self.stop_evt.set()

    # ---- 显示路（主线程） -------------------------------------------------

    def _display_loop(self) -> None:
        from .viz.windows import PreviewWindows  # 惰性导入：无显示模式不碰 GUI

        preview = PreviewWindows(self.cfg.display)
        # 录制属展示路：只写不读（铁律 2）；无显示窗口时本循环不运行，故不录制
        recorder = (
            StreamRecorder(self.cfg.display.record_dir, self.cfg.capture.target_fps)
            if self.cfg.display.record_dir
            else None
        )
        if recorder is not None and not self.mod.enabled:
            log.warning("modulation.enabled=False：无调制流可录，本次不会产出录制文件")
        try:
            last_seq = -1
            while not self.stop_evt.is_set():
                seq, crop = self.crop_slot.get()
                _, apair = self.attn_slot.get()
                if crop is not None and seq != last_seq:  # 与 perception/color 一致：按 seq 去重
                    last_seq = seq
                    attn = apair[1] if apair is not None else None
                    if self.mod.enabled and attn is not None:
                        focus, ambient = self.mod.apply(crop, attn)
                        if recorder is not None:
                            recorder.write("focus", focus)
                            recorder.write("ambient", ambient)
                    else:
                        focus = ambient = crop
                    key = preview.draw(crop, focus, ambient, attn, self.latest_result, self.fps)
                    if key in (ord("q"), 27):
                        break
                    if key == ord("h"):
                        preview.show_heat = not preview.show_heat
                    if key == ord("f"):
                        preview.toggle_fullscreen()
                elif (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
        finally:
            preview.close()
            if recorder is not None:
                recorder.close()
