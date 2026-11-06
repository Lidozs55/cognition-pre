import json
import threading

import numpy as np

from cognition_pre.pipeline import RealtimePipeline
from cognition_pre.perception import MockAttentionBackend
from cognition_pre.perception.base import AttentionBackend


def test_infer_world_default_is_attention():
    """缺省 infer_world 返回 (注意力, 注意力) 同对象——mock/NCut 关闭时世界=注意力，
    且管线上采样只做一次依赖该恒等（pipeline._perception_loop 的 is 判断）。"""
    b = MockAttentionBackend(animated=False)
    rgb = np.full((64, 64, 3), 127, np.uint8)
    a, w = b.infer_world(rgb)
    assert a is w
    np.testing.assert_array_equal(a, b.infer(rgb))


def test_color_uses_world_map_not_attention(tmp_path, cfg_factory, monkeypatch):
    """NCut 集成路由回归（2026-11-06）：主色计算必须吃世界图而非注意力图——
    注意力说主体在左（红半），世界图说主体在右（蓝半）；若路由错回注意力，
    焦点色会翻成红。展示路不在测试范围（无窗口），此处只锁计算路。"""
    from cognition_pre import pipeline as pl

    class DualBackend(AttentionBackend):
        name = "dual-fake"

        def infer(self, rgb_u8):
            g = np.zeros((4, 4), np.float32)
            g[:, :2] = 1.0  # 注意力：左半为"主体"
            return g

        def infer_world(self, rgb_u8):
            w = np.zeros((4, 4), np.float32)
            w[:, 2:] = 1.0  # 世界图：右半为"主体"
            return self.infer(rgb_u8), w

    class StaticSource:
        is_file = True

        def __init__(self, frame):
            self.frame = frame

        def open(self):
            return True

        def read(self):
            return self.frame.copy()

        def rewind(self):
            return True

        def close(self) -> None: ...

    frame = np.zeros((128, 128, 3), np.uint8)
    frame[:, :64] = (0, 0, 255)  # 左半红（BGR）
    frame[:, 64:] = (255, 0, 0)  # 右半蓝（BGR）
    monkeypatch.setattr(pl, "open_source", lambda _cfg: StaticSource(frame))
    monkeypatch.setattr(pl, "build_backend", lambda _cfg: DualBackend())

    cfg = cfg_factory(tmp_path)
    pipe = RealtimePipeline(cfg)
    pipe.run_blocking()
    assert pipe.latest_result is not None
    d = pipe.latest_result.to_dict()
    assert d["focus_rgb"][2] > 200 and d["focus_rgb"][0] < 50  # 焦点=蓝（世界图右半）
    assert d["ambient_rgb"][0] > 200 and d["ambient_rgb"][2] < 50  # 氛围=红（左半）


def test_mock_pipeline_end_to_end(tmp_path, cfg_factory):
    """无权重无摄像头跑通全链路：采集→mock 热图→双主色→EMA→JSONL 输出。"""
    cfg = cfg_factory(tmp_path, overrides={"color.hz": 5.0})
    pipe = RealtimePipeline(cfg)
    pipe.run_blocking()  # max_seconds=2.5 自动停

    assert pipe.latest_result is not None
    d = pipe.latest_result.to_dict()
    for key in ("focus_rgb", "ambient_rgb"):
        assert len(d[key]) == 3
        assert all(0 <= v <= 255 for v in d[key])
    assert 0.0 <= d["focus_conf"] <= 1.0

    lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 3
    last = json.loads(lines[-1])
    assert set(last) == {"ts", "frame_seq", "focus_rgb", "ambient_rgb", "focus_conf", "ambient_conf"}


def test_pipeline_runs_again_after_reset(tmp_path, cfg_factory):
    """stop 后线程已清理，可重复构建（冒烟的幂等性）。"""
    cfg = cfg_factory(tmp_path)
    RealtimePipeline(cfg).run_blocking()
    pipe2 = RealtimePipeline(cfg)
    pipe2.run_blocking()
    assert pipe2.latest_result is not None


def test_pipeline_stops_when_source_ends(tmp_path, cfg_factory, monkeypatch):
    """源正常结束（文件播完/OBS 掉线）必须停机——演示中途掉源不能挂死（P1 回归）。"""
    from cognition_pre import pipeline as pl

    class DyingSource:
        """读出一帧后即返回 None，模拟文件播完/摄像头拔出。"""

        is_file = True

        def open(self) -> bool:
            return True

        def read(self):
            if not getattr(self, "_done", False):
                self._done = True
                return np.zeros((48, 64, 3), np.uint8)
            return None

        def rewind(self) -> bool:
            return False

        def close(self) -> None: ...

    monkeypatch.setattr(pl, "open_source", lambda _cfg: DyingSource())
    cfg = cfg_factory(tmp_path, overrides={"max_seconds": 0.0})  # 关掉超时保险，暴露挂死
    pipe = RealtimePipeline(cfg)
    worker = threading.Thread(target=pipe.run_blocking, daemon=True)
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
