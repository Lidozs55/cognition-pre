import json
import threading

import numpy as np

from cognition_pre.pipeline import RealtimePipeline


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
