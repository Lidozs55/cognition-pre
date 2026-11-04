"""配置：dataclass 是单一事实源，YAML 仅作覆盖。

新增配置项的顺序：先在对应 dataclass 加字段与默认值 → 同步 configs/default.yaml → 再在消费方使用。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_origin, get_type_hints

import yaml


@dataclass
class CaptureConfig:
    source: str = "0"  # 摄像头索引 / 视频文件路径 / "test:split" 合成源
    backend: str = "auto"  # auto | dshow | msmf | ffmpeg（Windows 打 OBS 虚拟摄像头建议 dshow）
    target_fps: float = 30.0  # 文件源的播放节拍；摄像头源由设备决定
    loop_file: bool = True  # 文件源播完是否循环


@dataclass
class PreprocessConfig:
    square_crop: bool = True  # M2：中心方形裁切，边长 = min(H, W)
    model_input_size: int = 224  # 518 或 224；518 更准但更慢更耗显存


@dataclass
class AttentionConfig:
    backend: str = "mock"  # mock | dinov2
    hf_repo: str = "facebook/dinov2-small"
    device: str = "cpu"  # cpu | cuda
    fps: float = 10.0  # 热图更新频率；中间帧复用最近热图
    layer_start_frac: float = 0.6  # 取中后层 CLS→patch 注意力
    layer_end_drop: int = 1  # 丢弃最后 N 层（深层易糊）
    top_clip_pct: float = 0.05  # 归一后截掉前 5% 极值
    use_rollout: bool = False  # 预留：跨层 rollout 对照（Phase 1 实现）


@dataclass
class ColorConfig:
    hz: float = 10.0  # 主色计算频率（=数据输出频率，10fps）
    n_clusters: int = 5
    lam: float = 3.0  # λ：w = 1 + λ·A(x)。焦点/氛围分布分离的主杠杆：0.75 时 w∈[1,1.75]，
                      # 质量项碾压注意力信号导致两世界趋同；3.0 时注意区权重是背景 4 倍
    lab_weights: tuple[float, float, float] = (0.5, 1.0, 1.0)  # L 低、a/b 高
    sample_budget: int = 12000  # 抽样像素预算：8000 时多彩场景统计不稳；12000 实测双实例 ~21ms/轮（warm-start 后）
    kmeans_batch_size: int = 2048  # 大 batch 帮助 MiniBatchKMeans 早停收敛
    kmeans_n_init: int = 2
    min_mass_frac: float = 0.02
    seed: int = 0
    # 簇切换迟滞（Phase 2；注视稳定性的时序类似物——论文 Itti & Koch 2001 的 IOR
    # 方向相反，故不标 IOR）：挑战簇须以 (1+margin) 的分数优势才能夺主；与上一帧
    # 主色 Lab 距离 <= dist 的候选视为"在位色"。防同屏两簇分数接近时的主色闪烁。
    hysteresis_margin: float = 0.15
    hysteresis_dist: float = 0.12


@dataclass
class ModulationConfig:
    # 分段线性阈值化（ramp）：A<=lo 全黑（black_floor 残亮）、A>=hi 原始亮度、中间线性。
    # 焦点流对 A(x) 生效；氛围流对 1−A(x) 生效——阈值独立配置。
    focus_lo: float = 0.25
    focus_hi: float = 0.65
    ambient_lo: float = 0.25
    ambient_hi: float = 0.65
    black_floor: float = 0.0  # mask=0 时的残亮比例（0=真全黑；嫌死黑可给 0.05）
    enabled: bool = True


@dataclass
class StabilityConfig:
    # EMA 系数是"每次更新"语义：默认 0.05@10Hz ↔ 时间常数 τ≈2.0s；旧版 0.35@1.5Hz ↔ τ≈1.5s
    # （同量级，非严格等效）。改 color.hz 时按 α = 1 − exp(−1/(hz·τ)) 重调。
    ema_alpha: float = 0.05
    conf_ema_alpha: float = 0.03
    # 大跳变重置：新主色与 EMA 状态的 RGB 距离超过该值时直接置入而非 EMA 缓移，
    # 避免切场景时颜色"爬过"中间色相。0 关闭。
    jump_reset_dist: float = 60.0


@dataclass
class OutputConfig:
    jsonl_path: str = ""
    udp_host: str = ""
    udp_port: int = 0
    ws_url: str = ""
    log_every: float = 1.0  # 控制台打印间隔（秒）；0 关闭


@dataclass
class DisplayConfig:
    show: bool = True
    window_width: int = 320
    show_heatmap: bool = True
    fullscreen: bool = False  # 启动即全屏（运行中按 f 切换）；全屏时画布 letterbox 补黑边
    record_dir: str = ""  # 非空则把焦点/氛围两路调制流录制到该目录（仅展示路，永不回读）


@dataclass
class AppConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    modulation: ModulationConfig = field(default_factory=ModulationConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    max_seconds: float = 0.0  # >0 定时停（自检/批处理）


def _build(cls: type, data: dict[str, Any]):
    """由 dict 递归构建 dataclass，忽略未知键（告警），list 自动转 tuple 字段。"""
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        warnings.warn(f"{cls.__name__}: 忽略未知配置项 {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        t = hints.get(f.name)
        if hasattr(t, "__dataclass_fields__") and isinstance(v, dict):
            v = _build(t, v)
        elif get_origin(t) is tuple and isinstance(v, list):
            v = tuple(v)
        kwargs[f.name] = v
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> AppConfig:
    """path 为空返回内置默认；否则读 YAML 覆盖。"""
    if path is None:
        return AppConfig()
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _build(AppConfig, data)


def set_option(cfg: AppConfig, dotted: str, value: Any) -> None:
    """按 'section.field' 点路径改配置，供 CLI / 测试使用。"""
    parts = dotted.split(".")
    obj: Any = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise KeyError(f"未知配置项: {dotted}")
    cur = getattr(obj, parts[-1])
    if isinstance(cur, tuple) and isinstance(value, list):
        value = tuple(value)
    setattr(obj, parts[-1], value)
