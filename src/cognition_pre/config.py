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
    layer_start_frac: float = 0.92  # int(12*0.92)=11 → 仅末层 CLS→patch（官方 DINO 配方）。
    # 同帧 A/B 定案（scripts/compare_heatmaps.py）：末层最干净，中后层背景散斑严重；registers 变体无收益
    layer_end_drop: int = 1  # 与 start_frac 联合定层区间 [int(n*start_frac), n-end_drop)
    top_clip_pct: float = 0.05  # 归一后截掉前 5% 极值
    smooth_frac: float = 0.05  # 高斯核边长 = ceil(frac*网格边长)取奇：224²(16 网格)→1（不模糊）、
                               # 518²(37 网格)→3；背景散斑已由 norm_floor_pct 治理，平滑只做轻度并斑。
                               # 勿改回 sigma 语义：OpenCV 3σ 自动核（sigma=1 即 7×7）会抹平主体尖角（用户实测）
    norm_floor_pct: float = 0.0  # 可选背景平台抑制：该分位的值归零。默认关——无直接论文依据（LOST 谱系
                                 # 的背景阈值是符号/Otsu 类而非分位归零），且隐含"主体 <50% 画面"假设，
                                 # 特写会误伤主体。背景泄漏困扰时 0.5 起试；0 = min-max 归一


@dataclass
class ColorConfig:
    hz: float = 10.0  # 主色计算频率（=数据输出频率，10fps）
    n_clusters: int = 5  # 簇数上限：K 固定是 warm-start 的前提，动态类数放合并层（见 merge_dist）
    merge_dist: float = 0.10  # 近邻簇心合并阈值（聚类同空间、L 低权重）：同色系深浅碎片并为一族，
    # 同色系场景自动收敛到 2-3 个候选（2026-11-06 实测反馈"类数偏多"），真多彩不丢分辨率。0 关闭
    # 主色注意力加权：与展示路调制同形的 ramp（utils/ramp.py 同一份实现），A<lo 从该
    # 世界剔除。权重要能压到 0——旧式 w=1+λ·A 给背景保底权重、按面积碾压主体，λ 排
    # 不掉（实测被背景主导）；lo/hi 是两世界分离杠杆（趋同→调大 lo）。
    weight_lo: float = 0.2
    weight_hi: float = 0.8
    lab_weights: tuple[float, float, float] = (0.5, 1.0, 1.0)  # L 低、a/b 高
    sample_budget: int = 12000  # 抽样像素预算：8000 时多彩场景统计不稳；12000 实测双实例 ~21ms/轮（warm-start 后）。
    # 按本世界面积自适应消费（掩膜全分辨率先定，抽样只发生在本世界像素上）——小主体不再饥饿
    kmeans_batch_size: int = 2048  # 大 batch 帮助 MiniBatchKMeans 早停收敛
    kmeans_n_init: int = 2
    min_mass_frac: float = 0.02
    seed: int = 0
    # 簇切换迟滞（动量调制，Phase 2；注视稳定性时序类似物——Itti & Koch 2001 的 IOR
    # 方向相反，故不标 IOR）：挑战簇须有 (1 + margin×动量) 分数优势才夺主，动量三态
    # （连胜攒/被压制慢泄/换景快泄）见 dominant.py _select；margin 0.15→0.10 因旧式
    # 恒定门槛锁死过久（2026-11-05 反馈）。参考色 = 赢家 Lab 慢 EMA，Lab 距离 <= dist
    # 视为"在位色"。
    hysteresis_margin: float = 0.10
    hysteresis_dist: float = 0.12
    # 保护色惩罚（center-surround 对齐，2026-11-06）：焦点对比度基准 = 上一帧氛围主色
    # （Itti & Koch 2001 的 surround 即背景），贴背景色的候选打折；缺参照退回全帧均色。
    # 单向接线（氛围侧不接），避免双实例互推的反馈环。
    surround_contrast: bool = True


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
