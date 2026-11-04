# AGENTS.md — 仓库约定（人 / Agent 通用）

## 设计铁律（违反即返工）

1. **定位与读色分离**：注意力热图只产生权重（"谁赢"），颜色永远从原图（裁切帧）像素读取。
   任何"从热图/模型特征里取颜色"的实现都是错的。
2. **展示与计算分离**：`display/` 的调制输出只给人看，严禁作为任何计算的输入；
   出数只走 `color/` 的加权聚类。

## 工程约定

- **mock-first**：默认后端是 mock。任何改动必须保证 `uv run pytest -q` 全绿
  （测试不依赖摄像头、权重、GPU、窗口）。
- **包管理用 uv**：装环境 `uv sync`（默认含 dev 组，按 uv.lock 复现）；加依赖
  `uv add <pkg>`，可选功能 `uv sync --extra model|ws`；不要直接用 pip 或手工改动 .venv。
- **颜色空间边界**：管线内部 BGR（cv2 原生）；转 RGB 只发生在两处边界——
  模型输入（`preprocess.to_model_rgb`）与颜色计算（`pipeline._color_loop`）。
  新代码不要在别处悄悄转换。
- **依赖分层**：`uv sync` 的默认安装（core）不含 torch；torch/transformers 只属于
  `[model]` extra（`uv sync --extra model`），且只能在 `perception/dinov2_backend.py` 内延迟导入。
- **配置顺序**：新配置项先加 `config.py` 的 dataclass 字段（带默认值）→
  同步 `configs/default.yaml` → 再在消费方使用。未知键会被告警。
- **线程模型**：采集/感知/主色三个工作线程 + 主线程显示（cv2 GUI 必须主线程）。
  线程间只通过 `LatestSlot`（最新者赢）通信，不新增共享可变状态。
- **注释/文档用中文**；cv2 画布上的文字用英文（Hershey 字体不渲染中文）。
  日志避免非常规 unicode 符号（Windows 控制台 GBK）。

## 验证命令

```bash
uv run pytest -q                                              # 单测
uv run cognition-pre run --mock --no-display --max-seconds 3  # 冒烟（无摄像头无窗口）
uv run cognition-pre doctor                                   # 环境/摄像头自检
```

## 当前状态与下一步

- Phase 0/2/3 的无权重部分已完成：v1 打分器（对比度项 + 簇切换迟滞）、大跳变重置、
  支配度置信度、全景验证脚本（`scripts/validate_panorama.py`）、外发接收端
  （`scripts/receiver_demo.py`，UDP/WS 双通道 10Hz 实测通过）、双流录制（--record-dir）。
  本机调试：OBS 虚拟摄像头在 camera 1（dshow），camera 0 是纯黑实体摄像头。
- 2026-11-04：全仓切换 DINOv2（DINOv3 gated 权重申请被 Meta 拒绝，全部变体 403）。
  `--dinov2` 真权重全链路已在 OBS 虚拟摄像头 + CPU 上跑通（224² 实测 ~18fps，热图
  判读通过）。注意：transformers 5.x 必须 `attn_implementation="eager"` 才返回注意力
  权重（已固化在 backend，改动加载逻辑时勿丢）。
- 待办：真素材人眼评估（Phase 0 判读与 Phase 2 打分权重精调依赖它）；518²/224² 稳定性
  对比与 rollout 对照见 `docs/roadmap.md` Phase 1。
