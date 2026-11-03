# cognition-pre

> 用一个零标注的自监督视觉模型（DINOv3）实时算出"你在看哪"，据此把画面拆成两个世界——
> **焦点区**和**氛围区**——各自实时提取一个主色，同时输出两路"看得见注意力"的调制视频。

## 设计铁律

1. **定位与读色分离**：注意力热图只决定"谁赢"（权重），颜色永远从原图（裁切帧）读取。
2. **展示与计算分离**：调制视频只给人看，严禁进入任何颜色计算；出数一律走加权聚类。

```
OBS 虚拟摄像头（任意片源，30 fps）
        │
        ▼
  中心方形裁切 ──► 裁切帧（原分辨率，读色基准）
        │
        ▼
  缩放到模型尺寸（518² 或 224²）
        │
        ▼
  DINOv3-S 推理（10 fps）──► 注意力热图 A(x)∈[0,1]（上采样回裁切坐标）
        │
        ├──────────────────────────────┐
        ▼                              ▼
【主色计算，10 Hz】               【双流调制，全帧率】
  焦点色: 原图 × w=1+λA             焦点流: 低注意力区压暗
  氛围色: 原图 × w=1+λ(1−A)         氛围流: 高注意力区压暗
        │                              │
        ▼                              ▼
  加权聚类 + 打分              视频输出（纯展示，
        │                          不进任何计算）
        ▼
  焦点色 + 氛围色（10fps 数据输出）
```

完整系统设计与模块说明见 [docs/design.md](docs/design.md)；阶段计划见 [docs/roadmap.md](docs/roadmap.md)。

## 仓库结构

```
uv.lock                     # uv 锁文件（uv sync 按此复现环境）
configs/default.yaml        # 默认配置（与 config.py 的 dataclass 一一对应）
docs/                       # 设计文档 / 阶段计划
scripts/                    # 辅助脚本（生成演示视频 / 全景验证 / 外发接收端）
src/cognition_pre/
  config.py                 # 配置 dataclass（单一事实源）
  pipeline.py               # 实时管线编排（采集/感知/主色三线程 + 主线程显示）
  buffers.py                # "最新者赢"单槽缓冲
  preprocess.py             # M2 中心裁切 + 模型缩放
  input/frame_source.py     # M1 摄像头/文件/合成源
  perception/               # M3 注意力后端（mock | dinov3）+ 热图后处理
  color/dominant.py         # M4 加权 Mini-Batch K-Means + 打分器（对比度/迟滞）
  display/modulation.py     # M5 双流亮度调制 + 双流录制
  stability.py              # M6 色值 EMA + 大跳变重置
  output/                   # FrameResult + sinks（控制台/JSONL/UDP/WebSocket）
  viz/windows.py            # 预览画布（热图叠加 + 色板）
  cli.py                    # 命令行入口
tests/                      # pytest（全走 mock + 合成源，无摄像头无权重可跑）
```

## 快速开始

包管理用 [uv](https://docs.astral.sh/uv/)（不用 pip）：

```bash
uv sync        # 创建 .venv + 按 uv.lock 安装（含 dev 组；本项目本体 editable 安装）
```

### 1) 无权重冒烟（mock 注意力后端）

```bash
uv run cognition-pre doctor                                   # 依赖版本 + 摄像头探测
uv run cognition-pre run --mock --no-display --max-seconds 3  # 无窗口全链路自检
uv run cognition-pre run --mock                               # 有窗口预览（mock 热图会动）
uv run cognition-pre run --source test:split --mock           # 合成源，连摄像头都不用
```

### 2) 本地视频联调（无 OBS）

```bash
python scripts/make_demo_video.py --out data/demo.avi
uv run cognition-pre run --source data/demo.avi --mock
```

### 3) 接 OBS 虚拟摄像头

OBS 里：设置 → 直播/视频 → 开启"虚拟摄像头"（Start Virtual Camera），然后：

```bash
uv run cognition-pre doctor        # 找到 OBS 虚拟摄像头所在索引
uv run cognition-pre run --mock --source 1 --backend dshow   # 索引按 doctor 结果填
```

Windows 打开虚拟摄像头建议 `--backend dshow`（msmf 偶发打不开/黑帧）。

### 4) 真·DINOv3（Phase 1）

DINOv3 权重是 gated 仓库，需要：

1. 上 [HuggingFace 模型页](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
   登录并接受许可；
2. `uvx --from "huggingface_hub[cli]" hf auth login`（旧命令 `huggingface-cli` 已改名 `hf`），或设置环境变量 `HF_TOKEN`；
3. 安装模型依赖并运行：

```bash
uv sync --extra model   # CPU 版 torch；CUDA 版建议 uv pip install torch --index-url https://download.pytorch.org/whl/<cu版本>
uv run cognition-pre run --dinov3 --source 1 --device cuda
# 显存富余可上 518：--size 518（注意 docs/roadmap.md 里 518² 的显存说明）
```

### 输出与外发

```bash
# 主色结果（10Hz JSON）落到本地文件 + UDP/WebSocket 外发给灯带/特效
uv run cognition-pre run --mock --jsonl outputs/colors.jsonl --udp 127.0.0.1:9999 --ws ws://127.0.0.1:8765

# 另一终端：接收端"灯带"模拟器（终端色块实时显示焦点色/氛围色）
uv run python scripts/receiver_demo.py --udp 127.0.0.1:9999        # UDP 接收
uv run python scripts/receiver_demo.py --ws 127.0.0.1:8765         # WS 服务端（需 uv sync --extra ws）

# 录制两路调制视频（焦点流/氛围流，展示路产物，不进任何计算）
uv run cognition-pre run --mock --record-dir outputs/rec
```

协议即 `FrameResult.to_dict()`（conf 为主色**支配度**：胜者得分占全簇总分比例，供灯带淡入淡出）：

```json
{"ts": 1760000000.1, "frame_seq": 1024, "focus_rgb": [200, 30, 30], "ambient_rgb": [30, 60, 200],
 "focus_conf": 0.42, "ambient_conf": 0.35}
```

### Phase 0 全景专项验证

```bash
uv run python scripts/validate_panorama.py --source data/wide.avi --jsonl outputs/pano.jsonl
```

同一素材分别按"中心方形裁切"与"全画幅"逐帧算氛围主色，输出 jump（帧间跳变）/
drift（时间漂移）/diver（两几何偏差）指标，判读是否需要立项全画幅备选方案。

## 配置

所有参数集中在 `configs/default.yaml`（镜像 `src/cognition_pre/config.py` 中的 dataclass），
CLI 标志可临时覆盖。常用项：

| 项 | 默认 | 说明 |
| --- | --- | --- |
| `attention.fps` | 10 | 热图更新频率，中间帧复用最近热图 |
| `attention.top_clip_pct` | 0.05 | 归一后截掉前 5% 极值，防孤斑通吃 |
| `color.hz` | 10 | 主色计算频率（=数据输出频率） |
| `color.lam` | 0.75 | 注意力权重强度 λ：w = 1 + λ·A(x) |
| `color.hysteresis_margin` | 0.15 | 簇切换迟滞：挑战簇须有 (1+margin) 分数优势才夺主 |
| `color.lab_weights` | [0.5,1,1] | Lab 特征权重：L 低、a/b 高 |
| `modulation.gamma` | 0.3 | 调制下限 γ：mask = γ + (1−γ)·s(x) |
| `stability.ema_alpha` | 0.05 | 色值 EMA（按 10Hz 调校，τ≈2.0s；改 color.hz 按 α = 1 − exp(−1/(hz·τ)) 重调） |
| `stability.jump_reset_dist` | 60 | RGB 距离超此值直接置入（防换景爬色）；0 关闭 |

## 测试

```bash
uv run pytest -q             # 全部走 mock + 合成源，CPU 秒级跑完
```

## 约定（给人与 Agent）

见 [AGENTS.md](AGENTS.md)。核心三条：mock-first（改动后 mock 测试必须全绿）、
管线内部 BGR / 颜色计算与模型输入 RGB、core 安装不依赖 torch。
