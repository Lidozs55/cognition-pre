# QUICKSTART — 启动指南

极简版：装环境 → 基础启动 → 两个最常用参数（换大模型 / 换输入分辨率）。
完整设计与全部参数表见 [README.md](README.md) 与 [docs/design.md](docs/design.md)。

## 1. 环境安装（极简）

Python >= 3.10。

```powershell
# 装 uv（Windows；没有 uv 时执行一次）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
uv sync                # core 依赖：mock 全链路可跑（不含 torch）
uv sync --extra model  # --dinov2 需要：torch(CPU) + transformers
# 有 NVIDIA 显卡：torch 换 CUDA 版，按 https://pytorch.org 官方索引装
```

**只有 pip、不想装 uv 时**（pip 不读 uv.lock，版本以 pyproject.toml 下限为准）：

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows（Git Bash 用 source .venv/Scripts/activate）
pip install -e .                      # core
pip install -e ".[model]"             # --dinov2 需要
pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA 卡按需（可选）
```

装完自检（pip 环境把 `uv run` 换成直接调命令，下同）：

```bash
uv run cognition-pre doctor           # 依赖版本 + CUDA 可用性 + 摄像头探测
```

## 2. 基础启动

OBS 里开启虚拟摄像头（Start Virtual Camera）后：

```bash
uv run cognition-pre run --dinov2 --source 1 --backend dshow
```

- `--source 1`：摄像头索引，按 doctor 输出填（OBS 虚拟摄像头可能落在 0 或 1；本机调试在 1）
- `--backend dshow`：Windows 打开虚拟摄像头建议 dshow（msmf 偶发打不开/黑帧）
- 首次运行自动从 HuggingFace 拉取公开权重（无需 token，之后走本地缓存）
- 预览窗按 `f` 切换全屏，`q` / Ctrl+C 退出

无 OBS / 无摄像头冒烟自检：

```bash
uv run cognition-pre run --mock --no-display --max-seconds 3
```

## 3. 两个最常用参数

### 换更大的模型：--repo

```bash
uv run cognition-pre run --dinov2 --source 1 --backend dshow --repo facebook/dinov2-base
```

- 默认 `facebook/dinov2-small`（ViT-S/14，22M）；`--repo facebook/dinov2-base` 换
  ViT-B/14（86M），热图通常更稳
- base 在 CPU 上只有 ~4-6fps，**只在展示机（GPU）用**，配 `--device cuda`

### 换输入分辨率：--size

```bash
uv run cognition-pre run --dinov2 --source 1 --backend dshow --size 518
```

- `--size 224`（默认）：本机 CPU 实测 ~18fps，满足 10fps 热图率
- `--size 518`：热图判读优于 224，但 CPU 单帧 ~0.45s 跑不动 10fps，**仅展示机（GPU）用**
- 224/518 均被 patch 14 整除（16/37 网格），无对齐问题

参数可叠加（展示机全开示例）：

```bash
uv run cognition-pre run --dinov2 --source 1 --backend dshow --repo facebook/dinov2-base --size 518 --device cuda --fullscreen
```

## 4. 常用可选参数

```bash
--jsonl outputs/colors.jsonl   # 10Hz 主色结果落盘
--udp 127.0.0.1:9999           # UDP 外发（接收端：uv run python scripts/receiver_demo.py --udp 127.0.0.1:9999）
--ws ws://127.0.0.1:8765       # WebSocket 外发（需 uv sync --extra ws）
--record-dir outputs/rec       # 录制焦点/氛围两路调制流（仅展示路）
--no-ncut-color                # 主色世界掩膜退回注意力 ramp（A/B 对照与回退）
--no-surround-contrast         # 关闭保护色惩罚（A/B 对照）
```
