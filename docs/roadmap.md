# Roadmap

## Phase 0 — 骨架与无权重联调（本次完成）

- [x] 仓库结构 / 配置系统 / 三线程管线骨架
- [x] mock 注意力后端全链路：采集 → 热图 → 双主色 → EMA → 调制预览 → 输出
- [x] 冒烟：`cognition-pre run --mock --no-display --max-seconds 3`
- [x] 测试套件（mock + 合成源，CPU 秒级）
- [x] **全景图专项验证（M2 已知代价）**：工具已建 `scripts/validate_panorama.py`
      （中心裁切 vs 全画幅氛围色对比，输出 jump/drift/diver 指标）；合成宽幅素材上
      diver 均值 53.9（裁切显著改变氛围色）——**真实素材跑一轮并判读仍待人做**
- [ ] 用真素材（电影片段/直播流）人眼评估：焦点色是否"落在主角身上"、氛围色是否"像环境"

## Phase 1 — DINOv3 真权重接入

前置：HF 页面接受 DINOv3 许可 → `HF_TOKEN`（`uvx --from "huggingface_hub[cli]" hf auth login`）→ `uv sync --extra model`。

- [ ] 首次 `--dinov3 --source 0` 跑通，核对：
  - **518 与 vits16 组合**：518/16 不整除，patch embed 卷积 floor 后 patch 数恰为平方数，
    现有校验不拦截——显式验证 pos-embed 插值行为，否则从 CLI choices 移除 518
  - `num_prefix_tokens` 切片偏移（CLS + register tokens）
  - 层区间选择（`layer_start_frac=0.6`、末层丢弃）是否给出对象级热图
  - `output_attentions=True` 的显存开销（224² 轻松 / 518² 约 0.6GB fp32）
- [ ] 热图质量对照：中后层平均 vs attention rollout（`use_rollout` 预留项）
- [ ] `--size 518` 与 `--size 224` 的焦点色稳定性对比（同一素材）
- [ ] 性能：GPU 上热图实际 fps；不足则改 hook 精取所选层（省显存省算力）

## Phase 2 — 打分器细化（结合 Itti & Koch 2001）

论文在仓库根：`Itti_Koch_2001.pdf`。

- [x] 机制映射落地（v1 打分器）：center-surround/空间竞争 → 簇对比度项（近似：
      全局色彩显著性，无空间结构）；WTA → 迟滞选主（注视稳定性时序类似物，
      IOR 方向相反未采用）（`hysteresis_margin`/`hysteresis_dist`）
- [x] 簇切换迟滞：挑战簇优势不足 margin 时保持在位主色，抑制"主色闪烁"
- [x] `ColorStabilizer` 大跳变重置（`jump_reset_dist`：换景直接置入，不爬中间色相）
- [x] 置信度 = 主色支配度（胜者得分占全簇总分），进 `FrameResult`，供灯带淡入淡出
- [ ] 打分权重精调（对比度/彩度/紧凑度的系数）依赖真素材人眼评估反馈

## Phase 3 — 输出外发联调

- [x] 联调工具 `scripts/receiver_demo.py`（终端灯带模拟器；UDP 与 WS 双通道均已
      按 10Hz 端到端验证）
- [ ] WebSocket/UDP → 真实灯带/特效端到端（协议：`FrameResult.to_dict()` JSON）
- [ ] 断线重连与积压策略压测（`output/sinks.py` 已有 1s 退避 + 丢旧保实时）

## Phase 4 — 性能与精度

- [ ] 518² vs 224²：精度收益 vs 实时性权衡的量化报告（JSONL 数据已有出口）
- [ ] 感知线程剖析：resize / 推理 / 上采样各占比
- [ ] 视需要：DINOv3 patch 14 变体、torch.compile / ONNX Runtime 导出
