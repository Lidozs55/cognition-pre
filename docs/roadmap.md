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

## Phase 1 — DINOv2 真权重接入

> 2026-11-04：原 DINOv3 方案的 gated 权重申请被 Meta 拒绝（实测全部变体 403、
> "rejected by the repo's authors"），全仓切换到公开免审批的 DINOv2。同属 DINO 家族，
> "中后层 CLS→patch 注意力 ≈ 对象级分割"的用法不变；patch 16→14，且无 register token。

- [x] 首次 `--dinov2` 跑通（OBS 虚拟摄像头 + CPU），Phase 1 核对项结果：
  - patch 14：224/14=16、518/14=37 均整除（原 vits16 的 518 不整除问题不复存在）
  - prefix 切片偏移 = 1（DINOv2 无 register token，`num_prefix_tokens` 缺省路径恰好正确）
  - 层区间 [7, 11)（12 层 × 0.6、末层丢弃）热图有结构：屏幕场景热区落在文字/控件上
    （`scripts/check_backend.py` 落盘"裁切帧 | 热图叠加"并排图人工判读）
  - transformers 5.x 默认 SDPA 不物化注意力权重，必须 `attn_implementation="eager"`
    （已固化在 `dinov2_backend.py`，否则 `output_attentions=True` 拿到空 attentions）
- [x] 性能：CPU 224² 实测 ~18 fps，满足 10fps 热图率（本机无 NVIDIA 卡，不做 CUDA 路径）
- [ ] `--size 518` CPU 耗时实测，及 518² 与 224² 的焦点色稳定性对比（同一素材）
- [ ] 展示机（GPU）上 dinov2-base（ViT-B/14, 86M）热图 A/B：`--repo` 已备；本机 CPU
      预计 4-6fps 跑不动，只在展示机上比
- [ ] 热图质量对照：中后层平均 vs attention rollout（`use_rollout` 预留项）
- [ ] 真素材（电影片段/人物直播）人眼评估：热图是否落在主体上

## Phase 2 — 打分器细化（结合 Itti & Koch 2001）

论文在仓库根：`Itti_Koch_2001.pdf`。

- [x] 机制映射落地（v1 打分器）：center-surround/空间竞争 → 簇对比度项（近似：
      全局色彩显著性，无空间结构）；WTA → 迟滞选主（注视稳定性时序类似物，
      IOR 方向相反未采用）（`hysteresis_margin`/`hysteresis_dist`）
- [x] 簇切换迟滞：挑战簇优势不足 margin 时保持在位主色，抑制"主色闪烁"
- [x] `ColorStabilizer` 大跳变重置（`jump_reset_dist`：换景直接置入，不爬中间色相）
- [x] 置信度 = 主色支配度（胜者得分占全簇总分），进 `FrameResult`，供灯带淡入淡出
- [x] 主色闪烁治理（2026-11-04 机制诊断 + 第一批落地）：
  - 机制链：每轮重抽样重聚类 → 簇集合微抖 → 迟滞只护 0.12 Lab 内近色、远距竞争
    主色裸奔 → 赢家互换 RGB 距离 >60 触发 jump_reset 置入 → τ≈2s 的 EMA 失效 →
    10Hz 全幅闪烁（静态多彩素材必现）
  - 已落地：warm-start 簇心延续（簇集合时域连贯）+ 抽样预算 8000→12000（多彩场景
    统计不稳）+ 调制改双阈值 ramp（观感问题，非闪烁主因）+ λ 0.75→3.0（实测反馈
    焦点/氛围趋同，w∈[1,1.75] 时质量项碾压注意力信号）
  - 待真素材实测后启用：jump_reset 持续性门槛（连续 N 轮超阈值才置入，防闪 C）、
    hysteresis_margin 0.15→0.3（防闪 D 剩余项）
- [ ] 打分权重精调（对比度/彩度/紧凑度的系数）依赖真素材人眼评估反馈

## Phase 3 — 输出外发联调

- [x] 联调工具 `scripts/receiver_demo.py`（终端灯带模拟器；UDP 与 WS 双通道均已
      按 10Hz 端到端验证）
- [ ] WebSocket/UDP → 真实灯带/特效端到端（协议：`FrameResult.to_dict()` JSON）
- [ ] 断线重连与积压策略压测（`output/sinks.py` 已有 1s 退避 + 丢旧保实时）

## Phase 4 — 性能与精度

- [ ] 518² vs 224²：精度收益 vs 实时性权衡的量化报告（JSONL 数据已有出口）
- [ ] 感知线程剖析：resize / 推理 / 上采样各占比
- [ ] 视需要：torch.compile / ONNX Runtime 导出
