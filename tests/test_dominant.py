import numpy as np
import pytest
from conftest import make_cfg

from cognition_pre.color.dominant import ColorCandidate, WeightedDominantColor, _score


def _split_scene():
    """RGB：左半纯红 (200,30,30)，右半纯蓝 (30,60,200)；注意力全押左边。"""
    h = w = 160
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = (200, 30, 30)
    img[:, w // 2 :] = (30, 60, 200)
    s = np.zeros((h, w), np.float32)
    s[:, : w // 2] = 1.0
    return img, s


def test_focus_picks_attention_side():
    cfg = make_cfg(overrides={"color.seed": 0})
    img, s = _split_scene()
    res = WeightedDominantColor(cfg.color, invert=False, name="focus").compute(img, s)
    assert res.best is not None
    assert res.best.rgb[0] > 150 and res.best.rgb[2] < 100  # 红胜出


def test_ambient_picks_inverse_side():
    cfg = make_cfg(overrides={"color.seed": 0})
    img, s = _split_scene()
    res = WeightedDominantColor(cfg.color, invert=True, name="ambient").compute(img, s)
    assert res.best is not None
    assert res.best.rgb[2] > 150 and res.best.rgb[0] < 100  # 蓝胜出


def test_flat_image_still_returns():
    """平坦注意力（A=0 < weight_lo → 全零权重）退化为无权重，输出不断流。"""
    cfg = make_cfg()
    img = np.full((64, 64, 3), 128, np.uint8)
    s = np.zeros((64, 64), np.float32)
    res = WeightedDominantColor(cfg.color).compute(img, s)
    assert res.best is not None
    assert all(120 < v < 140 for v in res.best.rgb)


def test_candidates_sorted_and_scored():
    cfg = make_cfg()
    img, s = _split_scene()
    res = WeightedDominantColor(cfg.color).compute(img, s)
    scores = [c.score for c in res.candidates]
    assert scores == sorted(scores, reverse=True)
    assert res.candidates[0].score == res.best.score


def test_contrast_boosts_distinctive_cluster():
    """Phase 2 对比度项：注意力区的小块高对比色应胜过大面积背景色。"""
    cfg = make_cfg(overrides={"color.seed": 0})
    h = w = 160
    img = np.full((h, w, 3), 128, np.uint8)  # 大面积中性灰背景
    img[:, : w // 5] = (200, 30, 30)  # 左 20% 红
    s = np.zeros((h, w), np.float32)
    s[:, : w // 5] = 1.0  # 注意力全押红块
    res = WeightedDominantColor(cfg.color, invert=False, name="focus").compute(img, s)
    assert res.best is not None
    assert res.best.rgb[0] > 150  # 红胜出（v0 打分下灰会靠质量反超）
    assert res.best.contrast > 0.5  # 红块相对全帧均值显著醒目


def test_hysteresis_keeps_incumbent_on_close_scores():
    """Phase 2 动量化迟滞：连胜攒满动量后，挑战簇优势不足 (margin×动量) 时保持在位色。"""
    h = w = 160
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = (200, 30, 30)
    img[:, w // 2 :] = (30, 60, 200)

    s_red = np.zeros((h, w), np.float32)
    s_red[:, : w // 2] = 1.0  # 稳定期：注意力全在红
    # 挑战帧：注意力移向蓝但优势小——带内 ramp 值（0.7→w=0.83、0.85→w=1.0），蓝仅以
    # 约两成质量优势领先（<margin）；若用 0.8/1.0 会双双饱和到 w=1，注意力失去区分度
    s_blueish = np.zeros((h, w), np.float32)
    s_blueish[:, : w // 2] = 0.7
    s_blueish[:, w // 2 :] = 0.85

    keep_cfg = make_cfg(overrides={"color.seed": 0, "color.hysteresis_margin": 0.5})
    switch_cfg = make_cfg(overrides={"color.seed": 0, "color.hysteresis_margin": 0.0})
    for cfg in (keep_cfg, switch_cfg):
        cm = WeightedDominantColor(cfg.color, invert=False, name="focus")
        stable = None
        for _ in range(8):  # 连胜攒满动量（约 5 帧满，多攒几帧留余量）
            stable = cm.compute(img, s_red)
        assert stable.best.rgb[0] > 150  # 稳定期红在位
        second = cm.compute(img, s_blueish)
        # 挑战帧：蓝优势 ~20% < margin×动量（满额）→ 保红；margin=0 → 立即换蓝
        if cfg.color.hysteresis_margin:
            assert second.best.rgb[0] > 150
        else:
            assert second.best.rgb[2] > 150


def _cand(lab, score):
    return ColorCandidate(
        rgb=(0.0, 0.0, 0.0), lab01=lab, score=score,
        mass_frac=0.5, compactness=1.0, chroma=1.0, contrast=1.0,
    )


def test_hysteresis_momentum_builds_and_releases():
    """迟滞动量（2026-11-05 弱化）：保护力要连胜攒出来——动量低时小优势即可夺主，
    攒满后全额保护，换景（大优势）夺主并乘性泄放。合成候选直测 _select 状态机。"""
    cfg = make_cfg(overrides={"color.hysteresis_margin": 0.5})
    cm = WeightedDominantColor(cfg.color, invert=False, name="focus")
    red, blue = (0.5, 0.6, 0.0), (0.5, 0.0, 0.6)  # Lab 距离远超 hysteresis_dist=0.12

    # 帧1（动量 0）：无保护，分数赢家即主色；首帧只建立参考色，动量仍为 0
    assert cm._select([_cand(red, 1.0), _cand(blue, 0.95)]).lab01 == red
    assert cm._mom == pytest.approx(0.0)

    # 帧2：红保持 → 动量攒到 0.2（每帧 +0.2）
    cm._select([_cand(red, 1.0), _cand(blue, 0.95)])
    assert cm._mom == pytest.approx(0.2)

    # 帧3（动量 0.2，有效门槛 0.5×0.2=0.1）：蓝 15% 优势 > 有效门槛（虽 < 满额 margin）→ 蓝夺主
    assert cm._select([_cand(red, 1.0), _cand(blue, 1.15)]).lab01 == blue

    # 蓝连胜攒满动量
    for _ in range(6):
        cm._select([_cand(red, 1.0), _cand(blue, 1.1)])
    assert cm._mom == pytest.approx(1.0)

    # 满动量（有效门槛 0.5）：蓝 10% 分数劣势 → 在位蓝仍被保护；但红（分数赢家）被
    # 门槛压制 → 动量慢泄 ×0.85（否则满动量下小优势挑战永无出头之日，门槛自锁）
    assert cm._select([_cand(red, 1.0), _cand(blue, 0.9)]).lab01 == blue
    assert cm._mom == pytest.approx(0.85)

    # 换景级优势（2 倍）直接夺主，动量乘性泄放（0.85 → 0.51）、参考色跟新赢家
    assert cm._select([_cand(red, 2.0), _cand(blue, 1.0)]).lab01 == red
    assert cm._mom == pytest.approx(0.51)


def test_hysteresis_small_persistent_shift_takes_over():
    """门槛自锁修复（2026-11-06 复盘发现）：满动量在位后，仅 3% 持续分数优势的
    挑战者须在 ~1s 内夺主——旧实现把泄放只挂在"赢家换人"，而门槛恰恰阻止换人，
    小优势永无出头之日（与弱化初衷相反的稳态锁死）。"""
    cfg = make_cfg(overrides={"color.seed": 0})  # margin 0.10
    cm = WeightedDominantColor(cfg.color, invert=False, name="focus")
    red, blue = (0.5, 0.6, 0.0), (0.5, 0.0, 0.6)
    for _ in range(8):  # 红连胜攒满动量
        cm._select([_cand(red, 1.0), _cand(blue, 0.97)])
    assert cm._mom == pytest.approx(1.0)
    took = None
    for t in range(1, 13):  # 蓝 3% 持续优势（> 噪声量级的真实内容变化）
        if cm._select([_cand(red, 1.0), _cand(blue, 1.03)]).lab01 == blue:
            took = t
            break
    assert took is not None and took <= 10  # ~0.5-1s 内完成切换


def test_score_formula_chroma_tilt():
    """彩度系数 (0.45+0.55c)：灰→满彩端点 0.45/1.0（2026-11-05 鲜艳度影响力略调高）。"""
    assert _score(1.0, 0.0, 1.0, 1.0) == pytest.approx(0.45)  # 灰簇底线 ×0.45（旧式 0.55）
    assert _score(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)  # 满彩簇 ×1.0
    # 其余两项保持 (0.6+0.4x) 语义不变
    assert _score(1.0, 1.0, 0.0, 1.0) == pytest.approx(0.6)
    assert _score(1.0, 1.0, 1.0, 0.0) == pytest.approx(0.6)


def test_dominance_is_winner_score_share():
    cfg = make_cfg()
    img = np.full((64, 64, 3), 128, np.uint8)
    res = WeightedDominantColor(cfg.color).compute(img, np.zeros((64, 64), np.float32))
    assert res.best is not None
    assert 0.0 < res.dominance <= 1.0


def test_dominance_multi_cluster_share():
    """多簇时支配度 = 胜者得分占全簇总分比例（非恒 1）。
    注意场景须用带内注意力（0.5 与 1.0）：若另一侧 A=0 会被 ramp 整体剔除，
    世界只剩一族颜色 → 支配度恒 1（那也是正确语义，但测不了多簇分享）。"""
    cfg = make_cfg(overrides={"color.seed": 0})
    img, _ = _split_scene()
    s = np.full((160, 160), 0.5, np.float32)  # 蓝侧带内（w=0.5）
    s[:, :80] = 1.0  # 红侧满权重
    res = WeightedDominantColor(cfg.color, invert=False, name="focus").compute(img, s)
    assert res.best is not None and len(res.candidates) >= 2
    total = sum(c.score for c in res.candidates)
    assert 0.0 < res.dominance < 1.0
    assert abs(res.dominance - res.best.score / total) < 1e-6


def test_background_below_lo_excluded():
    """ramp 加权回归（2026-11-05 背景主导修复）：主体仅占 10% 画面但注意力高，
    大面积背景注意力 0.1 < weight_lo → 权重记 0、抽样前剔除，焦点色必须是主体色。
    旧式 w = 1+λ·A 下背景保底权重 1（λ=3 时 w=1.3），0.9×1.3 > 0.1×3.85，
    背景簇靠面积反超——权重压不到 0 就排不掉背景。"""
    cfg = make_cfg(overrides={"color.seed": 0})
    h = w = 160
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : 16] = (200, 30, 30)  # 左 10% 红（主体）
    img[:, 16:] = (60, 90, 170)  # 右 90% 蓝灰（背景）
    s = np.full((h, w), 0.1, np.float32)
    s[:, : 16] = 0.95  # 注意力高置信落在主体

    focus = WeightedDominantColor(cfg.color, invert=False, name="focus").compute(img, s)
    assert focus.best is not None
    assert focus.best.rgb[0] > 150 and focus.best.rgb[2] < 110  # 焦点=红

    ambient = WeightedDominantColor(cfg.color, invert=True, name="ambient").compute(img, s)
    assert ambient.best is not None
    assert ambient.best.rgb[2] > 130  # 氛围=大面积蓝灰（主体侧 1−A=0.05 < lo 已剔除）


def test_nearby_shades_merge_into_one():
    """动态类数（2026-11-06 实测反馈"类数偏多"）：同色系被 K-Means 拆出的深浅碎片
    按 merge_dist 并族——纯红渐变世界只出一个红候选，且任意两候选距离 >= merge_dist
    （族划分的不变式）。比静态砍 K 优雅：真多彩场景互距远、不丢分辨率。"""
    cfg = make_cfg(overrides={"color.seed": 0})
    h = w = 160
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :40] = (200, 30, 30)  # 深红
    img[:, 40:80] = (212, 46, 40)  # 浅红（同色系，加权 Lab 距离 << merge_dist）
    img[:, 80:] = (60, 90, 170)  # 背景蓝灰（注意力 < lo，已被 ramp 剔除）
    s = np.full((h, w), 0.05, np.float32)
    s[:, :80] = 1.0  # 焦点世界 = 两种红
    res = WeightedDominantColor(cfg.color, invert=False, name="focus").compute(img, s)
    assert res.best is not None
    assert res.best.rgb[0] > 170 and res.best.rgb[2] < 110  # 合并后仍是红
    assert len(res.candidates) == 1  # 两种红并成一族
    labs = [np.asarray(c.lab01) for c in res.candidates]
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            assert np.linalg.norm(labs[i] - labs[j]) >= cfg.color.merge_dist


def test_sample_world_adaptive_budget():
    """抽样预算自适应（2026-11-06）：掩膜全分辨率先定，抽样只发生在本世界像素上——
    小主体世界全取（旧全帧等步长网格按全帧面积定步长，1080p 下 5% 主体只得 ~600
    样本，其余预算漏给零权重背景格点）；超大世界按本世界面积定步长交掩膜，样本
    ≤ 预算且全在世界内；几乎无合格像素退化为全帧无权重。"""
    cfg = make_cfg()
    cm = WeightedDominantColor(cfg.color, invert=False, name="focus")
    h = w = 480  # n_all = 230400，预算 12000
    rgb = np.full((h, w, 3), 128, np.uint8)

    # 小主体（44² = 1936 px ≤ 预算）：全取，满注意力像素权重 = 1
    s = np.zeros((h, w), np.float32)
    s[:44, :44] = 1.0
    in_world = s > cfg.color.weight_lo
    x_rgb, ws = cm._sample_world(rgb, s, in_world, int(in_world.sum()), step=4)
    assert x_rgb.shape[0] == 1936 and np.all(ws == 1.0)

    # 全帧世界（超预算）：步长按 n_world 定（ceil 保证 ≤ 预算），全在世界内
    s_full = np.ones((h, w), np.float32)
    x_rgb2, ws2 = cm._sample_world(rgb, s_full, np.ones((h, w), bool), h * w, step=4)
    assert x_rgb2.shape[0] <= cfg.color.sample_budget
    assert x_rgb2.shape[0] > cfg.color.sample_budget // 4  # 步长取整不至于欠采过头
    assert np.all(ws2 > 0)

    # 本世界几乎无合格像素：全帧等步长、无权重（退化回退，输出不断流）
    x_rgb3, ws3 = cm._sample_world(
        rgb, np.zeros((h, w), np.float32), np.zeros((h, w), bool), 0, step=4
    )
    assert x_rgb3.shape[0] == 120 * 120 and np.all(ws3 == 1.0)


def test_surround_contrast_camouflage_penalty():
    """保护色惩罚（2026-11-06 center-surround 对齐）：焦点世界内两个等质量等彩度的
    候选，与背景主色同色者（保护色）对比度趋 0、输给异色者。Itti & Koch 2001 的
    surround 即背景；缺省（无参照）退回全帧均色，此时绿候选对比度不再为 0。"""
    cfg = make_cfg(overrides={"color.seed": 0})
    h = w = 160
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (30, 170, 60)  # 背景：绿（氛围世界主色）
    img[:, 40:80] = (235, 145, 30)  # 焦点世界右半：橙（与背景异色）
    s = np.full((h, w), 0.05, np.float32)
    s[:, :80] = 1.0  # 焦点世界 = 绿(保护色) + 橙，等面积等权重

    ambient = WeightedDominantColor(cfg.color, invert=True, name="ambient").compute(img, s)
    assert ambient.best is not None and ambient.best.rgb[1] > 120  # 氛围主色 = 绿

    focus = WeightedDominantColor(cfg.color, invert=False, name="focus").compute(
        img, s, surround_ref=ambient.best.lab01
    )
    green = next(c for c in focus.candidates if c.rgb[0] < 100)
    orange = next(c for c in focus.candidates if c.rgb[0] > 180)
    assert orange.contrast > green.contrast + 0.3  # 异色远比保护色"醒目"
    assert focus.best is orange  # 保护色候选被压，橙胜出（质量/彩度本完全打平）

    fallback = WeightedDominantColor(cfg.color, invert=False, name="focus").compute(img, s)
    green_fb = next(c for c in fallback.candidates if c.rgb[0] < 100)
    assert green_fb.contrast > 0.2  # 无参照时退回全帧均色，绿不再被判为"不醒目"


def test_warm_start_reuses_centers_and_stays_correct():
    """warm-start（防闪 A）：第二轮复用上轮簇心，正确性不变。"""
    cfg = make_cfg(overrides={"color.seed": 0})
    img, s = _split_scene()
    cm = WeightedDominantColor(cfg.color, invert=False, name="focus")
    first = cm.compute(img, s)
    assert first.best is not None
    assert cm._prev_centers is not None  # 已存簇心
    second = cm.compute(img, s)
    assert second.best is not None
    assert second.best.rgb[0] > 150  # 红仍胜出


def test_warm_start_survives_scene_change():
    """warm-start 不能锁死旧簇：换素材后赢家应跟新内容走。"""
    cfg = make_cfg(overrides={"color.seed": 0})
    h = w = 160
    img_red = np.zeros((h, w, 3), np.uint8)
    img_red[:] = (200, 30, 30)
    s = np.ones((h, w), np.float32)
    img_blue = np.zeros((h, w, 3), np.uint8)
    img_blue[:] = (30, 60, 200)

    cm = WeightedDominantColor(cfg.color, invert=False, name="focus")
    assert cm.compute(img_red, s).best.rgb[0] > 150
    res = cm.compute(img_blue, s)
    assert res.best is not None
    assert res.best.rgb[2] > 150  # 全画面换蓝后旧簇心不得绑架结果
