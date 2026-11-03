import numpy as np
from conftest import make_cfg

from cognition_pre.color.dominant import WeightedDominantColor


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
    """Phase 2 迟滞：挑战簇优势不足 margin 时保持上帧主色。"""
    h = w = 160
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = (200, 30, 30)
    img[:, w // 2 :] = (30, 60, 200)

    s_red = np.zeros((h, w), np.float32)
    s_red[:, : w // 2] = 1.0  # 帧1：注意力全在红
    s_blueish = np.zeros((h, w), np.float32)
    s_blueish[:, : w // 2] = 0.8  # 帧2：注意力移向蓝（蓝侧 1.0），蓝仅以 ~9% 分数优势领先（<margin）
    s_blueish[:, w // 2 :] = 1.0

    keep_cfg = make_cfg(overrides={"color.seed": 0, "color.hysteresis_margin": 0.5})
    switch_cfg = make_cfg(overrides={"color.seed": 0, "color.hysteresis_margin": 0.0})
    for cfg in (keep_cfg, switch_cfg):
        cm = WeightedDominantColor(cfg.color, invert=False, name="focus")
        first = cm.compute(img, s_red)
        assert first.best.rgb[0] > 150  # 帧1 红
        second = cm.compute(img, s_blueish)
        # 帧2 蓝的优势 < margin → 迟滞保红；margin=0 → 立即换蓝
        if cfg.color.hysteresis_margin:
            assert second.best.rgb[0] > 150
        else:
            assert second.best.rgb[2] > 150


def test_dominance_is_winner_score_share():
    cfg = make_cfg()
    img = np.full((64, 64, 3), 128, np.uint8)
    res = WeightedDominantColor(cfg.color).compute(img, np.zeros((64, 64), np.float32))
    assert res.best is not None
    assert 0.0 < res.dominance <= 1.0


def test_dominance_multi_cluster_share():
    """多簇时支配度 = 胜者得分占全簇总分比例（非恒 1）。"""
    cfg = make_cfg(overrides={"color.seed": 0})
    img, s = _split_scene()
    res = WeightedDominantColor(cfg.color, invert=False, name="focus").compute(img, s)
    assert res.best is not None
    total = sum(c.score for c in res.candidates)
    assert 0.0 < res.dominance < 1.0
    assert abs(res.dominance - res.best.score / total) < 1e-6
