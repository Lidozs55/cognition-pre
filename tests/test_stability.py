from cognition_pre.stability import ColorStabilizer


def test_first_value_passthrough():
    st = ColorStabilizer(ema_alpha=0.5)
    rgb, conf = st.update((255, 0, 0), 0.8)
    assert rgb == (255.0, 0.0, 0.0)
    assert conf == 0.8


def test_ema_converges():
    st = ColorStabilizer(ema_alpha=0.5, conf_ema_alpha=0.5, jump_reset_dist=0.0)
    st.update((0, 0, 0), 0.0)  # 先落一个非目标初值，避免首值直通造成的永真
    rgb = (0.0, 0.0, 0.0)
    conf = 0.0
    for _ in range(20):
        rgb, conf = st.update((255, 255, 255), 1.0)
    assert rgb[0] > 250
    assert conf > 0.95


def test_big_jump_resets_instead_of_crawling():
    """场景切换的大跳变直接置入，不"爬"过中间色相（Phase 2 规格）。"""
    st = ColorStabilizer(ema_alpha=0.3, conf_ema_alpha=0.3, jump_reset_dist=60.0)
    st.update((255, 0, 0), 1.0)
    rgb, _ = st.update((0, 0, 255), 1.0)  # 距离 360 >> 60 → 重置
    assert rgb[2] == 255.0 and rgb[0] == 0.0


def test_smoothing_damps_small_jumps():
    """阈值内的小幅波动仍走 EMA 部分吸收。"""
    st = ColorStabilizer(ema_alpha=0.3, conf_ema_alpha=0.3, jump_reset_dist=60.0)
    st.update((255, 0, 0), 1.0)
    rgb, _ = st.update((200, 0, 0), 1.0)  # 距离 55 < 60 → EMA
    assert 230 < rgb[0] < 250


def test_jump_reset_can_be_disabled():
    st = ColorStabilizer(ema_alpha=0.3, conf_ema_alpha=0.3, jump_reset_dist=0.0)
    st.update((255, 0, 0), 1.0)
    rgb, _ = st.update((0, 0, 255), 1.0)  # 关闭重置 → 旧 EMA 行为
    assert 40 < rgb[2] < 90
