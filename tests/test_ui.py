import numpy as np


def test_ui_helpers_provide_non_network_demo_data():
    from stat_arb_emrt_rl.ui.app import demo_emrt_curve, paper_pairs_frame

    pairs = paper_pairs_frame()
    curve = demo_emrt_curve(seed=7, n_steps=120)

    assert {"ticker_1", "ticker_2", "sector"}.issubset(pairs.columns)
    assert len(pairs) >= 10
    assert {"mean_reversion_speed", "emrt"}.issubset(curve.columns)
    assert np.isfinite(curve["emrt"]).any()
