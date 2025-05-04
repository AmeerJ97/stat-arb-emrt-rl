import numpy as np


def test_clean_subpackages_reexport_legacy_implementation():
    from stat_arb_emrt_rl.backtesting import RLStatArbBacktest
    from stat_arb_emrt_rl.core import compute_emrt
    from stat_arb_emrt_rl.data import FinancialLoaderProvider
    from stat_arb_emrt_rl.discovery import MultiCointConfig, MultiCointEngine
    from stat_arb_emrt_rl.rl import TabularQAgent

    emrt, taus = compute_emrt(np.sin(np.linspace(0, 12, 200)), C=0.25)

    assert np.isfinite(emrt)
    assert taus
    assert FinancialLoaderProvider.__name__ == "FinancialLoaderProvider"
    assert MultiCointConfig.__name__ == "MultiCointConfig"
    assert MultiCointEngine.__name__ == "MultiCointEngine"
    assert RLStatArbBacktest.__name__ == "RLStatArbBacktest"
    assert TabularQAgent.__name__ == "TabularQAgent"
