import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


def _plotting_dependencies_available() -> bool:
    try:
        import backtrader  # noqa: F401
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(
    _plotting_dependencies_available(),
    "plotting dependencies are not installed",
)
def test_plots():
    from stat_arb_emrt_rl.backtest.plotting import plot_pair_spreads

    dates = pd.bdate_range("2024-01-01", periods=5)
    pair_name = "AAA/BBB"
    pair = ("AAA", "BBB")

    class Analyzer:
        def __init__(self, data):
            self._data = data

        def get_analysis(self):
            return self._data

    captured = {}

    def fake_navigator(**kwargs):
        captured.update(kwargs)

    strategy = SimpleNamespace(
        analyzers=SimpleNamespace(
            spread_tracker=Analyzer(
                {
                    "spread_data": {pair_name: [0.2, 0.1, -0.1, 0.0, 0.2]},
                    "dates": {pair_name: list(dates)},
                    "means": {pair_name: [0.0] * len(dates)},
                    "std_devs": {pair_name: [0.2] * len(dates)},
                    "k_values": {pair_name: [1.5] * len(dates)},
                }
            ),
            returns=Analyzer({"returns": [0.01, -0.005, 0.002, 0.0, 0.003]}),
        ),
        trade_recorder=SimpleNamespace(get_trades=lambda: [], trade_pairs=[]),
        price_history={
            pair: {
                "dates": list(dates),
                "s1": [100, 101, 102, 103, 104],
                "s2": [50, 51, 50, 52, 53],
            }
        },
        start_date=dates[0].to_pydatetime(),
        params=SimpleNamespace(pairs=[pair], K=1.5, cointegration_lookback=90),
        hedge_ratios={pair: 1.0},
    )

    normalized = pd.DataFrame(
        {
            "Normalized AAA": [1.0, 1.01, 1.02, 1.03, 1.04],
            "Normalized BBB": [1.0, 1.02, 1.01, 1.04, 1.05],
        },
        index=dates,
    )

    with patch("stat_arb_emrt_rl.backtest.gui.PairNavigator", side_effect=fake_navigator):
        with patch("stat_arb_emrt_rl.backtest.plotting.FinancialLoader") as loader_cls:
            loader_cls.return_value.get_normalized_pair.return_value = normalized
            plot_pair_spreads(
                results=[strategy],
                pairs_to_plot=[pair_name],
                end_date=dates[-1].strftime("%Y-%m-%d"),
            )

    assert captured["pairs_data"][0]["pair_name"] == pair_name


if __name__ == "__main__":
    test_plots()
