import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from stat_arb_emrt_rl.data_provider import InMemoryProvider
from stat_arb_emrt_rl.rl_backtest import RLStatArbBacktest


class TestRLBacktestSmoke(unittest.TestCase):
    @patch("stat_arb_emrt_rl.rl_backtest.train_agent")
    @patch("stat_arb_emrt_rl.rl_backtest.run_rl_trading")
    @patch("stat_arb_emrt_rl.rl_backtest.optimize_spread_coefficients")
    @patch("stat_arb_emrt_rl.rl_backtest.OUOptimizer.optimize")
    def test_run_pair_offline_with_provider(
        self,
        mock_optimize,
        mock_optimize_spread,
        mock_run_rl_trading,
        mock_train_agent,
    ):
        dates_form = pd.bdate_range("2022-01-01", periods=80)
        dates_trade = pd.bdate_range("2023-01-01", periods=80)

        form_df = pd.DataFrame(
            {
                "Normalized MSFT": np.linspace(1.0, 1.1, len(dates_form)),
                "Normalized GOOGL": np.linspace(1.0, 1.08, len(dates_form)),
            },
            index=dates_form,
        )
        trade_df = pd.DataFrame(
            {
                "Normalized MSFT": np.linspace(1.0, 1.06, len(dates_trade)),
                "Normalized GOOGL": np.linspace(1.0, 1.03, len(dates_trade)),
            },
            index=dates_trade,
        )

        provider = InMemoryProvider(
            stock_data={},
            pair_data={
                "MSFT|GOOGL|2022-01-01|2022-12-31": form_df,
                "MSFT|GOOGL|2023-01-01|2023-12-31": trade_df,
            },
            universe=["MSFT", "GOOGL"],
        )

        mock_optimize.return_value = {
            "beta": 1.0,
            "mu": 0.8,
            "theta": 0.0,
            "sigma": 0.2,
            "log_likelihood": 1.0,
        }
        mock_optimize_spread.return_value = {
            "coefficients": {"S1": 1.0, "S2": 1.0},
            "emrt": 5.0,
            "spread": (form_df["Normalized MSFT"] - form_df["Normalized GOOGL"]).values,
        }
        mock_train_agent.return_value = object()
        mock_run_rl_trading.return_value = {
            "equity_curve": np.linspace(100.0, 105.0, len(dates_trade)),
            "total_trades": 3,
            "win_rate": 0.66,
            "daily_return_mean": 0.001,
            "daily_return_std": 0.01,
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.05,
            "total_return_pct": 5.0,
        }

        bt = RLStatArbBacktest(
            formation_start="2022-01-01",
            formation_end="2022-12-31",
            trading_start="2023-01-01",
            trading_end="2023-12-31",
            data_provider=provider,
        )
        result = bt.run_pair("MSFT", "GOOGL", "Technology")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["pair"], "MSFT-GOOGL")
        self.assertIn("DM", result)
        self.assertIn("OU", result)
        self.assertIn("RL", result)


if __name__ == "__main__":
    unittest.main()
