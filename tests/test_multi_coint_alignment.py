import unittest

import pandas as pd

from stat_arb_emrt_rl.data_provider import InMemoryProvider

try:
    from stat_arb_emrt_rl.multi_coint import MultiCointConfig, MultiCointEngine
    _HAS_MULTI_COINT_DEPS = True
except Exception:
    _HAS_MULTI_COINT_DEPS = False


@unittest.skipUnless(_HAS_MULTI_COINT_DEPS, "multi_coint optional dependencies are not installed")
class TestMultiCointAlignment(unittest.TestCase):
    def test_align_series_mixed_asset_uses_business_calendar(self):
        equity_idx = pd.bdate_range("2024-01-01", "2024-01-20")
        crypto_idx = pd.date_range("2024-01-01", "2024-01-20", freq="D")

        cfg = MultiCointConfig()
        cfg.min_data_points = 5
        provider = InMemoryProvider(stock_data={}, pair_data={}, universe=[])
        engine = MultiCointEngine(
            start_date="2024-01-01",
            end_date="2024-01-20",
            config=cfg,
            data_provider=provider,
        )
        engine.price_data = {
            "AAPL": pd.Series(range(len(equity_idx)), index=equity_idx, dtype=float),
            "BTC-USD": pd.Series(range(len(crypto_idx)), index=crypto_idx, dtype=float),
        }

        aligned = engine._align_series(["AAPL", "BTC-USD"])
        self.assertIsNotNone(aligned)
        assert aligned is not None
        self.assertFalse(aligned.empty)
        self.assertTrue((aligned.index.dayofweek < 5).all())


if __name__ == "__main__":
    unittest.main()
