import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from stat_arb_emrt_rl.financial_loader import FinancialLoader, FinancialLoaderConfig


class TestFinancialLoader(unittest.TestCase):
    def setUp(self):
        self.config = FinancialLoaderConfig()
        self.config.retry_attempts = 1
        self.config.rate_limit_delay = 0.0
        self.loader = FinancialLoader(config=self.config)

    @patch("yfinance.download")
    def test_get_stock_data_success(self, mock_download):
        """Test successful stock data retrieval"""
        test_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100],
                "Volume": [1e6],
            },
            index=[pd.Timestamp("2023-01-01")],
        )
        mock_download.return_value = test_data

        result = self.loader.get_stock_data("AAPL", "2023-01-01", "2023-01-02")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)

    @patch("yfinance.download")
    def test_cache_functionality(self, mock_download):
        """Test caching mechanism with TTL"""
        test_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100],
                "Volume": [1e6],
            },
            index=[pd.Timestamp("2023-01-01")],
        )
        mock_download.return_value = test_data

        # First call (cache miss)
        first = self.loader.get_stock_data("MSFT", "2023-01-01", "2023-01-02")
        self.assertEqual(self.loader.get_cache_metrics()["misses"], 1)
        self.assertIsNotNone(first)

        # Second call (cache hit)
        second = self.loader.get_stock_data("MSFT", "2023-01-01", "2023-01-02")
        self.assertEqual(self.loader.get_cache_metrics()["hits"], 1)
        self.assertIsNotNone(second)
        self.assertEqual(mock_download.call_count, 1)

    # @patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "test_key"})
    # @patch("stat_arb_emrt_rl.financial_loader.TimeSeries")  # Corrected patch target
    # @patch("stat_arb_emrt_rl.financial_loader.FinancialLoader._try_fallback_source")
    # @patch("yfinance.download")
    # def test_fallback_mechanism(self, mock_download, mock_fallback, mock_timeseries):
    #     mock_download.side_effect = Exception("YFinance failed")
    #     mock_fallback.return_value = pd.DataFrame(
    #         {"Open": [100], "High": [101], "Low": [99], "Close": [100], "Volume": [1e6]}
    #     )
    #     # Setup mock TimeSeries instance
    #     mock_ts_instance = MagicMock()
    #     mock_ts_instance.get_daily_adjusted.return_value = (pd.DataFrame(), None)
    #     mock_timeseries.return_value = mock_ts_instance

    #     self.loader.config.fallback_enabled = True
    #     self.loader._init_fallback_source()  # Re-initialize with mocked dependency
    #     result = self.loader.get_stock_data("GOOG", "2023-01-01", "2023-01-02")
    #     self.assertIsNotNone(result)
    #     self.assertEqual(len(result), 1)

    @patch("time.sleep")
    @patch("yfinance.download")
    def test_rate_limiting(self, mock_download, mock_sleep):
        """Test API rate limiting"""
        test_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100],
                "Volume": [1e6],
            },
            index=[pd.Timestamp("2023-01-01")],
        )
        mock_download.return_value = test_data

        self.loader.get_stock_data("AAPL", "2023-01-01", "2023-01-02")
        self.loader.get_stock_data("MSFT", "2023-01-01", "2023-01-02")
        self.assertEqual(mock_sleep.call_count, 0)

    @patch("yfinance.download")
    def test_primary_source_exception_returns_none(self, mock_download):
        mock_download.side_effect = Exception("API Error")
        result = self.loader.get_stock_data("FAIL", "2023-01-01", "2023-01-02")
        self.assertIsNone(result)

    def test_cache_eviction(self):
        """Test cache size management"""
        self.loader.config.max_tickers = 2
        dates = ["2023-01-01", "2023-01-02", "2023-01-03"]

        for i, date in enumerate(dates):
            self.loader._update_cache(f"key{i}", pd.DataFrame())

        self.assertEqual(len(self.loader.cache), 2)
        self.assertNotIn("key0", self.loader.cache)

    def test_mixed_pair_uses_business_calendar_in_auto_mode(self):
        business_idx = pd.bdate_range("2024-01-01", "2024-01-15")
        daily_idx = pd.date_range("2024-01-01", "2024-01-15", freq="D")

        equity_df = pd.DataFrame(
            {
                "Open": np.linspace(100, 110, len(business_idx)),
                "High": np.linspace(101, 111, len(business_idx)),
                "Low": np.linspace(99, 109, len(business_idx)),
                "Close": np.linspace(100, 110, len(business_idx)),
                "Volume": 1e6,
            },
            index=business_idx,
        )
        crypto_df = pd.DataFrame(
            {
                "Open": np.linspace(40, 45, len(daily_idx)),
                "High": np.linspace(41, 46, len(daily_idx)),
                "Low": np.linspace(39, 44, len(daily_idx)),
                "Close": np.linspace(40, 45, len(daily_idx)),
                "Volume": 2e6,
            },
            index=daily_idx,
        )

        with patch.object(
            self.loader,
            "get_stock_data",
            side_effect=[equity_df, crypto_df],
        ):
            result = self.loader.get_normalized_pair(
                "AAPL",
                "BTC-USD",
                "2024-01-01",
                "2024-01-15",
                calendar_mode="auto",
            )

        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertTrue((result.index.dayofweek < 5).all())

    def test_get_mixed_asset_history_returns_aligned_panel(self):
        business_idx = pd.bdate_range("2024-02-01", "2024-03-15")
        daily_idx = pd.date_range("2024-02-01", "2024-03-15", freq="D")

        equity_df = pd.DataFrame(
            {
                "Open": np.linspace(50, 55, len(business_idx)),
                "High": np.linspace(51, 56, len(business_idx)),
                "Low": np.linspace(49, 54, len(business_idx)),
                "Close": np.linspace(50, 55, len(business_idx)),
                "Volume": 5e5,
            },
            index=business_idx,
        )
        crypto_df = pd.DataFrame(
            {
                "Open": np.linspace(200, 220, len(daily_idx)),
                "High": np.linspace(201, 221, len(daily_idx)),
                "Low": np.linspace(199, 219, len(daily_idx)),
                "Close": np.linspace(200, 220, len(daily_idx)),
                "Volume": 7e5,
            },
            index=daily_idx,
        )

        with patch.object(
            self.loader,
            "get_stock_data",
            side_effect=[equity_df, crypto_df],
        ):
            panel = self.loader.get_mixed_asset_history(
                ["MSFT", "BTC-USD"],
                "2024-02-01",
                "2024-03-15",
                calendar_mode="business",
                normalize=True,
            )

        self.assertIsNotNone(panel)
        self.assertFalse(panel.empty)
        self.assertListEqual(list(panel.columns), ["MSFT", "BTC-USD"])
        self.assertTrue(np.allclose(panel.iloc[0].values, np.ones(2)))


if __name__ == "__main__":
    unittest.main()
