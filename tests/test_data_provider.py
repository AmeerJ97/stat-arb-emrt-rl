import unittest
from unittest.mock import MagicMock

import pandas as pd

from stat_arb_emrt_rl.data_provider import FinancialLoaderProvider, InMemoryProvider


class TestFinancialLoaderProvider(unittest.TestCase):
    def test_adapter_delegates_to_loader(self):
        loader = MagicMock()
        expected = pd.DataFrame({"Close": [1.0, 2.0]})
        loader.get_stock_data.return_value = expected
        loader.get_normalized_pair.return_value = expected
        loader.get_sp500_tickers.return_value = ["AAPL", "MSFT"]

        provider = FinancialLoaderProvider(loader=loader)

        result_stock = provider.get_stock_data("AAPL", "2024-01-01", "2024-01-05")
        result_pair = provider.get_normalized_pair("AAPL", "MSFT", "2024-01-01", "2024-01-05")
        result_universe = provider.get_sp500_tickers()

        self.assertTrue(result_stock.equals(expected))
        self.assertTrue(result_pair.equals(expected))
        self.assertEqual(result_universe, ["AAPL", "MSFT"])


class TestInMemoryProvider(unittest.TestCase):
    def test_in_memory_provider_returns_copies(self):
        stock_df = pd.DataFrame({"Close": [100.0, 101.0]})
        pair_df = pd.DataFrame({"Normalized AAPL": [1.0], "Normalized MSFT": [1.0]})
        provider = InMemoryProvider(
            stock_data={"AAPL": stock_df},
            pair_data={"AAPL|MSFT|2024-01-01|2024-01-05": pair_df},
            universe=["AAPL", "MSFT"],
        )

        returned_stock = provider.get_stock_data("AAPL", "2024-01-01", "2024-01-05")
        returned_pair = provider.get_normalized_pair("AAPL", "MSFT", "2024-01-01", "2024-01-05")

        self.assertIsNotNone(returned_stock)
        self.assertIsNotNone(returned_pair)
        self.assertEqual(provider.get_sp500_tickers(), ["AAPL", "MSFT"])

        returned_stock.iloc[0, 0] = -1.0
        self.assertNotEqual(stock_df.iloc[0, 0], returned_stock.iloc[0, 0])


if __name__ == "__main__":
    unittest.main()
