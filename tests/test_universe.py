import unittest
from unittest.mock import patch

import pandas as pd

from stat_arb_emrt_rl.financial_loader import FinancialLoader


class TestFinancialLoaderUniverse(unittest.TestCase):
    @patch("stat_arb_emrt_rl.financial_loader.requests.get")
    @patch("stat_arb_emrt_rl.financial_loader.pd.read_html")
    def test_get_sp500_tickers_from_wikipedia(self, mock_read_html, mock_get):
        loader = FinancialLoader()
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "<html></html>"
        mock_get.return_value.raise_for_status.return_value = None
        mock_read_html.return_value = [
            pd.DataFrame({"Symbol": ["AAPL", "MSFT", "GOOGL"]}),
        ]

        tickers = loader.get_sp500_tickers(refresh=True)
        self.assertEqual(tickers, ["AAPL", "MSFT", "GOOGL"])

    @patch("stat_arb_emrt_rl.financial_loader.requests.get")
    def test_get_sp500_tickers_fallback_on_failure(self, mock_get):
        loader = FinancialLoader()
        mock_get.side_effect = RuntimeError("network down")
        tickers = loader.get_sp500_tickers(refresh=True)
        self.assertGreater(len(tickers), 10)


if __name__ == "__main__":
    unittest.main()
