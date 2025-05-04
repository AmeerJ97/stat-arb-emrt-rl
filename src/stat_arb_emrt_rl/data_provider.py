from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import pandas as pd

from .financial_loader import FinancialLoader


class MarketDataProvider(Protocol):
    """Minimal provider interface for strategy and research pipelines."""

    def get_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        period: Optional[str] = None,
        reset_cache: bool = False,
    ) -> Optional[pd.DataFrame]:
        ...

    def get_normalized_pair(
        self,
        t1: str,
        t2: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        period: Optional[str] = None,
        calendar_mode: str = "auto",
    ) -> Optional[pd.DataFrame]:
        ...

    def get_sp500_tickers(self, refresh: bool = False) -> List[str]:
        ...


@dataclass
class FinancialLoaderProvider:
    """
    Adapter that exposes FinancialLoader through the MarketDataProvider interface.
    """

    loader: FinancialLoader

    def get_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        period: Optional[str] = None,
        reset_cache: bool = False,
    ) -> Optional[pd.DataFrame]:
        return self.loader.get_stock_data(
            ticker,
            start_date,
            end_date,
            interval=interval,
            period=period,
            reset_cache=reset_cache,
        )

    def get_normalized_pair(
        self,
        t1: str,
        t2: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        period: Optional[str] = None,
        calendar_mode: str = "auto",
    ) -> Optional[pd.DataFrame]:
        return self.loader.get_normalized_pair(
            t1,
            t2,
            start_date,
            end_date,
            interval=interval,
            period=period,
            calendar_mode=calendar_mode,
        )

    def get_sp500_tickers(self, refresh: bool = False) -> List[str]:
        return self.loader.get_sp500_tickers(refresh=refresh)


@dataclass
class InMemoryProvider:
    """
    Deterministic provider for tests and offline experiments.
    """

    stock_data: Dict[str, pd.DataFrame]
    pair_data: Dict[str, pd.DataFrame]
    universe: List[str]

    def _pair_key(self, t1: str, t2: str, start_date: str, end_date: str) -> str:
        return f"{t1}|{t2}|{start_date}|{end_date}"

    def get_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        period: Optional[str] = None,
        reset_cache: bool = False,
    ) -> Optional[pd.DataFrame]:
        _ = (start_date, end_date, interval, period, reset_cache)
        data = self.stock_data.get(ticker)
        return data.copy() if data is not None else None

    def get_normalized_pair(
        self,
        t1: str,
        t2: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        period: Optional[str] = None,
        calendar_mode: str = "auto",
    ) -> Optional[pd.DataFrame]:
        _ = (interval, period, calendar_mode)
        data = self.pair_data.get(self._pair_key(t1, t2, start_date, end_date))
        return data.copy() if data is not None else None

    def get_sp500_tickers(self, refresh: bool = False) -> List[str]:
        _ = refresh
        return list(self.universe)
