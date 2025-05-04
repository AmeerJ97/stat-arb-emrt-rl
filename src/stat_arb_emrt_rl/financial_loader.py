# ───────────────────────────────────────────────
# Standard Library Imports
# ───────────────────────────────────────────────
import os
import time
import pprint
import requests
from datetime import date, datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# ───────────────────────────────────────────────
# Third-Party Package Imports
# ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm
from dotenv import load_dotenv

# ───────────────────────────────────────────────
# Project-Specific Imports
# ───────────────────────────────────────────────
# from utilities_wrapper import _print_header
from .printing_system import (
    buffered_print,
    print_header,
    print_status,
    GREEN,
    RED,
    YELLOW,
    CYAN,
    PURPLE,
    BLUE,
    ENDC,
)


# ───────────────────────────────────────────────
# Load Environment Variables
# ───────────────────────────────────────────────
load_dotenv()


class FinancialLoaderConfig:
    """Configuration class for loader parameters"""

    def __init__(self):
        self.worker_count = int(os.getenv("FIN_LOADER_WORKERS", 8))
        self.retry_attempts = int(os.getenv("FIN_LOADER_RETRIES", 3))
        self.cache_ttl = int(os.getenv("FIN_LOADER_CACHE_TTL", 3600))  # 1 hour
        self.rate_limit_delay = float(os.getenv("FIN_LOADER_RATE_DELAY", 0.1))
        self.fallback_enabled = bool(os.getenv("FIN_LOADER_FALLBACK", False))
        self.max_tickers = int(os.getenv("FIN_LOADER_MAX_TICKERS", 500))


def retry_api_call(func):
    """Decorator for API call retry logic with exponential backoff"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        last_exception = None
        for attempt in range(self.config.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                delay = 2**attempt + self.config.rate_limit_delay
                buffered_print(
                    f"Retrying in {delay}s (Attempt {attempt+1})", "WARN")
                time.sleep(delay)
        buffered_print(f"Max retries exceeded: {str(last_exception)}", "ERROR")
        return None

    return wrapper


class FinancialLoader:
    """Enterprise-grade financial data loader with advanced features"""

    def __init__(self, config: FinancialLoaderConfig = None):
        print_header("Financial Loader Initialized")
        self.config = config or FinancialLoaderConfig()
        self.cache = {}
        self.required_columns = ["Open", "High", "Low", "Close", "Volume"]
        self._cache_hits = 0
        self._cache_misses = 0
        # self._init_fallback_source()
        self._last_api_call = datetime.min

    def _print_status(self, message: str, status: str = "INFO"):
        """Standardized status messages with colors"""
        colors = {
            "INFO": "\033[94m",  # Blue
            "SUCCESS": "\033[92m",  # Green
            "WARN": "\033[93m",  # Yellow
            "ERROR": "\033[91m",  # Red
            "CACHE": "\033[95m",  # Purple
            "CONFIG": "\033[96m",  # Cyan
        }
        reset = "\033[0m"
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = colors.get(status.upper(), "\033[0m")
        buffered_print(
            f"{color}[{timestamp}] [{status.ljust(7)}] {message}{reset}")

    def _get_cache_key(
        self, ticker: str, start_date: str, end_date: str, interval: str
    ) -> str:
        """Generate consistent cache key with validation"""
        return f"{ticker}|{start_date}|{end_date}|{interval}"

    def _validate_dates(self, start_date: str, end_date: str) -> Tuple[str, str]:
        """Validate and auto-correct date formats and ranges"""
        try:
            start = datetime.strptime(str(start_date), "%Y-%m-%d")
            end = datetime.strptime(str(end_date), "%Y-%m-%d")

            if start > datetime.now():
                buffered_print(
                    "Future start date detected, using today", "WARN")
                start_date = datetime.now().strftime("%Y-%m-%d")

            if start > end:
                buffered_print("Auto-correcting reversed date range", "WARN")
                return end_date, start_date

            return start_date, end_date
        except ValueError as e:
            raise ValueError(f"Invalid date format: {str(e)}. Use YYYY-MM-DD")

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            age = datetime.now() - entry["timestamp"]
            return age.total_seconds() < self.config.cache_ttl
        return False

    # def _update_cache(self, cache_key: str, data: pd.DataFrame):
    #     """Update cache with timestamp and size management"""
    #     # Enforce max cache size
    #     if len(self.cache) >= self.config.max_tickers:
    #         oldest_key = min(
    #             self.cache.keys(), key=lambda k: self.cache[k]["timestamp"]
    #         )
    #         del self.cache[oldest_key]

    #     self.cache[cache_key] = {"data": data, "timestamp": datetime.now()}
    def _update_cache(self, cache_key: str, data: pd.DataFrame):
        """Update cache with timestamp and size management"""
        # Ensure cache is a dictionary (defensive programming)
        if not isinstance(self.cache, dict):
            self.cache = {}
            buffered_print("Reset cache to dictionary", "WARN")

        # Enforce max cache size
        if len(self.cache) >= self.config.max_tickers:
            try:
                oldest_key = min(
                    self.cache.keys(), key=lambda k: self.cache[k]["timestamp"]
                )
                del self.cache[oldest_key]
            except (TypeError, AttributeError) as e:
                buffered_print(f"Cache cleanup failed: {str(e)}", "ERROR")
                self.cache = {}  # Reset cache if corrupted

        self.cache[cache_key] = {"data": data, "timestamp": datetime.now()}

    # @retry_api_call
    def get_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        period: Optional[str] = None,
        reset_cache: bool = False,
    ) -> Optional[pd.DataFrame]:
        cache_key = self._get_cache_key(ticker, start_date, end_date, interval)
        # if not reset_cache and self._is_cache_valid(cache_key):
        #     buffered_print(f"get_stock_data failed due to cache.", "ERROR")

        try:
            start_date, end_date = self._validate_dates(start_date, end_date)
        except ValueError as e:
            buffered_print(str(e), "ERROR")
            return None

        # Rate limiting
        elapsed = (datetime.now() - self._last_api_call).total_seconds()
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_api_call = datetime.now()

        # cache_key = self._get_cache_key(ticker, start_date, end_date, interval)
        if self._is_cache_valid(cache_key):
            self._cache_hits += 1
            # buffered_print(
            #     f"Cache hit for {ticker} ({start_date} to {end_date})", "CACHE"
            # )
            return self.cache[cache_key]["data"].copy()

        self._cache_misses += 1
        # buffered_print(
        # f"Downloading {ticker} data ({start_date} to {end_date})", "PROCESS"
        # )

        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
        except Exception as primary_error:
            # if self.fallback_source:
            # return self._try_fallback_source(ticker, start_date, end_date, interval)
            buffered_print(
                f"Primary source failed: {str(primary_error)} Ticker: {ticker}", "ERROR"
            )
            return None

        processed = self._process_data(data, ticker)
        if processed is not None:
            self._update_cache(cache_key, processed)
            # buffered_print(f"Cached {ticker} ({len(processed)} records)", "SUCCESS")
            return processed.copy()
        else:
            self._update_cache(cache_key, pd.DataFrame())
            return None

    def _process_data(self, data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Process raw DataFrame with enhanced validation"""
        if data.empty:
            buffered_print(f"No data available for {ticker}", "ERROR")
            return None

        try:
            data.index = pd.to_datetime(data.index)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            for col in self.required_columns:
                if col not in data.columns:
                    data[col] = np.nan  # Add missing columns with NaN values

            return data[self.required_columns].sort_index()
        except Exception as e:
            buffered_print(
                f"Processing failed for {ticker}: {str(e)}", "ERROR")
            return None

    def get_cache_metrics(self) -> Dict:
        """Get cache performance metrics"""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0
            ),
            "current_size": len(self.cache),
            "max_size": self.config.max_tickers,
        }

    def get_sp500_tickers(self, refresh: bool = False) -> List[str]:
        """Fetch S&P 500 components with enhanced status reporting and fallback"""
        cache_key = "sp500_tickers"

        if not refresh and cache_key in self.cache:
            buffered_print("Returning cached S&P 500 tickers", "CACHE")
            return self.cache[cache_key]

        buffered_print(
            "Fetching S&P 500 components from Wikipedia...", "START")
        try:
            import requests
            from io import StringIO

            wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

            # Use requests library with proper User-Agent to avoid HTTP 403
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(wiki_url, headers=headers, timeout=10)
            response.raise_for_status()

            # Use pd.read_html with the HTML content from requests
            tables = pd.read_html(StringIO(response.text), attrs={"id": "constituents"})
            wiki_tickers = tables[0]["Symbol"].tolist()
            self.cache[cache_key] = wiki_tickers
            buffered_print(
                f"Successfully fetched {len(wiki_tickers)} tickers", "SUCCESS"
            )
            return wiki_tickers
        except Exception as e:
            buffered_print(f"Wikipedia fetch failed: {str(e)}", "WARNING")

            # Fallback: use hardcoded top S&P 500 tickers if Wikipedia unavailable
            fallback_tickers = [
                'AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOG', 'GOOGL', 'AMZN', 'META',
                'BRK.B', 'JNJ', 'V', 'WMT', 'KO', 'PG', 'MCD', 'DIS', 'NFLX',
                'BA', 'MMM', 'JPM', 'BAC', 'WFC', 'USB', 'GS', 'MS', 'BLK', 'SPY',
                'PEP', 'AXP', 'CSCO', 'INTC', 'AMD', 'ORCL', 'CRM', 'ADBE', 'AVGO',
                'UBER', 'PYPL', 'SQ', 'COIN', 'MU', 'QCOM', 'ASML', 'TSM', 'LMT',
                'RTX', 'NOC', 'GD', 'F', 'GM', 'EXC', 'SO', 'DUK', 'NEE', 'AEP',
                'XOM', 'CVX', 'MPC', 'PSX', 'EOG', 'COP', 'SLB', 'HAL', 'OKE', 'MRO',
                'ORCL', 'IBM', 'SAP', 'UBER', 'LYFT', 'ABNB', 'ZM', 'OKTA', 'TWLO',
                'RUM', 'SHOP', 'ETSY', 'DASH', 'DDOG', 'SNOW', 'CRWD', 'PALO', 'NET',
                'ZS', 'WDAY', 'ANSS', 'SPLK', 'SUMO', 'FTNT', 'CHKP', 'PANW', 'VRNS'
            ]

            self.cache[cache_key] = fallback_tickers
            buffered_print(
                f"Using fallback: {len(fallback_tickers)} tickers loaded", "WARNING"
            )
            return fallback_tickers

    def get_low_beta_high_dividend(
        self,
        max_beta: float = 1.0,
        min_dividend: float = 0.02,
        max_momentum: float = 0.3,
    ) -> List[str]:
        """Stock screening with improved progress tracking"""
        cache_key = f"screen|{max_beta}|{min_dividend}|{max_momentum}"

        if cache_key in self.cache:
            buffered_print("Returning cached screening results", "CACHE")
            return self.cache[cache_key]

        tickers = self.get_sp500_tickers()
        qualified = []
        error_log = {}

        buffered_print(
            f"Screening {len(tickers)} stocks across 10 workers", "PROCESS")

        def process_ticker(ticker):
            try:
                data = yf.Ticker(ticker)
                info = data.info
                hist = data.history(period="1y")

                if len(hist) < 2:
                    return None

                metrics = {
                    "beta": info.get("beta", 2.0),
                    "div_yield": info.get("dividendYield", 0.0),
                    "momentum": (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1,
                }

                if (
                    metrics["beta"] < max_beta
                    and metrics["div_yield"] > min_dividend
                    and abs(metrics["momentum"]) < max_momentum
                ):
                    return ticker

            except Exception as e:
                error_log[ticker] = str(e)
            return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_ticker, t): t for t in tickers}
            progress = tqdm(
                as_completed(futures),
                total=len(tickers),
                desc="Screening Stocks",
                unit="ticker",
            )

            for future in progress:
                if result := future.result():
                    qualified.append(result)

        buffered_print(f"Qualified {len(qualified)} stocks", "RESULT")
        self.cache[cache_key] = qualified
        return qualified

    def _align_dataframes(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> List[pd.DataFrame]:
        """Align two dataframes on their indices with inner join using list return type"""
        aligned = df1.align(df2, join="inner", axis=0)
        return [aligned[0], aligned[1]]

    @staticmethod
    def _is_crypto_symbol(ticker: str) -> bool:
        return ticker.upper().endswith("-USD")

    def _align_pair_dataframes(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        calendar_mode: str = "intersection",
    ) -> List[pd.DataFrame]:
        """
        Align two market series using an explicit calendar policy.

        calendar_mode:
        - intersection: exact timestamp overlap only.
        - business: business-day calendar with bounded forward-fill.
        - all_days: daily calendar with bounded forward-fill.
        """
        if calendar_mode == "intersection":
            return self._align_dataframes(df1, df2)

        if df1.empty or df2.empty:
            return [pd.DataFrame(), pd.DataFrame()]

        start = max(df1.index.min(), df2.index.min())
        end = min(df1.index.max(), df2.index.max())
        if pd.isna(start) or pd.isna(end) or start > end:
            return [pd.DataFrame(), pd.DataFrame()]

        freq = "B" if calendar_mode == "business" else "D"
        calendar = pd.date_range(start=start, end=end, freq=freq)

        # Keep fill limits tight to avoid carrying stale prices too far.
        fill_limit = 3 if calendar_mode == "business" else 2
        a1 = df1.reindex(calendar).ffill(limit=fill_limit).dropna(how="any")
        a2 = df2.reindex(calendar).ffill(limit=fill_limit).dropna(how="any")
        a1, a2 = a1.align(a2, join="inner")
        return [a1, a2]

    def get_volatility_index(
        self,
        symbol: str = "^VIX",
        days: int = 20,
        as_of_date: Optional[Union[str, datetime, date]] = None,
    ) -> Tuple[float, float]:
        """Robust volatility index fetching with comprehensive error handling"""
        # Validate parameters with defaults
        if not isinstance(days, int) or days <= 0:
            days = 20
            buffered_print(
                f"Invalid days value, using default: {days}", "WARN")

        # Handle date input and ensure datetime type
        if as_of_date is None:
            as_of_date = datetime.now()
        elif isinstance(as_of_date, str):
            try:
                as_of_date = datetime.strptime(as_of_date, "%Y-%m-%d")
            except ValueError:
                buffered_print(
                    f"Invalid date format, using current date", "WARN")
                as_of_date = datetime.now()
        elif isinstance(as_of_date, date):
            as_of_date = datetime.combine(as_of_date, datetime.min.time())

        # Validate date range
        if as_of_date > datetime.now():
            buffered_print("Future date requested, using current date", "WARN")
            as_of_date = datetime.now()

        # Generate cache key
        date_str = as_of_date.strftime("%Y-%m-%d")
        cache_key = self._get_cache_key(
            symbol, f"{days}d", date_str, "volatility")

        # Return cached data if valid
        if self._is_cache_valid(cache_key):
            self._cache_hits += 1
            cached_data = self.cache[cache_key]["data"]
            return (cached_data["ma"], cached_data["last_close"])

        # Calculate date range with buffer
        end_date = as_of_date.strftime("%Y-%m-%d")
        start_date = (as_of_date - timedelta(days=days * 3)
                      ).strftime("%Y-%m-%d")

        try:
            # Attempt to fetch VIX data
            df = self.get_stock_data(
                symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )

            # Handle data fetch failure
            if df is None or df.empty:
                raise ValueError("No data returned from get_stock_data")

            # Filter to valid dates
            valid_dates = df[df.index <= pd.to_datetime(as_of_date)]
            if valid_dates.empty:
                raise ValueError("No data available before specified date")

            # Find last trading day
            last_close_date = valid_dates.index.max()
            closes = valid_dates["Close"].loc[:last_close_date]

            # Adjust window size if insufficient data
            if len(closes) < days:
                days = max(1, len(closes))
                buffered_print(f"Adjusted window size to {days} days", "WARN")

            # Calculate metrics
            ma = closes.rolling(window=days).mean().dropna().iloc[-1]
            last_close = closes.iloc[-1]

            # Validate calculations
            if np.isnan(ma) or np.isnan(last_close):
                raise ValueError("NaN values in calculated metrics")

            # Package results
            result = {
                "ma": round(float(ma), 2),
                "last_close": round(float(last_close), 2),
                "calculation_date": last_close_date.strftime("%Y-%m-%d"),
            }

            self._update_cache(cache_key, result)
            return (result["ma"], result["last_close"])

        except Exception as e:
            buffered_print(
                f"Volatility index error: {str(e)} - using defaults (20.0, 20.0)",
                "ERROR"
            )
            return (20.0, 20.0)

    # def get_volatility_index(
    #     self,
    #     symbol: str = "^VIX",
    #     days: int = 20,
    #     as_of_date: Optional[Union[str, datetime, date]] = None,
    # ) -> Tuple[float, float]:
    #     """Fetch X-day moving average and closing value for volatility index as of specific date"""
    #     # Validate parameters
    #     if not isinstance(days, int) or days <= 0:
    #         buffered_print(f"Invalid days value: {days}. Using default 20.", "WARN")
    #         days = 20

    #     # Handle date input and ensure datetime.datetime type
    #     if as_of_date is None:
    #         as_of_date = datetime.now()
    #     elif isinstance(as_of_date, str):
    #         try:
    #             as_of_date = datetime.strptime(as_of_date, "%Y-%m-%d")
    #         except ValueError:
    #             buffered_print(
    #                 f"Invalid date format: {as_of_date}. Using current date.", "WARN"
    #             )
    #             as_of_date = datetime.now()
    #     elif isinstance(as_of_date, date):
    #         # Convert date to datetime at midnight
    #         as_of_date = datetime.combine(as_of_date, datetime.min.time())

    #     # Now ensure comparison works with datetime objects
    #     if as_of_date > datetime.now():
    #         buffered_print("Future date requested, using current date", "WARN")
    #         as_of_date = datetime.now()

    #     # Generate cache key with date context
    #     date_str = as_of_date.strftime("%Y-%m-%d")
    #     cache_key = self._get_cache_key(symbol, f"{days}d", date_str, "volatility")

    #     if self._is_cache_valid(cache_key):
    #         self._cache_hits += 1
    #         cached_data = self.cache[cache_key]["data"]
    #         return (cached_data["ma"], cached_data["last_close"])

    #     # Calculate date range with buffer
    #     end_date = as_of_date.strftime("%Y-%m-%d")
    #     start_date = (as_of_date - timedelta(days=days * 3)).strftime("%Y-%m-%d")

    #     df = self.get_stock_data(
    #         symbol, start_date=start_date, end_date=end_date, interval="1d"
    #     )

    #     if df is None or df.empty:
    #         buffered_print(
    #             f"Failed to fetch {symbol} data for {date_str} - utilizing default values (20, 20)",
    #             "ERROR",
    #         )
    #         return (20.0, 20.0)

    #     # Filter to dates <= as_of_date
    #     valid_dates = df[df.index <= pd.to_datetime(as_of_date)]
    #     if valid_dates.empty:
    #         buffered_print(f"No data available before {date_str}", "ERROR")
    #         return (20.0, 20.0)

    #     # Find actual last trading day <= requested date
    #     last_close_date = valid_dates.index.max()
    #     closes = valid_dates["Close"].loc[:last_close_date]

    #     if len(closes) < days:
    #         buffered_print(
    #             f"Only {len(closes)}/{days} trading days before {date_str}", "WARN"
    #         )
    #         days = len(closes) if len(closes) > 0 else 1

    #     try:
    #         # Calculate metrics relative to as_of_date
    #         ma = closes.rolling(window=days).mean().dropna().iloc[-1]
    #         last_close = closes.iloc[-1]
    #     except (IndexError, KeyError) as e:
    #         buffered_print(f"Calculation error: {str(e)}", "ERROR")
    #         return (20.0, 20.0)

    #     # Validate calculations
    #     if np.isnan(ma) or np.isnan(last_close):
    #         buffered_print(f"NaN values detected for {symbol}", "ERROR")
    #         return (20.0, 20.0)

    #     # Package results
    #     result = {
    #         "ma": round(float(ma), 2),
    #         "last_close": round(float(last_close), 2),
    #         "calculation_date": last_close_date.strftime("%Y-%m-%d"),
    #     }

    #     self._update_cache(cache_key, result)
    #     # buffered_print(
    #     # f"Computed {days}D MA: {result['ma']} as of {result['calculation_date']}",
    #     # "SUCCESS",
    #     # )

    #     return (result["ma"], result["last_close"])

    #     # Validate results
    #     if np.isnan(ma) or np.isnan(last_close) or ma <= 0:
    #         buffered_print(f"Invalid calculated values for {symbol}", "ERROR")
    #         return (20.0, 20.0)

    #     # Package and cache results
    #     result = {"ma": round(float(ma), 2), "last_close": round(float(last_close), 2)}
    #     self._update_cache(cache_key, result)
    #     buffered_print(
    #         f"Computed {days}D MA: {result['ma']}, Last Close: {result['last_close']}",
    #         "SUCCESS",
    #     )

    #     return (result["ma"], result["last_close"])

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
        """Get normalized prices using merged Open/Close values for two tickers"""
        if calendar_mode == "auto":
            is_mixed = self._is_crypto_symbol(t1) != self._is_crypto_symbol(t2)
            calendar_mode = "business" if is_mixed else "intersection"

        cache_key = (
            f"pair|{t1}-{t2}|{calendar_mode}|{self._get_cache_key('', start_date, end_date, interval)}"
        )

        if self._is_cache_valid(cache_key):
            # buffered_print(f"Returning cached pair {t1}-{t2}", "CACHE")
            return self.cache[cache_key]["data"].copy()

        # buffered_print(f"Processing pair {t1}-{t2}", "PROCESS")

        # Fetch and validate data
        data1 = self.get_stock_data(t1, start_date, end_date, interval, period)
        data2 = self.get_stock_data(t2, start_date, end_date, interval, period)
        if data1 is None or data2 is None or data1.empty or data2.empty:
            buffered_print(f"Missing data for pair {t1}-{t2}", "ERROR")
            return None

        try:
            aligned = self._align_pair_dataframes(data1, data2, calendar_mode=calendar_mode)
            if aligned[0].empty or aligned[1].empty:
                buffered_print(f"No overlapping data for {t1}-{t2}", "WARN")
                return None

            # Merge Open and Close
            result = pd.DataFrame(
                columns=["Date", f"Merged {t1}", f"Merged {t2}"])

            for i in range(0, len(aligned[0])):
                date = aligned[0].index[i]

                result.loc[i * 2, f"Date"] = date
                result.loc[i * 2 + 1, f"Date"] = date

                result.loc[i * 2, f"Merged {t1}"] = aligned[0]["Open"].loc[
                    aligned[0].index[i]
                ]
                result.loc[i * 2, f"Merged {t2}"] = aligned[1]["Open"].loc[
                    aligned[1].index[i]
                ]

                result.loc[i * 2 + 1, f"Merged {t1}"] = aligned[0]["Close"].loc[
                    aligned[0].index[i]
                ]
                result.loc[i * 2 + 1, f"Merged {t2}"] = aligned[1]["Close"].loc[
                    aligned[1].index[i]
                ]

            result.set_index("Date", inplace=True)
            result.index = pd.to_datetime(result.index)
            result.sort_index(inplace=True)

            # Validate merged prices
            if result.iloc[0].min() <= 0:
                buffered_print(f"Invalid merged prices for {t1}-{t2}", "ERROR")
                return None

            # Normalize to initial merged value
            result[f"Normalized {t1}"] = (
                result[f"Merged {t1}"] / result[f"Merged {t1}"].iloc[0]
            )
            result[f"Normalized {t2}"] = (
                result[f"Merged {t2}"] / result[f"Merged {t2}"].iloc[0]
            )

            self._update_cache(cache_key, result)
            # buffered_print(f"Created merged-normalized pair {t1}-{t2}\n", "SUCCESS")
            return result.copy()

        except Exception as e:
            buffered_print(f"Pair processing failed: {str(e)}", "ERROR")
            return None

    def get_mixed_asset_history(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        calendar_mode: str = "business",
        normalize: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Build a single aligned price panel across equities and crypto.

        This is stable for mixed assets because the calendar policy is explicit.
        """
        if not tickers:
            return None

        series = {}
        for ticker in tickers:
            df = self.get_stock_data(ticker, start_date, end_date, interval=interval)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if len(close) >= 20:
                series[ticker] = close

        if len(series) < 2:
            return None

        panel = pd.concat(series, axis=1, join="outer", sort=True).sort_index()
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        panel = panel[(panel.index >= start) & (panel.index <= end)]
        if panel.empty:
            return None

        if calendar_mode in {"business", "all_days"}:
            freq = "B" if calendar_mode == "business" else "D"
            target_index = pd.date_range(start=panel.index.min(), end=panel.index.max(), freq=freq)
            panel = panel.reindex(target_index).ffill(limit=3)

        panel = panel.dropna(how="any")
        if panel.empty:
            return None

        if normalize:
            panel = panel / panel.iloc[0]

        return panel

    def plot_normalized_pair(
        self,
        t1: str,
        t2: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        show_table: bool = False,
        period: Optional[str] = None,
    ) -> None:
        """Enhanced visualization with formatted output and error handling"""
        df = self.get_normalized_pair(
            t1, t2, start_date, end_date, interval, period)
        if df is None:
            buffered_print("No data available for plotting", "ERROR")
            return

        # Display table if requested
        if show_table:
            buffered_print("Displaying sample data:", "INFO")
            buffered_print(f"\n{' Closing Prices '.center(80, '-')}")
            buffered_print(df[[f"Close {t1}", f"Close {t2}"]].tail(5))
            buffered_print(f"\n{' Normalized Prices '.center(80, '-')}")
            buffered_print(
                df[[f"Normalized {t1}", f"Normalized {t2}"]].tail(5))

        # Create plot
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            plt.suptitle(f"Price Analysis: {t1} vs {t2}", y=0.95)

            # Normalized plot
            df[[f"Normalized {t1}", f"Normalized {t2}"]].plot(
                ax=ax1, title="Normalized Price Comparison", style=["-", "--"]
            )

            # Absolute plot
            df[[f"Close {t1}", f"Close {t2}"]].plot(
                ax=ax2, title="Absolute Price Comparison", style=["-", "--"]
            )

            plt.tight_layout()
            plt.show()
            buffered_print("Plotting completed successfully", "SUCCESS")

        except Exception as e:
            buffered_print(f"Plotting failed: {str(e)}", "ERROR")

    def _init_fallback_source(self):
        """Initialize alternative data sources"""
        self.fallback_source = None
        if self.config.fallback_enabled:
            try:
                from alpha_vantage import TimeSeries

                api_key = os.getenv("ALPHAVANTAGE_API_KEY")
                if api_key:
                    self.fallback_source = TimeSeries(
                        api_key, output_format="pandas")
                    buffered_print("Fallback source initialized", "CONFIG")
                else:
                    buffered_print(
                        "Missing API key for fallback source", "WARN")
            except ImportError:
                buffered_print(
                    "Fallback source unavailable (install alpha_vantage)", "WARN"
                )

    # def _try_fallback_source(
    #     self, ticker: str, start_date: str, end_date: str, interval: str
    # ) -> Optional[pd.DataFrame]:
    #     """Attempt to use fallback data source"""
    #     buffered_print("Attempting fallback data source", "WARN")
    #     try:
    #         if interval != "1d":
    #             buffered_print("Fallback only supports daily data", "WARN")
    #             return None

    #         data, _ = self.fallback_source.get_daily_adjusted(
    #             symbol=ticker, outputsize="full"
    #         )
    #         df = data.copy()
    #         df = df.rename(
    #             columns={
    #                 "1. open": "Open",
    #                 "2. high": "High",
    #                 "3. low": "Low",
    #                 "4. close": "Close",
    #                 "6. volume": "Volume",
    #             }
    #         )
    #         df.index = pd.to_datetime(df.index)
    #         df = df.sort_index().loc[start_date:end_date]
    #         return self._process_data(df, ticker)
    #     except Exception as fallback_error:
    #         buffered_print(f"Fallback failed: {str(fallback_error)}", "ERROR")
    #         return None
