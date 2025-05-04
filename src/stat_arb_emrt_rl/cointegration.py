# ───────────────────────────────────────────────
# Standard Library Imports
# ───────────────────────────────────────────────
import os
import time
import json
import hashlib
from datetime import datetime, timedelta

# ───────────────────────────────────────────────
# Third-Party Package Imports
# ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

# ───────────────────────────────────────────────
# Typing and Project-Specific Imports
# ───────────────────────────────────────────────
from typing import Dict, List, Optional, Tuple
from .financial_loader import FinancialLoader
from .printing_system import print_header


class CointegrationConfig:
    """Configuration class for cointegration analysis parameters"""

    def __init__(self):
        self.max_workers = 18  # Conservative default
        self.p_value_threshold = 0.01
        self.cache_ttl = 3600 * 24  # 24 hours
        self.max_pairs = 25000
        self.heatmap_size = (20, 20)
        self.bulk_download_size = 25  # Number of tickers per bulk request
        self.min_request_interval = 2.0  # Seconds between bulk requests
        self.cache_dir = "coint_cache"
        self.serialization_format = "parquet"
        self.max_retries = 3
        self.retry_delay = 5.0  # Seconds between retries
        self.request_delay = 0.0


class CointegrationAnalyzer:
    def __init__(
        self,
        financial_loader: FinancialLoader,
        tickers: Optional[List[str]] = None,
        start_date: str = "2024-01-01",
        end_date: str = "2025-01-01",
        config: CointegrationConfig = None,
        reset_cache: bool = False,
    ):
        self.loader = financial_loader
        self.tickers = tickers or []
        self.start_date = start_date
        self.end_date = end_date
        self.config = config or CointegrationConfig()
        self.data = pd.DataFrame()
        self.coint_matrix = None
        self._init_cache()
        self._last_bulk_request = datetime.min
        self.reset_cache = reset_cache
        print_header(message="Cointegration Analyzer Initialized")

    def _init_cache(self):
        """Initialize cache directory structure"""
        os.makedirs(self.config.cache_dir, exist_ok=True)
        self.ticker_cache_dir = os.path.join(self.config.cache_dir, "tickers")
        os.makedirs(self.ticker_cache_dir, exist_ok=True)

    def _print_status(self, message: str, status: str = "INFO"):
        """Leverage FinancialLoader's status reporting"""
        self.loader._print_status(message, status)

    def _get_cache_key(self, items: List[str]) -> str:
        """Generate consistent cache key for multiple items"""
        base_key = "|".join(items + [self.start_date, self.end_date])
        return hashlib.md5(base_key.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> str:
        """Get file path for cached data"""
        return os.path.join(
            self.ticker_cache_dir, f"{key}.{self.config.serialization_format}"
        )

    def _is_cache_valid(self, file_path: str) -> bool:
        """Check if cache file is still valid"""
        if not os.path.exists(file_path):
            return False
        file_age = datetime.now().timestamp() - os.path.getmtime(file_path)
        return file_age < self.config.cache_ttl

    def _load_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        cache_path = self._get_cache_path(key)
        if not self._is_cache_valid(cache_path):
            return None

        try:
            if self.config.serialization_format == "parquet":
                return pd.read_parquet(cache_path)
            return pd.read_pickle(cache_path)
        except Exception as e:
            self._print_status(f"Cache load failed: {str(e)}", "ERROR")
            return None

    def _save_to_cache(self, key: str, data: pd.DataFrame):
        """Save data to cache"""
        cache_path = self._get_cache_path(key)
        try:
            if self.config.serialization_format == "parquet":
                data.to_parquet(cache_path)
            else:
                data.to_pickle(cache_path)
        except Exception as e:
            self._print_status(f"Cache save failed: {str(e)}", "ERROR")

    def _throttle_requests(self):
        """Enforce minimum time between bulk requests"""
        elapsed = (datetime.now() - self._last_bulk_request).total_seconds()
        if elapsed < self.config.min_request_interval:
            sleep_time = self.config.min_request_interval - elapsed
            time.sleep(sleep_time)
        self._last_bulk_request = datetime.now()

    def _download_ticker_batch(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Download a batch of tickers with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                self._throttle_requests()
                data = yf.download(
                    tickers,
                    start=self.start_date,
                    end=self.end_date,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                )
                return {ticker: data[ticker] for ticker in tickers if ticker in data}
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    self._print_status(
                        f"Failed batch download: {str(e)}", "ERROR")
                    return {}
                delay = self.config.retry_delay * (attempt + 1)
                self._print_status(
                    f"Retrying in {delay}s (Attempt {attempt + 1})", "WARN"
                )
                time.sleep(delay)
        return {}

    def _process_ticker_data(
        self, ticker: str, df: pd.DataFrame
    ) -> Optional[pd.Series]:
        """Process raw ticker data into stacked format"""
        if df.empty or len(df) < 30:
            self._print_status(
                f"Rejected {ticker}: insufficient history", "WARN")
            return None

        try:
            stacked = (
                df[["Open", "Close"]]
                .stack()
                .swaplevel()
                .sort_index()
                .rename_axis(index=["date", "price_type"])
                .rename(ticker)
            )
            return stacked
        except Exception as e:
            self._print_status(f"Error processing {ticker}: {str(e)}", "ERROR")
            return None

    def fetch_data(self, reset_cache=False):
        """Fetch all ticker data using batched downloads"""
        self._print_status(
            f"Fetching data for {len(self.tickers)} tickers...", "INFO")

        if reset_cache:
            self._print_status("Cache reset requested", "INFO")
            dataset_key = self._get_cache_key(["full_dataset"])
            cache_path = self._get_cache_path(dataset_key)
            if os.path.exists(cache_path):
                os.remove(cache_path)

        # Try to load full dataset from cache first
        dataset_key = self._get_cache_key(["full_dataset"])
        cached_data = self._load_from_cache(
            dataset_key) if not reset_cache else None
        if cached_data is not None:
            self.data = cached_data
            self._print_status("Loaded full dataset from cache", "CACHE")
            return

        # Process in batches
        all_data = {}
        for i in range(0, len(self.tickers), self.config.bulk_download_size):
            batch = self.tickers[i: i + self.config.bulk_download_size]
            batch_key = self._get_cache_key(batch)

            # Try to load batch from cache
            cached_batch = self._load_from_cache(batch_key)
            if cached_batch is not None:
                self._print_status(
                    f"Loaded batch {i//self.config.bulk_download_size + 1} from cache",
                    "CACHE",
                )
                all_data.update(cached_batch)
                continue

            # Download batch
            self._print_status(
                f"Downloading batch {i//self.config.bulk_download_size + 1}/"
                f"{len(self.tickers)//self.config.bulk_download_size + 1}",
                "PROCESS",
            )
            batch_data = self._download_ticker_batch(batch)

            # Process and cache batch
            processed_batch = {}
            for ticker, df in batch_data.items():
                processed = self._process_ticker_data(ticker, df)
                if processed is not None:
                    processed_batch[ticker] = processed

            if processed_batch:
                self._save_to_cache(batch_key, pd.DataFrame(processed_batch))
                all_data.update(processed_batch)

        if not all_data:
            raise ValueError("No valid ticker data fetched")

        self.data = pd.concat(all_data.values(), axis=1)
        self._save_to_cache(dataset_key, self.data)
        self._print_status(f"Data loaded: {self.data.shape}", "SUCCESS")

    # def clean_data(self):
    #     """Clean and align data across all tickers"""
    #     self.loader._print_status("Cleaning data...", "INFO")

    #     # Forward fill then drop remaining NAs
    #     self.data = self.data.ffill().bfill()
    #     self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
    #     self.data = self.data.dropna(axis=1, how="all")

    #     # Ensure minimum length
    #     min_length = 20  # Increased from 10
    #     self.data = self.data.loc[:, self.data.count() >= min_length]

    #     self.loader._print_status(f"Cleaned data shape: {self.data.shape}", "SUCCESS")
    def clean_data(self):
        """Clean and align data across all tickers"""
        self.loader._print_status("Cleaning data...", "INFO")

        # Fill missing values and drop invalid columns
        self.data = self.data.ffill().bfill().replace(
            [np.inf, -np.inf], np.nan)
        self.data = self.data.dropna(axis=1, how="all")

        # Align all tickers to common dates
        common_dates = self.data.index
        for col in self.data.columns:
            common_dates = common_dates.intersection(
                self.data[col].dropna().index)
        self.data = self.data.loc[common_dates]

        # Filter minimum length and non-constant tickers
        min_length = 20
        valid_cols = [
            col
            for col in self.data.columns
            if (self.data[col].count() >= min_length) and (self.data[col].nunique() > 1)
        ]
        self.data = self.data[valid_cols]

        self.loader._print_status(
            f"Cleaned data shape: {self.data.shape}", "SUCCESS")

    def _save_dataset_cache(self):
        """Save full dataset cache"""
        cache_path = os.path.join(
            self.config.cache_dir,
            f"full_dataset_{self._get_cache_key('dataset')}.{self.config.serialization_format}",
        )
        try:
            self.data.to_parquet(cache_path)
        except Exception as e:
            self._print_status(f"Dataset cache save failed: {str(e)}", "ERROR")

    def _load_dataset_cache(self) -> bool:
        """Try to load full dataset from cache"""
        cache_path = os.path.join(
            self.config.cache_dir,
            f"full_dataset_{self._get_cache_key('dataset')}.{self.config.serialization_format}",
        )
        if os.path.exists(cache_path) and self._is_cache_valid(cache_path):
            try:
                self.data = pd.read_parquet(cache_path)
                self._print_status("Loaded full dataset from cache", "CACHE")
                return True
            except Exception as e:
                self._print_status(
                    f"Dataset cache load failed: {str(e)}", "ERROR")
        return False

    def test_cointegration(self, ticker1: str, ticker2: str) -> float:
        """Perform cointegration test on aligned price series"""
        try:
            _, p_value, _ = coint(self.data[ticker1], self.data[ticker2])
            return p_value
        except Exception as e:
            self.loader._print_status(
                f"Cointegration test failed for {ticker1}-{ticker2}: {str(e)}", "ERROR"
            )
            return np.nan

    # def _cointegration_test(self, ticker1: str, ticker2: str) -> Tuple[float, bool]:
    #     """Perform cointegration test with enhanced validation and alignment"""
    #     try:
    #         # Get data for both tickers
    #         series1 = self.data[ticker1]
    #         series2 = self.data[ticker2]

    #         # Align the series by their indices (dates)
    #         aligned = pd.DataFrame({ticker1: series1, ticker2: series2}).dropna()
    #         if len(aligned) < 10:
    #             self._print_status(
    #                 f"Insufficient aligned data for {ticker1}-{ticker2} ({len(aligned)} points)",
    #                 "WARN",
    #             )
    #             return (np.nan, False)

    #         # Extract aligned series
    #         aligned_series1 = aligned[ticker1]
    #         aligned_series2 = aligned[ticker2]

    #         # Check for constant values which will cause coint to fail
    #         if aligned_series1.nunique() == 1 or aligned_series2.nunique() == 1:
    #             self._print_status(
    #                 f"Constant values detected for {ticker1}-{ticker2}", "WARN"
    #             )
    #             return (np.nan, False)

    #         # Perform cointegration test on aligned series
    #         _, p_value, _ = coint(aligned_series1, aligned_series2)
    #         return (p_value, True)

    #     except Exception as e:
    #         self._print_status(
    #             f"Cointegration test failed for {ticker1}-{ticker2}: {str(e)}", "ERROR"
    #         )
    #         return (np.nan, False)
    def _cointegration_test(self, ticker1: str, ticker2: str) -> Tuple[float, bool]:
        """Test pre-aligned series"""
        try:
            _, p_value, _ = coint(self.data[ticker1], self.data[ticker2])
            return (p_value, True)
        except Exception as e:
            self.loader._print_status(
                f"Cointegration test failed for {ticker1}-{ticker2}: {str(e)}", "ERROR"
            )
            return (np.nan, False)

    def build_coint_matrix(self) -> pd.DataFrame:
        """Build matrix with parallel pair testing and caching"""
        matrix_key = self._get_cache_key(["coint_matrix"])
        cache_path = self._get_cache_path(matrix_key)

        if os.path.exists(cache_path) and self._is_cache_valid(cache_path):
            try:
                self.coint_matrix = pd.read_parquet(cache_path)
                self._print_status(
                    "Loaded cointegration matrix from cache", "CACHE")
                return self.coint_matrix
            except Exception as e:
                self._print_status(
                    f"Matrix cache load failed: {str(e)}", "ERROR")

        if len(self.data) < 30:
            raise ValueError(
                "Insufficient data for meaningful cointegration analysis")

        self._print_status("Building cointegration matrix...", "INFO")
        n = len(self.tickers)

        matrix = np.full((n, n), np.nan)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        chunk_size = max(1, len(pairs) // (self.config.max_workers * 2))

        pair_results = Parallel(n_jobs=self.config.max_workers)(
            delayed(self._cointegration_test)(self.tickers[i], self.tickers[j])
            for i, j in tqdm(pairs, desc=f"Testing {len(pairs)} pairs")
        )

        for idx, (i, j) in enumerate(pairs):
            p_value, success = pair_results[idx]
            if success:
                matrix[i, j] = p_value
                matrix[j, i] = p_value
        np.fill_diagonal(matrix, 1.0)

        self.coint_matrix = pd.DataFrame(
            matrix, index=self.tickers, columns=self.tickers
        )

        try:
            self.coint_matrix.to_parquet(cache_path)
        except Exception as e:
            self._print_status(f"Matrix cache save failed: {str(e)}", "ERROR")

        return self.coint_matrix

    def get_significant_pairs(self) -> pd.DataFrame:
        """Get significant pairs with caching"""
        pairs_key = self._get_cache_key(["significant_pairs"])
        cache_path = self._get_cache_path(pairs_key)

        if os.path.exists(cache_path) and self._is_cache_valid(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                self._print_status(
                    f"Pairs cache load failed: {str(e)}", "ERROR")

        pairs = []
        for i in range(len(self.tickers)):
            for j in range(i + 1, len(self.tickers)):
                pval = self.coint_matrix.iloc[i, j]
                if pval < self.config.p_value_threshold:
                    pairs.append(
                        {
                            "Ticker1": self.tickers[i],
                            "Ticker2": self.tickers[j],
                            "P-Value": pval,
                        }
                    )

        result = pd.DataFrame(pairs).sort_values("P-Value")
        try:
            result.to_parquet(cache_path)
        except Exception as e:
            self._print_status(f"Pairs cache save failed: {str(e)}", "ERROR")

        return result

    def visualize_results(
        self, threshold: float = 0.05, save_path: Optional[str] = None
    ):
        """Generate heatmap visualization of cointegration matrix"""
        plt.figure(figsize=self.config.heatmap_size)
        mask = np.triu(np.ones_like(self.coint_matrix, dtype=bool))

        sns.heatmap(
            (self.coint_matrix < threshold) & (self.coint_matrix > 0),
            annot=False,
            mask=mask,
            cmap="viridis",
            cbar_kws={"label": f"p < {threshold}"},
        )

        plt.title("Cointegration Matrix (Engle-Granger Test)")
        plt.xlabel("Tickers")
        plt.ylabel("Tickers")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()


# ───────────────────────────────────────────────
# Cryptocurrency Tickers for Cointegration
# ───────────────────────────────────────────────
CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "ADA-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD",
    "ATOM-USD", "UNI-USD", "LTC-USD", "NEAR-USD", "FIL-USD",
    "APT-USD", "ARB-USD", "OP-USD", "DOGE-USD", "SHIB-USD",
]


def eg_analysis(
    loader: FinancialLoader,
    start_date: str = "2024-01-01",
    end_date: str = "2025-01-01",
    beta_threshold: float = 1.0,
    dividend_threshold: float = 0.02,
    p_value_threshold: float = 0.05,
    heatmap: bool = False,
    beta_flag: bool = False,
    reset_cache: bool = False,
    include_crypto: bool = False,
    crypto_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main analysis workflow using FinancialLoader integration.

    Args:
        include_crypto: Add crypto tickers to the S&P 500 universe
        crypto_only: Search only among crypto tickers (no equities)
    """
    config = CointegrationConfig()
    config.p_value_threshold = p_value_threshold

    # Get tickers using FinancialLoader screening
    if beta_flag:
        low_beta = loader.get_low_beta_high_dividend(
            max_beta=beta_threshold, min_dividend=dividend_threshold
        )

    if crypto_only:
        tickers = list(CRYPTO_TICKERS)
    else:
        sp500 = loader.get_sp500_tickers()
        tickers = list(set(sp500 + (low_beta if beta_flag else []))
                       )[: config.max_pairs]

        if include_crypto:
            tickers = list(set(tickers + CRYPTO_TICKERS))

    exclude_tickers = {"BF.B", "BRK.B", "LEN", "ETR", "AAPL", "TSLA"}
    tickers = [t for t in tickers if t not in exclude_tickers]

    # Initialize and process
    analyzer = CointegrationAnalyzer(
        loader,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        config=config,
        reset_cache=reset_cache,
    )

    analyzer.fetch_data(reset_cache=reset_cache)
    analyzer.clean_data()
    coint_matrix = analyzer.build_coint_matrix()
    significant_pairs = analyzer.get_significant_pairs()
    # print(f"Significant Pairs:\n{significant_pairs.reset_index(inplace=True)}")
    significant_pairs.reset_index(inplace=True)
    if heatmap:
        analyzer.visualize_results(threshold=p_value_threshold)

    return coint_matrix, significant_pairs
