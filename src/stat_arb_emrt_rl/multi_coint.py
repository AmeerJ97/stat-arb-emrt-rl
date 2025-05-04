# ───────────────────────────────────────────────
# Multi-Timeframe, Cross-Asset Cointegration Engine
# Supports: pairs (EG), N-asset groups (Johansen),
# equities + crypto, short/medium/long horizons
# ───────────────────────────────────────────────
import os
import hashlib
import time
import itertools
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm

from .financial_loader import FinancialLoader
from .data_provider import FinancialLoaderProvider, MarketDataProvider
from .printing_system import (
    buffered_print,
    print_header,
    print_section,
    GREEN,
    RED,
    YELLOW,
    CYAN,
    BLUE,
    BOLD,
    ENDC,
)
from .gpu_monitor import print_gpu_status

# ───────────────────────────────────────────────
# Asset Universes
# ───────────────────────────────────────────────
CRYPTO_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "ADA-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD",
    "ATOM-USD", "UNI-USD", "LTC-USD", "NEAR-USD", "FIL-USD",
    "APT-USD", "ARB-USD", "OP-USD", "DOGE-USD", "SHIB-USD",
    "AAVE-USD", "MKR-USD", "SNX-USD", "CRV-USD", "RUNE-USD",
]

# Timeframe definitions (trading days)
TIMEFRAMES = {
    "short": {"days": 63, "label": "Short (3mo)"},
    "medium": {"days": 252, "label": "Medium (1yr)"},
    "long": {"days": 504, "label": "Long (2yr)"},
}


# ───────────────────────────────────────────────
# Engle-Granger Pairwise Cointegration
# ───────────────────────────────────────────────
def eg_test(
    s1: np.ndarray,
    s2: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Engle-Granger cointegration test for a pair."""
    try:
        if len(s1) < 30 or len(s2) < 30:
            return 1.0, False
        if np.std(s1) < 1e-10 or np.std(s2) < 1e-10:
            return 1.0, False
        _, pvalue, _ = coint(s1, s2)
        return float(pvalue), pvalue < alpha
    except Exception:
        return 1.0, False


# ───────────────────────────────────────────────
# Johansen N-Asset Cointegration
# ───────────────────────────────────────────────
def johansen_test(
    data: np.ndarray,
    det_order: int = 0,
    k_ar_diff: int = 1,
    significance: int = 1,  # 0=90%, 1=95%, 2=99%
) -> Dict:
    """
    Johansen cointegration test for N-asset groups.

    Args:
        data: (T, N) array of price series
        det_order: -1=no const, 0=const, 1=const+trend
        k_ar_diff: number of lagged differences
        significance: 0=90%, 1=95%, 2=99%

    Returns:
        Dict with n_coint (number of cointegrating relations),
        trace_stats, crit_values, eigenvectors (hedge ratios)
    """
    try:
        T, N = data.shape
        if T < 30 or N < 2:
            return {"n_coint": 0, "valid": False}

        result = coint_johansen(data, det_order, k_ar_diff)

        # Trace test: compare trace stat to critical values
        trace_stats = result.lr1  # trace statistics
        crit_values = result.cvt[:, significance]  # critical values at chosen level

        # Count cointegrating relationships
        n_coint = 0
        for i in range(N):
            if trace_stats[i] > crit_values[i]:
                n_coint += 1
            else:
                break  # sequential testing stops at first failure

        # Extract cointegrating vectors (normalized)
        eigenvectors = result.evec
        # Normalize first element to 1 for interpretability
        if n_coint > 0 and abs(eigenvectors[0, 0]) > 1e-10:
            primary_vector = eigenvectors[:, 0] / eigenvectors[0, 0]
        else:
            primary_vector = eigenvectors[:, 0] if n_coint > 0 else np.zeros(N)

        return {
            "n_coint": n_coint,
            "valid": True,
            "trace_stats": trace_stats.tolist(),
            "crit_values": crit_values.tolist(),
            "eigenvectors": eigenvectors,
            "primary_vector": primary_vector,
            "eigenvalues": result.eig.tolist(),
        }

    except Exception as e:
        return {"n_coint": 0, "valid": False, "error": str(e)}


# ───────────────────────────────────────────────
# Multi-Timeframe Cointegration Scorer
# ───────────────────────────────────────────────
def multi_timeframe_score(
    s1: np.ndarray,
    s2: np.ndarray,
    timeframes: Dict = None,
    alpha: float = 0.05,
) -> Dict:
    """
    Test cointegration across short/medium/long windows.

    A pair cointegrated across all timeframes gets highest score.
    Score = weighted sum of (1 if cointegrated at timeframe, 0 otherwise)
    Weights: short=0.2, medium=0.4, long=0.4

    Returns:
        Dict with per-timeframe results and composite score
    """
    if timeframes is None:
        timeframes = TIMEFRAMES

    n = min(len(s1), len(s2))
    results = {}
    weights = {"short": 0.2, "medium": 0.4, "long": 0.4}
    composite = 0.0

    for tf_name, tf_config in timeframes.items():
        window = tf_config["days"]
        if n < window:
            # Use all available data if shorter than window
            window = n

        s1_window = s1[-window:]
        s2_window = s2[-window:]

        pval, is_coint = eg_test(s1_window, s2_window, alpha)

        # ADF test on spread for stationarity confirmation
        spread = s1_window - np.polyfit(s2_window, s1_window, 1)[0] * s2_window
        try:
            adf_stat, adf_pval, _, _, _, _ = adfuller(spread)
        except Exception:
            adf_stat, adf_pval = 0.0, 1.0

        results[tf_name] = {
            "pvalue": pval,
            "cointegrated": is_coint,
            "adf_pvalue": float(adf_pval),
            "stationary": adf_pval < alpha,
            "window": window,
        }

        w = weights.get(tf_name, 0.33)
        if is_coint:
            # Score inversely proportional to p-value
            composite += w * (1.0 - pval)

    results["composite_score"] = composite
    results["all_timeframes"] = all(
        r["cointegrated"] for r in results.values() if isinstance(r, dict) and "cointegrated" in r
    )

    return results


# ───────────────────────────────────────────────
# Cross-Asset Cointegration Engine
# ───────────────────────────────────────────────
class MultiCointConfig:
    """Configuration for multi-asset cointegration analysis."""

    def __init__(self):
        self.max_workers = 4             # conservative CPU usage
        self.p_value_threshold = 0.05
        self.cache_dir = "coint_cache"
        self.cache_ttl = 3600 * 24

        # Pair search
        self.max_pairs = 300
        self.min_data_points = 60
        self.pair_timeout = 1500         # max seconds for pair search phase

        # N-asset search
        self.max_group_size = 4          # max N for N-cointegration
        self.max_groups_to_test = 500    # limit combinatorial explosion
        self.johansen_significance = 1   # 95%

        # Multi-timeframe
        self.timeframes = TIMEFRAMES
        self.min_composite_score = 0.3   # minimum to qualify

        # Asset classes
        self.include_sp500 = True
        self.include_crypto = True
        self.crypto_tickers = CRYPTO_UNIVERSE
        self.exclude_tickers = {"BF.B", "BRK.B", "LEN", "ETR"}

        # Resource limits
        self.max_cpu_temp = 80           # pause if GPU/CPU too hot
        self.batch_size = 50             # process pairs in batches


class MultiCointEngine:
    """
    Multi-timeframe, cross-asset cointegration engine.

    Finds:
    1. Cross-asset pairs (stock-stock, crypto-crypto, stock-crypto)
    2. N-asset cointegrated groups via Johansen test
    3. Multi-timeframe stable pairs (short + medium + long)
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        config: Optional[MultiCointConfig] = None,
        data_provider: Optional[MarketDataProvider] = None,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.config = config or MultiCointConfig()
        self.loader = data_provider or FinancialLoaderProvider(FinancialLoader())

        self.price_data: Dict[str, pd.Series] = {}
        self.pair_results: List[Dict] = []
        self.group_results: List[Dict] = []

        os.makedirs(self.config.cache_dir, exist_ok=True)

    def _fetch_prices(self, tickers: List[str]) -> Dict[str, pd.Series]:
        """Fetch adjusted close prices for all tickers."""
        prices = {}

        for ticker in tqdm(tickers, desc="Fetching prices", colour="cyan"):
            try:
                df = self.loader.get_stock_data(
                    ticker, self.start_date, self.end_date,
                )
                if df is not None and len(df) >= self.config.min_data_points:
                    prices[ticker] = df["Close"].dropna()
            except Exception:
                pass

        return prices

    def _align_series(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        """Align price series on common dates with mixed-asset stabilization."""
        series_list = []
        for t in tickers:
            if t in self.price_data:
                series_list.append(self.price_data[t].rename(t))

        if len(series_list) < 2:
            return None

        has_crypto = any("-USD" in t for t in tickers)
        has_equity = any("-USD" not in t for t in tickers)
        if has_crypto and has_equity:
            # Mixed calendars: align to business days and forward-fill short gaps.
            df = pd.concat(series_list, axis=1, join="outer", sort=False).sort_index()
            business_index = pd.date_range(df.index.min(), df.index.max(), freq="B")
            df = df.reindex(business_index).ffill(limit=3).dropna()
        else:
            df = pd.concat(series_list, axis=1, join="inner").dropna()
        if len(df) < self.config.min_data_points:
            return None

        return df

    def _normalize(self, series: pd.Series) -> pd.Series:
        """Normalize to starting value of 1.0."""
        first = series.iloc[0]
        if abs(first) < 1e-10:
            return series
        return series / first

    def _classify_pair(self, t1: str, t2: str) -> str:
        """Classify pair as equity-equity, crypto-crypto, or cross-asset."""
        is_crypto_1 = "-USD" in t1
        is_crypto_2 = "-USD" in t2
        if is_crypto_1 and is_crypto_2:
            return "crypto-crypto"
        elif not is_crypto_1 and not is_crypto_2:
            return "equity-equity"
        else:
            return "cross-asset"

    # ─── Phase 1: Build Universe ───
    def build_universe(self) -> List[str]:
        """Build combined ticker universe."""
        print_section("Building Asset Universe", CYAN)

        tickers = []

        if self.config.include_sp500:
            sp500 = self.loader.get_sp500_tickers()
            sp500 = [t for t in sp500 if t not in self.config.exclude_tickers]
            tickers.extend(sp500)
            buffered_print(f"  S&P 500: {len(sp500)} tickers")

        if self.config.include_crypto:
            tickers.extend(self.config.crypto_tickers)
            buffered_print(f"  Crypto: {len(self.config.crypto_tickers)} tickers")

        tickers = list(set(tickers))
        buffered_print(f"  {BOLD}Total universe: {len(tickers)} tickers{ENDC}")

        # Fetch prices
        self.price_data = self._fetch_prices(tickers)
        available = list(self.price_data.keys())
        buffered_print(f"  Available after fetch: {len(available)} tickers")

        return available

    # ─── Phase 2: Pairwise Cointegration ───
    def find_pairs(
        self,
        tickers: Optional[List[str]] = None,
        max_pairs: Optional[int] = None,
        use_gpu: bool = True,
    ) -> pd.DataFrame:
        """
        Find cointegrated pairs with multi-timeframe scoring.

        When GPU available: fast GPU pass filters ALL pairs first,
        then CPU multi-TF scoring runs only on GPU-identified candidates.
        """
        if tickers is None:
            tickers = list(self.price_data.keys())

        max_pairs = max_pairs or self.config.max_pairs
        print_section("Pairwise Cointegration Search", YELLOW)

        # ── GPU Fast Pass: test ALL pairs in seconds ──
        gpu_candidates = set()
        if use_gpu:
            try:
                from .gpu_coint import gpu_cointegration_search, HAS_CUPY
                if HAS_CUPY:
                    # Build aligned price dict
                    aligned_prices = {}
                    min_len = min(
                        len(self.price_data[t]) for t in tickers if t in self.price_data
                    )
                    for t in tickers:
                        if t in self.price_data and len(self.price_data[t]) >= min_len:
                            vals = self.price_data[t].values if hasattr(self.price_data[t], 'values') else self.price_data[t]
                            # Normalize
                            if abs(vals[0]) > 1e-10:
                                aligned_prices[t] = vals[-min_len:] / vals[-min_len]
                            else:
                                aligned_prices[t] = vals[-min_len:]

                    valid_tickers = list(aligned_prices.keys())
                    n_all_pairs = len(valid_tickers) * (len(valid_tickers) - 1) // 2
                    buffered_print(
                        f"  {CYAN}GPU fast pass: screening {n_all_pairs} pairs...{ENDC}"
                    )

                    gpu_results = gpu_cointegration_search(
                        aligned_prices, valid_tickers, alpha=0.10,  # loose threshold for candidates
                    )

                    for r in gpu_results:
                        gpu_candidates.add((r["ticker1"], r["ticker2"]))
                        gpu_candidates.add((r["ticker2"], r["ticker1"]))

                    buffered_print(
                        f"  {GREEN}GPU found {len(gpu_results)} candidate pairs for multi-TF scoring{ENDC}"
                    )
            except Exception as e:
                buffered_print(f"  {YELLOW}GPU pass failed ({e}), using CPU only{ENDC}")

        # Generate pair candidates (filtered by GPU if available)
        all_pairs = list(itertools.combinations(tickers, 2))

        if gpu_candidates:
            # Only score GPU-identified candidates
            all_pairs = [(a, b) for a, b in all_pairs if (a, b) in gpu_candidates or (b, a) in gpu_candidates]
            buffered_print(f"  Narrowed to {len(all_pairs)} pairs for multi-TF scoring")
        elif len(all_pairs) > max_pairs:
            # Prioritize cross-asset pairs
            cross_pairs = [(a, b) for a, b in all_pairs if self._classify_pair(a, b) == "cross-asset"]
            same_pairs = [(a, b) for a, b in all_pairs if self._classify_pair(a, b) != "cross-asset"]
            np.random.shuffle(same_pairs)
            all_pairs = cross_pairs + same_pairs[:max_pairs - len(cross_pairs)]

        buffered_print(f"  Testing {len(all_pairs)} pairs...")

        def test_pair(t1, t2):
            aligned = self._align_series([t1, t2])
            if aligned is None:
                return None

            s1 = self._normalize(aligned[t1]).values
            s2 = self._normalize(aligned[t2]).values

            tf_result = multi_timeframe_score(
                s1, s2,
                timeframes=self.config.timeframes,
                alpha=self.config.p_value_threshold,
            )

            if tf_result["composite_score"] < self.config.min_composite_score:
                return None

            return {
                "ticker1": t1,
                "ticker2": t2,
                "pair_type": self._classify_pair(t1, t2),
                "composite_score": tf_result["composite_score"],
                "all_timeframes": tf_result["all_timeframes"],
                **{f"{k}_pval": tf_result[k]["pvalue"]
                   for k in TIMEFRAMES if k in tf_result and isinstance(tf_result[k], dict)},
                **{f"{k}_coint": tf_result[k]["cointegrated"]
                   for k in TIMEFRAMES if k in tf_result and isinstance(tf_result[k], dict)},
            }

        # Process in batches with resource monitoring
        all_results = []
        batch_size = self.config.batch_size
        for batch_start in range(0, len(all_pairs), batch_size):
            batch = all_pairs[batch_start:batch_start + batch_size]

            batch_results = Parallel(n_jobs=self.config.max_workers)(
                delayed(test_pair)(t1, t2) for t1, t2 in batch
            )
            all_results.extend(batch_results)

            done = min(batch_start + batch_size, len(all_pairs))
            found = sum(1 for r in all_results if r is not None)
            buffered_print(
                f"  Progress: {done}/{len(all_pairs)} tested, {found} found"
            )

        self.pair_results = [r for r in all_results if r is not None]
        df = pd.DataFrame(self.pair_results)

        if not df.empty:
            df = df.sort_values("composite_score", ascending=False)

        # Summary
        n_total = len(df)
        n_cross = len(df[df["pair_type"] == "cross-asset"]) if not df.empty else 0
        n_crypto = len(df[df["pair_type"] == "crypto-crypto"]) if not df.empty else 0
        n_equity = len(df[df["pair_type"] == "equity-equity"]) if not df.empty else 0
        n_all_tf = len(df[df["all_timeframes"] == True]) if not df.empty else 0

        print_section("Pair Results", GREEN)
        buffered_print(f"  Total cointegrated pairs: {n_total}")
        buffered_print(f"  Equity-Equity: {n_equity}")
        buffered_print(f"  Crypto-Crypto: {n_crypto}")
        buffered_print(f"  {BOLD}Cross-Asset (Stock-Crypto): {n_cross}{ENDC}")
        buffered_print(f"  Cointegrated at ALL timeframes: {n_all_tf}")

        return df

    # ─── Phase 3: N-Asset Cointegration Groups ───
    def find_groups(
        self,
        tickers: Optional[List[str]] = None,
        group_sizes: Optional[List[int]] = None,
        max_groups: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Find N-asset cointegrated groups via Johansen test.

        Tests combinations of 3-4 assets for multi-asset cointegration.
        Prioritizes groups that include both equities and crypto.
        """
        if tickers is None:
            tickers = list(self.price_data.keys())

        group_sizes = group_sizes or list(range(3, self.config.max_group_size + 1))
        max_groups = max_groups or self.config.max_groups_to_test

        print_section("N-Asset Cointegration Search (Johansen)", YELLOW)

        all_groups = []
        for size in group_sizes:
            combos = list(itertools.combinations(tickers, size))
            if len(combos) > max_groups:
                # Prioritize mixed groups (equity + crypto)
                mixed = [g for g in combos if any("-USD" in t for t in g) and any("-USD" not in t for t in g)]
                pure = [g for g in combos if g not in mixed]
                np.random.shuffle(pure)
                combos = mixed[:max_groups // 2] + pure[:max_groups // 2]

            all_groups.extend(combos)
            buffered_print(f"  Size {size}: testing {len(combos)} groups")

        def test_group(group):
            aligned = self._align_series(list(group))
            if aligned is None:
                return None

            # Normalize all series
            norm_data = np.column_stack([
                self._normalize(aligned[t]).values for t in group
            ])

            result = johansen_test(
                norm_data,
                det_order=0,
                k_ar_diff=1,
                significance=self.config.johansen_significance,
            )

            if not result["valid"] or result["n_coint"] == 0:
                return None

            has_crypto = any("-USD" in t for t in group)
            has_equity = any("-USD" not in t for t in group)

            return {
                "tickers": list(group),
                "group_size": len(group),
                "n_coint_relations": result["n_coint"],
                "primary_vector": result["primary_vector"].tolist(),
                "trace_stats": result["trace_stats"],
                "crit_values": result["crit_values"],
                "group_type": "cross-asset" if (has_crypto and has_equity) else (
                    "crypto" if has_crypto else "equity"
                ),
                "max_eigenvalue": max(result["eigenvalues"]),
            }

        results = Parallel(n_jobs=self.config.max_workers)(
            delayed(test_group)(g)
            for g in tqdm(all_groups, desc="Testing groups", colour="yellow")
        )

        self.group_results = [r for r in results if r is not None]
        df = pd.DataFrame(self.group_results)

        if not df.empty:
            df = df.sort_values("n_coint_relations", ascending=False)

        n_total = len(df)
        n_cross = len(df[df["group_type"] == "cross-asset"]) if not df.empty else 0

        print_section("Group Results", GREEN)
        buffered_print(f"  Total cointegrated groups: {n_total}")
        buffered_print(f"  {BOLD}Cross-asset groups: {n_cross}{ENDC}")
        if not df.empty:
            buffered_print(f"  Max cointegrating relations: {df['n_coint_relations'].max()}")

        return df

    # ─── Full Pipeline ───
    def run(
        self,
        find_n_groups: bool = True,
        max_pairs: int = 500,
        max_groups: int = 500,
        n_group_tickers: int = 50,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the full multi-timeframe, cross-asset cointegration pipeline.

        1. Build universe (SP500 + crypto)
        2. Find pairwise cointegration across timeframes
        3. Find N-asset groups via Johansen

        Returns (pairs_df, groups_df)
        """
        print_header("Multi-Timeframe Cross-Asset Cointegration Engine")
        print_gpu_status()

        # Phase 1: Universe
        available = self.build_universe()

        # Phase 2: Pairs
        pairs_df = self.find_pairs(available, max_pairs=max_pairs)

        # Phase 3: N-Asset Groups
        groups_df = pd.DataFrame()
        if find_n_groups:
            # Use top-scoring tickers from pair results for group search
            if not pairs_df.empty:
                top_tickers = set()
                for _, row in pairs_df.head(100).iterrows():
                    top_tickers.add(row["ticker1"])
                    top_tickers.add(row["ticker2"])
                # Always include some crypto
                crypto_in_data = [t for t in available if "-USD" in t]
                top_tickers.update(crypto_in_data[:10])
                group_tickers = list(top_tickers)[:n_group_tickers]
            else:
                group_tickers = available[:n_group_tickers]

            groups_df = self.find_groups(
                group_tickers,
                group_sizes=[3, 4],
                max_groups=max_groups,
            )

        # Save results
        if not pairs_df.empty:
            pairs_df.to_csv("coint_pairs_multi_tf.csv", index=False)
        if not groups_df.empty:
            groups_df.to_csv("coint_groups_johansen.csv", index=False)

        print_section("Pipeline Complete", GREEN)
        print_gpu_status()

        return pairs_df, groups_df

    def get_top_pairs(self, n: int = 20) -> List[Tuple[str, str]]:
        """Get top N pairs by composite score."""
        df = pd.DataFrame(self.pair_results)
        if df.empty:
            return []
        df = df.sort_values("composite_score", ascending=False).head(n)
        return [(row["ticker1"], row["ticker2"]) for _, row in df.iterrows()]

    def get_top_groups(self, n: int = 10) -> List[Dict]:
        """Get top N groups by cointegrating relations."""
        df = pd.DataFrame(self.group_results)
        if df.empty:
            return []
        return df.sort_values(
            ["n_coint_relations", "max_eigenvalue"], ascending=[False, False]
        ).head(n).to_dict("records")


# ───────────────────────────────────────────────
# CLI Entry Point
# ───────────────────────────────────────────────
def main():
    import sys

    config = MultiCointConfig()

    # Parse args
    if "--crypto-only" in sys.argv:
        config.include_sp500 = False
    if "--equity-only" in sys.argv:
        config.include_crypto = False
    if "--fast" in sys.argv:
        config.max_workers = 6

    # Date range
    start = "2023-01-01"
    end = "2025-01-01"
    for arg in sys.argv:
        if arg.startswith("--start="):
            start = arg.split("=")[1]
        elif arg.startswith("--end="):
            end = arg.split("=")[1]

    engine = MultiCointEngine(start, end, config)
    pairs_df, groups_df = engine.run(
        find_n_groups="--no-groups" not in sys.argv,
        max_pairs=300,
        max_groups=300,
    )

    if not pairs_df.empty:
        print(f"\n{BOLD}Top 15 Pairs:{ENDC}")
        cols = ["ticker1", "ticker2", "pair_type", "composite_score", "all_timeframes"]
        available_cols = [c for c in cols if c in pairs_df.columns]
        print(pairs_df[available_cols].head(15).to_string(index=False))

    if not groups_df.empty:
        print(f"\n{BOLD}Top 10 N-Asset Groups:{ENDC}")
        for _, row in groups_df.head(10).iterrows():
            tickers = row["tickers"]
            print(f"  {row['group_type']:>12} | n_coint={row['n_coint_relations']} | {tickers}")


if __name__ == "__main__":
    main()
