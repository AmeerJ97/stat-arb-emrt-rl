from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ..financial_loader import FinancialLoader
from ..multi_coint import MultiCointEngine, MultiCointConfig
from ..printing_system import (
    BOLD, CYAN, ENDC, GREEN, RED, YELLOW,
    buffered_print, print_header, print_section,
)

logger = logging.getLogger(__name__)

# Use most of the available cores
_N_WORKERS = max(1, os.cpu_count() - 2) if os.cpu_count() else 8


@dataclass
class WFAConfig:
    """Configuration for a Walk-Forward Analysis run."""

    overall_start: str = "2020-01-01"
    overall_end: str = "2025-06-09"

    train_months: int = 12
    test_months: int = 3
    step_months: int = 3
    anchored: bool = False

    pair_limit: int = 75
    initial_cash: int = 10000
    use_rl: bool = False
    include_crypto: bool = False
    crypto_only: bool = False

    refit_pairs: bool = True

    output_dir: str = "./wfa_results"


@dataclass
class WindowResult:
    """Results from a single WFA window."""

    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    pairs_discovered: List[Tuple[str, str]] = field(default_factory=list)
    pairs_traded: List[Tuple[str, str]] = field(default_factory=list)

    is_sharpe: float = 0.0
    is_cagr: float = 0.0
    is_max_dd: float = 0.0
    is_final_value: float = 0.0

    oos_sharpe: float = 0.0
    oos_cagr: float = 0.0
    oos_max_dd: float = 0.0
    oos_final_value: float = 0.0
    oos_total_trades: int = 0
    oos_win_rate: float = 0.0
    oos_profit_factor: float = 0.0

    oos_equity_curve: Optional[pd.Series] = None
    oos_daily_returns: Optional[pd.Series] = None
    oos_trades: List[Dict] = field(default_factory=list)

    @property
    def sharpe_decay(self) -> float:
        if self.oos_sharpe == 0:
            return float("inf") if self.is_sharpe > 0 else 0.0
        return self.is_sharpe / self.oos_sharpe


@dataclass
class WFAReport:
    """Aggregated results across all WFA windows."""

    config: WFAConfig
    windows: List[WindowResult] = field(default_factory=list)

    stitched_equity: Optional[pd.Series] = None
    stitched_returns: Optional[pd.Series] = None

    agg_sharpe: float = 0.0
    agg_cagr: float = 0.0
    agg_max_dd: float = 0.0
    agg_total_trades: int = 0
    agg_win_rate: float = 0.0

    avg_sharpe_decay: float = 0.0
    pair_turnover_rate: float = 0.0


class WalkForwardEngine:
    """Walk-Forward Analysis orchestrator.

    Pre-fetches all price data ONCE for the full date range, then slices
    per window. Each window runs GPU-accelerated pair discovery on the
    pre-loaded data (no re-downloading).
    """

    def __init__(self, config: WFAConfig) -> None:
        self.config = config
        self.loader = FinancialLoader()
        self.results: List[WindowResult] = []
        # Pre-fetched price data: {ticker: pd.Series of Close prices}
        self._full_price_data: Dict[str, pd.Series] = {}
        self._universe_tickers: List[str] = []

    def _prefetch_all_data(self) -> None:
        """Download ALL ticker data in ONE bulk yfinance call.

        Uses yf.download() with all tickers at once (internally threaded),
        then stores Close prices for slicing per window.
        """
        import yfinance as yf

        # Compute the earliest date we'll ever need (warmup for first OOS window)
        earliest_warmup = (
            datetime.strptime(self.config.overall_start, "%Y-%m-%d")
            - relativedelta(months=14)
        ).strftime("%Y-%m-%d")
        fetch_start = min(earliest_warmup, self.config.overall_start)
        fetch_end = self.config.overall_end

        print_section("Pre-fetching ALL data (bulk download)", CYAN)

        # Build ticker universe
        if self.config.crypto_only:
            from ..multi_coint import CRYPTO_UNIVERSE
            tickers = list(CRYPTO_UNIVERSE)
        else:
            sp500 = self.loader.get_sp500_tickers()
            exclude = {"BF.B", "BRK.B", "LEN", "ETR"}
            tickers = [t for t in sp500 if t not in exclude]
            if self.config.include_crypto or self.config.crypto_only:
                from ..multi_coint import CRYPTO_UNIVERSE
                tickers = list(set(tickers + list(CRYPTO_UNIVERSE)))

        self._universe_tickers = tickers
        buffered_print(
            f"  {len(tickers)} tickers | {fetch_start} → {fetch_end}"
        )

        # SINGLE bulk download — yfinance threads internally
        raw = yf.download(
            tickers,
            start=fetch_start,
            end=fetch_end,
            group_by="ticker",
            auto_adjust=True,
            progress=True,
            threads=True,
        )

        # Extract Close prices per ticker
        loaded = 0
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    close = raw["Close"].dropna()
                else:
                    close = raw[ticker]["Close"].dropna()
                if len(close) >= 60:
                    self._full_price_data[ticker] = close
                    loaded += 1
            except (KeyError, TypeError):
                continue

        buffered_print(
            f"{GREEN}  Bulk-downloaded {loaded}/{len(tickers)} tickers "
            f"in one call. Data cached in memory.{ENDC}"
        )

    def _slice_price_data(
        self, start_date: str, end_date: str
    ) -> Dict[str, pd.Series]:
        """Slice pre-fetched data to a specific date window."""
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        sliced = {}
        for ticker, series in self._full_price_data.items():
            s = series[(series.index >= start_ts) & (series.index <= end_ts)]
            if len(s) >= 60:  # min_data_points
                sliced[ticker] = s
        return sliced

    def _gpu_discover_pairs(
        self, train_start: str, train_end: str
    ) -> List[Tuple[str, str]]:
        """Run GPU cointegration directly on pre-fetched data.

        Skips the slow CPU multi-timeframe scoring entirely.
        The GPU batch ADF test at 180K pairs/sec is sufficient
        for WFA pair selection — multi-TF is for the discovery report,
        not for per-window pair selection in WFA.
        """
        from ..gpu_coint import gpu_cointegration_search, HAS_CUPY

        sliced = self._slice_price_data(train_start, train_end)
        tickers = list(sliced.keys())
        buffered_print(f"  {len(tickers)} tickers in window")

        if not HAS_CUPY:
            # Fallback: use CPU Engle-Granger via MultiCointEngine
            buffered_print(f"{YELLOW}  CuPy unavailable, falling back to CPU{ENDC}")
            mc_config = MultiCointConfig()
            mc_config.max_workers = _N_WORKERS
            mc_config.batch_size = 500
            mc_engine = MultiCointEngine(
                start_date=train_start, end_date=train_end, config=mc_config,
            )
            mc_engine.price_data = sliced
            pairs_df = mc_engine.find_pairs(tickers, max_pairs=self.config.pair_limit * 4, use_gpu=False)
            if pairs_df.empty:
                return []
            return [(r.ticker1, r.ticker2) for _, r in pairs_df.iterrows()]

        # Build aligned price dict for GPU (normalize to 1.0)
        min_len = min(len(sliced[t]) for t in tickers)
        aligned_prices = {}
        for t in tickers:
            vals = sliced[t].values
            if len(vals) >= min_len and abs(vals[0]) > 1e-10:
                aligned_prices[t] = vals[-min_len:] / vals[-min_len]

        valid_tickers = list(aligned_prices.keys())
        n_pairs = len(valid_tickers) * (len(valid_tickers) - 1) // 2
        buffered_print(f"  GPU screening {n_pairs} pairs...")

        # GPU batch cointegration — ~180K pairs/sec
        gpu_results = gpu_cointegration_search(
            aligned_prices, valid_tickers, alpha=0.05,
        )

        if not gpu_results:
            buffered_print(f"{YELLOW}  No cointegrated pairs found.{ENDC}")
            return []

        # Sort by p-value (lowest = strongest cointegration), take top N
        gpu_results.sort(key=lambda r: r["pvalue"])
        top = gpu_results[: self.config.pair_limit * 4]

        pairs = [(r["ticker1"], r["ticker2"]) for r in top]
        buffered_print(
            f"{GREEN}  GPU found {len(gpu_results)} cointegrated pairs, "
            f"using top {len(pairs)}{ENDC}"
        )
        return pairs

    def generate_windows(self) -> List[Tuple[str, str, str, str]]:
        """Generate (train_start, train_end, test_start, test_end) tuples."""
        overall_start = datetime.strptime(self.config.overall_start, "%Y-%m-%d")
        overall_end = datetime.strptime(self.config.overall_end, "%Y-%m-%d")
        anchor_start = overall_start

        windows: List[Tuple[str, str, str, str]] = []
        window_start = overall_start

        while True:
            train_start = anchor_start if self.config.anchored else window_start
            train_end = window_start + relativedelta(months=self.config.train_months)
            test_start = train_end
            test_end = test_start + relativedelta(months=self.config.test_months)

            if test_end > overall_end:
                break

            windows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))

            window_start += relativedelta(months=self.config.step_months)

        return windows

    def run_window(
        self,
        window_id: int,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ) -> WindowResult:
        """Execute a single WFA window: discover pairs IS, backtest IS + OOS."""
        from ..backtest.backtest_engine import run_backtest_headless

        result = WindowResult(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        # Step 1: Pair discovery on training window (GPU-accelerated)
        # Uses pre-fetched data sliced to window — NO Yahoo Finance downloads.
        # Goes straight to GPU cointegration search, skipping slow CPU multi-TF scoring.
        buffered_print(
            f"\n{BOLD}{CYAN}[Window {window_id}] "
            f"Discovering pairs (GPU): {train_start} → {train_end}{ENDC}"
        )
        try:
            pairs = self._gpu_discover_pairs(train_start, train_end)
        except Exception as e:
            buffered_print(
                f"{RED}[Window {window_id}] Pair discovery failed: {e}{ENDC}",
                "ERROR",
            )
            pairs = []

        pairs = pairs[: self.config.pair_limit]
        result.pairs_discovered = pairs

        if not pairs:
            buffered_print(
                f"{YELLOW}[Window {window_id}] No pairs found, skipping.{ENDC}",
                "WARNING",
            )
            result.oos_final_value = self.config.initial_cash
            return result

        # Step 2: In-sample backtest (for overfitting detection)
        buffered_print(
            f"{CYAN}[Window {window_id}] "
            f"IS backtest: {train_start} → {train_end}{ENDC}"
        )
        is_metrics = run_backtest_headless(
            loader=self.loader,
            pairs=pairs,
            start_date=train_start,
            end_date=train_end,
            initial_cash=self.config.initial_cash,
            use_rl=self.config.use_rl,
        )
        result.is_sharpe = is_metrics.get("sharpe", 0.0)
        result.is_cagr = is_metrics.get("cagr", 0.0)
        result.is_max_dd = is_metrics.get("max_drawdown", 0.0)
        result.is_final_value = is_metrics.get("final_value", self.config.initial_cash)

        # Step 3: Out-of-sample backtest with IS-fitted pairs
        # Prepend warm-up data so strategy has M=170 bars for initialization.
        warmup_start = (
            datetime.strptime(test_start, "%Y-%m-%d")
            - relativedelta(months=14)
        ).strftime("%Y-%m-%d")

        buffered_print(
            f"{CYAN}[Window {window_id}] "
            f"OOS backtest: {test_start} → {test_end} "
            f"(data from {warmup_start} for warm-up){ENDC}"
        )
        oos_metrics = run_backtest_headless(
            loader=self.loader,
            pairs=pairs,
            start_date=warmup_start,
            end_date=test_end,
            initial_cash=self.config.initial_cash,
            use_rl=self.config.use_rl,
            metrics_start_date=test_start,
        )

        result.oos_sharpe = oos_metrics.get("sharpe", 0.0)
        result.oos_cagr = oos_metrics.get("cagr", 0.0)
        result.oos_max_dd = oos_metrics.get("max_drawdown", 0.0)
        result.oos_final_value = oos_metrics.get("final_value", self.config.initial_cash)
        result.oos_total_trades = oos_metrics.get("total_trades", 0)
        result.oos_win_rate = oos_metrics.get("win_rate", 0.0)
        result.oos_profit_factor = oos_metrics.get("profit_factor", 0.0)
        result.oos_equity_curve = oos_metrics.get("equity_curve")
        result.oos_daily_returns = oos_metrics.get("daily_returns")
        result.oos_trades = oos_metrics.get("trades", [])
        result.pairs_traded = oos_metrics.get("pairs_traded", [])

        buffered_print(
            f"{GREEN}[Window {window_id}] "
            f"IS Sharpe={result.is_sharpe:.2f} | "
            f"OOS Sharpe={result.oos_sharpe:.2f} | "
            f"OOS Final=${result.oos_final_value:,.2f} | "
            f"Trades={result.oos_total_trades}{ENDC}"
        )

        return result

    def run(self) -> WFAReport:
        """Execute the full WFA pipeline."""
        print_header("Walk-Forward Analysis")

        windows = self.generate_windows()
        n_windows = len(windows)
        mode = "anchored (expanding)" if self.config.anchored else "sliding"

        buffered_print(
            f"{BOLD}Configuration:{ENDC}\n"
            f"  Date range: {self.config.overall_start} → {self.config.overall_end}\n"
            f"  Train: {self.config.train_months}mo | "
            f"Test: {self.config.test_months}mo | "
            f"Step: {self.config.step_months}mo\n"
            f"  Mode: {mode}\n"
            f"  Windows: {n_windows}\n"
            f"  Pair limit: {self.config.pair_limit}\n"
            f"  RL: {self.config.use_rl}\n"
            f"  Workers: {_N_WORKERS}\n"
        )

        if n_windows == 0:
            buffered_print(
                f"{RED}No valid windows generated. "
                f"Check date range and window sizes.{ENDC}",
                "ERROR",
            )
            return WFAReport(config=self.config)

        # ONE-TIME data download for the entire date range
        self._prefetch_all_data()

        # Run each window sequentially (Cerebro is not thread-safe)
        for i, (ts, te, os_, oe) in enumerate(windows):
            print_section(f"Window {i + 1}/{n_windows}", CYAN)
            window_result = self.run_window(i + 1, ts, te, os_, oe)
            self.results.append(window_result)
            gc.collect()

        # Build report
        report = WFAReport(config=self.config, windows=self.results)
        report.stitched_equity = self._stitch_equity_curves()
        report.stitched_returns = (
            report.stitched_equity.pct_change().dropna()
            if report.stitched_equity is not None and len(report.stitched_equity) > 1
            else None
        )

        agg = self._compute_aggregate_metrics(report.stitched_equity)
        report.agg_sharpe = agg.get("sharpe", 0.0)
        report.agg_cagr = agg.get("cagr", 0.0)
        report.agg_max_dd = agg.get("max_drawdown", 0.0)
        report.agg_total_trades = sum(w.oos_total_trades for w in self.results)

        total_closed = sum(
            w.oos_total_trades for w in self.results if w.oos_total_trades > 0
        )
        if total_closed > 0:
            report.agg_win_rate = (
                sum(w.oos_win_rate * w.oos_total_trades for w in self.results)
                / total_closed
            )

        report.pair_turnover_rate = self._compute_pair_turnover()

        valid_decays = [
            w.sharpe_decay for w in self.results
            if w.sharpe_decay != float("inf") and w.oos_total_trades > 0
        ]
        report.avg_sharpe_decay = (
            sum(valid_decays) / len(valid_decays) if valid_decays else 0.0
        )

        return report

    def _stitch_equity_curves(self) -> Optional[pd.Series]:
        """Chain OOS equity curves end-to-end, rescaling each to continue
        from where the previous window ended."""
        stitched_parts: List[pd.Series] = []
        carry_forward = float(self.config.initial_cash)

        for wr in self.results:
            curve = wr.oos_equity_curve
            if curve is None or curve.empty:
                continue

            start_val = float(curve.iloc[0])
            if start_val == 0:
                continue

            scale = carry_forward / start_val
            scaled = curve * scale
            stitched_parts.append(scaled)
            carry_forward = float(scaled.iloc[-1])

        if not stitched_parts:
            return None

        stitched = pd.concat(stitched_parts)
        # Remove duplicate indices (overlap at window boundaries)
        stitched = stitched[~stitched.index.duplicated(keep="last")]
        return stitched.sort_index()

    def _compute_aggregate_metrics(
        self, stitched: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Compute Sharpe, CAGR, MaxDD from the stitched OOS equity curve."""
        if stitched is None or len(stitched) < 2:
            return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}

        returns = stitched.pct_change().dropna()

        # Sharpe (annualized, assuming 252 trading days)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # CAGR
        total_days = (stitched.index[-1] - stitched.index[0]).days
        if total_days > 0 and float(stitched.iloc[0]) > 0:
            total_return = float(stitched.iloc[-1]) / float(stitched.iloc[0])
            years = total_days / 365.25
            cagr = total_return ** (1.0 / years) - 1.0 if years > 0 else 0.0
        else:
            cagr = 0.0

        # Max drawdown
        cummax = stitched.cummax()
        drawdown = (stitched - cummax) / cummax
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": max_dd}

    def _compute_pair_turnover(self) -> float:
        """Average Jaccard distance between consecutive windows' pair sets."""
        if len(self.results) < 2:
            return 0.0

        distances: List[float] = []
        for i in range(1, len(self.results)):
            prev = set(self.results[i - 1].pairs_discovered)
            curr = set(self.results[i].pairs_discovered)
            union = prev | curr
            if not union:
                continue
            intersection = prev & curr
            distances.append(1.0 - len(intersection) / len(union))

        return sum(distances) / len(distances) if distances else 0.0
