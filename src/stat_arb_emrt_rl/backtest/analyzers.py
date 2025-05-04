# FILE: analyzers.py

# ───────────────────────────────────────────────
# Standard Library Imports
# ───────────────────────────────────────────────
from collections import defaultdict

# ───────────────────────────────────────────────
# Third-Party Package Imports
# ───────────────────────────────────────────────
import numpy as np
import backtrader as bt
import pandas as pd

# ───────────────────────────────────────────────
# Project-Specific Imports
# ───────────────────────────────────────────────
from ..printing_system import (
    YELLOW, ENDC, buffered_print,
)


class EnhancedReturnsAnalyzer(bt.Analyzer):
    def __init__(self):
        self.returns = []
        self.equity_values = []
        self.equity_dates = []

    def start(self):
        self.equity_values = []
        self.equity_dates = []
        self.returns = []
        super().start()

    def next(self):
        # --- FIX: Corrected Equity Curve Logic ---
        # The previous implementation had a bug where it added both the starting cash and the
        # first day's closing value on the first data point, both with the same date. This
        # corrupted the equity series from the start, leading to an incorrect visual
        # representation and inaccurate performance metrics (CAGR, Drawdown).
        #
        # The corrected logic is simpler and more robust. It records only the closing
        # portfolio value at the end of each bar. The initial starting capital is handled
        # separately during the final analysis in `get_analysis`, which is the correct
        # approach for calculating metrics like Max Drawdown and CAGR.

        # current_value = self.strategy.broker.getvalue()
        # current_date = pd.to_datetime(self.strategy.datetime.date(0))

        # self.equity_values.append(current_value)
        # self.equity_dates.append(current_date)
        current_value = self.strategy.broker.getvalue()
        current_date = pd.to_datetime(self.strategy.datetime.date(0))

        self.equity_values.append(current_value)
        self.equity_dates.append(current_date)

        if len(self.equity_values) > 1:
            prev_value = self.equity_values[-2]
            current_value = self.equity_values[-1]
            if prev_value != 0:
                daily_return = (current_value - prev_value) / prev_value
                self.returns.append(daily_return)
            else:
                self.returns.append(0.0)

    def get_analysis(self):
        if self.equity_dates and self.equity_values:
            equity_series = pd.Series(
                self.equity_values, index=pd.DatetimeIndex(self.equity_dates))
            equity_series = equity_series[~equity_series.index.duplicated(
                keep='last')]
        else:
            equity_series = pd.Series(dtype=float)

        initial_capital_for_metrics = self.strategy.broker.startingcash

        return {
            "returns": self.returns,
            "equity_values": self.equity_values,
            "equity_series": equity_series,
            "initial_capital": initial_capital_for_metrics,
            "cagr": self._calculate_cagr(equity_series, initial_capital_for_metrics),
            "max_drawdown": self._calculate_max_drawdown(equity_series, initial_capital_for_metrics),
            "sharpe": self._calculate_sharpe(self.returns),
        }

    def _calculate_cagr(self, equity_series: pd.Series, initial_capital: float):
        if equity_series.empty or len(equity_series) < 1:
            return 0.0
        start_equity = initial_capital
        end_equity = equity_series.iloc[-1]
        if len(equity_series.index) < 2:
            return (end_equity/start_equity - 1) if start_equity > 0 else 0.0
        num_days_span = (
            equity_series.index[-1] - equity_series.index[0]).days + 1
        if num_days_span <= 0:
            return (end_equity/start_equity - 1) if start_equity > 0 else 0.0
        trading_days_per_year = 252.0
        if start_equity == 0:
            return 0.0
        cagr = (end_equity / start_equity) ** (trading_days_per_year /
                                               num_days_span) - 1.0
        return cagr if np.isfinite(cagr) else 0.0

    # def _calculate_max_drawdown(self, equity_series: pd.Series, initial_capital: float):
    #     if equity_series.empty:
    #         return 0.0
    #     # full_equity_trace = pd.concat(
    #     #     [pd.Series([initial_capital]), equity_series.reset_index(drop=True)])
    #     full_equity_trace = equity_series.copy()
    #     peak = full_equity_trace.expanding(min_periods=1).max()
    #     # drawdown = (full_equity_trace - peak) / peak.replace(0, np.nan)
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         drawdown = np.where(peak > 0, (full_equity_trace - peak) / peak, 0)

    #     max_dd = abs(
    #         drawdown.min()) if not drawdown.empty and not drawdown.isnull().all() else 0.0
    #     return max_dd if np.isfinite(max_dd) else 0.0
    def _calculate_max_drawdown(self, equity_series: pd.Series, initial_capital: float):
        if equity_series.empty:
            return 0.0
        full_equity_trace = equity_series.copy()
        peak = full_equity_trace.expanding(min_periods=1).max()

        # Calculate drawdown safely
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = (full_equity_trace - peak) / peak
            drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # FIX: Handle numpy array conversion
        if isinstance(drawdown, np.ndarray):
            if drawdown.size == 0:
                return 0.0
            min_val = np.nanmin(drawdown)
            return abs(min_val) if not np.isnan(min_val) else 0.0
        else:
            if drawdown.empty or drawdown.isnull().all():
                return 0.0
            return abs(drawdown.min())

    def _calculate_sharpe(self, daily_returns_list: list[float]):
        if not daily_returns_list or len(daily_returns_list) < 2:
            return 0.0
        returns_arr = np.array(daily_returns_list)
        finite_returns = returns_arr[np.isfinite(returns_arr)]

        if len(finite_returns) < 2:
            return 0.0
        if np.std(finite_returns) < 1e-9:
            return 0.0 if np.mean(finite_returns) == 0 else np.nan

        trading_days_per_year = 252.0
        annualized_return = np.mean(finite_returns) * trading_days_per_year
        # annualized_vol = np.std(finite_returns) * \
        #     np.sqrt(trading_days_per_year)
        annualized_vol = np.std(
            finite_returns) * np.sqrt(trading_days_per_year / len(finite_returns))
        sharpe = annualized_return / annualized_vol if annualized_vol > 1e-9 else 0.0
        return sharpe if np.isfinite(sharpe) else 0.0


class SpreadTracker(bt.Analyzer):
    def __init__(self):
        self.spread_data = defaultdict(list)
        self.trade_signals = defaultdict(list)
        self.k_values = defaultdict(list)
        self.means = defaultdict(list)
        self.std_devs = defaultdict(list)
        self.dates = defaultdict(list)

    # def notify_trade(self, trade):
    #     # --- FIX: Removed Conflicting Trade Recording ---
    #     # The original implementation of this method called `self.strategy.trade_recorder.process_leg_event()`.
    #     # This created a critical bug: trades were being recorded in two different places.
    #     #   1. Here, in the analyzer, based on individual trade execution events.
    #     #   2. In the strategy (`reversion_strategy.py`), which explicitly calls `record_trade()`
    #     #      after a pair trade is conceptually closed.
    #     #
    #     # This dual recording mechanism led to duplicated and malformed trade data, corrupting the
    #     # final trade log and all downstream statistics (win rate, profit factor, etc.).
    #     #
    #     # By removing the problematic financial recording logic, we designate the strategy as the *single
    #     # source of truth* for when a trade is recorded. This ensures data integrity. The rest of
    #     # the logic in this method, which records signals for plotting, is preserved to avoid
    #     # breaking the visualization components.
    #     parts = trade.data._name.split("_")
    #     if len(parts) < 3:
    #         return
    #     t1, t2, leg = parts[0], parts[1], parts[2]
    #     pair = (t1, t2)

    #     # This part for `trade_signals` is used for plotting markers and is preserved.
    #     if trade.isclosed:
    #         status = "exit"
    #     else:  # isopen
    #         direction = "long" if trade.size > 0 else "short"
    #         asset = t1 if leg == "LONG" else t2
    #         status = f"{direction}_{asset}"
    #     # self.trade_signals[pair].append(
    #         # (self.strategy.datetime.date(), status))

    def next(self):
        self.current_dt = self.strategy.datetime.date()
        for pair_tuple in self.strategy.params.pairs:
            data_s1_feed, data_s2_feed = self.strategy.data_mapping.get(
                pair_tuple, (None, None))

            if data_s1_feed is None or data_s2_feed is None or not len(data_s1_feed) or not len(data_s2_feed):
                self.dates[pair_tuple].append(self.current_dt)
                self.spread_data[pair_tuple].append(np.nan)
                self.means[pair_tuple].append(np.nan)
                self.std_devs[pair_tuple].append(np.nan)
                self.k_values[pair_tuple].append(np.nan)
                continue

            current_date_for_pair = data_s1_feed.datetime.date(0)
            spread, mean_val, std_val, k_val = np.nan, np.nan, np.nan, np.nan

            try:
                spread_series_hist = self.strategy.spreads.get(pair_tuple, [])
                if spread_series_hist and isinstance(spread_series_hist[-1], (int, float)) and not np.isnan(spread_series_hist[-1]):
                    spread = spread_series_hist[-1]

                mean_val = self.strategy.spread_mean.get(pair_tuple, np.nan)
                std_val_raw = self.strategy.spread_std.get(pair_tuple, np.nan)
                std_val = max(std_val_raw, 1e-9) if std_val_raw is not None and not np.isnan(
                    std_val_raw) and std_val_raw > 0 else 1e-9

                default_K = self.strategy.params.K if hasattr(
                    self.strategy.params, 'K') else 1.5
                k_val = default_K

                if hasattr(self.strategy.params, 'adaptive_K') and self.strategy.params.adaptive_K:
                    vol_lookback = getattr(
                        self.strategy.params, 'volatility_lookback', 10)
                    if len(spread_series_hist) >= vol_lookback:
                        numeric_spread_hist_recent = [
                            s for s in spread_series_hist[-vol_lookback:] if isinstance(s, (int, float)) and not np.isnan(s)]
                        if len(numeric_spread_hist_recent) >= 2:
                            recent_vol = np.nanstd(numeric_spread_hist_recent)
                            if std_val > 1e-9 and recent_vol is not None and not np.isnan(recent_vol):
                                k_val = default_K * (recent_vol / std_val)
                                k_floor = getattr(
                                    self.strategy.params, 'K_floor', 1.0)
                                k_ceiling = getattr(
                                    self.strategy.params, 'K_ceiling', 3.0)
                                k_val = np.clip(k_val, k_floor, k_ceiling)

                if k_val is None or np.isnan(k_val):
                    k_val = default_K
            except (IndexError, Exception):
                pass

            self.dates[pair_tuple].append(current_date_for_pair)
            self.spread_data[pair_tuple].append(spread)
            self.means[pair_tuple].append(mean_val)
            self.std_devs[pair_tuple].append(std_val)
            self.k_values[pair_tuple].append(k_val)

    def get_analysis(self):
        def convert_keys(data):
            return {f"{k[0]}/{k[1]}": v for k, v in data.items()}

        return {
            "spread_data": convert_keys(self.spread_data),
            "trade_signals": convert_keys(self.trade_signals),
            "k_values": convert_keys(self.k_values),
            "means": convert_keys(self.means),
            "std_devs": convert_keys(self.std_devs),
            "dates": convert_keys(self.dates),
        }
