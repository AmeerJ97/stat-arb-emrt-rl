# ───────────────────────────────────────────────
# Standard Library Imports
# ───────────────────────────────────────────────
import sys
import math
import time
import logging
import shutil
import warnings
import multiprocessing
from time import sleep
from datetime import timedelta
from collections import defaultdict
from queue import Queue, Empty
from threading import Thread

# ───────────────────────────────────────────────
# Third-Party Package Imports
# ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import backtrader as bt
from numba import jit
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import coint, adfuller
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ───────────────────────────────────────────────
# Project-Specific Imports
# ───────────────────────────────────────────────
from .trade_recorder import EnhancedTradeRecorder
from .printing_system import (
    GREEN,
    RED,
    YELLOW,
    CYAN,
    PURPLE,
    ORANGE,
    BLUE,
    BOLD,
    UNDERLINE,
    ENDC,
    StructuredMessage,
    print_header,
    print_trade,
    print_trade_exit,
    buffered_print,
    print_centered,
    format_drawdown_warning,
    format_volatility_alert,
    StrategyColors,
    TradeVisuals,
    print_section,
    # flush_queue,
    MAX_LINE_WIDTH,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


# Update the calculate_pair_stats function definition
def calculate_pair_stats(
    pair,
    historical_spreads,
    hedge_ratio,
    short_window,
    long_window,
    s1_price,
    s2_price,
):
    """Calculate pair statistics with error handling"""
    try:
        # Validate inputs
        # if not np.isfinite(s1_price) or not np.isfinite(s2_price):
        #     return (pair, None, None, None, None, None, historical_spreads)
        spread = s1_price - hedge_ratio * s2_price
        updated_spreads = historical_spreads + [spread]

        if len(updated_spreads) < long_window:
            return (pair, spread, None, None, None, 1e-5, updated_spreads)

        spread_window = np.array(updated_spreads[-long_window:])
        valid_spreads = spread_window[np.isfinite(spread_window)]

        if len(valid_spreads) < 2:
            return (pair, spread, None, None, None, 1e-5, updated_spreads)

        window_mean = np.nanmean(valid_spreads)
        window_std = max(np.nanstd(valid_spreads, ddof=1), 1e-5)
        z_score = (spread - window_mean) / window_std

        return (
            pair,
            spread,
            z_score,
            window_mean,
            np.nanmean(valid_spreads[-short_window:]),
            window_std,
            updated_spreads,
        )

    except Exception as e:
        buffered_print(f"Error in calculate_pair_stats for {pair}: {str(e)}")
        return (
            pair,
            spread if "spread" in locals() else 0,
            None,
            None,
            None,
            1e-5,
            historical_spreads,
        )


def is_cointegrated(s1_prices, s2_prices, window=90, alpha=0.055):
    """Enhanced cointegration check with threshold validation"""
    if len(s1_prices) < window or len(s2_prices) < window:
        return False

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # if s1_prices == s2_prices:
        if np.array_equal(s1_prices, s2_prices):
            # buffered_print(f"Coint warning check: {np.array_equal(s1_prices, s2_prices)}\n")
            buffered_print(
                f"Coint warning check: {np.array_equal(s1_prices, s2_prices)}"
            )
            return False
    try:
        _, pvalue, _ = coint(s1_prices[-window:], s2_prices[-window:])
        return pvalue < alpha
    except Exception as e:
        # buffered_print(f"Cointegration test exception: {e}.")
        buffered_print(f"Cointegration test exception: {e}.")
        return False


def cointegration_worker(pair, s1, s2, window=90):
    """Worker function for parallel cointegration checks"""
    return pair if is_cointegrated(s1, s2, window=window) else None


# @jit(nopython=True)
# def hurst(ts):
#     """Calculate Hurst exponent for trend detection"""
#     if len(ts) < 20 or np.all(ts == ts[0]):
#         return 0.5  # Neutral value for small samples
#     lags = range(2, 20)
#     tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
#     poly = np.polyfit(np.log(lags), np.log(tau), 1)
#     return poly[0] * 2.0


@jit(nopython=True)
def hurst(ts):
    if len(ts) < 20 or np.all(ts == ts[0]):
        return 0.5

    max_lag = min(20, len(ts) - 1)  # Ensure lags don't exceed data length
    if max_lag < 2:
        return 0.5
    lags = np.arange(2, max_lag)

    # lags = np.arange(2, 20)
    tau = np.empty(len(lags))

    # Compute tau without list comprehension
    for i in range(len(lags)):
        lag = lags[i]
        tau[i] = np.std(ts[lag:] - ts[:-lag])

    # Manual linear regression for slope
    log_lags = np.log(lags)
    log_tau = np.log(tau)
    n = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_tau)
    sum_xy = np.sum(log_lags * log_tau)
    sum_x2 = np.sum(log_lags**2)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

    return slope * 2.0


def is_trending(spread_series, lookback=90, threshold=0.7):
    """Determine if spread is trending using Hurst exponent"""
    if len(spread_series) < lookback:
        return False
    h = hurst(spread_series[-lookback:])
    return h > threshold


def estimate_half_life(spread: pd.Series) -> float:
    if len(spread) < 20:
        return np.inf  # Not enough data
    spread_lag = spread.shift(1).dropna()
    delta_spread = spread.diff().dropna()
    beta = np.polyfit(spread_lag, delta_spread, 1)[0]
    half_life = -np.log(2) / beta if beta < -1e-5 else np.inf
    return half_life


class MeanReversionStrategy(bt.Strategy):
    params = dict(
        pairs=[],
        rebalance_period=20,
        K=1.55,
        K_floor=1.25,
        K_ceiling=2.5,
        adaptive_K=True,
        risk=0.050,
        stop_loss=0.8,
        take_profit=2,
        M=170,
        short_window=3,
        long_window=20,
        max_workers=23,
        volatility_lookback=10,
        tcs_max=0.1,
        liquidation_threshold=-0.25,  # -25% PnL threshold
        trend_lookback=150,
        hurst_threshold=0.7,
        verbose=False,
        cointegration_lookback=150,
        # RL agent integration
        use_rl=False,              # Enable RL-based trading signals
        rl_training_paths=500,     # OU paths for RL training
        rl_training_epochs=3,      # Training epochs
        rl_lookback=4,             # State lookback window
        rl_threshold_k=3.0,       # State discretization threshold
    )

    def __init__(self):
        from .financial_loader import FinancialLoader
        from .ou_optimizer import OUOptimizer

        self.valid_pairs = set(self.params.pairs)  # type: ignore

        rl_mode = " [RL-Enhanced]" if self.params.use_rl else ""
        print_header(
            message=f"Mean Reversion Strategy Initialized with {len(self.params.pairs)} pairs{rl_mode}"
        )  # type: ignore

        # Core components
        self.loader = FinancialLoader()
        self.optimizer = OUOptimizer()
        self.data_mapping = {}
        for i, pair in enumerate(self.params.pairs):
            self.data_mapping[pair] = (
                self.datas[i * 2], self.datas[i * 2 + 1])

        self.pool = multiprocessing.Pool(self.params.max_workers)

        # RL agent integration
        self.rl_agents = {}       # {pair: TabularQAgent}
        self.rl_spreads = {pair: [] for pair in self.params.pairs}
        self.rl_ou_params = {}    # {pair: {mu, theta, sigma}}
        if self.params.use_rl:
            buffered_print(f"{CYAN}RL mode enabled: agents will be trained at first rebalance{ENDC}")

        # Initialize data structures
        self._init_data_structures()

        self._spread_analyzer = (
            self.analyzers.spread_tracker
            if hasattr(self.analyzers, "spread_tracker")
            else None
        )

        self.trade_recorder = EnhancedTradeRecorder()
        self._init_bar_shown = False
        self._init_pbar = None
        self.last_processed_date = None

        self.last_opened_spread_direction = {}

    def stop(self):
        from .printing_system import shutdown_printing

        # flush_queue()  # Flush remaining messages
        # shutdown_printing()  # Signal printer thread to stop
        if hasattr(self, "pool") and self.pool:
            self.pool.close()
            self.pool.join()
        super().stop()

    def _init_data_structures(self):
        """Initialize all tracking data structures"""

        self.start_date = None
        self.rebalance_counter = 0

        # Pair tracking
        self.spreads = {pair: [] for pair in self.params.pairs}
        self.hedge_ratios = {pair: 1.0 for pair in self.params.pairs}
        self.weights = {
            pair: 1.0 / len(self.params.pairs) for pair in self.params.pairs
        }
        self.price_history = {
            pair: {"s1": [], "s2": [], "dates": []} for pair in self.params.pairs
        }

        # Pairs Stats
        # self.spread_mean = {pair: 0.0 for pair in self.params.pairs}
        self.spread_mean = {pair: np.nan for pair in self.params.pairs}
        self.spread_std = {pair: 0.0 for pair in self.params.pairs}
        self.z_scores = {pair: 0.0 for pair in self.params.pairs}

        # Position tracking
        self.pair_positions = {pair: 0 for pair in self.params.pairs}
        self.entry_spreads = {pair: 0 for pair in self.params.pairs}
        self.entry_dates = {pair: None for pair in self.params.pairs}
        self.entry_hedge_ratios = {pair: 1.0 for pair in self.params.pairs}

        # Performance tracking
        self.trade_tcs = defaultdict(dict)
        self.exit_reasons = {}
        self.last_trade_date = None

        self.cum_negative_days = 0
        self.pnl_liquidation_counter = {pair: 0 for pair in self.params.pairs}
        self.invalid_pairs = []
        self.blacklisted_pairs = []

        self.entry_s1_price = {pair: 0.0 for pair in self.params.pairs}
        self.entry_s2_price = {pair: 0.0 for pair in self.params.pairs}
        self.entry_value = {pair: 0.0 for pair in self.params.pairs}

        self.active_trades = {}  # Format: {pair: {entry_data}}
        self.trade_history = defaultdict(list)  # Stores completed trades

    def record_trade_signal(self, pair, date, signal_type):
        if self._spread_analyzer:
            pair_key = f"{pair[0]}/{pair[1]}"
            self._spread_analyzer.trade_signals[pair_key].append(
                (date, signal_type))

    def next(self):
        """Main strategy logic"""

        if not self.params.pairs:
            return

        current_date = self.data_mapping[self.params.pairs[0]][0].datetime.date(
            0)
        current_equity = self.broker.getvalue()

        self.rebalance_counter += 1

        self._update_price_history()
        self._handle_initial_period(current_date, current_equity)

        if current_equity < self.initial_equity:
            self.cum_negative_days += 1

        if (
            self.rebalance_counter % self.params.rebalance_period == 0
        ) and self.rebalance_counter > self.params.M:
            self._execute_rebalance(current_date)

        if self.rebalance_counter == self.params.M:
            self._execute_rebalance(current_date)
            self._process_pairs(current_date, current_equity)

        for pair in self.params.pairs:
            if len(self.spreads[pair]) >= self.params.long_window:
                spread_window = np.array(
                    self.spreads[pair][-self.params.long_window:])
                self.spread_mean[pair] = np.mean(spread_window)
                self.spread_std[pair] = np.std(spread_window) + 1e-5

                latest_spread = self.spreads[pair][-1]
                self.z_scores[pair] = (
                    latest_spread - self.spread_mean[pair]
                ) / self.spread_std[pair]

        if self._spread_analyzer:
            self._spread_analyzer.next()

        if self.rebalance_counter > self.params.M:
            print_section(f"Date: {current_date.strftime('%Y-%m-%d')}", BLUE)
            self.last_processed_date = current_date

            self._process_pairs(current_date, current_equity)

    def _update_price_history(self):
        """Update price history for all pairs"""
        for pair in self.params.pairs:
            if pair not in self.data_mapping:
                continue

            data_s1, data_s2 = self.data_mapping[pair]

            if not (data_s1 and data_s2):
                continue  # Skip if data feeds are invalid

            s1_price = data_s1.close[0]
            s2_price = data_s2.close[0]

            # Check for NaN, infinity, or zero/negative prices
            if not (np.isfinite(s1_price)) or s1_price <= 0:
                buffered_print(
                    f"Invalid price for {pair[0]} at {data_s1.datetime.date(0)}"
                )
                continue
            if not (np.isfinite(s2_price)) or s2_price <= 0:
                buffered_print(
                    f"Invalid price for {pair[1]} at {data_s2.datetime.date(0)}"
                )
                continue

            common_dates = pd.to_datetime(data_s1.datetime.date(0))
            self.price_history[pair]["dates"].append(
                common_dates)  # Use aligned dates
            self.price_history[pair]["s1"].append(data_s1.close[0])
            self.price_history[pair]["s2"].append(data_s2.close[0])

    # def _handle_initial_period(self, current_date, current_equity):
    #     """Handle initial warm-up period with thread-safe updates"""
    #     if not self.start_date:
    #         self.start_date = current_date
    #         self.initial_equity = current_equity

    #     if self.rebalance_counter < self.params.M:
    #         # Process pairs without trading
    #         args_list = [
    #             (
    #                 pair,
    #                 list(self.spreads[pair]),
    #                 self.hedge_ratios[pair],
    #                 self.params.short_window,
    #                 self.params.long_window,
    #                 self.data_mapping[pair][0].close[0],
    #                 self.data_mapping[pair][1].close[0],
    #             )
    #             for pair in self.params.pairs
    #         ]

    #         # Calculate stats without multiprocessing for simplicity
    #         for args in args_list:
    #             result = calculate_pair_stats(*args)
    #             (
    #                 pair,
    #                 spread,
    #                 z_score,
    #                 long_ma,
    #                 short_ma,
    #                 spread_std,
    #                 updated_spreads,
    #             ) = result
    #             self.spreads[pair] = updated_spreads
    #             if spread is not None:
    #                 self.spread_mean[pair] = long_ma
    #                 self.spread_std[pair] = spread_std
    #                 self.z_scores[pair] = z_score

    #         # Show and update the loading bar based on rebalance_counter
    #         if not self._init_bar_shown:
    #             self._init_pbar = tqdm(
    #                 total=self.params.M,
    #                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    #                 colour="yellow",
    #                 desc=f"{YELLOW}Initializing Data",
    #             )
    #             self._init_bar_shown = True
    #             self._last_pbar_n = 0  # Initialize the attribute here

    #         if self._init_pbar is not None:
    #             progress = min(self.rebalance_counter, self.params.M)
    #             self._init_pbar.update(progress - self._last_pbar_n)
    #             self._last_pbar_n = progress

    #         if self.rebalance_counter >= self.params.M:
    #             if self._init_pbar is not None:
    #                 # Force complete to 100%
    #                 remaining = self.params.M - self._last_pbar_n
    #                 if remaining > 0:
    #                     self._init_pbar.update(remaining)
    #                     self._last_pbar_n = self.params.M

    #                 # self._init_pbar.update(remaining)
    #                 self._init_pbar.close()
    #                 self._init_pbar = None
    #                 print()  # Add newline after bar

    #         progress = min(self.rebalance_counter, self.params.M)
    #         if self._init_pbar is not None:
    #             self._init_pbar.update(progress - self._last_pbar_n)
    #             self._last_pbar_n = progress
    #             if self.rebalance_counter >= self.params.M:
    #                 self._init_pbar.close()
    #                 self._init_pbar = None

    #         # Close the bar when initialization is complete
    #     if self.rebalance_counter >= self.params.M and self._init_pbar is not None:
    #         self._init_pbar.close()
    #         self._init_pbar = None

    #         # Update the progress bar to match rebalance_counter
    #         # if self._init_pbar is not None:
    #         #     # tqdm only allows incremental updates, so track last position
    #         #     if not hasattr(self, "_last_pbar_n"):
    #         #         self._last_pbar_n = 0
    #         #     progress = min(self.rebalance_counter, self.params.M)
    #         #     delta = progress - self._last_pbar_n
    #         #     if delta > 0:
    #         #         self._init_pbar.update(delta)
    #         #         self._last_pbar_n = progress

    def _handle_initial_period(self, current_date, current_equity):
        """Handle initial warm-up period with thread-safe updates"""
        if not self.start_date:
            self.start_date = current_date
            self.initial_equity = current_equity

        if self.rebalance_counter < self.params.M:
            # Process pairs without trading
            args_list = [
                (
                    pair,
                    list(self.spreads[pair]),
                    self.hedge_ratios[pair],
                    self.params.short_window,
                    self.params.long_window,
                    self.data_mapping[pair][0].close[0],
                    self.data_mapping[pair][1].close[0],
                )
                for pair in self.params.pairs
            ]

            # Calculate stats without multiprocessing for simplicity
            for args in args_list:
                result = calculate_pair_stats(*args)
                (
                    pair,
                    spread,
                    z_score,
                    long_ma,
                    short_ma,
                    spread_std,
                    updated_spreads,
                ) = result
                self.spreads[pair] = updated_spreads
                if spread is not None:
                    self.spread_mean[pair] = long_ma
                    self.spread_std[pair] = spread_std
                    self.z_scores[pair] = z_score

            # Show and update the loading bar based on rebalance_counter
            if not self._init_bar_shown:
                self._init_pbar = tqdm(
                    total=self.params.M,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                    colour="yellow",
                    desc=f"{YELLOW}Initializing Data",
                )
                self._init_bar_shown = True
                self._last_pbar_n = 0  # Initialize the attribute here

            if self._init_pbar is not None:
                progress = min(self.rebalance_counter, self.params.M)
                self._init_pbar.update(progress - self._last_pbar_n)
                self._last_pbar_n = progress

            if self.rebalance_counter >= self.params.M:
                if self._init_pbar is not None:
                    # Force complete to 100%
                    remaining = self.params.M - self._last_pbar_n
                    if remaining > 0:
                        self._init_pbar.update(remaining)
                        self._last_pbar_n = self.params.M

                    self._init_pbar.close()
                    self._init_pbar = None
                    buffered_print("\n")  # Add newline after bar

        # Keep only the proper closing block with update logic
        if self.rebalance_counter >= self.params.M and self._init_pbar is not None:
            # Force complete to 100%
            remaining = self.params.M - self._last_pbar_n
            if remaining > 0:
                self._init_pbar.update(remaining)
                self._last_pbar_n = self.params.M
            self._init_pbar.close()
            self._init_pbar = None
            buffered_print("\n")

    def _execute_rebalance(self, current_date):
        """Enhanced rebalance execution with robust formatting"""

        # print_header(
        #     f"Portfolio Rebalance | Date: {current_date.strftime('%Y-%m-%d')} | Portfolio Value: ${self.broker.getvalue():,.2f}"
        # )

        self.rebalance_portfolio(current_date)

        # Format allocation table
        # allocation_data = {pair: self.weights[pair] for pair in self.params.pairs}
        # alloc_chart = format_rebalance_table(allocation_data)
        # buffered_print(alloc_chart)

        # Add clear visual separator using StructuredMessage with GREEN
        term_width = shutil.get_terminal_size().columns

        # separator_line = StructuredMessage("═" * term_width, color_code=GREEN)
        print_centered(f"{YELLOW}=" * term_width)

    def _process_pairs(self, current_date, current_equity):
        """Process all pairs for trading signals"""

        # Print date header

        args_list = self._prepare_pair_args()
        vix_ma, vix_last = self.loader.get_volatility_index(
            as_of_date=current_date, days=15
        )

        for args in args_list:
            result = calculate_pair_stats(*args)
            self._process_pair_result(result, vix_ma, vix_last, current_date)

    def _prepare_pair_args(self):
        """Prepare arguments for pair processing"""

        return [
            (
                pair,
                list(self.spreads[pair]),
                self.hedge_ratios[pair],
                self.params.short_window,
                self.params.long_window,
                self.data_mapping[pair][0].close[0],
                self.data_mapping[pair][1].close[0],
            )
            for pair in self.params.pairs
        ]

    def _process_pair_result(self, result, vix_ma, vix_last, current_date):
        """Process individual pair results with improved formatting"""

        (pair, spread, z_score, long_ma, short_ma,
         spread_std, updated_spreads) = result

        self.spreads[pair] = updated_spreads
        if spread is None:
            return

        # if self.spread_std[pair] > 2 * np.mean(list(self.spread_std.values())):
        #     vol_alert = format_volatility_alert(
        #         current_vol=self.spread_std[pair],
        #         historical_vol=np.mean(list(self.spread_std.values())),
        #     )
            # buffered_print(vol_alert)

        # Execute trading logic
        self.execute_trades(pair, z_score, spread_std,
                            vix_ma, vix_last, current_date)

    def _get_rl_action(self, pair):
        """Query the RL agent for a trading action on this pair."""
        from .rl_agent import extract_state, ACTIONS, get_valid_actions

        agent = self.rl_agents.get(pair)
        if agent is None:
            return None

        spread_arr = np.array(self.spreads[pair])
        if len(spread_arr) < self.params.rl_lookback + 1:
            return None

        scale = max(np.std(spread_arr), 1e-8)
        _, state_idx = extract_state(
            spread_arr, len(spread_arr) - 1,
            l=self.params.rl_lookback,
            k=self.params.rl_threshold_k,
            scale=scale,
        )

        # Map pair position to RL position space {0, 1}
        rl_position = 1 if self.pair_positions[pair] != 0 else 0

        action_idx = agent.select_action(state_idx, rl_position, training=False)
        return ACTIONS[action_idx]  # -1, 0, or +1

    def execute_trades(self, pair, z_score, spread_std, vix_ma, vix_last, current_date):
        """Execute trades with improved signal formatting.

        When use_rl=True, the RL agent's action is combined with z-score signals:
        - RL BUY (+1) + z_score below threshold → strong entry signal
        - RL HOLD (0) → defer to z-score only
        - RL SELL (-1) with open position → exit signal
        """

        data_s1, data_s2 = self.data_mapping[pair]

        # Dynamic Position Sizing
        position_size = self.calculate_position_size(pair)
        if position_size < 1:
            buffered_print(
                f"{YELLOW}Position size too small for {pair}. Requires {abs(position_size)} units{ENDC}"
            )
            return

        # Adaptive Threshold Calculation
        K = (
            self._calculate_adaptive_k(pair, spread_std)
            if self.params.adaptive_K
            else self.params.K
        )
        if abs(K) > 1.25 * self.params.K:
            position_size *= 2

        available_cash = self.broker.getcash()
        required_cash = position_size * (data_s1.close[0] + data_s2.close[0])
        if required_cash > available_cash * 0.9:
            return

        # Trade Confidence Score
        tcs = self._calculate_tcs(z_score, K)

        # RL agent signal (if enabled)
        rl_action = self._get_rl_action(pair) if self.params.use_rl else None

        # Existing Position Management
        if self.pair_positions[pair] != 0:
            # RL agent says SELL → exit immediately
            if rl_action == -1:
                self.close_position(pair, current_date, "RL Exit Signal", tcs)
                return
            self.check_stop_conditions(pair, K, tcs, current_date)
            return

        # Trade Entry Logic - combine z-score with RL
        zscore_entry = (z_score < -K or z_score > K) and tcs > self.params.tcs_max
        rl_confirms = rl_action == 1 if rl_action is not None else True

        entry_condition = (
            (self.pair_positions[pair] == 0)
            and zscore_entry
            and rl_confirms
            and pair not in self.blacklisted_pairs
        )

        if entry_condition:
            self._execute_new_trade(
                pair,
                data_s1,
                data_s2,
                position_size,
                z_score,
                vix_ma,
                vix_last,
                tcs,
                current_date,
            )

    def _calculate_adaptive_k(self, pair, spread_std):
        """Calculate dynamic K value based on volatility"""

        recent_vol = np.std(
            self.spreads[pair][-self.params.volatility_lookback:])
        K = self.params.K * (recent_vol / (spread_std + 1e-5))
        return np.clip(K, 1.0, 3.0)

    def _calculate_tcs(self, z_score, K):
        """Calculate Trade Confidence Score"""

        tcs = np.clip((abs(z_score) - K) / K, 0, 1)  # Ensures TCS ∈ [0,1]
        return np.clip(tcs, 0.0, 1.0)

    def calculate_position_size(self, pair):
        """Calculate Position Sizing Dynamically"""

        data_s1, data_s2 = self.data_mapping[pair]
        s1_price = data_s1.close[0]
        s2_price = data_s2.close[0]
        hr = self.hedge_ratios[pair]

        current_spread = s1_price - hr * s2_price

        spread_std = max(self.spread_std[pair], 1e-5)
        deviation = abs(current_spread -
                        self.spread_mean[pair]) / (spread_std + 1e-5)
        deviation = np.clip(deviation, 0.5, 2.0)

        allocated_risk = (
            self.broker.getvalue() * self.params.risk * self.weights[pair] * 10
        )
        position_size = (allocated_risk * deviation) / s1_price

        vol_ratio = self.spread_std[pair] / np.std(
            self.spreads[pair][-252:]
        )  # Annualized

        # Larger sizes in low-vol regimes
        size = max(1, min(int(position_size * (1 + 0.5 * (1 - vol_ratio))), 10))

        return size

    def _execute_new_trade(
        self, pair, data_s1, data_s2, size, z_score, vix_ma, vix_last, tcs, date
    ):
        """Enhanced trade execution with volatility-adjusted sizing"""

        try:
            # Price validation
            s1_price = max(data_s1.close[0], 0.01)
            s2_price = max(data_s2.close[0], 0.01)
            hedge_ratio = self.hedge_ratios[pair]

            # Volatility-adjusted sizing
            if vix_last > vix_ma * 1.15:
                size_multiplier = 0.7
                buffered_print(
                    f"{RED}Volatility scaling down size for {pair}{ENDC}")
            else:
                size_multiplier = 1.3

            # Position sizing validation
            max_position_size = (self.broker.getcash() * 0.8) / (
                s1_price + hedge_ratio * s2_price
            )
            validated_size = min(size * size_multiplier, max_position_size)
            validated_size = max(math.floor(validated_size), 1)

            # Dynamic direction handling
            if z_score < -self.params.K:
                self._open_long(
                    pair=pair,
                    data_s1=data_s1,
                    data_s2=data_s2,
                    size=validated_size,
                    z_score=z_score,
                    tcs=tcs,
                    date=date,
                    hedge_ratio=hedge_ratio,
                )
            elif z_score > self.params.K:
                self._open_short(
                    pair=pair,
                    data_s1=data_s1,
                    data_s2=data_s2,
                    size=validated_size,
                    z_score=z_score,
                    tcs=tcs,
                    date=date,
                    hedge_ratio=hedge_ratio,
                )

        except Exception as e:
            buffered_print(
                f"{RED}Trade execution failed for {pair}: {str(e)}{ENDC}")
            self.record_trade_signal(pair, date, "error")

    def _open_long(self, pair, data_s1, data_s2, size, z_score, tcs, date, hedge_ratio):
        """Precision long entry with slippage protection"""

        try:
            self.last_opened_spread_direction[pair] = "long_spread"

            s1_size = math.floor(size)
            s2_size = math.floor(size * hedge_ratio)

            self.buy(data=data_s1, size=s1_size, exectype=bt.Order.Close)
            self.sell(data=data_s2, size=s2_size, exectype=bt.Order.Close)

            self.pair_positions[pair] = 1
            print_trade(
                "LONG",
                pair,
                date,
                (data_s1.close[0], data_s2.close[0]),
                z_score,
                tcs,
                size,
                self.weights[pair],
            )

            self.active_trades[pair] = {
                "entry_spread": data_s1.close[0] - hedge_ratio * data_s2.close[0],
                "entry_size": size,
                "entry_direction": "LONG",
                "entry_hedge_ratio": hedge_ratio,
                "entry_prices": (data_s1.close[0], data_s2.close[0]),
                "entry_date": date,
            }

        except Exception as e:
            buffered_print(
                f"{RED}Long entry failed for {pair}: {str(e)}{ENDC}")

    def _open_short(
        self, pair, data_s1, data_s2, size, z_score, tcs, date, hedge_ratio
    ):
        """Precision short entry with hedge ratio validation"""

        try:
            self.last_opened_spread_direction[pair] = "short_spread"

            s1_size = math.floor(size)
            s2_size = math.floor(size * hedge_ratio)

            self.sell(data=data_s1, size=s1_size, exectype=bt.Order.Close)
            self.buy(data=data_s2, size=s2_size, exectype=bt.Order.Close)

            self.pair_positions[pair] = -1
            print_trade(
                "SHORT",
                pair,
                date,
                (data_s1.close[0], data_s2.close[0]),
                z_score,
                tcs,
                size,
                self.weights[pair],
            )

            self.active_trades[pair] = {
                "entry_spread": data_s1.close[0] - hedge_ratio * data_s2.close[0],
                "entry_size": size,
                "entry_direction": "SHORT",
                "entry_hedge_ratio": hedge_ratio,
                "entry_prices": (data_s1.close[0], data_s2.close[0]),
                "entry_date": date,
            }

        except Exception as e:
            buffered_print(
                f"{RED}Short entry failed for {pair}: {str(e)}{ENDC}")

    def check_stop_conditions(self, pair, K, tcs, current_date):
        """Enhanced stop condition checking"""

        data_s1, data_s2 = self.data_mapping[pair]
        current_spread = (
            data_s1.close[0] - self.entry_hedge_ratios[pair] * data_s2.close[0]
        )
        z_score = (current_spread -
                   self.spread_mean[pair]) / self.spread_std[pair]

        # Dynamic stop loss with volatility adjustment
        recent_vol = self._get_recent_volatility(pair)
        sl_multiplier = self.params.stop_loss + \
            (recent_vol / self.spread_std[pair])

        # Trailing stop
        entry_spread = self.entry_spreads[pair]
        trailing_stop = max(0.05, recent_vol * 2)
        spread_change = current_spread - entry_spread
        pnl = spread_change / \
            abs(entry_spread) if abs(entry_spread) > 1e-5 else 0
        if any(
            [
                (
                    (
                        pnl < self.params.liquidation_threshold * self.params.stop_loss
                        or pnl
                        > -self.params.liquidation_threshold * self.params.take_profit
                    )
                ),
            ]
        ):
            self.close_position(pair, current_date, "Dynamic Exit", tcs)
            liquidation_msg = format_drawdown_warning(
                drawdown=abs(pnl), threshold=self.params.liquidation_threshold
            )
            buffered_print(liquidation_msg, "RISK")

        # Long
        if self.pair_positions[pair] > 0:
            current_s1 = data_s1.close[0]
            current_s2 = data_s2.close[0]
            entry_s1 = self.entry_s1_price[pair]
            entry_s2 = self.entry_s2_price[pair]

            # Check if prices are worse than entry
            s1_worse = current_s1 < entry_s1
            s2_worse = current_s2 > entry_s2

            # Adjust exit threshold dynamically
            exit_threshold = -K * 0.3 if (s1_worse or s2_worse) else -K * 0.5

            if z_score >= exit_threshold:
                self.close_position(pair, current_date,
                                    "Long Exit (Adjusted)", tcs)
        # Short
        elif self.pair_positions[pair] < 0:
            current_s1 = data_s1.close[0]
            current_s2 = data_s2.close[0]
            entry_s1 = self.entry_s1_price[pair]
            entry_s2 = self.entry_s2_price[pair]

            # Check if prices are worse than entry
            s1_worse = current_s1 > entry_s1
            s2_worse = current_s2 < entry_s2

            # Adjust exit threshold dynamically
            exit_threshold = K * 0.25 if current_spread > entry_spread else K * 0.75

            if z_score <= exit_threshold:
                self.close_position(pair, current_date,
                                    "Short Exit (Adjusted)", tcs)

    def close_position(self, pair, date, reason, tcs):
        """Enhanced position closing with explicit trade reversal"""

        try:
            data_s1, data_s2 = self.data_mapping[pair]
            trade = self.active_trades.get(pair)
            if not trade:
                return

            current_s1 = max(data_s1.close[0], 0.01)
            current_s2 = max(data_s2.close[0], 0.01)
            original_size = trade["entry_size"]
            hedge_ratio = self.entry_hedge_ratios.get(
                pair, self.hedge_ratios[pair])

            current_spread = current_s1 - hedge_ratio * current_s2
            spread_change = current_spread - trade["entry_spread"]

            direction_factor = 1 if trade["entry_direction"] == "LONG" else -1
            pnl = (current_spread - trade["entry_spread"]
                   ) * original_size * direction_factor
            # pnl = (
            #     spread_change
            #     * original_size
            #     * (-1 if trade["entry_direction"] == "SHORT" else 1)
            # )
            # commission_rate = self.broker.getcommissioninfo(data_s1).p.commission
            # commission = (original_size * (trade["entry_prices"][0] + current_s1) +
            #             (original_size * hedge_ratio) * (trade["entry_prices"][1] + current_s2)) * commission_rate
            # net_pnl = pnl - commission

            if trade["entry_direction"] == "LONG":
                self.sell(data=data_s1, size=original_size)
                self.buy(data=data_s2, size=math.ceil(
                    original_size * hedge_ratio))
            else:
                self.buy(data=data_s1, size=original_size)
                self.sell(data=data_s2, size=math.ceil(
                    original_size * hedge_ratio))

            self.trade_history[pair].append(
                {
                    **trade,
                    "exit_date": date,
                    "exit_prices": (current_s1, current_s2),
                    "pnl": pnl,
                    "duration_days": (date - trade["entry_date"]).days,
                }
            )

            pair_index = self.params.pairs.index(pair)
            color_idx = pair_index % len(self.trade_recorder.COLOR_CYCLE)

            # Record trade with pair-based color_idx
            self.trade_recorder.record_trade(
                pair=pair,
                entry_dt=trade["entry_date"],
                exit_dt=date,
                pnl=pnl,
                size=trade["entry_size"],
                direction=trade["entry_direction"],
                entry_prices=trade["entry_prices"],
                exit_prices=(current_s1, current_s2),
                commission=0,
                hedge_ratio=trade["entry_hedge_ratio"],
                color_idx=color_idx,  # Critical addition
            )

            self.pair_positions[pair] = 0
            del self.active_trades[pair]
            self.record_trade_signal(pair, date, "exit")

            print_trade_exit(
                pair=pair,
                date=date,
                pnl=pnl,
                duration=(date - trade["entry_date"]).days,
                reason=reason,
            )

        except Exception as e:
            buffered_print(
                f"{RED}Error closing position for {pair}: {str(e)}{ENDC}")
            self.close(data=data_s1)
            self.close(data=data_s2)
            self.pair_positions[pair] = 0

    def get_net_profit(self, pair):
        """Calculate cumulative PnL for a pair from trade history"""

        return sum(trade["pnl"] for trade in self.trade_history.get(pair, []))

    def _get_recent_volatility(self, pair):
        """Get recent spread volatility"""

        spread_series = pd.Series(self.spreads[pair])
        if len(spread_series) >= 2:
            return (
                spread_series.ewm(
                    span=self.params.volatility_lookback, adjust=False)
                .std()
                .iloc[-1]
            )
        return self.spread_std[pair]

    def rebalance_portfolio(self, current_date):
        """Enhanced portfolio rebalancing"""

        # Add top banner and spacing
        term_width = shutil.get_terminal_size().columns
        print_centered(f"{YELLOW}{'=' * term_width}{ENDC}", YELLOW)

        # Phase 1: Filter invalid pairs
        self.invalid_pairs = list(
            set(self._identify_invalid_pairs(current_date)))

        # Print centered message with proper formatting
        msg = f"Rebalance: Identified {len(self.invalid_pairs)} invalid pairs out of {len(self.params.pairs)} total pairs....."
        print_centered(msg, YELLOW)

        # Phase 2: Optimize valid pairs
        ranked_pairs = self._optimize_valid_pairs(
            current_date, self.invalid_pairs)

        # Phase 3: Calculate new weights
        self._calculate_weights(ranked_pairs)

        # Phase 4: Cleanup and logging
        self.valid_pairs = set(
            [pair for pair in self.params.pairs if pair not in self.invalid_pairs]
        )

    def _identify_invalid_pairs(self, current_date):
        """Identify pairs to exclude"""

        hurst_invalid_pairs = []
        invalid_pairs = []

        for pair in self.params.pairs:
            if (
                hurst(np.array(self.spreads[pair]
                      [-self.params.trend_lookback:]))
                > self.params.hurst_threshold
            ):
                hurst_invalid_pairs.append(pair)

            # PNL threshold check to liquidate pair trade
            net_profit = self.get_net_profit(pair)
            # if (
            #     net_profit < self.params.liquidation_threshold
            #     and self.pair_positions[pair] != 0
            #     and self.pnl_liquidation_counter[pair] > 3
            # ):
            #     self.close_position(
            #         pair,
            #         current_date,
            #         f"{PURPLE}{pair} PNL Liquidation | Total pair net profit: {net_profit:.2%}{ENDC}",
            #         0,
            #     )

            if net_profit >= self.params.liquidation_threshold * 0.5:
                self.pnl_liquidation_counter[pair] += 1

            if (
                self.pnl_liquidation_counter[pair] > 5
                and pair not in self.blacklisted_pairs
            ):
                self.blacklisted_pairs.append(pair)

        max_workers = min(self.params.max_workers,
                          multiprocessing.cpu_count() - 1)

        filtered_pairs = [
            cointegration_worker(
                pair,
                self.price_history[pair]["s1"],
                self.price_history[pair]["s2"],
                window=self.params.cointegration_lookback,
            )
            for pair in self.params.pairs
        ]

        invalid_pairs = [
            pair
            for pair, result in zip(self.params.pairs, filtered_pairs)
            if result is None and pair in hurst_invalid_pairs
        ]

        return invalid_pairs

    def _optimize_valid_pairs(self, current_date, invalid_pairs):
        """Optimize valid pairs (Phase 2 of rebalance)"""

        msg = "Rebalance: Optimizing portfolio weights based on pair performance..."
        print_centered(msg, YELLOW)

        ranked_pairs = []

        # Optimize valid pairs using multiprocessing
        with multiprocessing.Pool(self.params.max_workers) as pool:
            tasks = [
                (
                    self.loader.get_normalized_pair(
                        t1=pair[0],
                        t2=pair[1],
                        start_date=self.start_date,
                        end_date=current_date.strftime("%Y-%m-%d"),
                    ),
                    pair[0],
                    pair[1],
                )
                for pair in self.params.pairs
                if pair not in invalid_pairs
            ]
            try:
                # Run all tasks and collect results directly
                results = pool.starmap(self.optimizer.optimize, tasks)
            except Exception as e:
                buffered_print(f"Optimization failed: {str(e)}")
                results = [None] * len(tasks)

        # Calculate Sharpe ratios and process results
        pair_sharpes = {}

        if hasattr(self, "trade_recorder"):
            analysis = self.trade_recorder.get_trades()
        else:
            analysis = []

        for pair in self.params.pairs:
            if pair in invalid_pairs:
                pair_sharpes[pair] = 0
                continue

            pair_trades = [t for t in analysis if t["pair"] == pair]

            if len(pair_trades) < 2:
                pair_sharpes[pair] = 0
                continue

            returns = np.array([t["pnl"] / self.broker.getvalue()
                               for t in pair_trades])
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            pair_sharpes[pair] = (
                avg_return / std_return) if std_return > 1e-5 else 0

        # Rank pairs and optionally train RL agents
        for pair, result in zip(
            (p for p in self.params.pairs if p not in invalid_pairs),
            results,
        ):
            if result:
                self.hedge_ratios[pair] = result["beta"]
                likelihood = result["log_likelihood"]
                mu = result["mu"]
                sigma = result["sigma"]
                sharpe_score = pair_sharpes.get(pair, 0)
                ranked_pairs.append(
                    (pair, likelihood, mu / sigma, sharpe_score))

                # Store OU params for RL
                self.rl_ou_params[pair] = {
                    "mu": mu, "theta": result["theta"], "sigma": sigma,
                }

                # Train RL agent per pair during rebalance
                if self.params.use_rl and mu > 0.001:
                    self._train_rl_agent(pair, result)

        return ranked_pairs

    def _train_rl_agent(self, pair, ou_result):
        """Train a tabular Q-learning agent for a specific pair."""
        from .rl_agent import TabularQAgent, RLAgentConfig, train_agent

        mu = max(ou_result["mu"], 0.3)
        theta = ou_result["theta"]
        sigma = max(ou_result["sigma"], 0.01)

        config = RLAgentConfig()
        config.ou_mu = mu
        config.ou_theta = theta
        config.ou_sigma = sigma
        config.n_training_paths = self.params.rl_training_paths
        config.n_epochs = self.params.rl_training_epochs
        config.lookback = self.params.rl_lookback
        config.threshold_k = self.params.rl_threshold_k

        agent = train_agent(
            config=config,
            ou_params_range={
                "mu_range": (max(0.1, mu * 0.5), mu * 2.0),
                "sigma_range": (max(0.005, sigma * 0.5), sigma * 2.0),
            },
            verbose=False,
        )
        self.rl_agents[pair] = agent
        buffered_print(
            f"{GREEN}RL agent trained for {pair[0]}-{pair[1]} "
            f"(mu={mu:.4f}, theta={theta:.4f}){ENDC}"
        )

    def _calculate_weights(self, ranked_pairs):
        """Phase 3: Calculate weights using old version's formula"""

        msg = "Rebalance: Calculating optimal portfolio weights for valid pairs..."
        print_centered(msg, YELLOW)

        if not ranked_pairs:
            self.weights = {pair: 0 for pair in self.params.pairs}
            return

        l_scores = [x[1] for x in ranked_pairs]
        mu_scores = [x[2] for x in ranked_pairs]
        sharpe_scores = [x[3] for x in ranked_pairs]

        # Normalize using old version's method
        tilde_li = (l_scores - np.min(l_scores)) / (
            np.max(l_scores) - np.min(l_scores) + 1e-8
        )
        tilde_mu = (mu_scores - np.min(mu_scores)) / (
            np.max(mu_scores) - np.min(mu_scores) + 1e-8
        )
        tilde_sharpe = (sharpe_scores - np.min(sharpe_scores)) / (
            np.max(sharpe_scores) - np.min(sharpe_scores) + 1e-8
        )

        combined_scores = tilde_li * 0.33 + tilde_mu * 0.33 + tilde_sharpe * 0.34

        sorted_pairs = sorted(
            zip(ranked_pairs, combined_scores), key=lambda x: -x[1])

        top_pairs = sorted_pairs[: int(0.35 * len(self.params.pairs))]

        # print_centered(
        #     f"{YELLOW}Ranked top pairs: {len(top_pairs)} / {len(self.params.pairs)}{ENDC}\n",
        #     YELLOW,
        # )

        if ranked_pairs:
            msg = f"Ranked top pairs: {len(top_pairs)} / {len(self.params.pairs)}"
            print_centered(msg, YELLOW)

            # Add spacing after bottom banner
            term_width = shutil.get_terminal_size().columns
            print_centered(f"{YELLOW}{'=' * (term_width-15)}{ENDC}", YELLOW)

        # print_centered(
        #     f"{YELLOW}{'='*(MAX_LINE_WIDTH-2)}{ENDC}",
        #     YELLOW
        # )

        total_score = sum(score for (_, _, _, _), score in top_pairs)
        for (pair, _, _, _), score in top_pairs:
            self.weights[pair] = score / \
                total_score if total_score > 1e-8 else 0

        # Zero out non-top pairs
        for pair in self.params.pairs:
            if pair not in [p[0] for p, _ in top_pairs]:
                self.weights[pair] = 0

    # def print_trade(self, *args, **kwargs):
    #     """Print trade information with consistent separators"""

    #     # ... existing code ...
    #     print_centered(f"{YELLOW}{'─'*(MAX_LINE_WIDTH-2)}{ENDC}", YELLOW)
