# ---- START OF FILE analysis_utils.py ----

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats  # For skew, kurtosis, norm, linregress
from statsmodels.tsa.stattools import coint, adfuller  # For Engle-Granger and ADF
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # For Johansen
from typing import Dict, List, Tuple, Optional


def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series, window: int = 20) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Calculates rolling Pearson and Spearman correlations between two series.

    Args:
        series1: First pandas Series.
        series2: Second pandas Series.
        window: Rolling window size.

    Returns:
        A tuple (rolling_pearson, rolling_spearman). Series contain NaN for initial periods.
        Returns (None, None) if input series are unsuitable.
    """
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series):
        return None, None
    if series1.empty or series2.empty:  # Check for empty series first
        return None, None

    # Align series by index, then drop NaNs from alignment
    df = pd.DataFrame({'s1': series1, 's2': series2})
    # Forward fill to handle cases where one series has data and other doesn't on a specific day,
    # then drop any remaining NaNs (e.g. at the very beginning if both are NaN)
    df = df.ffill().bfill().dropna()

    if len(df) < window:  # Check again after alignment and fill
        return None, None

    aligned_s1 = df['s1']
    aligned_s2 = df['s2']

    rolling_pearson = aligned_s1.rolling(
        window=window).corr(aligned_s2)  # Default is pearson

    # For Spearman, use apply with scipy.stats.spearmanr
    # The lambda function will receive a DataFrame window if raw=False
    # or two Series if we pass them separately to apply.
    # We need to ensure at least 2 non-NaN values for spearmanr.

    def spearman_on_window(window_data_s1, window_data_s2):
        # Drop NaNs within the current window for both series aligned
        valid_s1 = window_data_s1.dropna()
        valid_s2 = window_data_s2.dropna()
        # Align them again for the current window after dropping NaNs individually
        aligned_window = pd.DataFrame(
            {'s1': valid_s1, 's2': valid_s2}).dropna()
        if len(aligned_window) < 2:  # spearmanr needs at least 2 points
            return np.nan
        corr, _ = scipy_stats.spearmanr(
            aligned_window['s1'], aligned_window['s2'])
        return corr

    # Apply the function to the rolling windows of the original aligned series
    # Initialize with original index
    rolling_spearman = pd.Series(index=aligned_s1.index, dtype=float)

    # Iterate through windows to apply spearman correctly
    # This is less efficient than a direct rolling apply if pandas supported it,
    # but more robust for handling NaNs within windows for spearmanr.
    if len(aligned_s1) >= window:
        for i in range(window - 1, len(aligned_s1)):
            window_s1 = aligned_s1.iloc[i-window+1:i+1]
            window_s2 = aligned_s2.iloc[i-window+1:i+1]
            rolling_spearman.iloc[i] = spearman_on_window(window_s1, window_s2)
    else:  # Not enough data for any window
        rolling_spearman = pd.Series(np.nan, index=aligned_s1.index)

    return rolling_pearson, rolling_spearman


def simulate_pair_equity_curve(
    dates: List[pd.Timestamp],
    s1_prices: np.ndarray,
    s2_prices: np.ndarray,
    hedge_ratio: float,
    z_scores: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    initial_capital: float = 10000.0,
    fixed_trade_size_asset1: float = 100.0,
    commission_per_leg_trade: float = 0.0
) -> Tuple[pd.Series, List[Dict], Dict[str, float]]:
    """
    Simulates an equity curve for a pair based on z-score signals.

    Args:
        dates: List of timestamps for the data.
        s1_prices: Numpy array of prices for asset 1.
        s2_prices: Numpy array of prices for asset 2.
        hedge_ratio: Hedge ratio (Spread = S1 - HR * S2).
        z_scores: Numpy array of z-scores for the spread.
        entry_threshold: Z-score absolute value to enter a trade.
        exit_threshold: Z-score absolute value to exit a trade (closer to zero).
        initial_capital: Starting capital for the simulation.
        fixed_trade_size_asset1: Fixed number of shares of asset1 to trade.
        commission_per_leg_trade: Commission for a single leg (entry or exit). Total trade commission is 4x this.

    Returns:
        equity_curve_series: Pandas Series of equity values.
        simulated_trades: List of dictionaries, each representing a trade.
        performance_metrics: Dictionary of calculated performance metrics.
    """
    if not (len(dates) == len(s1_prices) == len(s2_prices) == len(z_scores)):
        # from ..printing_system import buffered_print # Local import for util
        # buffered_print(f"SimEquityCalc Error: Input array length mismatch. D:{len(dates)}, S1:{len(s1_prices)}, S2:{len(s2_prices)}, Z:{len(z_scores)}", "ERROR")
        raise ValueError("Input arrays must have the same length.")
    if len(dates) == 0:
        return pd.Series(dtype=float), [], calculate_performance_metrics(pd.Series(dtype=float), [], initial_capital)

    equity = initial_capital
    equity_curve_values = [initial_capital] * \
        len(dates)  # Pre-allocate with initial capital

    simulated_trades_list = []
    position = 0  # 0: flat, 1: long spread, -1: short spread
    entry_s1_price_val = 0.0
    entry_s2_price_val = 0.0
    entry_date_val = None

    # total_commission_per_round_trip = 4 * commission_per_leg_trade
    # pnl_s1 = (exit_s1_price - entry_s1_price_val) * fixed_trade_size_asset1
    # pnl_s2 = (entry_s2_price_val - exit_s2_price) * \
    #     (fixed_trade_size_asset1 * hedge_ratio)
    # commission = 4 * commission_per_leg_trade * fixed_trade_size_asset1
    # pnl = pnl_s1 + pnl_s2 - commission

    for i in range(len(dates)):  # Start from first data point to check for entry
        current_date = dates[i]
        current_s1_price = s1_prices[i]
        current_s2_price = s2_prices[i]
        current_z = z_scores[i]

        # Update equity curve value for the current day *before* any PnL realization from closing a trade
        # If a trade closes, its PnL is added, and then this equity value is recorded.
        # If no trade, equity remains same as previous day (or reflects ongoing PnL if we were tracking mark-to-market)
        # For this simulation, PnL is realized on close.
        if i > 0:  # For subsequent days, start with previous day's equity
            equity_curve_values[i] = equity_curve_values[i-1]

        pnl_from_closing_trade = 0.0

        # Handle exits first
        if position == 1:  # In a long spread (Long S1, Short S2)
            if current_z >= -exit_threshold:
                exit_s1_price = current_s1_price
                exit_s2_price = current_s2_price

                pnl_s1 = (exit_s1_price - entry_s1_price_val) * \
                    fixed_trade_size_asset1
                pnl_s2 = (entry_s2_price_val - exit_s2_price) * \
                    (fixed_trade_size_asset1 * hedge_ratio)  # Short S2
                pnl = pnl_s1 + pnl_s2

                equity += pnl
                pnl_from_closing_trade = pnl
                simulated_trades_list.append({
                    "entry_dt": entry_date_val, "exit_dt": current_date,
                    "entry_s1": entry_s1_price_val, "exit_s1": exit_s1_price,
                    "entry_s2": entry_s2_price_val, "exit_s2": exit_s2_price,
                    "pnl": pnl, "direction": "long_spread",
                    "duration_days": (current_date - entry_date_val).days if entry_date_val else 0
                })
                position = 0
        elif position == -1:  # In a short spread (Short S1, Long S2)
            if current_z <= exit_threshold:
                exit_s1_price = current_s1_price
                exit_s2_price = current_s2_price

                pnl_s1 = (entry_s1_price_val - exit_s1_price) * \
                    fixed_trade_size_asset1  # Short S1
                pnl_s2 = (exit_s2_price - entry_s2_price_val) * \
                    (fixed_trade_size_asset1 * hedge_ratio)  # Long S2
                pnl = pnl_s1 + pnl_s2

                equity += pnl
                pnl_from_closing_trade = pnl
                simulated_trades_list.append({
                    "entry_dt": entry_date_val, "exit_dt": current_date,
                    "entry_s1": entry_s1_price_val, "exit_s1": exit_s1_price,
                    "entry_s2": entry_s2_price_val, "exit_s2": exit_s2_price,
                    "pnl": pnl, "direction": "short_spread",
                    "duration_days": (current_date - entry_date_val).days if entry_date_val else 0
                })
                position = 0

        # Handle entries
        if position == 0:  # Can enter a new trade if flat
            if current_z < -entry_threshold:  # Long spread entry
                position = 1
                entry_s1_price_val = current_s1_price
                entry_s2_price_val = current_s2_price
                entry_date_val = current_date
            elif current_z > entry_threshold:  # Short spread entry
                position = -1
                entry_s1_price_val = current_s1_price
                entry_s2_price_val = current_s2_price
                entry_date_val = current_date

        equity_curve_values[i] = equity  # Record final equity for the day

    equity_curve_series = pd.Series(
        equity_curve_values, index=pd.DatetimeIndex(dates))

    performance_metrics = calculate_performance_metrics(
        equity_curve_series, simulated_trades_list, initial_capital)

    return equity_curve_series, simulated_trades_list, performance_metrics


def calculate_performance_metrics(equity_curve: pd.Series, trades: List[Dict], initial_capital: float) -> Dict[str, float]:
    """Calculates performance metrics from an equity curve and trade list."""
    metrics = {
        "cagr": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
        "max_drawdown": 0.0, "win_rate": 0.0, "avg_trade_duration": 0.0,
        "total_trades": len(trades), "profit_factor": 0.0, "final_equity": equity_curve.iloc[-1] if not equity_curve.empty else initial_capital,
        "net_profit_pct": 0.0
    }

    if equity_curve.empty or len(equity_curve) < 2:
        return metrics

    # Final Equity and Net Profit %
    # start_equity_val = initial_capital  # Use the passed initial capital
    start_equity_val = equity_curve.iloc[0] if not equity_curve.empty else initial_capital
    end_equity_val = equity_curve.iloc[-1]
    metrics["final_equity"] = end_equity_val
    if start_equity_val > 0:
        metrics["net_profit_pct"] = (
            end_equity_val - start_equity_val) / start_equity_val

    # CAGR
    num_days_total = (equity_curve.index[-1] - equity_curve.index[0]).days
    if num_days_total > 0 and start_equity_val > 0:
        metrics["cagr"] = (
            end_equity_val / start_equity_val) ** (252.0 / num_days_total) - 1.0

    # Daily returns for Sharpe and Sortino
    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) > 1:
        mean_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)

        # Sharpe Ratio
        sharpe = (mean_daily_return / std_daily_return) * \
            np.sqrt(252) if std_daily_return > 1e-9 else 0.0
        metrics["sharpe_ratio"] = sharpe if np.isfinite(sharpe) else 0.0

        # Sortino Ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(negative_returns) if len(
            negative_returns) > 1 else 0.0
        sortino = (mean_daily_return / downside_std) * \
            np.sqrt(252) if downside_std > 1e-9 else 0.0
        metrics["sortino_ratio"] = sortino if np.isfinite(sortino) else 0.0
    else:  # Handle cases with too few returns for std dev
        metrics["sharpe_ratio"] = 0.0
        metrics["sortino_ratio"] = 0.0

    # Max Drawdown
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    metrics["max_drawdown"] = abs(
        drawdown.min()) if not drawdown.empty and not drawdown.isnull().all() else 0.0

    # Trade-based metrics
    if trades:
        profitable_trades = sum(1 for t in trades if t["pnl"] > 0)
        metrics["win_rate"] = profitable_trades / \
            len(trades) if trades else 0.0

        # Ensure duration_days is not None
        total_duration = sum(t["duration_days"]
                             for t in trades if t["duration_days"] is not None)
        metrics["avg_trade_duration"] = total_duration / \
            len(trades) if trades else 0.0

        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        # Ensure it's negative PnLs sum
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        metrics["profit_factor"] = gross_profit / \
            gross_loss if gross_loss > 1e-9 else np.inf if gross_profit > 0 else 0.0

    return metrics


def run_rolling_engle_granger(series1: pd.Series, series2: pd.Series, window: int = 60) -> Optional[pd.Series]:
    # ... (existing - looks good, ensure window param is used from plotting.py)
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series) or \
       series1.empty or series2.empty or len(series1) < window or len(series2) < window:  # Ensure len(series1) is also checked
        return None
    df = pd.DataFrame({'s1': series1, 's2': series2}
                      ).dropna()  # Ensure alignment first
    if len(df) < window:
        return None

    rolling_p_values = pd.Series(index=df.index, dtype=float)
    for i in range(window - 1, len(df)):
        window_s1 = df['s1'].iloc[i-window+1:i+1]
        window_s2 = df['s2'].iloc[i-window+1:i+1]
        if window_s1.nunique() < 2 or window_s2.nunique() < 2:
            rolling_p_values.iloc[i] = np.nan
            continue
        try:
            with np.errstate(invalid='ignore'):
                _, p_val, _ = coint(window_s1, window_s2,
                                    trend='c', maxlag=0, autolag=None)
            rolling_p_values.iloc[i] = p_val
        except Exception:
            rolling_p_values.iloc[i] = np.nan
    return rolling_p_values


def run_rolling_adf_test(series: pd.Series, window: int = 90, regression='c', autolag='AIC') -> Optional[pd.Series]:
    """Performs rolling Augmented Dickey-Fuller test. Returns a Series of p-values."""
    if not isinstance(series, pd.Series) or series.empty or len(series.dropna()) < window:
        return None

    # Ensure series is float for adfuller
    series_clean = series.dropna().astype(float)
    if len(series_clean) < window:
        return None

    rolling_adf_p_values = pd.Series(index=series_clean.index, dtype=float)

    for i in range(window - 1, len(series_clean)):
        window_data = series_clean.iloc[i-window+1:i+1]
        if window_data.nunique() < 2 or len(window_data) < 20:  # ADF needs sufficient unique points
            rolling_adf_p_values.iloc[i] = np.nan
            continue
        try:
            _, p_val, _, _, _, _ = adfuller(
                window_data, regression=regression, autolag=autolag)
            rolling_adf_p_values.iloc[i] = p_val
        except Exception:
            rolling_adf_p_values.iloc[i] = np.nan

    return rolling_adf_p_values


def run_adf_test(series: pd.Series, regression='c', autolag='AIC') -> Optional[Tuple[float, float]]:
    """Performs Augmented Dickey-Fuller test for stationarity.
    Returns: (ADF Statistic, p-value) or None if test fails.
    """
    if not isinstance(series, pd.Series) or series.empty or series.nunique() < 2 or len(series.dropna()) < 20:
        return None
    try:
        result = adfuller(
            series.dropna(), regression=regression, autolag=autolag)
        return result[0], result[1]  # ADF Statistic, p-value
    except Exception:
        return None


def run_engle_granger_test(series1: pd.Series, series2: pd.Series) -> Optional[float]:
    """Performs Engle-Granger cointegration test."""
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series) or \
       series1.empty or series2.empty or len(series1) < 20 or len(series2) < 20:
        return None

    # Align series
    df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
    if len(df) < 20:
        return None

    if df['s1'].nunique() < 2 or df['s2'].nunique() < 2:
        return None  # Avoid constant series

    try:
        # Suppress warning from coint for constant series (already checked)
        with np.errstate(invalid='ignore'):
            # Simple test, trend 'c'
            _, p_value, _ = coint(df['s1'], df['s2'],
                                  trend='c', maxlag=0, autolag=None)
        return p_value
    except Exception:
        return None


def run_rolling_engle_granger(series1: pd.Series, series2: pd.Series, window: int = 60) -> Optional[pd.Series]:
    """Performs rolling Engle-Granger cointegration test."""
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series) or \
       series1.empty or series2.empty or len(series1) < window or len(series2) < window:
        return None

    df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
    if len(df) < window:
        return None

    rolling_p_values = pd.Series(index=df.index, dtype=float)

    for i in range(window - 1, len(df)):
        window_s1 = df['s1'].iloc[i-window+1:i+1]
        window_s2 = df['s2'].iloc[i-window+1:i+1]
        if window_s1.nunique() < 2 or window_s2.nunique() < 2:
            rolling_p_values.iloc[i] = np.nan
            continue
        try:
            with np.errstate(invalid='ignore'):
                _, p_val, _ = coint(window_s1, window_s2,
                                    trend='c', maxlag=0, autolag=None)
            rolling_p_values.iloc[i] = p_val
        except Exception:
            rolling_p_values.iloc[i] = np.nan

    return rolling_p_values


def run_johansen_test(series1: pd.Series, series2: pd.Series, det_order: int = 0, k_ar_diff: int = 1) -> Optional[Dict]:
    """Performs Johansen cointegration test."""
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series) or \
       series1.empty or series2.empty or len(series1) < 20 or len(series2) < 20:
        return None

    df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
    if len(df) < 20 or df['s1'].nunique() < 2 or df['s2'].nunique() < 2:
        return None

    try:
        # det_order: -1 no intercept, 0 constant, 1 linear trend
        # k_ar_diff: number of lags in levels (p) for VAR(p) model. k_ar_diff = p-1 for VECM.
        # Common choices: k_ar_diff = 1 (implies VAR(2) in levels)
        result = coint_johansen(df, det_order=det_order, k_ar_diff=k_ar_diff)
        return {
            "eigenvalues": result.eig,
            "trace_stat": result.lr1,
            # Critical values for trace statistic (90%, 95%, 99%)
            "trace_crit_vals": result.cvt,
            "max_eig_stat": result.lr2,
            "max_eig_crit_vals": result.cvm,  # Critical values for max eigenvalue statistic
            "cointegrating_vectors": result.evec
        }
    except Exception as e:
        # from ..printing_system import buffered_print # Local import
        # buffered_print(f"Johansen test error: {e}", "ERROR")
        return None


def estimate_spread_half_life(spread_series: pd.Series) -> Optional[float]:
    """Estimates the half-life of a mean-reverting spread series."""
    if not isinstance(spread_series, pd.Series) or spread_series.empty or spread_series.nunique() < 2 or len(spread_series) < 20:
        return None

    spread_series_clean = spread_series.dropna()
    if len(spread_series_clean) < 20:
        return None

    # Calculate lagged spread and delta spread
    spread_lagged = spread_series_clean.shift(1).dropna()
    delta_spread = spread_series_clean.diff().dropna()

    # Align them
    df_hl = pd.DataFrame(
        {'delta': delta_spread, 'lagged': spread_lagged}).dropna()
    if len(df_hl) < 2:
        return None

    # OLS regression: delta_spread = slope * spread_lagged + intercept
    # We only need the slope (lambda in OU notation, often denoted as (rho-1) or gamma)
    try:
        # slope is equivalent to (rho-1) in Y_t - Y_{t-1} = (rho-1)Y_{t-1} + intercept + error
        # or lambda in dY = lambda * Y * dt + ...
        # Adding a constant for the intercept
        X = pd.DataFrame({'const': 1.0, 'lagged_spread': df_hl['lagged']})
        y = df_hl['delta']

        # Using statsmodels for OLS to get p-values etc. if needed, but polyfit is simpler for just slope
        # For simple slope:
        slope, intercept = np.polyfit(df_hl['lagged'], df_hl['delta'], 1)

        if slope >= 0:  # Not mean-reverting or exploding
            return np.inf  # Or None, depending on how you want to signal this

        half_life = -np.log(2) / slope
        return half_life
    except Exception:
        return None


def calculate_rolling_half_life(spread_series: pd.Series, window: int = 60) -> Optional[pd.Series]:
    """Calculates rolling half-life of a spread series."""
    if not isinstance(spread_series, pd.Series) or spread_series.empty or len(spread_series) < window:
        return None

    rolling_hl = pd.Series(index=spread_series.index, dtype=float)

    for i in range(window - 1, len(spread_series)):
        window_spread = spread_series.iloc[i-window+1:i+1]
        hl = estimate_spread_half_life(window_spread)
        rolling_hl.iloc[i] = hl if hl is not None else np.nan

    return rolling_hl
# ---- END OF FILE analysis_utils.py ----
