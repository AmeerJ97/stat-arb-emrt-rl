# ───────────────────────────────────────────────
# Standard Library Imports
# ───────────────────────────────────────────────
from typing import Dict, List, Tuple, Optional

# ───────────────────────────────────────────────
# Third-Party Package Imports
# ───────────────────────────────────────────────
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# For LineCollection artists
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.patches import Patch  # For legend patches if needed
import numpy as np
import backtrader as bt

# ───────────────────────────────────────────────
# Project-Specific Imports
# ───────────────────────────────────────────────
from .utils import find_nearest_date_index, COLOR_CYCLE, get_pair_colors
from ..printing_system import buffered_print
from ..financial_loader import FinancialLoader

WINDOW_SIZE = 20
FIXED_COLORS = {  # Ensure this is defined or imported
    "spread": "#00FFAA", "mean": "#FFD700", "static_mean": "#FF00FF",
    "bands": "#666666", "price1": "#1F77B4", "price2": "#FF7F0E",
    "normalized_spread": "#FF69B4", "price_ratio": "#00FFFF",
    "rolling_mean_generic": "#FFFACD", "adaptive_k_annotation_bg": "#666666",
    "total_equity": "#4CAF50",
}


def plot_pair_spreads(results: List[bt.Strategy],
                      pairs_to_plot: List[str],
                      end_date: str,
                      initial_capital: float = 10000.0,  # New param
                      total_strategy_equity_data: Optional[pd.Series] = None):
    from .gui import PairNavigator

    plt.style.use("dark_background")

    if not results:
        buffered_print(
            "No strategy results provided to plot_pair_spreads.", "ERROR")
        return
    strategy = results[0]

    # --- Validate Strategy Attributes ---
    required_attrs = {
        'analyzers': ['spread_tracker', 'returns'], 'trade_recorder': None,
        # Added K for default
        'price_history': None, 'start_date': None, 'params': ['pairs', 'K', 'cointegration_lookback']
    }
    for attr, sub_attrs in required_attrs.items():
        if not hasattr(strategy, attr):
            buffered_print(
                f"Strategy object missing top-level attribute: {attr}", "ERROR")
            return
        # Check nested attributes (e.g., strategy.analyzers.spread_tracker)
        if sub_attrs:
            parent_obj = getattr(strategy, attr)
            for sub_attr in sub_attrs:
                if not hasattr(parent_obj, sub_attr):
                    buffered_print(
                        f"Strategy object missing nested attribute: {attr}.{sub_attr}", "ERROR")
                    return
    # --- End Validation ---

    spread_analysis = strategy.analyzers.spread_tracker.get_analysis()
    trade_analysis_dicts = strategy.trade_recorder.get_trades()
    loader = FinancialLoader()

    returns_analyzer = strategy.analyzers.returns.get_analysis()
    all_returns = pd.Series(dtype=np.float32)
    if "returns" in returns_analyzer and returns_analyzer["returns"]:
        try:
            all_returns = pd.Series(
                returns_analyzer["returns"],
                index=pd.date_range(
                    start=pd.to_datetime(strategy.start_date),
                    periods=len(returns_analyzer["returns"]),
                    freq="B",  # Business days often make more sense for returns
                ),
            ).astype(np.float32)
        except Exception as e:
            buffered_print(
                f"Error creating all_returns series: {e}", "WARNING")

    pair_plot_data_list: List[Dict] = []

    # --- Strategy Level Parameters ---
    strategy_k_param = getattr(strategy.params, 'K', 1.5)
    rolling_corr_window = getattr(
        strategy.params, 'correlation_window', WINDOW_SIZE * 2)
    # From MeanReversionStrategy.params
    coint_lookback_window = getattr(
        strategy.params, 'cointegration_lookback', 90)
    # Example, make it a param if needed
    coint_p_value_threshold = getattr(
        strategy.params, 'coint_significance_level', 0.05)
    half_life_lookback_window = getattr(
        strategy.params, 'half_life_window', 60)  # Example

    for pair_idx, pair_name_str in enumerate(pairs_to_plot):
        t1, t2 = pair_name_str.split("/")
        # current_pair_as_tuple = (t1, t2)
        current_pair_as_tuple = tuple(pair_name_str.split('/'))

        current_trade_pair_objects = [
            tp for tp in strategy.trade_recorder.trade_pairs
            if tp.pair == current_pair_as_tuple
        ]

        if pair_name_str not in spread_analysis["spread_data"]:
            buffered_print(
                f"Plotting: Missing spread_data for {pair_name_str}. Skipping.", "WARNING")
            continue

        dates_raw = spread_analysis["dates"].get(pair_name_str, [])
        raw_data_map = {
            "spreads": spread_analysis["spread_data"].get(pair_name_str, []),
            "means": spread_analysis["means"].get(pair_name_str, []),
            "std_devs": spread_analysis["std_devs"].get(pair_name_str, []),
            "k_values": spread_analysis["k_values"].get(pair_name_str, [])
        }
        if not dates_raw or not all(len(lst) == len(dates_raw) for lst in raw_data_map.values()):
            buffered_print(
                f"Plotting: Data inconsistency or missing lists for {pair_name_str}. Skipping.", "WARNING")
            continue

        valid_indices = [
            i for i in range(len(dates_raw))
            if not np.isnan(raw_data_map["spreads"][i])  # Only check spreads
        ]
        if len(valid_indices) < 2:
            buffered_print(
                f"Plotting: Insufficient valid (non-NaN) data points for {pair_name_str}. Skipping.", "WARNING")
            continue

        # Hedge ratio (ensure this logic is robust)
        hedge_ratio_for_pair = 1.0
        if hasattr(strategy, 'hedge_ratios') and isinstance(strategy.hedge_ratios, dict):
            hedge_ratio_for_pair = strategy.hedge_ratios.get(
                current_pair_as_tuple, 1.0)

        clean_datetime_dates = [pd.to_datetime(
            dates_raw[i]) for i in valid_indices]
        clean_spreads = np.array([raw_data_map["spreads"][i]
                                 for i in valid_indices], dtype=np.float32)
        clean_means = np.array([raw_data_map["means"][i]
                               for i in valid_indices], dtype=np.float32)
        clean_std = np.array([raw_data_map["std_devs"][i]
                             for i in valid_indices], dtype=np.float32)
        clean_k = np.array([raw_data_map["k_values"][i]
                           for i in valid_indices], dtype=np.float32)

        _actual_price_hist = strategy.price_history.get(
            current_pair_as_tuple, {"dates": [], "s1": [], "s2": []})
        _actual_datetime_dates = [pd.to_datetime(
            d) for d in _actual_price_hist.get("dates", [])]
        _actual_p1 = np.array(
            _actual_price_hist.get("s1", []), dtype=np.float32)
        _actual_p2 = np.array(
            _actual_price_hist.get("s2", []), dtype=np.float32)

        normalized_df = loader.get_normalized_pair(
            t1, t2,
            strategy.start_date.strftime(
                "%Y-%m-%d") if strategy.start_date else "1900-01-01",
            pd.to_datetime(end_date).strftime("%Y-%m-%d")
        )

        norm_spread_series: Optional[pd.Series] = None
        price_ratio_series: Optional[pd.Series] = None

        if normalized_df is not None and not normalized_df.empty:
            norm_t1_col, norm_t2_col = f"Normalized {t1}", f"Normalized {t2}"

            if norm_t1_col in normalized_df.columns and norm_t2_col in normalized_df.columns:
                norm_spread_series = (
                    normalized_df[norm_t1_col] - normalized_df[norm_t2_col]).dropna()
                safe_t2_norm = normalized_df[norm_t2_col].replace(
                    0, np.nan)  # Avoid division by zero
                price_ratio_series = (
                    normalized_df[norm_t1_col] / safe_t2_norm).dropna()

        actual_price_hist = strategy.price_history.get(
            current_pair_as_tuple, {"dates": [], "s1": [], "s2": []})
        actual_datetime_dates = [pd.to_datetime(
            d) for d in actual_price_hist.get("dates", [])]
        actual_p1 = np.array(actual_price_hist.get("s1", []), dtype=np.float32)
        actual_p2 = np.array(actual_price_hist.get("s2", []), dtype=np.float32)

        current_trade_pair_objects = [
            tp for tp in strategy.trade_recorder.trade_pairs
            if tp.pair == current_pair_as_tuple
        ]
        # trade_colors = [COLOR_CYCLE[tp.color_idx %
        #                             len(COLOR_CYCLE)] for tp in current_trade_pair_objects]
        trade_colors = [
            COLOR_CYCLE[i % len(COLOR_CYCLE)]
            for i in range(len(current_trade_pair_objects))
        ]

        stats = {"sharpe": 0.0, "max_drawdown": 0.0}
        current_pair_trades_dicts = [t for t in trade_analysis_dicts if t.get(
            "pair") == current_pair_as_tuple and t.get("exit_dt")]

        if current_pair_trades_dicts and not all_returns.empty:
            active_trade_days = pd.DatetimeIndex([])
            for trade_dict in current_pair_trades_dicts:  # Iterate through filtered trade dictionaries
                try:
                    entry_dt = pd.to_datetime(trade_dict["entry_dt"])
                    # exit_dt is guaranteed by filter
                    exit_dt = pd.to_datetime(trade_dict["exit_dt"])
                    active_trade_days = active_trade_days.union(
                        pd.date_range(entry_dt, exit_dt, freq='B'))

                except Exception:
                    pass

            relevant_returns = all_returns[all_returns.index.isin(
                active_trade_days.unique())]

            if len(relevant_returns) >= 2:
                mean_ret, std_ret = relevant_returns.mean(), relevant_returns.std()
                stats["sharpe"] = (
                    mean_ret / (std_ret if std_ret > 1e-9 else 1e-9)) * np.sqrt(252)
                cumulative_ret = (1 + relevant_returns).cumprod()
                peak = cumulative_ret.expanding(min_periods=1).max()
                drawdown = (cumulative_ret - peak) / peak
                stats["max_drawdown"] = abs(
                    drawdown.min()) if not drawdown.empty else 0.0
            stats = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                     for k, v in stats.items()}

        # Fallback: if not in hedge_ratios, try to get from an active trade for the pair
        elif hasattr(strategy, 'active_trades') and strategy.active_trades.get(current_pair_as_tuple):
            active_trade_info = strategy.active_trades[current_pair_as_tuple]
            if isinstance(active_trade_info, dict):  # Ensure it's the expected dict
                hedge_ratio_for_pair = active_trade_info.get(
                    "entry_hedge_ratio", 1.0)

        pair_plot_data_list.append({
            "pair_index": pair_idx, "pair_name": pair_name_str,
            "clean_dates": clean_datetime_dates, "clean_spreads": clean_spreads,
            "clean_means": clean_means, "clean_std": clean_std, "clean_k": clean_k,
            "normalized_df": normalized_df,
            "actual_dates": actual_datetime_dates, "actual_p1": actual_p1, "actual_p2": actual_p2,
            "trade_pairs_objects": current_trade_pair_objects, "trade_colors": trade_colors,
            "stats": stats, "colors": FIXED_COLORS,
            "rolling_p1": pd.Series(actual_p1).rolling(WINDOW_SIZE, min_periods=1).mean().values if len(actual_p1) > 0 else np.array([]),
            "rolling_p2": pd.Series(actual_p2).rolling(WINDOW_SIZE, min_periods=1).mean().values if len(actual_p2) > 0 else np.array([]),
            "rolling_spread": pd.Series(clean_spreads).rolling(WINDOW_SIZE, min_periods=1).mean().values if len(clean_spreads) > 0 else np.array([]),
            "rolling_norm_spread": norm_spread_series.rolling(WINDOW_SIZE, min_periods=1).mean() if norm_spread_series is not None and not norm_spread_series.empty else None,
            "rolling_ratio": price_ratio_series.rolling(WINDOW_SIZE, min_periods=1).mean() if price_ratio_series is not None and not price_ratio_series.empty else None,
            "hedge_ratio": hedge_ratio_for_pair,
            "strategy_k_param": strategy_k_param,
            "rolling_corr_window": rolling_corr_window,
            "coint_lookback_window": coint_lookback_window,
            "coint_p_value_threshold": coint_p_value_threshold,
            "half_life_lookback_window": half_life_lookback_window,
        })

    if pair_plot_data_list:
        # --- Prepare total strategy equity data for PairNavigator ---
        navigator_total_equity_data = None
        if total_strategy_equity_data is not None:
            navigator_total_equity_data = {
                "total_equity_curve": total_strategy_equity_data,
                "initial_capital": initial_capital
            }

        navigator = PairNavigator(
            pairs_data=pair_plot_data_list,
            colors=FIXED_COLORS.copy(),
            headless=False,
            total_strategy_equity_data=navigator_total_equity_data  # Pass here
        )
    else:
        buffered_print(
            "No valid data to visualize for any pair after processing.", "ERROR")


def format_axis(ax: plt.Axes, title: str):
    """Formats a Matplotlib Axes object with a dark theme and improved layout."""
    ax.set_title(title, color="white", fontsize=14, pad=15)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12, minticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    # Reverted to 20 for more Y-axis lines
    ax.yaxis.set_major_locator(plt.MaxNLocator(20, prune='both'))
    ax.tick_params(axis="x", rotation=30, labelsize=9, colors="lightgrey")
    ax.tick_params(axis="y", labelsize=9, colors="lightgrey")
    ax.grid(True, color="#555555", linestyle=":", alpha=0.5)
    ax.set_facecolor("#2a2a2a")

    for spine in ax.spines.values():
        spine.set_edgecolor('grey')

    handles, labels = ax.get_legend_handles_labels()
    unique_labels_dict = {}
    if handles:
        for handle, label in zip(handles, labels):
            if label and label not in unique_labels_dict:
                unique_labels_dict[label] = handle

    if unique_labels_dict:
        legend = ax.legend(unique_labels_dict.values(), unique_labels_dict.keys(),
                           loc="upper left", facecolor="#333333", edgecolor="darkgrey",
                           fontsize=8, framealpha=0.85)
        if legend:
            for text_obj in legend.get_texts():
                text_obj.set_color("white")
            if legend.get_title():
                legend.get_title().set_color("white")


def add_trade_markers(ax: plt.Axes, trade_pair_objects: List[any], trade_colors: List[str],
                      x_dates_numeric: np.ndarray, y_values: np.ndarray,
                      axis_type: str, asset_name: Optional[str] = None,
                      pair_name_for_debug: str = "") -> Tuple[List[plt.Artist], List[str]]:
    handles: List[plt.Artist] = []
    labels: List[str] = []

    if not trade_pair_objects:
        return handles, labels

    # Define styles for different marker types
    entry_filled_marker_style = {
        's': 90 * 1.2,
        'facecolor': None,  # To be set by 'color'
        'edgecolor': "white",
        'linewidth': 1.2,
        'zorder': 100
    }
    entry_unfilled_marker_style = {  # For '^', 'v' if treated as unfilled for edge purposes
        's': 90,
        'color': None,  # To be set by 'color'
        'zorder': 100
        # No edgecolor or linewidth if they cause issues, or use a very thin one of the same color
    }
    exit_marker_style = {  # For 'x'
        's': 90 * 1.5,
        'color': None,  # To be set by 'color'
        'zorder': 101
        # No edgecolor or linewidth for 'x'
    }

    marker_size, edge_width = 90, 1.2

    for i, trade_obj in enumerate(trade_pair_objects):
        if not hasattr(trade_obj, 'legs'):
            continue

        current_trade_color = trade_colors[i % len(trade_colors)]

        # Handle spread-level plots (e.g., equity curve, actual spread)
        # Using 'o' for entry (filled) and 'x' for exit (unfilled)
        if axis_type in ["spread", "normalized_spread", "price_ratio", "simulated_equity"]:
            # Entry marker ('o')
            if trade_obj.entry_dt:
                entry_date_num = mdates.date2num(trade_obj.entry_dt)
                entry_idx = find_nearest_date_index(
                    x_dates_numeric, entry_date_num)
                if entry_idx is not None and 0 <= entry_idx < len(y_values):
                    style = entry_filled_marker_style.copy()
                    # 'color' kwarg for scatter sets facecolor for filled markers if 'facecolor' is None
                    ax.scatter(
                        x_dates_numeric[entry_idx], y_values[entry_idx],
                        marker='o', color=current_trade_color, **style
                    )

            # Exit marker ('x')
            if trade_obj.exit_dt:
                exit_date_num = mdates.date2num(trade_obj.exit_dt)
                exit_idx = find_nearest_date_index(
                    x_dates_numeric, exit_date_num)
                if exit_idx is not None and 0 <= exit_idx < len(y_values):
                    style = exit_marker_style.copy()
                    style['color'] = current_trade_color
                    ax.scatter(
                        x_dates_numeric[exit_idx], y_values[exit_idx],
                        marker='x', **style
                    )

        # Handle asset-level plots (e.g., price comparison, normalized prices)
        # Using '^'/'v' for entry (can be filled) and 'x' for exit
        elif axis_type in ["price_comparison", "normalized_price"] and asset_name:
            for leg in trade_obj.legs:
                if leg['asset'] == asset_name:
                    # Entry marker ('^' or 'v')
                    if leg['entry_dt']:
                        entry_date_num = mdates.date2num(leg['entry_dt'])
                        entry_idx = find_nearest_date_index(
                            x_dates_numeric, entry_date_num)
                        if entry_idx is not None and 0 <= entry_idx < len(y_values):
                            marker_char = '^' if leg['entry_type'] == 'long' else 'v'
                            # Triangles are path-based and can take facecolor/edgecolor
                            style = entry_filled_marker_style.copy()
                            # Slightly smaller for triangles maybe
                            style['s'] = 90
                            # More subtle edge for triangles
                            style['edgecolor'] = 'lightgrey'
                            style['linewidth'] = 1.0
                            ax.scatter(
                                x_dates_numeric[entry_idx], y_values[entry_idx],
                                marker=marker_char, color=current_trade_color, **style
                            )

                    # Exit marker ('x')
                    if leg['exit_dt']:
                        exit_date_num = mdates.date2num(leg['exit_dt'])
                        exit_idx = find_nearest_date_index(
                            x_dates_numeric, exit_date_num)
                        if exit_idx is not None and 0 <= exit_idx < len(y_values):
                            style = exit_marker_style.copy()
                            style['color'] = current_trade_color
                            ax.scatter(
                                x_dates_numeric[exit_idx], y_values[exit_idx],
                                marker='x', **style
                            )

    return handles, labels
