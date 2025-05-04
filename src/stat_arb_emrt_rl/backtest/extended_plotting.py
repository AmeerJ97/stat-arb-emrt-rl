import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats  # For KDE, skew, kurtosis
from typing import Dict, List, Optional, Any

from .plotting import format_axis, FIXED_COLORS, WINDOW_SIZE
from .utils import find_nearest_date_index
from .analysis_utils import (
    simulate_pair_equity_curve,
    calculate_rolling_correlation,
    run_engle_granger_test, run_rolling_engle_granger,
    run_adf_test, run_rolling_adf_test, run_johansen_test,
    estimate_spread_half_life, calculate_rolling_half_life  # New
)
from ..printing_system import buffered_print


def _plot_simulated_equity_curve_content(data: Dict, ax: plt.Axes) -> Optional[plt.Axes]:
    pair_name = data.get('pair_name', 'N/A')
    try:
        # --- Data Extraction for Simulation ---
        actual_dates_dt = data.get("actual_dates")
        actual_p1 = data.get("actual_p1")
        actual_p2 = data.get("actual_p2")

        clean_dates_dt = data.get("clean_dates")
        # Used to derive z-scores or directly if pre-calculated
        clean_spreads = data.get("clean_spreads")
        clean_means = data.get("clean_means")
        clean_std = data.get("clean_std")

        hedge_ratio = data.get("hedge_ratio", 1.0)
        strategy_k_param = data.get(
            "strategy_k_param", 1.5)  # Strategy's K for entry

        # Basic validation
        if not all([actual_dates_dt, isinstance(actual_p1, np.ndarray), isinstance(actual_p2, np.ndarray),
                    clean_dates_dt, isinstance(clean_spreads, np.ndarray),
                    isinstance(clean_means, np.ndarray), isinstance(clean_std, np.ndarray)]):
            ax.text(0.5, 0.5, "Missing critical data for simulation.",
                    color="red", ha="center", va="center")
            format_axis(ax, f"Simulated Equity - {pair_name} (Data Error)")
            return None

        # --- Align data for simulation ---
        # Primary dates for simulation will be `clean_dates_dt` because z-scores are based on them.
        # We need to map `actual_p1` and `actual_p2` to these `clean_dates_dt`.
        df_raw_prices = pd.DataFrame(
            {'s1': actual_p1, 's2': actual_p2}, index=pd.DatetimeIndex(actual_dates_dt))

        # Create a DataFrame for z-score components using clean_dates_dt
        df_z_components = pd.DataFrame({
            # This is S1' - HR*S2' (prices at clean_dates)
            'spread_val': clean_spreads,
            'mean_val': clean_means,
            'std_val': clean_std
        }, index=pd.DatetimeIndex(clean_dates_dt))

        # Join raw prices to the z-score component dates, forward fill missing raw prices
        sim_df = df_z_components.join(
            df_raw_prices, how='left').ffill().bfill()

        # Drop rows where any essential data might still be NaN after ffill/bfill (e.g., if raw prices don't cover z-score range)
        sim_df.dropna(subset=['s1', 's2', 'spread_val',
                      'mean_val', 'std_val'], inplace=True)

        if sim_df.empty or len(sim_df) < 2:
            ax.text(0.5, 0.5, "Not enough aligned data for simulation.",
                    color="orange", ha="center", va="center")
            format_axis(ax, f"Simulated Equity - {pair_name} (Align Error)")
            return None

        sim_dates_list = sim_df.index.tolist()
        sim_s1_prices = sim_df['s1'].values
        sim_s2_prices = sim_df['s2'].values

        # Calculate Z-scores for simulation using the mean and std corresponding to clean_dates
        # The 'spread_val' from sim_df is S1-HR*S2 using prices at clean_dates.
        # Use np.maximum to avoid division by zero or very small std_val.
        sim_z_scores = (sim_df['spread_val'].values - sim_df['mean_val'].values) / \
            np.maximum(sim_df['std_val'].values, 1e-9)

        # --- Simulation ---
        equity_curve, trades, metrics = simulate_pair_equity_curve(
            dates=sim_dates_list,
            s1_prices=sim_s1_prices,
            s2_prices=sim_s2_prices,
            hedge_ratio=hedge_ratio,
            z_scores=sim_z_scores,
            entry_threshold=strategy_k_param,
            exit_threshold=0.1,  # Exit if |Z| < 0.1 (closer to mean)
            initial_capital=10000.0,
            commission_per_leg_trade=0.50  # Example commission
        )

        if equity_curve.empty:
            ax.text(0.5, 0.5, "Simulation failed or no trades occurred.",
                    color="orange", ha="center", va="center")
            format_axis(ax, f"Simulated Equity - {pair_name} (No Trades)")
            return None

        # --- Plotting Equity Curve ---
        equity_dates_numeric = mdates.date2num(
            equity_curve.index.to_pydatetime())
        ax.plot(equity_dates_numeric, equity_curve.values, color=FIXED_COLORS.get(
            "price1", "cyan"), label="Simulated Equity")

        # --- Mark Trades on Equity Curve ---
        for trade in trades:
            entry_dt_num = mdates.date2num(trade['entry_dt'])
            exit_dt_num = mdates.date2num(trade['exit_dt'])

            entry_equity_idx = find_nearest_date_index(
                equity_dates_numeric, entry_dt_num)
            exit_equity_idx = find_nearest_date_index(
                equity_dates_numeric, exit_dt_num)

            if entry_equity_idx is not None and entry_equity_idx < len(equity_curve):
                ax.plot(equity_dates_numeric[entry_equity_idx], equity_curve.values[entry_equity_idx],
                        marker='^' if trade['direction'] == 'long_spread' else 'v',
                        color='lime' if trade['direction'] == 'long_spread' else 'magenta',
                        markersize=7, alpha=0.9, markeredgecolor='white', mew=0.5)
            if exit_equity_idx is not None and exit_equity_idx < len(equity_curve):
                ax.plot(equity_dates_numeric[exit_equity_idx], equity_curve.values[exit_equity_idx],
                        marker='o', color='grey', markersize=5, alpha=0.9, markeredgecolor='white', mew=0.5)

        # --- Annotate with Metrics ---
        stats_text = (
            f"Final Equity: ${metrics['final_equity']:,.2f}\n"
            f"Net Profit: {metrics['net_profit_pct']:.2%}\n"
            f"CAGR: {metrics['cagr']:.2%}\n"
            f"Sharpe: {metrics['sharpe_ratio']:.2f}\n"
            f"Sortino: {metrics['sortino_ratio']:.2f}\n"
            f"Max DD: {metrics['max_drawdown']:.2%}\n"
            f"Trades: {metrics['total_trades']}\n"
            f"Win Rate: {metrics['win_rate']:.1%}\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Avg Duration: {metrics['avg_trade_duration']:.1f} days"
        )
        ax.text(0.75, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left',  # Adjust alignment
                bbox=dict(boxstyle='round,pad=0.3', fc='#2a2a2a', alpha=0.85, ec='grey'), color='white')

        ax.set_ylabel("Equity ($)", color='lightgrey')
        # This will call ax.legend()
        format_axis(ax, f"Simulated Equity Curve - {pair_name}")
        if ax.get_legend():  # Ensure legend is not None
            # Explicitly set legend location
            ax.legend(loc='upper left', fontsize=8)
        return ax

    except Exception as e:
        buffered_print(
            f"Error plotting simulated equity for {pair_name}: {type(e).__name__} - {e}", "ERROR")
        ax.text(0.5, 0.5, "Error during plot generation",
                color="red", ha="center", va="center")
        format_axis(ax, f"Simulated Equity - {pair_name} (Plot Error)")
        return None


def _plot_trade_return_histogram_content(data: Dict, ax: plt.Axes) -> Optional[plt.Axes]:
    pair_name = data.get('pair_name', 'N/A')
    try:
        trade_objects: List[Any] = data.get(
            "trade_pairs_objects", [])  # These are TradePair objects

        if not trade_objects:
            ax.text(0.5, 0.5, "No trades to analyze.",
                    color="orange", ha="center", va="center")
            format_axis(
                ax, f"Trade Return Histogram - {pair_name} (No Trades)")
            return ax

        pnl_values = []
        for tp_obj in trade_objects:
            if tp_obj.exit_dt is None or not tp_obj.legs:  # Ensure trade is complete
                continue

            # Calculate PnL for the TradePair object
            trade_pnl = 0
            if tp_obj.pnl_from_strategy is not None:  # Prioritize PnL if strategy calculated it
                trade_pnl = tp_obj.pnl_from_strategy
            else:  # Calculate from legs if not
                for leg in tp_obj.legs:
                    if leg.get('exit_price') is None or leg.get('entry_price') is None or leg.get('size') is None:
                        continue

                    leg_pnl = 0
                    if leg['entry_type'] == "long":
                        leg_pnl = (leg["exit_price"] -
                                   leg["entry_price"]) * leg['size']
                    elif leg['entry_type'] == "short":
                        leg_pnl = (leg["entry_price"] -
                                   leg["exit_price"]) * leg['size']
                    trade_pnl += leg_pnl

            if np.isfinite(trade_pnl):
                pnl_values.append(trade_pnl)

        if not pnl_values:
            ax.text(0.5, 0.5, "No valid PnL data from trades.",
                    color="orange", ha="center", va="center")
            format_axis(
                ax, f"Trade Return Histogram - {pair_name} (No PnL Data)")
            return ax

        # ✅ Add this line to create the series
        pnl_series = pd.Series(pnl_values)

        ax.hist(pnl_series, bins='auto', density=True, alpha=0.75,
                color=FIXED_COLORS.get("price2", "orangered"),
                label="Trade PnL Distribution", edgecolor='grey')

        # Add X-axis numerical formatting
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

        pnl_series = pd.Series(pnl_values)

        # Histogram
        ax.hist(pnl_series, bins='auto', density=True, alpha=0.75, color=FIXED_COLORS.get(
            "price2", "orangered"), label="Trade PnL Distribution", edgecolor='grey')

        # KDE
        if len(pnl_series) > 1:  # KDE needs at least 2 points
            try:
                kde = scipy_stats.gaussian_kde(pnl_series)
                x_range = np.linspace(
                    pnl_series.min() - pnl_series.std(), pnl_series.max() + pnl_series.std(), 200)
                ax.plot(x_range, kde(x_range), color=FIXED_COLORS.get(
                    "price1", "deepskyblue"), linewidth=2, label="KDE")
            except Exception as e_kde:  # Catch LinAlgError for singular matrix if all PnLs are same
                buffered_print(
                    f"KDE calculation failed for {pair_name}: {e_kde}", "WARNING")

        # Skewness and Kurtosis
        skew = pnl_series.skew() if len(pnl_series) > 0 else 0.0
        # Fisher's kurtosis (normal == 0)
        kurt = pnl_series.kurtosis() if len(pnl_series) > 0 else 0.0

        stats_text = f"Trades: {len(pnl_series)}\nMean PnL: {pnl_series.mean():.2f}\nStd Dev PnL: {pnl_series.std():.2f}\nSkewness: {skew:.2f}\nKurtosis: {kurt:.2f}"
        text_x_pos = 0.65 if ax.get_legend() and ax.get_legend(
        ).get_window_extent().x0 < ax.get_window_extent().width / 2 else 0.02
        ax.text(text_x_pos, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='left',  # ensure left align
                bbox=dict(boxstyle='round,pad=0.4', fc='#2a2a2a', alpha=0.85, ec='grey'), color='white')

        ax.set_xlabel("Profit/Loss per Trade ($)", color='lightgrey')
        ax.set_ylabel("Density", color='lightgrey')
        # Calls ax.legend()
        format_axis(ax, f"Trade Return Histogram - {pair_name}")
        if ax.get_legend():
            # Ensure legend is explicitly placed
            ax.legend(loc='upper left', fontsize=8)
        return ax

    except Exception as e:
        buffered_print(
            f"Error plotting trade return histogram for {pair_name}: {type(e).__name__} - {e}", "ERROR")
        ax.text(0.5, 0.5, "Error during plot generation",
                color="red", ha="center", va="center")
        format_axis(ax, f"Trade Return Histogram - {pair_name} (Plot Error)")
        return None

    except Exception as e:
        buffered_print(
            f"Error plotting trade return histogram for {pair_name}: {type(e).__name__} - {e}", "ERROR")
        import traceback
        traceback.print_exc()
        ax.text(0.5, 0.5, "Error during plot generation",
                color="red", ha="center", va="center")
        format_axis(ax, f"Trade Return Histogram - {pair_name} (Plot Error)")
        return None


def _plot_rolling_correlation_content(data: Dict, ax: plt.Axes) -> Optional[plt.Axes]:
    pair_name = data.get('pair_name', 'N/A')
    try:
        actual_dates_dt = data.get("actual_dates")  # List of datetime objects
        actual_p1 = data.get("actual_p1")  # np.ndarray
        actual_p2 = data.get("actual_p2")  # np.ndarray

        if not actual_dates_dt or not isinstance(actual_p1, np.ndarray) or not isinstance(actual_p2, np.ndarray) or \
           len(actual_dates_dt) != len(actual_p1) or len(actual_dates_dt) != len(actual_p2):
            ax.text(0.5, 0.5, "Insufficient or mismatched price data for correlation.",
                    color="orange", ha="center", va="center")
            format_axis(ax, f"Rolling Correlation - {pair_name} (Data Error)")
            return None

        series1 = pd.Series(actual_p1, index=pd.DatetimeIndex(actual_dates_dt))
        series2 = pd.Series(actual_p2, index=pd.DatetimeIndex(actual_dates_dt))

        # Use a default window or get from data if provided
        correlation_window = data.get(
            "rolling_corr_window", WINDOW_SIZE * 2)  # e.g., 40 days

        rolling_pearson, rolling_spearman = calculate_rolling_correlation(
            series1, series2, window=correlation_window)

        if rolling_pearson is None or rolling_spearman is None or rolling_pearson.empty or rolling_spearman.empty:
            ax.text(0.5, 0.5, "Could not calculate rolling correlation.",
                    color="orange", ha="center", va="center")
            format_axis(ax, f"Rolling Correlation - {pair_name} (Calc Error)")
            return None

        # Plotting
        # Ensure index is datetime for mdates.date2num
        if not isinstance(rolling_pearson.index, pd.DatetimeIndex):
            rolling_pearson.index = pd.to_datetime(rolling_pearson.index)
        if not isinstance(rolling_spearman.index, pd.DatetimeIndex):
            rolling_spearman.index = pd.to_datetime(rolling_spearman.index)

        corr_dates_numeric_p = mdates.date2num(
            rolling_pearson.index.to_pydatetime())
        corr_dates_numeric_s = mdates.date2num(
            rolling_spearman.index.to_pydatetime())

        ax.plot(corr_dates_numeric_p, rolling_pearson.values, color=FIXED_COLORS.get(
            "price1", "aqua"), label=f"Pearson ({correlation_window}d)", linewidth=1.5)
        ax.plot(corr_dates_numeric_s, rolling_spearman.values, color=FIXED_COLORS.get(
            "price2", "salmon"), label=f"Spearman ({correlation_window}d)", linestyle='--', linewidth=1.5)

        # Annotate correlation breakdowns
        threshold = 0.8
        # Ensure rolling_pearson.values is 1D numpy array before boolean indexing
        pearson_values_np = np.array(rolling_pearson.values)
        breakdown_mask = pearson_values_np < threshold

        # Filter dates and values using the mask
        # Need to get corresponding dates for these breakdown_mask points from corr_dates_numeric_p
        breakdown_dates_plot = corr_dates_numeric_p[breakdown_mask & np.isfinite(
            pearson_values_np)]  # also ensure finite
        breakdown_values_plot = pearson_values_np[breakdown_mask & np.isfinite(
            pearson_values_np)]

        if len(breakdown_dates_plot) > 0:
            ax.scatter(breakdown_dates_plot, breakdown_values_plot,
                       color='red', marker='o', s=20, label=f"Pearson < {threshold}", zorder=5, alpha=0.7)

        ax.axhline(0.0, color='white', linestyle=':', linewidth=1.0, alpha=0.6)
        ax.axhline(1.0, color='lightgreen', linestyle=':',
                   linewidth=1.0, alpha=0.6)
        ax.axhline(-1.0, color='lightcoral',
                   linestyle=':', linewidth=1.0, alpha=0.6)
        ax.set_ylim(-1.05, 1.05)

        ax.set_ylabel("Correlation Coefficient", color='lightgrey')
        format_axis(
            ax, f"Rolling Price Correlation ({correlation_window}d) - {pair_name}")
        return ax

    except Exception as e:
        buffered_print(
            f"Error plotting rolling correlation for {pair_name}: {type(e).__name__} - {e}", "ERROR")
        import traceback
        traceback.print_exc()
        ax.text(0.5, 0.5, "Error during plot generation",
                color="red", ha="center", va="center")
        format_axis(ax, f"Rolling Correlation - {pair_name} (Plot Error)")
        return None


def _plot_cointegration_tests_content(data: Dict, ax: plt.Axes) -> Optional[plt.Axes]:
    pair_name = data.get('pair_name', 'N/A')
    p_value_threshold = data.get(
        'coint_p_value_threshold', 0.05)  # For both EG and ADF
    # Used for rolling EG and rolling ADF
    coint_lookback = data.get('coint_lookback_window', 90)

    try:
        actual_dates_dt = data.get("actual_dates")
        actual_p1_arr = data.get("actual_p1")  # Price series 1
        actual_p2_arr = data.get("actual_p2")  # Price series 2

        # Spread series (calculated as S1 - HR*S2) - this is what we test for stationarity (ADF)
        # and use for half-life. Cointegration is between S1 and S2.
        # Dates for the calculated spread
        clean_dates_spread_dt = data.get("clean_dates")
        # The actual spread values S1-HR*S2
        spread_arr_for_adf = data.get("clean_spreads")
        # Needed if we reconstruct spread from S1, S2
        hedge_ratio = data.get("hedge_ratio", 1.0)

        # Validate price data for cointegration tests (EG, Johansen)
        if not actual_dates_dt or not isinstance(actual_p1_arr, np.ndarray) or \
           not isinstance(actual_p2_arr, np.ndarray) or len(actual_p1_arr) < coint_lookback:
            ax.text(0.5, 0.4, "Insufficient S1/S2 price data for Cointegration tests.",
                    color="orange", ha="center", va="center")
        else:
            s1_prices = pd.Series(
                actual_p1_arr, index=pd.DatetimeIndex(actual_dates_dt))
            s2_prices = pd.Series(
                actual_p2_arr, index=pd.DatetimeIndex(actual_dates_dt))

            # --- Static Engle-Granger (between S1 and S2) ---
            static_eg_p_value = run_engle_granger_test(s1_prices, s2_prices)
            eg_text = f"Static E-G (S1,S2) p-val: {static_eg_p_value:.3f}" if static_eg_p_value is not None else "Static E-G: N/A"
            eg_color = 'lightgreen' if static_eg_p_value is not None and static_eg_p_value < p_value_threshold else 'lightcoral'

            # --- Static Johansen (between S1 and S2) ---
            johansen_results = run_johansen_test(
                s1_prices, s2_prices, det_order=0, k_ar_diff=1)
            jo_text_lines = ["Johansen (S1,S2):"]
            jo_color = 'lightgrey'
            if johansen_results:
                trace_stat_r0 = johansen_results["trace_stat"][0]
                crit_val_95_r0 = johansen_results["trace_crit_vals"][0, 1]
                jo_text_lines.append(
                    f"  Trace(r=0): {trace_stat_r0:.2f} (Crit@95%: {crit_val_95_r0:.2f})")
                if trace_stat_r0 > crit_val_95_r0:
                    jo_text_lines.append("  Suggests Cointegration (Trace)")
                    if eg_color != 'lightgreen':
                        jo_color = 'lightgreen'  # Prioritize EG color if significant
                else:
                    jo_text_lines.append("  No Cointegration (Trace)")
                    if eg_color != 'lightgreen':
                        jo_color = 'lightcoral'
            else:
                jo_text_lines.append("  Test N/A")
                if eg_color != 'lightgreen':
                    jo_color = 'lightcoral'

            static_coint_info_text = f"{eg_text}\n" + "\n".join(jo_text_lines)
            text_bg_coint_color = eg_color if static_eg_p_value is not None and static_eg_p_value < p_value_threshold else jo_color
            text_bg_coint_color = text_bg_coint_color if text_bg_coint_color != 'lightgrey' else '#3a3a3a'
            ax.text(0.02, 0.98, static_coint_info_text, transform=ax.transAxes, fontsize=7,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', fc=text_bg_coint_color, alpha=0.85, ec='grey'), color='white')

            # --- Rolling Engle-Granger (between S1 and S2) ---
            rolling_eg_p_values = run_rolling_engle_granger(
                s1_prices, s2_prices, window=coint_lookback)
            if rolling_eg_p_values is not None and not rolling_eg_p_values.empty:
                if not isinstance(rolling_eg_p_values.index, pd.DatetimeIndex):
                    rolling_eg_p_values.index = pd.to_datetime(
                        rolling_eg_p_values.index)
                dates_num_eg = mdates.date2num(
                    rolling_eg_p_values.index.to_pydatetime())
                ax.plot(dates_num_eg, rolling_eg_p_values.values,
                        label=f"Rolling E-G p-val ({coint_lookback}d)", color=FIXED_COLORS["price_ratio"], linewidth=1.5)
                ax.fill_between(dates_num_eg, 0, p_value_threshold, where=rolling_eg_p_values.values < p_value_threshold,
                                facecolor='green', alpha=0.2, label='E-G Signif. Cointegration')  # Subtler fill
            else:
                ax.text(0.5, 0.7, "Rolling E-G: N/A", color="orange",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)

        # Validate spread data for ADF tests
        # Use same lookback for consistency
        if not clean_dates_spread_dt or not isinstance(spread_arr_for_adf, np.ndarray) or len(spread_arr_for_adf) < coint_lookback:
            ax.text(0.5, 0.6, "Insufficient Spread data for ADF tests.",
                    color="orange", ha="center", va="center")
        else:
            spread_series_for_adf = pd.Series(
                spread_arr_for_adf, index=pd.DatetimeIndex(clean_dates_spread_dt))

            # --- Static ADF Test (on the spread: S1 - HR*S2) ---
            adf_stat, adf_p_value = run_adf_test(
                spread_series_for_adf) if spread_series_for_adf is not None else (None, None)
            adf_text = f"Static ADF (Spread) p-val: {adf_p_value:.3f}" if adf_p_value is not None else "Static ADF: N/A"
            adf_color = 'lightgreen' if adf_p_value is not None and adf_p_value < p_value_threshold else 'lightcoral'
            ax.text(0.02, 0.80, adf_text, transform=ax.transAxes, fontsize=7,  # Position below coint text
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', fc=adf_color, alpha=0.85, ec='grey'), color='white')

            # --- Rolling ADF Test (on the spread) ---
            rolling_adf_p_values = run_rolling_adf_test(
                spread_series_for_adf, window=coint_lookback)  # Use same window
            if rolling_adf_p_values is not None and not rolling_adf_p_values.empty:
                if not isinstance(rolling_adf_p_values.index, pd.DatetimeIndex):
                    rolling_adf_p_values.index = pd.to_datetime(
                        rolling_adf_p_values.index)
                dates_num_adf = mdates.date2num(
                    rolling_adf_p_values.index.to_pydatetime())
                ax.plot(dates_num_adf, rolling_adf_p_values.values,
                        label=f"Rolling ADF p-val ({coint_lookback}d)", color=FIXED_COLORS["normalized_spread"], linestyle='-.', linewidth=1.5)
                ax.fill_between(dates_num_adf, 0, p_value_threshold, where=rolling_adf_p_values.values < p_value_threshold,
                                facecolor='cyan', alpha=0.2, label='ADF Signif. Stationarity')  # Different fill color
            else:
                ax.text(0.5, 0.5, "Rolling ADF: N/A", color="orange",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)

        ax.axhline(p_value_threshold, color='red', linestyle='--',
                   label=f"p-val Threshold ({p_value_threshold:.2f})", linewidth=1.2)
        ax.set_ylim(-0.05, 1.05)
        # Y-axis primarily for p-values
        ax.set_ylabel("p-value", color='lightgrey')
        format_axis(ax, f"Cointegration & Stationarity - {pair_name}")
        if ax.get_legend():
            # Use 'best' to avoid overlap with text
            ax.legend(loc='best', fontsize=7)
        return ax

    except Exception as e:
        buffered_print(
            f"Error plotting cointegration tests for {pair_name}: {type(e).__name__} - {e}", "ERROR")
        import traceback
        traceback.print_exc()
        ax.text(0.5, 0.5, "Error during plot generation",
                color="red", ha="center", va="center")
        format_axis(ax, f"Cointegration - {pair_name} (Plot Error)")
        return None


def _plot_half_life_content(data: Dict, ax: plt.Axes) -> Optional[plt.Axes]:
    pair_name = data.get('pair_name', 'N/A')
    # e.g. 3 months of daily data
    hl_lookback = data.get('half_life_lookback_window', 60)

    try:
        # Need spread series for half-life. 'clean_spreads' is S1-HR*S2 on 'clean_dates'
        clean_dates_dt = data.get("clean_dates")
        clean_spreads_arr = data.get("clean_spreads")

        if not clean_dates_dt or not isinstance(clean_spreads_arr, np.ndarray) or len(clean_spreads_arr) < hl_lookback:
            ax.text(0.5, 0.5, "Insufficient spread data for half-life estimation.",
                    color="orange", ha="center", va="center")
            format_axis(ax, f"Half-Life Estimation - {pair_name} (Data Error)")
            return None

        spread_series = pd.Series(
            clean_spreads_arr, index=pd.DatetimeIndex(clean_dates_dt))

        # --- Static Half-Life ---
        static_hl = estimate_spread_half_life(spread_series)
        hl_text = f"Static Half-Life: {static_hl:.2f} days" if static_hl is not None and np.isfinite(
            static_hl) else "Static Half-Life: N/A (or > inf)"

        # --- Rolling Half-Life ---
        rolling_hl_series = calculate_rolling_half_life(
            spread_series, window=hl_lookback)

        if rolling_hl_series is not None and not rolling_hl_series.empty:
            if not isinstance(rolling_hl_series.index, pd.DatetimeIndex):
                rolling_hl_series.index = pd.to_datetime(
                    rolling_hl_series.index)
            dates_num = mdates.date2num(
                rolling_hl_series.index.to_pydatetime())

            # Filter out extreme values for better plotting scale
            finite_rolling_hl = rolling_hl_series[np.isfinite(rolling_hl_series) & (
                # Cap at 3x window
                rolling_hl_series > 0) & (rolling_hl_series < hl_lookback * 3)]
            if not finite_rolling_hl.empty:
                ax.plot(mdates.date2num(finite_rolling_hl.index.to_pydatetime()), finite_rolling_hl.values,
                        label=f"Rolling Half-Life ({hl_lookback}d)", color=FIXED_COLORS["spread"])
            else:
                ax.text(0.5, 0.6, "Rolling Half-Life: Mostly non-finite values.",
                        color="orange", ha="center", va="center", transform=ax.transAxes)

            # Example thresholds for trade windows
            ax.axhline(5, color='lightgreen', linestyle=':',
                       label="Short-Term Reversion (HL < 5d)")
            ax.axhline(20, color='yellow', linestyle=':',
                       label="Medium-Term Reversion (HL < 20d)")
            # ax.set_ylim(0, hl_lookback * 1.5) # Dynamic Y-axis based on typical values
        else:
            ax.text(0.5, 0.6, "Rolling Half-Life: N/A", color="orange",
                    ha="center", va="center", transform=ax.transAxes)

        ax.text(0.02, 0.98, hl_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='#2a2a2a', alpha=0.8, ec='grey'), color='white')

        ax.set_ylabel("Half-Life (days)", color='lightgrey')
        format_axis(ax, f"Spread Half-Life Estimation - {pair_name}")
        if ax.get_legend():
            ax.legend(loc='upper right', fontsize=8)
        return ax

    except Exception as e:
        buffered_print(
            f"Error plotting half-life for {pair_name}: {type(e).__name__} - {e}", "ERROR")
        ax.text(0.5, 0.5, "Error during plot generation",
                color="red", ha="center", va="center")
        format_axis(ax, f"Half-Life Estimation - {pair_name} (Plot Error)")
        return None


def _plot_total_strategy_equity_content(data: Dict, ax: plt.Axes) -> Optional[plt.Axes]:
    """
    Plots the total strategy equity curve. 
    'data' here is expected to be the specific dictionary containing the total equity curve,
    not the per-pair data.
    """
    try:
        # pd.Series with DatetimeIndex
        equity_curve_series = data.get("total_equity_curve")
        initial_capital = data.get("initial_capital", 10000.0)

        if not isinstance(equity_curve_series, pd.Series) or equity_curve_series.empty:
            ax.text(0.5, 0.5, "Total strategy equity data not available.",
                    color="orange", ha="center", va="center")
            format_axis(ax, "Total Strategy Equity (Data Error)")
            return None

        # Calculate metrics for the total strategy equity
        # The simulate_pair_equity_curve's metrics calculator expects a list of trades, which we don't have for the *overall* strategy in this simplified way.
        # So, we'll calculate some high-level metrics directly.
        metrics = {}
        if len(equity_curve_series) >= 2:
            final_equity = equity_curve_series.iloc[-1]
            metrics['Final Equity'] = f"${final_equity:,.2f}"
            metrics['Net Profit %'] = f"{((final_equity - initial_capital) / initial_capital):.2%}"

            num_days = (
                equity_curve_series.index[-1] - equity_curve_series.index[0]).days
            if num_days > 0 and initial_capital > 0:
                metrics['CAGR'] = f"{((final_equity / initial_capital)**(252.0/num_days) - 1):.2%}"
            else:
                metrics['CAGR'] = "N/A"

            daily_returns = equity_curve_series.pct_change().dropna()
            if len(daily_returns) > 1:
                sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * \
                    np.sqrt(252) if np.std(daily_returns) > 1e-9 else 0.0
                metrics['Sharpe Ratio'] = f"{sharpe:.2f}" if np.isfinite(
                    sharpe) else "N/A"
            else:
                metrics['Sharpe Ratio'] = "N/A"

            peak = equity_curve_series.expanding(min_periods=1).max()
            drawdown = (equity_curve_series - peak) / peak
            max_dd = abs(
                drawdown.min()) if not drawdown.empty and not drawdown.isnull().all() else 0.0
            metrics['Max Drawdown'] = f"{max_dd:.2%}"
        else:
            metrics['Info'] = "Not enough data points"

        dates_numeric = mdates.date2num(
            equity_curve_series.index.to_pydatetime())
        ax.plot(dates_numeric, equity_curve_series.values, color=FIXED_COLORS.get(
            "price1", "lime"), label="Total Strategy Equity")

        stats_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', fc='#2a2a2a', alpha=0.85, ec='grey'), color='white')

        ax.set_ylabel("Portfolio Value ($)", color='lightgrey')
        format_axis(ax, "Total Strategy Equity Curve")
        if ax.get_legend():
            ax.legend(loc='upper left', fontsize=8)
        return ax

    except Exception as e:
        buffered_print(
            f"Error plotting total strategy equity: {type(e).__name__} - {e}", "ERROR")
        ax.text(0.5, 0.5, "Error generating total equity plot",
                color="red", ha="center", va="center")
        format_axis(ax, "Total Strategy Equity (Plot Error)")
        return None
