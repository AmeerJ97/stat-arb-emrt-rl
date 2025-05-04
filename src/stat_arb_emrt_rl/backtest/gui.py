# ───────────────────────────────────────────────
# Standard Library Imports
# ───────────────────────────────────────────────
from .extended_plotting import (
    _plot_simulated_equity_curve_content,
    _plot_trade_return_histogram_content,
    _plot_rolling_correlation_content,
    _plot_cointegration_tests_content,
    _plot_half_life_content,
    _plot_total_strategy_equity_content
)
from .utils import (
    find_nearest_date_index,
    COLOR_CYCLE,
    get_pair_colors,
    sanitize_filename,
)
from ..printing_system import buffered_print
from .plotting import add_trade_markers, format_axis, FIXED_COLORS, WINDOW_SIZE
try:
    import mplcursors
except ImportError:
    mplcursors = None
from matplotlib.collections import PathCollection, LineCollection
# Import WINDOW_SIZE

import pandas as pd
import numpy as np
# Corrected import
from matplotlib.backend_bases import Event, ResizeEvent, CloseEvent, MouseEvent, KeyEvent
from matplotlib.image import AxesImage
from matplotlib.text import Text
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.widgets import Button, TextBox
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import sys
import time
import random
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
from collections import defaultdict

# ───────────────────────────────────────────────
# Third-Party Package Imports
# ───────────────────────────────────────────────
import matplotlib
# Only use TkAgg if running in interactive mode (not discovery/headless)
# For headless/discovery mode, use Agg backend which doesn't require tkinter
if not ("--discover" in sys.argv or "--rl-backtest" in sys.argv):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        # Fall back to Agg if TkAgg is not available
        matplotlib.use("Agg")
else:
    # Use Agg backend for headless/discovery modes
    matplotlib.use("Agg")


try:
    import plotly.graph_objects as go
except ImportError:
    go = None

# ───────────────────────────────────────────────
# Project-Specific Imports
# ───────────────────────────────────────────────
# Assuming plotting.py is in the same directory (e.g. backtest/plotting.py and backtest/gui.py)


class PairNavigator:
    """
    Manages interactive visualization of trading pair data, including multiple plot types,
    navigation between pairs, and window geometry persistence.
    """

    def __init__(self,
                 pairs_data: List[Dict],
                 colors: Optional[Dict] = None,
                 headless: bool = False,
                 total_strategy_equity_data: Optional[Dict] = None):
        # ... (self.figure_types and self._figure_visibility are already correctly defined from previous response)
        # ... (rest of __init__ as in your last provided version)
        buffered_print("PairNavigator: Initializing...", "DEBUG")
        self._original_headless_param = headless
        self.headless = headless
        self.pairs_data = pairs_data
        self.colors = colors if colors is not None else FIXED_COLORS.copy()
        self.total_strategy_equity_data = total_strategy_equity_data
        self.window_geometries: Dict[str, str] = defaultdict(str)
        self.closed_geometries: Dict[str, str] = {}
        self.last_pair_geometries: Dict[str, str] = {}
        self.current_figures: List[plt.Figure] = []
        self.current_index: int = 0
        self.figure_types: List[str] = [
            "spread", "normalized_price", "normalized_spread",
            "price_comparison", "price_ratio",
            "simulated_equity", "trade_return_histogram", "rolling_correlation",
            "cointegration_tests", "half_life_estimation",
            "total_strategy_equity",
            "trade_legend",
        ]
        self._figure_visibility: Dict[str, bool] = {
            ft: (ft not in ["trade_legend", "simulated_equity",
                            "trade_return_histogram", "rolling_correlation",
                            "cointegration_tests", "half_life_estimation",
                            "total_strategy_equity"])
            for ft in self.figure_types
        }
        for ft_new in ["trade_legend", "simulated_equity", "trade_return_histogram",
                       "rolling_correlation", "cointegration_tests",
                       "half_life_estimation", "total_strategy_equity"]:
            self._figure_visibility[ft_new] = False
        self.last_resize_time: float = time.time()
        self.resize_debounce_interval: float = 0.3
        self.should_exit: bool = False
        self.control_fig: Optional[plt.Figure] = None
        self._button_refs: List[Button] = []
        self.search_box: Optional[TextBox] = None
        self.search_focused: bool = False
        self.cursors: List[Any] = []  # List of mplcursors.Cursor if available
        gui_init_successful = False
        if not self.headless:
            buffered_print(
                "PairNavigator: Attempting GUI initialization.", "DEBUG")
            try:
                self._init_gui()
                gui_init_successful = True
                buffered_print(
                    "PairNavigator: _init_gui completed successfully.", "DEBUG")
            except Exception as init_error:
                buffered_print(
                    f"GUI initialization failed: {type(init_error).__name__} - {init_error}. Falling back to headless mode.", "CRITICAL")
                import traceback
                traceback.print_exc()
                self.headless = True
        buffered_print(
            f"PairNavigator: Effective self.headless = {self.headless}", "DEBUG")
        if self.pairs_data or self.total_strategy_equity_data:
            if not self.headless:
                buffered_print(
                    "PairNavigator: Interactive mode. Calling _refresh_display_content_only() and render().", "DEBUG")
                try:
                    self._refresh_display_content_only()
                    self.render()
                except Exception as display_error:
                    buffered_print(
                        f"Error during initial interactive display/render: {type(display_error).__name__} - {display_error}", "CRITICAL")
                    if not self._original_headless_param:
                        buffered_print(
                            "PairNavigator: Critical error in interactive mode. Attempting fallback to headless.", "CRITICAL")
                        self.headless = True
                    self._render_headless()
            else:
                buffered_print(
                    "PairNavigator: Headless mode active. Calling _render_headless().", "INFO")
                self._render_headless()
        elif not self.headless and gui_init_successful:
            buffered_print(
                "PairNavigator: GUI initialized but no data to display. Waiting for events or data.", "INFO")
            self.render()
        else:
            buffered_print(
                "PairNavigator: No data to display or GUI not available. Navigator will not render plots.", "INFO")

    def _on_figure_close(self, event: CloseEvent, fig_type: str):
        fig = event.canvas.figure
        try:
            if hasattr(fig.canvas, 'manager') and hasattr(fig.canvas.manager, 'window') and \
               hasattr(fig.canvas.manager.window, 'geometry'):
                geom = fig.canvas.manager.window.geometry()
                self.closed_geometries[fig_type] = geom
                self.window_geometries[fig_type] = geom
                buffered_print(
                    f"Captured geometry for closed window '{fig_type}': {geom}", "DEBUG")
            else:
                buffered_print(
                    f"Cannot get geometry for closing '{fig_type}': manager or window issue.", "DEBUG")
        except Exception as e:
            buffered_print(
                f"Error getting geometry for closing window '{fig_type}': {e}", "DEBUG")

        self._figure_visibility[fig_type] = False

        if fig in self.current_figures:
            self.current_figures.remove(fig)

    def _create_figures(self):
        if not self.pairs_data or not (0 <= self.current_index < len(self.pairs_data)):
            buffered_print(
                f"Plotting: No data or invalid index ({self.current_index}) for creating figures. Cleaning up.", "WARNING")
            self._cleanup_figures(preserve_control_panel=True)
            return

        current_pair_plot_config = self.pairs_data[self.current_index]
        self._cleanup_figures(preserve_control_panel=True)

        newly_created_and_visible_figures: List[plt.Figure] = []

        figure_creator_map: Dict[str, Callable[[Optional[Dict]], Optional[plt.Figure]]] = {
            "spread": self._create_spread_plot,
            "normalized_price": self._create_normalized_price_plot,
            "normalized_spread": self._create_normalized_spread_plot,
            "price_comparison": self._create_price_comparison_plot,
            "price_ratio": self._create_price_ratio_plot,
            "trade_legend": self._create_legend_window,
            "simulated_equity": self._create_simulated_equity_plot,
            "trade_return_histogram": self._create_trade_return_histogram_plot,
            "rolling_correlation": self._create_rolling_correlation_plot,
            "cointegration_tests": self._create_cointegration_tests_plot,
            "half_life_estimation": self._create_half_life_estimation_plot,
            "total_strategy_equity": self._create_total_strategy_equity_plot,
        }

        for fig_type in self.figure_types:
            if not self._figure_visibility.get(fig_type, False):
                continue

            creator_func = figure_creator_map.get(fig_type)
            if not creator_func:
                buffered_print(
                    f"No creator function defined for figure type: '{fig_type}'", "WARNING")
                continue

            created_fig: Optional[plt.Figure] = None
            try:
                created_fig = creator_func(current_pair_plot_config)
                if created_fig:
                    created_fig._figure_type = fig_type
                    self._apply_saved_geometry(created_fig, fig_type)
                    created_fig.canvas.mpl_connect('close_event', partial(
                        self._on_figure_close, fig_type=fig_type))
                    newly_created_and_visible_figures.append(created_fig)
                    created_fig.canvas.draw_idle()
                else:
                    buffered_print(
                        f"Creator for '{fig_type}' returned None for pair '{current_pair_plot_config.get('pair_name', 'N/A')}'. Figure not created.", "DEBUG")
            except Exception as e:
                pair_name_debug = current_pair_plot_config.get(
                    'pair_name', 'UNKNOWN_PAIR')
                buffered_print(
                    f"CRITICAL ERROR creating figure '{fig_type}' for pair '{pair_name_debug}': {type(e).__name__} - {e}", "ERROR")
                if created_fig and plt.fignum_exists(created_fig.number):
                    plt.close(created_fig)

        self.current_figures = newly_created_and_visible_figures
        self._update_titles()
        self._lift_control_panel()

    def _create_legend_window(self, data: Dict) -> Optional[plt.Figure]:
        fig = None
        pair_name = data.get('pair_name', 'N/A')
        try:
            trade_pair_objects: List[Any] = data.get("trade_pairs_objects", [])
            trade_colors: List[str] = data.get("trade_colors", [])

            fig = plt.figure(
                figsize=(7, max(2 + len(trade_pair_objects) * 0.6, 4)),
                facecolor="#1a1a1a",
                num=f"Trade Legend - {pair_name}"
            )
            ax = fig.add_subplot(111)
            ax.axis('off')

            legend_handles: List[plt.Artist] = []
            legend_labels: List[str] = []

            if not trade_pair_objects:
                ax.text(0.5, 0.5, "No trades for this pair.",
                        color="white", ha="center", va="center", fontsize=12)
            else:
                for i, trade_obj in enumerate(trade_pair_objects):
                    if not all(hasattr(trade_obj, attr) for attr in ['entry_dt', 'exit_dt', 'direction', 'color_idx']):
                        buffered_print(
                            f"Legend: Malformed trade object at index {i} for pair {pair_name}", "WARNING")
                        continue

                    color = trade_colors[i % len(
                        trade_colors)] if trade_colors else FIXED_COLORS['price1']

                    entry_marker_char = '^'
                    if hasattr(trade_obj, 'direction') and trade_obj.direction and trade_obj.direction.lower() == "short":
                        entry_marker_char = 'v'

                    entry_legend_text = f"Trade {i+1} Entry"
                    if hasattr(trade_obj, 'direction') and trade_obj.direction:
                        entry_legend_text += f" ({trade_obj.direction.capitalize()})"

                    entry_handle = Line2D([0], [0], marker=entry_marker_char, color=color,
                                          markeredgecolor='white', markersize=8, linestyle='None')
                    legend_handles.append(entry_handle)
                    legend_labels.append(entry_legend_text)

                    if hasattr(trade_obj, 'exit_dt') and trade_obj.exit_dt:
                        exit_handle = Line2D([0], [0], marker='x', color=color,
                                             markeredgecolor='white', markersize=8, linestyle='None')
                        legend_handles.append(exit_handle)
                        legend_labels.append(f"Trade {i+1} Exit")

            if legend_handles:
                legend = ax.legend(
                    legend_handles, legend_labels, loc='center', ncol=min(2, (len(legend_handles) + 1) // 2),
                    frameon=True, facecolor="#2a2a2a", edgecolor="grey", fontsize=9,
                    title=f"Trades for {pair_name}", title_fontsize=11
                )
                if legend:  # Check if legend was created
                    for text_obj in legend.get_texts():
                        text_obj.set_color("white")
                    if legend.get_title():
                        legend.get_title().set_color("white")
            elif trade_pair_objects:
                ax.text(0.5, 0.5, "No displayable trade actions for legend.",
                        color="white", ha="center", va="center", fontsize=12)

            fig.tight_layout()
            return fig
        except Exception as e:
            buffered_print(
                f"Failed to create legend window for {pair_name}: {type(e).__name__} - {e}", "ERROR")
            if fig and plt.fignum_exists(fig.number):
                plt.close(fig)
            return None

    def _plot_spread_content(self, data: Dict, ax: plt.Axes) -> Optional[plt.Axes]:
        """Plots the content for the spread analysis on a given Axes object."""
        pair_name = data.get('pair_name', 'N/A')
        try:
            clean_dates_dt = data.get("clean_dates")
            if not clean_dates_dt or len(clean_dates_dt) < 2:
                buffered_print(
                    f"Spread content: Insufficient clean_dates for {pair_name}", "DEBUG")
                return None

            clean_dates_numeric = mdates.date2num(clean_dates_dt)

            clean_spreads = data.get("clean_spreads")
            clean_means = data.get("clean_means")
            clean_k = data.get("clean_k")
            clean_std = data.get("clean_std")

            if not (isinstance(clean_spreads, np.ndarray) and len(clean_spreads) == len(clean_dates_numeric) and
                    isinstance(clean_means, np.ndarray) and len(clean_means) == len(clean_dates_numeric) and
                    isinstance(clean_k, np.ndarray) and len(clean_k) == len(clean_dates_numeric) and
                    isinstance(clean_std, np.ndarray) and len(clean_std) == len(clean_dates_numeric)):
                buffered_print(
                    f"Spread content: Data length mismatch for {pair_name}.", "WARNING")
                return None

            plot_colors = data.get("colors", FIXED_COLORS)
            ax.plot(clean_dates_numeric, clean_spreads,
                    color=plot_colors["spread"], label="Spread", linewidth=1.5)
            ax.plot(clean_dates_numeric, clean_means,
                    color=plot_colors["mean"], linestyle="--", label="Dynamic Mean", linewidth=1.2)

            if len(clean_k) > 0 and len(clean_std) > 0:
                lower_band = clean_means - clean_k * clean_std
                upper_band = clean_means + clean_k * clean_std
                k_val_legend = f"k={clean_k[-1]:.2f}" if len(
                    clean_k) > 0 else "k"
                ax.fill_between(clean_dates_numeric, lower_band, upper_band, color=plot_colors["bands"],
                                alpha=0.25, label=f"Dynamic Bands (\u00B1{k_val_legend}\u03C3)")
                if len(clean_dates_numeric) > 0 and len(upper_band) > 0 and len(clean_k) > 0 and len(clean_means) > 0 and len(clean_std) > 0 and -1 < (len(clean_k)-1):  # Ensure indices are valid
                    ax.annotate(f"Z: \u00B1{clean_k[-1]:.2f}\u03C3",
                                xy=(clean_dates_numeric[-1],
                                    upper_band[-1]), xycoords='data',
                                xytext=(10, 5), textcoords='offset points', color="white", fontsize=9, alpha=0.8,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor=plot_colors["adaptive_k_annotation_bg"], edgecolor="none", alpha=0.7))

            rolling_spread_data = data.get("rolling_spread")
            if rolling_spread_data is not None and len(rolling_spread_data) == len(clean_dates_numeric):
                ax.plot(clean_dates_numeric, rolling_spread_data, color=plot_colors.get("rolling_mean_generic", "#FF00FF"),
                        linestyle=':', alpha=0.9, label='Roll Spread Mean', linewidth=1.2)

            if len(clean_spreads) > 0:
                static_mean_val = np.nanmean(clean_spreads)
                if not np.isnan(static_mean_val):
                    ax.axhline(static_mean_val, color=plot_colors.get(
                        "static_mean", "magenta"), linestyle=':', alpha=0.7, label='Static Spread Mean')
                if len(clean_dates_numeric) > 0:
                    ax.annotate(f'{clean_spreads[-1]:.2f}', (clean_dates_numeric[-1], clean_spreads[-1]),
                                textcoords="offset points", xytext=(8, 8), ha='left', color='white', fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="grey", lw=0.5, alpha=0.6))

            add_trade_markers(ax, data.get("trade_pairs_objects", []), data.get("trade_colors", []),
                              clean_dates_numeric, clean_spreads,
                              axis_type="spread", pair_name_for_debug=pair_name)

            format_axis(
                ax, f"Pair {data.get('pair_index', -1) + 1}: Spread Analysis - {pair_name}")
            if not self.headless:
                self._add_hover_tooltips(ax)
            return ax
        except Exception as e:
            buffered_print(
                f"Error plotting spread content for {pair_name}: {type(e).__name__} - {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return None

    def _plot_normalized_price_content(self, data: Dict, ax: plt.Axes) -> Optional[plt.Axes]:
        """Plots the content for the normalized price comparison on a given Axes object."""
        pair_name = data.get('pair_name', 'N/A')
        try:
            normalized_df = data.get("normalized_df")
            if normalized_df is None or normalized_df.empty:
                buffered_print(
                    f"NormPrice content: No normalized_df for {pair_name}", "DEBUG")
                return None
            t1, t2 = pair_name.split('/')
            norm_t1_col, norm_t2_col = f"Normalized {t1}", f"Normalized {t2}"
            if not (norm_t1_col in normalized_df.columns and norm_t2_col in normalized_df.columns):
                buffered_print(
                    f"NormPrice content: Missing cols in normalized_df for {pair_name}", "WARNING")
                return None
            if not isinstance(normalized_df.index, pd.DatetimeIndex):
                try:
                    normalized_df.index = pd.to_datetime(normalized_df.index)
                except:
                    buffered_print(
                        f"NormPrice content: Bad index for {pair_name}", "WARNING")
                    return None

            dates_numeric = mdates.date2num(
                normalized_df.index.to_pydatetime())
            norm_p1_values = normalized_df[norm_t1_col].values
            norm_p2_values = normalized_df[norm_t2_col].values
            plot_colors = data.get("colors", FIXED_COLORS)

            ax.plot(dates_numeric, norm_p1_values,
                    color=plot_colors["price1"], label=f"Norm {t1}", linewidth=1.5)
            ax.plot(dates_numeric, norm_p2_values,
                    color=plot_colors["price2"], label=f"Norm {t2}", linewidth=1.5)

            if len(norm_p1_values) > 0:
                static_mean_p1 = np.nanmean(norm_p1_values)
                if not np.isnan(static_mean_p1):
                    ax.axhline(
                        static_mean_p1, color=plot_colors["price1"], linestyle=':', alpha=0.6, label=f"Static Mean {t1}")
            if len(norm_p2_values) > 0:
                static_mean_p2 = np.nanmean(norm_p2_values)
                if not np.isnan(static_mean_p2):
                    ax.axhline(
                        static_mean_p2, color=plot_colors["price2"], linestyle=':', alpha=0.6, label=f"Static Mean {t2}")

            if len(norm_p1_values) >= WINDOW_SIZE:
                rolling_mean_p1 = pd.Series(norm_p1_values).rolling(
                    WINDOW_SIZE, min_periods=1).mean().values
                ax.plot(dates_numeric, rolling_mean_p1, color=plot_colors.get(
                    "rolling_mean_generic", "yellow"), linestyle='-.', alpha=0.7, label=f"Roll Mean {t1}")
            if len(norm_p2_values) >= WINDOW_SIZE:
                rolling_mean_p2 = pd.Series(norm_p2_values).rolling(
                    WINDOW_SIZE, min_periods=1).mean().values
                ax.plot(dates_numeric, rolling_mean_p2, color=plot_colors.get(
                    "rolling_mean_generic", "lightgreen"), linestyle='-.', alpha=0.7, label=f"Roll Mean {t2}")

            if len(dates_numeric) > 0:
                if len(norm_p1_values) > 0:
                    ax.annotate(f'{norm_p1_values[-1]:.2f}', (dates_numeric[-1], norm_p1_values[-1]), textcoords="offset points", xytext=(
                        8, 8), ha='left', color='white', fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc=plot_colors["price1"], alpha=0.7))
                if len(norm_p2_values) > 0:
                    ax.annotate(f'{norm_p2_values[-1]:.2f}', (dates_numeric[-1], norm_p2_values[-1]), textcoords="offset points", xytext=(
                        8, -15), ha='left', color='white', fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc=plot_colors["price2"], alpha=0.7))

            add_trade_markers(ax, data.get("trade_pairs_objects", []), data.get("trade_colors", [
            ]), dates_numeric, norm_p1_values, axis_type="normalized_price", asset_name=t1, pair_name_for_debug=pair_name)
            add_trade_markers(ax, data.get("trade_pairs_objects", []), data.get("trade_colors", [
            ]), dates_numeric, norm_p2_values, axis_type="normalized_price", asset_name=t2, pair_name_for_debug=pair_name)

            format_axis(
                ax, f"Pair {data.get('pair_index', -1) + 1}: Normalized Prices - {pair_name}")
            if not self.headless:
                self._add_hover_tooltips(ax)
            return ax
        except Exception as e:
            buffered_print(
                f"Error plotting norm price content for {pair_name}: {type(e).__name__} - {e}", "ERROR")
            return None

    def _plot_normalized_spread_content(self, data: Dict, ax: plt.Axes) -> Optional[plt.Axes]:
        pair_name = data.get('pair_name', 'N/A')
        try:
            normalized_df = data.get("normalized_df")
            if normalized_df is None or normalized_df.empty:
                return None
            t1, t2 = pair_name.split('/')
            norm_t1_col, norm_t2_col = f"Normalized {t1}", f"Normalized {t2}"
            if not (norm_t1_col in normalized_df.columns and norm_t2_col in normalized_df.columns):
                return None
            norm_spread_series = (
                normalized_df[norm_t1_col] - normalized_df[norm_t2_col]).dropna()
            if norm_spread_series.empty:
                return None
            if not isinstance(norm_spread_series.index, pd.DatetimeIndex):
                try:
                    norm_spread_series.index = pd.to_datetime(
                        norm_spread_series.index)
                except:
                    return None

            dates_numeric = mdates.date2num(
                norm_spread_series.index.to_pydatetime())
            norm_spread_values = norm_spread_series.values
            plot_colors = data.get("colors", FIXED_COLORS)

            ax.plot(dates_numeric, norm_spread_values,
                    color=plot_colors["normalized_spread"], label="Normalized Spread", linewidth=1.5)

            if len(norm_spread_values) > 0:
                mean_val = np.nanmean(norm_spread_values)
                if not np.isnan(mean_val):
                    ax.axhline(mean_val, color=plot_colors.get(
                        "mean", "cyan"), linestyle="--", label="Mean Norm Spread", linewidth=1.2)
                if len(dates_numeric) > 0:
                    ax.annotate(f'{norm_spread_values[-1]:.2f}', (dates_numeric[-1], norm_spread_values[-1]), textcoords="offset points", xytext=(
                        8, 8), ha='left', color='white', fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="grey", lw=0.5, alpha=0.6))

            rolling_norm_spread_data = data.get("rolling_norm_spread")
            if rolling_norm_spread_data is not None and not (isinstance(rolling_norm_spread_data, pd.Series) and rolling_norm_spread_data.empty):
                rolling_dates_for_plot = dates_numeric
                values_to_plot = None
                if isinstance(rolling_norm_spread_data, pd.Series):
                    if isinstance(rolling_norm_spread_data.index, pd.DatetimeIndex):
                        if rolling_norm_spread_data.index.equals(norm_spread_series.index):
                            values_to_plot = rolling_norm_spread_data.values
                        elif len(rolling_norm_spread_data.values) == len(dates_numeric):
                            values_to_plot = rolling_norm_spread_data.values
                    elif len(rolling_norm_spread_data.values) == len(dates_numeric):
                        values_to_plot = rolling_norm_spread_data.values
                elif isinstance(rolling_norm_spread_data, np.ndarray) and len(rolling_norm_spread_data) == len(dates_numeric):
                    values_to_plot = rolling_norm_spread_data

                if values_to_plot is not None and len(values_to_plot) == len(rolling_dates_for_plot):
                    ax.plot(rolling_dates_for_plot, values_to_plot, color=plot_colors.get(
                        "rolling_mean_generic", "#00FFFF"), linestyle=':', alpha=0.9, label='Rolling Mean', linewidth=1.2)

            add_trade_markers(ax, data.get("trade_pairs_objects", []), data.get("trade_colors", [
            ]), dates_numeric, norm_spread_values, axis_type="normalized_spread", pair_name_for_debug=pair_name)
            format_axis(
                ax, f"Pair {data.get('pair_index', -1) + 1}: Normalized Spread - {pair_name}")
            if not self.headless:
                self._add_hover_tooltips(ax)
            return ax
        except Exception as e:
            buffered_print(
                f"Error plotting norm spread content for {pair_name}: {type(e).__name__} - {e}", "ERROR")
            return None

    def _plot_price_comparison_content(self, data: Dict, axes: List[plt.Axes]) -> Optional[List[plt.Axes]]:
        """Plots the content for the price comparison on a list of two Axes objects."""
        if len(axes) != 2:
            buffered_print(
                "Price comparison content: Expected 2 axes.", "ERROR")
            return None
        ax1, ax2 = axes[0], axes[1]
        pair_name = data.get('pair_name', 'N/A')
        try:
            actual_dates_dt = data.get("actual_dates")
            if not actual_dates_dt or len(actual_dates_dt) < 2:
                return None
            actual_dates_numeric = mdates.date2num(actual_dates_dt)

            actual_p1 = data.get("actual_p1")
            actual_p2 = data.get("actual_p2")
            if not (isinstance(actual_p1, np.ndarray) and len(actual_p1) == len(actual_dates_numeric) and isinstance(actual_p2, np.ndarray) and len(actual_p2) == len(actual_dates_numeric)):
                return None
            t1, t2 = pair_name.split('/')
            plot_colors = data.get("colors", FIXED_COLORS)

            ax1.plot(actual_dates_numeric, actual_p1,
                     color=plot_colors["price1"], label=f"{t1} Price", linewidth=1.5)
            if len(actual_p1) > 0:
                static_mean_p1 = np.nanmean(actual_p1)
                if not np.isnan(static_mean_p1):
                    ax1.axhline(
                        static_mean_p1, color=plot_colors["price1"], linestyle=':', alpha=0.6, label=f"Static Mean {t1}")
            rolling_p1 = data.get("rolling_p1")
            if rolling_p1 is not None and len(rolling_p1) == len(actual_dates_numeric):
                ax1.plot(actual_dates_numeric, rolling_p1, color=plot_colors.get(
                    "rolling_mean_generic", "cyan"), linestyle="--", alpha=0.7, label=f"{t1} Roll Mean", linewidth=1.2)
            if len(actual_p1) > 0 and len(actual_dates_numeric) > 0:
                ax1.annotate(f'{actual_p1[-1]:.2f}', (actual_dates_numeric[-1], actual_p1[-1]), textcoords="offset points", xytext=(
                    8, 8), ha='left', color='white', fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc=plot_colors["price1"], alpha=0.7))
            add_trade_markers(ax1, data.get("trade_pairs_objects", []), data.get("trade_colors", [
            ]), actual_dates_numeric, actual_p1, axis_type="price_comparison", asset_name=t1, pair_name_for_debug=pair_name)
            format_axis(ax1, f"#{data.get('pair_index', -1) + 1}: {t1} Price")

            ax2.plot(actual_dates_numeric, actual_p2,
                     color=plot_colors["price2"], label=f"{t2} Price", linewidth=1.5)
            if len(actual_p2) > 0:
                static_mean_p2 = np.nanmean(actual_p2)
                if not np.isnan(static_mean_p2):
                    ax2.axhline(
                        static_mean_p2, color=plot_colors["price2"], linestyle=':', alpha=0.6, label=f"Static Mean {t2}")
            rolling_p2 = data.get("rolling_p2")
            if rolling_p2 is not None and len(rolling_p2) == len(actual_dates_numeric):
                ax2.plot(actual_dates_numeric, rolling_p2, color=plot_colors.get(
                    "rolling_mean_generic", "magenta"), linestyle="--", alpha=0.7, label=f"{t2} Roll Mean", linewidth=1.2)
            if len(actual_p2) > 0 and len(actual_dates_numeric) > 0:
                ax2.annotate(f'{actual_p2[-1]:.2f}', (actual_dates_numeric[-1], actual_p2[-1]), textcoords="offset points", xytext=(
                    8, 8), ha='left', color='white', fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc=plot_colors["price2"], alpha=0.7))

            add_trade_markers(ax2, data.get("trade_pairs_objects", []), data.get("trade_colors", [
            ]), actual_dates_numeric, actual_p2, axis_type="price_comparison", asset_name=t2, pair_name_for_debug=pair_name)
            format_axis(ax2, f"#{data.get('pair_index', -1) + 1}: {t2} Price")

            if not self.headless:
                self._add_hover_tooltips(ax1)
                self._add_hover_tooltips(ax2)
            return axes
        except Exception as e:
            buffered_print(
                f"Error plotting price comp content for {pair_name}: {type(e).__name__} - {e}", "ERROR")
            return None

    def _plot_price_ratio_content(self, data: Dict, ax: plt.Axes) -> Optional[plt.Axes]:
        """Plots the content for the price ratio on a given Axes object."""
        pair_name = data.get('pair_name', 'N/A')
        try:
            normalized_df = data.get("normalized_df")
            if normalized_df is None or normalized_df.empty:
                return None
            t1, t2 = pair_name.split('/')
            norm_t1_col, norm_t2_col = f"Normalized {t1}", f"Normalized {t2}"
            if not (norm_t1_col in normalized_df.columns and norm_t2_col in normalized_df.columns):
                return None
            safe_t2_norm = normalized_df[norm_t2_col].replace(0, np.nan)
            price_ratio_series = (
                normalized_df[norm_t1_col] / safe_t2_norm).dropna()
            if price_ratio_series.empty:
                return None
            if not isinstance(price_ratio_series.index, pd.DatetimeIndex):
                try:
                    price_ratio_series.index = pd.to_datetime(
                        price_ratio_series.index)
                except:
                    return None

            dates_numeric = mdates.date2num(
                price_ratio_series.index.to_pydatetime())
            price_ratio_values = price_ratio_series.values
            plot_colors = data.get("colors", FIXED_COLORS)

            ax.plot(dates_numeric, price_ratio_values,
                    color=plot_colors["price_ratio"], label="Price Ratio (S1/S2)", linewidth=1.5)
            ax.axhline(1.0, color="white", linestyle="--",
                       alpha=0.5, label="Ratio = 1.0", linewidth=1)

            if len(price_ratio_values) > 0:
                mean_val = np.nanmean(price_ratio_values)
                if not np.isnan(mean_val):
                    ax.axhline(mean_val, color=plot_colors.get(
                        "mean", "cyan"), linestyle=":", alpha=0.7, label="Mean Ratio", linewidth=1.2)
                if len(dates_numeric) > 0:
                    ax.annotate(f'{price_ratio_values[-1]:.3f}', (dates_numeric[-1], price_ratio_values[-1]), textcoords="offset points", xytext=(
                        8, 8), ha='left', color='white', fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="grey", lw=0.5, alpha=0.6))

            rolling_ratio_data = data.get("rolling_ratio")
            if rolling_ratio_data is not None:
                if isinstance(rolling_ratio_data, pd.Series) and not rolling_ratio_data.empty:
                    if isinstance(rolling_ratio_data.index, pd.DatetimeIndex) and rolling_ratio_data.index.equals(price_ratio_series.index):
                        ax.plot(dates_numeric, rolling_ratio_data.values, color=plot_colors.get(
                            "rolling_mean_generic", "#FF00FF"), linestyle=':', alpha=0.9, label='Rolling Ratio', linewidth=1.2)
                    elif len(rolling_ratio_data.values) == len(dates_numeric):
                        ax.plot(dates_numeric, rolling_ratio_data.values, color=plot_colors.get(
                            "rolling_mean_generic", "#FF00FF"), linestyle=':', alpha=0.9, label='Rolling Ratio (idx aligned)', linewidth=1.2)
                elif isinstance(rolling_ratio_data, np.ndarray) and len(rolling_ratio_data) == len(dates_numeric):
                    ax.plot(dates_numeric, rolling_ratio_data, color=plot_colors.get(
                        "rolling_mean_generic", "#FF00FF"), linestyle=':', alpha=0.9, label='Rolling Ratio', linewidth=1.2)

            add_trade_markers(ax, data.get("trade_pairs_objects", []), data.get("trade_colors", []),
                              dates_numeric, price_ratio_values,
                              axis_type="price_ratio", pair_name_for_debug=pair_name)

            format_axis(
                ax, f"Pair {data.get('pair_index', -1) + 1}: Price Ratio - {pair_name}")
            if not self.headless:
                self._add_hover_tooltips(ax)
            return ax
        except Exception as e:
            buffered_print(
                f"Error plotting price ratio content for {pair_name}: {type(e).__name__} - {e}", "ERROR")
            return None

    def _plot_legend_content(self, data: Dict, ax: plt.Axes) -> Optional[plt.Axes]:
        """Updates the content of an existing legend axes."""
        pair_name = data.get('pair_name', 'N/A')
        try:
            # ax.clear() # Clearing is done by the caller if needed
            ax.axis('off')

            trade_pair_objects: List[Any] = data.get("trade_pairs_objects", [])
            trade_colors: List[str] = data.get("trade_colors", [])
            legend_handles: List[plt.Artist] = []
            legend_labels: List[str] = []

            if not trade_pair_objects:
                ax.text(0.5, 0.5, "No trades for this pair.",
                        color="white", ha="center", va="center", fontsize=12)
            else:
                for i, trade_obj in enumerate(trade_pair_objects):
                    if not all(hasattr(trade_obj, attr) for attr in ['entry_dt', 'exit_dt', 'direction', 'color_idx']):
                        continue
                    color = trade_colors[i % len(
                        trade_colors)] if trade_colors else FIXED_COLORS['price1']
                    entry_marker_char = '^' if hasattr(
                        trade_obj, 'direction') and trade_obj.direction and trade_obj.direction.lower() == "long" else 'v'
                    entry_legend_text = f"Trade {i+1} Entry" + (f" ({trade_obj.direction.capitalize()})" if hasattr(
                        trade_obj, 'direction') and trade_obj.direction else "")

                    entry_handle = Line2D([0], [0], marker=entry_marker_char, color=color,
                                          markeredgecolor='white', markersize=8, linestyle='None')
                    if entry_legend_text not in legend_labels:
                        legend_handles.append(entry_handle)
                        legend_labels.append(entry_legend_text)

                    if hasattr(trade_obj, 'exit_dt') and trade_obj.exit_dt:
                        exit_legend_text = f"Trade {i+1} Exit"
                        exit_handle = Line2D(
                            [0], [0], marker='x', color=color, markeredgecolor='white', markersize=8, linestyle='None')
                        if exit_legend_text not in legend_labels:
                            legend_handles.append(exit_handle)
                            legend_labels.append(exit_legend_text)

            if legend_handles:
                # Remove any existing legend before creating a new one
                if ax.get_legend():
                    ax.get_legend().remove()  # Important for content update

                legend = ax.legend(legend_handles, legend_labels, loc='center', ncol=min(2, (len(legend_handles) + 1) // 2),
                                   frameon=True, facecolor="#2a2a2a", edgecolor="grey", fontsize=9,
                                   title=f"Trades for {pair_name}", title_fontsize=11)
                if legend:
                    for text_obj in legend.get_texts():
                        text_obj.set_color("white")
                    if legend.get_title():
                        legend.get_title().set_color("white")
            elif trade_pair_objects:
                ax.text(0.5, 0.5, "No displayable trade actions for legend.",
                        color="white", ha="center", va="center", fontsize=12)
            return ax
        except Exception as e:
            buffered_print(
                f"Error plotting legend content for {pair_name}: {type(e).__name__} - {e}", "ERROR")
            try:
                if ax:
                    ax.text(0.5, 0.5, "Legend Error", color="red",
                            ha="center", va="center", fontsize=10)
            except:
                pass
            return None

    def _create_spread_plot(self, data: Dict) -> Optional[plt.Figure]:
        pair_name = data.get('pair_name', 'N/A')
        fig = plt.figure(figsize=(18, 8), facecolor="#1a1a1a",
                         num=f"Spread Analysis - {pair_name}")
        ax = fig.add_subplot(111)
        if self._plot_spread_content(data, ax):
            fig.tight_layout()
            return fig
        plt.close(fig)
        return None

    def _create_normalized_price_plot(self, data: Dict) -> Optional[plt.Figure]:
        pair_name = data.get('pair_name', 'N/A')
        fig = plt.figure(figsize=(18, 8), facecolor="#1a1a1a",
                         num=f"Normalized Prices - {pair_name}")
        ax = fig.add_subplot(111)
        if self._plot_normalized_price_content(data, ax):
            fig.tight_layout()
            return fig
        plt.close(fig)
        return None

    def _create_normalized_spread_plot(self, data: Dict) -> Optional[plt.Figure]:
        pair_name = data.get('pair_name', 'N/A')
        fig = plt.figure(figsize=(18, 8), facecolor="#1a1a1a",
                         num=f"Normalized Spread - {pair_name}")
        ax = fig.add_subplot(111)
        if self._plot_normalized_spread_content(data, ax):
            fig.tight_layout()
            return fig
        plt.close(fig)
        return None

    def _create_price_comparison_plot(self, data: Dict) -> Optional[plt.Figure]:
        pair_name = data.get('pair_name', 'N/A')
        fig = plt.figure(figsize=(18, 10), facecolor="#1a1a1a",
                         num=f"Price Comparison - {pair_name}")
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        if self._plot_price_comparison_content(data, [ax1, ax2]):
            fig.suptitle(f"Pair {data.get('pair_index', -1) + 1}: Price Comparison - {pair_name}",
                         y=0.98, color="white", fontsize=16, fontweight='bold')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            return fig
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)
            return None

    def _create_price_ratio_plot(self, data: Dict) -> Optional[plt.Figure]:
        pair_name = data.get('pair_name', 'N/A')
        fig = plt.figure(figsize=(18, 8), facecolor="#1a1a1a",
                         num=f"Price Ratio - {pair_name}")
        ax = fig.add_subplot(111)
        if self._plot_price_ratio_content(data, ax):
            fig.tight_layout()
            return fig
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)
            return None

    def _add_hover_tooltips(self, ax: plt.Axes):
        """Adds hover tooltips to plottable artists in an Axes object, avoiding duplicates and improving style."""
        if self.headless:
            return

        # Filter existing cursors for this specific axes
        # (Keep your existing logic for managing self.cursors if it's working to avoid duplicates)
        for cur in list(self.cursors):  # Iterate over a copy if removing items
            if hasattr(cur, 'axes_list') and ax in cur.axes_list:
                # If cursor already exists for this ax, maybe just return or ensure it's active
                # For simplicity, let's assume new cursors are fine if old ones are cleaned up elsewhere (e.g., _cleanup_figures)
                # However, for this specific recursive issue, the main problem is the call below.
                pass  # No specific action if cursor exists, might be handled by cleanup

        plottable_artists = [
            child for child in ax.get_children()
            if isinstance(child, (Line2D, PathCollection)) and
            getattr(child, 'get_visible', lambda: True)() and
            not (isinstance(child, Line2D) and (child.get_linestyle()
                 in [':', '--', '-.']) and not child.get_label())
        ]

        if not plottable_artists:
            return

        if mplcursors is None:
            return

        try:
            cursor = mplcursors.cursor(plottable_artists, hover=True)
            self.cursors.append(cursor)

            @cursor.connect("add")
            def on_hover(sel):
                try:
                    artist_label = sel.artist.get_label()
                    if not artist_label or artist_label.startswith('_'):
                        artist_label = sel.artist.__class__.__name__
                        if artist_label == "PathCollection":
                            artist_label = "Data Point"

                    target_point = sel.target
                    text_content = f"{artist_label}"
                    if isinstance(target_point, (np.ndarray, list, tuple)) and len(target_point) >= 2:
                        x, y = target_point[0], target_point[1]
                        try:
                            if isinstance(x, (float, np.floating)) and x > 700000:
                                date_str = mdates.num2date(
                                    x).strftime("%Y-%m-%d")
                                text_content = f"{artist_label}\nDate: {date_str}\nValue: {y:.3f}"
                            else:
                                text_content = f"{artist_label}\nX: {x:.2f}\nY: {y:.3f}"
                        except (ValueError, TypeError):
                            text_content = f"{artist_label}\nX: {x:.2f}\nY: {y:.3f}"

                    sel.annotation.set_text(text_content)
                    sel.annotation.set_color("white")
                    sel.annotation.get_bbox_patch().set(facecolor="darkslategray", alpha=0.9,
                                                        edgecolor="silver", lw=0.5, boxstyle="round,pad=0.4")
                    if hasattr(sel.annotation, 'arrow_patch') and sel.annotation.arrow_patch:
                        sel.annotation.arrow_patch.set(
                            arrowstyle="-", facecolor="silver", alpha=0.7, ec="darkslategray")
                except Exception as e_hover:
                    sel.annotation.set_text("Info N/A")
                    buffered_print(
                        f"Tooltip internal error on '{artist_label}': {e_hover}", "DEBUG")
        except Exception as e_cursor:
            buffered_print(
                f"Failed to create mplcursor for axes '{ax.get_title()}': {type(e_cursor).__name__} - {e_cursor}", "WARNING")

    def _apply_saved_geometry(self, fig: plt.Figure, fig_type: str):
        """Restore geometry and ensure visibility. Manages deiconify and lift."""
        try:
            # Prioritize closed, then current, then last pair's geometry
            geom = self.closed_geometries.get(fig_type) or \
                self.window_geometries.get(fig_type) or \
                self.last_pair_geometries.get(fig_type)

            manager = fig.canvas.manager
            if hasattr(manager, "window") and manager.window is not None:
                if geom:
                    try:
                        manager.window.geometry(geom)
                    except Exception as e_geom:
                        buffered_print(
                            f"Error applying geometry to '{fig_type}': {e_geom}", "DEBUG")

                # Standard ways to show/raise a window
                if hasattr(manager.window, 'deiconify'):
                    manager.window.deiconify()  # Tk
                elif hasattr(manager.window, 'show'):
                    manager.window.show()       # Qt/Others

                if hasattr(manager.window, 'lift'):
                    manager.window.lift()         # Tk
                elif hasattr(manager.window, 'raise_'):
                    manager.window.raise_()   # Qt/Others

                if hasattr(manager.window, 'activateWindow'):
                    manager.window.activateWindow()  # Qt
                if hasattr(manager.window, 'focus_force'):
                    manager.window.focus_force()       # Tk

                fig.canvas.draw_idle()  # Ensure plot is updated visually
            else:
                buffered_print(
                    f"Cannot apply geometry: No 'window' attribute for manager of '{fig_type}'", "DEBUG")
        except Exception as e:  # Catch-all for any other backend-specific issues
            buffered_print(
                f"Generic window geometry/visibility error for '{fig_type}': {str(e)}", "DEBUG")

    def _refresh_display(self):
        """Closes old figures and creates/updates figures for the current pair."""
        if self.headless:
            self._render_headless()  # Headless mode generates and saves plots
            return

        if not self.pairs_data:  # No data to display
            buffered_print(
                "PairNavigator: No pairs_data available to refresh display.", "INFO")
            # Clean up any existing plot windows
            self._cleanup_figures(preserve_control_panel=True)
            return

        if not (0 <= self.current_index < len(self.pairs_data)):  # Index out of bounds
            buffered_print(
                f"PairNavigator: current_index ({self.current_index}) out of bounds for pairs_data (len {len(self.pairs_data)}). Resetting to 0.", "WARNING")
            self.current_index = 0
            # Still no data (e.g. if pairs_data was empty list)
            if not self.pairs_data:
                self._cleanup_figures(preserve_control_panel=True)
                return

        # This will close existing plot figures and recreate them based on _figure_visibility
        self._create_figures()

        # Small pause for UI to update, especially if many windows are created/updated
        plt.pause(0.05)

    def _refresh_display_content_only(self):
        """
        Updates the content of currently visible figures for the new pair or on visibility change.
        Creates figures if they are marked visible but don't exist.
        Does NOT close and reopen all figure windows unnecessarily.
        """
        if not self.pairs_data or not (0 <= self.current_index < len(self.pairs_data)):
            buffered_print(
                "Refresh content: No data or invalid index. Cleaning up.", "WARNING")
            self._cleanup_figures(preserve_control_panel=True)
            return

        current_pair_plot_config = self.pairs_data[self.current_index]
        pair_name = current_pair_plot_config.get("pair_name", "N/A")
        buffered_print(
            f"Refreshing content for pair: {pair_name} (Index: {self.current_index})", "DEBUG")

        for fig_type in self.figure_types:
            if self._figure_visibility.get(fig_type, False):
                # This figure type is intended to be visible.
                # _create_or_update_single_figure will either show existing or create new.
                self._create_or_update_single_figure(fig_type)
            else:
                # This figure type is intended to be hidden. Ensure it is.
                existing_fig = next((f for f in self.current_figures if hasattr(
                    f, "_figure_type") and f._figure_type == fig_type), None)
                if existing_fig and plt.fignum_exists(existing_fig.number):
                    manager = existing_fig.canvas.manager
                    if hasattr(manager, "window") and manager.window is not None:
                        # Check if it's currently not withdrawn (i.e., visible or iconified)
                        is_actually_visible = True
                        if hasattr(manager.window, 'state') and manager.window.state() == 'withdrawn':
                            is_actually_visible = False
                        elif hasattr(manager.window, 'isVisible') and not manager.window.isVisible():
                            is_actually_visible = False

                        if is_actually_visible:
                            self._capture_figure_geometry(existing_fig)
                            if hasattr(manager.window, 'withdraw'):
                                manager.window.withdraw()

        self._update_titles()
        self._lift_control_panel()

    # def toggle_figure(self, fig_type: str, event: Optional[Event] = None):
    #     """Toggles the visibility of a specific figure type using targeted updates."""
    #     # Toggle the desired visibility state
    #     new_visibility = not self._figure_visibility.get(fig_type, False)
    #     self._figure_visibility[fig_type] = new_visibility

    #     # Always update the figure regardless of new visibility state
    #     self._create_or_update_single_figure(fig_type)

    #     # Explicitly handle window show/hide
    #     existing_fig = next((f for f in self.current_figures if hasattr(
    #         f, "_figure_type") and f._figure_type == fig_type), None)
    #     if existing_fig and plt.fignum_exists(existing_fig.number):
    #         manager = existing_fig.canvas.manager
    #         if hasattr(manager, "window") and manager.window is not None:
    #             if new_visibility:
    #                 if hasattr(manager.window, 'deiconify'):
    #                     manager.window.deiconify()
    #                 self._apply_saved_geometry(existing_fig, fig_type)
    #             else:
    #                 if hasattr(manager.window, 'withdraw'):
    #                     manager.window.withdraw()
    def toggle_figure(self, fig_type: str, event: Optional[Event] = None):
        """Toggles the visibility of a specific figure type using targeted updates."""
        # Toggle the desired visibility state
        self._figure_visibility[fig_type] = not self._figure_visibility.get(
            fig_type, False)

        # Now update the figure (will handle show/hide)
        self._create_or_update_single_figure(fig_type)

    def _cleanup_figures(self, preserve_control_panel: bool = False):
        """Closes all currently managed plot figures. Optionally preserves control panel."""
        # Disconnect and clear mplcursors instances to prevent memory leaks or errors
        for cursor in self.cursors:
            try:
                cursor.disconnect_events()
            except Exception:
                pass  # Ignore errors during disconnect
        self.cursors = []

        # Iterate over a copy of the list as we are modifying it
        for fig in list(self.current_figures):
            if plt.fignum_exists(fig.number):
                try:
                    plt.close(fig)
                except Exception as e_close:
                    buffered_print(
                        f"Error closing figure number {fig.number}: {e_close}", "DEBUG")
        self.current_figures = []  # Clear the list of managed figures

        # Handle control panel separately
        if not preserve_control_panel and hasattr(self, "control_fig") and self.control_fig:
            if plt.fignum_exists(self.control_fig.number):
                try:
                    plt.close(self.control_fig)
                except Exception as e_ctrl_close:
                    buffered_print(
                        f"Error closing control panel: {e_ctrl_close}", "DEBUG")
            self.control_fig = None

    def toggle_figure(self, fig_type: str, event: Optional[Event] = None):
        """Toggles the visibility of a specific figure type using targeted updates."""
        # Toggle the desired visibility state
        self._figure_visibility[fig_type] = not self._figure_visibility.get(
            fig_type, False)

        # Now, either create/show or hide this specific figure
        self._create_or_update_single_figure(fig_type)
        # No full _refresh_display() needed here to avoid affecting other plots' states

        # No _capture_current_geometries needed here if we are only updating content
        # self.last_pair_geometries can be updated if _refresh_display (full) is ever called by nav.

    def next_pair(self, event: Optional[Event] = None):
        if self.pairs_data:
            self._capture_current_geometries()  # Still good to capture before changing index
            self.last_pair_geometries = self.window_geometries.copy()
            self.current_index = (self.current_index +
                                  1) % len(self.pairs_data)
            self._refresh_display_content_only()  # Ensure this is called
        self._lift_control_panel()

    def random_pair(self, event: Optional[Event] = None):
        if self.pairs_data:
            self._capture_current_geometries()
            self.last_pair_geometries = self.window_geometries.copy()
            self.current_index = random.randint(0, len(self.pairs_data) - 1)
            self._refresh_display_content_only()
        self._lift_control_panel()

    def restart(self, event: Optional[Event] = None):
        self._capture_current_geometries()
        self.last_pair_geometries = self.window_geometries.copy()
        self.current_index = 0
        if self.pairs_data:
            self._refresh_display_content_only()
        self._lift_control_panel()

    def _init_gui(self):
        """Initializes the control panel GUI elements."""
        buffered_print("PairNavigator: _init_gui starting...", "DEBUG")
        plt.ioff()

        try:
            # Adjusted figsize height from 9.5 to 11 to accommodate more buttons
            self.control_fig = plt.figure(
                figsize=(4, 11), facecolor="#1a1a1a", num="Pair Navigator Controls"
            )
            if not self.control_fig:
                buffered_print(
                    "PairNavigator: plt.figure() failed to create control_fig!", "CRITICAL")
                raise RuntimeError("Failed to create control_fig")

            self.control_fig.clf = lambda: None
            self.control_fig.clear = lambda: None
            buffered_print("PairNavigator: Control figure created.", "DEBUG")
        except Exception as e_fig:
            buffered_print(
                f"PairNavigator: Error creating control_fig: {e_fig}", "CRITICAL")
            raise

        try:
            # Adjust subplot to give a bit more room at the bottom if needed, top is already 0.95
            plt.subplots_adjust(left=0.1, right=0.9,
                                bottom=0.02, top=0.95, hspace=0.25)

            # Search Box Label (slightly adjust y if needed due to figure height change, but 0.94 should be fine)
            # y from 0.94 to 0.95, height 0.03 to 0.025
            ax_label = self.control_fig.add_axes([0.1, 0.95, 0.8, 0.025])
            ax_label.axis('off')
            ax_label.text(0.5, 0.5, 'Search Pair (T1/T2):', ha='center',
                          va='center', fontsize=9, color='white')  # smaller font

            # Search Box (adjust y)
            # y from 0.90 to 0.91, height 0.04 to 0.035
            ax_search = self.control_fig.add_axes([0.1, 0.91, 0.8, 0.035])
            self.search_box = TextBox(
                ax_search, '', initial='', color='#2a2a2a', hovercolor='#3a3a3a')
            if not self.search_box:
                buffered_print(
                    "PairNavigator: TextBox() failed to create search_box!", "CRITICAL")
                raise RuntimeError("Failed to create search_box")
            self.search_box.label.set_color('white')
            self.search_box.text_disp.set_color('white')
            self.search_box.text_disp.set_fontsize(
                9)  # Smaller font for text box display

            if hasattr(self.search_box, '_cursor'):
                self.search_box._cursor.set_color('white')
            elif hasattr(self.search_box, 'cursor_patch'):
                self.search_box.cursor_patch.set_facecolor('white')

            self.search_box.on_submit(self._on_search_submit)
            buffered_print("PairNavigator: Search box created.", "DEBUG")
        except Exception as e_search:
            buffered_print(
                f"PairNavigator: Error creating search box: {e_search}", "CRITICAL")
            raise

        try:
            self.control_fig.canvas.mpl_connect(
                "button_press_event", self._on_click_outside)
            # This will now use the new figure height for layout
            button_config = self._get_button_config()
            self._button_refs = []
            for config_idx, config in enumerate(button_config):
                try:
                    ax_btn = self.control_fig.add_axes(config["position"])
                    btn = Button(ax_btn, config["label"], color=config["color"],
                                 hovercolor=config.get("style", {}).get("hovercolor", config["color"]))
                    btn.on_clicked(self._safe_callback(config["callback"]))
                    style = config.get("style", {})
                    btn.label.set_color(style.get("text_color", "#ffffff"))
                    # Will be overridden by _get_button_config
                    btn.label.set_fontsize(style.get("fontsize", 10))
                    btn.label.set_fontweight(style.get("fontweight", "bold"))
                    self._button_refs.append(btn)
                except Exception as e_btn_item:
                    buffered_print(
                        f"Button '{config.get('label', 'UNKNOWN')}' creation failed: {str(e_btn_item)}", "ERROR")
            buffered_print("PairNavigator: Buttons created.", "DEBUG")
        except Exception as e_buttons:
            buffered_print(
                f"PairNavigator: Error creating UI buttons: {e_buttons}", "CRITICAL")
            raise

        try:
            mgr = self.control_fig.canvas.manager
            if hasattr(mgr, "window") and mgr.window is not None:
                if hasattr(mgr.window, "attributes"):
                    mgr.window.attributes("-topmost", 1)
                    # Keep slightly transparent
                    mgr.window.attributes("-alpha", 0.95)
                if hasattr(mgr.window, "resizable"):
                    mgr.window.resizable(False, False)
                try:
                    mgr.window.title("Nav Controls")  # Shorter title
                except:
                    try:
                        mgr.window.setWindowTitle("Nav Controls")
                    except:
                        pass

                if hasattr(mgr.window, 'winfo_screenwidth') and hasattr(mgr.window, 'geometry'):
                    screen_width = mgr.window.winfo_screenwidth()
                    # Adjusted height from 700 to 850 or 900
                    try:
                        mgr.window.geometry(f"300x900+{screen_width-350}+50")
                    except:
                        pass
            else:
                buffered_print(
                    "Control panel manager has no 'window' attribute. Skipping window setup.", "DEBUG")
            buffered_print(
                "PairNavigator: Window manager configured.", "DEBUG")
        except Exception as e_mgr:
            buffered_print(
                f"Control panel window configuration error: {str(e_mgr)}. Backend specific.", "WARNING")

        try:
            self.control_fig.canvas.mpl_connect(
                "key_press_event", self._on_key_press)
            self.control_fig.canvas.mpl_connect(
                "resize_event", self._on_figure_configure)  # Though resizing is false
            self.control_fig.canvas.mpl_connect(
                'close_event', self.exit_program)
            buffered_print(
                "PairNavigator: Control figure event handlers connected.", "DEBUG")
        except Exception as e_events:
            buffered_print(
                f"PairNavigator: Error connecting control_fig event handlers: {e_events}", "CRITICAL")
            raise

        buffered_print(
            "PairNavigator: _init_gui finished successfully.", "DEBUG")

    def _get_button_config(self) -> List[Dict]:
        blue_style = {"text_color": "white", "fontsize": 7,
                      "fontweight": "bold", "hovercolor": "#2E86C1"}
        gray_style = {"text_color": "white", "fontsize": 7,
                      "fontweight": "bold", "hovercolor": "#5D6D7E"}
        red_style = {"text_color": "white", "fontsize": 7,
                     "fontweight": "bold", "hovercolor": "#C0392B"}
        green_style = {"text_color": "white", "fontsize": 7,
                       "fontweight": "bold", "hovercolor": "#27AE60"}
        purple_style = {"text_color": "white", "fontsize": 7,
                        "fontweight": "bold", "hovercolor": "#8E44AD"}

        panel_width = 0.9
        panel_left = (1 - panel_width) / 2
        num_orig_plot_buttons = 6
        num_sim_plot_buttons = 3
        num_analysis_plot_buttons = 3
        num_nav_buttons = 3
        num_action_buttons = 2
        total_button_slots = num_orig_plot_buttons + num_sim_plot_buttons + \
            num_analysis_plot_buttons + num_nav_buttons + num_action_buttons
        button_area_top = 0.89
        button_area_bottom = 0.01
        button_area_height = button_area_top - button_area_bottom
        effective_button_area_height = button_area_height * 0.99
        slot_height_incl_spacing = effective_button_area_height / total_button_slots
        button_height = slot_height_incl_spacing * 0.90
        v_spacing = slot_height_incl_spacing - button_height
        config: List[Dict] = []
        current_y = button_area_top - button_height

        plot_button_types_orig = ["spread", "normalized_price",
                                  "normalized_spread", "price_comparison", "price_ratio", "trade_legend"]
        plot_button_labels_orig = [
            "Spread (1)", "Norm Prices (2)", "Norm Spread (3)", "Price Comp (4)", "Price Ratio (5)", "Trade Legend (L)"]
        for i, fig_type in enumerate(plot_button_types_orig):
            config.append({"label": plot_button_labels_orig[i], "color": "#1F77B4", "callback": partial(
                self.toggle_figure, fig_type), "position": (panel_left, current_y, panel_width, button_height), "style": blue_style})
            current_y -= (button_height + v_spacing)
        plot_button_types_sim = ["simulated_equity",
                                 "trade_return_histogram", "rolling_correlation"]
        plot_button_labels_sim = [
            "Sim Equity (6)", "Returns Hist (7)", "Roll Corr (8)"]
        for i, fig_type in enumerate(plot_button_types_sim):
            config.append({"label": plot_button_labels_sim[i], "color": "#2ECC71", "callback": partial(
                self.toggle_figure, fig_type), "position": (panel_left, current_y, panel_width, button_height), "style": green_style})
            current_y -= (button_height + v_spacing)
        plot_button_types_analysis = [
            "cointegration_tests", "half_life_estimation", "total_strategy_equity"]
        plot_button_labels_analysis = [
            "Coint Tests (9)", "Half-Life (0)", "Total Equity (-)"]
        for i, fig_type in enumerate(plot_button_types_analysis):
            config.append({"label": plot_button_labels_analysis[i], "color": "#9B59B6", "callback": partial(
                self.toggle_figure, fig_type), "position": (panel_left, current_y, panel_width, button_height), "style": purple_style})
            current_y -= (button_height + v_spacing)
        current_y -= v_spacing * 0.1
        nav_buttons_config = [("NEXT (SPACE)", self.next_pair),
                              ("RESTART (R)", self.restart), ("RANDOM (X)", self.random_pair)]
        for label, callback_func in nav_buttons_config:
            config.append({"label": label, "color": "#404040", "callback": callback_func, "position": (
                panel_left, current_y, panel_width, button_height), "style": gray_style})
            current_y -= (button_height + v_spacing)
        current_y -= v_spacing * 0.1
        action_buttons_config = [
            ("CLOSE ALL (C)", self.close_all, "#A93226"), ("EXIT (Q)", self.exit_program, "#E74C3C")]
        for label, callback_func, btn_color in action_buttons_config:
            config.append({"label": label, "color": btn_color, "callback": callback_func, "position": (
                panel_left, current_y, panel_width, button_height), "style": red_style})
            current_y -= (button_height + v_spacing)
        return config

    def _update_titles(self):
        """Update titles of all currently displayed figures with pair information."""
        # This method should now correctly handle window titles vs suptitle for all figures.
        # No change needed here specifically for the title issue if _figure_type is consistently set.
        if not self.pairs_data or not (0 <= self.current_index < len(self.pairs_data)):
            # For total strategy equity, title doesn't depend on current_index
            for fig_obj in self.current_figures:
                if hasattr(fig_obj, "_figure_type") and fig_obj._figure_type == "total_strategy_equity":
                    if plt.fignum_exists(fig_obj.number) and hasattr(fig_obj.canvas.manager, 'set_window_title'):
                        try:
                            fig_obj.canvas.manager.set_window_title(
                                "Total Strategy Equity")  # Static title
                        except Exception as e_title:
                            buffered_print(
                                f"Error setting title for total_strategy_equity: {e_title}", "DEBUG")
            return

        current_pair_info = self.pairs_data[self.current_index]
        pair_name = current_pair_info.get("pair_name", "N/A")
        pair_idx_display = current_pair_info.get("pair_index", -1) + 1

        for fig_obj in self.current_figures:
            if plt.fignum_exists(fig_obj.number) and hasattr(fig_obj, "_figure_type"):
                fig_type = cast(str, fig_obj._figure_type)

                # Default base title to fig_type
                fig_type_title_base = fig_type.replace("_", " ").title()

                # Get existing window title if available (manager.num or specific name)
                window_title_base = fig_type_title_base  # Fallback
                if hasattr(fig_obj.canvas.manager, 'get_window_title'):
                    current_win_title = fig_obj.canvas.manager.get_window_title()
                    # Attempt to extract base part if it follows "Base - Pair #X: NAME"
                    if " - Pair #" in current_win_title:
                        window_title_base = current_win_title.split(
                            " - Pair #")[0]
                    # If it's not just the figure number
                    elif current_win_title and current_win_title != str(fig_obj.number):
                        window_title_base = current_win_title  # Use it as is if not default
                elif hasattr(fig_obj.canvas.manager, 'num') and fig_obj.canvas.manager.num:
                    window_title_base = fig_obj.canvas.manager.num  # Often "Figure X"

                final_window_title = ""
                if fig_type == "total_strategy_equity":
                    final_window_title = "Total Strategy Equity"  # Fixed title
                elif fig_type == "price_comparison":
                    # Price comparison uses fig.suptitle, so window title can be simpler
                    final_window_title = f"Price Comparison - Pair #{pair_idx_display}: {pair_name}"
                else:
                    # For other plots, combine base title with pair info
                    final_window_title = f"{window_title_base} - Pair #{pair_idx_display}: {pair_name}"

                if hasattr(fig_obj.canvas.manager, 'set_window_title'):
                    try:
                        fig_obj.canvas.manager.set_window_title(
                            final_window_title)
                    except Exception as e_title:
                        buffered_print(
                            f"Error setting window title for {fig_type_title_base}: {e_title}", "DEBUG")

                # Update internal suptitle for plots like price_comparison
                if fig_type == "price_comparison" and hasattr(fig_obj, 'suptitle'):
                    try:
                        # Clear previous suptitle if it exists by drawing a new one
                        fig_obj.suptitle(f"Pair {pair_idx_display}: Price Comparison - {pair_name}",
                                         y=0.98, color="white", fontsize=16, fontweight='bold')
                        fig_obj.canvas.draw_idle()  # Redraw to reflect suptitle change
                    except Exception as e_suptitle:
                        buffered_print(
                            f"Error setting suptitle for price_comparison: {e_suptitle}", "DEBUG")

    def _create_simulated_equity_plot(self, data: Dict) -> Optional[plt.Figure]:
        pair_name = data.get('pair_name', 'N/A')
        fig = plt.figure(figsize=(18, 9), facecolor="#1a1a1a",
                         num=f"Simulated Equity - {pair_name}")
        ax = fig.add_subplot(111)
        if _plot_simulated_equity_curve_content(data, ax):
            fig.tight_layout(pad=1.5)
            return fig
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

    def _create_trade_return_histogram_plot(self, data: Dict) -> Optional[plt.Figure]:
        pair_name = data.get('pair_name', 'N/A')
        fig = plt.figure(figsize=(12, 7), facecolor="#1a1a1a",
                         num=f"Trade Returns Hist - {pair_name}")
        ax = fig.add_subplot(111)
        if _plot_trade_return_histogram_content(data, ax):
            fig.tight_layout(pad=1.5)
            return fig
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

    def _create_rolling_correlation_plot(self, data: Dict) -> Optional[plt.Figure]:
        pair_name = data.get('pair_name', 'N/A')
        fig = plt.figure(figsize=(18, 8), facecolor="#1a1a1a",
                         num=f"Rolling Correlation - {pair_name}")
        ax = fig.add_subplot(111)
        if _plot_rolling_correlation_content(data, ax):
            fig.tight_layout(pad=1.5)
            return fig
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

    # New
    def _create_cointegration_tests_plot(self, data: Dict) -> Optional[plt.Figure]:
        pair_name = data.get('pair_name', 'N/A')
        fig = plt.figure(figsize=(18, 9), facecolor="#1a1a1a",
                         num=f"Cointegration Tests - {pair_name}")
        ax = fig.add_subplot(111)
        if _plot_cointegration_tests_content(data, ax):
            fig.tight_layout(pad=1.5)
            return fig
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

    # New
    def _create_half_life_estimation_plot(self, data: Dict) -> Optional[plt.Figure]:
        pair_name = data.get('pair_name', 'N/A')
        fig = plt.figure(figsize=(18, 8), facecolor="#1a1a1a",
                         num=f"Half-Life Estimation - {pair_name}")
        ax = fig.add_subplot(111)
        if _plot_half_life_content(data, ax):
            fig.tight_layout(pad=1.5)
            return fig
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

    # New
    def _create_total_strategy_equity_plot(self, data: Optional[Dict] = None) -> Optional[plt.Figure]:
        if not self.total_strategy_equity_data:
            return None
        fig = plt.figure(figsize=(18, 9), facecolor="#1a1a1a",
                         num="Total Strategy Equity")
        ax = fig.add_subplot(111)
        if _plot_total_strategy_equity_content(self.total_strategy_equity_data, ax):
            fig.tight_layout(pad=1.5)
            return fig
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

    def _create_or_update_single_figure(self, fig_type_to_manage: str):
        current_pair_plot_config = None
        if fig_type_to_manage != "total_strategy_equity":
            if not self.pairs_data or not (0 <= self.current_index < len(self.pairs_data)):
                buffered_print(
                    f"Cannot update {fig_type_to_manage}: No valid pair data at current index.", "WARNING")
                return
            current_pair_plot_config = self.pairs_data[self.current_index]

        figure_creator_map: Dict[str, Callable[[Optional[Dict]], Optional[plt.Figure]]] = {
            "spread": self._create_spread_plot, "normalized_price": self._create_normalized_price_plot,
            "normalized_spread": self._create_normalized_spread_plot, "price_comparison": self._create_price_comparison_plot,
            "price_ratio": self._create_price_ratio_plot, "trade_legend": self._create_legend_window,
            "simulated_equity": self._create_simulated_equity_plot, "trade_return_histogram": self._create_trade_return_histogram_plot,
            "rolling_correlation": self._create_rolling_correlation_plot,
            "cointegration_tests": self._create_cointegration_tests_plot,
            "half_life_estimation": self._create_half_life_estimation_plot,
            "total_strategy_equity": self._create_total_strategy_equity_plot,
        }
        figure_content_plotter_map: Dict[str, Callable[[Dict, Any], Optional[Any]]] = {
            "spread": self._plot_spread_content, "normalized_price": self._plot_normalized_price_content,
            "normalized_spread": self._plot_normalized_spread_content, "price_comparison": self._plot_price_comparison_content,
            "price_ratio": self._plot_price_ratio_content, "trade_legend": self._plot_legend_content,
            "simulated_equity": _plot_simulated_equity_curve_content, "trade_return_histogram": _plot_trade_return_histogram_content,
            "rolling_correlation": _plot_rolling_correlation_content,
            "cointegration_tests": _plot_cointegration_tests_content,
            "half_life_estimation": _plot_half_life_content,
            "total_strategy_equity": _plot_total_strategy_equity_content,
        }

        existing_fig = next((f for f in self.current_figures if hasattr(
            f, "_figure_type") and f._figure_type == fig_type_to_manage), None)

        if self._figure_visibility.get(fig_type_to_manage, False):
            content_plotter = figure_content_plotter_map.get(
                fig_type_to_manage)
            if not content_plotter:
                buffered_print(
                    f"No content plotter for '{fig_type_to_manage}'", "WARNING")
                return

            data_for_plotter = current_pair_plot_config if fig_type_to_manage != "total_strategy_equity" else self.total_strategy_equity_data
            if not data_for_plotter:
                buffered_print(
                    f"No data available for plotter of '{fig_type_to_manage}'", "WARNING")
                return

            if existing_fig and plt.fignum_exists(existing_fig.number):
                if fig_type_to_manage == "total_strategy_equity" and hasattr(existing_fig, "_content_drawn_once") and existing_fig._content_drawn_once:
                    self._show_figure_window(existing_fig, fig_type_to_manage)
                    return

                for ax_obj in existing_fig.get_axes():
                    for cur_idx, c_obj in reversed(list(enumerate(self.cursors))):
                        if hasattr(c_obj, 'axes_list') and ax_obj in c_obj.axes_list:
                            try:
                                c_obj.disconnect_events()
                            except:
                                pass
                            self.cursors.pop(cur_idx)
                    ax_obj.clear()

                plot_successful = False
                axes_to_plot_on = existing_fig.get_axes()
                if axes_to_plot_on:  # Ensure axes exist
                    if fig_type_to_manage == "price_comparison" and len(axes_to_plot_on) == 2:
                        if content_plotter(data_for_plotter, axes_to_plot_on):
                            plot_successful = True
                    else:
                        if content_plotter(data_for_plotter, axes_to_plot_on[0]):
                            plot_successful = True

                if plot_successful:
                    if fig_type_to_manage == "total_strategy_equity":
                        existing_fig._content_drawn_once = True
                    # Apply geometry after content update
                    self._apply_saved_geometry(
                        existing_fig, fig_type_to_manage)
                    existing_fig.canvas.draw_idle()
                else:
                    self._figure_visibility[fig_type_to_manage] = False
                    if plt.fignum_exists(existing_fig.number):
                        plt.close(existing_fig)
                    if existing_fig in self.current_figures:
                        self.current_figures.remove(existing_fig)
            else:
                if existing_fig and existing_fig in self.current_figures:
                    self.current_figures.remove(existing_fig)
                creator_func = figure_creator_map.get(fig_type_to_manage)
                if creator_func:
                    created_fig = creator_func(
                        data_for_plotter if fig_type_to_manage != "total_strategy_equity" else None)
                    if created_fig:
                        created_fig._figure_type = fig_type_to_manage
                        if fig_type_to_manage == "total_strategy_equity":
                            created_fig._content_drawn_once = True
                        self._apply_saved_geometry(
                            created_fig, fig_type_to_manage)
                        created_fig.canvas.mpl_connect('close_event', partial(
                            self._on_figure_close, fig_type=fig_type_to_manage))
                        self.current_figures.append(created_fig)
                        created_fig.canvas.draw_idle()
                    else:
                        self._figure_visibility[fig_type_to_manage] = False

            fig_to_show = next((f for f in self.current_figures if hasattr(
                f, "_figure_type") and f._figure_type == fig_type_to_manage), None)
            if fig_to_show and plt.fignum_exists(fig_to_show.number):
                self._show_figure_window(fig_to_show, fig_type_to_manage)
        else:
            if existing_fig and plt.fignum_exists(existing_fig.number):
                manager = existing_fig.canvas.manager
                if hasattr(manager, "window") and manager.window is not None:
                    is_actually_visible = True
                    if hasattr(manager.window, 'state') and manager.window.state() == 'withdrawn':
                        is_actually_visible = False
                    elif hasattr(manager.window, 'isVisible') and not manager.window.isVisible():
                        is_actually_visible = False
                    if is_actually_visible:
                        self._capture_figure_geometry(existing_fig)
                        if hasattr(manager.window, 'withdraw'):
                            manager.window.withdraw()

    def render(self):
        """Manages the Matplotlib event loop and figure display."""
        buffered_print(
            f"PairNavigator: render() called. self.headless={self.headless}, self.should_exit={self.should_exit}", "DEBUG")
        if self.headless:
            buffered_print(
                "PairNavigator: Headless mode, render() call is a no-op here (headless processing done in __init__).", "INFO")
            return

        # Ensure figures are ready if they weren't created in __init__ (e.g., if data was loaded later)
        # However, current __init__ calls _refresh_display_content_only then render.
        if not self.current_figures and self.pairs_data:
            buffered_print(
                "PairNavigator.render(): No current figures but data exists. Refreshing display.", "DEBUG")
            self._refresh_display_content_only()  # Use content refresh

        if self.control_fig and plt.fignum_exists(self.control_fig.number):
            self._lift_control_panel()
            buffered_print("PairNavigator: Control panel lifted.", "DEBUG")

        if not plt.get_fignums() and not self.should_exit:
            buffered_print(
                "PairNavigator.render(): No Matplotlib figures to show.", "INFO")
            if not (self.control_fig and plt.fignum_exists(self.control_fig.number)):
                buffered_print(
                    "PairNavigator.render(): Control panel also not available. Render may exit.", "INFO")
                return
            else:  # Only control panel might be open
                buffered_print(
                    "PairNavigator: Only control panel seems to be active.", "DEBUG")

        if not self.should_exit:
            buffered_print(
                "PairNavigator: Calling plt.show(block=True)...", "INFO")
            try:
                plt.show(block=True)
                # Should only print after all GUI windows closed
                buffered_print("PairNavigator: plt.show() exited.", "INFO")
            except KeyboardInterrupt:
                buffered_print(
                    "KeyboardInterrupt caught in PairNavigator.render(). Exiting.", "INFO")
                self.exit_program()
            except Exception as e:
                buffered_print(
                    f"Error during plt.show() in PairNavigator: {type(e).__name__} - {e}", "CRITICAL")
                import traceback
                traceback.print_exc()
                self.exit_program()
        else:
            buffered_print(
                "PairNavigator: self.should_exit is True, skipping plt.show().", "INFO")

    def close_all(self, event: Optional[Event] = None):
        """
        Closes all plot windows managed by the navigator, preserving their last known geometries.
        The control panel remains open.
        """
        buffered_print("PairNavigator: Close All action triggered.", "DEBUG")
        # Capture geometries of currently visible windows before marking them for hiding
        # This ensures that if they are reopened, they try to use their last positions.
        for fig_obj in self.current_figures:
            if hasattr(fig_obj, "_figure_type") and plt.fignum_exists(fig_obj.number):
                fig_type = cast(str, fig_obj._figure_type)
                # Check if the figure is actually visible before capturing geometry
                manager = fig_obj.canvas.manager
                is_visible = True
                if hasattr(manager, "window") and manager.window is not None:
                    if hasattr(manager.window, 'state') and manager.window.state() == 'withdrawn':
                        is_visible = False
                    elif hasattr(manager.window, 'isVisible') and not manager.window.isVisible():
                        is_visible = False

                if is_visible:
                    # Updates self.window_geometries
                    self._capture_figure_geometry(fig_obj)
                    self.closed_geometries[fig_type] = self.window_geometries.get(
                        fig_type, "")  # Store as a "closed" geometry

        # Mark all figure types as not visible (user intends for them to be closed)
        for fig_type_to_hide in self.figure_types:
            # Don't override trade_legend's default of False unless it was explicitly made True
            if fig_type_to_hide != "trade_legend" or self._figure_visibility.get("trade_legend", False):
                self._figure_visibility[fig_type_to_hide] = False

        # Close actual plot windows
        self._cleanup_figures(preserve_control_panel=True)
        self._lift_control_panel()  # Ensure control panel is still accessible

    def exit_program(self, event: Optional[Event] = None):
        if self.should_exit:
            return
        self.should_exit = True
        buffered_print("PairNavigator: Exiting program...", "INFO")
        self._cleanup_figures(preserve_control_panel=False)
        # Note: sys.exit(0) can be problematic for some backends/environments.
        # Rely on plt.close('all') and the event loop naturally ending.

    def _on_search_submit(self, text: str):
        """Handles pair search submission, updating content of visible windows."""
        search_text = text.strip().upper()
        if self.search_box:
            self.search_box.set_val("")
        if not search_text:
            return
        parts = [p.strip() for p in search_text.split('/') if p.strip()]
        if len(parts) != 2:
            buffered_print(
                f"Invalid search: '{search_text}'. Use T1/T2", "WARNING")
            self._lift_control_panel()
            return

        target_pair_name = f"{parts[0]}/{parts[1]}"
        found_idx = next((idx for idx, pi_dict in enumerate(
            self.pairs_data) if pi_dict.get("pair_name", "").upper() == target_pair_name), -1)

        if found_idx != -1:
            # When search finds a pair, we might still want to capture geometries of the *old* pair's state
            # before switching to the new one if _refresh_display_content_only doesn't close windows.
            # self._capture_current_geometries() # Optional, depends on desired strictness
            # self.last_pair_geometries = self.window_geometries.copy() # Optional
            self.current_index = found_idx
            self._refresh_display_content_only()
        else:
            buffered_print(f"Pair '{target_pair_name}' not found.", "INFO")
        self._lift_control_panel()

    def _on_click_outside(self, event: MouseEvent):
        """Manages focus for the search box."""
        if not hasattr(self, 'search_box') or not self.search_box or not self.search_box.ax:
            return

        if event.inaxes != self.search_box.ax:
            if self.search_focused:
                self.search_focused = False
        else:
            self.search_focused = True

    def _on_key_press(self, event: KeyEvent):
        """Handles global key presses for navigation when control panel is focused."""
        if self.search_focused and self.search_box and event.inaxes == self.search_box.ax:
            # Added new keys
            if event.key in [" ", "r", "x", "c", "q", "l", "1", "2", "3", "4", "5", "6", "7", "8"]:
                return  # Don't trigger global shortcuts if typing these in search

        key_map: Dict[str, Callable] = {
            " ": self.next_pair, "space": self.next_pair,
            "r": self.restart, "x": self.random_pair,
            "c": self.close_all, "q": self.exit_program,
            "l": partial(self.toggle_figure, "trade_legend"),
            "1": partial(self.toggle_figure, "spread"),
            "2": partial(self.toggle_figure, "normalized_price"),
            "3": partial(self.toggle_figure, "normalized_spread"),
            "4": partial(self.toggle_figure, "price_comparison"),
            "5": partial(self.toggle_figure, "price_ratio"),
            "6": partial(self.toggle_figure, "simulated_equity"),
            "7": partial(self.toggle_figure, "trade_return_histogram"),
            "8": partial(self.toggle_figure, "rolling_correlation"),
            "9": partial(self.toggle_figure, "cointegration_tests"),
            "0": partial(self.toggle_figure, "half_life_estimation"),
            "-": partial(self.toggle_figure, "total_strategy_equity"),
            "_": partial(self.toggle_figure, "total_strategy_equity"),

        }
        action = key_map.get(event.key.lower())
        if action:
            action()
            if self.control_fig and plt.fignum_exists(self.control_fig.number):
                self.control_fig.canvas.draw_idle()
            self._lift_control_panel()

    def _safe_callback(self, func: Callable) -> Callable:
        """Wraps button callbacks to handle focus and errors."""
        def wrapped_callback(event: Event):
            try:
                if self.search_box and hasattr(event, 'inaxes') and event.inaxes != self.search_box.ax:
                    self.search_focused = False
                func(event)
            except Exception as e:
                buffered_print(
                    f"Error in GUI callback '{func.__name__}': {type(e).__name__} - {e}", "ERROR")
            finally:
                self._lift_control_panel()
        return wrapped_callback

    def _lift_control_panel(self):
        """Brings the control panel window to the top and attempts to focus it."""
        if self.control_fig and plt.fignum_exists(self.control_fig.number):
            try:
                manager = self.control_fig.canvas.manager
                if hasattr(manager, "window") and manager.window is not None:
                    if hasattr(manager.window, 'deiconify'):
                        manager.window.deiconify()
                    if hasattr(manager.window, 'lift'):
                        manager.window.lift()
                    elif hasattr(manager.window, 'raise_'):
                        manager.window.raise_()

                    if hasattr(manager.window, 'attributes'):
                        try:
                            manager.window.attributes('-topmost', 1)
                            # Ensure winfo_exists before calling attributes again in a lambda
                            if manager.window.winfo_exists():
                                manager.window.after(100, lambda: manager.window.attributes(
                                    '-topmost', 0) if manager.window.winfo_exists() else None)
                        except Exception:
                            pass
                    self.control_fig.canvas.draw_idle()
            except Exception as e:
                buffered_print(f"Error lifting control panel: {e}", "DEBUG")

    def _capture_figure_geometry(self, fig: plt.Figure):
        """Captures and stores the geometry of a given figure."""
        if not hasattr(fig, "_figure_type"):
            return

        fig_type = cast(str, fig._figure_type)
        try:
            if hasattr(fig.canvas, 'manager') and hasattr(fig.canvas.manager, 'window') and \
               hasattr(fig.canvas.manager.window, 'geometry'):
                geom = fig.canvas.manager.window.geometry()
                if geom:
                    self.window_geometries[fig_type] = geom
            else:
                buffered_print(
                    f"Cannot capture geometry: No window/geometry for manager of '{fig_type}'", "DEBUG")
        except Exception as e:
            buffered_print(
                f"Error capturing geometry for '{fig_type}': {e}", "DEBUG")

    def _capture_current_geometries(self):
        """Captures geometries of all currently visible and managed plot figures."""
        current_session_geoms: Dict[str, str] = {}
        for fig_obj in self.current_figures:
            if plt.fignum_exists(fig_obj.number) and hasattr(fig_obj, "_figure_type"):
                fig_type = cast(str, fig_obj._figure_type)
                try:
                    manager = fig_obj.canvas.manager
                    if hasattr(manager, 'window') and manager.window is not None:
                        is_visible = True
                        if hasattr(manager.window, 'state'):
                            if manager.window.state() == 'withdrawn':
                                is_visible = False
                        elif hasattr(manager.window, 'isVisible'):
                            if not manager.window.isVisible():
                                is_visible = False

                        if is_visible:
                            self._capture_figure_geometry(fig_obj)
                            if fig_type in self.window_geometries:
                                current_session_geoms[fig_type] = self.window_geometries[fig_type]
                except Exception as e:
                    buffered_print(
                        f"Error during capture_current_geometries for {fig_type}: {e}", "DEBUG")

        # Update last_pair_geometries with the geometries captured in this session
        # This ensures last_pair_geometries truly reflects the state before potential navigation
        self.last_pair_geometries = current_session_geoms.copy()

    def _on_figure_configure(self, event: ResizeEvent):
        """Debounced handler for figure resize/move events to save geometry."""
        current_time = time.time()
        if current_time - self.last_resize_time < self.resize_debounce_interval:
            return
        self.last_resize_time = current_time

        fig = event.canvas.figure
        if hasattr(fig, "_figure_type"):
            self._capture_figure_geometry(fig)
        elif fig == self.control_fig:
            pass

    def _show_figure_window(self, fig: plt.Figure, fig_type: str):
        """Ensures a figure window is visible and properly positioned."""
        if not plt.fignum_exists(fig.number):
            return

        manager = fig.canvas.manager
        if hasattr(manager, "window") and manager.window is not None:
            # First ensure it's not minimized/withdrawn
            if hasattr(manager.window, 'deiconify'):
                manager.window.deiconify()
            # Then apply any saved geometry
            self._apply_saved_geometry(fig, fig_type)
            # Finally bring to front
            if hasattr(manager.window, 'lift'):
                manager.window.lift()
            elif hasattr(manager.window, 'raise_'):
                manager.window.raise_()

    def _render_headless(self):
        """Generates and saves plots to './headless/' directory without displaying them."""
        buffered_print(
            "PairNavigator: Running in headless mode. Generating plots to disk.", "INFO")
        if not self.pairs_data:
            buffered_print("Headless: No data to plot.", "INFO")
            return

        headless_dir = "headless"
        if not os.path.exists(headless_dir):
            try:
                os.makedirs(headless_dir)
                buffered_print(f"Created directory: ./{headless_dir}", "INFO")
            except OSError as e:
                buffered_print(
                    f"Error creating directory ./{headless_dir}: {e}", "ERROR")
                return

        # Store original visibility to restore after headless plotting
        original_vis_state = self._figure_visibility.copy()

        for idx in range(len(self.pairs_data)):
            self.current_index = idx
            current_pair_plot_config = self.pairs_data[self.current_index]
            pair_name = current_pair_plot_config.get(
                "pair_name", f"Pair_{idx}")
            safe_pair_name = sanitize_filename(pair_name)
            buffered_print(f"Headless: Processing pair {pair_name}", "DEBUG")

            figure_creator_map = {
                "spread": self._create_spread_plot,
                "normalized_price": self._create_normalized_price_plot,
                "normalized_spread": self._create_normalized_spread_plot,
                "price_comparison": self._create_price_comparison_plot,
                "price_ratio": self._create_price_ratio_plot,
            }

            for fig_type, creator_func in figure_creator_map.items():
                fig: Optional[plt.Figure] = None
                try:
                    # Temporarily set to visible for creation
                    self._figure_visibility[fig_type] = True
                    fig = creator_func(current_pair_plot_config)
                    if fig:
                        filename = os.path.join(
                            headless_dir, f"plot_{safe_pair_name}_{fig_type}.png")
                        fig.savefig(filename, bbox_inches='tight', dpi=150)
                        buffered_print(f"Headless: Saved {filename}", "INFO")
                    else:
                        buffered_print(
                            f"Headless: Could not create '{fig_type}' for {pair_name}", "WARNING")
                except Exception as e:
                    buffered_print(
                        f"Headless: Error creating/saving '{fig_type}' for {pair_name}: {e}", "ERROR")
                finally:
                    if fig and plt.fignum_exists(fig.number):
                        plt.close(fig)
                    self._figure_visibility[fig_type] = original_vis_state.get(
                        fig_type, False)  # Restore original

        self._figure_visibility = original_vis_state  # Restore all original visibilities
        buffered_print("Headless: Plot generation complete.", "INFO")
