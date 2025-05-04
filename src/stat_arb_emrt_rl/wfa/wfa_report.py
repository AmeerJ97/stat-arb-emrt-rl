from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from ..printing_system import (
    BOLD, CYAN, ENDC, GREEN, RED, YELLOW,
    buffered_print, print_header, print_section,
)

from .wfa_engine import WFAReport


class WFAReporter:
    """Generate reports, CSVs, and plots from a WFA run."""

    def __init__(self, report: WFAReport) -> None:
        self.report = report
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        os.makedirs(self.report.config.output_dir, exist_ok=True)

    def to_csv(self) -> None:
        """Export window summary and stitched equity to CSV."""
        out = self.report.config.output_dir

        # Window summary
        rows = []
        for w in self.report.windows:
            rows.append({
                "window_id": w.window_id,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "pairs_discovered": len(w.pairs_discovered),
                "pairs_traded": len(w.pairs_traded),
                "is_sharpe": w.is_sharpe,
                "is_cagr": w.is_cagr,
                "is_max_dd": w.is_max_dd,
                "is_final_value": w.is_final_value,
                "oos_sharpe": w.oos_sharpe,
                "oos_cagr": w.oos_cagr,
                "oos_max_dd": w.oos_max_dd,
                "oos_final_value": w.oos_final_value,
                "oos_total_trades": w.oos_total_trades,
                "oos_win_rate": w.oos_win_rate,
                "oos_profit_factor": w.oos_profit_factor,
                "sharpe_decay": w.sharpe_decay if w.sharpe_decay != float("inf") else None,
            })

        df = pd.DataFrame(rows)
        summary_path = os.path.join(out, "wfa_window_summary.csv")
        df.to_csv(summary_path, index=False)
        buffered_print(f"{GREEN}Saved: {summary_path}{ENDC}")

        # Stitched equity
        if self.report.stitched_equity is not None and not self.report.stitched_equity.empty:
            eq_path = os.path.join(out, "wfa_stitched_equity.csv")
            eq_df = pd.DataFrame({
                "date": self.report.stitched_equity.index,
                "equity": self.report.stitched_equity.values,
            })
            eq_df.to_csv(eq_path, index=False)
            buffered_print(f"{GREEN}Saved: {eq_path}{ENDC}")

    def plot_stitched_equity(self) -> None:
        """Plot the stitched OOS equity curve with window boundaries."""
        eq = self.report.stitched_equity
        if eq is None or eq.empty:
            buffered_print(f"{YELLOW}No equity data to plot.{ENDC}", "WARNING")
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        ax.plot(eq.index, eq.values, color="#00d4ff", linewidth=1.5, label="OOS Equity")

        # Window boundaries
        for w in self.report.windows:
            ts = pd.Timestamp(w.test_start)
            ax.axvline(ts, color="#ff6b6b", linestyle="--", alpha=0.4, linewidth=0.8)

        ax.set_title("Walk-Forward Analysis: Stitched OOS Equity Curve",
                      color="white", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Portfolio Value ($)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#16213e", edgecolor="white", labelcolor="white")
        ax.grid(True, alpha=0.2, color="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        path = os.path.join(self.report.config.output_dir, "wfa_stitched_equity.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buffered_print(f"{GREEN}Saved: {path}{ENDC}")

    def plot_is_vs_oos(self, metric: str = "sharpe") -> None:
        """Bar chart comparing IS vs OOS metric per window."""
        windows = self.report.windows
        if not windows:
            return

        labels = [f"W{w.window_id}" for w in windows]
        is_vals = [getattr(w, f"is_{metric}", 0.0) for w in windows]
        oos_vals = [getattr(w, f"oos_{metric}", 0.0) for w in windows]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        ax.bar(x - width / 2, is_vals, width, label="In-Sample", color="#ff6b6b", alpha=0.8)
        ax.bar(x + width / 2, oos_vals, width, label="Out-of-Sample", color="#00d4ff", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color="white")
        ax.set_title(f"IS vs OOS {metric.upper()} by Window (Overfitting Detection)",
                      color="white", fontsize=13, fontweight="bold")
        ax.set_ylabel(metric.upper(), color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#16213e", edgecolor="white", labelcolor="white")
        ax.grid(True, axis="y", alpha=0.2, color="white")
        ax.axhline(0, color="white", linewidth=0.5, alpha=0.5)

        path = os.path.join(self.report.config.output_dir, f"wfa_is_vs_oos_{metric}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buffered_print(f"{GREEN}Saved: {path}{ENDC}")

    def plot_pair_turnover(self) -> None:
        """Heatmap of pair presence across windows."""
        windows = self.report.windows
        if not windows:
            return

        all_pairs = set()
        for w in windows:
            all_pairs.update(w.pairs_discovered)

        if not all_pairs:
            return

        pairs_sorted = sorted(all_pairs)
        # Limit to top 40 pairs by frequency to keep heatmap readable
        pair_freq = {}
        for p in pairs_sorted:
            pair_freq[p] = sum(1 for w in windows if p in w.pairs_discovered)
        top_pairs = sorted(pair_freq, key=pair_freq.get, reverse=True)[:40]

        matrix = np.zeros((len(top_pairs), len(windows)))
        for j, w in enumerate(windows):
            disc = set(w.pairs_discovered)
            for i, p in enumerate(top_pairs):
                matrix[i, j] = 1.0 if p in disc else 0.0

        fig, ax = plt.subplots(figsize=(max(8, len(windows) * 0.8), max(6, len(top_pairs) * 0.3)))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        ax.imshow(matrix, aspect="auto", cmap="YlGn", interpolation="nearest")
        ax.set_xticks(range(len(windows)))
        ax.set_xticklabels([f"W{w.window_id}" for w in windows], color="white", fontsize=8)
        ax.set_yticks(range(len(top_pairs)))
        ax.set_yticklabels([f"{p[0]}/{p[1]}" for p in top_pairs], color="white", fontsize=7)
        ax.set_title("Pair Presence Across Windows (Turnover Analysis)",
                      color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("Window", color="white")
        ax.tick_params(colors="white")

        path = os.path.join(self.report.config.output_dir, "wfa_pair_turnover.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buffered_print(f"{GREEN}Saved: {path}{ENDC}")

    def plot_rolling_oos_sharpe(self) -> None:
        """Plot OOS Sharpe ratio per window over time."""
        windows = [w for w in self.report.windows if w.oos_total_trades > 0]
        if not windows:
            return

        dates = [pd.Timestamp(w.test_start) for w in windows]
        sharpes = [w.oos_sharpe for w in windows]

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        ax.plot(dates, sharpes, "o-", color="#00d4ff", markersize=6, linewidth=1.5)
        ax.axhline(0, color="#ff6b6b", linestyle="--", alpha=0.5)

        ax.set_title("OOS Sharpe Ratio Over Time",
                      color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("Window Start", color="white")
        ax.set_ylabel("Sharpe Ratio", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, color="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        path = os.path.join(self.report.config.output_dir, "wfa_rolling_oos_sharpe.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buffered_print(f"{GREEN}Saved: {path}{ENDC}")

    def print_summary(self) -> None:
        """Print formatted terminal summary."""
        r = self.report
        print_header("Walk-Forward Analysis Results")

        mode = "Anchored (Expanding)" if r.config.anchored else "Sliding"
        buffered_print(
            f"{BOLD}Mode:{ENDC} {mode} | "
            f"Train: {r.config.train_months}mo | "
            f"Test: {r.config.test_months}mo | "
            f"Step: {r.config.step_months}mo"
        )
        buffered_print(
            f"{BOLD}Windows:{ENDC} {len(r.windows)} | "
            f"Date range: {r.config.overall_start} → {r.config.overall_end}\n"
        )

        # Per-window summary
        print_section("Window Results", CYAN)
        for w in r.windows:
            decay_str = f"{w.sharpe_decay:.2f}" if w.sharpe_decay != float("inf") else "INF"
            buffered_print(
                f"  W{w.window_id}: "
                f"[{w.test_start}→{w.test_end}] "
                f"IS_SR={w.is_sharpe:+.2f} "
                f"OOS_SR={w.oos_sharpe:+.2f} "
                f"Decay={decay_str} "
                f"Trades={w.oos_total_trades} "
                f"WR={w.oos_win_rate:.0%} "
                f"Final=${w.oos_final_value:,.0f} "
                f"Pairs={len(w.pairs_discovered)}"
            )

        # Aggregate
        print_section("Aggregate OOS Performance", GREEN)
        buffered_print(f"  {BOLD}Sharpe:{ENDC}     {r.agg_sharpe:.2f}")
        buffered_print(f"  {BOLD}CAGR:{ENDC}       {r.agg_cagr:.2%}")
        buffered_print(f"  {BOLD}Max DD:{ENDC}     {r.agg_max_dd:.2%}")
        buffered_print(f"  {BOLD}Total Trades:{ENDC} {r.agg_total_trades}")
        buffered_print(f"  {BOLD}Win Rate:{ENDC}   {r.agg_win_rate:.1%}")

        # Overfitting
        print_section("Overfitting Analysis", YELLOW)
        buffered_print(f"  {BOLD}Avg Sharpe Decay (IS/OOS):{ENDC} {r.avg_sharpe_decay:.2f}")
        buffered_print(f"  {BOLD}Pair Turnover Rate:{ENDC} {r.pair_turnover_rate:.1%}")

        if r.avg_sharpe_decay > 2.0:
            buffered_print(f"  {RED}WARNING: High Sharpe decay suggests overfitting.{ENDC}")
        elif r.avg_sharpe_decay > 1.5:
            buffered_print(f"  {YELLOW}CAUTION: Moderate Sharpe decay — monitor closely.{ENDC}")
        else:
            buffered_print(f"  {GREEN}Sharpe decay within acceptable range.{ENDC}")

        if r.pair_turnover_rate > 0.5:
            buffered_print(f"  {YELLOW}High pair turnover — cointegration relationships are unstable.{ENDC}")

    def generate_full_report(self) -> None:
        """Generate all CSVs, plots, and terminal summary."""
        self.print_summary()
        self.to_csv()
        self.plot_stitched_equity()
        self.plot_is_vs_oos("sharpe")
        self.plot_is_vs_oos("cagr")
        self.plot_rolling_oos_sharpe()
        self.plot_pair_turnover()
        buffered_print(
            f"\n{BOLD}{GREEN}Full WFA report saved to: "
            f"{self.report.config.output_dir}{ENDC}"
        )
