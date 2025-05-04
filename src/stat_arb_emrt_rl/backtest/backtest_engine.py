# ───────────────────────────────────────────────
# Set matplotlib backend before other imports
# ───────────────────────────────────────────────
import sys
from ..printing_system import (
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
    print_header,
    buffered_print,
    print_section,
    # flush_queue,
    MAX_LINE_WIDTH,
)
from .plotting import plot_pair_spreads, format_axis
from .gui import PairNavigator
from .analyzers import EnhancedReturnsAnalyzer, SpreadTracker
from ..reversion_strategy import MeanReversionStrategy
from ..data_provider import MarketDataProvider
import numpy as np
from tqdm import tqdm
import pandas as pd
import backtrader as bt
from time import sleep
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib
import logging

# Only use TkAgg if running in interactive mode (not discovery/headless/wfa)
if not ("--discover" in sys.argv or "--rl-backtest" in sys.argv or "--wfa" in sys.argv):
    try:
        matplotlib.use("TkAgg")  # Ensure interactive backend is set early
    except Exception:
        # Fall back to Agg if TkAgg is not available
        matplotlib.use("Agg")
else:
    # Use Agg backend for headless/discovery/wfa modes
    matplotlib.use("Agg")

logger = logging.getLogger(__name__)


@dataclass
class BacktestReport:
    final_value: float
    cagr: float
    sharpe: float
    max_drawdown: float
    equity_curve: Optional[pd.Series]
    daily_returns: Optional[pd.Series]
    trades: List[Dict]
    total_trades: int
    win_rate: float
    profit_factor: float
    pairs_traded: List[Tuple[str, str]]

    @classmethod
    def from_dict(cls, data: Dict) -> "BacktestReport":
        return cls(
            final_value=float(data.get("final_value", 0.0)),
            cagr=float(data.get("cagr", 0.0)),
            sharpe=float(data.get("sharpe", 0.0)),
            max_drawdown=float(data.get("max_drawdown", 0.0)),
            equity_curve=data.get("equity_curve"),
            daily_returns=data.get("daily_returns"),
            trades=list(data.get("trades", [])),
            total_trades=int(data.get("total_trades", 0)),
            win_rate=float(data.get("win_rate", 0.0)),
            profit_factor=float(data.get("profit_factor", 0.0)),
            pairs_traded=list(data.get("pairs_traded", [])),
        )


# ───────────────────────────────────────────────
# Shared helpers for backtest setup and result extraction
# ───────────────────────────────────────────────


def _setup_cerebro(
    loader: MarketDataProvider,
    pairs,
    start_date: str,
    end_date: str,
    initial_cash: int = 10000,
    use_rl: bool = False,
    reset_cache: bool = False,
    verbose: bool = True,
) -> Tuple[Optional[bt.Cerebro], List[Tuple[str, str]]]:
    """Create and configure a Cerebro instance with data feeds.

    Returns (cerebro, valid_pairs). Returns (None, []) if no valid pairs.
    """
    cerebro = bt.Cerebro(
        stdstats=False,
        maxcpus=None,
        exactbars=False,
        runonce=False,
    )
    valid_pairs = []
    all_names = set()

    pair_iter = (
        tqdm(
            pairs,
            desc=f"{YELLOW}Initializing Backtested Data: {start_date} to {end_date}",
            colour="yellow",
        )
        if verbose
        else pairs
    )

    for pair in pair_iter:
        t1, t2 = pair

        df1 = loader.get_stock_data(
            t1, start_date, end_date, reset_cache=reset_cache)
        df2 = loader.get_stock_data(
            t2, start_date, end_date, reset_cache=reset_cache)

        # Date filtering after alignment
        if df1 is not None:
            df1 = df1[df1.index <= pd.to_datetime(end_date)]
        if df2 is not None:
            df2 = df2[df2.index <= pd.to_datetime(end_date)]

        if df1 is None or len(df1) < 10:
            if verbose:
                buffered_print(f"{RED}Insufficient data for {t1}{ENDC}", "ERROR")
            continue

        if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
            if verbose:
                buffered_print(
                    f"{RED} Invalid data type for pair {pair}. \n {t1}/{t2} \n Skipping...",
                    "ERROR",
                )
            continue

        if df1 is None or df2 is None or df1.empty or df2.empty:
            if verbose:
                buffered_print(f"{RED} Skipping invalid pair: {pair}", "WARNING")
            continue

        df1, df2 = df1.align(df2, join="inner")
        df1 = df1.ffill().bfill()
        df2 = df2.ffill().bfill()

        if df1.empty or df2.empty:
            if verbose:
                buffered_print(
                    f"{RED} Skipping empty aligned pair: {pair}", "WARNING")
            continue

        if not df1.empty and not df2.empty:
            data1 = bt.feeds.PandasData(dataname=df1, name=f"{t1}_{t2}_LONG")
            data2 = bt.feeds.PandasData(dataname=df2, name=f"{t1}_{t2}_SHORT")

            all_names_list = [d._name for d in cerebro.datas]

            if data1._name in all_names_list or data2._name in all_names_list:
                if verbose:
                    buffered_print(
                        f"Skipping duplicate feeds for {pair}", "WARNING")
                continue

            cerebro.adddata(data1)
            cerebro.adddata(data2)
            valid_pairs.append((t1, t2))

    if verbose:
        buffered_print("\n")

    if not valid_pairs:
        if verbose:
            buffered_print(f"{RED} No valid pairs found.{ENDC}", "ERROR")
        return None, []

    cerebro.addstrategy(
        MeanReversionStrategy,
        pairs=valid_pairs,
        use_rl=use_rl,
    )

    cerebro.broker.set_cash(initial_cash)
    cerebro.broker.setcommission(commission=0.0005)

    cerebro.addanalyzer(EnhancedReturnsAnalyzer, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")
    cerebro.addanalyzer(SpreadTracker, _name="spread_tracker")
    cerebro.addanalyzer(bt.analyzers.TimeReturn,
                        _name='timereturn', timeframe=bt.TimeFrame.Days)

    return cerebro, valid_pairs


def _extract_results(
    strat,
    cerebro: bt.Cerebro,
    initial_cash: int,
    start_date: str,
    metrics_start_date: Optional[str] = None,
) -> Dict:
    """Extract structured metrics from a completed backtest run.

    If metrics_start_date is provided, equity curve and metrics are filtered
    to only include data from that date onward (used for WFA OOS windows
    where warm-up data precedes the actual test period).
    """
    result = {
        "final_value": cerebro.broker.getvalue(),
        "cagr": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "equity_curve": None,
        "daily_returns": None,
        "trades": [],
        "total_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "pairs_traded": [],
    }

    try:
        trade_analysis = strat.trade_recorder.get_trades()
        trade_summary = strat.trade_recorder.summarize()
        returns_analysis = strat.analyzers.returns.get_analysis()

        result["cagr"] = returns_analysis.get("cagr", 0.0)
        result["sharpe"] = returns_analysis.get("sharpe", 0.0)
        result["max_drawdown"] = returns_analysis.get("max_drawdown", 0.0)

        # Build equity curve from TimeReturn analyzer
        timereturn = strat.analyzers.timereturn.get_analysis()
        if timereturn:
            daily_rets = pd.Series(timereturn)
            daily_rets.index = pd.to_datetime(daily_rets.index)
            equity_curve = (1 + daily_rets).cumprod() * initial_cash
            initial_point = pd.Series(
                [initial_cash], index=[pd.Timestamp(start_date)]
            )
            equity_curve = pd.concat([initial_point, equity_curve])
        else:
            equity_curve = pd.Series(
                [initial_cash], index=[pd.Timestamp(start_date)]
            )

        # Filter to metrics_start_date if provided (WFA OOS mode)
        if metrics_start_date is not None:
            cutoff = pd.Timestamp(metrics_start_date)
            equity_curve = equity_curve[equity_curve.index >= cutoff]

            if len(equity_curve) > 1:
                # Recompute metrics on the filtered window only
                filtered_returns = equity_curve.pct_change().dropna()
                if len(filtered_returns) > 1 and filtered_returns.std() > 0:
                    result["sharpe"] = float(
                        (filtered_returns.mean() / filtered_returns.std())
                        * np.sqrt(252)
                    )
                else:
                    result["sharpe"] = 0.0

                total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
                if total_days > 0 and float(equity_curve.iloc[0]) > 0:
                    total_ret = float(equity_curve.iloc[-1]) / float(equity_curve.iloc[0])
                    years = total_days / 365.25
                    result["cagr"] = total_ret ** (1.0 / years) - 1.0 if years > 0 else 0.0

                cummax = equity_curve.cummax()
                dd = (equity_curve - cummax) / cummax
                result["max_drawdown"] = float(dd.min()) if len(dd) > 0 else 0.0

                result["final_value"] = float(equity_curve.iloc[-1])

        result["equity_curve"] = equity_curve
        result["daily_returns"] = equity_curve.pct_change().dropna() if len(equity_curve) > 1 else None

        # Trade data — filter by metrics_start_date if provided
        closed_trades = [t for t in trade_analysis if t.get("exit_dt") is not None]
        if metrics_start_date is not None:
            cutoff_dt = pd.Timestamp(metrics_start_date)
            closed_trades = [
                t for t in closed_trades
                if pd.Timestamp(t["entry_dt"]) >= cutoff_dt
            ]

        result["trades"] = closed_trades
        result["total_trades"] = len(closed_trades)

        if closed_trades:
            profitable = sum(1 for t in closed_trades if t["pnl"] > 0)
            result["win_rate"] = profitable / len(closed_trades)

            gross_profit = sum(t["pnl"] for t in closed_trades if t["pnl"] > 0)
            gross_loss = abs(sum(t["pnl"] for t in closed_trades if t["pnl"] <= 0))
            result["profit_factor"] = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

        # Pairs traded
        try:
            spread_data = strat.analyzers.spread_tracker.get_analysis().get("spread_data", {})
            result["pairs_traded"] = list(spread_data.keys())
        except Exception:
            result["pairs_traded"] = []

    except Exception as e:
        logger.warning("Result extraction error: %s", e)

    return result


def _default_result(initial_cash: int) -> Dict:
    return {
        "final_value": initial_cash,
        "cagr": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "equity_curve": None,
        "daily_returns": None,
        "trades": [],
        "total_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "pairs_traded": [],
    }


# ───────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────


def run_backtest_headless(
    loader: MarketDataProvider,
    pairs,
    start_date: str,
    end_date: str,
    initial_cash: int = 10000,
    use_rl: bool = False,
    reset_cache: bool = False,
    metrics_start_date: Optional[str] = None,
) -> Dict:
    """Silent backtest execution for WFA windows.

    Returns a dict with: final_value, cagr, sharpe, max_drawdown,
    equity_curve, daily_returns, trades, total_trades, win_rate,
    profit_factor, pairs_traded.

    If metrics_start_date is set, metrics are computed only from that
    date onward (warm-up data before it is used for strategy initialization
    but excluded from performance measurement).
    """
    cerebro, valid_pairs = _setup_cerebro(
        loader, pairs, start_date, end_date,
        initial_cash=initial_cash,
        use_rl=use_rl,
        reset_cache=reset_cache,
        verbose=False,
    )

    if cerebro is None:
        return _default_result(initial_cash)

    try:
        results = cerebro.run()
        strat = results[0]
    except Exception as e:
        logger.error("Headless backtest failed: %s", e)
        return _default_result(initial_cash)

    return _extract_results(strat, cerebro, initial_cash, start_date, metrics_start_date)


def run_backtest_report(
    loader: MarketDataProvider,
    pairs,
    start_date: str,
    end_date: str,
    initial_cash: int = 10000,
    use_rl: bool = False,
    reset_cache: bool = False,
    metrics_start_date: Optional[str] = None,
) -> BacktestReport:
    """
    Typed report API for production/research integrations.
    """
    raw = run_backtest_headless(
        loader=loader,
        pairs=pairs,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        use_rl=use_rl,
        reset_cache=reset_cache,
        metrics_start_date=metrics_start_date,
    )
    return BacktestReport.from_dict(raw)


def run_backtest(
    loader: MarketDataProvider,
    pairs,
    start_date,
    end_date,
    initial_cash=10000,
    reset_cache=False,
    use_rl=False,
):
    print_header("Backtest Engine Started")
    sleep(1)

    cerebro, valid_pairs = _setup_cerebro(
        loader, pairs, start_date, end_date,
        initial_cash=initial_cash,
        use_rl=use_rl,
        reset_cache=reset_cache,
        verbose=True,
    )

    if cerebro is None:
        return initial_cash

    # Execute Backtest with error handling
    try:
        results = cerebro.run()
        strat = results[0]

    except Exception as e:
        buffered_print(f"{RED}Backtest failed: {str(e)}{ENDC}", "ERROR")
        import traceback
        traceback.print_exc()
        return initial_cash

    try:
        trade_analysis = strat.trade_recorder.get_trades()
        trade_summary = strat.trade_recorder.summarize()
        returns_analysis = strat.analyzers.returns.get_analysis()

        # --- Extract total strategy equity ---
        total_strategy_equity_values = returns_analysis.get(
            "equity_values")
        if total_strategy_equity_values and strat.start_date:
            total_equity_dates = pd.date_range(
                start=pd.to_datetime(strat.start_date),
                periods=len(total_strategy_equity_values),
                freq='B'
            )
            initial_value = pd.Series([initial_cash],
                                      index=[pd.Timestamp(start_date)])
            total_strategy_equity_curve = pd.concat([
                initial_value,
                pd.Series(total_strategy_equity_values,
                          index=total_equity_dates)
            ])

            # --- Extract total strategy equity using TimeReturn ---
            timereturn = strat.analyzers.timereturn.get_analysis()

            if timereturn:
                daily_rets = pd.Series(timereturn)
                daily_rets.index = pd.to_datetime(daily_rets.index)

                equity_curve = (1 + daily_rets).cumprod() * initial_cash

                initial_point = pd.Series(
                    [initial_cash],
                    index=[pd.Timestamp(start_date)]
                )
                total_strategy_equity_curve = pd.concat(
                    [initial_point, equity_curve])
            else:
                total_strategy_equity_curve = pd.Series(
                    [initial_cash],
                    index=[pd.Timestamp(start_date)]
                )
        else:
            total_strategy_equity_curve = None
            buffered_print(
                "Could not retrieve total strategy equity values or start_date from strategy.", "WARNING")

    except Exception as e:
        buffered_print(f"{YELLOW}⚠ Analysis error: {str(e)}{ENDC}", "WARNING")
        trade_analysis = []
        trade_summary = {
            "total_trades": 0,
            "win_rate": 0,
            "gross_profit": 0,
            "gross_loss": 0,
            "profit_factor": 0,
            "avg_trade_duration": 0
        }
        returns_analysis = {
            "cagr": 0,
            "max_drawdown": 0,
            "sharpe": 0
        }
        total_strategy_equity_curve = None

    if total_strategy_equity_curve is None:
        total_strategy_equity_curve = pd.Series(
            [initial_cash],
            index=[pd.Timestamp(start_date)]
        )

    # Terminal Reporting
    print_section("Performance Summary", YELLOW)

    buffered_print(
        f"{BOLD}{GREEN}Initial Capital:{ENDC} ${initial_cash:,.2f}{ENDC}")
    buffered_print(
        f"{BOLD}{GREEN}Final Value:{ENDC} ${cerebro.broker.getvalue():,.2f}{ENDC}")
    buffered_print(
        f"{BOLD}{GREEN}CAGR:{ENDC} {returns_analysis['cagr']:.2%}{ENDC}")
    buffered_print(
        f"{BOLD}{GREEN}Max DD:{ENDC} {returns_analysis['max_drawdown']:.2%}{ENDC}")
    buffered_print(
        f"{BOLD}{GREEN}Sharpe:{ENDC} {returns_analysis['sharpe']:.2f}{ENDC}")

    # Trade Statistics
    print_section("Trade Analytics", YELLOW)

    closed_trades = [t for t in trade_analysis if t.get("exit_dt") is not None]
    total_trades = len(closed_trades)
    all_trades = trade_summary["total_trades"]
    if total_trades > 0:
        profitable = sum(1 for t in closed_trades if t["pnl"] > 0)
        win_rate = profitable / total_trades

        gross_profit = sum(t["pnl"] for t in closed_trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in closed_trades if t["pnl"] <= 0))
        profit_factor = gross_profit / \
            gross_loss if gross_loss != 0 else float("inf")

        sum_durations = sum(
            (t["exit_dt"] - t["entry_dt"]).days for t in closed_trades)
        avg_trade_duration = sum_durations / total_trades

        net_pnl = cerebro.broker.getvalue() - initial_cash
        net_pnl_pct = (cerebro.broker.getvalue() - initial_cash) / initial_cash

        returns_data = returns_analysis.get("returns", [])
        annualized_vol = (
            np.std(returns_data) *
            np.sqrt(252) if len(returns_data) >= 2 else 0
        )
        available_pairs = list(results[0].analyzers.spread_tracker.get_analysis()[
                               "spread_data"].keys())

        buffered_print(
            f"{BOLD}{GREEN}Total pairs Traded:{ENDC}{BOLD}{len(available_pairs)} pairs{ENDC}\n")
        buffered_print(
            f"{BOLD}{GREEN}Trades:{ENDC} {all_trades} | Win Rate: {win_rate:.1%}")
        buffered_print(
            f"{BOLD}{GREEN}Closed Trades:{ENDC} {total_trades} | Win Rate: {win_rate:.1%}")
        buffered_print(
            f"{BOLD}{GREEN}Total PNL:{ENDC} {net_pnl_pct:.2%}{ENDC}")
        buffered_print(
            f"{BOLD}{GREEN}Gross Profit/Loss:{ENDC} ${gross_profit:,.2f} / (${gross_loss:,.2f}){ENDC}")
        buffered_print(
            f"{BOLD}{GREEN}Profit Factor:{ENDC} {profit_factor:.2f}{ENDC}")
        buffered_print(
            f"{BOLD}{GREEN}Avg Trade Duration:{ENDC} {avg_trade_duration:.1f} days{ENDC}")
        buffered_print(
            f"{BOLD}{GREEN}Annualized Volatility:{ENDC} {annualized_vol:.2%}{ENDC}\n")

        buffered_print(
            f"{BOLD}{YELLOW}Total pairs Traded:{ENDC}\n{BOLD}{available_pairs}{ENDC}\n")
        # Plot specific pairs
        plot_pair_spreads(results, available_pairs, end_date, initial_capital=initial_cash,
                          total_strategy_equity_data=total_strategy_equity_curve)

    from ..printing_system import shutdown_printing
    shutdown_printing()

    return cerebro.broker.getvalue()
