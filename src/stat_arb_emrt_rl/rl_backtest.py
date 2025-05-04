# ───────────────────────────────────────────────
# RL-Enhanced Statistical Arbitrage Backtest
# Full pipeline: EMRT spread construction → RL training → Trading → Evaluation
# Implements Sections 3-5 of Ning & Lee (2024)
# ───────────────────────────────────────────────
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .financial_loader import FinancialLoader
from .data_provider import FinancialLoaderProvider, MarketDataProvider
from .ou_optimizer import OUOptimizer
from .emrt import compute_emrt, optimize_spread_coefficients, compare_spread_methods
from .rl_agent import (
    TabularQAgent,
    RLAgentConfig,
    train_agent,
    run_rl_trading,
    simulate_ou_spread,
    DEFAULT_LOOKBACK,
    DEFAULT_THRESHOLD,
)
from .printing_system import (
    buffered_print,
    print_header,
    print_section,
    print_status,
    GREEN,
    RED,
    YELLOW,
    CYAN,
    BLUE,
    BOLD,
    ENDC,
)


# ───────────────────────────────────────────────
# Paper's Stock Pairs (Section 5.3.2)
# ───────────────────────────────────────────────
PAPER_PAIRS = [
    ("MSFT", "GOOGL", "Technology"),
    ("CVS", "JNJ", "Healthcare"),
    ("CL", "KMB", "Consumer Goods"),
    ("V", "MA", "Financials"),
    ("GE", "BA", "Industrials"),
    ("OXY", "XOM", "Energy"),
    ("WELL", "VTR", "Real Estate"),
    ("PPG", "SHW", "Materials"),
    ("VZ", "TMUS", "Telecom"),
    ("CSX", "NSC", "Transportation"),
]

# Crypto pairs with known cointegration relationships
CRYPTO_PAIRS = [
    ("BTC-USD", "ETH-USD", "Crypto-L1"),
    ("SOL-USD", "AVAX-USD", "Crypto-L1-Alt"),
    ("LINK-USD", "DOT-USD", "Crypto-Oracle"),
    ("UNI-USD", "ATOM-USD", "Crypto-DeFi"),
    ("LTC-USD", "BTC-USD", "Crypto-Store"),
    ("MATIC-USD", "ARB-USD", "Crypto-L2"),
]


# ───────────────────────────────────────────────
# Benchmark: Distance Method Trading
# ───────────────────────────────────────────────
def run_dm_trading(
    spread: np.ndarray,
    mean_est: float,
    std_est: float,
    k: float = 1.0,
    initial_capital: float = 100.0,
    c: float = 0.001,
) -> Dict:
    """
    Distance Method (Gatev et al. 2006) benchmark.

    Entry: buy if X_t - x_bar < -k * s
    Exit:  close if X_t - x_bar > k * s

    Uses dollar-based PnL: invest capital, profit = shares * spread_change.
    """
    n = len(spread)
    position = 0
    capital = initial_capital
    equity = np.full(n, initial_capital)
    trade_log = []
    entry_price = 0.0
    entry_idx = 0
    n_shares = 0.0

    for t in range(1, n):
        deviation = spread[t] - mean_est

        if position == 0 and deviation < -k * std_est:
            position = 1
            entry_price = spread[t]
            entry_idx = t
            # Invest all capital: buy spread units
            unit_cost = max(abs(entry_price), 0.01)
            n_shares = capital / unit_cost
        elif position == 1 and deviation > k * std_est:
            dollar_pnl = n_shares * (spread[t] - entry_price)
            commission = n_shares * unit_cost * c
            capital += dollar_pnl - commission
            pnl_pct = dollar_pnl / (n_shares * unit_cost) if n_shares > 0 else 0
            trade_log.append({
                "entry_idx": entry_idx, "exit_idx": t,
                "entry_price": entry_price, "exit_price": spread[t],
                "pnl_pct": pnl_pct, "pnl_net": pnl_pct - c,
            })
            position = 0
            n_shares = 0.0

        # Mark to market
        if position == 1:
            mtm = n_shares * (spread[t] - entry_price)
            equity[t] = capital + mtm
        else:
            equity[t] = capital

    # Close at terminal
    if position == 1:
        dollar_pnl = n_shares * (spread[-1] - entry_price)
        capital += dollar_pnl
        pnl_pct = dollar_pnl / (n_shares * max(abs(entry_price), 0.01)) if n_shares > 0 else 0
        equity[-1] = capital
        trade_log.append({
            "entry_idx": entry_idx, "exit_idx": n - 1,
            "entry_price": entry_price, "exit_price": spread[-1],
            "pnl_pct": pnl_pct, "pnl_net": pnl_pct, "terminal": True,
        })

    return _compute_metrics(equity, trade_log, initial_capital, "DM")


# ───────────────────────────────────────────────
# Benchmark: OU Trading
# ───────────────────────────────────────────────
def run_ou_trading(
    spread: np.ndarray,
    theta_hat: float,
    sigma_eq: float,
    k: float = 0.5,
    initial_capital: float = 100.0,
    c: float = 0.001,
) -> Dict:
    """
    OU Mean Reversion Trading (Avellaneda & Lee 2010) benchmark.

    sigma_eq = sigma_hat / sqrt(2 * mu_hat)
    Entry: buy if X_t - theta_hat < -k * sigma_eq
    Exit:  close if X_t - theta_hat > k * sigma_eq

    Uses dollar-based PnL consistent with DM benchmark.
    """
    n = len(spread)
    position = 0
    capital = initial_capital
    equity = np.full(n, initial_capital)
    trade_log = []
    entry_price = 0.0
    entry_idx = 0
    n_shares = 0.0
    unit_cost = 0.01

    for t in range(1, n):
        deviation = spread[t] - theta_hat

        if position == 0 and deviation < -k * sigma_eq:
            position = 1
            entry_price = spread[t]
            entry_idx = t
            unit_cost = max(abs(entry_price), 0.01)
            n_shares = capital / unit_cost
        elif position == 1 and deviation > k * sigma_eq:
            dollar_pnl = n_shares * (spread[t] - entry_price)
            commission = n_shares * unit_cost * c
            capital += dollar_pnl - commission
            pnl_pct = dollar_pnl / (n_shares * unit_cost) if n_shares > 0 else 0
            trade_log.append({
                "entry_idx": entry_idx, "exit_idx": t,
                "entry_price": entry_price, "exit_price": spread[t],
                "pnl_pct": pnl_pct, "pnl_net": pnl_pct - c,
            })
            position = 0
            n_shares = 0.0

        if position == 1:
            mtm = n_shares * (spread[t] - entry_price)
            equity[t] = capital + mtm
        else:
            equity[t] = capital

    if position == 1:
        dollar_pnl = n_shares * (spread[-1] - entry_price)
        capital += dollar_pnl
        pnl_pct = dollar_pnl / (n_shares * max(abs(entry_price), 0.01)) if n_shares > 0 else 0
        equity[-1] = capital
        trade_log.append({
            "entry_idx": entry_idx, "exit_idx": n - 1,
            "entry_price": entry_price, "exit_price": spread[-1],
            "pnl_pct": pnl_pct, "pnl_net": pnl_pct, "terminal": True,
        })

    return _compute_metrics(equity, trade_log, initial_capital, "OU")


# ───────────────────────────────────────────────
# Metrics Computation
# ───────────────────────────────────────────────
def _compute_metrics(
    equity: np.ndarray,
    trade_log: List[Dict],
    initial_capital: float,
    method: str,
) -> Dict:
    """Compute standard performance metrics matching paper's Table 3/4."""
    daily_returns = np.diff(equity) / equity[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    total_trades = len(trade_log)
    winning = sum(1 for t in trade_log if t["pnl_net"] > 0)

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0

    daily_ret_mean = float(np.mean(daily_returns)) * 100 if len(daily_returns) > 0 else 0
    daily_ret_std = float(np.std(daily_returns)) * 100 if len(daily_returns) > 0 else 0
    sharpe = (
        float(np.mean(daily_returns) / np.std(daily_returns))
        if len(daily_returns) > 1 and np.std(daily_returns) > 1e-10
        else 0
    )
    cumul_pnl = (equity[-1] / initial_capital - 1) * 100

    return {
        "method": method,
        "equity_curve": equity,
        "trade_log": trade_log,
        "DailyRet": daily_ret_mean,
        "DailyStd": daily_ret_std,
        "DailySR": sharpe,
        "MaxDD": max_dd * 100,
        "CumulPnL": cumul_pnl,
        "total_trades": total_trades,
        "win_rate": winning / total_trades if total_trades > 0 else 0,
    }


# ───────────────────────────────────────────────
# Full Pipeline: Formation + Trading
# ───────────────────────────────────────────────
class RLStatArbBacktest:
    """
    Complete pipeline implementing the paper's experimental framework.

    Formation period: estimate OU params, compute EMRT betas, train RL agent
    Trading period: deploy RL agent vs. DM and OU benchmarks
    """

    def __init__(
        self,
        formation_start: str = "2022-01-01",
        formation_end: str = "2022-12-31",
        trading_start: str = "2023-01-01",
        trading_end: str = "2023-12-31",
        initial_capital: float = 100.0,
        rl_config: Optional[RLAgentConfig] = None,
        data_provider: Optional[MarketDataProvider] = None,
    ):
        self.formation_start = formation_start
        self.formation_end = formation_end
        self.trading_start = trading_start
        self.trading_end = trading_end
        self.initial_capital = initial_capital
        self.rl_config = rl_config or RLAgentConfig()

        self.loader = data_provider or FinancialLoaderProvider(FinancialLoader())
        self.optimizer = OUOptimizer()

        self.pair_results: Dict[str, Dict] = {}
        self.agents: Dict[str, TabularQAgent] = {}

    def run_pair(
        self,
        t1: str,
        t2: str,
        sector: str = "",
    ) -> Optional[Dict]:
        """
        Run the full pipeline for one pair.

        Phase 1 (Formation): OU estimation, EMRT beta, RL training
        Phase 2 (Trading): Execute DM, OU, and RL methods
        """
        pair_key = f"{t1}-{t2}"
        print_header(f"Processing Pair: {pair_key} ({sector})")

        # ─── Phase 1: Formation Period ───
        print_section("Phase 1: Formation Period", CYAN)

        # Get formation data
        formation_df = self.loader.get_normalized_pair(
            t1, t2, self.formation_start, self.formation_end,
        )
        if formation_df is None or formation_df.empty:
            buffered_print(f"{RED}No formation data for {pair_key}{ENDC}")
            return None

        # OU parameter estimation
        ou_result = self.optimizer.optimize(formation_df, t1, t2)
        if ou_result is None:
            buffered_print(f"{RED}OU optimization failed for {pair_key}{ENDC}")
            return None

        beta_ou = ou_result["beta"]
        mu_hat = ou_result["mu"]
        theta_hat = ou_result["theta"]
        sigma_hat = ou_result["sigma"]

        buffered_print(
            f"{GREEN}OU Params: beta={beta_ou:.4f}, mu={mu_hat:.4f}, "
            f"theta={theta_hat:.4f}, sigma={sigma_hat:.4f}{ENDC}"
        )

        # Formation spread for EMRT
        norm_s1 = formation_df[f"Normalized {t1}"].values
        norm_s2 = formation_df[f"Normalized {t2}"].values

        # EMRT-optimized beta
        emrt_result = optimize_spread_coefficients(
            price_series={"S1": norm_s1, "S2": norm_s2},
            reference_ticker="S1",
            coeff_range=(-3.0, 3.0),
            coeff_step=0.01,
        )
        beta_emrt = emrt_result["coefficients"]["S2"]

        buffered_print(
            f"{GREEN}Betas: DM=1.0, OU={beta_ou:.4f}, EMRT={beta_emrt:.4f} | "
            f"EMRT={emrt_result['emrt']:.2f}{ENDC}"
        )

        # Train RL agent on simulated OU paths
        print_section("RL Agent Training", YELLOW)

        rl_config = RLAgentConfig()
        rl_config.ou_mu = max(mu_hat, 0.5)  # clamp for stability
        rl_config.ou_theta = theta_hat
        rl_config.ou_sigma = max(sigma_hat, 0.01)
        rl_config.n_training_paths = 1_000  # reduced for speed; scale up for production
        rl_config.n_epochs = 5

        agent = train_agent(
            config=rl_config,
            ou_params_range={
                "mu_range": (max(0.3, mu_hat * 0.5), mu_hat * 2.0),
                "sigma_range": (max(0.01, sigma_hat * 0.5), sigma_hat * 2.0),
            },
            verbose=True,
        )
        self.agents[pair_key] = agent

        # ─── Phase 2: Trading Period ───
        print_section("Phase 2: Trading Period", CYAN)

        trading_df = self.loader.get_normalized_pair(
            t1, t2, self.trading_start, self.trading_end,
        )
        if trading_df is None or trading_df.empty:
            buffered_print(f"{RED}No trading data for {pair_key}{ENDC}")
            return None

        trade_s1 = trading_df[f"Normalized {t1}"].values
        trade_s2 = trading_df[f"Normalized {t2}"].values

        # Formation period statistics for DM benchmark
        formation_spread_dm = norm_s1 - 1.0 * norm_s2
        dm_mean = float(np.mean(formation_spread_dm))
        dm_std = float(np.std(formation_spread_dm))

        # Equilibrium volatility for OU benchmark
        sigma_eq = sigma_hat / np.sqrt(2 * max(mu_hat, 1e-5))

        # Construct trading spreads with each beta
        spread_dm = trade_s1 - 1.0 * trade_s2
        spread_ou = trade_s1 - beta_ou * trade_s2
        spread_emrt = trade_s1 - beta_emrt * trade_s2

        # Run all three methods
        dm_results = run_dm_trading(
            spread_dm, dm_mean, dm_std,
            k=1.0, initial_capital=self.initial_capital,
        )

        ou_results = run_ou_trading(
            spread_ou, theta_hat, sigma_eq,
            k=0.5, initial_capital=self.initial_capital,
        )

        rl_results = run_rl_trading(
            agent, spread_emrt, theta_hat,
            initial_capital=self.initial_capital,
            c=rl_config.transaction_cost,
        )
        rl_results["method"] = "RL"

        # Store results
        pair_result = {
            "pair": pair_key,
            "sector": sector,
            "betas": {"DM": 1.0, "OU": beta_ou, "EMRT": beta_emrt},
            "ou_params": ou_result,
            "emrt": emrt_result["emrt"],
            "DM": dm_results,
            "OU": ou_results,
            "RL": rl_results,
        }
        self.pair_results[pair_key] = pair_result

        # Print comparison
        self._print_pair_comparison(pair_result)

        return pair_result

    def run_all_paper_pairs(self) -> pd.DataFrame:
        """Run all 10 pairs from the paper's experimental setup."""
        print_header("Running All Paper Pairs")

        for t1, t2, sector in PAPER_PAIRS:
            try:
                self.run_pair(t1, t2, sector)
            except Exception as e:
                buffered_print(f"{RED}Failed {t1}-{t2}: {str(e)}{ENDC}")

        return self.generate_summary_table()

    def run_crypto_pairs(self) -> pd.DataFrame:
        """Run all predefined crypto pairs."""
        print_header("Running Crypto Pairs")

        for t1, t2, sector in CRYPTO_PAIRS:
            try:
                self.run_pair(t1, t2, sector)
            except Exception as e:
                buffered_print(f"{RED}Failed {t1}-{t2}: {str(e)}{ENDC}")

        return self.generate_summary_table()

    def run_custom_pairs(self, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Run on custom pairs with optional sector labels."""
        for t1, t2 in pairs:
            try:
                self.run_pair(t1, t2)
            except Exception as e:
                buffered_print(f"{RED}Failed {t1}-{t2}: {str(e)}{ENDC}")

        return self.generate_summary_table()

    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate performance summary table matching paper's Tables 3 & 4.
        """
        rows = []
        for pair_key, pr in self.pair_results.items():
            for method in ["DM", "OU", "RL"]:
                m = pr[method]
                rows.append({
                    "Pair": pair_key,
                    "Method": method,
                    "Beta": pr["betas"].get(method, pr["betas"].get("EMRT", 1.0)),
                    "DailyRet (%)": m.get("DailyRet", m.get("daily_return_mean", 0) * 100),
                    "DailyStd (%)": m.get("DailyStd", m.get("daily_return_std", 0) * 100),
                    "DailySR": m.get("DailySR", m.get("sharpe_ratio", 0)),
                    "MaxDD (%)": m.get("MaxDD", m.get("max_drawdown", 0) * 100),
                    "CumulPnL (%)": m.get("CumulPnL", m.get("total_return_pct", 0)),
                    "Trades": m.get("total_trades", 0),
                    "Win Rate": m.get("win_rate", 0),
                })

        df = pd.DataFrame(rows)

        print_section("Performance Summary (All Pairs)", GREEN)
        self._print_summary_table(df)

        return df

    def _print_pair_comparison(self, pr: Dict):
        """Print single pair comparison."""
        print_section(f"Results: {pr['pair']} ({pr['sector']})", GREEN)

        header = f"{'Method':<8} {'DailyRet%':>10} {'DailyStd%':>10} {'SR':>8} {'MaxDD%':>8} {'CumulPnL%':>10} {'Trades':>7}"
        buffered_print(f"{BOLD}{header}{ENDC}")
        buffered_print("-" * len(header))

        for method in ["DM", "OU", "RL"]:
            m = pr[method]
            daily_ret = m.get("DailyRet", m.get("daily_return_mean", 0) * 100)
            daily_std = m.get("DailyStd", m.get("daily_return_std", 0) * 100)
            sr = m.get("DailySR", m.get("sharpe_ratio", 0))
            mdd = m.get("MaxDD", m.get("max_drawdown", 0) * 100)
            cpnl = m.get("CumulPnL", m.get("total_return_pct", 0))
            trades = m.get("total_trades", 0)

            color = GREEN if method == "RL" else CYAN
            buffered_print(
                f"{color}{method:<8} {daily_ret:>10.4f} {daily_std:>10.4f} "
                f"{sr:>8.4f} {mdd:>8.4f} {cpnl:>10.4f} {trades:>7}{ENDC}"
            )

    def _print_summary_table(self, df: pd.DataFrame):
        """Print formatted summary table."""
        # Pivot for cleaner display
        for method in ["DM", "OU", "RL"]:
            subset = df[df["Method"] == method]
            if subset.empty:
                continue

            color = GREEN if method == "RL" else YELLOW
            print_section(f"{method} Method", color)

            for _, row in subset.iterrows():
                buffered_print(
                    f"  {row['Pair']:<12} "
                    f"Ret={row['DailyRet (%)']:>8.4f}% "
                    f"Std={row['DailyStd (%)']:>8.4f}% "
                    f"SR={row['DailySR']:>7.4f} "
                    f"DD={row['MaxDD (%)']:>8.4f}% "
                    f"PnL={row['CumulPnL (%)']:>8.4f}% "
                    f"Trades={int(row['Trades']):>3}"
                )

        # Aggregate comparison
        print_section("Aggregate by Method", BLUE)
        agg = df.groupby("Method").agg({
            "DailyRet (%)": "mean",
            "DailySR": "mean",
            "CumulPnL (%)": "mean",
            "MaxDD (%)": "mean",
        }).round(4)
        buffered_print(str(agg))


# ───────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────
def plot_wealth_comparison(
    pair_result: Dict,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Plot wealth evolution for DM, OU, and RL methods.
    Matches paper's Figure 3.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for method, color, label in [
        ("DM", "blue", "DM"),
        ("OU", "orange", "OU"),
        ("RL", "green", "RL"),
    ]:
        m = pair_result[method]
        equity = m.get("equity_curve", np.array([]))
        if len(equity) > 0:
            ax.plot(equity, color=color, label=label, alpha=0.8)

    ax.set_title(f"Total Wealth: {pair_result['pair']} ({pair_result.get('sector', '')})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Total Wealth ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_rl_actions(
    pair_result: Dict,
    figsize: Tuple[int, int] = (14, 8),
):
    """
    Plot spread with RL buy/sell signals.
    Matches paper's Figure 2.
    """
    import matplotlib.pyplot as plt

    rl = pair_result["RL"]
    equity = rl.get("equity_curve", np.array([]))
    actions = rl.get("actions", np.array([]))
    positions = rl.get("positions", np.array([]))

    if len(equity) == 0:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Spread + signals
    ax1.plot(equity, color="gray", alpha=0.5, label="Equity")
    buy_mask = actions == 1
    sell_mask = actions == -1
    x = np.arange(len(actions))
    if buy_mask.any():
        ax1.scatter(x[buy_mask], equity[buy_mask], color="green", marker="^",
                    s=40, label="Buy", zorder=5)
    if sell_mask.any():
        ax1.scatter(x[sell_mask], equity[sell_mask], color="red", marker="v",
                    s=40, label="Sell", zorder=5)
    ax1.set_title(f"RL Trading Actions: {pair_result['pair']}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Position over time
    ax2.fill_between(x, positions, alpha=0.3, color="blue", step="post")
    ax2.set_ylabel("Position")
    ax2.set_xlabel("Time")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Flat", "Long"])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ───────────────────────────────────────────────
# CLI Entry Point
# ───────────────────────────────────────────────
def main():
    """Run the full paper replication with default parameters."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    print_header("Advanced Statistical Arbitrage with Reinforcement Learning")
    buffered_print(f"{CYAN}Replicating Ning & Lee (2024) experimental setup{ENDC}")

    backtest = RLStatArbBacktest(
        formation_start="2022-01-01",
        formation_end="2022-12-31",
        trading_start="2023-01-01",
        trading_end="2023-12-31",
        initial_capital=100.0,
    )

    summary = backtest.run_all_paper_pairs()

    # Plot wealth curves for all pairs
    for pair_key, result in backtest.pair_results.items():
        fig = plot_wealth_comparison(result)
        if fig:
            plt.show(block=False)

    # Save summary
    summary.to_csv("rl_backtest_results.csv", index=False)
    buffered_print(f"\n{GREEN}Results saved to rl_backtest_results.csv{ENDC}")

    from .printing_system import shutdown_printing
    shutdown_printing()

    plt.show()


if __name__ == "__main__":
    main()
