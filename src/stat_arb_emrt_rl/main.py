# ───────────────────────────────────────────────
# Third-Party Package Imports
# ───────────────────────────────────────────────
import sys
import numpy as np

# ───────────────────────────────────────────────
# Project-Specific Imports
# ───────────────────────────────────────────────
from .financial_loader import FinancialLoader
from .cointegration import eg_analysis
from .backtest.backtest_engine import run_backtest
from .gpu_monitor import print_gpu_status, check_gpu_safe
from .printing_system import (
    GREEN,
    RED,
    YELLOW,
    CYAN,
    ENDC,
    BOLD,
    buffered_print,
    print_section,
    print_header,
)


def main():
    print_header(
        message="Statistical Arbitrage: Mean Reversion Pair Trade Strategy"
    )
    print_gpu_status()

    # ── Parse CLI flags ──
    use_rl = "--rl" in sys.argv
    include_crypto = "--crypto" in sys.argv
    crypto_only = "--crypto-only" in sys.argv
    run_rl_backtest = "--rl-backtest" in sys.argv

    if use_rl:
        buffered_print(f"{CYAN}RL-Enhanced mode enabled{ENDC}")
    if include_crypto or crypto_only:
        buffered_print(f"{CYAN}Crypto pairs {'only' if crypto_only else 'included'}{ENDC}")

    # ── Multi-coint discovery mode ──
    if "--discover" in sys.argv:
        from .multi_coint import MultiCointEngine, MultiCointConfig

        mc_config = MultiCointConfig()
        mc_config.include_sp500 = not crypto_only
        mc_config.include_crypto = True

        engine = MultiCointEngine(
            start_date="2023-01-01",
            end_date="2025-01-01",
            config=mc_config,
        )
        pairs_df, groups_df = engine.run(
            find_n_groups="--groups" in sys.argv,
            max_pairs=500,
            max_groups=300,
        )

        if not pairs_df.empty:
            buffered_print(f"\n{BOLD}Top 20 Pairs:{ENDC}")
            for _, r in pairs_df.head(20).iterrows():
                buffered_print(
                    f"  {r['ticker1']:>10} - {r['ticker2']:<10} "
                    f"{r['pair_type']:<14} score={r['composite_score']:.3f}"
                )
        return

    # ── RL-only backtest mode (standalone paper replication) ──
    if run_rl_backtest:
        import matplotlib
        matplotlib.use("TkAgg")
        from .rl_backtest import RLStatArbBacktest

        backtest = RLStatArbBacktest(
            formation_start="2022-01-01",
            formation_end="2022-12-31",
            trading_start="2023-01-01",
            trading_end="2023-12-31",
            initial_capital=100.0,
        )
        summary = backtest.run_all_paper_pairs()
        summary.to_csv("rl_backtest_results.csv", index=False)
        buffered_print(f"{GREEN}RL backtest results saved to rl_backtest_results.csv{ENDC}")
        return

    # ── Walk-Forward Analysis mode ──
    if "--wfa" in sys.argv:
        from .wfa import WalkForwardEngine, WFAConfig
        from .wfa.wfa_report import WFAReporter

        wfa_config = WFAConfig(
            overall_start="2020-01-01",
            overall_end="2025-06-09",
            train_months=12,
            test_months=3,
            step_months=3,
            anchored="--anchored" in sys.argv,
            pair_limit=75,
            initial_cash=10000,
            use_rl=use_rl,
            include_crypto=include_crypto,
            crypto_only=crypto_only,
        )

        engine = WalkForwardEngine(wfa_config)
        report = engine.run()

        reporter = WFAReporter(report)
        reporter.generate_full_report()
        return

    # ── Configuration ──
    config = {
        "start_date": "2024-06-01",
        "end_date": "2026-04-08",
        "coint_start_date": "2023-01-01",
        "coint_end_date": "2024-06-01",
        "pair_limit": 75,
        "initial_cash": 10000,
        "beta_only": False,
        "cache_path": "./ticker_cache/",
        "reset_cache": False,
        "include_crypto": include_crypto,
        "crypto_only": crypto_only,
    }

    # ── Load or Generate Pairs ──
    cache_suffix = "_crypto" if crypto_only else ("_with_crypto" if include_crypto else "")
    cache_file = (
        f"{config['cache_path']}ticker_pairs_{config['coint_start_date']}_"
        f"{config['coint_end_date']}_{config['beta_only']}{cache_suffix}.npy"
    )

    try:
        cached_pairs = np.load(cache_file).tolist()
        buffered_print(f"{GREEN}Loaded {len(cached_pairs)} cached pairs.{ENDC}")

    except FileNotFoundError:
        buffered_print(f"{YELLOW}No cached pairs found. Generating new...{ENDC}")

        _, ticker_pairs = eg_analysis(
            FinancialLoader(),
            config["coint_start_date"],
            config["coint_end_date"],
            include_crypto=include_crypto,
            crypto_only=crypto_only,
        )

        cached_pairs = [
            (row.Ticker1, row.Ticker2) for _, row in ticker_pairs.iterrows()
        ]
        np.save(cache_file, np.array(cached_pairs))

    # ── GPU safety check before heavy compute ──
    if not check_gpu_safe(max_temp=80):
        buffered_print(f"{RED}GPU too hot - reduce workload or wait{ENDC}")

    # ── Execute Backtest ──
    final_value = run_backtest(
        loader=FinancialLoader(),
        pairs=cached_pairs[: config["pair_limit"]],
        start_date=config["start_date"],
        end_date=config["end_date"],
        initial_cash=config["initial_cash"],
    )

    # ── Results ──
    print_section("Backtest Completed", GREEN)
    buffered_print(
        f"Initial: ${config['initial_cash']:,.2f} → Final: ${final_value:,.2f}"
    )
    print_gpu_status()


if __name__ == "__main__":
    main()
