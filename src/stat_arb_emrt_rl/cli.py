"""Command-line entry point for the EMRT statistical arbitrage toolkit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import __version__
from .rl_backtest import PAPER_PAIRS, RLStatArbBacktest


def _parse_pairs(values: Iterable[str] | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw in values or []:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            sep = ":" if ":" in item else "-"
            if sep not in item:
                raise argparse.ArgumentTypeError(
                    f"Pair '{item}' must use TICKER1:TICKER2 or TICKER1-TICKER2"
                )
            left, right = item.split(sep, 1)
            pairs.append((left.strip().upper(), right.strip().upper()))
    return pairs


def _write_frame(df: pd.DataFrame, output: str, path: str | None = None) -> None:
    if output == "json":
        text = df.to_json(orient="records", indent=2)
    elif output == "csv":
        text = df.to_csv(index=False)
    else:
        text = df.to_string(index=False)

    if path:
        Path(path).write_text(text, encoding="utf-8")
    else:
        print(text)


def cmd_paper_pairs(args: argparse.Namespace) -> int:
    df = pd.DataFrame(PAPER_PAIRS, columns=["ticker_1", "ticker_2", "sector"])
    _write_frame(df, args.output, args.path)
    return 0


def cmd_emrt_demo(args: argparse.Namespace) -> int:
    from .ui.app import demo_emrt_curve

    df = demo_emrt_curve(seed=args.seed, n_steps=args.steps)
    _write_frame(df, args.output, args.path)
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    from .multi_coint import MultiCointConfig, MultiCointEngine

    config = MultiCointConfig()
    config.include_sp500 = not args.crypto_only
    config.include_crypto = not args.equity_only
    config.max_pairs = args.max_pairs
    config.max_workers = args.workers

    engine = MultiCointEngine(args.start, args.end, config)
    pairs_df, groups_df = engine.run(
        find_n_groups=not args.no_groups,
        max_pairs=args.max_pairs,
        max_groups=args.max_groups,
        n_group_tickers=args.group_tickers,
    )
    result = {
        "pairs": pairs_df.head(args.limit).to_dict("records"),
        "groups": groups_df.head(args.limit).to_dict("records"),
    }
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    runner = RLStatArbBacktest(
        formation_start=args.formation_start,
        formation_end=args.formation_end,
        trading_start=args.trading_start,
        trading_end=args.trading_end,
        initial_capital=args.initial_capital,
    )

    if args.paper_pairs:
        summary = runner.run_all_paper_pairs()
    else:
        pairs = _parse_pairs(args.pair)
        if not pairs:
            raise SystemExit("backtest requires --paper-pairs or at least one --pair")
        summary = runner.run_custom_pairs(pairs)

    _write_frame(summary, args.output, args.path)
    return 0


def cmd_wfa(args: argparse.Namespace) -> int:
    from .wfa import WFAConfig, WFAReporter, WalkForwardEngine

    config = WFAConfig(
        overall_start=args.start,
        overall_end=args.end,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        anchored=args.anchored,
        pair_limit=args.pair_limit,
        initial_cash=args.initial_cash,
        use_rl=args.use_rl,
        include_crypto=args.include_crypto,
        crypto_only=args.crypto_only,
        output_dir=args.output_dir,
    )
    if args.dry_run:
        print(json.dumps(config.__dict__, indent=2, default=str))
        return 0

    report = WalkForwardEngine(config).run()
    WFAReporter(report).generate_full_report()
    return 0


def cmd_streamlit(args: argparse.Namespace) -> int:
    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("ui") / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), *args.streamlit_args]
    streamlit_cli.main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emrt-rl",
        description="EMRT and reinforcement-learning statistical arbitrage workflows.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    paper_pairs = subparsers.add_parser("paper-pairs", help="List paper benchmark pairs.")
    paper_pairs.add_argument("--output", choices=["table", "json", "csv"], default="table")
    paper_pairs.add_argument("--path", help="Optional output file.")
    paper_pairs.set_defaults(func=cmd_paper_pairs)

    demo = subparsers.add_parser("emrt-demo", help="Run an offline EMRT simulation demo.")
    demo.add_argument("--steps", type=int, default=500)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--output", choices=["table", "json", "csv"], default="table")
    demo.add_argument("--path", help="Optional output file.")
    demo.set_defaults(func=cmd_emrt_demo)

    discover = subparsers.add_parser("discover", help="Run multi-timeframe cointegration discovery.")
    discover.add_argument("--start", default="2023-01-01")
    discover.add_argument("--end", default="2025-01-01")
    discover.add_argument("--max-pairs", type=int, default=300)
    discover.add_argument("--max-groups", type=int, default=300)
    discover.add_argument("--group-tickers", type=int, default=50)
    discover.add_argument("--limit", type=int, default=20)
    discover.add_argument("--workers", type=int, default=4)
    discover.add_argument("--no-groups", action="store_true")
    discover.add_argument("--equity-only", action="store_true")
    discover.add_argument("--crypto-only", action="store_true")
    discover.set_defaults(func=cmd_discover)

    backtest = subparsers.add_parser("backtest", help="Run EMRT/RL paper-style backtests.")
    backtest.add_argument("--paper-pairs", action="store_true")
    backtest.add_argument("--pair", action="append", help="Ticker pair, for example MSFT:GOOGL.")
    backtest.add_argument("--formation-start", default="2022-01-01")
    backtest.add_argument("--formation-end", default="2022-12-31")
    backtest.add_argument("--trading-start", default="2023-01-01")
    backtest.add_argument("--trading-end", default="2023-12-31")
    backtest.add_argument("--initial-capital", type=float, default=100.0)
    backtest.add_argument("--output", choices=["table", "json", "csv"], default="table")
    backtest.add_argument("--path", help="Optional output file.")
    backtest.set_defaults(func=cmd_backtest)

    wfa = subparsers.add_parser("wfa", help="Run walk-forward analysis.")
    wfa.add_argument("--start", default="2020-01-01")
    wfa.add_argument("--end", default="2025-06-09")
    wfa.add_argument("--train-months", type=int, default=12)
    wfa.add_argument("--test-months", type=int, default=3)
    wfa.add_argument("--step-months", type=int, default=3)
    wfa.add_argument("--pair-limit", type=int, default=75)
    wfa.add_argument("--initial-cash", type=int, default=10000)
    wfa.add_argument("--output-dir", default="./wfa_results")
    wfa.add_argument("--anchored", action="store_true")
    wfa.add_argument("--use-rl", action="store_true")
    wfa.add_argument("--include-crypto", action="store_true")
    wfa.add_argument("--crypto-only", action="store_true")
    wfa.add_argument("--dry-run", action="store_true")
    wfa.set_defaults(func=cmd_wfa)

    streamlit = subparsers.add_parser("streamlit", help="Launch the Streamlit interface.")
    streamlit.add_argument("streamlit_args", nargs=argparse.REMAINDER)
    streamlit.set_defaults(func=cmd_streamlit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        if args.command == "streamlit":
            args.streamlit_args.extend(unknown)
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
