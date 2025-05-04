# Usage

## Offline Demo

Use the demo command when validating the package without downloading market data:

```bash
emrt-rl emrt-demo --output table
```

This generates synthetic mean-reverting spreads and computes EMRT across several
mean-reversion speeds.

## Paper Pair List

```bash
emrt-rl paper-pairs --output csv --path paper_pairs.csv
```

The benchmark list is the ten pair setup used by the paper implementation path.

## Discovery

```bash
emrt-rl discover --start 2023-01-01 --end 2025-01-01 --no-groups
```

Discovery builds a ticker universe, fetches adjusted close data, aligns calendars,
and runs multi-timeframe cointegration scoring. Remove `--no-groups` to include
Johansen group search.

## Backtest

```bash
emrt-rl backtest --pair MSFT:GOOGL
```

The backtest runs the two-period paper workflow:

1. Formation period: normalize prices, estimate OU parameters, optimize EMRT beta,
   and train the tabular RL agent.
2. Trading period: compare distance method, OU trading, and RL trading.

## Walk-Forward Analysis

```bash
emrt-rl wfa --dry-run
emrt-rl wfa --start 2020-01-01 --end 2025-06-09
```

The CLI exposes a dry-run path for checking configuration before launching the
full market-data workflow.
