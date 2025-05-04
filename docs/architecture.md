# Architecture

```text
src/stat_arb_emrt_rl/
  core/              Clean EMRT and OU public imports
  data/              Market-data provider public imports
  discovery/         Cointegration public imports
  rl/                RL public imports
  backtesting/       Paper-style benchmark public imports
  backtest/          Legacy Backtrader engine and plotting internals
  wfa/               Walk-forward engine and reports
  ui/                Streamlit interface
  *.py               Preserved implementation modules from the manual EMRT repo
```

The package intentionally keeps legacy module names importable under
`stat_arb_emrt_rl.*`. The new subsystem packages provide a cleaner surface for
new code while preserving compatibility with the working implementation.

## Data Flow

```text
market data -> normalized pair prices -> spread construction
            -> EMRT/OU fitting -> benchmark/RL trading
            -> summaries, plots, WFA reports, Streamlit display
```

## Side Effects

Runtime logs and generated research outputs are kept outside version control:

- `logs/`
- `coint_cache/`
- `ticker_cache/`
- `wfa_results/`
- generated CSV, PNG, NPY, and report artifacts

## Extension Points

Use `MarketDataProvider` for tests or alternate data sources. The tests use
`InMemoryProvider` so behavior can be verified without network access.
