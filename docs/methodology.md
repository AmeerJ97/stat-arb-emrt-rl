# Methodology

The implementation follows the shape of Ning and Lee's EMRT/RL statistical
arbitrage workflow while keeping the code organized as a reusable package.

## EMRT

`stat_arb_emrt_rl.emrt` implements the empirical mean reversion time procedure:

1. Detect important local minima and maxima using a standard-deviation-scaled
   threshold.
2. Build a sequence alternating between extremes and subsequent mean crossings.
3. Compute the average time from each important extreme to the next crossing.

The spread optimizer fixes one reference coefficient and searches over candidate
coefficients for the remaining asset to minimize EMRT.

## Baselines

`stat_arb_emrt_rl.rl_backtest` keeps the original three-way comparison:

- Distance method benchmark with thresholded spread deviations
- OU benchmark using optimized OU parameters
- RL method using the EMRT-optimized spread

## Reinforcement Learning

`stat_arb_emrt_rl.rl_agent` uses a tabular Q-learning policy. State comes from
recent spread movement signs, valid actions depend on current position, and the
reward function includes transaction costs.

## Discovery And WFA

`stat_arb_emrt_rl.multi_coint` supports multi-timeframe pair scoring and Johansen
group discovery. `stat_arb_emrt_rl.wfa` wraps repeated in-sample discovery and
out-of-sample backtesting windows to expose overfitting and pair turnover.

## Package Boundaries

The cleaned subsystem packages are thin re-export layers over the proven legacy
implementation:

- `core`: EMRT and OU primitives
- `data`: market-data adapters
- `discovery`: cointegration engines
- `rl`: RL agent and trading helpers
- `backtesting`: paper-style benchmark runners

This keeps imports coherent without rewriting the numerical behavior.
