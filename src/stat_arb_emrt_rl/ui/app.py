"""Streamlit interface for exploring the EMRT/RL stat-arb workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..emrt import compute_emrt
from ..rl_backtest import PAPER_PAIRS


def paper_pairs_frame() -> pd.DataFrame:
    """Return the paper benchmark pairs as a display-ready frame."""
    return pd.DataFrame(PAPER_PAIRS, columns=["ticker_1", "ticker_2", "sector"])


def demo_emrt_curve(seed: int = 7, n_steps: int = 500) -> pd.DataFrame:
    """Compute an offline EMRT sensitivity curve over synthetic OU speeds."""
    rng = np.random.default_rng(seed)
    rows = []
    for mean_reversion_speed in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        t = np.linspace(0, mean_reversion_speed * 8 * np.pi, n_steps)
        spread = np.sin(t) + 0.05 * rng.standard_normal(n_steps)
        emrt, tau_sequence = compute_emrt(spread, C=0.5)
        rows.append(
            {
                "mean_reversion_speed": mean_reversion_speed,
                "emrt": emrt,
                "tau_count": len(tau_sequence),
                "spread_std": float(np.std(spread)),
            }
        )
    return pd.DataFrame(rows)


def _render_overview(st) -> None:
    st.subheader("Research Workflow")
    pairs = paper_pairs_frame()
    left, middle, right = st.columns(3)
    left.metric("Benchmark pairs", len(pairs))
    middle.metric("Core methods", "DM / OU / RL")
    right.metric("Default mode", "Offline demo")

    st.dataframe(pairs, hide_index=True, use_container_width=True)

    st.subheader("EMRT Sensitivity")
    demo = demo_emrt_curve()
    st.line_chart(demo.set_index("mean_reversion_speed")["emrt"])
    st.dataframe(demo, hide_index=True, use_container_width=True)


def _render_discovery(st) -> None:
    st.subheader("Cointegration Discovery")
    with st.form("discovery-form"):
        cols = st.columns(4)
        start = cols[0].text_input("Start", value="2023-01-01")
        end = cols[1].text_input("End", value="2025-01-01")
        max_pairs = cols[2].number_input("Max pairs", min_value=10, max_value=5000, value=300)
        workers = cols[3].number_input("Workers", min_value=1, max_value=16, value=4)
        no_groups = st.checkbox("Skip Johansen group search", value=True)
        run = st.form_submit_button("Run discovery")

    if run:
        from ..multi_coint import MultiCointConfig, MultiCointEngine

        config = MultiCointConfig()
        config.max_pairs = int(max_pairs)
        config.max_workers = int(workers)
        engine = MultiCointEngine(start, end, config)
        pairs_df, groups_df = engine.run(find_n_groups=not no_groups, max_pairs=int(max_pairs))
        st.dataframe(pairs_df, use_container_width=True)
        if not groups_df.empty:
            st.dataframe(groups_df, use_container_width=True)


def _render_backtest(st) -> None:
    st.subheader("EMRT / RL Backtest")
    with st.form("backtest-form"):
        cols = st.columns(4)
        pair = cols[0].text_input("Pair", value="MSFT:GOOGL")
        formation_start = cols[1].text_input("Formation start", value="2022-01-01")
        formation_end = cols[2].text_input("Formation end", value="2022-12-31")
        initial_capital = cols[3].number_input("Capital", min_value=1.0, value=100.0)
        trading_start = st.text_input("Trading start", value="2023-01-01")
        trading_end = st.text_input("Trading end", value="2023-12-31")
        run = st.form_submit_button("Run backtest")

    if run:
        from ..rl_backtest import RLStatArbBacktest

        t1, t2 = [part.strip().upper() for part in pair.replace("-", ":").split(":", 1)]
        runner = RLStatArbBacktest(
            formation_start=formation_start,
            formation_end=formation_end,
            trading_start=trading_start,
            trading_end=trading_end,
            initial_capital=float(initial_capital),
        )
        runner.run_pair(t1, t2)
        st.dataframe(runner.generate_summary_table(), hide_index=True, use_container_width=True)


def _render_reference(st) -> None:
    st.subheader("Reference")
    st.markdown(
        """
        This project is an independent implementation inspired by:

        Ning, B. and Lee, C. (2024). *Advanced Statistical Arbitrage with Reinforcement Learning*.

        The repository includes citation metadata and links to the paper for attribution; it does
        not redistribute the paper text.
        """
    )


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="EMRT RL Stat Arb", layout="wide")
    st.title("EMRT RL Stat Arb")
    st.caption("Empirical mean reversion, cointegration discovery, and RL trading research toolkit.")

    page = st.sidebar.radio(
        "Workspace",
        ["Overview", "Discovery", "Backtest", "Reference"],
        index=0,
    )

    if page == "Overview":
        _render_overview(st)
    elif page == "Discovery":
        _render_discovery(st)
    elif page == "Backtest":
        _render_backtest(st)
    else:
        _render_reference(st)


if __name__ == "__main__":
    render()
