# ───────────────────────────────────────────────
# Empirical Mean Reversion Time (EMRT)
# Implements Section 3 of Ning & Lee (2024)
# "Advanced Statistical Arbitrage with Reinforcement Learning"
# ───────────────────────────────────────────────
import numpy as np
from typing import Dict, List, Optional, Tuple

from .printing_system import (
    buffered_print,
    print_header,
    print_section,
    GREEN,
    YELLOW,
    CYAN,
    ENDC,
)


# ───────────────────────────────────────────────
# Important Extremes Detection (Section 3.1)
# ───────────────────────────────────────────────
def find_important_extremes(
    ts: np.ndarray,
    C: float = 2.0,
) -> Tuple[List[int], List[int], List[str]]:
    """
    Find important local minima and maxima of a time series.
    Based on Fink and Gandhi (2007) as cited in the paper.

    A point X_m is an important minimum if there exist indices i <= m <= j such that:
      - X_m is the minimum among X_i, ..., X_j
      - X_i - X_m >= C * s  and  X_j - X_m >= C * s

    Similarly for important maxima.

    Args:
        ts: 1-D time series array
        C: threshold constant (paper uses C=2)

    Returns:
        (min_indices, max_indices, types) where types is ["min"/"max", ...]
    """
    n = len(ts)
    s = np.std(ts)
    if s < 1e-10:
        return [], [], []

    threshold = C * s
    min_indices = []
    max_indices = []

    # Find important minima
    for m in range(n):
        # Find the largest window [i, j] where ts[m] is minimum
        # and endpoint deviations exceed threshold
        i = m
        j = m

        # Extend left
        while i > 0 and ts[i - 1] >= ts[m]:
            i -= 1

        # Extend right
        while j < n - 1 and ts[j + 1] >= ts[m]:
            j += 1

        # Check if ts[m] is true minimum in [i, j]
        if np.argmin(ts[i:j + 1]) + i == m:
            left_dev = ts[i] - ts[m] if i < m else 0
            right_dev = ts[j] - ts[m] if j > m else 0
            if left_dev >= threshold and right_dev >= threshold:
                min_indices.append(m)

    # Find important maxima
    for m in range(n):
        i = m
        j = m

        while i > 0 and ts[i - 1] <= ts[m]:
            i -= 1

        while j < n - 1 and ts[j + 1] <= ts[m]:
            j += 1

        if np.argmax(ts[i:j + 1]) + i == m:
            left_dev = ts[m] - ts[i] if i < m else 0
            right_dev = ts[m] - ts[j] if j > m else 0
            if left_dev >= threshold and right_dev >= threshold:
                max_indices.append(m)

    # Combine and sort
    all_indices = sorted(set(min_indices + max_indices))
    types = ["min" if idx in min_indices else "max" for idx in all_indices]

    return min_indices, max_indices, types


# ───────────────────────────────────────────────
# Empirical Mean Reversion Time Calculation
# ───────────────────────────────────────────────
def compute_emrt(
    ts: np.ndarray,
    C: float = 2.0,
) -> Tuple[float, List[int]]:
    """
    Compute Empirical Mean Reversion Time (EMRT) for a time series.

    The sequence {tau_n} is constructed inductively:
      - tau_1: first local extreme
      - tau_2: first crossing of the sample mean after tau_1
      - tau_3: first local extreme after tau_2
      - tau_4: first crossing of the sample mean after tau_3
      ...

    EMRT r = (2/N) * sum_{n even} (tau_n - tau_{n-1})

    This measures the average time from an extreme to the next mean crossing.

    Args:
        ts: 1-D time series
        C: threshold for important extremes

    Returns:
        (emrt_value, tau_sequence)
    """
    n = len(ts)
    if n < 10:
        return float("inf"), []

    theta_hat = np.mean(ts)
    min_indices, max_indices, _ = find_important_extremes(ts, C)
    extreme_set = set(min_indices + max_indices)

    # Build tau sequence
    taus = []
    t = 0
    looking_for = "extreme"  # alternate between "extreme" and "crossing"

    while t < n:
        if looking_for == "extreme":
            # Find next local extreme after t
            found = False
            for idx in sorted(extreme_set):
                if idx >= t:
                    taus.append(idx)
                    t = idx + 1
                    looking_for = "crossing"
                    found = True
                    break
            if not found:
                break

        elif looking_for == "crossing":
            # Find first mean crossing after t
            found = False
            for i in range(t, n):
                # Check if ts crosses theta_hat between i-1 and i
                if i > 0:
                    prev_sign = np.sign(ts[i - 1] - theta_hat)
                    curr_sign = np.sign(ts[i] - theta_hat)
                    if prev_sign != curr_sign or abs(ts[i] - theta_hat) < 1e-10:
                        taus.append(i)
                        t = i + 1
                        looking_for = "extreme"
                        found = True
                        break
            if not found:
                break

    # Compute EMRT from even-indexed taus (reversion segments)
    if len(taus) < 2:
        return float("inf"), taus

    reversion_times = []
    for i in range(1, len(taus), 2):  # even-numbered (0-indexed: 1, 3, 5, ...)
        reversion_times.append(taus[i] - taus[i - 1])

    if not reversion_times:
        return float("inf"), taus

    emrt = float(np.mean(reversion_times))
    return emrt, taus


# ───────────────────────────────────────────────
# EMRT-Based Spread Optimization (Grid Search)
# ───────────────────────────────────────────────
def optimize_spread_coefficients(
    price_series: Dict[str, np.ndarray],
    reference_ticker: str,
    coeff_range: Tuple[float, float] = (-3.0, 3.0),
    coeff_step: float = 0.01,
    C: float = 2.0,
    variance_bound: Optional[float] = None,
) -> Dict:
    """
    Find optimal portfolio coefficients by minimizing EMRT via grid search.

    For n assets S_1, ..., S_n, form spread X = sum(a_i * S_i).
    Fix a_1 = 1 (reference), search over a_2, ..., a_n.

    From paper Section 3: "We set the first coefficient a_1 = 1.
    We then evaluate the EMRT of Y for each coefficient a_i,
    where a_i in [-3.00, -2.99, ..., 2.99, 3.00]."

    Args:
        price_series: dict mapping ticker -> normalized price array
        reference_ticker: the ticker with fixed coefficient a_1 = 1
        coeff_range: (min, max) range for other coefficients
        coeff_step: step size for grid search
        C: EMRT threshold
        variance_bound: optional upper bound M on sample variance

    Returns:
        Dict with optimal coefficients, EMRT, and spread
    """
    tickers = list(price_series.keys())
    if reference_ticker not in tickers:
        raise ValueError(f"Reference ticker {reference_ticker} not in price_series")

    other_tickers = [t for t in tickers if t != reference_ticker]
    ref_prices = price_series[reference_ticker]

    if len(other_tickers) == 0:
        raise ValueError("Need at least 2 tickers")

    # For pairs (2 tickers), do simple 1-D grid search
    if len(other_tickers) == 1:
        other_ticker = other_tickers[0]
        other_prices = price_series[other_ticker]

        coeffs = np.arange(coeff_range[0], coeff_range[1] + coeff_step, coeff_step)
        best_emrt = float("inf")
        best_coeff = 0.0
        best_spread = None

        for coeff in coeffs:
            spread = ref_prices - coeff * other_prices

            # Variance bound check
            if variance_bound is not None and np.var(spread) > variance_bound:
                continue

            # Mean constraint: we want spread centered near its mean
            emrt, _ = compute_emrt(spread, C)

            if emrt < best_emrt:
                best_emrt = emrt
                best_coeff = coeff
                best_spread = spread.copy()

        return {
            "coefficients": {reference_ticker: 1.0, other_ticker: best_coeff},
            "emrt": best_emrt,
            "spread": best_spread,
        }

    # For multi-asset: iterative coordinate descent over each coefficient
    # (full grid search over n-1 dimensions is intractable)
    coeffs = {t: 0.0 for t in other_tickers}
    best_emrt = float("inf")

    for iteration in range(3):  # 3 passes of coordinate descent
        for ticker in other_tickers:
            other_prices = price_series[ticker]
            search_range = np.arange(
                coeff_range[0], coeff_range[1] + coeff_step, coeff_step
            )

            best_c = coeffs[ticker]
            for c in search_range:
                coeffs[ticker] = c

                # Build spread with current coefficients
                spread = ref_prices.copy()
                for t in other_tickers:
                    spread = spread - coeffs[t] * price_series[t]

                if variance_bound is not None and np.var(spread) > variance_bound:
                    continue

                emrt, _ = compute_emrt(spread, C)
                if emrt < best_emrt:
                    best_emrt = emrt
                    best_c = c

            coeffs[ticker] = best_c

    # Compute final spread
    final_spread = ref_prices.copy()
    for t in other_tickers:
        final_spread = final_spread - coeffs[t] * price_series[t]

    all_coeffs = {reference_ticker: 1.0}
    all_coeffs.update(coeffs)

    return {
        "coefficients": all_coeffs,
        "emrt": best_emrt,
        "spread": final_spread,
    }


def compare_spread_methods(
    s1_prices: np.ndarray,
    s2_prices: np.ndarray,
    ou_beta: float,
    C: float = 2.0,
) -> Dict:
    """
    Compare three spread construction methods as in paper Table 2:
      1. Distance Method (DM): B = 1.0
      2. OU Method: B = ou_beta (from MLE)
      3. EMRT Method: B = argmin EMRT

    Args:
        s1_prices: normalized prices of ticker 1
        s2_prices: normalized prices of ticker 2
        ou_beta: beta from OU MLE optimization
        C: EMRT threshold

    Returns:
        Dict with results for each method
    """
    results = {}

    # Distance Method
    dm_spread = s1_prices - 1.0 * s2_prices
    dm_emrt, _ = compute_emrt(dm_spread, C)
    results["DM"] = {"beta": 1.0, "emrt": dm_emrt, "spread": dm_spread}

    # OU Method
    ou_spread = s1_prices - ou_beta * s2_prices
    ou_emrt, _ = compute_emrt(ou_spread, C)
    results["OU"] = {"beta": ou_beta, "emrt": ou_emrt, "spread": ou_spread}

    # EMRT Method
    emrt_result = optimize_spread_coefficients(
        price_series={"S1": s1_prices, "S2": s2_prices},
        reference_ticker="S1",
        C=C,
    )
    emrt_beta = emrt_result["coefficients"]["S2"]
    results["EMRT"] = {
        "beta": emrt_beta,
        "emrt": emrt_result["emrt"],
        "spread": emrt_result["spread"],
    }

    return results
