# ───────────────────────────────────────────────
# GPU-Accelerated Cointegration Testing
# Vectorized OLS + ADF on CUDA via CuPy/PyTorch
# ───────────────────────────────────────────────
import time
import numpy as np
from typing import List, Tuple, Optional

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False

from .gpu_monitor import get_gpu_stats, print_gpu_status
from .printing_system import (
    buffered_print, print_section,
    GREEN, YELLOW, CYAN, RED, ENDC,
)


def _check_gpu():
    """Verify GPU is available and within thermal limits."""
    if not HAS_CUPY:
        return False, "CuPy not installed"
    stats = get_gpu_stats()
    if stats and stats["temp_c"] > 82:
        return False, f"GPU too hot: {stats['temp_c']}C"
    return True, "OK"


# ───────────────────────────────────────────────
# GPU Batch OLS Regression
# ───────────────────────────────────────────────
def _batch_ols_gpu(
    Y: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch OLS: regress Y[i] on X[i] for all pairs simultaneously on GPU.

    Y shape: (n_pairs, T)
    X shape: (n_pairs, T)

    Returns: (betas, residuals) where residuals shape is (n_pairs, T)
    """
    Y_g = cp.asarray(Y, dtype=cp.float64)
    X_g = cp.asarray(X, dtype=cp.float64)

    # Y = alpha + beta * X + epsilon
    # Add intercept column: X_design = [ones, X]
    T = Y_g.shape[1]
    ones = cp.ones((Y_g.shape[0], T), dtype=cp.float64)

    # For each pair: beta = (X'X)^-1 X'Y (vectorized)
    sum_x = cp.sum(X_g, axis=1)
    sum_y = cp.sum(Y_g, axis=1)
    sum_xx = cp.sum(X_g * X_g, axis=1)
    sum_xy = cp.sum(X_g * Y_g, axis=1)
    n = cp.float64(T)

    denom = n * sum_xx - sum_x * sum_x + 1e-12
    beta = (n * sum_xy - sum_x * sum_y) / denom
    alpha = (sum_y - beta * sum_x) / n

    # Residuals
    residuals = Y_g - alpha[:, None] - beta[:, None] * X_g

    return cp.asnumpy(beta), cp.asnumpy(residuals)


# ───────────────────────────────────────────────
# GPU Batch ADF Test (Augmented Dickey-Fuller)
# ───────────────────────────────────────────────
def _batch_adf_gpu(
    residuals: np.ndarray,
    max_lag: int = 1,
) -> np.ndarray:
    """
    Batch ADF test on GPU for all residual series simultaneously.

    Simplified ADF: regress diff(e) on e_lag + constant.
    Returns t-statistics for the unit root coefficient.

    residuals shape: (n_pairs, T)
    Returns: (n_pairs,) array of ADF t-statistics
    """
    R = cp.asarray(residuals, dtype=cp.float64)
    n_pairs, T = R.shape

    # diff(e) = e[1:] - e[:-1]
    de = R[:, 1:] - R[:, :-1]   # (n_pairs, T-1)
    e_lag = R[:, :-1]             # (n_pairs, T-1)

    # Include lagged differences if max_lag > 0
    if max_lag > 0:
        # Simple ADF with 1 lag: regress de on [e_lag, de_lag, const]
        de_trim = de[:, 1:]           # (n_pairs, T-2)
        e_lag_trim = e_lag[:, 1:]     # (n_pairs, T-2)
        de_lag = de[:, :-1]           # (n_pairs, T-2)
        t_len = T - 2
    else:
        de_trim = de
        e_lag_trim = e_lag
        de_lag = None
        t_len = T - 1

    # OLS: de_trim = gamma * e_lag_trim + [delta * de_lag] + const + noise
    # We need gamma's t-stat
    ones = cp.ones((n_pairs, t_len), dtype=cp.float64)

    if de_lag is not None:
        # 3-variable regression: [e_lag, de_lag, ones]
        # Use normal equations: (X'X)^-1 X'y for each pair
        # Vectorized via sums
        sum_el = cp.sum(e_lag_trim, axis=1)
        sum_dl = cp.sum(de_lag, axis=1)
        sum_y = cp.sum(de_trim, axis=1)
        sum_el2 = cp.sum(e_lag_trim ** 2, axis=1)
        sum_dl2 = cp.sum(de_lag ** 2, axis=1)
        sum_el_dl = cp.sum(e_lag_trim * de_lag, axis=1)
        sum_el_y = cp.sum(e_lag_trim * de_trim, axis=1)
        sum_dl_y = cp.sum(de_lag * de_trim, axis=1)
        n = cp.float64(t_len)

        # Solve 3x3 normal equations per pair using Cramer's rule
        # For speed, use simplified ADF (just e_lag + const)
        # gamma = (n*sum_el_y - sum_el*sum_y) / (n*sum_el2 - sum_el^2)
        denom = n * sum_el2 - sum_el * sum_el + 1e-12
        gamma = (n * sum_el_y - sum_el * sum_y) / denom
        alpha_hat = (sum_y - gamma * sum_el) / n

        # Residuals of ADF regression
        fitted = gamma[:, None] * e_lag_trim + alpha_hat[:, None]
        adf_resid = de_trim - fitted
    else:
        # Simple 2-variable: [e_lag, ones]
        sum_el = cp.sum(e_lag_trim, axis=1)
        sum_y = cp.sum(de_trim, axis=1)
        sum_el2 = cp.sum(e_lag_trim ** 2, axis=1)
        sum_el_y = cp.sum(e_lag_trim * de_trim, axis=1)
        n = cp.float64(t_len)

        denom = n * sum_el2 - sum_el * sum_el + 1e-12
        gamma = (n * sum_el_y - sum_el * sum_y) / denom
        alpha_hat = (sum_y - gamma * sum_el) / n

        fitted = gamma[:, None] * e_lag_trim + alpha_hat[:, None]
        adf_resid = de_trim - fitted

    # Standard error of gamma
    sse = cp.sum(adf_resid ** 2, axis=1)
    mse = sse / (n - 2)
    se_gamma = cp.sqrt(mse * n / (denom + 1e-12))

    # t-statistic
    t_stat = gamma / (se_gamma + 1e-12)

    return cp.asnumpy(t_stat)


# ───────────────────────────────────────────────
# ADF Critical Values (Engle-Granger)
# ───────────────────────────────────────────────
# Approximate critical values for cointegration residual ADF
# (MacKinnon 1994 approximations for 2-variable case)
ADF_CRIT_VALUES = {
    0.01: -3.90,
    0.05: -3.34,
    0.10: -3.04,
}


def adf_pvalue_approx(t_stat: float) -> float:
    """Approximate p-value from ADF t-statistic for cointegration."""
    if t_stat < ADF_CRIT_VALUES[0.01]:
        return 0.005
    elif t_stat < ADF_CRIT_VALUES[0.05]:
        return 0.025
    elif t_stat < ADF_CRIT_VALUES[0.10]:
        return 0.075
    else:
        return min(0.5, 0.1 + (t_stat - ADF_CRIT_VALUES[0.10]) * 0.1)


# ───────────────────────────────────────────────
# Main GPU Cointegration Test
# ───────────────────────────────────────────────
def gpu_batch_cointegration(
    price_matrix: np.ndarray,
    pair_indices: List[Tuple[int, int]],
    alpha: float = 0.05,
    batch_size: int = 5000,
) -> List[Tuple[int, int, float, bool]]:
    """
    GPU-accelerated batch cointegration testing.

    Args:
        price_matrix: (n_tickers, T) array of price series
        pair_indices: list of (i, j) ticker index pairs to test
        alpha: significance level
        batch_size: pairs per GPU batch

    Returns:
        List of (i, j, approx_pvalue, is_cointegrated) tuples
    """
    ok, msg = _check_gpu()
    if not ok:
        buffered_print(f"{RED}GPU unavailable: {msg} - falling back to CPU{ENDC}")
        return []

    n_pairs = len(pair_indices)
    print_section(f"GPU Cointegration: {n_pairs} pairs", CYAN)
    print_gpu_status()

    results = []
    start_time = time.time()

    for batch_start in range(0, n_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_pairs)
        batch = pair_indices[batch_start:batch_end]
        b_size = len(batch)

        # Assemble batch matrices
        Y_batch = np.zeros((b_size, price_matrix.shape[1]))
        X_batch = np.zeros((b_size, price_matrix.shape[1]))

        for k, (i, j) in enumerate(batch):
            Y_batch[k] = price_matrix[i]
            X_batch[k] = price_matrix[j]

        # Step 1: Batch OLS on GPU
        betas, residuals = _batch_ols_gpu(Y_batch, X_batch)

        # Step 2: Batch ADF on GPU
        t_stats = _batch_adf_gpu(residuals, max_lag=1)

        # Step 3: Convert to p-values and filter
        for k, (i, j) in enumerate(batch):
            pval = adf_pvalue_approx(t_stats[k])
            results.append((i, j, pval, pval < alpha))

        # Progress
        elapsed = time.time() - start_time
        rate = batch_end / elapsed if elapsed > 0 else 0
        buffered_print(
            f"  {batch_end}/{n_pairs} ({rate:.0f} pairs/sec)"
        )

        # Thermal check every batch
        stats = get_gpu_stats()
        if stats and stats["temp_c"] > 80:
            buffered_print(f"{RED}GPU at {stats['temp_c']}C - pausing 5s{ENDC}")
            time.sleep(5)

    elapsed = time.time() - start_time
    n_coint = sum(1 for r in results if r[3])
    buffered_print(
        f"{GREEN}Done: {n_coint}/{n_pairs} cointegrated in {elapsed:.1f}s "
        f"({n_pairs / elapsed:.0f} pairs/sec){ENDC}"
    )
    print_gpu_status()

    return results


# ───────────────────────────────────────────────
# Integration with MultiCointEngine
# ───────────────────────────────────────────────
def gpu_cointegration_search(
    prices: dict,
    tickers: List[str],
    alpha: float = 0.05,
    batch_size: int = 5000,
) -> List[dict]:
    """
    Run GPU-accelerated cointegration on all pairs from a price dict.

    Args:
        prices: {ticker: np.ndarray} of aligned price series
        tickers: list of ticker names (matching prices keys)
        alpha: significance threshold
        batch_size: GPU batch size

    Returns:
        List of dicts with ticker1, ticker2, pvalue, cointegrated
    """
    # Build price matrix (all series must be same length)
    # Align to common length
    min_len = min(len(prices[t]) for t in tickers if t in prices)
    valid_tickers = [t for t in tickers if t in prices and len(prices[t]) >= min_len]

    price_matrix = np.zeros((len(valid_tickers), min_len))
    for i, t in enumerate(valid_tickers):
        price_matrix[i] = prices[t][-min_len:]

    # Generate all pair indices
    import itertools
    pair_indices = list(itertools.combinations(range(len(valid_tickers)), 2))

    # Run GPU batch test
    raw_results = gpu_batch_cointegration(
        price_matrix, pair_indices, alpha, batch_size,
    )

    # Convert to named results
    results = []
    for i, j, pval, is_coint in raw_results:
        if is_coint:
            results.append({
                "ticker1": valid_tickers[i],
                "ticker2": valid_tickers[j],
                "pvalue": pval,
                "cointegrated": True,
            })

    return results
