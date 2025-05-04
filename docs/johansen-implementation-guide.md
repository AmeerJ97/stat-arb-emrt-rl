# Johansen Cointegration Testing for N-Asset Groups
## Complete Implementation Guide for statsmodels

This guide provides practical implementation details for using `statsmodels.tsa.vector_ar.vecm.coint_johansen` to test for cointegration among multiple assets (N-asset groups), with focus on interpretation and handling real-world challenges.

---

## 1. Exact API: `coint_johansen` Function Signature

### Function Location
```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen
```

### Complete Function Signature
```python
coint_johansen(
    endog,           # pd.DataFrame with shape (n_obs, n_assets). Each column is a time series.
    det_order=-1,    # Deterministic terms: -1 (none), 0 (constant), 1 (const + trend)
    k_ar_diff=1      # Lags in differences: k_ar_diff = p-1 where p is VAR order
)
```

### Parameters Explained

| Parameter | Type | Values | Meaning |
|-----------|------|--------|---------|
| **endog** | DataFrame | (n_obs, n_assets) | DataFrame with n time series as columns. Each row is one time point. Must have ≥20 observations. |
| **det_order** | int | -1, 0, 1 | **-1**: No deterministic terms (intercept). **0**: Constant term only. **1**: Constant + linear time trend. |
| **k_ar_diff** | int | ≥1 | Number of lags in the differenced VAR. For VAR(p), use k_ar_diff = p-1. Default 1 implies VAR(2). |

### Return Object (JohansenResults)

The function returns a results object with the following attributes:

```python
result = coint_johansen(df, det_order=0, k_ar_diff=1)

# Key attributes:
result.eig           # Eigenvalues (n_assets,) - sorted descending
result.evec          # Eigenvectors / Cointegrating vectors (n_assets, n_assets)
                     # Each column is a cointegrating vector (hedge ratios)

result.lr1           # Trace statistic (n_assets,) - test H0: r cointegrating relationships
result.cvt           # Trace critical values (n_assets, 3) - columns: 90%, 95%, 99%

result.lr2           # Max eigenvalue statistic (n_assets,) - test H0: exactly r relationships
result.cvm           # Max eigenvalue critical values (n_assets, 3) - columns: 90%, 95%, 99%

result.nobs          # Number of observations used
result.k_ar_diff     # Lag order used (k_ar_diff parameter)
result.det_order_diff  # Deterministic order in differenced system
```

---

## 2. Interpreting Trace and Eigenvalue Statistics

### Conceptual Overview

Both statistics test the **rank of cointegration**:
- **Rank 0**: No cointegrating relationships (all series independent)
- **Rank 1**: 1 cointegrating relationship (all series together form 1 stationary portfolio)
- **Rank r**: r linearly independent cointegrating relationships

### Trace Statistic Interpretation

**Test formulation:**
- H0: Number of cointegrating relationships ≤ r
- HA: Number of cointegrating relationships > r

**Decision rule:** If `trace_stat[r] > critical_value[r, 1]` (95% level), reject H0 → there are MORE than r relationships.

**Interpretation algorithm:**
1. Start from r=0
2. If `trace_stat[0] > critical_values[0, 1]`: At least 1 cointegrating relationship exists
3. If also `trace_stat[1] > critical_values[1, 1]`: At least 2 exist
4. Continue until you fail to reject (first r where stat ≤ critical value)
5. The number of cointegrating relationships = **last r where you rejected H0 + 1**

### Max Eigenvalue Interpretation

**Test formulation:**
- H0: Number of cointegrating relationships = r
- HA: Number of cointegrating relationships = r+1

**Decision rule:** If `max_eig_stat[r] > critical_value[r, 1]`, reject H0 → there are MORE than r relationships.

**Interpretation:** Directly reads the rank - first index where you fail to reject H0 is your rank.

### Practical Example: 3 Assets

```python
# Example output from coint_johansen:
# eig:              [0.45, 0.22, 0.05]       (eigenvalues)
# trace_stat:       [62.3, 23.8, 4.2]        (trace statistics)
# trace_crit_vals:  [[13.4, 15.4, 20.0],     (90%, 95%, 99%)
#                    [7.3, 9.2, 12.9],
#                    [0.5, 2.1, 4.8]]

# Trace test at 95% (index 1):
# r=0: trace_stat[0]=62.3 > critical[0,1]=15.4 → REJECT H0
#      → At least 1 cointegrating relationship exists
# r=1: trace_stat[1]=23.8 > critical[1,1]=9.2  → REJECT H0
#      → At least 2 cointegrating relationships exist
# r=2: trace_stat[2]=4.2  < critical[2,1]=2.1  → FAIL TO REJECT H0
#      → Not significant, stop

# Conclusion: Rank = 2 (two cointegrating relationships among 3 assets)
```

### Code Implementation: Determining Rank

```python
def determine_cointegration_rank(johansen_result, significance_level=0.05):
    """
    Determine the number of cointegrating relationships using trace statistic.

    Args:
        johansen_result: Output from coint_johansen()
        significance_level: 0.10 (90%), 0.05 (95%), 0.01 (99%)

    Returns:
        int: Number of cointegrating relationships
    """
    trace_stats = johansen_result.lr1
    crit_vals = johansen_result.cvt

    # Map significance level to column index
    level_map = {0.10: 0, 0.05: 1, 0.01: 2}
    crit_col = level_map.get(significance_level, 1)  # Default to 95%

    # Count how many trace statistics exceed critical value
    # (Trace test is cumulative)
    rank = 0
    for r in range(len(trace_stats)):
        if trace_stats[r] > crit_vals[r, crit_col]:
            rank = r + 1  # Update rank (cumulative rejection)
        else:
            break  # First failure to reject stops the test

    return rank
```

---

## 3. Extracting Cointegrating Vectors (Hedge Ratios)

### What Are Cointegrating Vectors?

Cointegrating vectors are eigenvectors from the Johansen test. Each vector represents:
- **Hedge ratios / Portfolio weights**: How to combine N assets into a stationary spread
- **Columns of evec matrix**: Each column is one cointegrating vector
- **Interpretation**: If vector is [1.0, -0.5, 0.3], the spread is `1.0*P1 - 0.5*P2 + 0.3*P3`

### Extraction Logic

```python
def extract_cointegrating_vectors(johansen_result, rank):
    """
    Extract the cointegrating vectors (eigenvectors) corresponding to
    the number of cointegrating relationships (rank).

    Args:
        johansen_result: Output from coint_johansen()
        rank: Number of cointegrating relationships (from determine_rank)

    Returns:
        ndarray: Shape (n_assets, rank) - each column is a hedging ratio vector
    """
    # evec has shape (n_assets, n_assets)
    # We want the first 'rank' columns (correspond to largest eigenvalues)
    evec = johansen_result.evec
    cointegrating_vectors = evec[:, :rank]

    return cointegrating_vectors
```

### Practical Example: Using Hedge Ratios to Form Spreads

```python
import numpy as np
import pandas as pd

# Suppose we have 3 assets and rank=2 (2 cointegrating relationships)
prices = pd.DataFrame({
    'BTC': [60000, 61000, 59500, ...],
    'ETH': [3000, 3050, 2980, ...],
    'SOL': [150, 155, 148, ...]
})

result = coint_johansen(prices, det_order=0, k_ar_diff=1)
rank = determine_cointegration_rank(result, significance_level=0.05)
cointegrating_vecs = extract_cointegrating_vectors(result, rank)

# cointegrating_vecs shape: (3, 2) - 3 assets, 2 relationship vectors

# Form the first spread (first cointegrating relationship)
hedge_ratio_1 = cointegrating_vecs[:, 0]
spread_1 = (hedge_ratio_1[0] * prices['BTC'] +
            hedge_ratio_1[1] * prices['ETH'] +
            hedge_ratio_1[2] * prices['SOL'])

# Spread_1 is stationary and can be mean-reverted for trading signals
mean_spread = spread_1.mean()
std_spread = spread_1.std()
z_score = (spread_1 - mean_spread) / std_spread
```

### Normalization of Hedge Ratios

By default, eigenvectors are not normalized. To make them interpretable (sum to 1 or first element = 1):

```python
def normalize_hedge_ratios(cointegrating_vectors, method='sum'):
    """
    Normalize cointegrating vectors for easier interpretation.

    Args:
        cointegrating_vectors: (n_assets, rank) array
        method: 'sum' (sum to 1), 'first' (first element = 1), or 'norm' (L2 norm = 1)

    Returns:
        ndarray: Normalized vectors
    """
    normalized = cointegrating_vectors.copy().astype(float)

    if method == 'sum':
        # Normalize so each vector sums to 1
        for i in range(normalized.shape[1]):
            vec_sum = normalized[:, i].sum()
            if vec_sum != 0:
                normalized[:, i] /= vec_sum

    elif method == 'first':
        # Normalize so first element = 1
        for i in range(normalized.shape[1]):
            if normalized[0, i] != 0:
                normalized[:, i] /= normalized[0, i]

    elif method == 'norm':
        # L2 norm = 1
        for i in range(normalized.shape[1]):
            norm = np.linalg.norm(normalized[:, i])
            if norm != 0:
                normalized[:, i] /= norm

    return normalized
```

---

## 4. Selecting the Lag Order (k_ar_diff)

### What is k_ar_diff?

- **k_ar_diff = p - 1** where p is the VAR(p) model order
- Controls how much history the model uses
- Higher lag order → more flexible but fewer observations
- Lower lag order → more observations but may miss dynamics

### Selection Methods

#### Method 1: Information Criteria (Recommended)

```python
from statsmodels.tsa.api import VAR

def select_lag_order_by_ic(prices_df, maxlag=12):
    """
    Select lag order using Akaike and Bayesian information criteria.

    Args:
        prices_df: DataFrame with prices as columns
        maxlag: Maximum lag order to test

    Returns:
        dict: Recommended lag orders from different criteria
    """
    model = VAR(prices_df)
    lag_results = model.select_lags(maxlag=maxlag)

    # lag_results gives p (VAR order), we need k_ar_diff = p - 1
    recommended_p = {
        'aic': lag_results.aic,
        'bic': lag_results.bic,
        'hq': lag_results.hq,
        'fpe': lag_results.fpe
    }

    recommended_k_ar_diff = {k: v-1 for k, v in recommended_p.items()}

    return recommended_k_ar_diff

# Usage:
lag_orders = select_lag_order_by_ic(prices, maxlag=12)
# lag_orders might be: {'aic': 2, 'bic': 1, 'hq': 2, 'fpe': 2}
# Common choice: use BIC (more conservative, fewer lags)
k_ar_diff = lag_orders['bic']
```

#### Method 2: ACF/PACF Visual Inspection

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Check autocorrelation of price differences
price_diff = prices.diff().dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(price_diff.iloc[:, 0], lags=20, ax=ax1)
plot_pacf(price_diff.iloc[:, 0], lags=20, ax=ax2)
plt.show()

# PACF cuts off after lag k → use k_ar_diff = k
```

#### Method 3: Conservative/Domain Knowledge

```python
# Common practical choices:
# - Daily data: k_ar_diff = 1 (VAR(2)) - default, works well for most cases
# - Daily data with strong autocorrelation: k_ar_diff = 2-5
# - High-frequency (minute/hourly): k_ar_diff = 1-3
# - Weekly/Monthly: k_ar_diff = 1-2

# For most equity/crypto pairs:
k_ar_diff = 1  # Conservative, preserves observations
```

### Trade-offs

| k_ar_diff | Observations Available | Model Flexibility | Recommendation |
|-----------|------------------------|-------------------|-----------------|
| 1         | Maximum                | Moderate          | **Start here** |
| 2-3       | Reduced                | Higher            | If ACF suggests lag dependency |
| 4+        | Significantly reduced  | Very high         | Only with 1000+ observations |

---

## 5. Handling Mixed-Scale Assets (e.g., BTC at $60k vs Stock at $150)

### The Problem

Raw prices at vastly different scales cause numerical instability:
- Regression coefficients become extreme
- Eigenvector interpretation becomes difficult
- Cointegration test may fail due to numerical issues

### Solution 1: Price Normalization (Most Recommended)

Normalize each price series to start at 1.0:

```python
def normalize_prices(prices_df):
    """
    Normalize prices so each series starts at 1.0.
    Preserves percentage changes (returns) - the relevant relationship.

    Args:
        prices_df: DataFrame with price columns

    Returns:
        DataFrame: Normalized prices
    """
    return prices_df / prices_df.iloc[0]

# Usage:
prices_raw = pd.DataFrame({
    'BTC': [60000, 61000, 59500, ...],
    'ETH': [3000, 3050, 2980, ...],
    'SOL': [150, 155, 148, ...]
})

prices_normalized = normalize_prices(prices_raw)
# BTC starts at 1.0, ETH starts at 1.0, SOL starts at 1.0
# Now scales are comparable

result = coint_johansen(prices_normalized, det_order=0, k_ar_diff=1)
```

### Solution 2: Log-Returns (Alternative)

Use log-returns instead of prices (works for cointegration of returns):

```python
def log_returns(prices_df):
    """
    Calculate log returns: log(P_t / P_{t-1})
    """
    return np.log(prices_df / prices_df.shift(1)).dropna()

log_ret = log_returns(prices_raw)
# Now all series are on return scale (percentage changes)

result = coint_johansen(log_ret, det_order=0, k_ar_diff=1)
```

### Solution 3: Standardization (Z-Scores)

Standardize to mean 0, std dev 1:

```python
def standardize_prices(prices_df):
    """
    Standardize each price series to mean 0, std dev 1.
    Useful for visual comparison.
    """
    return (prices_df - prices_df.mean()) / prices_df.std()

prices_std = standardize_prices(prices_raw)
result = coint_johansen(prices_std, det_order=0, k_ar_diff=1)
```

### Recommended Approach

**For trading strategy (cointegration of prices):**
1. Use **normalized prices** (divide by first price)
2. Test with k_ar_diff=1, det_order=0
3. Extract hedge ratios from normalized result
4. Apply hedge ratios to RAW prices for actual trading

**Code:**
```python
prices_raw = pd.DataFrame({'BTC': [...], 'ETH': [...], 'SOL': [...]})

# Test on normalized
prices_norm = prices_raw / prices_raw.iloc[0]
result = coint_johansen(prices_norm, det_order=0, k_ar_diff=1)
rank = determine_cointegration_rank(result, significance_level=0.05)
hedge_ratios = extract_cointegrating_vectors(result, rank)

# Apply to raw prices for trading
hedge_ratios_normalized = normalize_hedge_ratios(hedge_ratios, method='first')
spread = (hedge_ratios_normalized[:, 0] * prices_raw.values).sum(axis=1)
# 'spread' is the actual dollar spread to trade
```

---

## 6. Typical Significance Thresholds

### Standard Thresholds

| Significance Level | Column in Critical Values | Use Case |
|-------------------|---------------------------|----------|
| 90% | 0 | Exploratory/screening |
| **95%** | **1** | **Standard, recommended** |
| 99% | 2 | Very conservative (fewer false positives) |

### Johansen-Specific Guidance

```python
# Standard thresholds for different scenarios:

# Strict test (99% confidence) - few false positives but may miss real relationships
significance_level = 0.01
crit_col = 2

# Balanced test (95% confidence) - RECOMMENDED
significance_level = 0.05
crit_col = 1

# Exploratory (90%) - find candidates for further analysis
significance_level = 0.10
crit_col = 0
```

### Interpretation Thresholds for Market Use

```python
def assess_cointegration_strength(trace_stat, crit_values):
    """
    Assess strength of cointegration relationship.

    Returns:
        str: 'strong', 'moderate', 'weak', 'not significant'
    """
    stat = trace_stat
    strong_95 = crit_values[2]   # 99%
    medium_95 = crit_values[1]   # 95%
    weak_95 = crit_values[0]     # 90%

    if stat > strong_95:
        return 'strong'
    elif stat > medium_95:
        return 'moderate'
    elif stat > weak_95:
        return 'weak'
    else:
        return 'not_significant'

# Usage:
for r in range(len(result.lr1)):
    strength = assess_cointegration_strength(
        result.lr1[r],
        result.cvt[r]
    )
    print(f"Relationship {r+1}: {strength}")
```

---

## Complete Working Example: 3-Asset Portfolio

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Load or create price data for 3 assets
prices = pd.DataFrame({
    'BTC': [60000, 61000, 59500, 62000, ...],  # ~500+ observations
    'ETH': [3000, 3050, 2980, 3100, ...],
    'SOL': [150, 155, 148, 160, ...]
})

# Step 1: Normalize prices (different scales)
prices_norm = prices / prices.iloc[0]

# Step 2: Select lag order
from statsmodels.tsa.api import VAR
model = VAR(prices_norm)
lag_results = model.select_lags(maxlag=12)
k_ar_diff = lag_results.bic - 1  # BIC criterion, convert to k_ar_diff

# Step 3: Run Johansen test
result = coint_johansen(prices_norm, det_order=0, k_ar_diff=k_ar_diff)

# Step 4: Determine rank
rank = determine_cointegration_rank(result, significance_level=0.05)
print(f"Number of cointegrating relationships: {rank}")

# Step 5: Extract and normalize hedge ratios
cointegrating_vecs = extract_cointegrating_vectors(result, rank)
hedge_ratios = normalize_hedge_ratios(cointegrating_vecs, method='sum')

# Step 6: Form spreads and test for mean reversion
for i in range(rank):
    weights = hedge_ratios[:, i]
    spread = (weights * prices.values).sum(axis=1)
    mean_spread = spread.mean()
    std_spread = spread.std()

    print(f"\nSpread {i+1}:")
    print(f"  Weights: BTC={weights[0]:.3f}, ETH={weights[1]:.3f}, SOL={weights[2]:.3f}")
    print(f"  Half-life (mean reversion): {estimate_spread_half_life(spread):.1f} days")
    print(f"  Current zscore: {(spread.iloc[-1] - mean_spread) / std_spread:.2f}")
```

---

## Summary Table: All 6 Questions

| Question | Answer |
|----------|--------|
| **API** | `coint_johansen(df, det_order, k_ar_diff)` returns eigenvalues, eigenvectors, trace/max stats, critical values |
| **Interpretation** | Count cumulative rejections of H0 using trace test; rank = first r that fails to reject |
| **Hedge Ratios** | First `rank` columns of `result.evec`; normalize for interpretability |
| **Lag Order** | Use VAR lag selection (BIC), ACF/PACF, or default k_ar_diff=1 |
| **Mixed Scales** | Normalize prices by first value to preserve percentage changes and comparability |
| **Thresholds** | Use 95% (crit_col=1) as standard; 99% for strict, 90% for exploratory |

---

## References & Further Reading

- **statsmodels documentation**: https://www.statsmodels.org/stable/vector_ar.html#cointegration
- **Johansen (1988, 1991)** original papers on trace and max eigenvalue tests
- **Engle & Granger (1987)** on cointegration concepts (pairwise case)
- **Harris & Sollis (2003)** "Applied Time Series Modelling and Forecasting"
