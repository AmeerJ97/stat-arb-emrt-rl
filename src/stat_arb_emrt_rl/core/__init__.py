"""Core EMRT and OU optimization primitives."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from ..emrt import compute_emrt, find_important_extremes, optimize_spread_coefficients
from ..ou_optimizer import OUOptimizationResult, OUOptimizer, OUOptimizerConfig


def optimize_ou_params(
    series: np.ndarray, dt: float = 1.0 / 252.0
) -> Tuple[float, float, float]:
    """
    Convenience wrapper for Ornstein-Uhlenbeck parameter estimation.
    
    Args:
        series: 1-D time series array
        dt: Time step (default 1/252 for daily)
        
    Returns:
        (theta, mu, sigma)
    """
    optimizer = OUOptimizer()
    # OUOptimizer expects a DataFrame with specific columns for its full API,
    # but we can simulate a single-series optimization by passing dummy data
    # or updating the test to use the class directly.
    # Actually, let's just make the test use the class directly to be more "rigorous".
    # But for a "wrapper", we'll provide a simpler version.
    
    # Internal estimation logic for a single series (MLE)
    n = len(series)
    if n < 2:
        return 0.0, np.mean(series), 0.0
        
    x = series[:-1]
    y = series[1:]
    
    # Simple linear regression: y = a*x + b + error
    # y = x + theta*(mu - x)*dt + sigma*sqrt(dt)*eps
    # y - x = (theta*mu*dt) - (theta*dt)*x + error
    
    dy = y - x
    params = np.polyfit(x, dy, 1) # [slope, intercept]
    
    # slope = -theta * dt
    # intercept = theta * mu * dt
    
    theta = -params[0] / dt
    mu = params[1] / (theta * dt + 1e-10)
    
    residuals = dy - (params[0] * x + params[1])
    sigma = np.std(residuals) / np.sqrt(dt)
    
    return theta, mu, sigma


__all__ = [
    "OUOptimizationResult",
    "OUOptimizer",
    "OUOptimizerConfig",
    "compute_emrt",
    "find_important_extremes",
    "optimize_spread_coefficients",
    "optimize_ou_params",
]
