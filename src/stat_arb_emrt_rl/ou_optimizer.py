# ───────────────────────────────────────────────
# Standard Library Imports
# ───────────────────────────────────────────────
import logging
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

# ───────────────────────────────────────────────
# Third-Party Package Imports
# ───────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import linregress

# ───────────────────────────────────────────────
# Project-Specific Imports
# ───────────────────────────────────────────────
from .printing_system import (
    buffered_print,
    print_header,
    print_status,
    print_boxed,
    print_section,
    GREEN,
    RED,
    YELLOW,
    CYAN,
    BLUE,
    ENDC,
)


class OUOptimizerConfig:
    """Configuration for OU Optimizer parameters"""

    def __init__(self):
        self.beta_bounds = (-6.0, 6.0)
        self.epsilon = 1e-8  # Numerical stability
        self.max_iter = 5000
        self.mu_min = 0
        self.optimization_method = "bounded"


@dataclass(frozen=True)
class OUOptimizationResult:
    beta: float
    mu: float
    theta: float
    sigma: float
    log_likelihood: float


class OUOptimizer:
    """Optimizes OU parameters with enhanced data utilization and efficiency"""

    def __init__(self, config: Optional[OUOptimizerConfig] = None):
        self.config = config or OUOptimizerConfig()
        self._validate_config()
        print_header(message="OU Optimizer Initialized")
        time.sleep(0.1)

    def _validate_config(self):
        if self.config.beta_bounds[0] >= self.config.beta_bounds[1]:
            raise ValueError("Invalid beta bounds: lower bound must be < upper bound")

    # def _print_status(self, message: str, level: str = "INFO"):
    #     """Standardized logging similar to FinancialLoader"""
    #     log_method = getattr(logger, level.lower(), logger.info)
    #     log_method(message)

    def _initial_beta(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate initial beta using scaled spread with numerical safeguards"""
        # Add epsilon to avoid division by zero
        x_std = np.std(x) + self.config.epsilon
        y_std = np.std(y) + self.config.epsilon
        scale_factor = y_std / x_std
        scaled_x = x * scale_factor

        # Use linear regression with check for valid slope
        if len(scaled_x) < 2 or np.all(scaled_x == scaled_x[0]):
            return np.clip(0.0, *self.config.beta_bounds)  # Fallback to zero

        slope, _, _, _, _ = linregress(scaled_x, y)
        return np.clip(slope / scale_factor, *self.config.beta_bounds)

    def _log_likelihood(
        self, beta: float, x: np.ndarray, y: np.ndarray, d_t: float
    ) -> float:
        """Compute OU log-likelihood for combined Open/Close spread"""
        spread = y - beta * x
        n = len(spread)
        if n < 2:
            print_status("Insufficient data points for OU calculation", "WARNING")
            return -np.inf

        Xt = np.asarray(spread[:-1], dtype=np.float64)
        Xt1 = np.asarray(spread[1:], dtype=np.float64)

        # Efficient parameter estimation using matrix operations
        sum_Xt = Xt.sum()
        sum_Xt1 = Xt1.sum()
        sum_Xt_sq = (Xt**2).sum()
        sum_Xt_Xt1 = (Xt * Xt1).sum()

        theta = (sum_Xt1 * sum_Xt_sq - sum_Xt * sum_Xt_Xt1) / (
            n * sum_Xt_sq - sum_Xt**2 + self.config.epsilon
        )

        mu_num = sum_Xt_Xt1 - theta * sum_Xt - theta * sum_Xt1 + n * theta**2
        mu_den = sum_Xt_sq - 2 * theta * sum_Xt + n * theta**2
        # mu = -np.log(
        #     max(mu_num / (mu_den + self.config.epsilon), self.config.epsilon) / d_t
        # )
        raw_mu = mu_num / (mu_den + self.config.epsilon)
        # mu = -np.log(1 + np.exp(raw_mu))  # Softplus to enforce μ > 0
        mu = np.log1p(np.exp(raw_mu))  # Softplus to enforce μ > 0
        # mu = -np.log(np.exp(raw_mu))  # Softplus to enforce μ > 0

        # Apply softplus transformation to ensure μ > 0
        # mu = np.log(1 + np.exp(raw_mu))  # Constrained μ

        sigma_sq = (1 / (n * d_t)) * (
            (Xt1 - Xt * np.exp(-mu * d_t) - theta * (1 - np.exp(-mu * d_t))) ** 2
        ).sum()
        sigma = np.sqrt(sigma_sq + self.config.epsilon)

        # Log-likelihood calculation using vectorized operations
        ll = (
            -n / 2 * np.log(2 * np.pi * sigma_sq)
            - (1 / (2 * sigma_sq))
            * (
                (Xt1 - Xt * np.exp(-mu * d_t) - theta * (1 - np.exp(-mu * d_t))) ** 2
            ).sum()
        )

        # self._log(f"μ: {mu:.6f}, θ: {theta:.6f}, σ: {sigma:.6f}, β: {beta:.6f}", "DEBUG")

        return ll

    def optimize_typed(
        self, df: pd.DataFrame, t1: str, t2: str, d_t: float = 1.0
    ) -> Optional[OUOptimizationResult]:
        """Typed OU optimization result for contract-safe callers."""
        try:
            if not all(
                col in df.columns for col in [f"Normalized {t1}", f"Normalized {t2}"]
            ):
                missing = {f"Normalized {t1}", f"Normalized {t2}"} - set(df.columns)
                raise ValueError(f"Missing columns: {missing}")

            # Initial beta estimation
            y = df[f"Normalized {t1}"].values.astype(np.float64)
            x = df[f"Normalized {t2}"].values.astype(np.float64)

            # Add validation
            if len(y) < 30 or len(x) < 30:
                print_status("Insufficient data points", "ERROR")
                return None

            # Force numpy arrays
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)

            # Initial beta estimation
            beta_init = self._initial_beta(x, y)
            # print_status(f"Initial beta estimate: {beta_init:.4f}")

            # Optimization
            result = minimize_scalar(
                fun=lambda b: -self._log_likelihood(b, x, y, d_t),
                bounds=self.config.beta_bounds,
                method=self.config.optimization_method,
                options={"maxiter": self.config.max_iter},
            )

            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")

            beta_opt = result.x
            final_ll = -result.fun

            # Parameter estimation
            spread = y - beta_opt * x

            theta_est = np.nanmean(spread)

            X = spread[:-1].reshape(-1, 1)
            Y = np.diff(spread) + self.config.epsilon

            # Robust μ estimation using linear regression
            try:
                coefficient = np.linalg.lstsq(X * d_t, Y, rcond=None)[0][0]
                mu_est = -coefficient  # Added negative sign
                mu_est = max(mu_est, self.config.mu_min)
            except np.linalg.LinAlgError:
                mu_est = self.config.mu_min

            sigma_est = np.sqrt(np.nanmean(np.square(Y - mu_est * X.flatten() * d_t)))

            return OUOptimizationResult(
                beta=float(beta_opt),
                mu=float(mu_est),
                theta=float(theta_est),
                sigma=float(sigma_est),
                log_likelihood=float(final_ll),
            )

        except Exception as e:
            print_status(f"Optimization error: {str(e)}", "ERROR")
            return None

    def optimize(
        self, df: pd.DataFrame, t1: str, t2: str, d_t: float = 1.0
    ) -> Optional[Dict]:
        """
        Backward-compatible dict-returning API.
        New code should prefer `optimize_typed`.
        """
        typed = self.optimize_typed(df, t1, t2, d_t=d_t)
        if typed is None:
            return None
        return asdict(typed)
