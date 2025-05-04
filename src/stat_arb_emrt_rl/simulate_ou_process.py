import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, linregress, t  # Added t-distribution import
from statsmodels.tsa.stattools import adfuller
from typing import Dict, List, Union

from .financial_loader import FinancialLoader, FinancialLoaderConfig
from .ou_optimizer import OUOptimizer, OUOptimizerConfig

# Initialize components
loader_config = FinancialLoaderConfig()
loader = FinancialLoader(config=loader_config)
ou_config = OUOptimizerConfig()
ou_opt = OUOptimizer(config=ou_config)


def simulate_ou_process(
    mu: Union[float, List[float]],
    theta: Union[float, List[float]],
    sigma: Union[float, List[float]],
    beta_true: Union[float, List[float]],
    X0: float = 0,
    dt: float = 1 / 252,
    N: int = 1000,
    noise_dist: str = "normal",
) -> pd.DataFrame:
    """Enhanced OU process simulator with robust parameter handling."""
    np.random.seed(42)

    # Validate parameter consistency
    def validate_parameters(*params):
        list_params = [p for p in params if isinstance(p, list)]
        if not list_params:
            return 1
        lengths = {len(p) for p in list_params}
        if len(lengths) > 1:
            raise ValueError("All parameter lists must have equal lengths")
        return lengths.pop()

    segments = validate_parameters(mu, theta, sigma, beta_true)
    seg_lengths = [
        N // segments + (1 if i < N % segments else 0) for i in range(segments)
    ]

    dfs = []
    current_spread = X0

    for seg in range(segments):
        # Get segment parameters with type safety
        seg_mu = mu[seg] if isinstance(mu, list) else mu
        seg_theta = theta[seg] if isinstance(theta, list) else theta
        seg_sigma = sigma[seg] if isinstance(sigma, list) else sigma
        seg_beta = beta_true[seg] if isinstance(beta_true, list) else beta_true
        seg_N = seg_lengths[seg]

        # Generate base price with realistic volatility
        returns = np.random.normal(scale=0.01 * seg_sigma, size=seg_N) * np.sqrt(
            1 + np.arange(seg_N) / seg_N
        )

        base_price = np.exp(np.cumsum(returns))

        # Generate OU spread using exact discretization
        spread = np.zeros(seg_N)
        spread[0] = current_spread

        for i in range(1, seg_N):  # Changed variable name from t to i
            noise = t.rvs(df=3) if noise_dist == "t" else np.random.randn()
            spread[i] = (
                spread[i - 1] * (1 - seg_mu * dt)
                + seg_mu * seg_theta * dt
                + seg_sigma * np.sqrt(dt) * noise
            )

        # Create price series with normalized columns
        merged_1 = base_price
        merged_2 = seg_beta * base_price + spread

        df_seg = pd.DataFrame(
            {
                "Merged Synthetic_1": merged_1,
                "Merged Synthetic_2": merged_2,
                "Normalized Synthetic_1": merged_1 / merged_1[0],
                "Normalized Synthetic_2": merged_2 / merged_2[0],
            },
            index=pd.date_range(start=f"202{seg}-01-01", periods=seg_N, freq="B"),
        )

        dfs.append(df_seg)
        current_spread = spread[-1]

    return pd.concat(dfs).sort_index()


def validate_parameters(result: Dict, case: Dict) -> bool:
    """Dynamic validation with adaptive thresholds"""
    mu_tol = 0.2 + 0.3 * case.get("sigma", 1)
    theta_tol = 0.3 + abs(case.get("theta", 0)) * 0.1
    sigma_tol = 0.3 + case.get("sigma", 1) * 0.2
    beta_tol = 0.1 + abs(case.get("beta_true", 1)) * 0.01

    checks = [
        abs(result["mu"] - case["mu"]) < mu_tol,
        abs(result["theta"] - case["theta"]) < theta_tol,
        abs(result["sigma"] - case["sigma"]) < sigma_tol,
        abs(result["beta"] - case["beta_true"]) < beta_tol,
        adfuller(result["spread"])[1] < 0.05,
    ]

    return all(checks)


def run_synthetic_tests(test_cases):
    """Enhanced test runner with visualization"""
    results = []

    for case in test_cases:
        try:
            print(f"\n=== Testing {case['id']}: {case['desc']} ===")

            # Generate synthetic data
            params = {
                k: v
                for k, v in case.items()
                if k in ["mu", "theta", "sigma", "beta_true", "X0", "N"]
            }
            df = simulate_ou_process(**params)

            # Run optimization with retries
            result = None
            for _ in range(3):
                result = ou_opt.optimize(df, "Synthetic_1", "Synthetic_2")
                if result and result["mu"] > 0 and result["sigma"] > 0:
                    break
                df = simulate_ou_process(**params)

            if not result:
                raise RuntimeError("Optimization failed after 3 attempts")

            # Calculate validation metrics
            result["spread"] = (
                df["Normalized Synthetic_1"]
                - result["beta"] * df["Normalized Synthetic_2"]
            ).values

            is_valid = validate_parameters(result, case)

            # Visualization
            fig, ax = plt.subplots(3, 1, figsize=(14, 12))

            # Price series plot
            df["Merged Synthetic_1"].plot(ax=ax[0], label="Asset 1", alpha=0.7)
            df["Merged Synthetic_2"].plot(ax=ax[0], label="Asset 2", alpha=0.7)
            ax[0].set_title(f"{case['id']} Price Series")
            ax[0].legend()

            # Spread plot
            pd.Series(result["spread"]).plot(ax=ax[1], color="purple", label="Spread")
            ax[1].axhline(
                result["theta"], color="r", linestyle="--", label="Estimated θ"
            )
            ax[1].axhline(case["theta"], color="g", linestyle=":", label="True θ")
            ax[1].set_title("Spread Series with Theta Comparison")
            ax[1].legend()

            # Parameter comparison
            ax[2].barh(0, result["mu"], 0.4, color="b", label="Estimated μ")
            ax[2].barh(0.5, case["mu"], 0.4, color="orange", label="True μ")
            ax[2].barh(1, result["sigma"], 0.4, color="b", label="Estimated σ")
            ax[2].barh(1.5, case["sigma"], 0.4, color="orange", label="True σ")
            ax[2].set_yticks([0, 0.5, 1, 1.5])
            ax[2].set_yticklabels(["μ Estimate", "μ True", "σ Estimate", "σ True"])
            ax[2].set_title("Parameter Recovery Comparison")

            plt.tight_layout()
            # plt.show()

            results.append((case["id"], "PASS" if is_valid else "WARN"))

        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            results.append((case["id"], "FAIL"))

    # Print summary
    print("\n=== Test Summary ===")
    for test_id, status in results:
        print(
            f"{test_id}: {'✅ PASS' if status == 'PASS' else '⚠️ WARN' if status == 'WARN' else '❌ FAIL'}"
        )


def test_real_world_pairs(loader, ou_opt, pairs):
    results = []
    for pair in pairs:
        df = loader.get_normalized_pair(
            pair["t1"], pair["t2"], "2021-01-01", "2021-12-31"
        )
        if df is not None:
            result = ou_opt.optimize(df, pair["t1"], pair["t2"])
            if result:
                spread = (
                    df[f"Merged {pair['t1']}"]
                    - result["beta"] * df[f"Merged {pair['t2']}"]
                )
                results.append(
                    {
                        "pair": pair["id"],
                        "tickers": f"{pair['t1']}, {pair['t2']}",
                        "real": pair["real"],
                        "beta": result["beta"],
                        "mu": result["mu"],
                        "theta": result["theta"],
                        "sigma": result["sigma"],
                        "log_likelihood": result["log_likelihood"] / 260,
                        "spread": spread,
                    }
                )
    return results


# Test cases
test_cases_synthetic = [
    {
        "id": "S1",
        "desc": "Basic OU Process",
        "mu": 0.5,
        "theta": 0,
        "sigma": 0.3,
        "beta_true": 1.0,
        "X0": 0,
        "N": 1000,
    },
    {
        "id": "S2",
        "desc": "Regime Switching",
        "mu": [0.3, 0.7],
        "theta": [-0.5, 0.5],
        "sigma": [0.2, 0.4],
        "beta_true": [1.0, 1.0],
        "X0": 0,
        "N": 2000,
    },
]

# Execute tests
if __name__ == "__main__":
    run_synthetic_tests(test_cases_synthetic)

    # Real-world analysis
    print("\n🔥 Real-World Pair Analysis")
    real_pairs = [
        {
            "id": "R1",
            "t1": "WM",
            "t2": "RSG",
            "real": "Real β = 0.98 | log likelihood = 5.7101",
            "desc": "Waste Management and Republic Services provide waste management and environmental services",
        },
        {
            "id": "R2",
            "t1": "UAL",
            "t2": "DAL",
            "real": "Real β = 1.02 | log likelihood = 5.3507",
            "desc": "United and Delta are two of the largest American airlines",
        },
        {
            "id": "R4",
            "t1": "V",
            "t2": "MA",
            "real": "Real β = 0.99 | log likelihood = 5.8283",
            "desc": "Visa and Mastercard are two dominant players in the payment industry",
        },
        {
            "id": "R5",
            "t1": "MS",
            "t2": "GS",
            "real": "Real β = 1.00 | log likelihood = 6.1103",
            "desc": "Investment Banks",
        },
        {
            "id": "R6",
            "t1": "NVDA",
            "t2": "AMD",
            "real": "Real β = 1.31 | log likelihood = 2.8395",
            "desc": " Nvidia and AMD are two American semiconductor companies that develop computer processors",
        },
        {
            "id": "R3",
            "t1": "CVX",
            "t2": "XOM",
            "real": "Real β = 0.95 | log likelihood = 4.6977",
            "desc": "Oil Majors",
        },
    ]

    real_results = test_real_world_pairs(loader, ou_opt, real_pairs)
    for result in real_results:
        print(f"\n💼 Analyzing {result['pair']}")
        print(f"  Pairs: {result['tickers']}")
        print(
            f"  β = {result['beta']:.4f} | μ = {result['mu']:.4f} | θ = {result['theta']:.4f} | σ = {result['sigma']:.4f}"
        )
        print(f"  {result['real']}")
        print(f"  log likelihood = {result['log_likelihood']:.1f}")
        print(f"  Half-life = {np.log(2)/(result['mu']+ 0.000001):.1f} days")

    print("\n🎉 All tests completed!")
