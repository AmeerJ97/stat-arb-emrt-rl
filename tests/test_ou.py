import unittest
import numpy as np
import pandas as pd
from stat_arb_emrt_rl.ou_optimizer import OUOptimizer, OUOptimizerConfig


class TestOUOptimizer(unittest.TestCase):
    def setUp(self):
        self.config = OUOptimizerConfig()
        self.optimizer = OUOptimizer(config=self.config)

    def test_known_beta(self):
        """Test optimization with synthetic data and known beta."""
        rng = np.random.default_rng(42)
        beta_true = 1.5
        x = rng.normal(size=1000)
        y = beta_true * x + rng.normal(scale=0.1, size=1000)
        df = pd.DataFrame({"Normalized T1": y, "Normalized T2": x})
        result = self.optimizer.optimize(df, "T1", "T2")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["beta"], beta_true, delta=0.1)

    def test_missing_required_columns(self):
        """Test graceful failure when required normalized columns are missing."""
        rng = np.random.default_rng(7)
        x = rng.normal(size=1000)
        y = 1.2 * x + rng.normal(scale=0.2, size=1000)
        df = pd.DataFrame({"Merged T1": y, "Merged T2": x})
        result = self.optimizer.optimize(df, "T1", "T2")
        self.assertIsNone(result)

    def test_optimize_typed_contract(self):
        rng = np.random.default_rng(99)
        x = rng.normal(size=500)
        y = 0.8 * x + rng.normal(scale=0.05, size=500)
        df = pd.DataFrame({"Normalized T1": y, "Normalized T2": x})

        result = self.optimizer.optimize_typed(df, "T1", "T2")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(hasattr(result, "beta"))
        self.assertTrue(hasattr(result, "mu"))
        self.assertTrue(hasattr(result, "theta"))
        self.assertTrue(hasattr(result, "sigma"))
        self.assertTrue(hasattr(result, "log_likelihood"))


if __name__ == "__main__":
    unittest.main()
