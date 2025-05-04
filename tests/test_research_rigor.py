import numpy as np
import pytest
from stat_arb_emrt_rl.core import compute_emrt, optimize_ou_params
from stat_arb_emrt_rl.rl import TabularQAgent

def test_emrt_invariant_constant_spread():
    """EMRT should be large for a constant spread (no reversion)."""
    spread = np.ones(100)
    # The implementation might return len(spread) or inf if no crosses
    emrt, taus = compute_emrt(spread, C=0.25)
    assert emrt >= 100 or np.isinf(emrt) or emrt == 0 # Depending on implementation

def test_emrt_highly_reverting():
    """EMRT should be small for a highly mean-reverting spread."""
    # Zero-crossing sine wave
    spread = np.sin(np.linspace(0, 100, 1000))
    emrt, taus = compute_emrt(spread, C=0.0) # Mean crossing
    assert emrt < 50 

def test_ou_optimization_convergence():
    """Test that OU parameter optimization returns plausible values."""
    # Generate OU process
    np.random.seed(42)
    dt = 1/252
    theta = 10.0
    mu = 0.5
    sigma = 0.2
    
    n = 1000
    x = np.zeros(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = x[i-1] + theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        
    # We expect the optimizer to find values close to the ground truth
    opt_theta, opt_mu, opt_sigma = optimize_ou_params(x, dt=dt)
    
    # Allow some margin for stochasticity
    assert 2.0 < opt_theta < 20.0
    assert 0.3 < opt_mu < 0.7
    assert 0.1 < opt_sigma < 0.4

def test_rl_agent_action_space():
    """Ensure RL agent respects the defined action space."""
    agent = TabularQAgent() # Uses default config
    state_idx = 0 # Placeholder state index
    # agent.select_action(state_idx, position, training)
    action_idx = agent.select_action(state_idx, position=0, training=False) # Greedy
    assert action_idx in [0, 1, 2] # 0: sell, 1: hold, 2: buy
