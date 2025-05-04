# ───────────────────────────────────────────────
# Reinforcement Learning Agent for Statistical Arbitrage
# Implements Section 4 of Ning & Lee (2024)
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
    BLUE,
    ENDC,
)


# ───────────────────────────────────────────────
# Constants from Paper (Section 4.2 & 5.2)
# ───────────────────────────────────────────────
DEFAULT_LOOKBACK = 4        # l = 4 (state window)
DEFAULT_THRESHOLD = 3.0     # k = 3% (price change threshold)
N_D_VALUES = 4              # {-2, -1, +1, +2}
ACTIONS = {0: -1, 1: 0, 2: 1}  # index -> action value
ACTION_LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}
N_ACTIONS = 3

# Mapping d-values to indices for Q-table
D_TO_INDEX = {-2: 0, -1: 1, 1: 2, 2: 3}
INDEX_TO_D = {0: -2, 1: -1, 2: 1, 3: 2}


# ───────────────────────────────────────────────
# State Space Construction (Section 4.2)
# ───────────────────────────────────────────────
def compute_d(
    price_prev: float,
    price_curr: float,
    k: float = DEFAULT_THRESHOLD,
    scale: float = 0.0,
) -> int:
    """
    Discretize a single price change into {-2, -1, +1, +2}.

    When scale > 0, uses absolute change / scale for threshold comparison.
    This makes the encoding scale-invariant (works for spreads near zero).

    When scale == 0, uses percentage change (paper's original formulation):
      pi_i = ((P_i - P_{i-1}) / P_{i-1}) * 100
      d_i = +2 if pi_i > k
      d_i = +1 if 0 < pi_i <= k
      d_i = -1 if -k <= pi_i < 0
      d_i = -2 if pi_i < -k
    """
    if scale > 1e-10:
        # Scale-invariant mode: normalize change by spread std
        # A change of 1 std maps to the threshold boundary
        change = (price_curr - price_prev) / scale
        # k=3 means "3% of std is the big-move threshold"
        # Calibrated so ~25% of moves fall in each bin for OU process
        threshold = 0.15  # approx 1-step std fraction for daily OU
    else:
        # Percentage mode (paper's original)
        if abs(price_prev) < 1e-10:
            return 1
        change = (price_curr - price_prev) / price_prev * 100.0
        threshold = k

    if change > threshold:
        return 2
    elif change > 0:
        return 1
    elif change > -threshold:
        return -1
    else:
        return -2


def state_to_index(state_vec: np.ndarray) -> int:
    """
    Convert length-l vector of d-values to flat Q-table index.
    Uses base-4 encoding: index = sum(d_index[i] * 4^i)
    Total states = 4^l (256 for l=4)
    """
    idx = 0
    for i, d in enumerate(state_vec):
        idx += D_TO_INDEX[int(d)] * (N_D_VALUES ** i)
    return idx


def index_to_state(index: int, l: int = DEFAULT_LOOKBACK) -> np.ndarray:
    """Reverse mapping from Q-table index back to d-values."""
    state = np.zeros(l, dtype=int)
    for i in range(l):
        d_idx = index % N_D_VALUES
        state[i] = INDEX_TO_D[d_idx]
        index //= N_D_VALUES
    return state


def extract_state(
    spread_prices: np.ndarray,
    t: int,
    l: int = DEFAULT_LOOKBACK,
    k: float = DEFAULT_THRESHOLD,
    scale: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Extract state vector at time t from spread price series.

    Args:
        spread_prices: array of spread prices
        t: current time index (must be >= l)
        l: lookback window length
        k: threshold for price change discretization
        scale: if > 0, use scale-invariant mode (spread_std)

    Returns:
        (state_vector, state_index) tuple
    """
    if t < l:
        raise ValueError(f"t={t} must be >= lookback l={l}")

    window = spread_prices[t - l: t + 1]  # l+1 prices -> l returns
    d_vec = np.array([
        compute_d(window[i], window[i + 1], k, scale)
        for i in range(l)
    ])
    return d_vec, state_to_index(d_vec)


# ───────────────────────────────────────────────
# Action Constraints (Section 4.2)
# ───────────────────────────────────────────────
def get_valid_actions(position: int) -> List[int]:
    """
    Position-dependent action constraints from the paper.

    position = 0 (flat):  can HOLD (1) or BUY (2)
    position = 1 (long):  can SELL (0) or HOLD (1)

    The paper does not allow short-selling from flat.
    Returns list of valid action INDICES.
    """
    if position == 0:
        return [1, 2]   # hold or buy
    elif position == 1:
        return [0, 1]   # sell or hold
    else:
        raise ValueError(f"Invalid position: {position}. Must be 0 or 1.")


# ───────────────────────────────────────────────
# Reward Function (Equation 5)
# ───────────────────────────────────────────────
def compute_reward(
    action: int,
    spread_xt: float,
    theta: float,
    c: float = 0.001,
) -> float:
    """
    R_{t+1} = A_t * (theta - X_t) - c * |A_t|

    Intuition:
    - Buy (+1) when X_t < theta: positive reward (buy low, expect reversion up)
    - Sell (-1) when X_t > theta: positive reward (sell high, expect reversion down)
    - Transaction cost c penalizes every non-hold action
    """
    return action * (theta - spread_xt) - c * abs(action)


# ───────────────────────────────────────────────
# Cumulative Return (Equation 6)
# ───────────────────────────────────────────────
def compute_cumulative_return(
    rewards: np.ndarray,
    spread_xt: float,
    position_it: int,
    r: float = 0.05,
    dt: float = 1.0 / 252,
) -> float:
    """
    G_t = sum_{s=t+1}^{T} e^{-r(s-t)*dt} * R_s + I_T * X_T

    The terminal term I_T * X_T accounts for mark-to-market of any
    open position at the end of the trading period.
    """
    T = len(rewards)
    if T == 0:
        return position_it * spread_xt

    discount_factors = np.exp(-r * np.arange(1, T + 1) * dt)
    return float(np.dot(discount_factors, rewards) + position_it * spread_xt)


# ───────────────────────────────────────────────
# Tabular Q-Learning Agent (Equations 3 & 4)
# ───────────────────────────────────────────────
class RLAgentConfig:
    """Configuration for RL agent hyperparameters."""

    def __init__(self):
        # State space
        self.lookback = DEFAULT_LOOKBACK
        self.threshold_k = DEFAULT_THRESHOLD

        # Q-learning hyperparameters (from paper Section 5.2)
        self.alpha = 0.1          # learning rate
        self.gamma = 0.99         # discount factor
        self.epsilon = 0.1        # exploration rate (training)

        # Training configuration
        self.n_training_paths = 10_000   # number of OU paths
        self.n_epochs = 10               # training epochs
        self.transaction_cost = 0.001    # c in reward function

        # OU simulation parameters (paper defaults)
        self.ou_mu = 1.0
        self.ou_theta = 1.0
        self.ou_sigma = 0.1
        self.ou_T = 252          # trading days per year
        self.interest_rate = 0.05


class TabularQAgent:
    """
    Tabular Q-learning agent for mean reversion trading.
    Implements the MDP framework from Section 4.1-4.2 of the paper.

    Q-table shape: [n_states, n_actions] = [4^l, 3]
    """

    def __init__(self, config: Optional[RLAgentConfig] = None):
        self.config = config or RLAgentConfig()
        self.n_states = N_D_VALUES ** self.config.lookback
        self.n_actions = N_ACTIONS

        # Initialize Q-table to zeros
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

        # Training metrics
        self.episode_rewards: List[float] = []
        self.td_errors: List[float] = []
        self.training_complete = False

    def select_action(
        self,
        state_idx: int,
        position: int,
        training: bool = True,
    ) -> int:
        """
        Epsilon-greedy action selection with position constraints.

        During training: random valid action with prob epsilon.
        During testing: epsilon=0, pure greedy.

        Returns action INDEX (0=sell, 1=hold, 2=buy).
        """
        valid = get_valid_actions(position)

        if training and np.random.random() < self.config.epsilon:
            return int(np.random.choice(valid))

        # Greedy: pick valid action with highest Q-value
        q_valid = [(a, self.Q[state_idx, a]) for a in valid]
        best_action = max(q_valid, key=lambda x: x[1])[0]
        return best_action

    def update(
        self,
        state_idx: int,
        action_idx: int,
        reward: float,
        next_state_idx: int,
        next_position: int,
        done: bool = False,
    ) -> float:
        """
        Q-learning update (Equation 4):
        Q_new(S_t, A_t) <- Q(S_t, A_t) + alpha * (
            R_{t+1} + gamma * max_a Q(S_{t+1}, a) - Q(S_t, A_t)
        )

        The max is taken over VALID actions in the next state.
        """
        if done:
            td_target = reward
        else:
            next_valid = get_valid_actions(next_position)
            max_next_q = max(self.Q[next_state_idx, a] for a in next_valid)
            td_target = reward + self.config.gamma * max_next_q

        td_error = td_target - self.Q[state_idx, action_idx]
        self.Q[state_idx, action_idx] += self.config.alpha * td_error

        return abs(td_error)

    def freeze(self):
        """Switch to pure exploitation (testing mode per paper)."""
        self.config.epsilon = 0.0
        self.training_complete = True

    def get_q_stats(self) -> Dict:
        """Get summary statistics of the Q-table."""
        nonzero_mask = self.Q != 0
        return {
            "total_states": self.n_states,
            "visited_states": int(np.any(nonzero_mask, axis=1).sum()),
            "q_mean": float(np.mean(self.Q[nonzero_mask])) if nonzero_mask.any() else 0,
            "q_std": float(np.std(self.Q[nonzero_mask])) if nonzero_mask.any() else 0,
            "q_max": float(np.max(self.Q)),
            "q_min": float(np.min(self.Q[nonzero_mask])) if nonzero_mask.any() else 0,
        }

    def save(self, filepath: str):
        """Save Q-table and config to file."""
        np.savez(
            filepath,
            Q=self.Q,
            lookback=self.config.lookback,
            threshold_k=self.config.threshold_k,
            alpha=self.config.alpha,
            gamma=self.config.gamma,
            epsilon=self.config.epsilon,
        )

    @classmethod
    def load(cls, filepath: str) -> "TabularQAgent":
        """Load Q-table and config from file."""
        data = np.load(filepath)
        config = RLAgentConfig()
        config.lookback = int(data["lookback"])
        config.threshold_k = float(data["threshold_k"])
        config.alpha = float(data["alpha"])
        config.gamma = float(data["gamma"])
        config.epsilon = float(data["epsilon"])

        agent = cls(config=config)
        agent.Q = data["Q"]
        return agent


# ───────────────────────────────────────────────
# OU Path Simulation for Training
# ───────────────────────────────────────────────
def simulate_ou_spread(
    mu: float = 1.0,
    theta: float = 1.0,
    sigma: float = 0.1,
    n_steps: int = 252,
    x0: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a single OU process path using Euler-Maruyama discretization.
    Optimized for speed in training loops (no DataFrame overhead).

    dX_t = mu(theta - X_t)dt + sigma*dW_t
    """
    dt = 1.0 / n_steps
    spread = np.zeros(n_steps + 1)
    spread[0] = theta if x0 is None else x0

    noise = np.random.randn(n_steps) * np.sqrt(dt)
    for i in range(n_steps):
        spread[i + 1] = (
            spread[i] * (1 - mu * dt)
            + mu * theta * dt
            + sigma * noise[i]
        )

    return spread


def simulate_ou_spread_batch(
    n_paths: int,
    mu: float = 1.0,
    theta: float = 1.0,
    sigma: float = 0.1,
    n_steps: int = 252,
) -> np.ndarray:
    """
    Generate multiple OU paths at once using vectorized operations.
    Returns shape (n_paths, n_steps+1).
    """
    dt = 1.0 / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = theta

    noise = np.random.randn(n_paths, n_steps) * np.sqrt(dt)
    for i in range(n_steps):
        paths[:, i + 1] = (
            paths[:, i] * (1 - mu * dt)
            + mu * theta * dt
            + sigma * noise[:, i]
        )

    return paths


# ───────────────────────────────────────────────
# Training Loop (Section 5.2)
# ───────────────────────────────────────────────
def train_episode(
    agent: TabularQAgent,
    spread: np.ndarray,
    theta: float,
    c: float = 0.001,
    l: int = DEFAULT_LOOKBACK,
    k: float = DEFAULT_THRESHOLD,
) -> Tuple[float, int]:
    """
    Train the agent on a single OU path.

    Uses scale-invariant state encoding with the spread's std dev.

    Returns:
        (total_reward, n_trades) tuple
    """
    position = 0
    total_reward = 0.0
    n_trades = 0
    scale = max(np.std(spread), 1e-8)

    for t in range(l, len(spread) - 1):
        _, state_idx = extract_state(spread, t, l, k, scale)

        action_idx = agent.select_action(state_idx, position, training=True)
        action_val = ACTIONS[action_idx]

        # Update position (clipped to {0, 1})
        new_position = np.clip(position + action_val, 0, 1)

        # Compute reward
        reward = compute_reward(action_val, spread[t], theta, c)

        # Next state
        is_done = (t == len(spread) - 2)
        if not is_done:
            _, next_state_idx = extract_state(spread, t + 1, l, k, scale)
        else:
            next_state_idx = state_idx  # terminal

        # Q-update
        td_error = agent.update(
            state_idx, action_idx, reward,
            next_state_idx, new_position,
            done=is_done,
        )

        if action_val != 0:
            n_trades += 1

        position = new_position
        total_reward += reward

    return total_reward, n_trades


def train_agent(
    config: Optional[RLAgentConfig] = None,
    ou_params_range: Optional[Dict] = None,
    verbose: bool = True,
) -> TabularQAgent:
    """
    Full training pipeline from Section 5.2.

    Trains on simulated OU paths with optional parameter diversity
    for robustness across different mean-reversion regimes.

    Args:
        config: agent configuration
        ou_params_range: dict with optional ranges for mu, sigma
            e.g. {"mu_range": (0.5, 5.0), "sigma_range": (0.05, 0.3)}
        verbose: print training progress

    Returns:
        Trained TabularQAgent with frozen epsilon
    """
    config = config or RLAgentConfig()
    agent = TabularQAgent(config=config)

    if verbose:
        print_header("RL Agent Training Started")
        buffered_print(f"{CYAN}States: {agent.n_states} | "
                       f"Actions: {agent.n_actions} | "
                       f"Paths: {config.n_training_paths} | "
                       f"Epochs: {config.n_epochs}{ENDC}")

    mu_range = (config.ou_mu, config.ou_mu)
    sigma_range = (config.ou_sigma, config.ou_sigma)
    if ou_params_range:
        mu_range = ou_params_range.get("mu_range", mu_range)
        sigma_range = ou_params_range.get("sigma_range", sigma_range)

    for epoch in range(config.n_epochs):
        epoch_rewards = []
        epoch_trades = []

        for ep in range(config.n_training_paths):
            # Sample OU parameters for diversity
            mu = np.random.uniform(*mu_range)
            sigma = np.random.uniform(*sigma_range)

            # Generate OU path
            spread = simulate_ou_spread(
                mu=mu,
                theta=config.ou_theta,
                sigma=sigma,
                n_steps=config.ou_T,
            )

            # Run one training episode
            ep_reward, ep_trades = train_episode(
                agent, spread, config.ou_theta,
                c=config.transaction_cost,
                l=config.lookback,
                k=config.threshold_k,
            )
            epoch_rewards.append(ep_reward)
            epoch_trades.append(ep_trades)

        agent.episode_rewards.extend(epoch_rewards)

        if verbose:
            stats = agent.get_q_stats()
            buffered_print(
                f"{GREEN}Epoch {epoch + 1}/{config.n_epochs} | "
                f"Mean Reward: {np.mean(epoch_rewards):.4f} | "
                f"Std: {np.std(epoch_rewards):.4f} | "
                f"Avg Trades: {np.mean(epoch_trades):.1f} | "
                f"States Visited: {stats['visited_states']}/{stats['total_states']}{ENDC}"
            )

    # Freeze for deployment
    agent.freeze()

    if verbose:
        final_stats = agent.get_q_stats()
        print_section("Training Complete", GREEN)
        buffered_print(f"Q-table: {final_stats}")

    return agent


# ───────────────────────────────────────────────
# Trading Execution with Trained Agent
# ───────────────────────────────────────────────
def run_rl_trading(
    agent: TabularQAgent,
    spread_prices: np.ndarray,
    theta: float,
    initial_capital: float = 100.0,
    c: float = 0.001,
    l: int = DEFAULT_LOOKBACK,
    k: float = DEFAULT_THRESHOLD,
) -> Dict:
    """
    Execute the trained RL agent on a spread price series.

    This mirrors the paper's testing phase where epsilon=0.

    Args:
        agent: trained (frozen) TabularQAgent
        spread_prices: actual spread prices during trading period
        theta: estimated long-term mean of spread
        initial_capital: starting capital
        c: transaction cost
        l: lookback window
        k: state threshold

    Returns:
        Dict with trading results, equity curve, actions, etc.
    """
    n = len(spread_prices)
    if n < l + 2:
        raise ValueError(f"Spread series too short: {n} < {l + 2}")

    position = 0
    capital = initial_capital
    positions = np.zeros(n, dtype=int)
    actions_taken = np.zeros(n, dtype=int)
    rewards = np.zeros(n)
    equity_curve = np.full(n, initial_capital)
    trade_log: List[Dict] = []

    entry_price = 0.0
    entry_idx = 0
    n_shares = 0.0
    unit_cost = 0.01

    # Scale-invariant state encoding
    scale = max(np.std(spread_prices), 1e-8)

    for t in range(l, n):
        _, state_idx = extract_state(spread_prices, t, l, k, scale)

        # Agent decides (epsilon=0 for testing)
        action_idx = agent.select_action(state_idx, position, training=False)
        action_val = ACTIONS[action_idx]

        new_position = np.clip(position + action_val, 0, 1)

        reward = compute_reward(action_val, spread_prices[t], theta, c)
        rewards[t] = reward

        # Track trades with dollar-based PnL
        if action_val == 1 and position == 0:
            # BUY: entering long
            entry_price = spread_prices[t]
            entry_idx = t
            unit_cost = max(abs(entry_price), 0.01)
            n_shares = capital / unit_cost
        elif action_val == -1 and position == 1:
            # SELL: closing long
            exit_price = spread_prices[t]
            dollar_pnl = n_shares * (exit_price - entry_price)
            commission = n_shares * unit_cost * c
            capital += dollar_pnl - commission
            pnl_pct = dollar_pnl / (n_shares * unit_cost) if n_shares > 0 else 0
            trade_log.append({
                "entry_idx": entry_idx,
                "exit_idx": t,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "pnl_net": pnl_pct - c,
            })
            n_shares = 0.0

        # Mark to market
        if new_position == 1 and n_shares > 0:
            mtm = n_shares * (spread_prices[t] - entry_price)
            equity_curve[t] = capital + mtm
        else:
            equity_curve[t] = capital

        positions[t] = new_position
        actions_taken[t] = action_val
        position = new_position

    # Terminal position handling
    if position == 1:
        exit_price = spread_prices[-1]
        dollar_pnl = n_shares * (exit_price - entry_price)
        capital += dollar_pnl
        pnl_pct = dollar_pnl / (n_shares * max(abs(entry_price), 0.01)) if n_shares > 0 else 0
        equity_curve[-1] = capital
        trade_log.append({
            "entry_idx": entry_idx,
            "exit_idx": n - 1,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_net": pnl_pct,
            "terminal": True,
        })

    # Compute performance metrics
    cumul_return = compute_cumulative_return(
        rewards[l:], spread_prices[-1], position,
    )

    total_trades = len(trade_log)
    winning_trades = sum(1 for t in trade_log if t["pnl_net"] > 0)
    daily_returns = np.diff(equity_curve[l:]) / equity_curve[l:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    return {
        "equity_curve": equity_curve,
        "positions": positions,
        "actions": actions_taken,
        "rewards": rewards,
        "trade_log": trade_log,
        "cumul_return": cumul_return,
        "final_capital": capital,
        "total_return_pct": (capital / initial_capital - 1) * 100,
        "total_trades": total_trades,
        "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
        "daily_return_mean": float(np.mean(daily_returns)) if len(daily_returns) > 0 else 0,
        "daily_return_std": float(np.std(daily_returns)) if len(daily_returns) > 0 else 0,
        "sharpe_ratio": (
            float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
            if len(daily_returns) > 1 and np.std(daily_returns) > 1e-10
            else 0
        ),
        "max_drawdown": _compute_max_drawdown(equity_curve[l:]),
    }


def _compute_max_drawdown(equity: np.ndarray) -> float:
    """Compute maximum drawdown from equity curve."""
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(np.min(drawdown))
