import numpy as np
from agents.base import Agent
from config import (
    QL_ALPHA, QL_GAMMA,
    QL_EPSILON_START, QL_EPSILON_END, QL_EPSILON_DECAY,
    NUM_PRICE_LEVELS, RANDOM_SEED
)

class QLearningAgent(Agent):
    # State Space: (my_last_price_idx, opp_last_price_idx) -> NUM_PRICE LEVELS^2 states
    # Action Space: price_idx is in [0, ..., NUM_PRICE_LEVELS - 1]
    def __init__(self, alpha=QL_ALPHA, gamma=QL_GAMMA, epsilon_start=QL_EPSILON_START, epsilon_end=QL_EPSILON_END,
                 epsilon_decay=QL_EPSILON_DECAY, seed=RANDOM_SEED):
        super().__init__(name="Q-Learner")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed=seed)

        n = NUM_PRICE_LEVELS
        self.Q = np.zeros((n, n, n))

        self._last_state = None
        self._last_action = None

    def _get_state(self, history, price_grid):
        if not history:
            mid = len(price_grid) // 2
            return (mid, mid)
        last = history[-1]
        my_idx = self._price_to_idx(last["my_price"], price_grid)
        opp_idx = self._price_to_idx(last["opp_price"], price_grid)
        return (my_idx, opp_idx)

    def _price_to_idx(self, price, price_grid):
        return int(np.argmin(np.abs(np.array(price_grid) - price)))

    def choose_price(self, price_grid, history):
        state = self._get_state(history, price_grid)
        s_my, s_opp = state

        if self.rng.random() < self.epsilon:
            action_idx = self.rng.integers(0, len(price_grid))
        else:
            action_idx = int(np.argmax(self.Q[s_my, s_opp, :]))

        self._last_state = state
        self._last_action = action_idx

        price = price_grid[action_idx]
        self.last_price = price
        return price

    def update(self, my_price, opp_price, my_profit, opp_profit):
        if self._last_state is None:
            return

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        s_my, s_opp = self._last_state
        a = self._last_action

        reward = my_price

        current_q = self.Q[s_my, s_opp, a]
        next_best = np.max(self.Q[a, :, :])
        td_target = reward + self.gamma * next_best
        self.Q[s_my, s_opp, a] += self.alpha * (td_target - current_q)

    def reset(self) -> None:
        super().reset()
        n = NUM_PRICE_LEVELS
        self.Q = np.zeros((n, n, n))
        self.epsilon = QL_EPSILON_START
        self._last_state = None
        self._last_action = None

    @property
    def exploration_rate(self):
        return self.epsilon

