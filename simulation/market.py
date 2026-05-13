import numpy as np
from prompt_toolkit.shortcuts import progress_dialog

from config import MARGINAL_COST, MONOPOLY_PRICE, NUM_PRICE_LEVELS

class BertrandMarket:
    # Symmetric Bertrand duopoly market
    # Two firms simultaneously set prices
    # Profit = (price - marginal_cost) * quantity
    # Price grid is discretized between marginal_cost and monopoly_price
    def __init__(self, marginal_cost=MARGINAL_COST, monopoly_price=MONOPOLY_PRICE, num_price_levels=NUM_PRICE_LEVELS):
        self.marginal_cost = marginal_cost
        self.monopoly_price = monopoly_price

        self.price_grid = list(np.linspace(marginal_cost, monopoly_price, num_price_levels))
        self.price_grid = [round(p, 6) for p in self.price_grid]

        # Competitive Nash price = martinal cost
        self.competitive_price = marginal_cost

    def step(self, prices):
        min_price = min(prices)
        winners = [i for i, p in enumerate(prices) if p == min_price]
        share = 1.0 / len(winners)

        profits = []

        for i, price in enumerate(prices):
            if i in winners:
                profits.append((price - self.marginal_cost) * share)
            else:
                profits.append(0)
        return profits

    def collusion_price(self):
        return self.monopoly_price

    def collusion_index(self, avg_price):
        span = self.monopoly_price - self.competitive_price
        if span == 0:
            return 0.0
        return max(0.0, min(1.0, (avg_price - self.competitive_price) / span))